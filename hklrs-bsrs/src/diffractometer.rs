//! The diffractometer device: the reals that drive each geometry axis, the
//! wavelength and the samples, read as one reading set with h, k, l.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use async_trait::async_trait;
use bsrs::core::msg::{ConfigurableObj, DynLocation, LocatableObj, NamedObj, ReadableObj};
use bsrs::core::status::Status;
use bsrs::core::{BsrsError, ConfigureArgs, DataKey, Dtype, ReadingValue};
use bsrs::event_model::compose::now;
use bsrs::event_model::{SignalMetadata, make_datakey};
use futures::future::{join_all, try_join_all};
use futures::try_join;
use hklrs::nalgebra::{Matrix3, Point3, Vector3};
use hklrs::{
    AxisKind, Deg, Geometry, InverseRequest, Lattice, Mode, ModeSpec, Orientation, Picker,
    Reflection, Sample, Target, Wavelength, forward,
};
use serde_json::{Value, json};

use crate::DiffError;
use crate::samples::{Measured, Samples};

/// A real axis: moved and located as one value, and read by a plan.
pub trait RealAxis: LocatableObj + ReadableObj {}

impl<T: LocatableObj + ReadableObj + ?Sized> RealAxis for T {}

#[derive(Clone)]
pub enum WavelengthSource {
    Fixed(Wavelength),
    /// A readable with exactly one field, in Å.
    Signal(Arc<dyn ReadableObj>),
}

impl WavelengthSource {
    async fn read(&self) -> Result<Wavelength, BsrsError> {
        let signal = match self {
            Self::Fixed(w) => return Ok(*w),
            Self::Signal(s) => s,
        };
        let reading = signal.read_dyn().await?;
        let name = signal.name().to_owned();
        if reading.len() != 1 {
            let mut fields: Vec<String> = reading.into_keys().collect();
            fields.sort();
            return Err(DiffError::WavelengthFields {
                signal: name,
                fields,
            }
            .into());
        }
        let value = reading.into_values().next().expect("one field").value;
        let angstrom = value.as_f64().ok_or_else(|| DiffError::WavelengthValue {
            signal: name,
            value: value.clone(),
        })?;
        Ok(Wavelength::new(angstrom).map_err(DiffError::from)?)
    }
}

#[derive(Clone)]
struct State {
    wavelength: WavelengthSource,
    samples: Samples,
    /// None only for a geometry no [`ModeSpec`] builds on.
    mode: Option<ModeSpec>,
    /// The last h, k, l the pseudo axes were sent to; their setpoint.
    target: Option<[f64; 3]>,
}

impl State {
    fn set_wavelength(&mut self, wavelength: Wavelength) -> Result<(), DiffError> {
        match &self.wavelength {
            WavelengthSource::Fixed(_) => {
                self.wavelength = WavelengthSource::Fixed(wavelength);
                Ok(())
            }
            WavelengthSource::Signal(s) => Err(DiffError::WavelengthFromSignal {
                signal: s.name().to_owned(),
            }),
        }
    }
}

/// Configuration fields: key suffix, dtype, shape, units.
const CONFIG: [(&str, Dtype, &[u64], Option<&str>); 7] = [
    ("geometry", Dtype::String, &[], None),
    ("sample", Dtype::String, &[], None),
    // a, b, c in Å; alpha, beta, gamma in degrees
    ("lattice", Dtype::Array, &[6], None),
    ("U", Dtype::Array, &[3, 3], None),
    ("UB", Dtype::Array, &[3, 3], Some("1/angstrom")),
    ("wavelength", Dtype::Number, &[], Some("angstrom")),
    // null for a geometry without modes
    ("mode", Dtype::String, &[], None),
];

const SETTABLE: [&str; 3] = ["sample", "wavelength", "mode"];

/// What a plan solves its trajectory from.
pub(crate) struct Snapshot {
    /// Readbacks in geometry axis order, degrees and metres.
    pub current: Vec<f64>,
    pub wavelength: Wavelength,
    pub sample: Sample,
    pub mode: Option<ModeSpec>,
}

pub struct Diffractometer {
    name: String,
    geometry: Geometry,
    /// `reals[i]` drives `geometry.axes()[i]`.
    reals: Vec<Arc<dyn RealAxis>>,
    state: RwLock<State>,
    /// Held by a pseudo-axis move from its setpoint through its last real's
    /// set, so concurrent moves reach the reals in setpoint order.
    moving: futures::lock::Mutex<()>,
}

impl Diffractometer {
    /// `reals` pairs every geometry axis name, in any order, with the real
    /// that drives it.
    pub fn new(
        name: &str,
        geometry: Geometry,
        reals: Vec<(&str, Arc<dyn RealAxis>)>,
        wavelength: WavelengthSource,
        sample: Sample,
    ) -> Result<Self, DiffError> {
        let mut slots: Vec<Option<Arc<dyn RealAxis>>> = vec![None; geometry.axes().len()];
        for (axis, real) in reals {
            let i = geometry.axis_index(axis)?;
            if slots[i].replace(real).is_some() {
                return Err(DiffError::DuplicateAxis {
                    name: axis.to_owned(),
                });
            }
        }
        let reals = slots
            .into_iter()
            .zip(geometry.axes())
            .map(|(real, axis)| {
                real.ok_or_else(|| DiffError::MissingAxis {
                    name: axis.name().to_owned(),
                })
            })
            .collect::<Result<_, _>>()?;
        // hkl's default: the engine's first mode
        let mode = ModeSpec::names()
            .map(ModeSpec::new)
            .find(|m| m.build(&geometry).is_ok());
        Ok(Self {
            name: name.to_owned(),
            geometry,
            reals,
            state: RwLock::new(State {
                wavelength,
                samples: Samples::new(sample),
                mode,
                target: None,
            }),
            moving: futures::lock::Mutex::new(()),
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The reals in geometry axis order.
    pub fn reals(&self) -> &[Arc<dyn RealAxis>] {
        &self.reals
    }

    fn key(&self, field: &str) -> String {
        format!("{}_{field}", self.name)
    }

    /// Reading keys of the axis readbacks, in geometry axis order.
    pub fn axis_keys(&self) -> Vec<String> {
        self.geometry
            .axes()
            .iter()
            .map(|a| self.key(a.name()))
            .collect()
    }

    pub fn hkl_keys(&self) -> [String; 3] {
        ["h", "k", "l"].map(|c| self.key(c))
    }

    // Every mutation is computed before the guard is written, so a lock
    // poisoned by a panic elsewhere still holds a whole state.
    fn state(&self) -> RwLockReadGuard<'_, State> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn state_mut(&self) -> RwLockWriteGuard<'_, State> {
        self.state.write().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn current_sample(&self) -> Sample {
        self.state().samples.current().sample.clone()
    }

    pub fn sample_names(&self) -> Vec<String> {
        self.state().samples.names()
    }

    pub fn add_sample(&self, sample: Sample) -> Result<(), DiffError> {
        self.state_mut().samples.add(sample)
    }

    pub fn select_sample(&self, name: &str) -> Result<(), DiffError> {
        self.state_mut().samples.select(name)
    }

    /// Removes sample `name`. The current sample changes only when it is
    /// the one removed, to the first remaining name; the last sample stays.
    pub fn remove_sample(&self, name: &str) -> Result<Sample, DiffError> {
        self.state_mut().samples.remove(name)
    }

    pub fn set_lattice(&self, lattice: Lattice) {
        self.state_mut().samples.current_mut().sample.lattice = lattice;
    }

    /// Sets U of the current sample from UB = U·B with its lattice's B.
    pub fn set_ub(&self, ub: &Matrix3<f64>) -> Result<(), DiffError> {
        let mut state = self.state_mut();
        let sample = &mut state.samples.current_mut().sample;
        let b_inverse = sample
            .lattice
            .b_matrix()
            .try_inverse()
            .expect("B has a positive diagonal for every valid Lattice");
        sample.orientation = Orientation::from_u_matrix(&(ub * b_inverse))?;
        Ok(())
    }

    pub fn set_wavelength(&self, wavelength: Wavelength) -> Result<(), DiffError> {
        self.state_mut().set_wavelength(wavelength)
    }

    /// The hkl mode the pseudo axes solve in; at construction the first of
    /// [`ModeSpec::names`] this geometry builds.
    pub fn mode(&self) -> Option<ModeSpec> {
        self.state().mode.clone()
    }

    pub fn set_mode(&self, mode: ModeSpec) -> Result<(), DiffError> {
        self.build_mode(&mode)?;
        self.state_mut().mode = Some(mode);
        Ok(())
    }

    fn build_mode<'g>(&'g self, mode: &ModeSpec) -> Result<Mode<'g>, DiffError> {
        mode.build(&self.geometry)
            .map_err(|source| DiffError::Mode {
                mode: mode.name.clone(),
                source,
            })
    }

    /// The last h, k, l the pseudo axes were sent to.
    pub fn target(&self) -> Option<[f64; 3]> {
        self.state().target
    }

    /// Records reflection `name` of the current sample.
    pub fn add_reflection(&self, name: &str, measured: Measured) -> Result<(), DiffError> {
        self.reflection(&measured)?;
        let mut state = self.state_mut();
        let entry = state.samples.current_mut();
        if entry.reflections.contains_key(name) {
            return Err(DiffError::DuplicateReflection {
                sample: entry.sample.name.clone(),
                name: name.to_owned(),
            });
        }
        entry.reflections.insert(name.to_owned(), measured);
        Ok(())
    }

    /// Orients the current sample from two of its reflections (Busing &
    /// Levy) and returns its UB.
    pub fn compute_ub(&self, primary: &str, secondary: &str) -> Result<Matrix3<f64>, DiffError> {
        let mut state = self.state_mut();
        let entry = state.samples.current_mut();
        let orientation = Orientation::busing_levy(
            self.reflection(entry.reflection(primary)?)?,
            self.reflection(entry.reflection(secondary)?)?,
        )?;
        entry.sample.orientation = orientation;
        Ok(entry.sample.ub())
    }

    fn reflection(&self, m: &Measured) -> Result<Reflection, DiffError> {
        let values = self.geometry.values_from_degrees(&m.positions)?;
        Ok(Reflection::measure(
            Vector3::from(m.hkl),
            &self.geometry,
            &values,
            m.wavelength,
            &self.point(),
        )?)
    }

    /// A point on the scattered beam in the detector-arm frame: the hkl
    /// detector, at any distance along the beam.
    fn point(&self) -> Point3<f64> {
        Point3::from(self.geometry.beam_direction().into_inner())
    }

    /// Readbacks of every real in geometry axis order, degrees for rotations
    /// and metres for translations.
    pub async fn positions(&self) -> Result<Vec<f64>, BsrsError> {
        let located = try_join_all(self.reals.iter().map(|r| r.locate_dyn())).await?;
        Ok(located.into_iter().map(|l| l.readback).collect())
    }

    pub async fn wavelength(&self) -> Result<Wavelength, BsrsError> {
        let source = self.state().wavelength.clone();
        source.read().await
    }

    /// Miller indices of the current sample at `positions` (geometry axis
    /// order, degrees for rotations and metres for translations).
    pub fn hkl(
        &self,
        positions: &[f64],
        wavelength: Wavelength,
    ) -> Result<Vector3<f64>, DiffError> {
        let values = self.geometry.values_from_degrees(positions)?;
        let state = self.state();
        let sample = &state.samples.current().sample;
        Ok(forward(&self.geometry, sample, &values, wavelength, &self.point())?.hkl)
    }

    /// Axis values that reach `hkl` with the current sample and mode, the
    /// solution closest to `current`, which also gives the held axes. Both
    /// in geometry axis order, degrees for rotations and metres for
    /// translations.
    pub fn inverse(
        &self,
        hkl: [f64; 3],
        current: &[f64],
        wavelength: Wavelength,
    ) -> Result<Vec<f64>, DiffError> {
        let current = self.geometry.values_from_degrees(current)?;
        let (mode, ub) = {
            let state = self.state();
            (state.mode.clone(), state.samples.current().sample.ub())
        };
        let mode = self.require_mode(mode.as_ref())?;
        let values = self.solve(&mode, ub, wavelength, hkl, &current)?;
        Ok(self.geometry.values_to_degrees(&values)?)
    }

    pub(crate) fn require_mode(&self, mode: Option<&ModeSpec>) -> Result<ModeSpec, DiffError> {
        mode.cloned().ok_or_else(|| DiffError::NoMode {
            geometry: self.geometry.name().to_owned(),
        })
    }

    /// Radian axis values reaching `hkl` in `mode`, closest to `seed`.
    fn solve(
        &self,
        mode: &ModeSpec,
        ub: Matrix3<f64>,
        wavelength: Wavelength,
        hkl: [f64; 3],
        seed: &[f64],
    ) -> Result<Vec<f64>, DiffError> {
        let mode = self.build_mode(mode)?;
        let request = InverseRequest {
            mode: &mode,
            ub,
            wavelength,
            target: Target::Hkl(Vector3::from(hkl)),
            current: seed,
        };
        request
            .pick(Picker::Closest)
            .map_err(|source| DiffError::Solve { hkl, source })
    }

    /// The readbacks, the wavelength, and the sample and mode from one read
    /// of the state.
    pub(crate) async fn snapshot(&self) -> Result<Snapshot, BsrsError> {
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        let state = self.state();
        Ok(Snapshot {
            current,
            wavelength,
            sample: state.samples.current().sample.clone(),
            mode: state.mode.clone(),
        })
    }

    /// Axis values for each point of `path` in its mode, each picked closest
    /// to the previous point's solution and the first to the readbacks, so
    /// the path takes no branch jump.
    pub(crate) fn solve_path(
        &self,
        snapshot: &Snapshot,
        path: &[([f64; 3], ModeSpec)],
    ) -> Result<Vec<Vec<f64>>, DiffError> {
        let ub = snapshot.sample.ub();
        let mut seed = self.geometry.values_from_degrees(&snapshot.current)?;
        let mut solved = Vec::with_capacity(path.len());
        for (index, (hkl, mode)) in path.iter().enumerate() {
            let point = |source| DiffError::Point {
                index,
                source: Box::new(source),
            };
            let values = self
                .solve(mode, ub, snapshot.wavelength, *hkl, &seed)
                .map_err(point)?;
            let degrees = self.geometry.values_to_degrees(&values);
            solved.push(degrees.map_err(|e| point(e.into()))?);
            seed = values;
        }
        Ok(solved)
    }

    pub(crate) fn set_target(&self, target: Option<[f64; 3]>) {
        self.state_mut().target = target;
    }

    /// Configuration values by field, without the device-name prefix.
    pub(crate) fn configuration(
        &self,
        sample: &Sample,
        mode: Option<&str>,
        wavelength: Wavelength,
    ) -> Vec<(&'static str, Value)> {
        let l = &sample.lattice;
        let deg = |r| Deg::from(r).0;
        let values = [
            json!(self.geometry.name()),
            json!(sample.name),
            json!([
                l.a(),
                l.b(),
                l.c(),
                deg(l.alpha()),
                deg(l.beta()),
                deg(l.gamma())
            ]),
            rows(&sample.u().to_rotation_matrix().into_inner()),
            rows(&sample.ub()),
            json!(wavelength.angstrom()),
            json!(mode),
        ];
        CONFIG.iter().map(|c| c.0).zip(values).collect()
    }

    /// Replaces component `index` of the pseudo setpoint (the readback h, k,
    /// l before the first move), solves closest to the readbacks and sets
    /// every real; the setpoint is kept only when the solve succeeds.
    pub(crate) async fn set_component(
        &self,
        index: usize,
        value: f64,
    ) -> Result<Status, BsrsError> {
        let _gate = self.moving.lock().await;
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        let saved = self.target();
        let mut target = match saved {
            Some(t) => t,
            None => self.hkl(&current, wavelength)?.into(),
        };
        target[index] = value;
        let values = self.inverse(target, &current, wavelength)?;
        self.state_mut().target = Some(target);
        let sets = self.reals.iter().zip(values).map(|(r, v)| r.set_dyn(v));
        let statuses = join_all(sets).await;
        Ok(statuses.iter().fold(Status::done(), |all, s| all.and(s)))
    }

    /// Component `index` of the pseudo setpoint and of the h, k, l at the
    /// readbacks.
    pub(crate) async fn locate_component(&self, index: usize) -> Result<DynLocation, BsrsError> {
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        let readback = self.hkl(&current, wavelength)?[index];
        let setpoint = self.target().map_or(readback, |t| t[index]);
        Ok(DynLocation { setpoint, readback })
    }

    /// Delivers a pause's stop to every real.
    pub(crate) async fn stop_reals(&self, success: bool) -> Result<(), BsrsError> {
        try_join_all(self.reals.iter().map(|r| r.stop_on_pause(success))).await?;
        Ok(())
    }

    /// Sets configuration by key, all or none: `{name}_sample` selects a
    /// sample, `{name}_wavelength` sets a fixed wavelength in Å.
    pub fn configure(&self, values: &HashMap<String, Value>) -> Result<(), DiffError> {
        let mut state = self.state_mut();
        let mut next = state.clone();
        for (key, value) in values.iter().collect::<BTreeMap<_, _>>() {
            let bad = |expected| DiffError::ConfigValue {
                key: key.clone(),
                value: value.clone(),
                expected,
            };
            match key
                .strip_prefix(&self.name)
                .and_then(|f| f.strip_prefix('_'))
            {
                Some("sample") => next
                    .samples
                    .select(value.as_str().ok_or_else(|| bad("string"))?)?,
                Some("wavelength") => {
                    let angstrom = value.as_f64().ok_or_else(|| bad("number"))?;
                    next.set_wavelength(Wavelength::new(angstrom)?)?;
                }
                Some("mode") => {
                    let mode = ModeSpec::new(value.as_str().ok_or_else(|| bad("string"))?);
                    self.build_mode(&mode)?;
                    next.mode = Some(mode);
                }
                _ => {
                    return Err(DiffError::ConfigKey {
                        key: key.clone(),
                        settable: SETTABLE.map(|f| self.key(f)).to_vec(),
                    });
                }
            }
        }
        *state = next;
        Ok(())
    }
}

pub(crate) fn datakey(key: &str, dtype: Dtype, shape: &[u64], units: Option<&str>) -> DataKey {
    let numpy = if dtype == Dtype::String {
        "|S40"
    } else {
        "<f8"
    };
    let meta = SignalMetadata {
        units: units.map(str::to_owned),
        ..SignalMetadata::default()
    };
    let shape = shape.iter().map(|&n| Some(n)).collect();
    make_datakey(
        format!("hklrs://{key}"),
        dtype,
        shape,
        Some(numpy.into()),
        meta,
    )
}

fn rows(m: &Matrix3<f64>) -> Value {
    json!(
        (0..3)
            .map(|i| [m[(i, 0)], m[(i, 1)], m[(i, 2)]])
            .collect::<Vec<_>>()
    )
}

impl NamedObj for Diffractometer {
    fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl ReadableObj for Diffractometer {
    /// Every axis readback and the h, k, l computed from those same
    /// readbacks.
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        let (positions, wavelength) = futures::try_join!(self.positions(), self.wavelength())?;
        let hkl = self.hkl(&positions, wavelength)?;
        let t = now();
        let values = positions.iter().chain(hkl.iter());
        let keys = self.axis_keys().into_iter().chain(self.hkl_keys());
        Ok(keys
            .zip(values)
            .map(|(k, &v)| (k, ReadingValue::new(json!(v), t)))
            .collect())
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        let axes = self.geometry.axes().iter().map(|a| {
            let units = match a.kind() {
                AxisKind::Rotation { .. } | AxisKind::RotationAbout { .. } => Some("deg"),
                AxisKind::Translation { .. } => Some("m"),
                AxisKind::Parameter => None,
            };
            (self.key(a.name()), units)
        });
        let hkl = self.hkl_keys().into_iter().map(|k| (k, None));
        Ok(axes
            .chain(hkl)
            .map(|(k, units)| {
                let key = datakey(&k, Dtype::Number, &[], units);
                (k, key)
            })
            .collect())
    }

    fn hint_fields(&self) -> Option<Vec<String>> {
        Some(self.hkl_keys().to_vec())
    }

    fn as_configurable(&self) -> Option<&dyn ConfigurableObj> {
        Some(self)
    }
}

#[async_trait]
impl ConfigurableObj for Diffractometer {
    async fn read_configuration_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        let wavelength = self.wavelength().await?;
        let fields = {
            let state = self.state();
            let mode = state.mode.as_ref().map(|m| m.name.as_str());
            self.configuration(&state.samples.current().sample, mode, wavelength)
        };
        let t = now();
        Ok(fields
            .into_iter()
            .map(|(field, v)| (self.key(field), ReadingValue::new(v, t)))
            .collect())
    }

    async fn describe_configuration_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        Ok(CONFIG
            .iter()
            .map(|&(field, dtype, shape, units)| {
                let k = self.key(field);
                let key = datakey(&k, dtype, shape, units);
                (k, key)
            })
            .collect())
    }

    async fn configure_dyn(&self, args: ConfigureArgs) -> Result<(), BsrsError> {
        Ok(self.configure(&args.values)?)
    }
}
