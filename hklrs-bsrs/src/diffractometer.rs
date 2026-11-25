//! The diffractometer device: the reals that drive each geometry axis, the
//! wavelength and the samples, read as one reading set with h, k, l and ψ.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{PI, TAU};
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
    AxisKind, Deg, Geometry, InverseError, InverseRequest, Lattice, Mode, ModeSpec, ModeTable,
    Orientation, Picker, Reflection, Sample, Target, Wavelength, forward,
};
use serde_json::{Value, json};

use crate::DiffError;
use crate::config::Config;
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
    /// Its limits and cut points change; its axes and stacks do not.
    geometry: Geometry,
    wavelength: WavelengthSource,
    samples: Samples,
    /// None only for a geometry without hkl's default mode.
    mode: Option<ModeSpec>,
    /// The last h, k, l the pseudo axes were sent to; their setpoint.
    target: Option<[f64; 3]>,
}

impl State {
    /// Miller indices of the current sample at `positions` (degrees and
    /// metres).
    fn hkl(&self, positions: &[f64], wavelength: Wavelength) -> Result<Vector3<f64>, DiffError> {
        let geometry = &self.geometry;
        let values = geometry.values_from_degrees(positions)?;
        let sample = &self.samples.current().sample;
        Ok(forward(geometry, sample, &values, wavelength, &point(geometry))?.hkl)
    }

    /// Positions reaching `hkl` in `mode`, the solution closest to
    /// `current`; degrees and metres.
    fn inverse(
        &self,
        mode: &ModeSpec,
        hkl: [f64; 3],
        current: &[f64],
        wavelength: Wavelength,
    ) -> Result<Vec<f64>, DiffError> {
        let geometry = &self.geometry;
        let seed = geometry.values_from_degrees(current)?;
        let ub = self.samples.current().sample.ub();
        let values = solve(geometry, mode, ub, wavelength, hkl, &seed)?;
        Ok(geometry.values_to_degrees(&values)?)
    }

    /// The reference and ψ (rad) of the device's mode, a psi mode.
    fn psi(&self) -> Result<(Vector3<f64>, f64), DiffError> {
        let mode = require_mode(&self.geometry, self.mode.as_ref())?;
        mode.psi().ok_or_else(|| DiffError::NoPsi {
            mode: mode.name.clone(),
        })
    }

    /// Solves `setpoint` closest to `current` and, only when that succeeds,
    /// commits it with the mode it was solved in; returns the positions.
    fn commit(
        &mut self,
        setpoint: Setpoint,
        current: &[f64],
        wavelength: Wavelength,
    ) -> Result<Vec<f64>, DiffError> {
        let mut mode = require_mode(&self.geometry, self.mode.as_ref())?.clone();
        let target = match setpoint {
            Setpoint::Hkl(hkl) => hkl,
            Setpoint::Component { component, value } => {
                // before the first move, the h, k, l at the readbacks
                let mut target = match self.target {
                    Some(target) => target,
                    None => self.hkl(current, wavelength)?.into(),
                };
                match component {
                    Component::Hkl(index) => target[index] = value,
                    Component::Psi => mode = mode.with_psi(self.psi()?.0, value.to_radians()),
                }
                target
            }
        };
        let positions = self.inverse(&mode, target, current, wavelength)?;
        self.target = Some(target);
        self.mode = Some(mode);
        Ok(positions)
    }

    /// `component` of the pseudo setpoint and at the readbacks `current`.
    fn locate(
        &self,
        component: Component,
        current: &[f64],
        wavelength: Wavelength,
    ) -> Result<DynLocation, DiffError> {
        match component {
            Component::Hkl(index) => {
                let readback = self.hkl(current, wavelength)?[index];
                let setpoint = self.target.map_or(readback, |t| t[index]);
                Ok(DynLocation { setpoint, readback })
            }
            Component::Psi => {
                let (reference, setpoint) = self.psi()?;
                let values = self.geometry.values_from_degrees(current)?;
                let ub = self.samples.current().sample.ub();
                let principal = hklrs::inverse::psi(&self.geometry, &ub, &reference, &values)
                    .map_err(|source| DiffError::Psi {
                        reference: reference.into(),
                        source,
                    })?;
                // on the setpoint's branch: within half a turn of it
                let turn = (principal - setpoint).rem_euclid(TAU);
                let readback = setpoint + if turn > PI { turn - TAU } else { turn };
                Ok(DynLocation {
                    setpoint: setpoint.to_degrees(),
                    readback: readback.to_degrees(),
                })
            }
        }
    }

    /// `component` at the readbacks `current` as [`State::locate`] reads it
    /// back, but NaN where it has no value: ψ outside a psi mode or where it
    /// is undefined, which `locate` refuses.
    fn reading(
        &self,
        component: Component,
        current: &[f64],
        wavelength: Wavelength,
    ) -> Result<f64, DiffError> {
        match self.locate(component, current, wavelength) {
            Ok(location) => Ok(location.readback),
            Err(
                DiffError::NoMode { .. }
                | DiffError::NoPsi { .. }
                | DiffError::Psi {
                    source: InverseError::PsiUndefined,
                    ..
                },
            ) => Ok(f64::NAN),
            Err(e) => Err(e),
        }
    }

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
    pub geometry: Geometry,
    /// Readbacks in geometry axis order, degrees and metres.
    pub current: Vec<f64>,
    pub wavelength: Wavelength,
    pub sample: Sample,
    pub mode: Option<ModeSpec>,
}

impl Snapshot {
    pub fn mode(&self) -> Result<&ModeSpec, DiffError> {
        require_mode(&self.geometry, self.mode.as_ref())
    }

    /// Axis values for each point of `path` in its mode, each picked closest
    /// to the previous point's solution and the first to the readbacks, so
    /// the path takes no branch jump.
    pub fn solve_path(&self, path: &[([f64; 3], ModeSpec)]) -> Result<Vec<Vec<f64>>, DiffError> {
        let geometry = &self.geometry;
        let ub = self.sample.ub();
        let mut seed = geometry.values_from_degrees(&self.current)?;
        let mut solved = Vec::with_capacity(path.len());
        for (index, (hkl, mode)) in path.iter().enumerate() {
            let point = |source| DiffError::Point {
                index,
                source: Box::new(source),
            };
            let values = solve(geometry, mode, ub, self.wavelength, *hkl, &seed).map_err(point)?;
            let degrees = geometry.values_to_degrees(&values);
            solved.push(degrees.map_err(|e| point(e.into()))?);
            seed = values;
        }
        Ok(solved)
    }
}

fn require_mode<'m>(
    geometry: &Geometry,
    mode: Option<&'m ModeSpec>,
) -> Result<&'m ModeSpec, DiffError> {
    mode.ok_or_else(|| DiffError::NoMode {
        geometry: geometry.name().to_owned(),
    })
}

pub(crate) fn build_mode<'g>(
    geometry: &'g Geometry,
    mode: &ModeSpec,
) -> Result<Mode<'g>, DiffError> {
    mode.build(geometry).map_err(|source| DiffError::Mode {
        mode: mode.name.clone(),
        source,
    })
}

/// hkl's default mode of the preset `geometry` is named after, as a new
/// engine list has it, where it builds on `geometry`.
pub(crate) fn default_mode(geometry: &Geometry) -> Option<ModeSpec> {
    let mode = ModeTable::of(geometry.name()).ok()?.default_mode()?;
    mode.build(geometry).is_ok().then_some(mode)
}

/// The one psi mode of `geometry`'s [`ModeTable`] that builds on it.
pub(crate) fn psi_mode(geometry: &Geometry) -> Result<String, DiffError> {
    match <[String; 1]>::try_from(psi_modes(geometry)) {
        Ok([one]) => Ok(one),
        Err(candidates) => Err(DiffError::PsiMode {
            geometry: geometry.name().to_owned(),
            candidates,
        }),
    }
}

/// The psi modes of `geometry`'s [`ModeTable`] that build on it.
fn psi_modes(geometry: &Geometry) -> Vec<String> {
    let Ok(table) = ModeTable::of(geometry.name()) else {
        return Vec::new();
    };
    // a psi mode refuses a geometry by its axes, whatever the reference
    let builds = |name: &&str| {
        let mode = ModeSpec::new(*name).with_psi(Vector3::z(), 0.0);
        mode.build(geometry).is_ok()
    };
    table
        .psi_modes()
        .filter(builds)
        .map(str::to_owned)
        .collect()
}

/// [`Diffractometer::set_limits`] on `geometry`.
pub(crate) fn set_limits(
    geometry: &mut Geometry,
    name: &str,
    min: f64,
    max: f64,
) -> Result<(), DiffError> {
    let rotation = geometry.axes()[geometry.axis_index(name)?].is_rotation();
    let value = |v: f64| if rotation { v.to_radians() } else { v };
    geometry
        .set_limits(name, value(min), value(max))
        .map_err(|e| match e {
            hklrs::Error::AxisLimits { name, .. } => DiffError::Limits { name, min, max },
            e => e.into(),
        })
}

/// Radian axis values reaching `hkl` in `mode`, closest to `seed`.
fn solve(
    geometry: &Geometry,
    mode: &ModeSpec,
    ub: Matrix3<f64>,
    wavelength: Wavelength,
    hkl: [f64; 3],
    seed: &[f64],
) -> Result<Vec<f64>, DiffError> {
    let mode = build_mode(geometry, mode)?;
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

/// A point on the scattered beam in the detector-arm frame: the hkl
/// detector, at any distance along the beam.
fn point(geometry: &Geometry) -> Point3<f64> {
    Point3::from(geometry.beam_direction().into_inner())
}

pub(crate) fn reflection(geometry: &Geometry, m: &Measured) -> Result<Reflection, DiffError> {
    let values = geometry.values_from_degrees(&m.positions)?;
    Ok(Reflection::measure(
        Vector3::from(m.hkl),
        geometry,
        &values,
        m.wavelength,
        &point(geometry),
    )?)
}

/// Configuration values by field, without the device-name prefix.
pub(crate) fn configuration(
    geometry: &Geometry,
    sample: &Sample,
    mode: Option<&str>,
    wavelength: Wavelength,
) -> Vec<(&'static str, Value)> {
    let l = &sample.lattice;
    let deg = |r| Deg::from(r).0;
    let values = [
        json!(geometry.name()),
        json!(sample.name),
        json!([
            l.a(),
            l.b(),
            l.c(),
            deg(l.alpha()),
            deg(l.beta()),
            deg(l.gamma())
        ]),
        json!(rows(&sample.u().to_rotation_matrix().into_inner())),
        json!(rows(&sample.ub())),
        json!(wavelength.angstrom()),
        json!(mode),
    ];
    CONFIG.iter().map(|c| c.0).zip(values).collect()
}

/// One component of the pseudo setpoint.
#[derive(Clone, Copy)]
pub(crate) enum Component {
    /// h, k or l by index.
    Hkl(usize),
    /// ψ of the device's psi mode, degrees.
    Psi,
}

impl Component {
    pub(crate) fn units(self) -> Option<&'static str> {
        match self {
            Component::Hkl(_) => None,
            Component::Psi => Some("deg"),
        }
    }
}

/// Where a [`Diffractometer::move_to`] goes.
pub(crate) enum Setpoint {
    /// One component replaced, the others kept: h, k or l in the hkl
    /// setpoint, ψ in the mode with the hkl setpoint held.
    Component {
        component: Component,
        value: f64,
    },
    Hkl([f64; 3]),
}

pub struct Diffractometer {
    name: String,
    /// `reals[i]` drives `geometry.axes()[i]`.
    reals: Vec<Arc<dyn RealAxis>>,
    state: RwLock<State>,
    /// Held by [`Diffractometer::move_to`] from its setpoint through its
    /// last real's set, so concurrent moves reach the reals in setpoint
    /// order.
    moving: futures::lock::Mutex<()>,
    /// The plan whose run holds the reals: [`Diffractometer::move_to`]
    /// refuses while it is set. Set by [`Diffractometer::claim`] and cleared
    /// when its [`Scanning`] drops.
    scan: std::sync::Mutex<Option<&'static str>>,
    /// The reading's ψ key, on a geometry with a psi mode. Fixed at
    /// construction, so a mode change cannot change the reading's keys
    /// during a run.
    psi_key: Option<String>,
}

/// A scan's hold on the reals, from before it reads the readbacks it solves
/// from to the end of its run; dropping it releases them.
pub(crate) struct Scanning {
    dx: Arc<Diffractometer>,
}

impl Scanning {
    /// Drops the pseudo setpoint: the reals move outside the pseudo axes.
    /// No move is in flight while a scan holds the reals.
    pub fn clear_target(&self) {
        self.dx.state_mut().target = None;
    }
}

impl Drop for Scanning {
    fn drop(&mut self) {
        *self.dx.scan() = None;
    }
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
        let mode = default_mode(&geometry);
        let psi_key = (!psi_modes(&geometry).is_empty()).then(|| format!("{name}_psi"));
        Ok(Self {
            name: name.to_owned(),
            reals,
            state: RwLock::new(State {
                geometry,
                wavelength,
                samples: Samples::new(sample),
                mode,
                target: None,
            }),
            moving: futures::lock::Mutex::new(()),
            scan: std::sync::Mutex::new(None),
            psi_key,
        })
    }

    /// A copy of the geometry with its current limits and cut points.
    pub fn geometry(&self) -> Geometry {
        self.state().geometry.clone()
    }

    /// The reals in geometry axis order.
    pub fn reals(&self) -> &[Arc<dyn RealAxis>] {
        &self.reals
    }

    pub(crate) fn key(&self, field: &str) -> String {
        format!("{}_{field}", self.name)
    }

    /// Reading keys of the axis readbacks, in geometry axis order.
    pub fn axis_keys(&self) -> Vec<String> {
        let state = self.state();
        let axes = state.geometry.axes().iter();
        axes.map(|a| self.key(a.name())).collect()
    }

    pub fn hkl_keys(&self) -> [String; 3] {
        ["h", "k", "l"].map(|c| self.key(c))
    }

    // Every mutation is computed before the guard is written, so a lock
    // poisoned by a panic elsewhere still holds a whole state. No guard is
    // taken while another is held.
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

    /// Limits axis `name` to `[min, max]`, degrees for a rotation and metres
    /// for a translation; an infinite end is unbounded. A solution outside
    /// the limits after wrapping to the cut point is rejected.
    pub fn set_limits(&self, name: &str, min: f64, max: f64) -> Result<(), DiffError> {
        set_limits(&mut self.state_mut().geometry, name, min, max)
    }

    /// Represents rotation `name` in [cut, cut + 360) degrees.
    pub fn set_cut_point(&self, name: &str, cut: f64) -> Result<(), DiffError> {
        Ok(self
            .state_mut()
            .geometry
            .set_cut_point(name, cut.to_radians())?)
    }

    /// Everything [`Diffractometer::load_config`] sets.
    pub fn config(&self) -> Config {
        let state = self.state();
        Config::capture(
            &state.geometry,
            &state.samples,
            state.mode.as_ref(),
            &state.wavelength,
        )
    }

    /// Sets the limits and cut points, the samples and the current one, the
    /// mode and a fixed wavelength from `config`, all or none. Its geometry
    /// must be this device's by name and axes.
    pub fn load_config(&self, config: &Config) -> Result<(), DiffError> {
        let mut state = self.state_mut();
        let mut next = state.clone();
        let (geometry, samples, mode) = config.load(&next.geometry)?;
        next.geometry = geometry;
        next.samples = samples;
        next.mode = mode;
        if let Some(angstrom) = config.wavelength {
            next.set_wavelength(Wavelength::new(angstrom)?)?;
        }
        *state = next;
        Ok(())
    }

    /// The hkl mode the pseudo axes solve in; at construction hkl's default
    /// mode of the geometry's preset.
    pub fn mode(&self) -> Option<ModeSpec> {
        self.state().mode.clone()
    }

    pub fn set_mode(&self, mode: ModeSpec) -> Result<(), DiffError> {
        let mut state = self.state_mut();
        build_mode(&state.geometry, &mode)?;
        state.mode = Some(mode);
        Ok(())
    }

    /// The last h, k, l the pseudo axes were sent to.
    pub fn target(&self) -> Option<[f64; 3]> {
        self.state().target
    }

    /// Records reflection `name` of the current sample.
    pub fn add_reflection(&self, name: &str, measured: Measured) -> Result<(), DiffError> {
        let mut state = self.state_mut();
        reflection(&state.geometry, &measured)?;
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
        let mut guard = self.state_mut();
        let state = &mut *guard;
        let entry = state.samples.current_mut();
        let orientation = Orientation::busing_levy(
            reflection(&state.geometry, entry.reflection(primary)?)?,
            reflection(&state.geometry, entry.reflection(secondary)?)?,
        )?;
        entry.sample.orientation = orientation;
        Ok(entry.sample.ub())
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
        self.state().hkl(positions, wavelength)
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
        let state = self.state();
        let mode = require_mode(&state.geometry, state.mode.as_ref())?;
        state.inverse(mode, hkl, current, wavelength)
    }

    /// The readbacks, the wavelength, and the geometry, sample and mode from
    /// one read of the state.
    pub(crate) async fn snapshot(&self) -> Result<Snapshot, BsrsError> {
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        let state = self.state();
        Ok(Snapshot {
            geometry: state.geometry.clone(),
            current,
            wavelength,
            sample: state.samples.current().sample.clone(),
            mode: state.mode.clone(),
        })
    }

    fn scan(&self) -> std::sync::MutexGuard<'_, Option<&'static str>> {
        self.scan.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Holds the reals for `plan`'s run, once any move in flight has set its
    /// reals; refused while another scan holds them.
    pub(crate) async fn claim(self: &Arc<Self>, plan: &'static str) -> Result<Scanning, DiffError> {
        let _gate = self.moving.lock().await;
        let mut scan = self.scan();
        if let Some(holder) = *scan {
            return Err(DiffError::Scanning {
                plan: holder.to_owned(),
            });
        }
        *scan = Some(plan);
        Ok(Scanning { dx: self.clone() })
    }

    /// The only move that solves and commits a pseudo setpoint: solves
    /// `setpoint` closest to the readbacks and sets every real; the
    /// setpoint is kept only when the solve succeeds. Refused while a scan
    /// holds the reals.
    pub(crate) async fn move_to(&self, setpoint: Setpoint) -> Result<Status, BsrsError> {
        let _gate = self.moving.lock().await;
        if let Some(plan) = *self.scan() {
            return Err(DiffError::Scanning {
                plan: plan.to_owned(),
            }
            .into());
        }
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        // one write lock, so a set_mode meanwhile is not lost to a ψ commit
        let values = self.state_mut().commit(setpoint, &current, wavelength)?;
        let sets = self.reals.iter().zip(values).map(|(r, v)| r.set_dyn(v));
        let statuses = join_all(sets).await;
        Ok(statuses.iter().fold(Status::done(), |all, s| all.and(s)))
    }

    /// `component` of the pseudo setpoint and at the readbacks.
    pub(crate) async fn locate(&self, component: Component) -> Result<DynLocation, BsrsError> {
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        Ok(self.state().locate(component, &current, wavelength)?)
    }

    /// h, k, l and, on a geometry with a psi mode, ψ at `current` under one
    /// state, as the reading carries them: ψ is NaN where it has no value.
    pub(crate) fn pseudos(
        &self,
        current: &[f64],
        wavelength: Wavelength,
    ) -> Result<(Vector3<f64>, Option<f64>), DiffError> {
        let state = self.state();
        let psi = self.psi_key.as_ref();
        let psi = psi.map(|_| state.reading(Component::Psi, current, wavelength));
        Ok((state.hkl(current, wavelength)?, psi.transpose()?))
    }

    /// `component` at the readbacks for a reading: NaN where it has no value.
    pub(crate) async fn reading(&self, component: Component) -> Result<f64, BsrsError> {
        let (current, wavelength) = try_join!(self.positions(), self.wavelength())?;
        Ok(self.state().reading(component, &current, wavelength)?)
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
                    build_mode(&next.geometry, &mode)?;
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

pub(crate) fn rows(m: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|i| [m[(i, 0)], m[(i, 1)], m[(i, 2)]])
}

impl NamedObj for Diffractometer {
    fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl ReadableObj for Diffractometer {
    /// Every axis readback and the h, k, l computed from those same
    /// readbacks, and on a geometry with a psi mode ψ as the ψ axis reads
    /// it.
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        let (positions, wavelength) = futures::try_join!(self.positions(), self.wavelength())?;
        let (hkl, psi) = self.pseudos(&positions, wavelength)?;
        let t = now();
        let values = positions.iter().chain(hkl.iter()).chain(psi.iter());
        let keys = self.axis_keys().into_iter().chain(self.hkl_keys());
        let keys = keys.chain(self.psi_key.clone());
        Ok(keys
            .zip(values)
            .map(|(k, &v)| (k, ReadingValue::new(json!(v), t)))
            .collect())
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        let state = self.state();
        let axes = state.geometry.axes().iter().map(|a| {
            let units = match a.kind() {
                AxisKind::Rotation { .. } | AxisKind::RotationAbout { .. } => Some("deg"),
                AxisKind::Translation { .. } => Some("m"),
                AxisKind::Parameter => None,
            };
            (self.key(a.name()), units)
        });
        let hkl = self.hkl_keys().into_iter().zip(0..);
        let hkl = hkl.map(|(k, i)| (k, Component::Hkl(i).units()));
        let psi = self.psi_key.clone().map(|k| (k, Component::Psi.units()));
        Ok(axes
            .chain(hkl)
            .chain(psi)
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
            let sample = &state.samples.current().sample;
            configuration(&state.geometry, sample, mode, wavelength)
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
