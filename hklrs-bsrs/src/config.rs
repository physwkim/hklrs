//! A diffractometer's configuration as data: the geometry's limits and cut
//! points, the samples with their lattice, U and reflections, the current
//! sample, the mode and a fixed wavelength. Degrees for rotations, metres
//! for translations, Å for lattice constants and wavelengths.

use std::collections::BTreeMap;
use std::path::Path;

use hklrs::nalgebra::{Matrix3, Vector3};
use hklrs::{Deg, Geometry, Lattice, ModeSpec, Orientation, Sample, Wavelength};
use serde::{Deserialize, Serialize};

use crate::diffractometer::{
    WavelengthSource, build_mode, default_mode, reflection, rows, set_limits,
};
use crate::samples::{Entry, Samples};
use crate::{DiffError, Measured};

/// The [`Config::schema_version`] this crate reads and writes.
pub const SCHEMA_VERSION: u64 = 1;

/// Axis values by geometry axis name.
pub(crate) type Positions = BTreeMap<String, f64>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u64,
    pub geometry: GeometryConfig,
    pub samples: Vec<SampleConfig>,
    /// The current sample's name.
    pub sample: String,
    /// None only for a geometry without hkl modes.
    pub mode: Option<ModeConfig>,
    /// Å; None for a device that reads it from a signal.
    pub wavelength: Option<f64>,
}

/// Checked against the device's geometry, which it does not replace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryConfig {
    pub name: String,
    /// In geometry axis order.
    pub axes: Vec<AxisConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisConfig {
    pub name: String,
    /// `[min, max]`; null is unbounded.
    pub limits: [Option<f64>; 2],
    /// Degrees; given exactly for a rotation.
    pub cut_point: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleConfig {
    pub name: String,
    pub lattice: LatticeConfig,
    /// Rows; the orientation, whether set or computed from reflections.
    #[serde(rename = "U")]
    pub u: [[f64; 3]; 3],
    pub reflections: Vec<ReflectionConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeConfig {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReflectionConfig {
    pub name: String,
    pub hkl: [f64; 3],
    /// Every geometry axis by name.
    pub positions: Positions,
    pub wavelength: f64,
}

/// A [`ModeSpec`] with ψ in degrees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hkl2: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psi: Option<PsiConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PsiConfig {
    pub reference: [f64; 3],
    /// Degrees.
    pub angle: f64,
}

impl From<&ModeConfig> for ModeSpec {
    fn from(m: &ModeConfig) -> Self {
        ModeSpec {
            name: m.name.clone(),
            hkl2: m.hkl2.map(Vector3::from),
            psi: m
                .psi
                .map(|p| (Vector3::from(p.reference), p.angle.to_radians())),
        }
    }
}

impl From<&ModeSpec> for ModeConfig {
    fn from(m: &ModeSpec) -> Self {
        ModeConfig {
            name: m.name.clone(),
            hkl2: m.hkl2.map(Into::into),
            psi: m.psi.map(|(reference, angle)| PsiConfig {
                reference: reference.into(),
                angle: angle.to_degrees(),
            }),
        }
    }
}

/// `positions` in geometry axis order: every axis, no other name.
pub(crate) fn ordered(geometry: &Geometry, positions: &Positions) -> Result<Vec<f64>, DiffError> {
    for name in positions.keys() {
        geometry.axis_index(name)?;
    }
    geometry
        .axes()
        .iter()
        .map(|a| {
            positions
                .get(a.name())
                .copied()
                .ok_or_else(|| DiffError::MissingPosition {
                    axis: a.name().to_owned(),
                })
        })
        .collect()
}

pub(crate) fn named(geometry: &Geometry, values: &[f64]) -> Positions {
    let names = geometry.axes().iter().map(|a| a.name().to_owned());
    names.zip(values.iter().copied()).collect()
}

fn in_sample(sample: &str) -> impl Fn(DiffError) -> DiffError + '_ {
    move |source| DiffError::InSample {
        sample: sample.to_owned(),
        source: Box::new(source),
    }
}

impl Config {
    pub(crate) fn capture(
        geometry: &Geometry,
        samples: &Samples,
        mode: Option<&ModeSpec>,
        wavelength: &WavelengthSource,
    ) -> Self {
        let axes = geometry.axes().iter().map(|a| {
            let user = |v: f64| {
                if !v.is_finite() {
                    None
                } else if a.is_rotation() {
                    Some(v.to_degrees())
                } else {
                    Some(v)
                }
            };
            let (min, max) = a.limits();
            AxisConfig {
                name: a.name().to_owned(),
                limits: [user(min), user(max)],
                cut_point: a.cut_point().map(f64::to_degrees),
            }
        });
        let configs = samples.entries().map(|entry| {
            let s = &entry.sample;
            let l = &s.lattice;
            let deg = |r| Deg::from(r).0;
            let reflections = entry.reflections.iter().map(|(name, m)| ReflectionConfig {
                name: name.clone(),
                hkl: m.hkl,
                positions: named(geometry, &m.positions),
                wavelength: m.wavelength.angstrom(),
            });
            SampleConfig {
                name: s.name.clone(),
                lattice: LatticeConfig {
                    a: l.a(),
                    b: l.b(),
                    c: l.c(),
                    alpha: deg(l.alpha()),
                    beta: deg(l.beta()),
                    gamma: deg(l.gamma()),
                },
                u: rows(&s.u().to_rotation_matrix().into_inner()),
                reflections: reflections.collect(),
            }
        });
        Config {
            schema_version: SCHEMA_VERSION,
            geometry: GeometryConfig {
                name: geometry.name().to_owned(),
                axes: axes.collect(),
            },
            samples: configs.collect(),
            sample: samples.current_name().to_owned(),
            mode: mode.map(ModeConfig::from),
            wavelength: match wavelength {
                WavelengthSource::Fixed(w) => Some(w.angstrom()),
                WavelengthSource::Signal(_) => None,
            },
        }
    }

    /// `device` with this configuration's limits and cut points, and the
    /// samples and mode built against it.
    pub(crate) fn load(
        &self,
        device: &Geometry,
    ) -> Result<(Geometry, Samples, Option<ModeSpec>), DiffError> {
        let geometry = self.load_geometry(device)?;
        let entries = self
            .samples
            .iter()
            .map(|s| s.load(&geometry).map_err(in_sample(&s.name)))
            .collect::<Result<_, _>>()?;
        let samples = Samples::from_entries(entries, &self.sample)?;
        let mode = match &self.mode {
            Some(m) => {
                let spec = ModeSpec::from(m);
                build_mode(&geometry, &spec)?;
                Some(spec)
            }
            None if default_mode(&geometry).is_some() => {
                return Err(DiffError::ConfigMissing {
                    field: "mode".into(),
                });
            }
            None => None,
        };
        Ok((geometry, samples, mode))
    }

    fn load_geometry(&self, device: &Geometry) -> Result<Geometry, DiffError> {
        let found: Vec<&str> = self.geometry.axes.iter().map(|a| a.name.as_str()).collect();
        let expected: Vec<&str> = device.axes().iter().map(|a| a.name()).collect();
        if self.geometry.name != device.name() || found != expected {
            return Err(DiffError::ConfigGeometry {
                found: self.geometry.name.clone(),
                found_axes: found.iter().map(|&n| n.to_owned()).collect(),
                expected: device.name().to_owned(),
                expected_axes: expected.iter().map(|&n| n.to_owned()).collect(),
            });
        }
        let mut geometry = device.clone();
        for axis in &self.geometry.axes {
            let [min, max] = axis.limits;
            let min = min.unwrap_or(f64::NEG_INFINITY);
            let max = max.unwrap_or(f64::INFINITY);
            set_limits(&mut geometry, &axis.name, min, max)?;
            match axis.cut_point {
                Some(cut) => geometry.set_cut_point(&axis.name, cut.to_radians())?,
                None if geometry.axes()[geometry.axis_index(&axis.name)?].is_rotation() => {
                    return Err(DiffError::ConfigMissing {
                        field: format!("geometry.axes.{}.cut_point", axis.name),
                    });
                }
                None => {}
            }
        }
        Ok(geometry)
    }

    pub fn from_json(text: &str) -> Result<Self, DiffError> {
        let parse = |e: serde_json::Error| DiffError::ConfigParse {
            format: "hklrs JSON",
            message: e.to_string(),
        };
        let config: Self = serde_json::from_str(text).map_err(parse)?;
        config.check_schema("hklrs JSON")
    }

    /// An hklrs configuration; [`Config::from_hklpy2`] reads hklpy2's.
    pub fn from_yaml(text: &str) -> Result<Self, DiffError> {
        let parse = |e: serde_saphyr::Error| DiffError::ConfigParse {
            format: "hklrs YAML",
            message: e.to_string(),
        };
        let config: Self = serde_saphyr::from_str(text).map_err(parse)?;
        config.check_schema("hklrs YAML")
    }

    fn check_schema(self, format: &'static str) -> Result<Self, DiffError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(DiffError::ConfigSchema {
                format,
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(self)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("every field serializes")
    }

    pub fn to_yaml(&self) -> String {
        serde_saphyr::to_string(self).expect("every field serializes")
    }

    /// `.json` is hklrs JSON; `.yml` and `.yaml` are hklpy2's when they
    /// carry its `_header` and hklrs YAML otherwise.
    pub fn read(path: &Path) -> Result<Self, DiffError> {
        let format = Format::of(path)?;
        let text = std::fs::read_to_string(path).map_err(|e| file_error(path, &e))?;
        match format {
            Format::Json => Self::from_json(&text),
            Format::Yaml if crate::hklpy2::is_hklpy2(&text) => Self::from_hklpy2(&text),
            Format::Yaml => Self::from_yaml(&text),
        }
    }

    /// As hklrs JSON for `.json`, hklrs YAML for `.yml` and `.yaml`.
    pub fn write(&self, path: &Path) -> Result<(), DiffError> {
        let text = match Format::of(path)? {
            Format::Json => self.to_json(),
            Format::Yaml => self.to_yaml(),
        };
        std::fs::write(path, text).map_err(|e| file_error(path, &e))
    }
}

impl SampleConfig {
    fn load(&self, geometry: &Geometry) -> Result<Entry, DiffError> {
        let l = &self.lattice;
        let lattice = Lattice::from_degrees(l.a, l.b, l.c, l.alpha, l.beta, l.gamma)?;
        let u = Matrix3::from_fn(|i, j| self.u[i][j]);
        let orientation = Orientation::from_u_matrix(&u)?;
        let mut reflections = BTreeMap::new();
        for r in &self.reflections {
            let in_reflection = |source| DiffError::InReflection {
                name: r.name.clone(),
                source: Box::new(source),
            };
            let measured = r.load(geometry).map_err(in_reflection)?;
            if reflections.insert(r.name.clone(), measured).is_some() {
                return Err(DiffError::DuplicateReflection {
                    sample: self.name.clone(),
                    name: r.name.clone(),
                });
            }
        }
        Ok(Entry {
            sample: Sample::new(&self.name, lattice, orientation),
            reflections,
        })
    }
}

impl ReflectionConfig {
    pub(crate) fn load(&self, geometry: &Geometry) -> Result<Measured, DiffError> {
        let measured = Measured {
            hkl: self.hkl,
            positions: ordered(geometry, &self.positions)?,
            wavelength: Wavelength::new(self.wavelength)?,
        };
        reflection(geometry, &measured)?;
        Ok(measured)
    }
}

enum Format {
    Json,
    Yaml,
}

impl Format {
    fn of(path: &Path) -> Result<Self, DiffError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("json") => Ok(Self::Json),
            Some("yml" | "yaml") => Ok(Self::Yaml),
            _ => Err(DiffError::ConfigFile {
                path: path.display().to_string(),
                message: "the extension is not .json, .yml or .yaml".into(),
            }),
        }
    }
}

fn file_error(path: &Path, e: &std::io::Error) -> DiffError {
    DiffError::ConfigFile {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}
