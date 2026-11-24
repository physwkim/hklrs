//! hklpy2's configuration file (schema 3, hkl_soleil solver, hkl engine)
//! read into a [`Config`], so its samples, limits and mode carry over: the
//! sample frame of an hkl geometry is hklrs's.

use std::collections::BTreeMap;

use hklrs::nalgebra::Matrix3;
use hklrs::presets::preset;
use hklrs::{Geometry, Lattice, ModeTable, Orientation, Reflection, Sample};
use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::DiffError;
use crate::config::{
    AxisConfig, Config, GeometryConfig, LatticeConfig, ModeConfig, ReflectionConfig,
    SCHEMA_VERSION, SampleConfig,
};
use crate::diffractometer::reflection;

const SCHEMA: u64 = 3;
const FORMAT: &str = "hklpy2 YAML";
/// Relative difference of two UBs still taken as equal: the file holds U, UB
/// and the reflections rounded to 12 decimals.
const UB_TOLERANCE: f64 = 1e-6;

#[derive(Deserialize)]
struct File {
    #[serde(rename = "_header")]
    header: Header,
    axes: Axes,
    sample_name: String,
    samples: BTreeMap<String, SampleFile>,
    /// By local real axis name.
    constraints: BTreeMap<String, Constraint>,
    solver: Solver,
    beam: Beam,
    /// Held axis values by mode.
    #[serde(default)]
    presets: BTreeMap<String, BTreeMap<String, f64>>,
}

#[derive(Deserialize)]
struct Header {
    config_schema_version: u64,
}

#[derive(Deserialize)]
struct Axes {
    pseudo_axes: Vec<String>,
    real_axes: Vec<String>,
    /// Local name to solver name.
    axes_xref: BTreeMap<String, String>,
    /// The parameters of every mode, one value each.
    #[serde(default)]
    extra_axes: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct SampleFile {
    lattice: LatticeFile,
    reflections: BTreeMap<String, ReflectionFile>,
    /// The first two orient the sample.
    reflections_order: Vec<String>,
    #[serde(rename = "U")]
    u: [[f64; 3]; 3],
    /// With 2π.
    #[serde(rename = "UB")]
    ub: [[f64; 3]; 3],
}

#[derive(Deserialize)]
struct LatticeFile {
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    angle_units: String,
    length_units: String,
}

#[derive(Deserialize)]
struct ReflectionFile {
    geometry: String,
    /// By local pseudo axis name.
    pseudos: BTreeMap<String, f64>,
    /// By local real axis name.
    reals: BTreeMap<String, f64>,
    reals_units: String,
    wavelength: f64,
    wavelength_units: String,
}

#[derive(Deserialize)]
struct Constraint {
    low_limit: f64,
    high_limit: f64,
    cut_point: f64,
    class: String,
}

#[derive(Deserialize)]
struct Solver {
    name: String,
    geometry: String,
    engine: String,
    mode: String,
}

#[derive(Deserialize)]
struct Beam {
    wavelength: f64,
    wavelength_units: String,
}

/// Whether `text` is hklpy2's: a mapping with its `_header`.
pub(crate) fn is_hklpy2(text: &str) -> bool {
    serde_saphyr::from_str::<BTreeMap<String, IgnoredAny>>(text)
        .is_ok_and(|m| m.contains_key("_header"))
}

fn unsupported(field: &str, value: impl std::fmt::Debug) -> DiffError {
    DiffError::Hklpy2Unsupported {
        field: field.to_owned(),
        value: format!("{value:?}"),
    }
}

fn require(field: &str, value: &str, accepted: &[&str]) -> Result<(), DiffError> {
    if accepted.contains(&value) {
        Ok(())
    } else {
        Err(unsupported(field, value))
    }
}

fn missing(field: String) -> DiffError {
    DiffError::ConfigMissing { field }
}

impl Config {
    /// hklpy2's exported configuration. U and the lattice orient each sample.
    /// hklpy2 leaves UB stale when U is set and U stale when UB is
    /// (PY2-21), so where the file's UB is not U·B the UB of the two
    /// orienting reflections decides which is meant; a sample without them,
    /// or whose reflections match neither, is rejected. hklpy2's `digits`,
    /// `forward_solution_function` and `auxiliary_axes` have no counterpart
    /// and are not read; presets are rejected.
    pub fn from_hklpy2(text: &str) -> Result<Self, DiffError> {
        let file: File = serde_saphyr::from_str(text).map_err(|e| DiffError::ConfigParse {
            format: FORMAT,
            message: e.to_string(),
        })?;
        if file.header.config_schema_version != SCHEMA {
            return Err(DiffError::ConfigSchema {
                format: FORMAT,
                found: file.header.config_schema_version,
                expected: SCHEMA,
            });
        }
        require("solver.name", &file.solver.name, &["hkl_soleil"])?;
        require("solver.engine", &file.solver.engine, &["hkl"])?;
        require(
            "beam.wavelength_units",
            &file.beam.wavelength_units,
            &["angstrom"],
        )?;
        for (mode, held) in &file.presets {
            if !held.is_empty() {
                return Err(unsupported(&format!("presets.{mode}"), held));
            }
        }
        let geometry = preset(&file.solver.geometry)?;
        let names = Names::new(&file, &geometry)?;
        let axes = names
            .reals
            .iter()
            .map(|(solver, local)| {
                let c = file
                    .constraints
                    .get(local)
                    .ok_or_else(|| missing(format!("constraints.{local}")))?;
                require(
                    &format!("constraints.{local}.class"),
                    &c.class,
                    &["LimitsConstraint"],
                )?;
                Ok(AxisConfig {
                    name: solver.clone(),
                    limits: [Some(c.low_limit), Some(c.high_limit)],
                    cut_point: Some(c.cut_point),
                })
            })
            .collect::<Result<_, DiffError>>()?;
        let samples = file
            .samples
            .iter()
            .map(|(name, s)| {
                s.config(name, &geometry, &names)
                    .map_err(|source| DiffError::InSample {
                        sample: name.clone(),
                        source: Box::new(source),
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(Config {
            schema_version: SCHEMA_VERSION,
            geometry: GeometryConfig {
                name: file.solver.geometry.clone(),
                axes,
            },
            samples,
            sample: file.sample_name,
            mode: Some(mode(&geometry, &file.solver.mode, &file.axes.extra_axes)?),
            wavelength: Some(file.beam.wavelength),
        })
    }
}

/// The file's local axis names and the solver (hklrs) names they stand for.
struct Names {
    /// (solver, local) of each real axis, in the file's order.
    reals: Vec<(String, String)>,
    /// The local names of h, k and l.
    hkl: [String; 3],
}

impl Names {
    fn new(file: &File, geometry: &Geometry) -> Result<Self, DiffError> {
        let xref = |local: &String| {
            let solver = file.axes.axes_xref.get(local);
            let solver = solver.ok_or_else(|| missing(format!("axes.axes_xref.{local}")))?;
            Ok((solver.clone(), local.clone()))
        };
        if let Some(a) = geometry.axes().iter().find(|a| !a.is_rotation()) {
            // reals_units is one unit for every axis
            return Err(unsupported("translation axis", a.name()));
        }
        let reals = file.axes.real_axes.iter().map(xref);
        let pseudos: BTreeMap<String, String> = file
            .axes
            .pseudo_axes
            .iter()
            .map(xref)
            .collect::<Result<_, DiffError>>()?;
        let local = |solver: &str| {
            let local = pseudos.get(solver).cloned();
            local.ok_or_else(|| missing(format!("pseudo axis {solver}")))
        };
        Ok(Self {
            reals: reals.collect::<Result<_, DiffError>>()?,
            hkl: [local("h")?, local("k")?, local("l")?],
        })
    }

    fn solver(&self, local: &str) -> Result<&str, DiffError> {
        let found = self.reals.iter().find(|(_, l)| l == local);
        let found = found.ok_or_else(|| missing(format!("axes.real_axes {local}")))?;
        Ok(&found.0)
    }
}

impl SampleFile {
    fn config(
        &self,
        name: &str,
        geometry: &Geometry,
        names: &Names,
    ) -> Result<SampleConfig, DiffError> {
        let l = &self.lattice;
        require("lattice.angle_units", &l.angle_units, &["degrees"])?;
        require("lattice.length_units", &l.length_units, &["angstrom"])?;
        let lattice = Lattice::from_degrees(l.a, l.b, l.c, l.alpha, l.beta, l.gamma)?;
        let reflections: Vec<ReflectionConfig> = self
            .reflections
            .iter()
            .map(|(r, file)| {
                file.config(r, geometry.name(), names)
                    .map_err(|source| DiffError::InReflection {
                        name: r.clone(),
                        source: Box::new(source),
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(SampleConfig {
            u: self.orientation(&lattice, geometry, &reflections)?,
            name: name.to_owned(),
            lattice: LatticeConfig {
                a: l.a,
                b: l.b,
                c: l.c,
                alpha: l.alpha,
                beta: l.beta,
                gamma: l.gamma,
            },
            reflections,
        })
    }

    /// The file's U where its UB is U·B; otherwise the U of whichever of
    /// the two the orienting reflections give.
    fn orientation(
        &self,
        lattice: &Lattice,
        geometry: &Geometry,
        reflections: &[ReflectionConfig],
    ) -> Result<[[f64; 3]; 3], DiffError> {
        let ub = Matrix3::from_fn(|i, j| self.ub[i][j]);
        let u_b = Matrix3::from_fn(|i, j| self.u[i][j]) * lattice.b_matrix();
        if same(&ub, &u_b) {
            return Ok(self.u);
        }
        let [r1, r2, ..] = &self.reflections_order[..] else {
            return Err(DiffError::Hklpy2UbUndecided {
                ub: Box::new(self.ub),
                u_b: Box::new(rows(&u_b)),
            });
        };
        let measure = |name: &String| -> Result<Reflection, DiffError> {
            let r = reflections.iter().find(|r| &r.name == name);
            let r = r.ok_or_else(|| missing(format!("reflections.{name}")))?;
            reflection(geometry, &r.load(geometry)?)
        };
        let oriented = Orientation::busing_levy(measure(r1)?, measure(r2)?)?;
        let sample = Sample::new("", *lattice, oriented);
        let measured = sample.ub();
        match (same(&measured, &ub), same(&measured, &u_b)) {
            (true, false) => Ok(rows(&sample.u().to_rotation_matrix().into_inner())),
            (false, true) => Ok(self.u),
            _ => Err(DiffError::Hklpy2UbUnmatched {
                ub: Box::new(self.ub),
                u_b: Box::new(rows(&u_b)),
                reflections: [r1.clone(), r2.clone()],
                measured: Box::new(rows(&measured)),
            }),
        }
    }
}

/// Equal to `b` within [`UB_TOLERANCE`] of its largest element; never for
/// NaN.
fn same(a: &Matrix3<f64>, b: &Matrix3<f64>) -> bool {
    let tolerance = UB_TOLERANCE * b.amax();
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
}

fn rows(m: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[(i, j)]))
}

impl ReflectionFile {
    fn config(
        &self,
        name: &str,
        geometry: &str,
        names: &Names,
    ) -> Result<ReflectionConfig, DiffError> {
        require("geometry", &self.geometry, &[geometry])?;
        // hklpy2 records its angles as angstrom (PY2-13)
        require("reals_units", &self.reals_units, &["angstrom", "degrees"])?;
        require("wavelength_units", &self.wavelength_units, &["angstrom"])?;
        let index = |local: &String| {
            self.pseudos
                .get(local)
                .copied()
                .ok_or_else(|| missing(format!("pseudos.{local}")))
        };
        let [h, k, l] = &names.hkl;
        let positions = self
            .reals
            .iter()
            .map(|(local, &v)| Ok((names.solver(local)?.to_owned(), v)))
            .collect::<Result<_, DiffError>>()?;
        Ok(ReflectionConfig {
            name: name.to_owned(),
            hkl: [index(h)?, index(k)?, index(l)?],
            positions,
            wavelength: self.wavelength,
        })
    }
}

/// hklpy2 keeps one value per parameter name across modes; mode `name` of
/// `geometry`'s preset takes exactly those hkl declares for it.
fn mode(
    geometry: &Geometry,
    name: &str,
    extras: &BTreeMap<String, f64>,
) -> Result<ModeConfig, DiffError> {
    let declared = ModeTable::of(geometry.name())
        .and_then(|table| table.parameters(name))
        .map_err(|source| DiffError::Mode {
            mode: name.to_owned(),
            source,
        })?;
    let parameters = declared.iter().map(|p| {
        let value = extras.get(p.name).copied();
        let value = value.ok_or_else(|| missing(format!("axes.extra_axes.{}", p.name)))?;
        Ok((p.name.to_owned(), value))
    });
    Ok(ModeConfig {
        name: name.to_owned(),
        parameters: parameters.collect::<Result<_, DiffError>>()?,
    })
}
