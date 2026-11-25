//! Inverse: axis values that bring a [`Target`] about under a [`Mode`].
//!
//! A Q target gives the three hkl equations R_sample·q = k̂f − k̂i, with q in
//! units of k = 2π/λ and k̂f the detector rotation applied to the beam
//! direction (a detector at infinity, as in hkl). Where no written axis
//! turns the sample or the beam, Q and k̂i stay where `current` puts them,
//! so the target fixes k̂f = k̂i + R_sample·q: it must lie on the Ewald
//! sphere, and gives the two equations of k̂f's direction.
//! [`Target::Eulerians`] gives three equations of the kappa arm's rotation;
//! [`Target::Detector`] none. The mode's extra [`Equation`]s close the
//! system over its written axes, and its [`Slits`], if any, are turned into
//! the surface after. Other axes stay at `current`.

mod analytic;
pub mod aps_04id_polar;
pub mod e4c;
pub mod e6c;
pub mod esrf_bm28_psic;
pub mod esrf_id01_psic;
mod jet;
pub mod k4c;
pub mod k6c;
pub mod kappa;
mod mode;
pub mod petra3_p09_eh2;
pub mod petra3_p23_4c;
pub mod petra3_p23_6c;
pub mod pseudo;
pub mod pyfai;
pub mod soleil_nanoscopium_robot;
pub mod soleil_sirius_turret;
pub mod soleil_sixs_med;
mod solve;
mod spec;
mod system;
mod validate;
pub mod zaxis;

use std::f64::consts::{PI, TAU};

use nalgebra::{Matrix3, Unit, UnitQuaternion, Vector3};

use crate::{Error, Geometry, Poses, Wavelength};

pub use mode::{Equation, Eulerian, Mode, Slits};
pub use spec::{ModeParameter, ModeSpec, ModeTable, PsiEngine};

/// Relative excess of |Q| over 2k still solved, pulled onto 2k: UB·h alone
/// rounds by ~1e-15, so a strict test would turn exact backscatter targets
/// Unreachable or not by rounding.
const REACH: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// Miller indices; Q = UB·h in the sample frame.
    Hkl(Vector3<f64>),
    /// Q in the sample frame, Å⁻¹ with 2π.
    QSample(Vector3<f64>),
    /// No Q: the mode's equations alone place the detector, as hkl's `q`,
    /// `q2`, `tth2` and `qper_qpar` engines do.
    Detector,
    /// (omega, chi, phi), rad: the kappa arm of a [`Mode::new_kappa`] mode
    /// turns to the rotation of this eulerian cradle, chi modulo 2π
    /// ([`kappa::KappaArm::to_kappa`]), at either kappa image, as hkl's
    /// `eulerians` engine does.
    Eulerians([f64; 3]),
}

impl Target {
    /// How many equations the target itself gives `mode`.
    fn rows(&self, mode: &Mode) -> usize {
        match self {
            Target::Hkl(_) | Target::QSample(_) if fixes_kf(mode) => 2,
            Target::Hkl(_) | Target::QSample(_) | Target::Eulerians(_) => 3,
            Target::Detector => 0,
        }
    }
}

/// No written axis of `mode` turns the sample or the beam: a Q target fixes
/// k̂f in the lab.
fn fixes_kf(mode: &Mode) -> bool {
    let g = mode.geometry();
    let turns = |i: &usize| g.sample_stack().contains(i) || g.source_stack().contains(i);
    !mode.written().iter().any(turns)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Picker {
    /// Nearest to `current`: the head of the sorted set.
    Closest,
    /// The first solution within limits the search reached from `current`.
    First,
}

#[derive(Clone, Debug)]
pub struct InverseRequest<'a> {
    /// The mode, which carries the geometry it was validated against.
    pub mode: &'a Mode<'a>,
    /// UB, Å⁻¹ with 2π.
    pub ub: Matrix3<f64>,
    pub wavelength: Wavelength,
    pub target: Target,
    /// Values of every axis in geometry order: the held values and the seed.
    pub current: &'a [f64],
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum InverseError {
    #[error("|Q| = {q} Å⁻¹ exceeds 4π/λ = {q_max} Å⁻¹")]
    Unreachable { q: f64, q_max: f64 },
    #[error("no axis values satisfy the mode")]
    NoSolution,
    #[error(
        "Q = {q:?} Å⁻¹ misses the Ewald sphere by {miss} Å⁻¹, and no written axis turns \
         the sample or the beam"
    )]
    OffEwaldSphere { q: [f64; 3], miss: f64 },
    #[error("{what} is not finite")]
    NonFinite { what: &'static str },
    #[error("{what} is zero")]
    ZeroReference { what: &'static str },
    #[error("axis index {index} is outside the geometry's {count} axes")]
    NoSuchAxis { index: usize, count: usize },
    #[error("axis {name:?} is not a rotation")]
    NotRotation { name: String },
    #[error("axis {name:?} is written twice")]
    WrittenTwice { name: String },
    #[error("axis {name:?} is not in the {holder} stack")]
    NotInStack { name: String, holder: &'static str },
    #[error(
        "{written} written axes need {written} equations, the target and the mode give {equations}"
    )]
    EquationCount { written: usize, equations: usize },
    #[error(
        "turning the slits on axis {name:?} would move k̂f: it must be the innermost detector axis and turn about the beam"
    )]
    SlitsMoveDetector { name: String },
    #[error("emergence {emergence} rad is outside [-π/2, π/2]")]
    EmergenceRange { emergence: f64 },
    #[error("psi is undefined: Q, k̂f × k̂i or the reference projected normal to Q is null")]
    PsiUndefined,
    #[error("kappa arm tilt alpha = {alpha} rad is outside (0, π/2]")]
    KappaAlpha { alpha: f64 },
    #[error("|chi| = {} rad exceeds the kappa reach 2·alpha = {reach} rad", chi.abs())]
    ChiOutOfReach { chi: f64, reach: f64 },
    #[error(
        "{names:?} are not a kappa arm: komega, kappa, kphi adjacent in the sample stack, \
         komega and kphi about one direction"
    )]
    NotKappaArm { names: [String; 3] },
    #[error("a kappa equation needs a mode made with Mode::new_kappa")]
    NoKappaArm,
    #[error("geometry {geometry:?} is not a preset: it has no named modes")]
    NotPreset { geometry: String },
    #[error("{geometry} has no mode {name:?}")]
    UnknownMode { geometry: String, name: String },
    #[error("mode {mode:?} needs parameter {parameter:?}")]
    MissingParameter {
        mode: String,
        parameter: &'static str,
    },
    #[error("mode {mode:?} takes no parameter {parameter:?}")]
    UnknownParameter { mode: String, parameter: String },
    #[error("mode {mode:?} takes parameter {parameter:?} as 0 or 1, not {value}")]
    NotSwitch {
        mode: String,
        parameter: &'static str,
        value: f64,
    },
    #[error(transparent)]
    Geometry(#[from] Error),
    #[error(
        "the mode does not isolate the solutions: its Jacobian is singular at one \
         (a continuum, or a multiple root)"
    )]
    Degenerate,
    /// Every solution, sorted by distance to `current`.
    #[error(
        "every one of the {} solutions takes an axis outside its limits",
        .0.len()
    )]
    NoSolutionInLimits(Vec<Vec<f64>>),
}

/// Residue of `a` modulo 2π in (−π, π].
fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r > PI { r - TAU } else { r }
}

fn finite(v: &[f64], what: &'static str) -> Result<(), InverseError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(InverseError::NonFinite { what })
    }
}

impl<'a> InverseRequest<'a> {
    /// Every solution found, written axes wrapped to their cut points, held
    /// axes exactly `current`, every axis within its limits, sorted by
    /// distance to `current`; `Degenerate` when one is not
    /// isolated, as where the mode leaves a rotation free, and for Q = 0;
    /// `NoSolutionInLimits` when every one is outside the limits.
    pub fn solve(&self) -> Result<Vec<Vec<f64>>, InverseError> {
        Ok(self.search()?.solutions)
    }

    pub fn pick(&self, picker: Picker) -> Result<Vec<f64>, InverseError> {
        let mut found = self.search()?;
        let i = match picker {
            Picker::Closest => 0,
            Picker::First => found.first,
        };
        Ok(found.solutions.swap_remove(i))
    }

    fn search(&self) -> Result<validate::Found, InverseError> {
        let system = self.system()?;
        analytic::search(&system)
            .unwrap_or_else(|| solve::search(&system))
            .into_sorted()
    }

    /// The target's and the mode's equations, one per written axis, once
    /// they pass the reach gate.
    fn system(&self) -> Result<system::System<'a>, InverseError> {
        let geometry = self.mode.geometry();
        let poses = geometry.poses(self.current)?;
        finite(self.ub.as_slice(), "UB")?;
        let (written, extra) = (self.mode.written().len(), self.mode.extra());
        let equations = self.target.rows(self.mode) + extra.len();
        if written != equations {
            return Err(InverseError::EquationCount { written, equations });
        }
        let k = self.wavelength.wavenumber();
        let q_max = 2.0 * k;
        reach(detector_q(extra), q_max)?;
        let goal = match self.target {
            Target::Hkl(h) => {
                finite(h.as_slice(), "target hkl")?;
                self.q_goal(self.ub * h, &poses)?
            }
            Target::QSample(q) => {
                finite(q.as_slice(), "target Q")?;
                self.q_goal(q, &poses)?
            }
            Target::Detector => system::Goal::Detector,
            Target::Eulerians(eulerians) => {
                let kappa = self.mode.kappa().ok_or(InverseError::NoKappaArm)?;
                let at = kappa.arm().to_kappa(eulerians, kappa::Branch::Same)?;
                let rotation = kappa
                    .axes()
                    .iter()
                    .zip(at)
                    .fold(UnitQuaternion::identity(), |r, (&i, v)| {
                        r * geometry.axes()[i].transform(v).rotation
                    });
                system::Goal::Arm {
                    eulerians,
                    rotation,
                }
            }
        };
        Ok(system::System::new(
            self.mode,
            self.ub / k,
            goal,
            k,
            self.current,
        ))
    }

    /// The goal of Q = `q` in the sample frame, Å⁻¹, pulled onto 2k past
    /// the reach gate: R_sample·q = k̂f − k̂i in units of k, or k̂f itself
    /// where the mode [`fixes_kf`], `poses` those of `current`.
    fn q_goal(&self, q: Vector3<f64>, poses: &Poses) -> Result<system::Goal, InverseError> {
        let k = self.wavelength.wavenumber();
        let q_max = 2.0 * k;
        let q_norm = q.norm();
        reach(q_norm, q_max)?;
        let pulled = if q_norm > q_max {
            q * (q_max / q_norm)
        } else {
            q
        } / k;
        if fixes_kf(self.mode) {
            let ki = poses.source * self.mode.geometry().beam_direction().into_inner();
            let kf = ki + poses.sample.rotation * pulled;
            let miss = kf.norm() - 1.0;
            if miss.abs() > REACH {
                return Err(InverseError::OffEwaldSphere {
                    q: q.into(),
                    miss: miss * k,
                });
            }
            return Ok(system::Goal::ray(Unit::new_normalize(kf)));
        }
        // k̂f = k̂i then, which leaves the sample rotation free in any mode
        // that turns it.
        if q_norm == 0.0 {
            return Err(InverseError::Degenerate);
        }
        Ok(system::Goal::Q(pulled))
    }
}

/// `Unreachable` for a |Q| of `q` Å⁻¹ beyond `q_max` = 2k.
fn reach(q: f64, q_max: f64) -> Result<(), InverseError> {
    if q > q_max * (1.0 + REACH) {
        Err(InverseError::Unreachable { q, q_max })
    } else {
        Ok(())
    }
}

/// The largest |Q| the extra equations ask for, Å⁻¹: a qper and the qpar
/// of the same surface count as their hypotenuse.
fn detector_q(extra: &[Equation]) -> f64 {
    let component = |surface: &Vector3<f64>, per: bool| {
        extra
            .iter()
            .find_map(|e| match *e {
                Equation::Qper { surface: s, qper } if per && s == *surface => Some(qper),
                Equation::Qpar { surface: s, qpar } if !per && s == *surface => Some(qpar),
                _ => None,
            })
            .unwrap_or(0.0)
    };
    extra
        .iter()
        .map(|e| match e {
            Equation::QNorm { q } | Equation::QSigned { q } => q.abs(),
            Equation::Qper { surface, .. } | Equation::Qpar { surface, .. } => {
                component(surface, true).hypot(component(surface, false))
            }
            _ => 0.0,
        })
        .fold(0.0, f64::max)
}

/// ψ of `reference` (Miller indices) at `values`: the angle about Q̂ from
/// n = (k̂f × k̂i) × Q̂ to the reference projected normal to Q, positive when
/// n × reference is along Q̂ (hkl's psi_constant init, `hkl_psi_write`).
/// Independent of the wavelength.
pub fn psi(
    geometry: &Geometry,
    ub: &Matrix3<f64>,
    reference: &Vector3<f64>,
    values: &[f64],
) -> Result<f64, InverseError> {
    finite(ub.as_slice(), "UB")?;
    finite(reference.as_slice(), "psi reference")?;
    system::psi_at(geometry, &(ub * reference), values)?.ok_or(InverseError::PsiUndefined)
}
