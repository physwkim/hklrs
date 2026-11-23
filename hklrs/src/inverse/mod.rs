//! Inverse: axis values that bring a target Q into reflection under a
//! [`Mode`].
//!
//! Every mode implies the three hkl equations R_sample·q = k̂f − k̂i, with q in
//! units of k = 2π/λ and k̂f the detector rotation applied to the beam
//! direction (a detector at infinity, as in hkl); the mode's extra
//! [`Equation`]s close the system over its written axes. Non-written axes
//! stay at `current`.

pub mod e4c;
pub mod e6c;
pub mod esrf_bm28_psic;
pub mod esrf_id01_psic;
mod jet;
pub mod k4c;
pub mod kappa;
mod mode;
mod solve;
mod system;
mod validate;
pub mod zaxis;

use std::f64::consts::{PI, TAU};

use nalgebra::{Matrix3, Vector3};

use crate::{Error, Geometry, Wavelength};

pub use mode::{Equation, Eulerian, Mode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// Miller indices; Q = UB·h in the sample frame.
    Hkl(Vector3<f64>),
    /// Q in the sample frame, Å⁻¹ with 2π.
    QSample(Vector3<f64>),
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
    #[error("{written} written axes need {written} equations, the mode has {equations}")]
    EquationCount { written: usize, equations: usize },
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
    #[error(transparent)]
    Geometry(#[from] Error),
    #[error(
        "the mode does not isolate the solutions: its Jacobian is singular at one \
         (a continuum, or a multiple root)"
    )]
    Degenerate,
    /// Every solution, sorted by distance to `current`.
    #[error(
        "every one of the {} solutions takes a written axis outside its limits",
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

impl InverseRequest<'_> {
    /// Every solution found, written axes wrapped to their cut points and
    /// within their limits, held axes exactly `current` (limits or not),
    /// sorted by distance to `current`; `Degenerate` when one is not
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
        let geometry = self.mode.geometry();
        geometry.poses(self.current)?;
        finite(self.ub.as_slice(), "UB")?;
        let q = match self.target {
            Target::Hkl(h) => {
                finite(h.as_slice(), "target hkl")?;
                self.ub * h
            }
            Target::QSample(q) => {
                finite(q.as_slice(), "target Q")?;
                q
            }
        };
        let k = self.wavelength.wavenumber();
        let (q_norm, q_max) = (q.norm(), 2.0 * k);
        if q_norm > q_max {
            return Err(InverseError::Unreachable { q: q_norm, q_max });
        }
        // k̂f = k̂i then, which leaves the sample rotation free in any mode.
        if q_norm == 0.0 {
            return Err(InverseError::Degenerate);
        }
        let system = system::System::new(self.mode, self.ub / k, q / k, self.current);
        solve::search(&system).into_sorted()
    }
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
