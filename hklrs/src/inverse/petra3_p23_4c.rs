//! hkl's PETRA3 P23 4C "hkl" engine modes (`hkl-engine-petra3-p23-4c.c`),
//! for any geometry with sample axes omega_t, mu and detector axes gamma,
//! delta. Axes a mode does not write hold at `current`.

use nalgebra::Vector3;

use super::{Equation, InverseError, Mode};
use crate::Geometry;

fn written<'g>(
    geometry: &'g Geometry,
    name: &str,
    axes: &[&str],
) -> Result<Mode<'g>, InverseError> {
    Mode::named(geometry, name, axes, |_| vec![])
}

/// hkl's default mode: gamma held. No bisector equation, as in hkl: the
/// three hkl equations already fix the three written axes.
pub fn bissector_vertical(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "bissector_vertical", &["omega_t", "mu", "delta"])
}

pub fn lifting_detector_omega_t(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "lifting_detector_omega_t",
        &["omega_t", "gamma", "delta"],
    )
}

pub fn lifting_detector_mu(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting_detector_mu", &["mu", "gamma", "delta"])
}

/// delta held; no bisector equation, as for [`bissector_vertical`].
pub fn bissector_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "bissector_horizontal",
        &["omega_t", "mu", "gamma"],
    )
}

/// ψ of `reference` (Miller indices) held at `psi` rad, see
/// [`super::e4c::psi_constant`].
pub fn psi_constant(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "psi_constant",
        &["omega_t", "mu", "gamma", "delta"],
        |_| vec![Equation::PsiConstant { reference, psi }],
    )
}
