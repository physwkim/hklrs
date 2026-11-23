//! hkl's APS POLAR "hkl" engine modes (`hkl-engine-aps-04id-polar.c`), for
//! any geometry with sample axes tau, mu, chi, phi and detector axes tau,
//! gamma, delta. tau tilts both, so with the surface normal along mu's axis
//! the incidence α is tau and the exit β is delta. Axes a mode does not write
//! hold at `current`.

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

/// hkl's default mode.
pub fn constant_phi_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "4-circles constant phi horizontal",
        &["mu", "chi", "gamma"],
    )
}

/// α (tau) held.
pub fn zaxis_alpha_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "zaxis + alpha-fixed", &["mu", "gamma", "delta"])
}

/// β (delta) held. hkl 5.1.9 writes tau, gamma, delta here, the axes of
/// [`lifting_detector_tau`], which move β and hold mu.
pub fn zaxis_beta_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "zaxis + beta-fixed", &["tau", "mu", "gamma"])
}

/// α = β: tau ≡ delta (mod 2π).
pub fn zaxis_alpha_eq_beta(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "zaxis + alpha=beta",
        &["tau", "mu", "gamma", "delta"],
        |w| vec![Equation::Equal { a: w[0], b: w[3] }],
    )
}

/// gamma ≡ 2·mu (mod 2π).
pub fn bissecting_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "4-circles bissecting horizontal",
        &["mu", "chi", "phi", "gamma"],
        |w| {
            vec![Equation::Bisector {
                sample: w[0],
                detector: w[3],
            }]
        },
    )
}

pub fn constant_mu_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "4-circles constant mu horizontal",
        &["chi", "phi", "gamma"],
    )
}

pub fn constant_chi_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "4-circles constant chi horizontal",
        &["mu", "phi", "gamma"],
    )
}

pub fn lifting_detector_tau(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector tau", &["tau", "gamma", "delta"])
}

pub fn lifting_detector_mu(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector mu", &["mu", "gamma", "delta"])
}

pub fn lifting_detector_chi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector chi", &["chi", "gamma", "delta"])
}

pub fn lifting_detector_phi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector phi", &["phi", "gamma", "delta"])
}

/// ψ of `reference` (Miller indices) held at `psi` rad, see
/// [`super::e4c::psi_constant`].
pub fn psi_constant_horizontal(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "psi constant horizontal",
        &["mu", "chi", "phi", "gamma"],
        |_| vec![Equation::PsiConstant { reference, psi }],
    )
}

/// As [`psi_constant_horizontal`], writing tau and delta instead of mu and
/// gamma.
pub fn psi_constant_vertical(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "psi constant vertical",
        &["tau", "chi", "phi", "delta"],
        |_| vec![Equation::PsiConstant { reference, psi }],
    )
}
