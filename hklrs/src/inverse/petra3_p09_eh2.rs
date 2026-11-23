//! hkl's PETRA3 P09 EH2 "hkl" engine modes (`hkl-engine-petra3-p09-eh2.c`),
//! for any geometry with sample axes mu, omega, chi, phi and detector axes
//! mu, delta, gamma. mu tilts both, so with the surface normal along omega's
//! axis the incidence α is mu and the exit β is gamma. Axes a mode does not
//! write hold at `current`.

use super::{Equation, InverseError, Mode};
use crate::Geometry;

fn written<'g>(
    geometry: &'g Geometry,
    name: &str,
    axes: &[&str],
) -> Result<Mode<'g>, InverseError> {
    Mode::named(geometry, name, axes, |_| vec![])
}

/// hkl's default mode: α (mu) held.
pub fn zaxis_alpha_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "zaxis + alpha-fixed",
        &["omega", "delta", "gamma"],
    )
}

/// β (gamma) held. hkl 5.1.9 writes mu, delta, gamma here, the axes of
/// [`lifting_detector_mu`], which move β and hold omega.
pub fn zaxis_beta_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "zaxis + beta-fixed", &["mu", "omega", "delta"])
}

/// α = β: mu ≡ gamma (mod 2π).
pub fn zaxis_alpha_eq_beta(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "zaxis + alpha=beta",
        &["mu", "omega", "delta", "gamma"],
        |w| vec![Equation::Equal { a: w[0], b: w[3] }],
    )
}

/// delta ≡ 2·omega (mod 2π).
pub fn bissecting_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "4-circles bissecting horizontal",
        &["omega", "chi", "phi", "delta"],
        |w| {
            vec![Equation::Bisector {
                sample: w[0],
                detector: w[3],
            }]
        },
    )
}

pub fn constant_omega_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "4-circles constant omega horizontal",
        &["chi", "phi", "delta"],
    )
}

pub fn constant_chi_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "4-circles constant chi horizontal",
        &["omega", "phi", "delta"],
    )
}

pub fn constant_phi_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "4-circles constant phi horizontal",
        &["omega", "chi", "delta"],
    )
}

pub fn lifting_detector_mu(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector mu", &["mu", "delta", "gamma"])
}

pub fn lifting_detector_omega(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "lifting detector omega",
        &["omega", "delta", "gamma"],
    )
}

pub fn lifting_detector_chi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector chi", &["chi", "delta", "gamma"])
}

pub fn lifting_detector_phi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector phi", &["phi", "delta", "gamma"])
}
