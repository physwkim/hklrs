//! hkl's 6-circle "hkl" engine modes (`hkl-engine-e6c.c`: E6C), for any
//! geometry with sample axes mu, omega, chi, phi and detector axes gamma,
//! delta. Axes a mode does not write hold at `current`.

use nalgebra::Vector3;

use super::{Equation, InverseError, Mode};
use crate::Geometry;

const VERTICAL: [&str; 4] = ["omega", "chi", "phi", "delta"];
const HORIZONTAL: [&str; 4] = ["mu", "chi", "phi", "gamma"];

fn written<'g>(
    geometry: &'g Geometry,
    name: &str,
    axes: &[&str],
) -> Result<Mode<'g>, InverseError> {
    Mode::named(geometry, name, axes, |_| vec![])
}

/// delta ≡ 2·omega (mod 2π).
pub fn bissector_vertical(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "bissector_vertical", &VERTICAL, |w| {
        vec![Equation::Bisector {
            sample: w[0],
            detector: w[3],
        }]
    })
}

pub fn constant_omega_vertical(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "constant_omega_vertical",
        &["chi", "phi", "delta"],
    )
}

pub fn constant_chi_vertical(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "constant_chi_vertical",
        &["omega", "phi", "delta"],
    )
}

pub fn constant_phi_vertical(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "constant_phi_vertical",
        &["omega", "chi", "delta"],
    )
}

pub fn lifting_detector_phi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting_detector_phi", &["phi", "gamma", "delta"])
}

pub fn lifting_detector_omega(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(
        geometry,
        "lifting_detector_omega",
        &["omega", "gamma", "delta"],
    )
}

pub fn lifting_detector_mu(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting_detector_mu", &["mu", "gamma", "delta"])
}

/// `hkl2` (Miller indices) in diffraction as well.
pub fn double_diffraction_vertical(
    geometry: &Geometry,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "double_diffraction_vertical", &VERTICAL, |_| {
        vec![Equation::DoubleDiffraction { hkl2 }]
    })
}

/// omega ≡ 0 (mod π) and gamma ≡ 2·mu (mod 2π).
pub fn bissector_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "bissector_horizontal",
        &["mu", "omega", "chi", "phi", "gamma"],
        |w| {
            vec![
                Equation::ZeroModPi { axis: w[1] },
                Equation::Bisector {
                    sample: w[0],
                    detector: w[4],
                },
            ]
        },
    )
}

/// `hkl2` (Miller indices) in diffraction as well.
pub fn double_diffraction_horizontal(
    geometry: &Geometry,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "double_diffraction_horizontal",
        &HORIZONTAL,
        |_| vec![Equation::DoubleDiffraction { hkl2 }],
    )
}

/// ψ of `reference` (Miller indices) held at `psi` rad, see
/// [`super::e4c::psi_constant`].
pub fn psi_constant_vertical(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "psi_constant_vertical", &VERTICAL, |_| {
        vec![Equation::PsiConstant { reference, psi }]
    })
}

/// hkl's psi engine, as [`super::e4c::psi`].
pub fn psi_vertical(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "psi_vertical", &VERTICAL, |_| {
        vec![Equation::PsiConstant { reference, psi }]
    })
}

/// As [`psi_constant_vertical`], writing gamma instead of delta.
pub fn psi_constant_horizontal(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "psi_constant_horizontal",
        &["omega", "chi", "phi", "gamma"],
        |_| vec![Equation::PsiConstant { reference, psi }],
    )
}

pub fn constant_mu_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "constant_mu_horizontal", &["chi", "phi", "gamma"])
}
