//! hkl's K4CV "hkl" engine modes (`hkl-engine-k4c.c`), for any geometry
//! with a kappa arm komega, kappa, kphi ([`Mode::new_kappa`]) and detector
//! axis tth. Every mode writes all four axes; the constant_* modes hold an
//! eulerian of the arm at a parameter. hkl's two functions per mode (one per
//! eulerian image) and its appended kappa twins are the either-image
//! [`Eulerian`] rows and the search's twin candidates.

use nalgebra::Vector3;

use super::{Equation, Eulerian, InverseError, Mode};
use crate::Geometry;

const ALL: [&str; 4] = ["komega", "kappa", "kphi", "tth"];

/// tth ≡ 2·omega (mod 2π), omega the arm's eulerian.
pub fn bissector(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "bissector", &ALL, |w| {
        vec![Equation::Kappa(Eulerian::Bisector { detector: w[3] })]
    })
}

/// The arm's eulerian omega held at `omega` rad (mod π).
pub fn constant_omega(geometry: &Geometry, omega: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_omega", &ALL, |_| {
        vec![Equation::Kappa(Eulerian::Omega(omega))]
    })
}

/// The arm's eulerian |chi| held at |`chi`| rad.
pub fn constant_chi(geometry: &Geometry, chi: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_chi", &ALL, |_| {
        vec![Equation::Kappa(Eulerian::Chi(chi))]
    })
}

/// The arm's eulerian phi held at `phi` rad (mod π).
pub fn constant_phi(geometry: &Geometry, phi: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_phi", &ALL, |_| {
        vec![Equation::Kappa(Eulerian::Phi(phi))]
    })
}

/// `hkl2` (Miller indices) in diffraction as well.
pub fn double_diffraction(
    geometry: &Geometry,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "double_diffraction", &ALL, |_| {
        vec![Equation::DoubleDiffraction { hkl2 }]
    })
}

/// ψ of `reference` (Miller indices) held at `psi` rad, as
/// [`super::e4c::psi_constant`].
pub fn psi_constant(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "psi_constant", &ALL, |_| {
        vec![Equation::PsiConstant { reference, psi }]
    })
}
