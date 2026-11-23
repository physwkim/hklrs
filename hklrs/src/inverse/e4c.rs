//! hkl's 4-circle "hkl" engine modes (`hkl-engine-e4c.c`: E4CV, E4CH, SOLEIL
//! MARS), for any geometry with sample axes omega, chi, phi and detector
//! axis tth. The constant_* modes hold their axis at `current`.

use nalgebra::Vector3;

use super::{Equation, InverseError, Mode};
use crate::Geometry;

const ALL: [&str; 4] = ["omega", "chi", "phi", "tth"];

fn mode<'g>(
    geometry: &'g Geometry,
    name: &str,
    written: &[&str],
    extra: impl FnOnce(&[usize]) -> Vec<Equation>,
) -> Result<Mode<'g>, InverseError> {
    let w = written
        .iter()
        .map(|n| geometry.axis_index(n))
        .collect::<Result<Vec<_>, _>>()?;
    Mode::new(geometry, name, &w, extra(&w))
}

/// tth ≡ 2·omega (mod 2π).
pub fn bissector(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    mode(geometry, "bissector", &ALL, |w| {
        vec![Equation::Bisector {
            sample: w[0],
            detector: w[3],
        }]
    })
}

pub fn constant_omega(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    mode(
        geometry,
        "constant_omega",
        &["chi", "phi", "tth"],
        |_| vec![],
    )
}

pub fn constant_chi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    mode(
        geometry,
        "constant_chi",
        &["omega", "phi", "tth"],
        |_| vec![],
    )
}

pub fn constant_phi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    mode(
        geometry,
        "constant_phi",
        &["omega", "chi", "tth"],
        |_| vec![],
    )
}

/// `hkl2` (Miller indices) in diffraction as well.
pub fn double_diffraction(
    geometry: &Geometry,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    mode(geometry, "double_diffraction", &ALL, |_| {
        vec![Equation::DoubleDiffraction { hkl2 }]
    })
}

/// ψ of `reference` (Miller indices) held at `psi` rad; hkl's init step,
/// which takes ψ from the current axes, is [`super::psi`].
pub fn psi_constant(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    mode(geometry, "psi_constant", &ALL, |_| {
        vec![Equation::PsiConstant { reference, psi }]
    })
}
