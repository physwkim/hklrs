//! hkl's K6C "hkl" engine modes (`hkl-engine-k6c.c`), for any geometry with
//! sample axes mu and a kappa arm komega, kappa, kphi ([`Mode::new_kappa`])
//! and detector axes gamma, delta. The vertical modes hold mu and gamma, the
//! horizontal ones delta. hkl's two functions per mode and its appended kappa
//! twins are the either-image [`Eulerian`] rows and the search's twin
//! candidates; a twin that moves a held arm axis is not a solution (HKL-16).

use nalgebra::Vector3;

use super::{Equation, Eulerian, InverseError, Mode};
use crate::Geometry;

const VERTICAL: [&str; 4] = ["komega", "kappa", "kphi", "delta"];
const HORIZONTAL: [&str; 5] = ["mu", "komega", "kappa", "kphi", "gamma"];

/// delta ≡ 2·omega (mod 2π), omega the arm's eulerian.
pub fn bissector_vertical(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "bissector_vertical", &VERTICAL, |w| {
        vec![Equation::Kappa(Eulerian::Bisector { detector: w[3] })]
    })
}

/// The arm's eulerian omega held at `omega` rad (mod π).
pub fn constant_omega_vertical(geometry: &Geometry, omega: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_omega_vertical", &VERTICAL, |_| {
        vec![Equation::Kappa(Eulerian::Omega(omega))]
    })
}

/// The arm's eulerian |chi| held at |`chi`| rad.
pub fn constant_chi_vertical(geometry: &Geometry, chi: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_chi_vertical", &VERTICAL, |_| {
        vec![Equation::Kappa(Eulerian::Chi(chi))]
    })
}

/// The arm's eulerian phi held at `phi` rad (mod π).
pub fn constant_phi_vertical(geometry: &Geometry, phi: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_phi_vertical", &VERTICAL, |_| {
        vec![Equation::Kappa(Eulerian::Phi(phi))]
    })
}

pub fn lifting_detector_kphi(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "lifting_detector_kphi",
        &["kphi", "gamma", "delta"],
        |_| vec![],
    )
}

pub fn lifting_detector_komega(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "lifting_detector_komega",
        &["komega", "gamma", "delta"],
        |_| vec![],
    )
}

pub fn lifting_detector_mu(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "lifting_detector_mu",
        &["mu", "gamma", "delta"],
        |_| vec![],
    )
}

/// `hkl2` (Miller indices) in diffraction as well.
pub fn double_diffraction_vertical(
    geometry: &Geometry,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "double_diffraction_vertical", &VERTICAL, |_| {
        vec![Equation::DoubleDiffraction { hkl2 }]
    })
}

/// The arm's eulerian omega ≡ 0 (mod π) and gamma ≡ 2·mu (mod 2π).
pub fn bissector_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "bissector_horizontal", &HORIZONTAL, |w| {
        vec![
            Equation::Kappa(Eulerian::Omega(0.0)),
            Equation::Bisector {
                sample: w[0],
                detector: w[4],
            },
        ]
    })
}

/// The arm's eulerian omega ≡ 0 and phi ≡ `phi` rad (mod π).
pub fn constant_phi_horizontal(geometry: &Geometry, phi: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_phi_horizontal", &HORIZONTAL, |_| {
        vec![
            Equation::Kappa(Eulerian::Omega(0.0)),
            Equation::Kappa(Eulerian::Phi(phi)),
        ]
    })
}

/// The arm's eulerian omega ≡ 0 (mod π), kphi held.
pub fn constant_kphi_horizontal(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "constant_kphi_horizontal",
        &["mu", "komega", "kappa", "gamma"],
        |_| vec![Equation::Kappa(Eulerian::Omega(0.0))],
    )
}

/// `hkl2` (Miller indices) in diffraction as well, the arm's eulerian
/// omega ≡ 0 (mod π).
pub fn double_diffraction_horizontal(
    geometry: &Geometry,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "double_diffraction_horizontal",
        &HORIZONTAL,
        |_| {
            vec![
                Equation::DoubleDiffraction { hkl2 },
                Equation::Kappa(Eulerian::Omega(0.0)),
            ]
        },
    )
}

/// ψ of `reference` (Miller indices) held at `psi` rad, as
/// [`super::e4c::psi_constant`].
pub fn psi_constant_vertical(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "psi_constant_vertical", &VERTICAL, |_| {
        vec![Equation::PsiConstant { reference, psi }]
    })
}

/// The surface `normal` (sample-holder frame) at `incidence` and `azimuth`
/// rad ([`Equation::Incidence`], [`Equation::Azimuth`]). hkl's init step
/// takes its normal as the lab direction at the init geometry: the
/// sample-holder normal is that direction under the inverse sample rotation
/// there.
pub fn constant_incidence(
    geometry: &Geometry,
    normal: Vector3<f64>,
    incidence: f64,
    azimuth: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "constant_incidence",
        &["komega", "kappa", "kphi", "gamma", "delta"],
        |_| {
            vec![
                Equation::Incidence { normal, incidence },
                Equation::Azimuth { normal, azimuth },
            ]
        },
    )
}
