//! hkl's K6C and SOLEIL SIRIUS KAPPA "hkl" engine modes
//! (`hkl-engine-k6c.c`), for any geometry with sample axes mu and a kappa arm
//! komega, kappa, kphi ([`Mode::new_kappa`]) and the two detector axes of
//! its [`Detectors`]. The vertical modes hold mu and the horizontal
//! detector, the horizontal ones the vertical detector. hkl's two functions
//! per mode and its appended kappa twins are the either-image [`Eulerian`]
//! rows and the search's twin candidates; a twin that moves a held arm axis
//! is not a solution (HKL-16).

use nalgebra::Vector3;

use super::{Equation, Eulerian, InverseError, Mode};
use crate::Geometry;

/// The detector axis the vertical modes write and the one the horizontal
/// modes write; the lifting and incidence modes write both, horizontal
/// first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detectors {
    pub vertical: &'static str,
    pub horizontal: &'static str,
}

/// hkl's K6C.
pub const K6C: Detectors = Detectors {
    vertical: "delta",
    horizontal: "gamma",
};

/// hkl's SOLEIL SIRIUS KAPPA, whose detector stack is delta then gamma.
pub const SOLEIL_SIRIUS_KAPPA: Detectors = Detectors {
    vertical: "gamma",
    horizontal: "delta",
};

impl Detectors {
    fn vertical(self) -> [&'static str; 4] {
        ["komega", "kappa", "kphi", self.vertical]
    }

    fn horizontal(self) -> [&'static str; 5] {
        ["mu", "komega", "kappa", "kphi", self.horizontal]
    }

    fn lifting(self, sample: &'static str) -> [&'static str; 3] {
        [sample, self.horizontal, self.vertical]
    }
}

/// The vertical detector ≡ 2·omega (mod 2π), omega the arm's eulerian.
pub fn bissector_vertical(geometry: &Geometry, d: Detectors) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "bissector_vertical", &d.vertical(), |w| {
        vec![Equation::Kappa(Eulerian::Bisector { detector: w[3] })]
    })
}

/// The arm's eulerian omega held at `omega` rad (mod π).
pub fn constant_omega_vertical(
    geometry: &Geometry,
    d: Detectors,
    omega: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_omega_vertical", &d.vertical(), |_| {
        vec![Equation::Kappa(Eulerian::Omega(omega))]
    })
}

/// The arm's eulerian |chi| held at |`chi`| rad.
pub fn constant_chi_vertical(
    geometry: &Geometry,
    d: Detectors,
    chi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_chi_vertical", &d.vertical(), |_| {
        vec![Equation::Kappa(Eulerian::Chi(chi))]
    })
}

/// The arm's eulerian phi held at `phi` rad (mod π).
pub fn constant_phi_vertical(
    geometry: &Geometry,
    d: Detectors,
    phi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_phi_vertical", &d.vertical(), |_| {
        vec![Equation::Kappa(Eulerian::Phi(phi))]
    })
}

pub fn lifting_detector_kphi(geometry: &Geometry, d: Detectors) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "lifting_detector_kphi",
        &d.lifting("kphi"),
        |_| vec![],
    )
}

pub fn lifting_detector_komega(
    geometry: &Geometry,
    d: Detectors,
) -> Result<Mode<'_>, InverseError> {
    let written = d.lifting("komega");
    Mode::named_kappa(geometry, "lifting_detector_komega", &written, |_| vec![])
}

pub fn lifting_detector_mu(geometry: &Geometry, d: Detectors) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "lifting_detector_mu",
        &d.lifting("mu"),
        |_| vec![],
    )
}

/// `hkl2` (Miller indices) in diffraction as well.
pub fn double_diffraction_vertical(
    geometry: &Geometry,
    d: Detectors,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    let written = d.vertical();
    Mode::named_kappa(geometry, "double_diffraction_vertical", &written, |_| {
        vec![Equation::DoubleDiffraction { hkl2 }]
    })
}

/// The arm's eulerian omega ≡ 0 (mod π) and the horizontal detector ≡ 2·mu
/// (mod 2π).
pub fn bissector_horizontal(geometry: &Geometry, d: Detectors) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "bissector_horizontal", &d.horizontal(), |w| {
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
pub fn constant_phi_horizontal(
    geometry: &Geometry,
    d: Detectors,
    phi: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(geometry, "constant_phi_horizontal", &d.horizontal(), |_| {
        vec![
            Equation::Kappa(Eulerian::Omega(0.0)),
            Equation::Kappa(Eulerian::Phi(phi)),
        ]
    })
}

/// The arm's eulerian omega ≡ 0 (mod π), kphi held.
pub fn constant_kphi_horizontal(
    geometry: &Geometry,
    d: Detectors,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "constant_kphi_horizontal",
        &["mu", "komega", "kappa", d.horizontal],
        |_| vec![Equation::Kappa(Eulerian::Omega(0.0))],
    )
}

/// `hkl2` (Miller indices) in diffraction as well, the arm's eulerian
/// omega ≡ 0 (mod π).
pub fn double_diffraction_horizontal(
    geometry: &Geometry,
    d: Detectors,
    hkl2: Vector3<f64>,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "double_diffraction_horizontal",
        &d.horizontal(),
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
    d: Detectors,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    psi_mode(geometry, "psi_constant_vertical", d, reference, psi)
}

/// hkl's K6C psi engine, as [`super::e4c::psi`].
pub fn psi_vertical(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    psi_mode(geometry, "psi_vertical", K6C, reference, psi)
}

/// hkl's SOLEIL SIRIUS KAPPA psi engine: [`psi_vertical`] on its
/// [`Detectors`].
pub fn psi_vertical_soleil_sirius_kappa(
    geometry: &Geometry,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'_>, InverseError> {
    let name = "psi_vertical_soleil_sirius_kappa";
    psi_mode(geometry, name, SOLEIL_SIRIUS_KAPPA, reference, psi)
}

fn psi_mode<'g>(
    geometry: &'g Geometry,
    name: &str,
    d: Detectors,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'g>, InverseError> {
    Mode::named_kappa(geometry, name, &d.vertical(), |_| {
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
    d: Detectors,
    normal: Vector3<f64>,
    incidence: f64,
    azimuth: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "constant_incidence",
        &["komega", "kappa", "kphi", d.horizontal, d.vertical],
        |_| {
            vec![
                Equation::Incidence { normal, incidence },
                Equation::Azimuth { normal, azimuth },
            ]
        },
    )
}
