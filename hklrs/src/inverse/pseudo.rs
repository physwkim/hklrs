//! hkl's detector and kappa pseudo-axis engines (`hkl-pseudoaxis-common-q.c`,
//! `-tth.c`, `-eulerians.c`), the inverses of [`crate::pseudo`]'s: each
//! mode takes the engine's pseudo values for a [`super::Target::Detector`]
//! request, [`eulerians`] for a [`super::Target::Eulerians`] one. Every
//! preset with the engine names its written axes alike.

use nalgebra::Vector3;

use super::{Equation, InverseError, Mode};
use crate::Geometry;

const GAMMA_DELTA: [&str; 2] = ["gamma", "delta"];

/// hkl's q engine on tth: [`crate::pseudo::q_signed`] at `q` Å⁻¹.
pub fn q(geometry: &Geometry, q: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "q", &["tth"], |_| vec![Equation::QSigned { q }])
}

/// hkl's q2 engine: |Q| at `q` Å⁻¹ and Q's azimuth about k̂i at `alpha`
/// rad ([`crate::pseudo::q2`]).
pub fn q2(geometry: &Geometry, q: f64, alpha: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "q2", &GAMMA_DELTA, |_| {
        vec![Equation::QNorm { q }, Equation::Qaz { qaz: alpha }]
    })
}

/// hkl's tth2 engine: 2θ at `tth` and Q's azimuth at `alpha` rad
/// ([`crate::pseudo::tth2`]).
pub fn tth2(geometry: &Geometry, tth: f64, alpha: f64) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "tth2", &GAMMA_DELTA, |_| {
        vec![Equation::Tth { tth }, Equation::Qaz { qaz: alpha }]
    })
}

/// hkl's qper_qpar engine: Q's components normal and parallel to the
/// sample-holder `surface` at `qper` and `qpar` Å⁻¹
/// ([`crate::pseudo::qper_qpar`]).
pub fn qper_qpar(
    geometry: &Geometry,
    surface: Vector3<f64>,
    qper: f64,
    qpar: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "qper_qpar", &GAMMA_DELTA, |_| {
        vec![
            Equation::Qper { surface, qper },
            Equation::Qpar { surface, qpar },
        ]
    })
}

/// hkl's eulerians engine on the kappa arm. hkl's `solutions` parameter
/// picks one kappa image; the request returns both.
pub fn eulerians(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named_kappa(
        geometry,
        "eulerians",
        &["komega", "kappa", "kphi"],
        |_| vec![],
    )
}
