//! hkl's ESRF BM28 PSIC "hkl" engine modes (`hkl-engine-esrf-bm28-psic.c`),
//! for any geometry with sample axes mu, eta, chi, phi and detector axes nu,
//! delta. Axes a mode does not write hold at `current`.

use super::{InverseError, Mode};
use crate::Geometry;

pub fn constant_nu_coplanar(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "constant_nu_coplanar",
        &["eta", "phi", "delta"],
        |_| vec![],
    )
}

pub fn constant_delta_coplanar(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "constant_delta_coplanar",
        &["eta", "phi", "nu"],
        |_| vec![],
    )
}

pub fn constant_eta_noncoplanar(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "constant_eta_noncoplanar",
        &["phi", "nu", "delta"],
        |_| vec![],
    )
}
