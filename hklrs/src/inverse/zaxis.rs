//! hkl's ZAXIS "hkl" engine modes (`hkl-engine-zaxis.c`), for any geometry
//! with sample axes mu, omega and detector axes mu, delta, gamma. Axes a mode
//! does not write hold at `current`.

use super::{Equation, InverseError, Mode};
use crate::Geometry;

pub fn zaxis(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "zaxis", &["omega", "delta", "gamma"], |_| vec![])
}

/// gamma ≡ mu (mod 2π).
pub fn reflectivity(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "reflectivity",
        &["mu", "omega", "delta", "gamma"],
        |w| vec![Equation::Equal { a: w[0], b: w[3] }],
    )
}
