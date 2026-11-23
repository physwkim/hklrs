//! hkl's SOLEIL SIRIUS TURRET "hkl" engine mode
//! (`hkl-engine-soleil-sirius-turret.c`), for any geometry with sample axis
//! thetah and detector axes delta, gamma. Axes it does not write, basepitch
//! (in both stacks), alphay and alphax, hold at `current`.

use super::{InverseError, Mode};
use crate::Geometry;

/// hkl's only mode.
pub fn lifting_detector_thetah(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "lifting_detector_thetah",
        &["thetah", "delta", "gamma"],
        |_| vec![],
    )
}
