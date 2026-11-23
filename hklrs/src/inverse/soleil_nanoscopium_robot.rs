//! hkl's SOLEIL NANOSCOPIUM ROBOT "hkl" engine modes
//! (`hkl-engine-soleil-nanoscopium-robot.c`), for any geometry with sample
//! axes rz, rs, rx and detector axes delta, gamma. Each writes one sample
//! axis; the other two and the detector length r hold at `current`.

use super::{InverseError, Mode};
use crate::Geometry;

fn written<'g>(
    geometry: &'g Geometry,
    name: &str,
    axes: &[&str],
) -> Result<Mode<'g>, InverseError> {
    Mode::named(geometry, name, axes, |_| vec![])
}

/// hkl's default mode.
pub fn lifting_detector_rz(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector rz", &["rz", "delta", "gamma"])
}

pub fn lifting_detector_rs(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector rs", &["rs", "delta", "gamma"])
}

pub fn lifting_detector_rx(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    written(geometry, "lifting detector rx", &["rx", "delta", "gamma"])
}
