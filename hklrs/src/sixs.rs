//! SIXS detector calibration: central pixel, sample-detector distance and
//! detector rotation, ported from `hkl_binoculars_detector_2d_sixs_calibration`
//! (binoculars-ng 7be4f79, detectors-2d.c:463).

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

use crate::{Error, PixelDetector};

/// Moves `det` so that its pixel `centre` (slow, fast) lies at `(sdd, 0, 0)`,
/// then rotates it by `detrot` radians about x. The mask is kept.
/// binoculars zeroes only y and z of the central pixel and adds `sdd` to its
/// x, which differs for a detector whose central pixel has x != 0 such as
/// the Cirpad (BIN-9).
///
/// binoculars' `normalize_flag` is not ported; it maps pixels onto the unit
/// sphere, which no [`PixelDetector`] can hold.
pub fn sixs(
    det: &PixelDetector,
    centre: (usize, usize),
    sdd: f64,
    detrot: f64,
) -> Result<PixelDetector, Error> {
    for (what, value) in [("sdd", sdd), ("detrot", detrot)] {
        if !value.is_finite() {
            return Err(Error::NotFinite { what, value });
        }
    }
    let c = det.pixel(centre.0, centre.1)?;
    let shift = Vector3::new(sdd, 0.0, 0.0) - c.coords;
    let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), detrot);
    Ok(det.transformed(&Isometry3::from_parts(Translation3::from(rot * shift), rot)))
}
