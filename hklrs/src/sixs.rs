//! SIXS detector calibration: central pixel, sample-detector distance and
//! detector rotation, ported from `hkl_binoculars_detector_2d_sixs_calibration`
//! (binoculars-ng 7be4f79, detectors-2d.c:463).

use nalgebra::{UnitQuaternion, Vector3};

use crate::{Error, PixelDetector};

/// Moves `det` so that its pixel `centre` (slow, fast) lies at `y = z = 0` at
/// distance `sdd` along x, then rotates it by `detrot` radians about x. Only
/// y and z of the central pixel are zeroed, as binoculars does: its x is kept
/// and `sdd` is added to it. The mask is kept.
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
    let shift = Vector3::new(sdd, -c.y, -c.z);
    let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), detrot);
    let out = PixelDetector::new(
        det.shape(),
        rot * (det.origin().coords + shift),
        rot * det.e_slow(),
        rot * det.e_fast(),
    )?;
    match det.mask() {
        Some(mask) => out.with_mask(mask.to_vec()),
        None => Ok(out),
    }
}
