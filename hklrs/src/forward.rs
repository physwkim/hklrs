//! Forward: axis values and a detector point to Q, hkl and read-only pseudos.

use nalgebra::{Point3, Unit, Vector3};

use crate::{Error, Geometry, Rad, Sample, Wavelength};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forward {
    /// Unit scattered-beam direction, lab frame.
    pub kf: Unit<Vector3<f64>>,
    /// Q = k_f − k_i in the lab frame, Å⁻¹ with 2π.
    pub q_lab: Vector3<f64>,
    /// Q in the sample frame (sample stack undone).
    pub q_sample: Vector3<f64>,
    pub hkl: Vector3<f64>,
    /// Scattering angle 2θ between k_i and k_f, in [0, π].
    pub tth: Rad,
    /// |Q| = 4π sin θ / λ, Å⁻¹.
    pub q: f64,
}

/// Angle between two vectors, accurate at 0 and π.
pub fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> Rad {
    Rad(f64::atan2(a.cross(b).norm(), a.dot(b)))
}

/// `point` is a detector point in the detector-arm frame, metres
/// ([`crate::PixelDetector::pixel`]).
pub fn forward(
    geometry: &Geometry,
    sample: &Sample,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
) -> Result<Forward, Error> {
    let poses = geometry.poses(values)?;
    let scattered = poses.detector * point - poses.sample_position();
    let kf = Unit::try_new(scattered, 0.0).ok_or(Error::DetectorAtSample)?;
    let ki = geometry.beam_direction();
    let q_lab = wavelength.wavenumber() * (kf.into_inner() - ki.into_inner());
    let q_sample = poses.sample.rotation.inverse() * q_lab;
    let tth = angle_between(&ki, &kf);
    Ok(Forward {
        kf,
        q_lab,
        q_sample,
        hkl: sample.hkl(&q_sample),
        tth,
        // from 2θ, not |q_lab|: the difference of two unit vectors loses
        // relative precision near the direct beam
        q: 2.0 * wavelength.wavenumber() * (tth.0 / 2.0).sin(),
    })
}
