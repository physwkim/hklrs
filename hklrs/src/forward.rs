//! Forward: axis values and a detector point to Q, hkl and read-only pseudos.

use nalgebra::{Point3, Unit, Vector3};

use crate::{Error, Geometry, Poses, Rad, Sample, Wavelength};

/// The sample-independent part of the forward kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scattering {
    /// Unit scattered-beam direction, lab frame.
    pub kf: Unit<Vector3<f64>>,
    /// Unit incident-beam direction, lab frame (source stack applied).
    pub ki: Unit<Vector3<f64>>,
    /// Q = k_f − k_i in the lab frame, Å⁻¹ with 2π.
    pub q_lab: Vector3<f64>,
    /// Q in the sample frame (sample stack undone).
    pub q_sample: Vector3<f64>,
    /// Scattering angle 2θ between k_i and k_f, in [0, π].
    pub tth: Rad,
    /// |Q| = 4π sin θ / λ, Å⁻¹.
    pub q: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forward {
    pub scattering: Scattering,
    pub hkl: Vector3<f64>,
}

/// Angle between two vectors, accurate at 0 and π.
pub fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> Rad {
    Rad(f64::atan2(a.cross(b).norm(), a.dot(b)))
}

/// `point` is a detector point in the detector-arm frame, metres
/// ([`crate::PixelDetector::pixel`]).
pub fn scattering(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
) -> Result<Scattering, Error> {
    let poses = geometry.poses(values)?;
    scattering_at(
        geometry,
        &poses,
        wavelength,
        &detector_position(&poses, point),
    )
}

/// Detector point `point` (detector-arm frame) relative to the sample at
/// `poses`: lab frame, metres.
pub fn detector_position(poses: &Poses, point: &Point3<f64>) -> Vector3<f64> {
    poses.detector * point - poses.sample_position()
}

/// k̂f, the direction of a [`detector_position`] `p`; a point at the sample
/// has none.
pub fn scattered_direction(p: &Vector3<f64>) -> Result<Unit<Vector3<f64>>, Error> {
    Unit::try_new(*p, 0.0).ok_or(Error::DetectorAtSample)
}

/// The kernel behind [`scattering`] and the batch map, from the
/// [`detector_position`] `p` at `poses`: a frame computes its poses once.
pub fn scattering_at(
    geometry: &Geometry,
    poses: &Poses,
    wavelength: Wavelength,
    p: &Vector3<f64>,
) -> Result<Scattering, Error> {
    let kf = scattered_direction(p)?;
    let ki = poses.source * geometry.beam_direction();
    let q_lab = wavelength.wavenumber() * (kf.into_inner() - ki.into_inner());
    let tth = angle_between(&ki, &kf);
    Ok(Scattering {
        kf,
        ki,
        q_lab,
        q_sample: poses.sample.rotation.inverse() * q_lab,
        tth,
        // from 2θ, not |q_lab|: the difference of two unit vectors loses
        // relative precision near the direct beam
        q: 2.0 * wavelength.wavenumber() * (tth.0 / 2.0).sin(),
    })
}

/// [`scattering`] plus the Miller indices of its Q for `sample`.
pub fn forward(
    geometry: &Geometry,
    sample: &Sample,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
) -> Result<Forward, Error> {
    let scattering = scattering(geometry, values, wavelength, point)?;
    Ok(Forward {
        hkl: sample.hkl(&scattering.q_sample),
        scattering,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ki_follows_the_source_stack_and_empty_is_the_beam() {
        let wl = Wavelength::new(1.0).unwrap();
        let p = Point3::new(1.0, 0.0, 0.0);
        let plain = Geometry::builder("p")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let f = scattering(&plain, &[0.4], wl, &p).unwrap();
        assert_eq!(f.ki, plain.beam_direction());
        let src = Geometry::builder("s")
            .source_rotation("tilt", [0.0, 1.0, 0.0])
            .build()
            .unwrap();
        let f = scattering(&src, &[0.1], wl, &p).unwrap();
        let want = Vector3::new(0.1_f64.cos(), 0.0, -0.1_f64.sin());
        assert!((f.ki.into_inner() - want).norm() < 1e-15);
        let k = wl.wavenumber();
        assert!((f.q_lab - k * (f.kf.into_inner() - f.ki.into_inner())).norm() < 1e-15);
    }

    /// HKL-1: `hkl_source_init` tests the signed components and rejects a
    /// beam along −x, −y or −z.
    #[test]
    fn hkl_1_any_non_zero_beam_direction_is_valid() {
        let wl = Wavelength::new(1.54).unwrap();
        for beam in [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-0.3, -0.5, -0.8],
            [2.0, 0.0, 0.0],
        ] {
            let geo = Geometry::builder("b")
                .detector_rotation("tth", [0.0, 0.0, 1.0])
                .beam_direction(beam)
                .build()
                .unwrap();
            let want = Vector3::from(beam).normalize();
            assert!((geo.beam_direction().into_inner() - want).norm() < 1e-16);
            let f = scattering(&geo, &[0.0], wl, &Point3::from(want)).unwrap();
            assert_eq!(f.ki.into_inner(), want, "{beam:?}");
            assert!(f.q_lab.norm() < 1e-15, "{beam:?}: {}", f.q_lab);
        }
        assert_eq!(
            Geometry::builder("b").beam_direction([0.0; 3]).build(),
            Err(Error::ZeroVector {
                what: "beam direction".into()
            })
        );
    }

    /// HKL-7: hkl's kf at zero detector angle is +x whatever the source
    /// direction, so a beam along z gives Q = (2π/λ, 0, −2π/λ) at 2θ = 0.
    /// Here the zero-angle kf is the beam, and 2θ is measured from it.
    #[test]
    fn hkl_7_zero_angle_kf_is_the_beam_direction() {
        let wl = Wavelength::new(1.54).unwrap();
        let geo = Geometry::builder("z")
            .sample_rotation("omega", [0.0, -1.0, 0.0])
            .detector_rotation("tth", [0.0, -1.0, 0.0])
            .beam_direction([0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let point = crate::PixelDetector::point(geo.beam_direction(), 0.5)
            .unwrap()
            .pixel(0, 0)
            .unwrap();
        let f = scattering(&geo, &[0.0, 0.0], wl, &point).unwrap();
        assert_eq!(f.kf, f.ki);
        assert_eq!(f.q_lab, Vector3::zeros());
        let tth = 0.6;
        let f = scattering(&geo, &[0.0, tth], wl, &point).unwrap();
        assert!((f.tth.0 - tth).abs() < 1e-15);
        let q = 2.0 * wl.wavenumber() * (tth / 2.0).sin();
        assert!((f.q_lab.norm() - q).abs() < 1e-14 * q);
    }

    /// XU-2: xu's point-detector variant with a detector translation and
    /// `sampledis` rotates a detector vector it never initialised (Q = NaN).
    /// Here kf comes from the detector point and the sample origin with and
    /// without either translation, against kf built by hand.
    #[test]
    fn xu_2_point_detector_follows_detector_and_sample_translations() {
        let wl = Wavelength::new(1.54).unwrap();
        // the sample displacement outermost, so it is fixed in the lab
        let geo = Geometry::builder("t")
            .sample_translation("sx", [1.0, 0.0, 0.0])
            .sample_translation("sy", [0.0, 1.0, 0.0])
            .sample_translation("sz", [0.0, 0.0, 1.0])
            .sample_rotation("om", [0.0, -1.0, 0.0])
            .detector_rotation("tth", [0.0, -1.0, 0.0])
            .detector_translation("dist", [1.0, 0.0, 0.0])
            .build()
            .unwrap();
        let (distance, tth) = (0.5, 20f64.to_radians());
        let point = Point3::new(distance, 0.0, 0.0);
        let arm = nalgebra::UnitQuaternion::from_axis_angle(&-Vector3::y_axis(), tth);
        let radial = 4.0 * std::f64::consts::PI * 10f64.to_radians().sin() / 1.54;
        for (dist, sd) in [
            (0.0, [0.0; 3]),
            (0.2, [0.0; 3]),
            (0.0, [0.01, -0.02, 0.03]),
            (0.2, [0.01, -0.02, 0.03]),
        ] {
            let values = [sd[0], sd[1], sd[2], 0.3, tth, dist];
            let f = scattering(&geo, &values, wl, &point).unwrap();
            let kf =
                (arm * Vector3::new(distance + dist, 0.0, 0.0) - Vector3::from(sd)).normalize();
            let want = wl.wavenumber() * (kf - Vector3::x());
            assert!(
                (f.q_lab - want).norm() < 1e-14,
                "dist {dist} sd {sd:?}: {}",
                f.q_lab
            );
            if sd == [0.0; 3] {
                assert!((f.q - radial).abs() < 1e-15 * radial, "{}", f.q);
            }
        }
    }
}
