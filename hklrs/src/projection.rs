//! Projections (design §7): the per-pixel coordinates a frame is binned on,
//! each variant naming its own axes. Every coordinate comes from the
//! forward kernel's per-pixel output ([`q_map`], [`detector_positions`]),
//! in hklrs units: Å⁻¹ with 2π, radians.
//!
//! binoculars 7be4f79 counterparts (hkl-binoculars.c): `QSample` is qcustom
//! QX_QY_QZ on a HORIZONTAL surface with uq = 0 (:826-848), `QSurface` the
//! same on any surface and uq, `QparQper` qcustom QPAR_QPER (:923-947),
//! `Hkl` the hkl space (:1507-1570), `Angles` the angles space (:341-389).
//! `QLab` has none.

use nalgebra::{Rotation3, Unit, Vector3};

use crate::grid::{Coord, Projected};
use crate::{
    Error, Geometry, Image, Orientation, PixelDetector, Planes, Sample, Wavelength, angle_between,
    detector_positions, q_map,
};

/// What a frame is binned on.
#[derive(Clone, Debug, PartialEq)]
pub enum Projection {
    /// Q in the lab frame.
    QLab,
    /// Q in the sample frame (sample stack undone).
    QSample,
    /// Q in a frame fixed to the sample, on the columns of `frame` (its
    /// axes in the sample frame): frame⁻¹ Q_sample. binoculars' QxQyQz
    /// takes frame Rx(−π/2) for a VERTICAL surface, which puts the normal ŷ
    /// on qz, times Rz(−omega_offset) (Python binoculars 93956b2, sixs.py
    /// :376-390) or euler_xyz(uqx, uqy, uqz) (hkl-binoculars.c:800-817).
    QSurface { frame: Rotation3<f64> },
    /// Miller indices of Q for the sample, through its UB.
    Hkl(Sample),
    /// Q split along a surface normal given in the sample frame:
    /// qper = Q·n and qpar = |Q − (Q·n) n|. binoculars fixes n per surface
    /// orientation (hkl-binoculars.c:801-812, qpar/qper at :741-744,
    /// :762): ẑ for HORIZONTAL and ŷ for VERTICAL, rotated by euler_xyz(uqx,
    /// uqy, uqz).
    QparQper { normal: Unit<Vector3<f64>> },
    /// Direction of the scattered beam in the lab frame, with x the beam
    /// at zero source and z up: delta_lab = atan2(z, x), gamma_lab =
    /// atan2(y, √(x²+z²)), and tth the angle from the incident beam.
    Angles,
}

impl Projection {
    /// The coordinates, the same table [`Projection::project`] labels its
    /// planes with.
    pub fn coords(&self) -> &[Coord] {
        const QLAB: &[Coord] = &[
            Coord::linear("qx_lab"),
            Coord::linear("qy_lab"),
            Coord::linear("qz_lab"),
        ];
        const QSAMPLE: &[Coord] = &[
            Coord::linear("qx"),
            Coord::linear("qy"),
            Coord::linear("qz"),
        ];
        const QSURFACE: &[Coord] = &[
            Coord::linear("qx_surface"),
            Coord::linear("qy_surface"),
            Coord::linear("qz_surface"),
        ];
        const HKL: &[Coord] = &[Coord::linear("H"), Coord::linear("K"), Coord::linear("L")];
        const QPAR_QPER: &[Coord] = &[Coord::linear("qpar"), Coord::linear("qper")];
        const ANGLES: &[Coord] = &[
            Coord::angle("delta_lab"),
            Coord::angle("gamma_lab"),
            Coord::angle("tth"),
        ];
        match self {
            Self::QLab => QLAB,
            Self::QSample => QSAMPLE,
            Self::QSurface { .. } => QSURFACE,
            Self::Hkl(_) => HKL,
            Self::QparQper { .. } => QPAR_QPER,
            Self::Angles => ANGLES,
        }
    }

    /// The coordinates of every pixel of `detector` for one frame at axis
    /// `values`, over the detector's shape.
    pub fn project(
        &self,
        geometry: &Geometry,
        values: &[f64],
        wavelength: Wavelength,
        detector: &PixelDetector,
    ) -> Result<Projected, Error> {
        let planes = match self {
            Self::QLab => {
                let rotation = geometry.poses(values)?.sample.rotation;
                map(&q_map(geometry, values, wavelength, detector)?, |q| {
                    (rotation * q).into()
                })
            }
            Self::QSample => {
                let q = q_map(geometry, values, wavelength, detector)?;
                q.into_planes().into()
            }
            Self::QSurface { frame } => {
                let to_frame = frame.inverse();
                map(&q_map(geometry, values, wavelength, detector)?, |q| {
                    (to_frame * q).into()
                })
            }
            Self::Hkl(sample) => {
                // U once per frame, not per pixel; hkl() is forward's kernel
                let fixed = Sample::new(
                    &sample.name,
                    sample.lattice,
                    Orientation::Manual { u: sample.u() },
                );
                map(&q_map(geometry, values, wavelength, detector)?, |q| {
                    fixed.hkl(&q).into()
                })
            }
            Self::QparQper { normal } => {
                map(&q_map(geometry, values, wavelength, detector)?, |q| {
                    let qper = q.dot(normal);
                    [(q - qper * normal.into_inner()).norm(), qper]
                })
            }
            Self::Angles => {
                let poses = geometry.poses(values)?;
                let ki = (poses.source * geometry.beam_direction()).into_inner();
                // binoculars takes acos(x) of the unnormalised pixel
                // position (hkl-binoculars.c:369, BIN-13)
                map(&detector_positions(geometry, values, detector)?, |p| {
                    [
                        p.z.atan2(p.x),
                        p.y.atan2(p.x.hypot(p.z)),
                        angle_between(&ki, &p).0,
                    ]
                })
            }
        };
        Projected::new(self.coords().to_vec(), planes)
    }
}

/// `f` of every pixel's vector, one plane per output coordinate, each over
/// the planes' shape.
fn map<const N: usize>(planes: &Planes, f: impl Fn(Vector3<f64>) -> [f64; N]) -> Vec<Image<f64>> {
    let values = Image::from_fn(planes.shape(), |s, j| f(planes.get(s, j)));
    (0..N)
        .map(|k| Image::from_fn(values.shape(), |s, j| values.get(s, j)[k]))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use nalgebra::{Matrix3, UnitQuaternion};

    use super::*;
    use crate::absorption::{Masked, Weight};
    use crate::grid::{Binning, CoordKind, Cube};
    use crate::normalise::Normalised;
    use crate::presets::preset;
    use crate::{Lattice, Rad, forward};

    fn setup() -> (Geometry, Vec<f64>, Wavelength, PixelDetector) {
        let g = preset("K6C").unwrap();
        // mu, komega, kappa, kphi, gamma, delta
        let values = g
            .values_from_degrees(&[3.0, 20.0, -30.0, 40.0, 12.0, 25.0])
            .unwrap();
        let wavelength = Wavelength::new(1.54).unwrap();
        let d = PixelDetector::new(
            (3, 4),
            Vector3::new(0.5, -0.002, -0.0015),
            Vector3::new(0.0, 0.0, 0.001),
            Vector3::new(0.0, 0.001, 0.0),
        )
        .unwrap();
        (g, values, wavelength, d)
    }

    fn sample() -> Sample {
        let u = UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3);
        Sample::new(
            "s",
            Lattice::new(4.1, 5.2, 6.3, Rad(1.5), Rad(1.6), Rad(1.7)).unwrap(),
            Orientation::Manual { u },
        )
    }

    fn frame() -> Rotation3<f64> {
        Rotation3::from_euler_angles(-0.4, 0.25, 1.1)
    }

    fn all() -> [Projection; 6] {
        [
            Projection::QLab,
            Projection::QSample,
            Projection::QSurface { frame: frame() },
            Projection::Hkl(sample()),
            Projection::QparQper {
                normal: Unit::new_normalize(Vector3::new(0.2, -0.1, 1.0)),
            },
            Projection::Angles,
        ]
    }

    #[test]
    fn every_pixel_matches_forward() {
        let (g, values, wavelength, d) = setup();
        let s = sample();
        let n = Unit::new_normalize(Vector3::new(0.2, -0.1, 1.0));
        let out = all().map(|p| p.project(&g, &values, wavelength, &d).unwrap());
        let (slow, fast) = d.shape();
        for i in 0..slow {
            for j in 0..fast {
                let px = i * fast + j;
                let f = forward(&g, &s, &values, wavelength, &d.pixel(i, j).unwrap()).unwrap();
                let sc = f.scattering;
                let at = |k: usize| -> Vec<f64> {
                    out[k].planes().iter().map(|p| p.get(i, j)).collect()
                };
                let close =
                    |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
                assert!(close(&at(0), sc.q_lab.as_slice()), "QLab {px}");
                assert!(close(&at(1), sc.q_sample.as_slice()), "QSample {px}");
                let on_frame = frame().matrix().tr_mul(&sc.q_sample);
                assert!(close(&at(2), on_frame.as_slice()), "QSurface {px}");
                assert!(close(&at(3), f.hkl.as_slice()), "Hkl {px}");
                let qper = sc.q_sample.dot(&n);
                let qpar = (sc.q_sample.norm_squared() - qper * qper).sqrt();
                assert!(close(&at(4), &[qpar, qper]), "QparQper {px}");
                let kf = sc.kf;
                let angles = [kf.z.atan2(kf.x), kf.y.atan2(kf.x.hypot(kf.z)), sc.tth.0];
                assert!(close(&at(5), &angles), "Angles {px}");
            }
        }
    }

    /// binoculars' VERTICAL surface frame Rx(−π/2): (qx, −qz, qy) of
    /// Q_sample, to the bit, the surface normal ŷ on the third axis.
    #[test]
    fn vertical_surface_puts_the_normal_on_qz() {
        let (g, values, wavelength, d) = setup();
        let vertical = Rotation3::from_matrix_unchecked(Matrix3::new(
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, -1.0, 0.0,
        ));
        let s = Projection::QSample
            .project(&g, &values, wavelength, &d)
            .unwrap();
        let v = Projection::QSurface { frame: vertical }
            .project(&g, &values, wavelength, &d)
            .unwrap();
        let [qx, qy, qz] = [0, 1, 2].map(|k| s.planes()[k].data());
        let [vx, vy, vz] = [0, 1, 2].map(|k| v.planes()[k].data());
        for i in 0..qx.len() {
            assert_eq!([vx[i], vy[i], vz[i]], [qx[i], -qz[i], qy[i]], "{i}");
        }
    }

    #[test]
    fn bin8_names_come_from_the_projection() {
        // BIN-8: binoculars labelled a 2θ axis `q`; here the planes are
        // labelled by the variant that computed them, name and kind
        let (g, values, wavelength, d) = setup();
        for p in all() {
            let out = p.project(&g, &values, wavelength, &d).unwrap();
            assert_eq!(out.coords(), p.coords());
            assert_eq!(out.planes().len(), p.coords().len());
            let mut cube = Cube::default();
            let binnings = vec![Binning::new(1e-3, 0.0).unwrap(); p.coords().len()];
            let one = Normalised::new(1.0, Weight::new(1.0, Masked::Opaque));
            cube.add_growing(&binnings, &out, &Image::from_fn(out.shape(), |_, _| one))
                .unwrap();
            let axes: Vec<&Coord> = cube.axes().iter().map(|a| a.coord()).collect();
            assert_eq!(axes, p.coords().iter().collect::<Vec<_>>());
            let angle = matches!(p, Projection::Angles);
            for a in cube.axes() {
                assert_eq!(a.kind() == CoordKind::Angle, angle, "{}", a.name());
            }
        }
        assert_eq!(Projection::Angles.coords()[2], Coord::angle("tth"));
    }

    #[test]
    fn angles_follow_the_scattered_beam() {
        // gamma 0, delta 90° on K6C sends kf straight up: delta_lab = 90°,
        // gamma_lab = 0, tth = 90°
        let g = preset("K6C").unwrap();
        let values = g
            .values_from_degrees(&[0.0, 0.0, 0.0, 0.0, 0.0, 90.0])
            .unwrap();
        let wavelength = Wavelength::new(1.0).unwrap();
        let d = PixelDetector::point(Unit::new_normalize(Vector3::x()), 2.0).unwrap();
        let a = Projection::Angles
            .project(&g, &values, wavelength, &d)
            .unwrap();
        let v: Vec<f64> = a.planes().iter().map(|p| p.get(0, 0)).collect();
        for (x, y) in v.iter().zip([FRAC_PI_2, 0.0, FRAC_PI_2]) {
            assert!((x - y).abs() < 1e-12, "{v:?}");
        }
    }
}
