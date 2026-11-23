//! Batch forward: one frame's axis values over a whole detector.
//!
//! Every pixel goes through [`scattering_at`], the kernel [`crate::forward`]
//! uses; the map only decides which part of its result to store.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::absorption::{Masked, Weight};
use crate::{Error, Geometry, PixelDetector, Poses, Scattering, Wavelength, scattering_at};

/// Axis values (axis order, radians / metres) and wavelength of one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub values: Vec<f64>,
    pub wavelength: Wavelength,
}

/// Three contiguous f64 planes (x, y, z), row-major (slow, fast).
#[derive(Clone, Debug, PartialEq)]
pub struct Planes {
    shape: (usize, usize),
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// Q in the sample frame per pixel, Å⁻¹ with 2π.
pub type QMap = Planes;

impl Planes {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The vector at pixel (slow, fast); panics outside the shape.
    pub fn get(&self, slow: usize, fast: usize) -> Vector3<f64> {
        assert!(slow < self.shape.0 && fast < self.shape.1);
        let i = slow * self.shape.1 + fast;
        Vector3::new(self.x[i], self.y[i], self.z[i])
    }
}

/// Fills `N` planes of `detector`'s shape, one rayon task per row. `f` sees
/// the frame's poses, the lab-frame sample-to-pixel vector and the kernel's
/// [`Scattering`] for each pixel; it overwrites `blank` at every pixel.
fn fill<T: Copy + Send, const N: usize>(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
    blank: T,
    f: impl Fn(&Poses, &Vector3<f64>, &Scattering) -> [T; N] + Sync,
) -> Result<[Vec<T>; N], Error> {
    let (slow, fast) = detector.shape();
    let poses = geometry.poses(&frame.values)?;
    let mut planes: [Vec<T>; N] = std::array::from_fn(|_| vec![blank; slow * fast]);
    let mut chunks: Vec<_> = planes.iter_mut().map(|p| p.chunks_mut(fast)).collect();
    let rows: Vec<[&mut [T]; N]> = (0..slow)
        .map(|_| std::array::from_fn(|k| chunks[k].next().expect("one chunk per row")))
        .collect();
    rows.into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..fast {
                let point: Point3<f64> = detector.pixel(i, j)?;
                let (p, s) = scattering_at(geometry, &poses, frame.wavelength, &point)?;
                for (plane, v) in row.iter_mut().zip(f(&poses, &p, &s)) {
                    plane[j] = v;
                }
            }
            Ok(())
        })?;
    Ok(planes)
}

fn planes3(shape: (usize, usize), [x, y, z]: [Vec<f64>; 3]) -> Planes {
    Planes { shape, x, y, z }
}

/// Q in the sample frame for every pixel.
pub fn q_map(geometry: &Geometry, frame: &Frame, detector: &PixelDetector) -> Result<QMap, Error> {
    let planes = fill(geometry, frame, detector, 0.0, |_, _, s| s.q_sample.into())?;
    Ok(planes3(detector.shape(), planes))
}

/// Pixel positions relative to the sample, lab frame, metres: the `p` of
/// the forward kernel.
pub fn detector_positions(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
) -> Result<Planes, Error> {
    let planes = fill(geometry, frame, detector, 0.0, |_, p, _| (*p).into())?;
    Ok(planes3(detector.shape(), planes))
}

/// One weight per pixel, row-major (slow, fast): a positive finite
/// [`Weight`], or the reason the pixel has none, so a masked pixel holds no
/// number to read.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    shape: (usize, usize),
    pub data: Vec<Result<Weight, Masked>>,
}

impl Weights {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The weight at pixel (slow, fast); panics outside the shape.
    pub fn get(&self, slow: usize, fast: usize) -> Result<Weight, Masked> {
        assert!(slow < self.shape.0 && fast < self.shape.1);
        self.data[slow * self.shape.1 + fast]
    }
}

/// Every pixel overwrites it.
const BLANK: Result<Weight, Masked> = Err(Masked::Opaque);

fn weights(shape: (usize, usize), [data]: [Vec<Result<Weight, Masked>>; 1]) -> Weights {
    Weights { shape, data }
}

/// Solid angle of each pixel, sr: `|(e_slow × e_fast) · k̂f| / |p|²`, with
/// the pixel area vector rotated by the detector pose. A detector whose
/// steps span no area (a point, or a linear detector built with a zero
/// `e_slow` rather than a pixel height) has none and gets
/// [`Error::NoPixelArea`]; a pixel seen edge-on is [`Masked::EdgeOn`].
pub fn solid_angle(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
) -> Result<Weights, Error> {
    let shape = detector.shape();
    let area = detector.e_slow().cross(&detector.e_fast());
    let norm = area.norm();
    if norm.is_nan() || norm <= 0.0 {
        return Err(Error::NoPixelArea { shape });
    }
    let data = fill(geometry, frame, detector, BLANK, |poses, p, s| {
        let omega = (poses.detector.rotation * area).dot(&s.kf).abs() / p.norm_squared();
        [Weight::new(omega, Masked::EdgeOn)]
    })?;
    Ok(weights(shape, data))
}

/// Polarisation factor of the scattered intensity for a beam whose electric
/// field lies along `e` (a lab-frame vector, projected normal to k̂i) with
/// degree `factor` (1 fully along `e`, 0 unpolarised, −1 fully across):
/// `P = ½(1+f)(1−(ê·k̂f)²) + ½(1−f)(1−(ê⊥·k̂f)²)`, ê⊥ = k̂i × ê. With
/// `e` the horizontal direction this is pyFAI's
/// `½(1 + cos²2θ − f cos 2χ sin²2θ)`. Where it vanishes the pixel is
/// [`Masked::Extinct`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polarisation {
    pub e: Vector3<f64>,
    pub factor: f64,
}

pub fn polarisation(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
    polarisation: Polarisation,
) -> Result<Weights, Error> {
    let f = polarisation.factor;
    if !(-1.0..=1.0).contains(&f) {
        return Err(Error::PolarisationFactor { value: f });
    }
    // k̂i is the same for every pixel of the frame
    let poses = geometry.poses(&frame.values)?;
    let ki = (poses.source * geometry.beam_direction()).into_inner();
    let e = polarisation.e - ki * polarisation.e.dot(&ki);
    let norm = e.norm();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(Error::ZeroVector {
            what: "polarisation vector normal to the beam".to_owned(),
        });
    }
    let e = e / norm;
    let across = ki.cross(&e);
    let data = fill(geometry, frame, detector, BLANK, |_, _, s| {
        let kf = s.kf.into_inner();
        let p = 0.5 * (1.0 + f) * (1.0 - e.dot(&kf).powi(2))
            + 0.5 * (1.0 - f) * (1.0 - across.dot(&kf).powi(2));
        [Weight::new(p, Masked::Extinct)]
    })?;
    Ok(weights(detector.shape(), data))
}

#[cfg(test)]
mod tests {
    use nalgebra::Unit;

    use super::*;
    use crate::{Lattice, Orientation, Sample, forward};

    /// Sample ω about z, detector 2θ about z and ν about y, a detector
    /// translation along x: every pose component enters p.
    fn geometry() -> Geometry {
        Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .detector_translation("dx", [1.0, 0.0, 0.0])
            .detector_rotation("tt", [0.0, 0.0, 1.0])
            .detector_rotation("nu", [0.0, 1.0, 0.0])
            .build()
            .unwrap()
    }

    fn frame() -> Frame {
        Frame {
            values: vec![0.31, 0.002, 0.57, -0.23],
            wavelength: Wavelength::new(1.54).unwrap(),
        }
    }

    fn detector() -> PixelDetector {
        PixelDetector::new(
            (5, 7),
            Vector3::new(0.3, -0.01, -0.02),
            Vector3::new(0.0, 0.0, 1e-3),
            Vector3::new(0.0, 2e-3, 0.0),
        )
        .unwrap()
    }

    fn sample() -> Sample {
        Sample::new(
            "s",
            Lattice::cubic(4.0).unwrap(),
            Orientation::from_u_matrix(&nalgebra::Matrix3::identity()).unwrap(),
        )
    }

    #[test]
    fn batch_equals_per_pixel_forward_exactly() {
        let (g, f, d) = (geometry(), frame(), detector());
        let map = q_map(&g, &f, &d).unwrap();
        assert_eq!(map.shape(), (5, 7));
        for i in 0..5 {
            for j in 0..7 {
                let one = forward(
                    &g,
                    &sample(),
                    &f.values,
                    f.wavelength,
                    &d.pixel(i, j).unwrap(),
                )
                .unwrap();
                let got = map.get(i, j);
                assert!(got.iter().all(|v| v.is_finite()));
                assert_eq!(got, one.scattering.q_sample, "pixel ({i}, {j})");
            }
        }
    }

    #[test]
    fn position_direction_is_kf_and_p_is_the_kernels() {
        let (g, f, d) = (geometry(), frame(), detector());
        let pos = detector_positions(&g, &f, &d).unwrap();
        let poses = g.poses(&f.values).unwrap();
        for i in 0..5 {
            for j in 0..7 {
                let one = forward(
                    &g,
                    &sample(),
                    &f.values,
                    f.wavelength,
                    &d.pixel(i, j).unwrap(),
                )
                .unwrap();
                let p = pos.get(i, j);
                let want = poses.detector * d.pixel(i, j).unwrap() - poses.sample_position();
                assert_eq!(p, want);
                let dir = Unit::new_normalize(p);
                let diff = (dir.into_inner() - one.scattering.kf.into_inner()).norm();
                assert!(diff < 1e-15, "pixel ({i}, {j}): {diff:e}");
            }
        }
    }

    fn point_at(direction: [f64; 3]) -> PixelDetector {
        PixelDetector::point(Unit::new_normalize(direction.into()), 0.5).unwrap()
    }

    fn one(g: &Geometry, d: &PixelDetector, pol: Polarisation) -> Result<f64, Masked> {
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        polarisation(g, &f, d, pol)
            .unwrap()
            .get(0, 0)
            .map(Weight::get)
    }

    fn om_only() -> Geometry {
        Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .build()
            .unwrap()
    }

    #[test]
    fn polarisation_at_90_degrees_in_and_out_of_the_plane() {
        let g = om_only();
        // beam along x: the field along y is horizontal, along z vertical
        let h = |factor| Polarisation {
            e: Vector3::y(),
            factor,
        };
        let v = |factor| Polarisation {
            e: Vector3::z(),
            factor,
        };
        // beam along x, in the plane means kf along y; along the field
        // nothing scatters
        let inp = point_at([0.0, 1.0, 0.0]);
        let out = point_at([0.0, 0.0, 1.0]);
        let (none, all) = (Err(Masked::Extinct), Ok(1.0));
        for (d, ph, pv) in [(&inp, none, all), (&out, all, none)] {
            assert_eq!(one(&g, d, h(1.0)), ph);
            assert_eq!(one(&g, d, v(1.0)), pv);
            assert_eq!(one(&g, d, h(-1.0)), pv);
            assert_eq!(one(&g, d, h(0.0)), Ok(0.5));
        }
    }

    #[test]
    fn polarisation_matches_the_2theta_chi_form() {
        let g = om_only();
        let (tth, chi) = (0.7_f64, 0.4_f64);
        // ki = x, horizontal e = z × x = y: χ is measured from y towards z
        let d = point_at([tth.cos(), tth.sin() * chi.cos(), tth.sin() * chi.sin()]);
        for f in [1.0, 0.3, 0.0, -0.6] {
            let want = 0.5 * (1.0 + tth.cos().powi(2) - f * (2.0 * chi).cos() * tth.sin().powi(2));
            let got = one(
                &g,
                &d,
                Polarisation {
                    e: Vector3::y(),
                    factor: f,
                },
            )
            .unwrap();
            assert!((got - want).abs() < 1e-15, "f = {f}: {got} vs {want}");
        }
    }

    #[test]
    fn polarisation_factor_outside_the_unit_interval_is_an_error() {
        let g = om_only();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let pol = Polarisation {
            e: Vector3::y(),
            factor: f64::NAN,
        };
        assert!(matches!(
            polarisation(&g, &f, &point_at([0.0, 1.0, 0.0]), pol),
            Err(Error::PolarisationFactor { .. })
        ));
    }

    #[test]
    fn polarisation_vector_is_projected_normal_to_the_beam() {
        let g = om_only();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let d = point_at([0.3, 0.8, 0.5]);
        let pol = |e| Polarisation { e, factor: 0.7 };
        let w = |e| polarisation(&g, &f, &d, pol(e)).map(|w| w.get(0, 0).unwrap().get());
        // the beam-parallel part of e and its length do not matter
        let a = w(Vector3::y()).unwrap();
        let b = w(Vector3::new(5.0, 3.0, 0.0)).unwrap();
        assert!((a - b).abs() < 1e-15);
        // along the beam, or not finite, there is no direction
        for e in [
            Vector3::x(),
            Vector3::zeros(),
            Vector3::new(f64::NAN, 0.0, 0.0),
        ] {
            assert!(matches!(w(e), Err(Error::ZeroVector { .. })), "{e:?}");
        }
    }

    /// Detector at `d` along x, steps (a along z, b along y); pixel (0, 0)
    /// is at normal incidence.
    fn flat(d: f64, a: f64, b: f64) -> PixelDetector {
        PixelDetector::new(
            (4, 3),
            Vector3::new(d, 0.0, 0.0),
            Vector3::new(0.0, 0.0, a),
            Vector3::new(0.0, b, 0.0),
        )
        .unwrap()
    }

    #[test]
    fn solid_angle_at_normal_incidence_and_tilted() {
        let g = om_only();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let (d, a, b) = (0.4_f64, 1e-3, 2e-3);
        let w = solid_angle(&g, &f, &flat(d, a, b)).unwrap();
        let normal = a * b / (d * d);
        assert!((w.get(0, 0).unwrap().get() - normal).abs() < 1e-12 * normal);
        // pixel (3, 2) sits at (d, 2b, 3a): cos α = d / r
        let r2 = d * d + (2.0 * b).powi(2) + (3.0 * a).powi(2);
        let tilted = a * b * d / r2.powf(1.5);
        assert!((w.get(3, 2).unwrap().get() - tilted).abs() < 1e-12 * tilted);
    }

    #[test]
    fn solid_angle_follows_the_detector_pose() {
        // detector turned by tt about z: the area vector turns with it, so
        // the on-axis pixel stays at normal incidence
        let g = Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .detector_rotation("tt", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let f = Frame {
            values: vec![0.0, 0.9],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let (d, a, b) = (0.4_f64, 1e-3, 2e-3);
        let w = solid_angle(&g, &f, &flat(d, a, b)).unwrap();
        let normal = a * b / (d * d);
        assert!((w.get(0, 0).unwrap().get() - normal).abs() < 1e-12 * normal);
    }

    #[test]
    fn solid_angle_of_a_linear_detector_uses_its_row_height() {
        let g = om_only();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let (d, h, b) = (0.4_f64, 5e-3, 2e-3);
        // one row along y, pixel height h along z
        let row = |e_slow| {
            PixelDetector::new(
                (1, 4),
                Vector3::new(d, 0.0, 0.0),
                e_slow,
                Vector3::new(0.0, b, 0.0),
            )
            .unwrap()
        };
        let w = solid_angle(&g, &f, &row(Vector3::new(0.0, 0.0, h))).unwrap();
        assert_eq!(w.shape(), (1, 4));
        let normal = h * b / (d * d);
        assert!((w.get(0, 0).unwrap().get() - normal).abs() < 1e-12 * normal);
        let r2 = d * d + (3.0 * b).powi(2);
        let tilted = h * b * d / r2.powf(1.5);
        assert!((w.get(0, 3).unwrap().get() - tilted).abs() < 1e-12 * tilted);
        // built with a zero e_slow it has no height to give
        assert_eq!(
            solid_angle(&g, &f, &row(Vector3::zeros())),
            Err(Error::NoPixelArea { shape: (1, 4) })
        );
    }

    #[test]
    fn solid_angle_needs_a_pixel_area() {
        let g = om_only();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        assert_eq!(
            solid_angle(&g, &f, &point_at([1.0, 0.0, 0.0])),
            Err(Error::NoPixelArea { shape: (1, 1) })
        );
    }

    /// A detector plane through the sample is seen edge-on at every pixel.
    #[test]
    fn solid_angle_masks_a_pixel_seen_edge_on() {
        let g = om_only();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let d = PixelDetector::new(
            (2, 3),
            Vector3::new(0.0, 0.3, 0.0),
            Vector3::new(0.0, 0.0, 1e-3),
            Vector3::new(0.0, 2e-3, 0.0),
        )
        .unwrap();
        let w = solid_angle(&g, &f, &d).unwrap();
        assert!(w.data.iter().all(|w| *w == Err(Masked::EdgeOn)), "{w:?}");
    }

    #[test]
    fn a_pixel_on_the_sample_is_an_error() {
        let g = Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let d = PixelDetector::new(
            (2, 2),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert_eq!(q_map(&g, &f, &d), Err(Error::DetectorAtSample));
    }
}
