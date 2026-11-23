//! Batch forward: one frame's axis values over a whole detector.
//!
//! Every pixel goes through [`scattering_at`], the kernel [`crate::forward`]
//! uses; the map only decides which part of its result to store.

use nalgebra::Vector3;
use rayon::prelude::*;

use crate::absorption::{Absorber, Attenuation, Masked, SlabFrame, Weight};
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

/// A pixel as [`fill`] visits it: `module` indexes the detector's modules
/// and `local` is the pixel (slow, fast) in that module.
#[derive(Clone, Copy)]
struct At {
    module: usize,
    local: (usize, usize),
}

/// Fills `N` planes of `detector`'s shape, one rayon task per row. `f` sees
/// the frame's poses, where the pixel is, the lab-frame sample-to-pixel
/// vector and the kernel's [`Scattering`] for each pixel; as the modules
/// tile the image, it overwrites `blank` at every pixel.
fn fill<T: Copy + Send, const N: usize>(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
    blank: T,
    f: impl Fn(&Poses, At, &Vector3<f64>, &Scattering) -> [T; N] + Sync,
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
            for (k, m) in detector.modules().iter().enumerate() {
                let ((a1, a2), (n1, n2)) = (m.at(), m.shape());
                if !(a1..a1 + n1).contains(&i) {
                    continue;
                }
                for l in 0..n2 {
                    let at = At {
                        module: k,
                        local: (i - a1, l),
                    };
                    let point = m.pixel(i - a1, l);
                    let (p, s) = scattering_at(geometry, &poses, frame.wavelength, &point)?;
                    for (plane, v) in row.iter_mut().zip(f(&poses, at, &p, &s)) {
                        plane[a2 + l] = v;
                    }
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
    let planes = fill(geometry, frame, detector, 0.0, |_, _, _, s| {
        s.q_sample.into()
    })?;
    Ok(planes3(detector.shape(), planes))
}

/// Pixel positions relative to the sample, lab frame, metres: the `p` of
/// the forward kernel.
pub fn detector_positions(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
) -> Result<Planes, Error> {
    let planes = fill(geometry, frame, detector, 0.0, |_, _, p, _| (*p).into())?;
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

/// Solid angle of each pixel, sr: `|A · k̂f| / |p|²`, with `A` the pixel's
/// area vector (its widths along the module axes times `slow × fast`)
/// rotated by the detector pose. A detector with a module whose pixels span
/// no area (a point axis, as a point or an xu linear detector has) gets
/// [`Error::NoPixelArea`]; a pixel seen edge-on is [`Masked::EdgeOn`].
pub fn solid_angle(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
) -> Result<Weights, Error> {
    let modules = detector.modules();
    if modules.iter().any(|m| m.normal().is_none()) {
        return Err(Error::NoPixelArea);
    }
    let data = fill(geometry, frame, detector, BLANK, |poses, at, p, s| {
        let (ls, lf) = at.local;
        let area = modules[at.module].area(ls, lf);
        let omega = (poses.detector.rotation * area).dot(&s.kf).abs() / p.norm_squared();
        [Weight::new(omega, Masked::EdgeOn)]
    })?;
    Ok(weights(detector.shape(), data))
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
    let data = fill(geometry, frame, detector, BLANK, |_, _, _, s| {
        let kf = s.kf.into_inner();
        let p = 0.5 * (1.0 + f) * (1.0 - e.dot(&kf).powi(2))
            + 0.5 * (1.0 - f) * (1.0 - across.dot(&kf).powi(2));
        [Weight::new(p, Masked::Extinct)]
    })?;
    Ok(weights(detector.shape(), data))
}

/// Absorption weight per pixel (design §5.1): the sample slab's times the
/// detector path's, at the kernel's k̂f and p. Every μ is taken at the frame
/// energy and F computed once, before the pixel loop, which sees only the
/// resulting [`SlabFrame`] and each module's
/// [`crate::absorption::OrientedPath`]: the detector layers lie along the
/// module's normal, and a window or sensor on a module with none is
/// [`Error::NoPixelArea`]. Where both sides mask, the pixel carries the
/// sample's reason.
pub fn absorption<A: Attenuation>(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
    absorber: &Absorber<A>,
) -> Result<Weights, Error> {
    let poses = geometry.poses(&frame.values)?;
    let energy = frame.wavelength.energy();
    let sample = SlabFrame::new(
        &absorber.slab,
        &absorber.beam,
        absorber.sample.mu(energy),
        &(poses.source * geometry.beam_direction()),
        &poses.sample.rotation,
    )?;
    let path = absorber.detector.at(energy);
    let paths = detector
        .modules()
        .iter()
        .map(|m| path.oriented(m.normal().map(|n| poses.detector.rotation * n)))
        .collect::<Result<Vec<_>, _>>()?;
    let data = fill(geometry, frame, detector, BLANK, |_, at, p, s| {
        [sample
            .weight(&s.kf)
            .and_then(|a| a.times(paths[at.module].weight(&s.kf, p.norm())?))]
    })?;
    Ok(weights(detector.shape(), data))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use nalgebra::Unit;

    use super::*;
    use crate::absorption::{Beam, DetectorPath, Face, Layer, Slab};
    use crate::xu::Linear;
    use crate::{Energy, Lattice, Module, Orientation, PerMetre, PixelAxis, Sample, forward};

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
            Err(Error::NoPixelArea)
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
            Err(Error::NoPixelArea)
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

    fn per_metre(v: f64) -> PerMetre {
        PerMetre::new(v).unwrap()
    }

    /// 5 mm square, `normal` in the sample frame.
    fn face(normal: [f64; 3]) -> Face {
        Face::new(normal.into(), Vector3::z(), 5e-3, 5e-3).unwrap()
    }

    /// Far smaller than the face.
    fn beam() -> Beam {
        Beam::top_hat(Vector3::z(), [1e-4, 2e-4]).unwrap()
    }

    fn absorber(slab: Slab) -> Absorber<PerMetre> {
        Absorber {
            slab,
            beam: beam(),
            sample: per_metre(1e4),
            detector: DetectorPath {
                air: Some(per_metre(1.0)),
                window: Some(Layer::new(per_metre(1e3), 25e-6).unwrap()),
                sensor: Some(Layer::new(per_metre(3e4), 450e-6).unwrap()),
            },
        }
    }

    // In `geometry()`'s frame, a sample face with normal +y faces the beam
    // and the detector, one with −x the beam only, one with −y neither.

    /// Bit for bit SlabFrame's times DetectorPath's at the kernel's k̂f and
    /// p, in reflection and in transmission.
    #[test]
    fn absorption_equals_slab_frame_and_detector_path_at_the_kernels_kf_and_p() {
        let (g, f, d) = (geometry(), frame(), detector());
        let poses = g.poses(&f.values).unwrap();
        let ki = poses.source * g.beam_direction();
        let normal = poses.detector.rotation * d.modules()[0].normal().unwrap();
        for (n, reflection) in [([0.0, 1.0, 0.0], true), ([-1.0, 0.0, 0.0], false)] {
            let a = absorber(Slab::new(face(n), 1e-4).unwrap());
            let w = absorption(&g, &f, &d, &a).unwrap();
            let path = a.detector.oriented(Some(normal)).unwrap();
            let sample =
                SlabFrame::new(&a.slab, &a.beam, a.sample, &ki, &poses.sample.rotation).unwrap();
            let n_lab = poses.sample.rotation * Vector3::from(n);
            for i in 0..5 {
                for j in 0..7 {
                    let point = d.pixel(i, j).unwrap();
                    let (p, s) = scattering_at(&g, &poses, f.wavelength, &point).unwrap();
                    assert_eq!(s.kf.dot(&n_lab) > 0.0, reflection, "{n:?} ({i}, {j})");
                    let want = sample.weight(&s.kf).unwrap();
                    let want = want.times(path.weight(&s.kf, p.norm()).unwrap());
                    assert_eq!(w.get(i, j), want, "{n:?} ({i}, {j})");
                    assert!(want.is_ok(), "{n:?} ({i}, {j})");
                }
            }
        }
    }

    /// Each side's mask reaches the pixel; where both mask, the sample's
    /// reason wins.
    #[test]
    fn absorption_masks_carry_their_reason() {
        let (g, f, d) = (geometry(), frame(), detector());
        let every = |a: &Absorber<PerMetre>, want: Masked| {
            let w = absorption(&g, &f, &d, a).unwrap();
            assert!(w.data.iter().all(|r| *r == Err(want)), "{want:?}: {w:?}");
        };
        let lit = Slab::new(face([0.0, 1.0, 0.0]), 1e-4).unwrap();
        let unlit = Slab::new(face([0.0, -1.0, 0.0]), 1e-4).unwrap();
        // optical depth 1e4 at normal incidence: transmits 0 in f64
        let opaque = DetectorPath {
            window: Some(Layer::new(per_metre(1e7), 1e-3).unwrap()),
            ..Default::default()
        };
        every(&absorber(unlit), Masked::Unlit);
        every(
            &absorber(Slab::bulk(face([-1.0, 0.0, 0.0]))),
            Masked::ThroughBulk,
        );
        every(
            &Absorber {
                detector: opaque,
                ..absorber(lit)
            },
            Masked::Opaque,
        );
        every(
            &Absorber {
                detector: opaque,
                ..absorber(unlit)
            },
            Masked::Unlit,
        );
    }

    /// Two modules with different normals: each pixel's layers lie along its
    /// own module's.
    #[test]
    fn absorption_orients_each_module_along_its_own_normal() {
        let (g, f) = (geometry(), frame());
        let axis = |v: [f64; 3], n: usize, pitch: f64| {
            PixelAxis::uniform(Unit::new_normalize(v.into()), n, pitch).unwrap()
        };
        let modules = vec![
            Module::new(
                (0, 0),
                [0.3, -0.01, -0.02].into(),
                axis([0.0, 0.0, 1.0], 2, 1e-3),
                axis([0.0, 1.0, 0.0], 3, 2e-3),
            )
            .unwrap(),
            Module::new(
                (2, 0),
                [0.3, -0.01, 0.01].into(),
                axis([-0.4, 0.0, 1.0], 2, 1e-3),
                axis([0.0, 1.0, 0.0], 3, 2e-3),
            )
            .unwrap(),
        ];
        let d = PixelDetector::from_modules((4, 3), modules).unwrap();
        let poses = g.poses(&f.values).unwrap();
        let ki = poses.source * g.beam_direction();
        let a = absorber(Slab::new(face([0.0, 1.0, 0.0]), 1e-4).unwrap());
        let sample =
            SlabFrame::new(&a.slab, &a.beam, a.sample, &ki, &poses.sample.rotation).unwrap();
        let paths: Vec<_> = d
            .modules()
            .iter()
            .map(|m| {
                let n = poses.detector.rotation * m.normal().unwrap();
                a.detector.oriented(Some(n)).unwrap()
            })
            .collect();
        assert_ne!(paths[0], paths[1]);
        let w = absorption(&g, &f, &d, &a).unwrap();
        for i in 0..4 {
            for j in 0..3 {
                let (p, s) =
                    scattering_at(&g, &poses, f.wavelength, &d.pixel(i, j).unwrap()).unwrap();
                let path = paths[i / 2];
                let want = sample
                    .weight(&s.kf)
                    .unwrap()
                    .times(path.weight(&s.kf, p.norm()).unwrap());
                assert_eq!(w.get(i, j), want, "({i}, {j})");
                assert!(want.is_ok(), "({i}, {j})");
            }
        }
    }

    /// Air alone needs no normal, so it weighs a point detector and an xu
    /// row; a window or sensor there is an error.
    #[test]
    fn absorption_layers_need_a_module_normal() {
        let (g, f) = (geometry(), frame());
        let row = Linear::new("z+".parse().unwrap(), 2.0, 5, 0.3, 1e-4)
            .detector(Vector3::x_axis())
            .unwrap();
        let air = Absorber {
            detector: DetectorPath {
                air: Some(per_metre(1.0)),
                ..Default::default()
            },
            ..absorber(Slab::new(face([0.0, 1.0, 0.0]), 1e-4).unwrap())
        };
        for d in [point_at([1.0, -0.03, -0.07]), row] {
            assert!(d.modules().iter().all(|m| m.normal().is_none()));
            let w = absorption(&g, &f, &d, &air).unwrap();
            assert!(w.data.iter().all(Result::is_ok), "{w:?}");
            for detector in [
                DetectorPath {
                    window: Some(Layer::new(per_metre(1e3), 25e-6).unwrap()),
                    ..air.detector
                },
                DetectorPath {
                    sensor: Some(Layer::new(per_metre(3e4), 450e-6).unwrap()),
                    ..air.detector
                },
            ] {
                let a = Absorber { detector, ..air };
                assert_eq!(absorption(&g, &f, &d, &a), Err(Error::NoPixelArea));
            }
        }
    }

    /// Records the energy of every call. The RefCell makes it !Sync, so the
    /// pixel loop, which `fill` requires to be Sync, cannot reach it.
    struct Recording(RefCell<Vec<f64>>);

    impl Attenuation for Recording {
        fn mu(&self, energy: Energy) -> PerMetre {
            self.0.borrow_mut().push(energy.kev());
            per_metre(1e3)
        }
    }

    #[test]
    fn absorption_takes_each_mu_once_per_frame_at_its_energy() {
        let (g, d) = (geometry(), detector());
        let r = Recording(RefCell::new(Vec::new()));
        let a = Absorber {
            slab: Slab::new(face([0.0, 1.0, 0.0]), 1e-4).unwrap(),
            beam: beam(),
            sample: &r,
            detector: DetectorPath {
                air: Some(&r),
                window: Some(Layer::new(&r, 25e-6).unwrap()),
                sensor: Some(Layer::new(&r, 450e-6).unwrap()),
            },
        };
        let mut want = Vec::new();
        for angstrom in [1.54, 1.0] {
            let f = Frame {
                wavelength: Wavelength::new(angstrom).unwrap(),
                ..frame()
            };
            let w = absorption(&g, &f, &d, &a).unwrap();
            assert!(w.data.iter().all(Result::is_ok), "{w:?}");
            want.extend([f.wavelength.energy().kev(); 4]);
        }
        assert_eq!(*r.0.borrow(), want);
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
