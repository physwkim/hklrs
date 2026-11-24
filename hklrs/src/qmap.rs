//! Batch forward: one frame's axis values over a whole detector.
//!
//! Every pixel's position comes from [`detector_position`] at the frame's
//! poses, and what depends on the scattered beam from [`scattered_direction`]
//! or [`scattering_at`], the kernels [`crate::forward`] uses. A map takes the
//! wavelength only if its result depends on it.

use nalgebra::Vector3;
use rayon::prelude::*;

use crate::absorption::{Absorber, Attenuation, Masked, SlabFrame, Weight};
use crate::{
    Error, Geometry, Image, PixelDetector, Poses, Wavelength, detector_position,
    scattered_direction, scattering_at,
};

/// Three contiguous f64 planes (x, y, z) of one shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Planes {
    x: Image<f64>,
    y: Image<f64>,
    z: Image<f64>,
}

/// Q in the sample frame per pixel, Å⁻¹ with 2π.
pub type QMap = Planes;

impl Planes {
    pub fn shape(&self) -> (usize, usize) {
        self.x.shape()
    }

    pub fn x(&self) -> &Image<f64> {
        &self.x
    }

    pub fn y(&self) -> &Image<f64> {
        &self.y
    }

    pub fn z(&self) -> &Image<f64> {
        &self.z
    }

    pub fn into_planes(self) -> [Image<f64>; 3] {
        [self.x, self.y, self.z]
    }

    /// The vector at pixel (slow, fast); panics outside the shape.
    pub fn get(&self, slow: usize, fast: usize) -> Vector3<f64> {
        Vector3::new(
            self.x.get(slow, fast),
            self.y.get(slow, fast),
            self.z.get(slow, fast),
        )
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
/// where the pixel is and its [`detector_position`] at `poses`; as the
/// modules tile the image, it overwrites `blank` at every pixel.
fn fill<T: Copy + Send, const N: usize>(
    poses: &Poses,
    detector: &PixelDetector,
    blank: T,
    f: impl Fn(At, &Vector3<f64>) -> Result<[T; N], Error> + Sync,
) -> Result<[Image<T>; N], Error> {
    let (slow, fast) = detector.shape();
    let mut planes: [Image<T>; N] =
        std::array::from_fn(|_| Image::from_fn((slow, fast), |_, _| blank));
    let mut chunks: Vec<_> = planes
        .iter_mut()
        .map(|p| p.data_mut().chunks_mut(fast))
        .collect();
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
                    let p = detector_position(poses, &m.pixel(i - a1, l));
                    for (plane, v) in row.iter_mut().zip(f(at, &p)?) {
                        plane[a2 + l] = v;
                    }
                }
            }
            Ok(())
        })?;
    Ok(planes)
}

fn planes3([x, y, z]: [Image<f64>; 3]) -> Planes {
    Planes { x, y, z }
}

/// Q in the sample frame for every pixel at axis `values` (axis order,
/// radians / metres).
pub fn q_map(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    detector: &PixelDetector,
) -> Result<QMap, Error> {
    let poses = geometry.poses(values)?;
    let planes = fill(&poses, detector, 0.0, |_, p| {
        Ok(scattering_at(geometry, &poses, wavelength, p)?
            .q_sample
            .into())
    })?;
    Ok(planes3(planes))
}

/// Pixel positions relative to the sample, lab frame, metres: the
/// [`detector_position`] of every pixel. A pixel at the sample is the zero
/// vector here; only what needs its k̂f fails on it.
pub fn detector_positions(
    geometry: &Geometry,
    values: &[f64],
    detector: &PixelDetector,
) -> Result<Planes, Error> {
    let poses = geometry.poses(values)?;
    let planes = fill(&poses, detector, 0.0, |_, p| Ok((*p).into()))?;
    Ok(planes3(planes))
}

/// One weight per pixel: a positive finite [`Weight`], or the reason the
/// pixel has none, so a masked pixel holds no number to read.
pub type Weights = Image<Result<Weight, Masked>>;

impl Weights {
    /// Weights given as numbers: where `reasons`, of the shape of `values`,
    /// has a [`Masked`] the pixel has that reason, elsewhere its value,
    /// which must be positive and finite or the pixel is [`Masked::Opaque`].
    pub fn from_values(
        values: &Image<f64>,
        reasons: &Image<Option<Masked>>,
    ) -> Result<Self, Error> {
        reasons.check("weight reasons", values.shape())?;
        Ok(Image::from_fn(values.shape(), |s, f| {
            match reasons.get(s, f) {
                Some(m) => Err(m),
                None => Weight::new(values.get(s, f), Masked::Opaque),
            }
        }))
    }
}

/// Every pixel overwrites it.
const BLANK: Result<Weight, Masked> = Err(Masked::Opaque);

/// Solid angle of each pixel, sr: `|A · k̂f| / |p|²`, with `A` the pixel's
/// area vector (its widths along the module axes times `slow × fast`)
/// rotated by the detector pose. A detector with a module whose pixels span
/// no area (a point axis, as a point or an xu linear detector has) gets
/// [`Error::NoPixelArea`]; a pixel seen edge-on is [`Masked::EdgeOn`].
pub fn solid_angle(
    geometry: &Geometry,
    values: &[f64],
    detector: &PixelDetector,
) -> Result<Weights, Error> {
    let modules = detector.modules();
    if modules.iter().any(|m| m.normal().is_none()) {
        return Err(Error::NoPixelArea);
    }
    let poses = geometry.poses(values)?;
    let [weights] = fill(&poses, detector, BLANK, |at, p| {
        let kf = scattered_direction(p)?;
        let (ls, lf) = at.local;
        let area = modules[at.module].area(ls, lf);
        let omega = (poses.detector.rotation * area).dot(&kf).abs() / p.norm_squared();
        Ok([Weight::new(omega, Masked::EdgeOn)])
    })?;
    Ok(weights)
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
    values: &[f64],
    detector: &PixelDetector,
    polarisation: Polarisation,
) -> Result<Weights, Error> {
    let f = polarisation.factor;
    if !(-1.0..=1.0).contains(&f) {
        return Err(Error::PolarisationFactor { value: f });
    }
    // k̂i is the same for every pixel of the frame
    let poses = geometry.poses(values)?;
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
    let [weights] = fill(&poses, detector, BLANK, |_, p| {
        let kf = scattered_direction(p)?.into_inner();
        let p = 0.5 * (1.0 + f) * (1.0 - e.dot(&kf).powi(2))
            + 0.5 * (1.0 - f) * (1.0 - across.dot(&kf).powi(2));
        Ok([Weight::new(p, Masked::Extinct)])
    })?;
    Ok(weights)
}

/// Absorption weight per pixel (design §5.1): the sample slab's times the
/// detector path's, at the pixel's k̂f and p. Every μ is taken at the energy
/// of `wavelength` and F computed once, before the pixel loop, which sees only the
/// resulting [`SlabFrame`] and each module's
/// [`crate::absorption::OrientedPath`]: the detector layers lie along the
/// module's normal, and a window or sensor on a module with none is
/// [`Error::NoPixelArea`]. Where both sides mask, the pixel carries the
/// sample's reason.
pub fn absorption<A: Attenuation>(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    detector: &PixelDetector,
    absorber: &Absorber<A>,
) -> Result<Weights, Error> {
    let poses = geometry.poses(values)?;
    let energy = wavelength.energy();
    let sample = SlabFrame::new(
        &absorber.slab,
        &absorber.beam,
        absorber.sample.mu(energy)?,
        &(poses.source * geometry.beam_direction()),
        &poses.sample.rotation,
    )?;
    let path = absorber.detector.at(energy)?;
    let paths = detector
        .modules()
        .iter()
        .map(|m| path.oriented(m.normal().map(|n| poses.detector.rotation * n)))
        .collect::<Result<Vec<_>, _>>()?;
    let [weights] = fill(&poses, detector, BLANK, |at, p| {
        let kf = scattered_direction(p)?;
        Ok([sample
            .weight(&kf)
            .and_then(|a| a.times(paths[at.module].weight(&kf, p.norm())?))])
    })?;
    Ok(weights)
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

    const VALUES: [f64; 4] = [0.31, 0.002, 0.57, -0.23];

    fn wavelength() -> Wavelength {
        Wavelength::new(1.54).unwrap()
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

    /// Reasons of the transposed shape have the values' pixel count.
    #[test]
    fn from_values_refuses_reasons_of_another_shape() {
        let values = Image::new((1, 3), vec![1.0, 2.0, 0.0]).unwrap();
        let reasons = Image::new((3, 1), vec![None; 3]).unwrap();
        assert_eq!(
            Weights::from_values(&values, &reasons),
            Err(Error::ShapeMismatch {
                what: "weight reasons",
                shape: (3, 1),
                expected: (1, 3)
            })
        );
        let reasons = Image::new((1, 3), vec![None, Some(Masked::EdgeOn), None]).unwrap();
        let w = Weights::from_values(&values, &reasons).unwrap();
        assert_eq!(w.shape(), (1, 3));
        assert_eq!(w.get(0, 0), Weight::new(1.0, Masked::Opaque));
        assert_eq!(w.get(0, 1), Err(Masked::EdgeOn));
        assert_eq!(w.get(0, 2), Err(Masked::Opaque));
    }

    #[test]
    fn batch_equals_per_pixel_forward_exactly() {
        let (g, d, wl) = (geometry(), detector(), wavelength());
        let map = q_map(&g, &VALUES, wl, &d).unwrap();
        assert_eq!(map.shape(), (5, 7));
        for i in 0..5 {
            for j in 0..7 {
                let one = forward(&g, &sample(), &VALUES, wl, &d.pixel(i, j).unwrap()).unwrap();
                let got = map.get(i, j);
                assert!(got.iter().all(|v| v.is_finite()));
                assert_eq!(got, one.scattering.q_sample, "pixel ({i}, {j})");
            }
        }
    }

    #[test]
    fn position_direction_is_kf_and_p_is_the_kernels() {
        let (g, d) = (geometry(), detector());
        let pos = detector_positions(&g, &VALUES, &d).unwrap();
        let poses = g.poses(&VALUES).unwrap();
        for i in 0..5 {
            for j in 0..7 {
                let pixel = d.pixel(i, j).unwrap();
                let one = forward(&g, &sample(), &VALUES, wavelength(), &pixel).unwrap();
                let p = pos.get(i, j);
                let want = poses.detector * pixel - poses.sample_position();
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
        polarisation(g, &[0.0], d, pol)
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
        let pol = Polarisation {
            e: Vector3::y(),
            factor: f64::NAN,
        };
        assert!(matches!(
            polarisation(&g, &[0.0], &point_at([0.0, 1.0, 0.0]), pol),
            Err(Error::PolarisationFactor { .. })
        ));
    }

    #[test]
    fn polarisation_vector_is_projected_normal_to_the_beam() {
        let g = om_only();
        let d = point_at([0.3, 0.8, 0.5]);
        let pol = |e| Polarisation { e, factor: 0.7 };
        let w = |e| polarisation(&g, &[0.0], &d, pol(e)).map(|w| w.get(0, 0).unwrap().get());
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
        let (d, a, b) = (0.4_f64, 1e-3, 2e-3);
        let w = solid_angle(&g, &[0.0], &flat(d, a, b)).unwrap();
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
        let (d, a, b) = (0.4_f64, 1e-3, 2e-3);
        let w = solid_angle(&g, &[0.0, 0.9], &flat(d, a, b)).unwrap();
        let normal = a * b / (d * d);
        assert!((w.get(0, 0).unwrap().get() - normal).abs() < 1e-12 * normal);
    }

    #[test]
    fn solid_angle_of_a_linear_detector_uses_its_row_height() {
        let g = om_only();
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
        let w = solid_angle(&g, &[0.0], &row(Vector3::new(0.0, 0.0, h))).unwrap();
        assert_eq!(w.shape(), (1, 4));
        let normal = h * b / (d * d);
        assert!((w.get(0, 0).unwrap().get() - normal).abs() < 1e-12 * normal);
        let r2 = d * d + (3.0 * b).powi(2);
        let tilted = h * b * d / r2.powf(1.5);
        assert!((w.get(0, 3).unwrap().get() - tilted).abs() < 1e-12 * tilted);
        // built with a zero e_slow it has no height to give
        assert_eq!(
            solid_angle(&g, &[0.0], &row(Vector3::zeros())),
            Err(Error::NoPixelArea)
        );
    }

    #[test]
    fn solid_angle_needs_a_pixel_area() {
        let g = om_only();
        assert_eq!(
            solid_angle(&g, &[0.0], &point_at([1.0, 0.0, 0.0])),
            Err(Error::NoPixelArea)
        );
    }

    /// A detector plane through the sample is seen edge-on at every pixel.
    #[test]
    fn solid_angle_masks_a_pixel_seen_edge_on() {
        let g = om_only();
        let d = PixelDetector::new(
            (2, 3),
            Vector3::new(0.0, 0.3, 0.0),
            Vector3::new(0.0, 0.0, 1e-3),
            Vector3::new(0.0, 2e-3, 0.0),
        )
        .unwrap();
        let w = solid_angle(&g, &[0.0], &d).unwrap();
        assert!(w.data().iter().all(|w| *w == Err(Masked::EdgeOn)), "{w:?}");
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

    /// Bit for bit SlabFrame's times DetectorPath's at the kernels' k̂f and
    /// p, in reflection and in transmission.
    #[test]
    fn absorption_equals_slab_frame_and_detector_path_at_the_kernels_kf_and_p() {
        let (g, d) = (geometry(), detector());
        let poses = g.poses(&VALUES).unwrap();
        let ki = poses.source * g.beam_direction();
        let normal = poses.detector.rotation * d.modules()[0].normal().unwrap();
        for (n, reflection) in [([0.0, 1.0, 0.0], true), ([-1.0, 0.0, 0.0], false)] {
            let a = absorber(Slab::new(face(n), 1e-4).unwrap());
            let w = absorption(&g, &VALUES, wavelength(), &d, &a).unwrap();
            let path = a.detector.oriented(Some(normal)).unwrap();
            let sample =
                SlabFrame::new(&a.slab, &a.beam, a.sample, &ki, &poses.sample.rotation).unwrap();
            let n_lab = poses.sample.rotation * Vector3::from(n);
            for i in 0..5 {
                for j in 0..7 {
                    let p = detector_position(&poses, &d.pixel(i, j).unwrap());
                    let kf = scattered_direction(&p).unwrap();
                    assert_eq!(kf.dot(&n_lab) > 0.0, reflection, "{n:?} ({i}, {j})");
                    let want = sample.weight(&kf).unwrap();
                    let want = want.times(path.weight(&kf, p.norm()).unwrap());
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
        let (g, d) = (geometry(), detector());
        let every = |a: &Absorber<PerMetre>, want: Masked| {
            let w = absorption(&g, &VALUES, wavelength(), &d, a).unwrap();
            assert!(w.data().iter().all(|r| *r == Err(want)), "{want:?}: {w:?}");
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
        let g = geometry();
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
        let poses = g.poses(&VALUES).unwrap();
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
        let w = absorption(&g, &VALUES, wavelength(), &d, &a).unwrap();
        for i in 0..4 {
            for j in 0..3 {
                let p = detector_position(&poses, &d.pixel(i, j).unwrap());
                let kf = scattered_direction(&p).unwrap();
                let path = paths[i / 2];
                let want = sample
                    .weight(&kf)
                    .unwrap()
                    .times(path.weight(&kf, p.norm()).unwrap());
                assert_eq!(w.get(i, j), want, "({i}, {j})");
                assert!(want.is_ok(), "({i}, {j})");
            }
        }
    }

    /// Air alone needs no normal, so it weighs a point detector and an xu
    /// row; a window or sensor there is an error.
    #[test]
    fn absorption_layers_need_a_module_normal() {
        let g = geometry();
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
            let w = absorption(&g, &VALUES, wavelength(), &d, &air).unwrap();
            assert!(w.data().iter().all(Result::is_ok), "{w:?}");
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
                assert_eq!(
                    absorption(&g, &VALUES, wavelength(), &d, &a),
                    Err(Error::NoPixelArea)
                );
            }
        }
    }

    /// Records the energy of every call. The RefCell makes it !Sync, so the
    /// pixel loop, which `fill` requires to be Sync, cannot reach it.
    struct Recording(RefCell<Vec<f64>>);

    impl Attenuation for Recording {
        fn mu(&self, energy: Energy) -> Result<PerMetre, Error> {
            self.0.borrow_mut().push(energy.kev());
            Ok(per_metre(1e3))
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
            let wl = Wavelength::new(angstrom).unwrap();
            let w = absorption(&g, &VALUES, wl, &d, &a).unwrap();
            assert!(w.data().iter().all(Result::is_ok), "{w:?}");
            want.extend([wl.energy().kev(); 4]);
        }
        assert_eq!(*r.0.borrow(), want);
    }

    /// Pixel (0, 0) sits at the sample: it has a position, the zero
    /// vector, but no k̂f, so every map that needs one fails.
    #[test]
    fn a_pixel_on_the_sample_has_a_position_but_no_kf() {
        let g = om_only();
        let d = PixelDetector::new(
            (2, 2),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let pos = detector_positions(&g, &[0.0], &d).unwrap();
        assert_eq!(pos.get(0, 0), Vector3::zeros());
        assert_eq!(pos.get(1, 1), Vector3::new(0.0, 1.0, 1.0));
        let at_sample = Some(Error::DetectorAtSample);
        let wl = Wavelength::new(1.0).unwrap();
        assert_eq!(q_map(&g, &[0.0], wl, &d).err(), at_sample);
        assert_eq!(solid_angle(&g, &[0.0], &d).err(), at_sample);
        let pol = Polarisation {
            e: Vector3::y(),
            factor: 0.0,
        };
        assert_eq!(polarisation(&g, &[0.0], &d, pol).err(), at_sample);
        let a = absorber(Slab::new(face([-1.0, 0.0, 0.0]), 1e-4).unwrap());
        assert_eq!(absorption(&g, &[0.0], wl, &d, &a).err(), at_sample);
    }
}
