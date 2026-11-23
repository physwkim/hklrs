//! Calibration of synthetic stacks: parameters recovered, σ with its stated
//! coverage, misses typed, directions searched, and data generated in xu's
//! (reversed) circle order not fitting as truth (XU-1).

use hklrs::calib::{
    AreaModel, AreaStart, CalFrame, CalibError, FrameKind, Param, ParamId, Peak, Reject,
    calibrate_area,
};
use hklrs::xu::{Area, Direction};
use hklrs::{Deg, Geometry};

fn dir(text: &str) -> Direction {
    text.parse().unwrap()
}

/// Detector circles, outermost first, the beam along x.
fn geometry(circles: &[(&str, [f64; 3])]) -> Geometry {
    let mut b = Geometry::builder("synth");
    for (name, axis) in circles {
        b = b.detector_rotation(name, *axis);
    }
    b.beam_direction([1.0, 0.0, 0.0]).build().unwrap()
}

const Z: [f64; 3] = [0.0, 0.0, 1.0];
const MINUS_Y: [f64; 3] = [0.0, -1.0, 0.0];

fn truth() -> Area {
    let mut a = Area::new(
        dir("z+"),
        dir("y+"),
        [400.3, 395.7],
        [800, 800],
        1.0,
        [2.0e-3, 2.1e-3],
    );
    a.detrot = Deg(1.5);
    a
}

/// Where the direct beam lands for `angles` on `area`.
fn beam_pixel(geometry: &Geometry, area: &Area, angles: &[f64]) -> [f64; 2] {
    let poses = geometry
        .poses(&geometry.values_from_degrees(angles).unwrap())
        .unwrap();
    let ki = poses.source * geometry.beam_direction();
    let to_arm = poses.detector.inverse();
    area.detector(geometry.beam_direction())
        .unwrap()
        .locate(
            to_arm * poses.sample_position(),
            to_arm.rotation * ki.into_inner(),
        )
        .unwrap()
}

/// xorshift64* and Box–Muller: enough for a coverage count, no dependency.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// `n` circles, angles drawn within ±`span` degrees.
fn angle_sets(n: usize, count: usize, span: f64, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..count)
        .map(|_| (0..n).map(|_| (2.0 * rng.unit() - 1.0) * span).collect())
        .collect()
}

fn frames(
    g: &Geometry,
    area: &Area,
    angles: &[Vec<f64>],
    sigma: f64,
    rng: Option<&mut Rng>,
) -> Vec<CalFrame> {
    let mut rng = rng;
    angles
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut pos = beam_pixel(g, area, a);
            if let Some(r) = rng.as_deref_mut() {
                pos[0] += sigma * r.normal();
                pos[1] += sigma * r.normal();
            }
            CalFrame::new(
                i,
                a.clone(),
                Peak {
                    pos,
                    sigma: [sigma; 2],
                },
                FrameKind::PrimaryBeam,
            )
            .unwrap()
        })
        .collect()
}

fn model(g: Geometry, directions: Option<[Direction; 2]>) -> AreaModel {
    AreaModel {
        geometry: g,
        sample: None,
        offset_axis: None,
        nch: [800, 800],
        directions,
    }
}

fn start() -> AreaStart {
    AreaStart {
        cch1: Param::free(393.0),
        cch2: Param::free(402.0),
        pwidth1: Param::free(2.06e-3),
        pwidth2: Param::free(2.15e-3),
        distance: Param::fixed(1.0),
        tiltazimuth: Param::fixed(0.0),
        tilt: Param::fixed(0.0),
        detrot: Param::free(0.0),
        angle_offset: Param::fixed(0.0),
        wavelength: Param::fixed(1.54),
    }
}

fn circles(n: usize) -> Vec<(&'static str, [f64; 3])> {
    [("a1", Z), ("a2", MINUS_Y), ("a3", Z)][..n].to_vec()
}

fn value(a: &Area, id: ParamId) -> f64 {
    match id {
        ParamId::Cch1 => a.cch1,
        ParamId::Cch2 => a.cch2,
        ParamId::Pwidth1 => a.pwidth1,
        ParamId::Pwidth2 => a.pwidth2,
        ParamId::Detrot => a.detrot.0,
        _ => unreachable!(),
    }
}

const FREE: [ParamId; 5] = [
    ParamId::Cch1,
    ParamId::Cch2,
    ParamId::Pwidth1,
    ParamId::Pwidth2,
    ParamId::Detrot,
];

#[test]
fn exact_frames_round_trip_on_one_to_three_circles() {
    for n in 1..=3 {
        let g = geometry(&circles(n));
        let mut rng = Rng(7 + n as u64);
        let angles = angle_sets(n, 24, 9.0, &mut rng);
        let fs = frames(&g, &truth(), &angles, 0.02, None);
        let fit = calibrate_area(&model(g, Some([dir("z+"), dir("y+")])), &fs, &[start()]);
        // one circle moves the beam along a line: the 2-D parameters cannot all be told apart
        if n == 1 {
            assert!(
                matches!(fit, Err(CalibError::Unidentifiable { .. })),
                "{fit:?}"
            );
            continue;
        }
        let fit = fit.unwrap();
        for id in FREE {
            let want = value(&truth(), id);
            let got = value(fit.value(), id);
            assert!(
                (got - want).abs() <= want.abs() * 1e-7 + 1e-9,
                "{n} circles {id:?}: {got} vs {want}"
            );
        }
        assert!(fit.rms() < 1e-5, "rms {}", fit.rms());
    }
}

#[test]
fn sigma_covers_the_truth_over_200_noise_seeds() {
    let g = geometry(&circles(2));
    let angles = angle_sets(2, 30, 9.0, &mut Rng(99));
    let (mut inside, mut total) = ([0usize; 5], 0usize);
    let mut rms = 0.0;
    for seed in 1..=200u64 {
        let mut rng = Rng(0x9E37_79B9 ^ seed.wrapping_mul(0x1000_0000_01B3));
        let fs = frames(&g, &truth(), &angles, 0.05, Some(&mut rng));
        let fit = calibrate_area(
            &model(g.clone(), Some([dir("z+"), dir("y+")])),
            &fs,
            &[start()],
        )
        .unwrap();
        for (k, id) in FREE.iter().enumerate() {
            let s = fit.sigma(*id).unwrap();
            if (value(fit.value(), *id) - value(&truth(), *id)).abs() <= s {
                inside[k] += 1;
            }
        }
        rms += fit.rms();
        total += 1;
    }
    for (k, id) in FREE.iter().enumerate() {
        let frac = inside[k] as f64 / total as f64;
        // 68.3 % ± 3 binomial standard deviations of 200 runs
        assert!((0.58..=0.79).contains(&frac), "{id:?}: {frac}");
    }
    let mean_rms = rms / total as f64;
    assert!((0.9..=1.05).contains(&mean_rms), "mean rms {mean_rms}");
}

#[test]
fn data_in_xus_circle_order_does_not_fit_as_truth() {
    // xu's calibration applies the detector circles in the reverse order of
    // its Ang2Q (XU-1). The same circles listed the other way round generate
    // data the forward model of this crate must not mistake for truth.
    let right = geometry(&circles(2));
    let reversed = geometry(&[("a2", MINUS_Y), ("a1", Z)]);
    let angles = angle_sets(2, 25, 12.0, &mut Rng(5));
    // angles stay keyed by axis name: the reversed geometry takes (a2, a1)
    let swapped: Vec<Vec<f64>> = angles.iter().map(|a| vec![a[1], a[0]]).collect();
    let m = model(right.clone(), Some([dir("z+"), dir("y+")]));

    let own = frames(&right, &truth(), &angles, 0.01, None);
    let fit = calibrate_area(&m, &own, &[start()]).unwrap();
    assert!(fit.rms() < 1e-4);

    let other = frames(&reversed, &truth(), &swapped, 0.01, None);
    // the same angle values mean a different beam position
    assert_ne!(
        beam_pixel(&right, &truth(), &angles[0]),
        beam_pixel(&reversed, &truth(), &swapped[0])
    );
    // at the true pixel widths, detrot and distance the data leave a large
    // misfit however the chip centre is placed
    let mut at_truth = start();
    at_truth.pwidth1 = Param::fixed(2.0e-3);
    at_truth.pwidth2 = Param::fixed(2.1e-3);
    at_truth.detrot = Param::fixed(1.5);
    let fit = calibrate_area(&m, &other, &[at_truth]).unwrap();
    assert!(fit.rms() > 100.0, "rms {}", fit.rms());
    // freed, the fit either fails with a typed error or lands off the truth
    match calibrate_area(&m, &other, &[start()]) {
        Ok(fit) => assert!((fit.value().pwidth1 - 2.0e-3).abs() > 1e-5),
        Err(CalibError::NotConverged { .. } | CalibError::Unidentifiable { .. }) => {}
        Err(e) => panic!("{e:?}"),
    }
}

#[test]
fn only_a_ray_behind_or_parallel_is_rejected_off_the_chip_keeps_its_residual() {
    let g = geometry(&circles(2));
    let mut angles = angle_sets(2, 20, 9.0, &mut Rng(11));
    angles.push(vec![0.0, 95.0]); // the detector turned away from the beam
    let fs = frames_unchecked(&g, &truth(), &angles);
    // a start whose chip centre is 150 pixels off sends the outer frames
    // off the chip at the start; they keep their residuals and pull the fit in
    let mut s = start();
    s.cch1 = Param::free(truth().cch1 + 150.0);
    s.cch2 = Param::free(truth().cch2 - 150.0);
    let fit = calibrate_area(&model(g, Some([dir("z+"), dir("y+")])), &fs, &[s]).unwrap();
    assert_eq!(fit.rejected().len(), 1);
    assert_eq!(fit.rejected()[0].0, 20);
    assert!(matches!(
        fit.rejected()[0].1,
        Reject::RayBehind | Reject::RayParallel
    ));
    assert_eq!(fit.dof(), 2 * 20 - 5);
    assert!(fit.rms() < 1e-3, "rms {}", fit.rms());
    assert!((fit.value().cch1 - truth().cch1).abs() < 1e-3);
}

/// Like [`frames`] but a frame whose beam misses the chip still gets a peak
/// (pixel 0, 0): what a caller would hold for it is unknowable.
fn frames_unchecked(g: &Geometry, area: &Area, angles: &[Vec<f64>]) -> Vec<CalFrame> {
    angles
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let poses = g.poses(&g.values_from_degrees(a).unwrap()).unwrap();
            let ki = poses.source * g.beam_direction();
            let to_arm = poses.detector.inverse();
            let pos = area
                .detector(g.beam_direction())
                .unwrap()
                .locate(
                    to_arm * poses.sample_position(),
                    to_arm.rotation * ki.into_inner(),
                )
                .unwrap_or([0.0, 0.0]);
            CalFrame::new(
                i,
                a.clone(),
                Peak {
                    pos,
                    sigma: [0.02; 2],
                },
                FrameKind::PrimaryBeam,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn directions_are_searched_among_the_beam_perpendicular_candidates() {
    let g = geometry(&circles(2));
    let angles = angle_sets(2, 24, 9.0, &mut Rng(21));
    let fs = frames(&g, &truth(), &angles, 0.02, None);
    // detrot held: the search picks the true directions
    let mut s = start();
    s.detrot = Param::fixed(1.5);
    let fit = calibrate_area(&model(g.clone(), None), &fs, &[s]).unwrap();
    assert_eq!(fit.value().dir1, dir("z+"));
    assert_eq!(fit.value().dir2, dir("y+"));

    // detrot free: a quarter turn about the beam with the directions swapped
    // puts every pixel in the same place, so it is one detector, returned
    // with detrot in (-45, 45]
    let fit = calibrate_area(&model(g, None), &fs, &[start()]).unwrap();
    let a = fit.value();
    assert_eq!((a.dir1, a.dir2), (dir("z+"), dir("y+")));
    assert!((a.detrot.0 - 1.5).abs() < 1e-6, "{:?}", a.detrot);
}

#[test]
fn frames_along_one_pixel_row_leave_the_slow_direction_ambiguous() {
    // all frames at one slow pixel: z+ and z- fit identically with cch1
    // mirrored, so no direction can be chosen
    let g = geometry(&circles(2));
    let angles: Vec<Vec<f64>> = (0..12).map(|i| vec![-6.0 + i as f64, 0.0]).collect();
    let mut flat = truth();
    flat.detrot = Deg(0.0);
    let fs = frames(&g, &flat, &angles, 0.02, None);
    let mut s = start();
    s.pwidth1 = Param::fixed(2.0e-3);
    s.detrot = Param::fixed(0.0);
    match calibrate_area(&model(g, None), &fs, &[s]) {
        Err(CalibError::AmbiguousDirections {
            best, runner_up, ..
        }) => {
            assert_eq!(best[1], dir("y+"));
            assert_eq!(runner_up[1], dir("y+"));
            assert_ne!(best[0], runner_up[0]);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_far_start_converges_to_a_local_minimum_that_only_the_rms_exposes() {
    let g = geometry(&circles(2));
    let angles = angle_sets(2, 24, 9.0, &mut Rng(31));
    let mut fs = frames(&g, &truth(), &angles, 0.02, None);
    // a frame whose peak sits somewhere else entirely, so no parameters fit it
    let bad = CalFrame::new(
        0,
        angles[0].clone(),
        Peak {
            pos: [10.0, 10.0],
            sigma: [1e-3; 2],
        },
        FrameKind::PrimaryBeam,
    )
    .unwrap();
    fs[0] = bad;
    let mut s = start();
    s.cch1 = Param::free(700.0);
    // the solver is deterministic: from cch1 = 700 it converges (a true
    // stopping rule, not the iteration cap) to a minimum at cch1 ≈ 82
    // (truth 400.3) and returns Ok; only the rms says it is wrong. The
    // outlier is not the cause: it holds under 5 % of the χ².
    let m = model(g.clone(), Some([dir("z+"), dir("y+")]));
    let fit = calibrate_area(&m, &fs, &[s]).unwrap();
    assert!(fit.rejected().is_empty());
    assert!((fit.rms() - 16138.0).abs() < 1.0, "rms {}", fit.rms());
    assert!(
        (fit.value().cch1 - 81.7).abs() < 0.1,
        "{}",
        fit.value().cch1
    );
    assert_eq!(fit.stop(), hklrs::calib::Stop::Cost);
    let detector = fit.value().detector(g.beam_direction()).unwrap();
    let chi2: Vec<f64> = fs
        .iter()
        .map(|f| {
            let poses = g
                .poses(&g.values_from_degrees(f.angles()).unwrap())
                .unwrap();
            let ki = poses.source * g.beam_direction();
            let to_arm = poses.detector.inverse();
            let p = f.peak();
            let module = detector
                .module(p.pos[0].round() as usize, p.pos[1].round() as usize)
                .unwrap();
            let at = detector
                .locate_on(
                    module,
                    to_arm * poses.sample_position(),
                    to_arm.rotation * ki.into_inner(),
                )
                .unwrap();
            (0..2)
                .map(|a| ((at[a] - p.pos[a]) / p.sigma[a]).powi(2))
                .sum()
        })
        .collect();
    let total: f64 = chi2.iter().sum();
    let m_res = 2.0 * fs.len() as f64;
    assert!(
        (total - m_res * fit.rms().powi(2)).abs() < 1e-6 * total,
        "χ² {total}"
    );
    assert!(chi2[0] < 0.05 * total, "bad frame {} of {total}", chi2[0]);
}

mod reflections {
    use std::f64::consts::TAU;

    use hklrs::nalgebra::{UnitQuaternion, Vector3};
    use hklrs::{Lattice, Orientation, Sample};

    use super::*;

    const LAMBDA: f64 = 1.54;

    fn sample() -> Sample {
        Sample::new(
            "cubic",
            Lattice::from_degrees(4.0, 4.0, 4.0, 90.0, 90.0, 90.0).unwrap(),
            Orientation::Manual {
                u: UnitQuaternion::identity(),
            },
        )
    }

    /// Sample circle `om` about z, then the xu detector circles.
    fn geometry() -> Geometry {
        Geometry::builder("refl")
            .sample_rotation("om", Z)
            .detector_rotation("a1", Z)
            .detector_rotation("a2", MINUS_Y)
            .beam_direction([1.0, 0.0, 0.0])
            .build()
            .unwrap()
    }

    /// Pixel of the ray k·k̂ᵢ + R_s UB hkl from the sample.
    fn pixel(g: &Geometry, area: &Area, angles: &[f64], hkl: Vector3<f64>) -> [f64; 2] {
        let poses = g.poses(&g.values_from_degrees(angles).unwrap()).unwrap();
        let ki = poses.source * g.beam_direction();
        let q = poses.sample.rotation * (sample().ub() * hkl);
        let to_arm = poses.detector.inverse();
        area.detector(g.beam_direction())
            .unwrap()
            .locate(
                to_arm * poses.sample_position(),
                to_arm.rotation * ((TAU / LAMBDA) * ki.into_inner() + q),
            )
            .unwrap()
    }

    /// `om` of the Bragg condition for an in-plane `hkl`, and the arm angle
    /// that points at the reflection.
    fn bragg(hkl: [f64; 2]) -> (f64, f64) {
        let q = TAU / 4.0 * (hkl[0].powi(2) + hkl[1].powi(2)).sqrt();
        let k = TAU / LAMBDA;
        let phi = hkl[1].atan2(hkl[0]);
        let om = (-q / (2.0 * k)).acos() - phi;
        let (qx, qy) = (q * (phi + om).cos(), q * (phi + om).sin());
        let a1 = qy.atan2(k + qx);
        (om.to_degrees(), a1.to_degrees())
    }

    fn frames(g: &Geometry, area: &Area, kind: fn(Vector3<f64>) -> FrameKind) -> Vec<CalFrame> {
        let mut out = Vec::new();
        let mut rng = Rng(3);
        for _ in 0..16 {
            let a = [
                0.0,
                (2.0 * rng.unit() - 1.0) * 9.0,
                (2.0 * rng.unit() - 1.0) * 9.0,
            ];
            out.push(CalFrame::new(
                out.len(),
                a.to_vec(),
                Peak {
                    pos: pixel(g, area, &a, Vector3::zeros()),
                    sigma: [0.02; 2],
                },
                kind(Vector3::zeros()),
            ));
        }
        for hkl in [
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
        ] {
            let (om, a1) = bragg(hkl);
            let a = [om, a1, 0.5];
            let h = Vector3::new(hkl[0], hkl[1], 0.0);
            out.push(CalFrame::new(
                out.len(),
                a.to_vec(),
                Peak {
                    pos: pixel(g, area, &a, h),
                    sigma: [0.02; 2],
                },
                FrameKind::Reflection(h),
            ));
        }
        out.into_iter().map(Result::unwrap).collect()
    }

    fn model() -> AreaModel {
        AreaModel {
            sample: Some(sample()),
            ..super::model(geometry(), Some([dir("z+"), dir("y+")]))
        }
    }

    #[test]
    fn wavelength_is_fitted_from_reflections_and_pinned_down_by_them() {
        let g = geometry();
        let fs = frames(&g, &truth(), |_| FrameKind::PrimaryBeam);
        let mut s = start();
        s.wavelength = Param::free(1.50);
        let fit = calibrate_area(&model(), &fs, &[s]).unwrap();
        let wl = fit
            .free()
            .iter()
            .find(|f| f.id == ParamId::Wavelength)
            .unwrap();
        assert!((wl.value - LAMBDA).abs() < 1e-7 * LAMBDA, "{}", wl.value);
        assert!(wl.sigma > 0.0 && wl.sigma < 1e-3);
        assert!((fit.value().cch1 - 400.3).abs() < 1e-5);
        assert!(fit.rejected().is_empty());
    }

    #[test]
    fn reflection_frames_need_a_sample() {
        let g = geometry();
        let fs = frames(&g, &truth(), |_| FrameKind::PrimaryBeam);
        let m = AreaModel {
            sample: None,
            ..model()
        };
        assert!(matches!(
            calibrate_area(&m, &fs, &[start()]),
            Err(CalibError::BadFrame { index: 16, .. })
        ));
    }

    #[test]
    fn a_frame_is_a_reflection_by_its_kind_not_by_a_nonzero_hkl() {
        // XU-8: a (0 0 0) reflection scatters nowhere, so it lies on the
        // direct beam like a primary-beam frame, and a frame that is a
        // reflection stays one with any hkl
        let g = geometry();
        let as_primary = frames(&g, &truth(), |_| FrameKind::PrimaryBeam);
        let as_zero = frames(&g, &truth(), FrameKind::Reflection);
        let m = model();
        let a = calibrate_area(&m, &as_primary, &[start()]).unwrap();
        let b = calibrate_area(&m, &as_zero, &[start()]).unwrap();
        let (a, b) = (a.value(), b.value());
        assert!((a.cch1 - b.cch1).abs() < 1e-9 && (a.cch2 - b.cch2).abs() < 1e-9);
        assert!((a.pwidth1 - b.pwidth1).abs() < 1e-12 && (a.detrot.0 - b.detrot.0).abs() < 1e-9);
    }
}
