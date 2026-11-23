//! hkl calibration against the xrayutilities golden (area_detector_calib_hkl):
//! hklrs fits the wavelength and sample tilt it is given the means to, keeps
//! reflections whose indices contain zeros (XU-8), and has no start tuple to
//! crash on (XU-7).

use std::path::PathBuf;

use hklrs::calib::{
    AreaHklStart, AreaModel, AreaStart, CalFrame, CalibError, FrameKind, Param, ParamId, Peak,
    calibrate_area_hkl,
};
use hklrs::nalgebra::{UnitQuaternion, Vector3};
use hklrs::xu::Direction;
use hklrs::{Geometry, Lattice, Orientation, Sample};
use serde_json::Value;

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../golden/xu_calib_hkl.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn dir(text: &str) -> Direction {
    text.parse().unwrap()
}

fn model(doc: &Value) -> AreaModel {
    let a = f(&doc["lattice"]);
    AreaModel {
        geometry: Geometry::builder("xu-hkl")
            .sample_rotation("om", [0.0, -1.0, 0.0])
            .detector_rotation("a1", [0.0, 0.0, 1.0])
            .detector_rotation("a2", [0.0, -1.0, 0.0])
            .beam_direction([1.0, 0.0, 0.0])
            .build()
            .unwrap(),
        sample: Some(Sample::new(
            "cubic",
            Lattice::from_degrees(a, a, a, 90.0, 90.0, 90.0).unwrap(),
            Orientation::Manual {
                u: UnitQuaternion::identity(),
            },
        )),
        offset_axis: Some("a1".into()),
        nch: [800, 800],
        directions: Some([dir("z+"), dir("y+")]),
    }
}

fn frames(doc: &Value) -> Vec<CalFrame> {
    doc["frames"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, fr)| {
            let h = Vector3::new(f(&fr["hkl"][0]), f(&fr["hkl"][1]), f(&fr["hkl"][2]));
            // by what the frame is, not by whether its indices are nonzero
            let kind = if h == Vector3::zeros() {
                FrameKind::PrimaryBeam
            } else {
                FrameKind::Reflection(h)
            };
            CalFrame::new(
                i,
                vec![f(&fr["sang"]), f(&fr["angle1"]), f(&fr["angle2"])],
                Peak {
                    pos: [f(&fr["xu_pixel"][0]), f(&fr["xu_pixel"][1])],
                    sigma: [f(&fr["sigma_pixel"][0]), f(&fr["sigma_pixel"][1])],
                },
                kind,
            )
            .unwrap()
        })
        .collect()
}

fn start(doc: &Value) -> AreaHklStart {
    AreaHklStart {
        area: AreaStart {
            cch1: Param::free(390.0),
            cch2: Param::free(405.0),
            pwidth1: Param::free(0.0021),
            pwidth2: Param::free(0.0019),
            distance: Param::fixed(1.0),
            tiltazimuth: Param::fixed(0.0),
            tilt: Param::fixed(0.0),
            detrot: Param::fixed(f(&doc["truth"]["detrot"])),
            angle_offset: Param::fixed(0.0),
            wavelength: Param::free(1.5),
        },
        sample_tilt: Param::free(0.2),
        sample_azimuth: Param::free(20.0),
    }
}

#[test]
fn xu_crashes_on_its_default_start_and_stalls_on_a_working_one() {
    let doc = golden();
    let runs = &doc["runs"];
    // XU-7: the default start is a tuple whose last item xu assigns to
    assert!(
        runs[0]["exception"]
            .as_str()
            .unwrap()
            .starts_with("TypeError")
    );
    assert_eq!(runs[0]["raised_at"], "sample_align.py:1692");
    // with a tuple start xu returns, with wavelength and tilt where it began
    let xu = &runs[3]["result"];
    assert_eq!(f(&xu[11]), 1.5);
    assert!((f(&xu[9]) - 0.3).abs() > 1.0);
}

#[test]
fn the_wavelength_and_sample_tilt_are_recovered_from_the_reflections() {
    let doc = golden();
    let fit = calibrate_area_hkl(&model(&doc), &frames(&doc), &[start(&doc)]).unwrap();
    let truth = &doc["truth"];
    let h = fit.value();
    for (id, got, want) in [
        (ParamId::Cch1, h.area.cch1, f(&truth["cch1"])),
        (ParamId::Cch2, h.area.cch2, f(&truth["cch2"])),
        (ParamId::Pwidth1, h.area.pwidth1, f(&truth["pwidth1"])),
        (ParamId::Pwidth2, h.area.pwidth2, f(&truth["pwidth2"])),
        (ParamId::Wavelength, h.wavelength, f(&truth["wavelength"])),
        (
            ParamId::SampleTilt,
            h.sample_tilt.0,
            f(&truth["sampletilt"]),
        ),
    ] {
        let s = fit.sigma(id).unwrap();
        assert!(
            (got - want).abs() <= 5.0 * s,
            "{id:?}: {got} vs {want} ± {s}"
        );
    }
    // xu's run on the same images ends at its start wavelength
    let xu = f(&doc["runs"][3]["result"][11]);
    assert!((h.wavelength - 1.54).abs() * 100.0 < (xu - 1.54).abs());
}

#[test]
fn reflections_with_zero_indices_are_kept_by_their_kind() {
    // XU-8: xu keeps a reflection image only when it is strong or has no zero
    // index, so the weak (0 0 L) images are dropped (9 of 21 survive)
    let doc = golden();
    assert_eq!(doc["runs"][1]["images_peak_found"], 9);
    let fs = frames(&doc);
    assert_eq!(fs.len(), 21);
    let fit = calibrate_area_hkl(&model(&doc), &fs, &[start(&doc)]).unwrap();
    assert!(fit.rejected().is_empty());
    assert!(
        fs.iter()
            .filter(|f| matches!(f.kind(), FrameKind::Reflection(_)))
            .count()
            == 12
    );
}

#[test]
fn primary_beam_and_reflection_frames_are_both_required() {
    let doc = golden();
    let all = frames(&doc);
    let only = |reflections: bool| -> Vec<CalFrame> {
        all.iter()
            .filter(|f| matches!(f.kind(), FrameKind::Reflection(_)) == reflections)
            .cloned()
            .collect()
    };
    for fs in [only(true), only(false)] {
        assert!(matches!(
            calibrate_area_hkl(&model(&doc), &fs, &[start(&doc)]),
            Err(CalibError::BadProblem)
        ));
    }
}

mod linear {
    use std::f64::consts::TAU;

    use hklrs::Deg;
    use hklrs::calib::{LinearHklStart, LinearModel, LinearStart, Peak1, calibrate_linear_hkl};
    use hklrs::xu::Linear;

    use super::*;

    const A: f64 = 10.86;
    const LAMBDA: f64 = 1.54;

    fn sample() -> Sample {
        Sample::new(
            "cubic",
            Lattice::from_degrees(A, A, A, 90.0, 90.0, 90.0).unwrap(),
            Orientation::Manual {
                u: UnitQuaternion::identity(),
            },
        )
    }

    /// Sample circle and detector circle both about x+, beam y+, detector
    /// direction z+.
    fn model() -> LinearModel {
        LinearModel {
            geometry: Geometry::builder("linear-hkl")
                .sample_rotation("om", [1.0, 0.0, 0.0])
                .detector_rotation("tth", [1.0, 0.0, 0.0])
                .beam_direction([0.0, 1.0, 0.0])
                .build()
                .unwrap(),
            sample: Some(sample()),
            offset_axis: None,
            nch: 1000,
            direction: Some(dir("z+")),
        }
    }

    /// xu's `Rz(azimuth)·Ry(tilt)` misorientation of the sample.
    fn misorientation(tilt: f64, azimuth: f64) -> hklrs::nalgebra::Matrix3<f64> {
        let (sp, cp) = azimuth.to_radians().sin_cos();
        let (sc, cc) = tilt.to_radians().sin_cos();
        hklrs::nalgebra::Matrix3::new(cp * cc, -sp, cp * sc, sp * cc, cp, sp * sc, -sc, 0.0, cc)
    }

    fn frames(m: &LinearModel, tilt: f64, azimuth: f64) -> Vec<CalFrame> {
        let truth = Linear::new(dir("z+"), 300.3, 1000, 1.0, 2.0e-4);
        let detector = truth.detector(m.geometry.beam_direction()).unwrap();
        let channel = |angles: [f64; 2], hkl: Option<Vector3<f64>>| {
            let g = &m.geometry;
            let poses = g.poses(&g.values_from_degrees(&angles).unwrap()).unwrap();
            let ki = poses.source * g.beam_direction();
            let dir = match hkl {
                None => ki.into_inner(),
                Some(h) => {
                    (TAU / LAMBDA) * ki.into_inner()
                        + poses.sample.rotation
                            * (misorientation(tilt, azimuth) * sample().ub() * h)
                }
            };
            let to_arm = poses.detector.inverse();
            detector
                .locate(to_arm * poses.sample_position(), to_arm.rotation * dir)
                .unwrap()[1]
        };
        let mut out = Vec::new();
        let mut push = |angles: [f64; 2], hkl: Option<Vector3<f64>>| {
            let kind = hkl.map_or(FrameKind::PrimaryBeam, FrameKind::Reflection);
            out.push(
                CalFrame::new(
                    out.len(),
                    angles.to_vec(),
                    Peak1 {
                        pos: channel(angles, hkl),
                        sigma: 0.05,
                    }
                    .into(),
                    kind,
                )
                .unwrap(),
            );
        };
        for tth in [-1.5, -0.5, 0.5, 1.5] {
            push([0.0, tth], None);
        }
        for l in [2.0, 4.0, 6.0, 8.0] {
            let q = TAU * l / A;
            let k = TAU / LAMBDA;
            let om = (q / (2.0 * k)).asin();
            let tth = (q * om.cos()).atan2(k - q * om.sin()).to_degrees();
            for d in [-0.6, 0.0, 0.6] {
                push([om.to_degrees(), tth + d], Some(Vector3::new(0.0, 0.0, l)));
            }
        }
        out
    }

    #[test]
    fn wavelength_and_pixel_size_are_recovered_from_reflections() {
        let m = model();
        let s = LinearHklStart {
            line: LinearStart {
                cch: Param::free(290.0),
                pwidth: Param::free(2.1e-4),
                distance: Param::fixed(1.0),
                tilt: Param::fixed(0.0),
                angle_offset: Param::fixed(0.0),
                wavelength: Param::free(1.5),
            },
            sample_tilt: Param::fixed(0.0),
            sample_azimuth: Param::fixed(0.0),
        };
        let fit = calibrate_linear_hkl(&m, &frames(&m, 0.0, 0.0), &[s]).unwrap();
        let h = fit.value();
        assert!((h.line.cch - 300.3).abs() < 1e-5, "{}", h.line.cch);
        assert!((h.line.pwidth - 2e-4).abs() < 1e-9, "{}", h.line.pwidth);
        assert!((h.wavelength - LAMBDA).abs() < 1e-7, "{}", h.wavelength);
        assert_eq!(h.sample_tilt, Deg(0.0));
        // a linear fit needs both kinds of frame too
        let only_primary: Vec<_> = frames(&m, 0.0, 0.0)
            .into_iter()
            .filter(|f| f.kind() == FrameKind::PrimaryBeam)
            .collect();
        assert!(matches!(
            calibrate_linear_hkl(&m, &only_primary, &[s]),
            Err(CalibError::BadProblem)
        ));
    }

    fn free_tilt(azimuth: Param) -> LinearHklStart {
        LinearHklStart {
            line: LinearStart {
                cch: Param::free(290.0),
                pwidth: Param::free(2.1e-4),
                distance: Param::fixed(1.0),
                tilt: Param::fixed(0.0),
                angle_offset: Param::fixed(0.0),
                wavelength: Param::free(1.5),
            },
            sample_tilt: Param::free(0.2),
            sample_azimuth: azimuth,
        }
    }

    // On a linear detector behind one detector circle, (0 0 L) reflections
    // see the sample misorientation (tilt 0.3°, azimuth 30° here) only through
    // its in-plane part tilt·sin(azimuth) = 0.15°: the out-of-plane part
    // cos(azimuth)·tilt moves the scattered ray along the strip's width,
    // which the channel does not measure. The two cannot be separated.
    #[test]
    fn tilt_and_azimuth_together_are_unidentifiable_from_one_circle() {
        let m = model();
        let fs = frames(&m, 0.3, 30.0);
        let err = calibrate_linear_hkl(&m, &fs, &[free_tilt(Param::free(20.0))]).unwrap_err();
        assert!(
            matches!(&err, CalibError::Unidentifiable { params } if params == &[ParamId::SampleAzimuth]),
            "{err:?}"
        );
    }

    #[test]
    fn with_the_azimuth_known_the_tilt_is_recovered() {
        let m = model();
        let fs = frames(&m, 0.3, 30.0);
        let fit = calibrate_linear_hkl(&m, &fs, &[free_tilt(Param::fixed(30.0))]).unwrap();
        let h = fit.value();
        let s = fit.sigma(ParamId::SampleTilt).unwrap();
        assert!(
            (h.sample_tilt.0 - 0.3).abs() <= 5.0 * s.max(1e-9),
            "{:?} ± {s}",
            h.sample_tilt
        );
        assert!((h.sample_tilt.0 - 0.3).abs() < 1e-6, "{:?}", h.sample_tilt);
        assert!((h.wavelength - LAMBDA).abs() < 1e-7, "{}", h.wavelength);
        assert!((h.line.cch - 300.3).abs() < 1e-5, "{}", h.line.cch);
        // the identifiable combination survives a wrong azimuth: tilt·sin(az)
        let wrong = calibrate_linear_hkl(&m, &fs, &[free_tilt(Param::fixed(90.0))]).unwrap();
        let t = wrong.value().sample_tilt.0;
        assert!((t - 0.3 * 30f64.to_radians().sin()).abs() < 1e-3, "{t}");
    }
}
