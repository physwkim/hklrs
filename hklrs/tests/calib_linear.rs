//! Linear-detector calibration against the xrayutilities golden
//! (psd_chdeg / linear_detector_calib): one detector circle `x+`, beam `y+`,
//! detector direction `z+`.

use std::path::PathBuf;

use hklrs::Geometry;
use hklrs::calib::{
    CalFrame, FrameKind, LinearModel, LinearStart, Param, ParamId, Peak1, calibrate_linear,
};
use serde_json::Value;

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../golden/xu_calib_linear.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn model() -> LinearModel {
    LinearModel {
        geometry: Geometry::builder("xu-linear")
            .detector_rotation("a", [1.0, 0.0, 0.0])
            .beam_direction([0.0, 1.0, 0.0])
            .build()
            .unwrap(),
        sample: None,
        offset_axis: None,
        nch: 1000,
        direction: Some("z+".parse().unwrap()),
    }
}

fn frames(case: &Value) -> Vec<CalFrame> {
    let (a, c, s) = (
        floats(&case["angles"]),
        floats(&case["xu_channel"]),
        floats(&case["sigma_channel"]),
    );
    (0..a.len())
        .map(|i| {
            CalFrame::new(
                i,
                vec![a[i]],
                Peak1 {
                    pos: c[i],
                    sigma: s[i],
                }
                .into(),
                FrameKind::PrimaryBeam,
            )
            .unwrap()
        })
        .collect()
}

fn start(tilt: Param) -> LinearStart {
    LinearStart {
        cch: Param::free(290.0),
        pwidth: Param::free(2.1e-4),
        distance: Param::fixed(1.0),
        tilt,
        angle_offset: Param::fixed(0.0),
        wavelength: Param::fixed(1.54),
    }
}

fn near(got: f64, want: f64, rel: f64) {
    assert!((got - want).abs() <= want.abs() * rel, "{got} vs {want}");
}

#[test]
fn tilt_free_recovers_the_truth_from_xus_channels() {
    let doc = golden();
    for case in doc["cases"].as_array().unwrap() {
        let tilt = case["true_tilt"].as_f64().unwrap();
        let fit = calibrate_linear(&model(), &frames(case), &[start(Param::free(1.0))]).unwrap();
        let l = fit.value();
        // xu's channels come from a local line fit of Ang2Q: good to ~1e-6
        near(l.cch, 300.3, 1e-6);
        near(l.pwidth, 2e-4, 1e-6);
        assert!((l.tilt.0 - tilt).abs() < 1e-4, "{:?} vs {tilt}", l.tilt);
        assert!(fit.rms() < 1e-3, "rms {}", fit.rms());
        assert!(fit.sigma(ParamId::Tilt).is_some());
    }
}

#[test]
fn tilt_held_at_zero_matches_xus_psd_chdeg_and_misfits_tilted_data() {
    let doc = golden();
    // psd_chdeg with usetilt False is the same least-squares line in tan:
    // hklrs agrees with it to 1e-9 and, on tilted data, shows the misfit
    for case in doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["usetilt"] == false)
    {
        let xu = floats(&case["psd_chdeg"]["result"]);
        let fit = calibrate_linear(&model(), &frames(case), &[start(Param::fixed(0.0))]).unwrap();
        let l = fit.value();
        near(l.pwidth, -xu[0], 1e-9);
        near(l.cch, xu[1], 1e-9);
        if case["true_tilt"].as_f64().unwrap() == 0.0 {
            assert!(fit.rms() < 1e-3);
        } else {
            assert!(fit.rms() > 10.0, "rms {}", fit.rms());
        }
    }
}

#[test]
fn the_direction_is_found_among_the_four_perpendicular_to_the_beam() {
    let doc = golden();
    let case = &doc["cases"][2];
    let m = LinearModel {
        direction: None,
        ..model()
    };
    let fit = calibrate_linear(&m, &frames(case), &[start(Param::free(1.0))]).unwrap();
    assert_eq!(fit.value().dir, "z+".parse().unwrap());
}
