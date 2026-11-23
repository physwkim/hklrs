//! The SIXS importer against binoculars-ng 7be4f79 (golden/bin_7be4f79_sixs.c):
//! corner and central pixel positions after calibration. Where binoculars'
//! pixels differ before calibration (BIN-11, BIN-12) or its central pixel
//! keeps an x (BIN-9), the golden is moved by exactly those differences.

pub mod common;

use common::{f, load, vec3};
use hklrs::nalgebra::{Rotation3, Vector3};
use hklrs::{Detector, Error, sixs};
use serde_json::Value;

const TOL: f64 = 1e-14;

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

/// (slow, fast) and position of each dumped pixel.
fn pixels(v: &Value) -> Vec<((usize, usize), Vector3<f64>)> {
    let row = |p: &Value| {
        let p = p.as_array().unwrap();
        ((u(&p[0]), u(&p[1])), vec3(&Value::Array(p[2..].to_vec())))
    };
    v.as_array().unwrap().iter().map(row).collect()
}

/// binoculars' calibrated pixel is `rot (p_b + (sdd, -c_b.y, -c_b.z))`, hklrs'
/// `rot (p_h + (sdd, 0, 0) - c_h)`; they differ by
/// `rot ((p_h - p_b) - (c_h - c_b) - c_b.x x)`.
#[test]
fn sixs_matches_binoculars() {
    let doc = load("bin_7be4f79_sixs.json");
    let cases = doc["cases"].as_array().unwrap();
    let ids: Vec<u64> = cases.iter().map(|c| c["id"].as_u64().unwrap()).collect();
    for det in Detector::ALL {
        assert!(ids.contains(&(det as u64)), "{} has no case", det.name());
    }
    for c in cases {
        let det = Detector::try_from(c["id"].as_u64().unwrap() as u32).unwrap();
        let pd = det.pixel_detector();
        let centre = (u(&c["iy0"]), u(&c["ix0"]));
        let (sdd, detrot) = (f(&c["sdd"]), f(&c["detrot"]));
        let out = sixs(&pd, centre, sdd, detrot).unwrap();
        assert_eq!(out.mask(), pd.mask(), "{}", det.name());
        let rot = Rotation3::from_axis_angle(&Vector3::x_axis(), detrot);
        let before = pixels(&c["before"]);
        let (centre_px, c_b) = before[4];
        assert_eq!(centre_px, centre);
        let c_h = pd.pixel(centre.0, centre.1).unwrap().coords;
        for (((s, fa), want), (at, p_b)) in pixels(&c["pixels"]).into_iter().zip(before) {
            assert_eq!((s, fa), at);
            let p_h = pd.pixel(s, fa).unwrap().coords;
            let moved = rot * ((p_h - p_b) - (c_h - c_b) - Vector3::x() * c_b.x);
            let err = (out.pixel(s, fa).unwrap().coords - (want + moved))
                .abs()
                .max();
            assert!(
                err < TOL,
                "{} {c} pixel ({s}, {fa}) off by {err}",
                det.name()
            );
        }
    }
}

/// BIN-9: the Cirpad's central pixel ends at x = sdd, where binoculars leaves
/// it at sdd plus its x before calibration.
#[test]
fn bin_9_cirpad_central_pixel_at_sdd() {
    let doc = load("bin_7be4f79_sixs.json");
    let mut seen = 0;
    for c in doc["cases"].as_array().unwrap() {
        if c["id"].as_u64() != Some(Detector::Cirpad as u64) {
            continue;
        }
        let centre = (u(&c["iy0"]), u(&c["ix0"]));
        let (sdd, detrot) = (f(&c["sdd"]), f(&c["detrot"]));
        let out = sixs(&Detector::Cirpad.pixel_detector(), centre, sdd, detrot).unwrap();
        let rot = Rotation3::from_axis_angle(&Vector3::x_axis(), detrot);
        let got = out.pixel(centre.0, centre.1).unwrap().coords;
        assert!((got - rot * Vector3::new(sdd, 0.0, 0.0)).abs().max() < TOL);
        let (_, before) = pixels(&c["before"])[4];
        let (_, binoculars) = pixels(&c["pixels"])[4];
        assert!(before.x.abs() > 1e-3, "{c}");
        assert!((binoculars.x - (sdd + before.x)).abs() < TOL, "{c}");
        seen += 1;
    }
    assert!(seen > 0);
}

#[test]
fn centre_outside_the_detector_is_an_error() {
    let det = Detector::Merlin.pixel_detector();
    assert_eq!(
        sixs(&det, (256, 0), 1.0, 0.0),
        Err(Error::PixelOutside {
            slow: 256,
            fast: 0,
            shape: (256, 256)
        })
    );
}
