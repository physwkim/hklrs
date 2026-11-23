//! The SIXS importer against binoculars-ng 7be4f79 (golden/bin_7be4f79_sixs.c):
//! corner and central pixel positions after calibration.

pub mod common;

use common::{f, load, vec3};
use hklrs::{Detector, Error, sixs};

const TOL: f64 = 1e-14;

#[test]
fn sixs_matches_binoculars() {
    let doc = load("bin_7be4f79_sixs.json");
    let cases = doc["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for c in cases {
        let det = Detector::try_from(c["id"].as_u64().unwrap() as u32).unwrap();
        let (ix0, iy0) = (
            c["ix0"].as_u64().unwrap() as usize,
            c["iy0"].as_u64().unwrap() as usize,
        );
        let out = sixs(
            &det.pixel_detector().unwrap(),
            (iy0, ix0),
            f(&c["sdd"]),
            f(&c["detrot"]),
        )
        .unwrap();
        assert_eq!(
            out.mask(),
            det.pixel_detector().unwrap().mask(),
            "{}",
            det.name()
        );
        for p in c["pixels"].as_array().unwrap() {
            let (s, fa) = (
                p[0].as_u64().unwrap() as usize,
                p[1].as_u64().unwrap() as usize,
            );
            let want = vec3(&serde_json::Value::Array(
                p.as_array().unwrap()[2..].to_vec(),
            ));
            let err = (out.pixel(s, fa).unwrap().coords - want).abs().max();
            assert!(
                err < TOL,
                "{} {c} pixel ({s}, {fa}) off by {err}",
                det.name()
            );
        }
    }
}

#[test]
fn centre_outside_the_detector_is_an_error() {
    let det = Detector::Merlin.pixel_detector().unwrap();
    assert_eq!(
        sixs(&det, (256, 0), 1.0, 0.0),
        Err(Error::PixelOutside {
            slow: 256,
            fast: 0,
            shape: (256, 256)
        })
    );
}
