//! Area-detector calibration against the xrayutilities golden: hklrs
//! recovers the truth xu generated the data from, and xu's own fit does not
//! (XU-1: its calibration model applies the detector circles in reverse
//! order; XU-6: it reports a run-away fit as a result).

use std::path::PathBuf;

use hklrs::calib::{
    AreaModel, AreaStart, CalFrame, CalibError, FrameKind, Param, ParamId, Peak, calibrate_area,
};
use hklrs::xu::{Area, Direction};
use hklrs::{Geometry, Wavelength};
use serde_json::Value;

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../golden/xu_calib_area.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn dir(text: &str) -> Direction {
    text.parse().unwrap()
}

/// xu's circles: outer `z+` (angle1), arm `y-` (angle2), beam `x+`.
fn geometry() -> Geometry {
    Geometry::builder("xu-area")
        .detector_rotation("a1", [0.0, 0.0, 1.0])
        .detector_rotation("a2", [0.0, -1.0, 0.0])
        .beam_direction([1.0, 0.0, 0.0])
        .build()
        .unwrap()
}

fn model() -> AreaModel {
    AreaModel {
        geometry: geometry(),
        sample: None,
        offset_axis: None,
        nch: [800, 800],
        directions: Some([dir("z+"), dir("y+")]),
    }
}

fn frames(doc: &Value, pixel: &str) -> Vec<CalFrame> {
    doc["frames"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, fr)| {
            let at = |k: &str| [f(&fr[k][0]), f(&fr[k][1])];
            CalFrame::new(
                i,
                vec![f(&fr["angle1"]), f(&fr["angle2"])],
                Peak {
                    pos: at(pixel),
                    sigma: at("sigma_pixel"),
                },
                FrameKind::PrimaryBeam,
            )
            .unwrap()
        })
        .collect()
}

fn start(detrot: Param) -> AreaStart {
    AreaStart {
        cch1: Param::free(390.0),
        cch2: Param::free(405.0),
        pwidth1: Param::free(0.0021),
        pwidth2: Param::free(0.0019),
        distance: Param::fixed(1.0),
        tiltazimuth: Param::fixed(0.0),
        tilt: Param::fixed(0.0),
        detrot,
        angle_offset: Param::fixed(0.0),
        wavelength: Param::fixed(1.54),
    }
}

fn value(fit: &hklrs::calib::Fit<Area>) -> Area {
    fit.value().clone()
}

#[test]
fn exact_pixels_recover_the_truth_xu_misses_by_its_circle_order() {
    let doc = golden();
    let exact = &doc["fits"][2];
    // xu on the same exact pixels: pwidth 2.05e-3 / 1.95e-3 against 2e-3 (XU-1)
    let (xu1, xu2) = (
        f(&exact["fitted"]["pwidth1"]),
        f(&exact["fitted"]["pwidth2"]),
    );
    assert!((xu1 - 2e-3).abs() > 4e-5 && (xu2 - 2e-3).abs() > 4e-5);
    let fit = calibrate_area(
        &model(),
        &frames(&doc, "true_pixel"),
        &[start(Param::fixed(2.0))],
    )
    .unwrap();
    let a = value(&fit);
    // the golden's exact pixels come from a local linear fit of xu's Ang2Q
    // (golden/xu_calib_area.py beampix), good to about 1e-5 relative
    for (got, want) in [
        (a.cch1, 401.3),
        (a.cch2, 396.7),
        (a.pwidth1, 2e-3),
        (a.pwidth2, 2e-3),
    ] {
        assert!((got - want).abs() <= want * 1e-4, "{got} vs {want}");
    }
    // two orders of magnitude closer than xu's fit of the same pixels
    assert!((a.pwidth1 - 2e-3).abs() * 100.0 < (xu1 - 2e-3).abs());
    assert!((a.pwidth2 - 2e-3).abs() * 100.0 < (xu2 - 2e-3).abs());
    assert!(fit.rejected().is_empty());
}

#[test]
fn xu_peak_pixels_recover_the_truth_within_their_errors() {
    let doc = golden();
    let fit = calibrate_area(
        &model(),
        &frames(&doc, "xu_pixel"),
        &[start(Param::fixed(2.0))],
    )
    .unwrap();
    let a = value(&fit);
    for (id, got, want) in [
        (ParamId::Cch1, a.cch1, 401.3),
        (ParamId::Cch2, a.cch2, 396.7),
        (ParamId::Pwidth1, a.pwidth1, 2e-3),
        (ParamId::Pwidth2, a.pwidth2, 2e-3),
    ] {
        let s = fit.sigma(id).unwrap();
        assert!(
            (got - want).abs() <= 4.0 * s,
            "{id:?}: {got} vs {want} ± {s}"
        );
    }
}

#[test]
fn detrot_free_from_zero_converges_where_xu_ran_off_with_info_12() {
    let doc = golden();
    let xu = &doc["fits"][0];
    assert_eq!(xu["odr"]["info"], 12);
    assert!(f(&xu["fitted"]["detrot"]) > 1e6);
    let fit = calibrate_area(
        &model(),
        &frames(&doc, "true_pixel"),
        &[start(Param::free(0.0))],
    )
    .unwrap();
    assert!(
        (value(&fit).detrot.0 - 2.0).abs() < 1e-4,
        "{:?}",
        value(&fit).detrot
    );
    assert!(fit.sigma(ParamId::Detrot).is_some());
}

#[test]
fn a_fixed_wavelength_nobody_uses_is_not_fitted() {
    // primary-beam frames carry no wavelength: freeing it is unidentifiable
    let doc = golden();
    let mut s = start(Param::fixed(2.0));
    s.wavelength = Param::free(1.54);
    let err = calibrate_area(&model(), &frames(&doc, "true_pixel"), &[s]).unwrap_err();
    assert!(
        matches!(&err, CalibError::Unidentifiable { params } if params == &[ParamId::Wavelength]),
        "{err:?}"
    );
    let _ = Wavelength::new(1.54).unwrap();
}
