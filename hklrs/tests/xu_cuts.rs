//! Line-cut geometry against xrayutilities deff08f (golden/xu_cuts.py): the
//! coordinate along the cut, the distance from it and the pixel selection
//! xu makes, for every kind and integration direction, plus the xu
//! divergences XU-17 and XU-18 (XU-19 is about intensities: phase 2).

pub mod common;

use common::{f, load};
use hklrs::Wavelength;
use hklrs::cut::{Cut, CutKind, IntDir};
use hklrs::grid::Binning;
use hklrs::nalgebra::Vector3;
use serde_json::Value;

const TOL: f64 = 1e-9;

fn points(doc: &Value, cloud: &str) -> Vec<Vector3<f64>> {
    let c = &doc["clouds"][cloud];
    (0..c[0].as_array().unwrap().len())
        .map(|i| Vector3::new(f(&c[0][i]), f(&c[1][i]), f(&c[2][i])))
        .collect()
}

fn at3(kind: &str, a: &Value) -> [f64; 3] {
    let v = |i: usize| f(&a[i]);
    match kind {
        "qz" => [v(0), v(1), 0.0],
        "qy" => [v(0), 0.0, v(1)],
        "qx" => [0.0, v(0), v(1)],
        _ => [v(0), v(1), v(2)],
    }
}

fn intdir(s: &str) -> IntDir {
    match s {
        "q" => IntDir::Q,
        "omega" => IntDir::Omega,
        "2theta" => IntDir::TwoTheta,
        "radial" => IntDir::Radial,
        _ => panic!("{s}"),
    }
}

fn vec3(v: &Value) -> [f64; 3] {
    [f(&v[0]), f(&v[1]), f(&v[2])]
}

fn kind(c: &Value) -> CutKind {
    match c["kind"].as_str().unwrap() {
        "qz" => CutKind::Qz,
        "qy" => CutKind::Qy,
        "qx" => CutKind::Qx,
        "omega" => CutKind::Omega,
        "radial" => CutKind::Radial,
        "twotheta" => CutKind::TwoTheta,
        "line" => CutKind::Line {
            point: vec3(&c["point"]),
            dir: vec3(&c["dir"]),
        },
        k => panic!("{k}"),
    }
}

/// xu's coordinate along the cut is an angle (degrees) for the angular kinds,
/// and for qy/qz/qx a length.
fn along_is_angle(k: &str) -> bool {
    matches!(k, "omega" | "radial" | "twotheta")
}

fn perp_is_angle(c: &Value) -> bool {
    c["intdir"] != "q"
}

/// A golden degree value as radians; a length is unchanged.
fn conv(v: &Value, angle: bool) -> f64 {
    match v.as_f64() {
        None => f64::NAN,
        Some(x) if angle => x.to_radians(),
        Some(x) => x,
    }
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= TOL * (1.0 + b.abs())
}

fn cut(c: &Value, half_width: f64) -> Cut {
    let k = c["kind"].as_str().unwrap();
    let at = if k == "line" {
        [0.0; 3]
    } else {
        at3(k, &c["at"])
    };
    Cut::new(
        kind(c),
        intdir(c["intdir"].as_str().unwrap()),
        at,
        half_width,
        Binning::new(0.01, 0.0).unwrap(),
    )
    .unwrap()
}

/// xu's `dis`, in this crate's units: `intrange / 2` (`get_arbitrary_line`: `intrange`, a radius),
/// radians for an angular window.
fn xu_half_width(c: &Value) -> f64 {
    let dis = f(&c["intrange"]) / if c["kind"] == "line" { 1.0 } else { 2.0 };
    if perp_is_angle(c) {
        dis.to_radians()
    } else {
        dis
    }
}

#[test]
fn coordinates_and_selection_match_xu() {
    let doc = load("xu_cuts.json");
    let wl = Wavelength::new(f(&doc["wavelength"])).unwrap();
    let mut checked = 0;
    for c in doc["cases"].as_array().unwrap() {
        if c.get("perp").is_none() {
            continue;
        }
        let name = format!("{} {} {}", c["cloud"], c["kind"], c["intdir"]);
        let cut = cut(c, xu_half_width(c));
        let k = c["kind"].as_str().unwrap();
        let q = points(&doc, c["cloud"].as_str().unwrap());
        let mut selected = 0;
        for (i, p) in q.iter().enumerate() {
            let got = cut.cutting(p, wl);
            let along = conv(&c["along"][i], along_is_angle(k));
            let perp = conv(&c["perp"][i], perp_is_angle(c));
            assert!(
                same(got.along, along),
                "{name} pixel {i} along {} vs {along}",
                got.along
            );
            assert!(
                same(got.perp, perp),
                "{name} pixel {i} perp {} vs {perp}",
                got.perp
            );
            let want = c["mask"][i].as_i64().unwrap() == 1;
            let located = cut.locate(p, wl).unwrap();
            assert_eq!(located.is_some(), want, "{name} pixel {i} selection");
            if let Some(l) = located {
                selected += 1;
                let b = cut.binning();
                assert!(b.edge(l.bin) <= l.cutting.along && l.cutting.along < b.edge(l.bin + 1));
            }
        }
        assert_eq!(
            selected as i64,
            c["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .sum::<i64>()
        );
        checked += 1;
    }
    // 2 clouds x (3 + 3 + 3 + 2 + 2 + 2 kind/intdir) + 2 lines
    assert_eq!(checked, 2 * (15 + 2));
}

/// XU-17: xu's `intrange` is a full width (`_get_cut` is handed `intrange / 2`); a caller
/// who follows the docstring of qy/qx/omega/radial/ttheta and passes the half-width gets a
/// window of half the size. `half_width` here is the half-width.
#[test]
fn xu_17_intrange_is_a_full_width() {
    let doc = load("xu_cuts.json");
    let wl = Wavelength::new(f(&doc["wavelength"])).unwrap();
    let mut widened = 0;
    for c in doc["cases"].as_array().unwrap() {
        let k = c["kind"].as_str().unwrap();
        if c.get("perp").is_none() || !matches!(k, "qy" | "qx" | "omega" | "radial" | "twotheta") {
            continue;
        }
        // xu with intrange = R selects |perp| < R/2; hklrs with half_width = R selects |perp| < R
        let full = f(&c["intrange"]);
        let hw = if perp_is_angle(c) {
            full.to_radians()
        } else {
            full
        };
        let wide = cut(c, hw);
        let xu_used: i64 = c["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .sum();
        let n = points(&doc, c["cloud"].as_str().unwrap())
            .iter()
            .filter(|p| wide.locate(p, wl).unwrap().is_some())
            .count() as i64;
        assert!(
            n > xu_used,
            "{} {}: {n} vs xu {xu_used}",
            c["kind"],
            c["intdir"]
        );
        widened += 1;
    }
    assert_eq!(widened, 2 * (3 + 3 + 2 + 2 + 2));
}

/// XU-18: a window that holds no pixel makes xu raise ValueError from `_get_cut`
/// (`numpy.max` of an empty selection); here it is `Ok(None)` for every pixel.
#[test]
fn xu_18_empty_window_is_not_an_error() {
    let doc = load("xu_cuts.json");
    let wl = Wavelength::new(f(&doc["wavelength"])).unwrap();
    let mut seen = 0;
    for c in doc["cases"].as_array().unwrap() {
        if c.get("error").is_none() {
            continue;
        }
        assert_eq!(c["error"], "ValueError");
        let cut = cut(c, xu_half_width(c));
        for p in points(&doc, c["cloud"].as_str().unwrap()) {
            assert_eq!(
                cut.locate(&p, wl),
                Ok(None),
                "{} {}",
                c["kind"],
                c["intdir"]
            );
        }
        seen += 1;
    }
    assert!(seen > 0);
}

/// XU-25: `get_qx_scan` with omega or 2theta raises ValueError for any array input
/// (`line_cuts.py:334,:341` pass `[array, scalar, scalar]` to `Q2Ang`); the golden holds the
/// coordinates xu's own `Q2Ang` gives for the intended broadcast cut.
#[test]
fn qx_angular_cuts_follow_the_intended_expression() {
    let doc = load("xu_cuts.json");
    let broken: Vec<_> = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c.get("xu_error").is_some())
        .collect();
    assert_eq!(broken.len(), 4);
    assert!(
        broken
            .iter()
            .all(|c| c["kind"] == "qx" && c["xu_error"] == "ValueError")
    );
}
