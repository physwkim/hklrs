//! The xrayutilities detector importer against xrayutilities deff08f
//! (golden/xu_detector.py): pixel positions at zero detector angles and Q in
//! the sample frame (UB = I) at each angle set.

pub mod common;

use common::{f, load, vec3};
use hklrs::nalgebra::Unit;
use hklrs::xu::{Area, Direction, Linear};
use hklrs::{Deg, Geometry, PixelDetector, Wavelength, scattering};
use serde_json::Value;

const POS_TOL: f64 = 1e-14;
const Q_TOL: f64 = 1e-12;

fn doc() -> Value {
    load("xu_detector.json")
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn direction(v: &Value) -> Direction {
    v.as_str().unwrap().parse().unwrap()
}

fn array(v: &Value) -> &Vec<Value> {
    v.as_array().unwrap()
}

fn geometry(doc: &Value) -> Geometry {
    let mut b = Geometry::builder("xu");
    for (i, a) in array(&doc["sample_axes"]).iter().enumerate() {
        b = b.sample_rotation(&format!("s{i}"), direction(a).vector().into_inner().into());
    }
    for (i, a) in array(&doc["detector_axes"]).iter().enumerate() {
        b = b.detector_rotation(&format!("d{i}"), direction(a).vector().into_inner().into());
    }
    b.beam_direction(vec3(&doc["r_i"]).into()).build().unwrap()
}

fn area(p: &Value) -> Area {
    Area {
        tilt: Deg(f(&p["tilt"])),
        tiltazimuth: Deg(f(&p["tiltazimuth"])),
        detrot: Deg(f(&p["detrot"])),
        nav: [u(&p["nav"][0]), u(&p["nav"][1])],
        roi: p["roi"]
            .as_array()
            .map(|r| [u(&r[0]), u(&r[1]), u(&r[2]), u(&r[3])]),
        ..Area::new(
            direction(&p["dir1"]),
            direction(&p["dir2"]),
            [f(&p["cch1"]), f(&p["cch2"])],
            [u(&p["nch1"]), u(&p["nch2"])],
            f(&p["distance"]),
            [f(&p["pwidth1"]), f(&p["pwidth2"])],
        )
    }
}

fn linear(p: &Value) -> Linear {
    Linear {
        tilt: Deg(f(&p["tilt"])),
        nav: u(&p["nav"]),
        roi: p["roi"].as_array().map(|r| [u(&r[0]), u(&r[1])]),
        ..Linear::new(
            direction(&p["dir"]),
            f(&p["cch"]),
            u(&p["nch"]),
            f(&p["distance"]),
            f(&p["pwidth"]),
        )
    }
}

fn case<'a>(doc: &'a Value, kind: &str, name: &str) -> &'a Value {
    array(&doc[kind])
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no {kind} case {name}"))
}

/// Golden values of one case as (slow, fast) -> value; a linear case is one row.
fn grid(v: &Value, linear: bool) -> Vec<Vec<&Value>> {
    if linear {
        vec![array(v).iter().collect()]
    } else {
        array(v)
            .iter()
            .map(|row| array(row).iter().collect())
            .collect()
    }
}

fn check(doc: &Value, det: &PixelDetector, c: &Value, linear: bool) {
    let geometry = geometry(doc);
    let wavelength = Wavelength::new(f(&doc["wavelength"])).unwrap();
    let pos = grid(&c["pos"], linear);
    assert_eq!(det.shape(), (pos.len(), pos[0].len()), "shape");
    let qs: Vec<_> = array(&c["q"]).iter().map(|q| grid(q, linear)).collect();
    for (angles, q) in array(&doc["angles"]).iter().zip(&qs) {
        let user: Vec<f64> = array(angles).iter().map(f).collect();
        let values = geometry.values_from_degrees(&user).unwrap();
        for (s, row) in pos.iter().enumerate() {
            for (k, want) in row.iter().enumerate() {
                let pixel = det.pixel(s, k).unwrap();
                let err = (pixel.coords - vec3(want)).norm();
                assert!(err <= POS_TOL, "pixel ({s}, {k}) off by {err} m");
                let got = scattering(&geometry, &values, wavelength, &pixel)
                    .unwrap()
                    .q_sample;
                let err = (got - vec3(q[s][k])).norm();
                assert!(err <= Q_TOL, "Q ({s}, {k}) at {user:?} off by {err}");
            }
        }
    }
}

fn check_area(name: &str) {
    let doc = doc();
    let c = case(&doc, "area", name);
    let beam = Unit::new_normalize(vec3(&doc["r_i"]));
    check(&doc, &area(&c["params"]).detector(beam).unwrap(), c, false);
}

fn check_linear(name: &str) {
    let doc = doc();
    let c = case(&doc, "linear", name);
    let beam = Unit::new_normalize(vec3(&doc["r_i"]));
    check(&doc, &linear(&c["params"]).detector(beam).unwrap(), c, true);
}

#[test]
fn area_plain() {
    check_area("plain");
}

#[test]
fn area_tilt() {
    check_area("tilt");
}

#[test]
fn area_tilt_tiltazimuth() {
    check_area("tilt_tiltazimuth");
}

#[test]
fn area_detrot() {
    check_area("detrot");
}

#[test]
fn area_roi() {
    check_area("roi");
}

#[test]
fn area_nav() {
    check_area("nav");
}

#[test]
fn area_nav_roi() {
    check_area("nav_roi");
}

#[test]
fn area_combined() {
    check_area("combined");
}

#[test]
fn linear_plain() {
    check_linear("plain");
}

#[test]
fn linear_tilt() {
    check_linear("tilt");
}

#[test]
fn linear_roi() {
    check_linear("roi");
}

#[test]
fn linear_nav() {
    check_linear("nav");
}

#[test]
fn linear_nav_roi() {
    check_linear("nav_roi");
}

#[test]
fn linear_combined() {
    check_linear("combined");
}

/// XU-13: xu's own Nav output sits on the first raw pixel of a block
/// aligned to pixel 0, which is why the golden takes Nav cases from the
/// block-grid reference detector instead.
#[test]
fn xu13_nav_pixel_is_the_first_pixel_of_a_pixel_0_aligned_block() {
    let doc = doc();
    for (kind, linear) in [("area", false), ("linear", true)] {
        for c in array(&doc[kind])
            .iter()
            .filter(|c| !c["xu_nav_pos"].is_null())
        {
            let p = &c["params"];
            let (nav, start) = if linear {
                let start = p["roi"].as_array().map_or(0, |r| u(&r[0]));
                ([1, u(&p["nav"])], [0, start])
            } else {
                let start = p["roi"].as_array().map_or([0, 0], |r| [u(&r[0]), u(&r[2])]);
                ([u(&p["nav"][0]), u(&p["nav"][1])], start)
            };
            let full = grid(&c["full_pos"], linear);
            for (k1, row) in grid(&c["xu_nav_pos"], linear).iter().enumerate() {
                for (k2, got) in row.iter().enumerate() {
                    let j1 = nav[0] * (start[0] / nav[0] + k1);
                    let j2 = nav[1] * (start[1] / nav[1] + k2);
                    let err = (vec3(got) - vec3(full[j1][j2])).norm();
                    assert!(
                        err <= POS_TOL,
                        "{kind} {}: ({k1}, {k2}) off by {err}",
                        c["name"]
                    );
                }
            }
        }
    }
}
