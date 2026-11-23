//! PONI import against pyFAI 2026.9.0 `Geometry.calc_pos_zyx`
//! (golden/pyfai_poni.py).

pub mod common;

use common::{f, load};
use hklrs::Rad;
use hklrs::nalgebra::{Unit, Vector3};
use hklrs::poni::{LabAxes, Orientation, Poni};
use serde_json::Value;

/// Metres; pixel positions are about 0.3 m from the sample.
const TOL: f64 = 1e-14;

fn poni(g: &Value, case: &Value) -> Poni {
    let n = |v: &Value| v.as_u64().unwrap() as usize;
    Poni {
        dist: f(&g["dist"]),
        poni1: f(&g["poni1"]),
        poni2: f(&g["poni2"]),
        rot1: Rad(f(&case["rot1"])),
        rot2: Rad(f(&case["rot2"])),
        rot3: Rad(f(&case["rot3"])),
        pixel1: f(&g["pixel1"]),
        pixel2: f(&g["pixel2"]),
        shape: (n(&g["shape"][0]), n(&g["shape"][1])),
        orientation: Orientation::try_from(case["orientation"].as_u64().unwrap() as u32).unwrap(),
    }
}

/// Lab directions of pyFAI axes 1, 2, 3 for `up` normal to `beam`.
fn axes(beam: Vector3<f64>, up: Vector3<f64>) -> [Vector3<f64>; 3] {
    [up, up.cross(&beam), beam]
}

/// Largest distance between the hklrs pixel and pyFAI's over a case; NaN
/// when any is NaN.
fn worst_case(g: &Value, case: &Value, p: &Poni, beam: Vector3<f64>, up: Vector3<f64>) -> f64 {
    let lab = LabAxes::new(Unit::new_unchecked(beam), Unit::new_unchecked(up)).unwrap();
    let det = p.detector(&lab).unwrap();
    assert_eq!(det.shape(), p.shape);
    let [a1, a2, a3] = axes(beam, up);
    let mut worst = 0.0f64;
    for (px, zyx) in g["pixels"]
        .as_array()
        .unwrap()
        .iter()
        .zip(case["zyx"].as_array().unwrap())
    {
        let (t3, t1, t2) = (f(&zyx[0]), f(&zyx[1]), f(&zyx[2]));
        let want = a1 * t1 + a2 * t2 + a3 * t3;
        let ix = |k: usize| px[k].as_u64().unwrap() as usize;
        let got = det.pixel(ix(0), ix(1)).unwrap().coords;
        let d = (got - want).norm();
        if d > worst || d.is_nan() {
            worst = d;
        }
    }
    worst
}

fn labs() -> [(&'static str, Vector3<f64>, Vector3<f64>); 2] {
    let beam = Vector3::new(0.3, -0.5, 0.81).normalize();
    let up = beam.cross(&Vector3::new(0.2, 0.7, -0.4)).normalize();
    [
        // hkl presets: x along the beam, z up
        ("hkl", Vector3::x(), Vector3::z()),
        ("oblique", beam, up),
    ]
}

#[test]
fn import_matches_pyfai_calc_pos_zyx() {
    let g = load("pyfai_poni.json");
    let cases = g["cases"].as_array().unwrap();
    let mut seen = std::collections::BTreeSet::new();
    let mut worst = 0.0f64;
    for case in cases {
        let p = poni(&g, case);
        for (name, beam, up) in labs() {
            let w = worst_case(&g, case, &p, beam, up);
            worst = worst.max(w);
            assert!(
                w <= TOL,
                "{name} lab axes, orientation {:?}, {}: {w} m",
                p.orientation,
                case["rots"]
            );
        }
        seen.insert((p.orientation as u32, case["rots"].as_str().unwrap()));
    }
    eprintln!("PONI: worst pixel distance to pyFAI {worst} m");
    // every orientation with rot1, rot2, rot3 each alone and all together
    for o in 1..=4 {
        for r in ["none", "rot1", "rot2", "rot3", "all", "all_large"] {
            assert!(seen.contains(&(o, r)), "missing orientation {o} {r}");
        }
    }
}

/// BIN-3: binoculars turned rot3 the other way and its test, built on the
/// same axis, could not see it. Each rot3 case must reject the opposite sign.
#[test]
fn bin_3_rot3_sign_is_pyfai_s() {
    let g = load("pyfai_poni.json");
    let mut n = 0;
    for case in g["cases"].as_array().unwrap() {
        let p = poni(&g, case);
        if p.rot3.0 == 0.0 {
            continue;
        }
        let flipped = Poni {
            rot3: Rad(-p.rot3.0),
            ..p
        };
        for (name, beam, up) in labs() {
            assert!(worst_case(&g, case, &p, beam, up) <= TOL);
            let w = worst_case(&g, case, &flipped, beam, up);
            assert!(w > 1e-3, "{name}: rot3 sign not seen ({w} m)");
        }
        n += 1;
    }
    assert_eq!(n, 4 * 3, "rot3, all and all_large in every orientation");
}
