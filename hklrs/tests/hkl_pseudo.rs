//! hkl 5.1.9 read-only pseudos (golden/hkl_pseudo.py): q (E4CV, E4CH, K4CV,
//! MARS), q2 and qper_qpar (every preset that has them), at hkl's own axes.
//!
//! hkl's kf ignores detector translations; where a preset has one, hklrs's
//! full pose can differ, so each preset's agreement is asserted, not assumed.
//! HKL-11: hkl's `acos` angle returns 0 for 2θ = 1e-6° (true |Q| 7.12e-8), so
//! there the independent value in the golden is the reference, not hkl's.

use hklrs::Axis;
use hklrs::nalgebra::Vector3;
use hklrs::presets::preset;
use hklrs::pseudo::{q_at, q2_at, qper_qpar_at, tth2_at};
use hklrs::{PixelDetector, Wavelength};
use serde_json::Value;

/// hkl's own `acos` loses angles below this 2θ, rad (HKL-11: 1e-6 rad gives
/// 6.7e-5 relative error, 1e-8 gives 0).
fn load(file: &str) -> Value {
    let path = format!("{}/../golden/{file}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// A golden canonical value (degrees, millimetres) as the user value
/// `Geometry::values_from_degrees` takes (degrees, metres).
fn canonical_to_user(axis: &Axis, v: f64) -> f64 {
    if axis.is_rotation() { v } else { v * 1e-3 }
}

const HKL11_TTH: f64 = 1e-7;
/// Same, for the 2θ value itself (degrees, absolute 1e-9 tolerance).
const HKL11_TTH2: f64 = 1e-4;
const TOL: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL * (1.0 + b.abs())
}

fn wrap_deg(a: f64) -> f64 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

#[test]
fn pseudos_match_hkl() {
    let golden = load("hkl_pseudo.json");
    let mut seen = std::collections::BTreeMap::<(String, String), usize>::new();
    let mut hkl11 = 0;
    for (i, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let name = case["geometry"].as_str().unwrap();
        let engine = case["engine"].as_str().unwrap();
        let geo = preset(name).unwrap();
        let user: Vec<f64> = geo
            .axes()
            .iter()
            .map(|a| canonical_to_user(a, f(&case["axes"][a.name()])))
            .collect();
        let values = geo.values_from_degrees(&user).unwrap();
        let wl = Wavelength::new(f(&case["wavelength"])).unwrap();
        let point = PixelDetector::point(geo.beam_direction(), 1.0)
            .unwrap()
            .pixel(0, 0)
            .unwrap();
        let tth = f(&case["numpy"]["tth"]).to_radians();
        let ctx = format!("{name} {engine} #{i}");
        let hkl = |k: usize| f(&case["pseudo"][k]);
        // hkl's value, or the independent one where hkl's own angle is wrong
        let (q_ref, hkl11_case) = if tth < HKL11_TTH {
            (f(&case["numpy"]["q"]), true)
        } else {
            (hkl(0), false)
        };
        hkl11 += usize::from(hkl11_case && engine == "q");
        match engine {
            "q" => {
                let want = if hkl11_case {
                    f(&case["numpy"]["q_signed"])
                } else {
                    q_ref
                };
                let got = q_at(&geo, &values, wl, &point).unwrap();
                assert!(close(got, want), "{ctx}: q {got} vs {want}");
            }
            "q2" => {
                let (q, alpha) = q2_at(&geo, &values, wl, &point).unwrap();
                assert!(close(q, q_ref), "{ctx}: q {q} vs {q_ref}");
                let d = wrap_deg(alpha.0.to_degrees() - hkl(1));
                // alpha of a k_f on the x axis is atan2 of rounding noise
                if f(&case["numpy"]["q"]) > 1e-3 {
                    assert!(d.abs() < 1e-7, "{ctx}: alpha differs by {d} deg");
                }
            }
            "tth2" => {
                let (t, alpha) = tth2_at(&geo, &values, wl, &point).unwrap();
                // hkl's acos 2θ is off by ~1e-16/2θ rad; below 1e-4 rad the
                // independent atan2 value is the reference (HKL-11)
                let want = if tth < HKL11_TTH2 {
                    f(&case["numpy"]["tth"])
                } else {
                    hkl(0)
                };
                assert!(
                    close(t.0.to_degrees(), want),
                    "{ctx}: tth {} vs {want}",
                    t.0.to_degrees()
                );
                let d = wrap_deg(alpha.0.to_degrees() - hkl(1));
                if tth > 1e-9 {
                    assert!(d.abs() < 1e-7, "{ctx}: alpha differs by {d} deg");
                }
            }
            "qper_qpar" => {
                let n = &case["parameters"];
                let n = Vector3::new(f(&n[0]), f(&n[1]), f(&n[2]));
                let (qper, qpar) = qper_qpar_at(&geo, &values, wl, &point, &n).unwrap();
                assert!(close(qper, hkl(0)), "{ctx}: qper {qper} vs {}", hkl(0));
                // qpar's sign is rounding noise when Q·(ki×n) = 0; magnitude always
                let noise = f(&case["numpy"]["qpar_sign_dot"]).abs() < 1e-9;
                let ok = if noise {
                    close(qpar.abs(), hkl(1).abs())
                } else {
                    close(qpar, hkl(1))
                };
                assert!(ok, "{ctx}: qpar {qpar} vs {}", hkl(1));
            }
            e => panic!("engine {e}"),
        }
        *seen
            .entry((name.to_owned(), engine.to_owned()))
            .or_default() += 1;
    }
    assert_eq!(seen.len(), 4 + 11 * 3, "{seen:?}");
    // per q preset: the zero position and 2θ = 1e-6° (the HKL-11 case)
    assert_eq!(hkl11, 8);
}

/// The four presets with engine `q` and no other: sanity that the golden
/// covers the stated presets.
#[test]
fn golden_covers_the_q_presets() {
    let golden = load("hkl_pseudo.json");
    let names: std::collections::BTreeSet<&str> = golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["engine"] == "q")
        .map(|c| c["geometry"].as_str().unwrap())
        .collect();
    assert_eq!(
        names.into_iter().collect::<Vec<_>>(),
        ["E4CH", "E4CV", "K4CV", "SOLEIL MARS"]
    );
    let _: &Value = &golden;
}
