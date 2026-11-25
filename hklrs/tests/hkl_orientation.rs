//! Busing-Levy orientation parity: golden/hkl_orientation.json (hkl 5.1.9
//! `compute_UB_busing_levy` on two reflections its own solver found, plus the
//! hklpy2 Si (400)/(040) case). `UB_reference` is the generator's numpy UB
//! from the same angles with the correct B.

mod common;
mod lattice_check;

use common::{canonical_to_user, f, load, mat, vec3};
use hklrs::nalgebra::{Matrix3, Point3};
use hklrs::{Geometry, Orientation, Reflection, Sample, Wavelength, presets};
use lattice_check::{hkl_b_correct, lattice};
use serde_json::Value;

/// hkl's solver stops at this residual, so U recovered from its angles is
/// only this close to the U the angles were solved for.
const SOLVER: f64 = 1e-6;

fn reflection(geo: &Geometry, r: &Value, axes_key: &str, to_user: bool) -> Reflection {
    let user: Vec<f64> = geo
        .axes()
        .iter()
        .map(|a| {
            let v = f(&r[axes_key][a.name()]);
            if to_user { canonical_to_user(a, v) } else { v }
        })
        .collect();
    Reflection::measure(
        vec3(&r["hkl"]),
        geo,
        &geo.values_from_degrees(&user).unwrap(),
        Wavelength::new(f(&r["wavelength"])).unwrap(),
        &Point3::from(geo.beam_direction().into_inner()),
    )
    .unwrap()
}

fn max_diff(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    (a - b).abs().max()
}

#[test]
fn busing_levy_matches_hkl() {
    let g = load("hkl_orientation.json");
    let (mut ok, mut hkl12) = (0, 0);
    for case in g["cases"].as_array().unwrap() {
        let name = case["geometry"].as_str().unwrap();
        let ctx = format!("{name} {} λ={}", case["sample"], case["wavelength"]);
        if let Some(e) = case.get("error") {
            // TwoC has no hkl engine
            assert_eq!(name, "TwoC", "{ctx}: unexpected error {e}");
            continue;
        }
        let geo = presets::preset(name).unwrap();
        let r: Vec<Reflection> = case["reflections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| reflection(&geo, r, "axes_canonical_deg_mm", true))
            .collect();
        let lattice = lattice(&case["lattice"]);
        let sample = Sample::new(
            "golden",
            lattice,
            Orientation::busing_levy(r[0].clone(), r[1].clone()).unwrap(),
        );
        let ub = sample.ub();

        let d = max_diff(&ub, &mat(&case["UB_reference"]));
        assert!(d < 1e-12, "{ctx}: UB vs reference {d}");
        let d = max_diff(&ub, &mat(&case["UB"]));
        if hkl_b_correct(&lattice) {
            assert!(d < 1e-12, "{ctx}: UB vs hkl {d}");
            // where hkl's B is wrong its solver found the angles for
            // U_true·B_hkl, so U_true is not recoverable with the right B
            let u = sample.u().to_rotation_matrix().into_inner();
            let d = max_diff(&u, &mat(&case["U_true"]));
            assert!(d < SOLVER, "{ctx}: U vs U_true {d}");
        } else {
            // upstream-defects.md HKL-12: fails when hkl's B is fixed
            assert!(d > 1e-4, "{ctx}: hkl's UB now agrees ({d})");
            hkl12 += 1;
        }
        ok += 1;
    }
    assert_eq!((ok, hkl12), (168, 42));
}

#[test]
fn busing_levy_matches_hklpy2_si() {
    let h = &load("hkl_orientation.json")["hklpy2"];
    let geo = presets::preset(h["geometry"].as_str().unwrap()).unwrap();
    let r: Vec<Reflection> = h["reflections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| reflection(&geo, r, "axes", false))
        .collect();
    let sample = Sample::new(
        "Si",
        lattice(&h["lattice"]),
        Orientation::busing_levy(r[0].clone(), r[1].clone()).unwrap(),
    );
    let d = max_diff(&sample.ub(), &mat(&h["UB_reference"]));
    assert!(d < 1e-12, "UB vs reference {d}");
    // hklpy2 reports UB rounded to 12 decimals
    let d = max_diff(&sample.ub(), &mat(&h["UB"]));
    assert!(d < 1e-11, "UB vs hklpy2 {d}");
}
