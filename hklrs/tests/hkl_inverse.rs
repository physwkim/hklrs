//! hkl 5.1.9 inverse golden (golden/hkl_inverse.py), E4CV and E4CH.
//!
//! Per case: the mode writes hkl's axes; every hkl solution is in the hklrs
//! set within MATCH rad; every hklrs solution passes the independent check
//! (extras beyond hkl's are counted); |Q| > 4π/λ is `Unreachable`; where hkl
//! fails on a reachable target, what hklrs finds is counted and checked the
//! same way.

mod common;
mod inverse_check;

use hklrs::inverse::e4c;
use hklrs::nalgebra::Vector3;
use hklrs::presets::preset;
use hklrs::{Geometry, InverseError, InverseRequest, Mode, Target, Wavelength};
use inverse_check::{largest, wrap};
use serde_json::Value;

/// hkl's own residual leaves its angles within ~1e-7 rad of the exact
/// solution.
const MATCH: f64 = 1e-6;

fn angular_distance(a: &[f64], b: &[f64]) -> f64 {
    largest(a.iter().zip(b).map(|(x, y)| wrap(x - y).abs()))
}

fn mode<'g>(geo: &'g Geometry, case: &Value) -> Mode<'g> {
    let names = case["parameter_names"].as_array().unwrap();
    let param = |n: &str| {
        let i = names.iter().position(|p| p == n).unwrap();
        common::f(&case["parameters"][i])
    };
    let hkl2 = || Vector3::new(param("h2"), param("k2"), param("l2"));
    match case["mode"].as_str().unwrap() {
        "bissector" => e4c::bissector(geo),
        "constant_omega" => e4c::constant_omega(geo),
        "constant_chi" => e4c::constant_chi(geo),
        "constant_phi" => e4c::constant_phi(geo),
        "double_diffraction" => e4c::double_diffraction(geo, hkl2()),
        "psi_constant" => e4c::psi_constant(geo, hkl2(), param("psi").to_radians()),
        m => panic!("mode {m}"),
    }
    .unwrap()
}

/// Axis values, geometry order, of a golden `{name: canonical value}` object.
fn axes(geo: &Geometry, named: &Value) -> Vec<f64> {
    let user: Vec<f64> = geo
        .axes()
        .iter()
        .map(|a| common::canonical_to_user(a, common::f(&named[a.name()])))
        .collect();
    geo.values_from_degrees(&user).unwrap()
}

#[derive(Default)]
struct Counts {
    cases: usize,
    unreachable: usize,
    hkl_solutions: usize,
    hklrs_solutions: usize,
    extras: usize,
    hkl_failed: usize,
    hkl_failed_solved: usize,
    hkl_failed_solutions: usize,
}

fn run(geometry: &str, mode_name: &str) {
    let golden = common::load("hkl_inverse.json");
    let geo = preset(geometry).unwrap();
    let mut n = Counts::default();
    let mut failures = Vec::new();
    for (index, case) in golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, c)| c["geometry"] == geometry && c["mode"] == mode_name)
    {
        n.cases += 1;
        let mode = mode(&geo, case);
        let written: Vec<&str> = mode
            .written()
            .iter()
            .map(|&i| geo.axes()[i].name())
            .collect();
        assert_eq!(
            case["writable_axes"],
            serde_json::json!(written),
            "case {index}"
        );
        let current = axes(&geo, &case["start_axes"]);
        let hkl = common::vec3(&case["target_hkl"]);
        let ub = common::mat(&case["UB"]);
        let wavelength = Wavelength::new(common::f(&case["wavelength"])).unwrap();
        let result = InverseRequest {
            mode: &mode,
            ub,
            wavelength,
            target: Target::Hkl(hkl),
            current: &current,
        }
        .solve();
        let mut fail = |what: String| failures.push(format!("case {index}: {what}"));
        if case["unreachable_by_q"].as_bool().unwrap() {
            n.unreachable += 1;
            if !matches!(result, Err(InverseError::Unreachable { .. })) {
                fail(format!("|Q| > 4π/λ gave {result:?}"));
            }
            continue;
        }
        let ours = match result {
            Ok(s) => s,
            Err(InverseError::NoSolution) => Vec::new(),
            Err(e) => {
                fail(format!("{e}"));
                continue;
            }
        };
        for s in &ours {
            if let Err(e) = inverse_check::check(&mode, &ub, wavelength, &hkl, &current, s) {
                fail(e);
            }
        }
        n.hklrs_solutions += ours.len();
        let theirs: Vec<Vec<f64>> = case["solutions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| axes(&geo, &s["axes"]).into_iter().map(wrap).collect())
            .collect();
        n.hkl_solutions += theirs.len();
        for t in &theirs {
            let nearest = ours
                .iter()
                .map(|s| angular_distance(s, t))
                .fold(f64::INFINITY, f64::min);
            if nearest > MATCH {
                fail(format!("hkl solution {t:?} not found, nearest {nearest:e}"));
            }
        }
        n.extras += ours
            .iter()
            .filter(|s| !theirs.iter().any(|t| angular_distance(s, t) <= MATCH))
            .count();
        if case.get("error").is_some() {
            n.hkl_failed += 1;
            n.hkl_failed_solved += usize::from(!ours.is_empty());
            n.hkl_failed_solutions += ours.len();
        }
    }
    println!(
        "{geometry} {mode_name}: {} cases, {} unreachable; hkl {} solutions, all found; \
         hklrs {} solutions, {} beyond hkl's; hkl failed {} reachable cases, hklrs solved {} \
         of them with {} solutions",
        n.cases,
        n.unreachable,
        n.hkl_solutions,
        n.hklrs_solutions,
        n.extras,
        n.hkl_failed,
        n.hkl_failed_solved,
        n.hkl_failed_solutions,
    );
    assert!(n.cases > 0, "no {geometry} {mode_name} cases");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

macro_rules! golden {
    ($($test:ident: $geometry:literal $mode:literal,)*) => {
        $(#[test]
        fn $test() {
            run($geometry, $mode);
        })*
    };
}

golden! {
    e4cv_bissector: "E4CV" "bissector",
    e4cv_constant_omega: "E4CV" "constant_omega",
    e4cv_constant_chi: "E4CV" "constant_chi",
    e4cv_constant_phi: "E4CV" "constant_phi",
    e4cv_double_diffraction: "E4CV" "double_diffraction",
    e4cv_psi_constant: "E4CV" "psi_constant",
    e4ch_bissector: "E4CH" "bissector",
    e4ch_constant_omega: "E4CH" "constant_omega",
    e4ch_constant_chi: "E4CH" "constant_chi",
    e4ch_constant_phi: "E4CH" "constant_phi",
    e4ch_double_diffraction: "E4CH" "double_diffraction",
    e4ch_psi_constant: "E4CH" "psi_constant",
}
