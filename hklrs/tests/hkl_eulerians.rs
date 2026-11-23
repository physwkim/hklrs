//! hkl 5.1.9 "eulerians" golden (golden/hkl_eulerians.py), K4CV and K6C.
//!
//! Forward: hklrs's (omega, chi, phi) equal hkl's modulo 2π, and the
//! eulerian preset (E4CV, E6C) at them has the kappa preset's sample
//! rotation. Inverse: hklrs's kappa position and its twin are hkl's
//! solutions modulo 2π, other axes held; chi past 2α is an error.

// pub: this test reads only some of the shared golden readers.
pub mod common;

use std::f64::consts::TAU;

use hklrs::inverse::kappa::{Branch, KappaArm};
use hklrs::presets::{KAPPA_ALPHA_DEG, preset};
use hklrs::{Deg, Geometry, InverseError};
use serde_json::Value;

/// As hkl_inverse.rs. Near |chi| = 2α kappa is ill-conditioned in chi and
/// hkl and hklrs each carry ~1e-8 rad there.
const MATCH: f64 = 1e-6;
/// hklrs's own round trip: its kappa position has the eulerians' rotation.
const ROTATION: f64 = 1e-12;

fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r > TAU / 2.0 { r - TAU } else { r }
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| wrap(x - y).abs() <= tol)
}

fn arm() -> KappaArm {
    KappaArm::new(Deg(KAPPA_ALPHA_DEG).into()).unwrap()
}

fn branch(case: &Value) -> Branch {
    match common::f(&case["solutions"]) {
        1.0 => Branch::Same,
        0.0 => Branch::Opposite,
        s => panic!("solutions {s}"),
    }
}

fn rad(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| common::f(x).to_radians())
        .collect()
}

/// Every axis of `geo` from a golden `{name: deg}` object, radians.
fn axes(geo: &Geometry, named: &Value) -> Vec<f64> {
    geo.axes()
        .iter()
        .map(|a| common::f(&named[a.name()]).to_radians())
        .collect()
}

const KAPPA: [&str; 3] = ["komega", "kappa", "kphi"];

struct Pair {
    kappa: Geometry,
    eulerian: Geometry,
    /// Indices of komega, kappa, kphi; omega, chi, phi sit at the same ones.
    at: [usize; 3],
}

impl Pair {
    fn new(geometry: &str) -> Self {
        let kappa = preset(geometry).unwrap();
        let eulerian = preset(&geometry.replace('K', "E")).unwrap();
        let at = KAPPA.map(|n| kappa.axis_index(n).unwrap());
        for (i, n) in at.iter().zip(["omega", "chi", "phi"]) {
            assert_eq!(eulerian.axis_index(n).unwrap(), *i);
        }
        Self {
            kappa,
            eulerian,
            at,
        }
    }

    fn triple(&self, values: &[f64]) -> [f64; 3] {
        self.at.map(|i| values[i])
    }

    fn with(&self, values: &[f64], triple: [f64; 3]) -> Vec<f64> {
        let mut v = values.to_vec();
        for (i, x) in self.at.iter().zip(triple) {
            v[*i] = x;
        }
        v
    }

    /// Largest element difference of the sample rotations of the kappa
    /// preset at `values` and the eulerian preset at `values` with
    /// `eulerian` written in.
    fn rotation_gap(&self, values: &[f64], eulerian: [f64; 3]) -> f64 {
        let r = |g: &Geometry, v: &[f64]| {
            g.poses(v)
                .unwrap()
                .sample
                .rotation
                .to_rotation_matrix()
                .into_inner()
        };
        (r(&self.kappa, values) - r(&self.eulerian, &self.with(values, eulerian))).amax()
    }
}

fn cases<'a>(golden: &'a Value, geometry: &'a str, direction: &'a str) -> Vec<&'a Value> {
    let c: Vec<&Value> = golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["geometry"] == geometry && c["direction"] == direction)
        .collect();
    assert!(!c.is_empty(), "no {geometry} {direction} cases");
    c
}

fn forward(geometry: &str) {
    let golden = common::load("hkl_eulerians.json");
    let (pair, arm) = (Pair::new(geometry), arm());
    let mut failures = Vec::new();
    let all = cases(&golden, geometry, "forward");
    for (n, case) in all.iter().enumerate() {
        let values = axes(&pair.kappa, &case["axes"]);
        let ours = arm.to_eulerian(pair.triple(&values), branch(case));
        let theirs = rad(&case["eulerians"]);
        if !close(&ours, &theirs, MATCH) {
            failures.push(format!("case {n}: hklrs {ours:?}, hkl {theirs:?}"));
        }
        let gap = pair.rotation_gap(&values, ours);
        if gap > ROTATION {
            failures.push(format!("case {n}: eulerian rotation off by {gap:e}"));
        }
    }
    println!("{geometry} forward: {} cases", all.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[derive(Default)]
struct Counts {
    solved: usize,
    out_of_reach: usize,
    hkl19: usize,
}

fn inverse(geometry: &str) {
    let golden = common::load("hkl_eulerians.json");
    let (pair, arm) = (Pair::new(geometry), arm());
    let mut n = Counts::default();
    let mut failures = Vec::new();
    for (index, case) in cases(&golden, geometry, "inverse").iter().enumerate() {
        let mut fail = |what: String| failures.push(format!("case {index}: {what}"));
        let e = rad(&case["eulerians"]);
        let eulerian = [e[0], e[1], e[2]];
        let start = axes(&pair.kappa, &case["start_axes"]);
        let result = arm.to_kappa(eulerian, branch(case));
        let reachable = wrap(eulerian[1]).abs() <= arm.reach();
        match (case.get("error"), result) {
            (Some(text), Err(InverseError::ChiOutOfReach { chi, reach })) => {
                n.out_of_reach += 1;
                // hkl says "0° < chi < 50°" for a limit of 2α = 100° (HKL-9).
                assert!(text.as_str().unwrap().contains("chi < 50°"), "case {index}");
                if reachable || reach != arm.reach() || !close(&[chi], &[eulerian[1]], MATCH) {
                    fail(format!("chi {chi} reported against reach {reach}"));
                }
            }
            // hkl rejects chi outside (−180°, 180°] whose residue is
            // within ±2α (HKL-19).
            (Some(_), Ok(k)) if reachable && eulerian[1].abs() > TAU / 2.0 => {
                n.hkl19 += 1;
                let gap = pair.rotation_gap(&pair.with(&start, k), eulerian);
                if gap > ROTATION {
                    fail(format!("HKL-19 solution {k:?} off by {gap:e}"));
                }
            }
            (None, Ok(k)) => {
                n.solved += 1;
                let ours = [k, arm.twin(k)].map(|t| pair.with(&start, t));
                let theirs: Vec<Vec<f64>> = case["solution_axes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| axes(&pair.kappa, s))
                    .collect();
                for t in &theirs {
                    if !ours.iter().any(|o| close(o, t, MATCH)) {
                        fail(format!("hkl solution {t:?} not in {ours:?}"));
                    }
                }
                for o in &ours {
                    if !theirs.iter().any(|t| close(o, t, MATCH)) {
                        fail(format!("hklrs solution {o:?} not in hkl's {theirs:?}"));
                    }
                    let gap = pair.rotation_gap(o, eulerian);
                    if gap > ROTATION {
                        fail(format!("solution {o:?} off by {gap:e}"));
                    }
                }
            }
            (hkl, ours) => fail(format!("hkl {hkl:?}, hklrs {ours:?}")),
        }
    }
    println!(
        "{geometry} inverse: {} solved as hkl, {} past 2α, {} solved where hkl rejects (HKL-19)",
        n.solved, n.out_of_reach, n.hkl19
    );
    assert!(n.solved > 0 && n.out_of_reach > 0 && n.hkl19 > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn k4cv_forward() {
    forward("K4CV");
}

#[test]
fn k6c_forward() {
    forward("K6C");
}

#[test]
fn k4cv_inverse() {
    inverse("K4CV");
}

#[test]
fn k6c_inverse() {
    inverse("K6C");
}
