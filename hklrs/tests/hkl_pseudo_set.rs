//! hkl 5.1.9 pseudo-axis engine solves (golden/hkl_pseudo_set.py) through
//! hklrs's modes for them: hkl's psi engine as `e4c::psi` and its kin, with
//! hkl's hkl0 as the request's `Target::Hkl`; q, q2, tth2 and qper_qpar as
//! `inverse::pseudo`'s under `Target::Detector`; eulerians under
//! `Target::Eulerians`.
//!
//! Every hkl solution is in the hklrs set within MATCH rad, and every hklrs
//! solution holds the other axes and reads back to the target through
//! hklrs's forward pseudo. MED2+3 turns its slits after every engine, as its
//! hkl solution list does; hkl's coarser slits fit is matched after one
//! Newton step. A kappa twin hkl appends that moves a held axis (HKL-16) is
//! the twin of an hklrs solution and is counted. A target outside (−π, π] gives the set of its
//! equivalent inside. hkl fails there: psi and alpha (HKL-38), chi
//! (HKL-19); and psi set before initialisation has no hkl solution at hkl0
//! (HKL-39).

mod common;

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use hklrs::inverse::kappa::{Branch, KappaArm};
use hklrs::inverse::{self, aps_04id_polar, e4c, e6c, k4c, k6c, petra3_p23_6c, soleil_sixs_med};
use hklrs::nalgebra::{Matrix3, Point3, Vector3};
use hklrs::presets::{KAPPA_ALPHA_DEG, preset};
use hklrs::{
    Deg, Geometry, InverseError, InverseRequest, Mode, Target, Wavelength, pseudo, scattering,
};
use serde_json::Value;

/// As hkl_inverse.rs: hkl's own residual leaves its angles within ~1e-7 rad.
const MATCH: f64 = 1e-6;
/// hklrs's read-back miss allowed: rad, Å⁻¹ or Miller units.
const READ: f64 = 1e-9;
/// Axis values closer than this (rad, every axis) are one solution.
const SAME: f64 = 1e-9;

fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r > PI { r - TAU } else { r }
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| wrap(x - y).abs() <= tol)
}

fn same_sets(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    let within =
        |x: &[Vec<f64>], y: &[Vec<f64>]| x.iter().all(|x| y.iter().any(|y| close(x, y, SAME)));
    a.len() == b.len() && within(a, b) && within(b, a)
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

fn list(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(common::f).collect()
}

/// Where the angle a target carries in degrees, if any, sits in `target`.
fn seam_index(engine: &str) -> Option<usize> {
    match engine {
        "psi" => Some(0),
        "q2" | "tth2" | "eulerians" => Some(1),
        _ => None,
    }
}

/// The mode and target of `case`; the target values, rad or Å⁻¹, are
/// what [`read_back`] gives at a solution.
fn request<'g>(geo: &'g Geometry, case: &Value) -> (Mode<'g>, Target, Vec<f64>) {
    let p = list(&case["parameters"]);
    let t = list(&case["target"]);
    let rad = |i: usize| t[i].to_radians();
    let vector = || Vector3::new(p[0], p[1], p[2]);
    let geometry = case["geometry"].as_str().unwrap();
    let (mode, target, read) = match case["engine"].as_str().unwrap() {
        "psi" => {
            let (reference, psi) = (vector(), rad(0));
            let mode = match geometry {
                "E4CV" | "E4CH" | "SOLEIL MARS" => e4c::psi(geo, reference, psi),
                "K4CV" => k4c::psi(geo, reference, psi),
                "E6C" => e6c::psi_vertical(geo, reference, psi),
                "PETRA3 P23 6C" => petra3_p23_6c::psi_vertical(geo, reference, psi),
                "K6C" => k6c::psi_vertical(geo, reference, psi),
                "SOLEIL SIRIUS KAPPA" => k6c::psi_vertical_soleil_sirius_kappa(geo, reference, psi),
                "APS POLAR" => aps_04id_polar::psi_vertical(geo, reference, psi),
                g => panic!("psi on {g}"),
            };
            let hkl0 = common::vec3(&case["hkl0"]);
            (mode, Target::Hkl(hkl0), vec![psi])
        }
        "q" => (inverse::pseudo::q(geo, t[0]), Target::Detector, t),
        "q2" => (
            inverse::pseudo::q2(geo, t[0], rad(1)),
            Target::Detector,
            vec![t[0], rad(1)],
        ),
        "tth2" => (
            inverse::pseudo::tth2(geo, rad(0), rad(1)),
            Target::Detector,
            vec![rad(0), rad(1)],
        ),
        "qper_qpar" => (
            inverse::pseudo::qper_qpar(geo, vector(), t[0], t[1]),
            Target::Detector,
            t,
        ),
        "eulerians" => {
            let e = [rad(0), rad(1), rad(2)];
            (
                inverse::pseudo::eulerians(geo),
                Target::Eulerians(e),
                e.to_vec(),
            )
        }
        e => panic!("engine {e}"),
    };
    // MED2+3's solution list fits the slits after every engine.
    let mode = match geometry {
        "SOLEIL SIXS MED2+3" => mode.and_then(soleil_sixs_med::slits),
        _ => mode,
    };
    (mode.unwrap(), target, read)
}

/// `values` after one Newton step on the slits equation over the slits
/// axis: hkl stops its slits fit at |surface·normal| < 1e-6, coarser than
/// MATCH where the equation turns slowly. `None` without slits.
fn slits_newton(mode: &Mode, values: &[f64]) -> Option<Vec<f64>> {
    let slits = mode.slits()?;
    let at = |v: f64| {
        let mut x = values.to_vec();
        x[slits.axis()] = v;
        x
    };
    let f = |v: f64| {
        let p = mode.geometry().poses(&at(v)).unwrap();
        (p.sample.rotation * slits.surface()).dot(&(p.detector.rotation * slits.normal()))
    };
    let (v, h) = (values[slits.axis()], 1e-6);
    Some(at(wrap(v - f(v) * 2.0 * h / (f(v + h) - f(v - h)))))
}

/// The written axes then the slits axis: what the solve moves.
fn moved(mode: &Mode) -> Vec<usize> {
    let slits = mode.slits().map(|s| s.axis());
    mode.written().iter().copied().chain(slits).collect()
}

/// The engine's pseudo values at `values` through hklrs's forward pseudo,
/// as [`request`]'s target values, and for psi the miss of hkl0 too; for
/// eulerians on the kappa image whose (omega, chi, phi) is nearer `want`.
fn read_back(
    mode: &Mode,
    case: &Value,
    ub: &Matrix3<f64>,
    wavelength: Wavelength,
    want: &[f64],
    values: &[f64],
) -> Vec<f64> {
    let geo = mode.geometry();
    let point = Point3::from(geo.beam_direction().into_inner());
    let p = list(&case["parameters"]);
    match case["engine"].as_str().unwrap() {
        "psi" => {
            let reference = Vector3::new(p[0], p[1], p[2]);
            let s = scattering(geo, values, wavelength, &point).unwrap();
            let hkl = ub.lu().solve(&s.q_sample).unwrap();
            let miss = (hkl - common::vec3(&case["hkl0"])).amax();
            let psi = inverse::psi(geo, ub, &reference, values).unwrap();
            // the miss reads back as hkl0 reached: the target's own value
            vec![if miss <= READ { psi } else { f64::NAN }]
        }
        "q" => vec![pseudo::q_at(geo, values, wavelength, &point).unwrap()],
        "q2" => {
            let (q, alpha) = pseudo::q2_at(geo, values, wavelength, &point).unwrap();
            vec![q, alpha.0]
        }
        "tth2" => {
            let (tth, alpha) = pseudo::tth2_at(geo, values, wavelength, &point).unwrap();
            vec![tth.0, alpha.0]
        }
        "qper_qpar" => {
            let surface = Vector3::new(p[0], p[1], p[2]);
            let (qper, qpar) =
                pseudo::qper_qpar_at(geo, values, wavelength, &point, &surface).unwrap();
            vec![qper, qpar]
        }
        "eulerians" => {
            let kappa = mode.kappa().unwrap();
            let at = kappa.axes().map(|i| values[i]);
            [Branch::Same, Branch::Opposite]
                .map(|b| kappa.arm().to_eulerian(at, b).to_vec())
                .into_iter()
                .min_by(|a, b| miss(a, want).total_cmp(&miss(b, want)))
                .unwrap()
        }
        e => panic!("engine {e}"),
    }
}

/// Largest difference of pseudo values, angles modulo 2π; NaN when one is.
fn miss(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| wrap(x - y).abs())
        .fold(0.0, |m, d| if d > m || d.is_nan() { d } else { m })
}

/// `m <= tol`, false for a NaN miss.
fn within(m: f64, tol: f64) -> bool {
    m <= tol
}

/// One golden case through hklrs.
struct Run {
    index: usize,
    engine: String,
    /// hkl's solutions, every axis, rad.
    theirs: Vec<Vec<f64>>,
    hkl_failed: bool,
    ours: Result<Vec<Vec<f64>>, InverseError>,
}

#[derive(Default)]
struct Counts {
    cases: usize,
    hkl_solutions: usize,
    hklrs_solutions: usize,
    extras: usize,
    /// hkl failed and hklrs solved, away from a seam.
    hkl_misses: usize,
    /// hkl solutions in the set after [`slits_newton`].
    slits_newton: usize,
    /// hkl's kappa twins of an hklrs solution that move held axes (HKL-16),
    /// by preset.
    held_twins: BTreeMap<String, usize>,
}

/// `values` with komega, kappa, kphi at their twin, as hkl's solution list
/// appends it whatever the mode writes (HKL-16); `None` without a kappa arm.
fn twin(geo: &Geometry, values: &[f64]) -> Option<Vec<f64>> {
    let arm = KappaArm::new(Deg(KAPPA_ALPHA_DEG).into()).unwrap();
    let at = ["komega", "kappa", "kphi"].map(|a| geo.axis_index(a).ok());
    let at = at.iter().copied().collect::<Option<Vec<_>>>()?;
    let mut out = values.to_vec();
    for (&i, v) in at.iter().zip(arm.twin([0, 1, 2].map(|k| values[at[k]]))) {
        out[i] = v;
    }
    Some(out)
}

#[test]
fn pseudo_engine_sets_match_hkl() {
    let golden = common::load("hkl_pseudo_set.json");
    let wavelength = Wavelength::new(common::f(&golden["wavelength"])).unwrap();
    let mut failures = Vec::new();
    let mut runs = Vec::new();
    let mut counts: BTreeMap<String, Counts> = BTreeMap::new();
    let mut hkl39 = 0;
    for (index, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let name = case["geometry"].as_str().unwrap();
        let engine = case["engine"].as_str().unwrap().to_owned();
        let geo = preset(name).unwrap();
        let ub = common::mat(&golden["geometries"][name]["UB"]);
        let (mode, target, want) = request(&geo, case);
        let mut fail =
            |what: String| failures.push(format!("case {index} {name} {engine}: {what}"));
        if !case["initialized"].as_bool().unwrap() {
            // HKL-39: none of hkl's solutions holds Q at hkl0.
            if case["solutions_at_hkl0"] != 0 {
                fail("hkl solved at hkl0 before initialisation".into());
            }
        } else {
            let written: Vec<&str> = mode
                .written()
                .iter()
                .map(|&i| geo.axes()[i].name())
                .collect();
            if case["written"] != serde_json::json!(written) {
                fail(format!("writes {written:?}, hkl {}", case["written"]));
            }
        }
        let current = axes(&geo, &case["start_axes"]);
        let moves = moved(&mode);
        let ours = InverseRequest {
            mode: &mode,
            ub,
            wavelength,
            target,
            current: &current,
        }
        .solve();
        let theirs: Vec<Vec<f64>> = case
            .get("solutions")
            .map_or(&Vec::new(), |s| s.as_array().unwrap())
            .iter()
            .map(|s| axes(&geo, s))
            .collect();
        let n = counts.entry(engine.clone()).or_default();
        n.cases += 1;
        n.hkl_solutions += theirs.len();
        match &ours {
            Ok(set) => {
                n.hklrs_solutions += set.len();
                for s in set {
                    let held = (0..s.len())
                        .filter(|i| !moves.contains(i))
                        .all(|i| s[i].to_bits() == current[i].to_bits());
                    let got = read_back(&mode, case, &ub, wavelength, &want, s);
                    let m = miss(&got, &want);
                    if !held || !within(m, READ) {
                        fail(format!(
                            "{s:?} reads back {got:?} for {want:?}, held {held}"
                        ));
                    }
                }
                n.extras += set
                    .iter()
                    .filter(|s| !theirs.iter().any(|t| close(s, t, MATCH)))
                    .count();
                for t in &theirs {
                    let shifted = (0..t.len())
                        .any(|i| !moves.contains(&i) && wrap(t[i] - current[i]).abs() > MATCH);
                    let twin_of = |s: &Vec<f64>| twin(&geo, s).is_some_and(|w| close(&w, t, MATCH));
                    let found = |t: &[f64]| set.iter().any(|s| close(s, t, MATCH));
                    if shifted && set.iter().any(twin_of) {
                        *n.held_twins.entry(name.to_owned()).or_default() += 1;
                    } else if found(t) {
                    } else if slits_newton(&mode, t).is_some_and(|p| found(&p)) {
                        n.slits_newton += 1;
                    } else {
                        fail(format!("hkl solution {t:?} not in {set:?}"));
                    }
                }
                if !case["initialized"].as_bool().unwrap() {
                    hkl39 += usize::from(!set.is_empty());
                }
            }
            // kappa = 0 and chi = 0: komega + kphi alone is fixed, as omega + phi.
            Err(InverseError::Degenerate) if engine == "eulerians" && wrap(want[1]) == 0.0 => {
                let [komega, _, kphi] = mode.kappa().unwrap().axes();
                let sum = |e: &[f64]| [e[1], e[0] + e[2]];
                for t in &theirs {
                    let mut turned = t.clone();
                    turned[komega] += 0.3;
                    turned[kphi] -= 0.3;
                    for v in [t, &turned] {
                        let got = read_back(&mode, case, &ub, wavelength, &want, v);
                        if !within(miss(&sum(&got), &sum(&want)), MATCH) {
                            fail(format!("Degenerate, but {v:?} reads back {got:?}"));
                        }
                    }
                }
            }
            Err(InverseError::NoSolution) if theirs.is_empty() => {}
            Err(e) => fail(format!("{e:?}, hkl {theirs:?}")),
        }
        runs.push(Run {
            index,
            engine,
            hkl_failed: theirs.is_empty(),
            theirs,
            ours,
        });
    }
    // Seams: a target angle outside (−180°, 180°] against its equivalent.
    let cases = golden["cases"].as_array().unwrap();
    let mut seams: BTreeMap<(String, &str), [usize; 3]> = BTreeMap::new();
    for run in &runs {
        let case = &cases[run.index];
        let Some(k) = seam_index(&run.engine) else {
            continue;
        };
        let target = list(&case["target"]);
        let a = target[k];
        let solved = run.ours.as_ref().is_ok_and(|s| !s.is_empty());
        let defect = if run.engine == "eulerians" {
            "HKL-19"
        } else {
            "HKL-38"
        };
        if run.engine != "eulerians" && a.abs() >= 180.0 {
            let n = seams.entry((run.engine.clone(), defect)).or_default();
            n[0] += 1;
            n[1] += usize::from(run.hkl_failed && solved);
        }
        if a > -180.0 && a <= 180.0 {
            continue;
        }
        let inside = (a + 180.0).rem_euclid(360.0) - 180.0;
        let inside = if inside == -180.0 { 180.0 } else { inside };
        let equivalent = runs.iter().find(|r| {
            let c = &cases[r.index];
            let mut t = list(&c["target"]);
            let same_case = c["geometry"] == case["geometry"]
                && c["engine"] == case["engine"]
                && c["parameters"] == case["parameters"]
                && c["start_axes"] == case["start_axes"];
            same_case && t[k] == inside && {
                t[k] = a;
                t == target
            }
        });
        let Some(equivalent) = equivalent else {
            failures.push(format!("case {}: no equivalent at {inside}", run.index));
            continue;
        };
        if a.abs() > 180.0 && !run.theirs.is_empty() {
            failures.push(format!("case {}: hkl solved at {a} ({defect})", run.index));
        }
        let agree = match (&run.ours, &equivalent.ours) {
            (Ok(a), Ok(b)) => same_sets(a, b),
            (Err(a), Err(b)) => a == b,
            _ => false,
        };
        if !agree {
            failures.push(format!(
                "case {} at {a}: {:?}, at {inside} {:?}",
                run.index, run.ours, equivalent.ours
            ));
        }
        if run.engine == "eulerians" {
            let n = seams.entry((run.engine.clone(), defect)).or_default();
            n[0] += 1;
            n[1] += usize::from(solved);
        }
        seams.entry((run.engine.clone(), defect)).or_default()[2] += 1;
    }
    for run in &runs {
        let case = &cases[run.index];
        let seam =
            seam_index(&run.engine).is_some_and(|k| common::f(&case["target"][k]).abs() >= 180.0);
        if run.hkl_failed
            && !seam
            && case["initialized"].as_bool().unwrap()
            && run.ours.as_ref().is_ok_and(|s| !s.is_empty())
        {
            counts.get_mut(&run.engine).unwrap().hkl_misses += 1;
        }
    }
    for (engine, n) in &counts {
        println!(
            "{engine}: {} cases, hkl {} solutions, all in hklrs's {} ({} beyond hkl's, {} \
             after a Newton step on the slits) but kappa twins moving held axes (HKL-16) {:?}; \
             hkl failed {} cases off the seams that hklrs solves",
            n.cases,
            n.hkl_solutions,
            n.hklrs_solutions,
            n.extras,
            n.slits_newton,
            n.held_twins,
            n.hkl_misses
        );
    }
    for ((engine, defect), [at, solved, outside]) in &seams {
        println!(
            "{engine} {defect}: {at} targets at |angle| >= 180, hkl fails and hklrs solves \
             {solved}; {outside} outside (-180, 180] give their equivalent's set"
        );
    }
    let psi_cases = runs.iter().filter(|r| r.engine == "psi").count();
    println!(
        "psi HKL-39: {hkl39} of the uninitialised cases solved at hkl0 by hklrs, none by hkl ({psi_cases} psi cases)"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    for key in [
        ("psi", "HKL-38"),
        ("q2", "HKL-38"),
        ("tth2", "HKL-38"),
        ("eulerians", "HKL-19"),
    ] {
        let n = seams[&(key.0.to_owned(), key.1)];
        assert!(n[1] > 0 && n[2] > 0, "{key:?}: {n:?}");
    }
    assert_eq!(
        hkl39, 9,
        "HKL-39: every uninitialised psi case solves at hkl0"
    );
}
