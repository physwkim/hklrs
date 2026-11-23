//! diffcalc-core 059ec8a inverse golden (golden/diffcalc_inverse.py) against
//! the closed-form E6C solver `hklrs::inverse::analytic`.
//!
//! Per case the golden's UB, reference and surface (You's φ frame) enter in
//! hklrs's E6C frame through `frame()ᵀ`. Every returned position must bring
//! hkl into reflection through hklrs `scattering` and meet each constraint by
//! the golden's own definition, recomputed here from hklrs poses; every valid
//! diffcalc solution and every isolated solution of the golden's numeric
//! solve must be among them. Where the numeric solve found a continuous
//! family the answer is `Degenerate`; where it found nothing, no solution.
//! `dc_ids` mark cases where diffcalc's own answer is wrong
//! (upstream-defects.md); they are judged by the same physics.

use std::collections::BTreeMap;

use hklrs::inverse::analytic::{self, Constraints, Error};
use hklrs::nalgebra::{Matrix3, Point3, Vector3};
use hklrs::presets::preset;
use hklrs::{Geometry, Wavelength, scattering};
use serde_json::Value;

/// A diffcalc or numeric solution matches a returned one within this (rad).
const MATCH: f64 = 1e-6;
/// Largest hkl miss and constraint miss (rad, sine for arcsine angles).
const HKL: f64 = 1e-9;
const CONSTRAINT: f64 = 1e-8;

/// hklrs E6C axis order (mu, omega, chi, phi, gamma, delta) in You's names.
const ORDER: [&str; 6] = ["mu", "eta", "chi", "phi", "nu", "delta"];
const DET: [&str; 4] = ["delta", "nu", "qaz", "naz"];
const REF: [&str; 7] = [
    "alpha",
    "beta",
    "psi",
    "a_eq_b",
    "betain",
    "betaout",
    "bin_eq_bout",
];
const SAMP: [&str; 4] = ["mu", "eta", "chi", "phi"];

/// The golden file; tests/common is not used because this binary needs
/// only its readers and would leave the rest dead. Python wrote the virtual
/// ψ of the backscatter case as a bare `NaN`, read here as null.
fn load() -> Value {
    let path = format!(
        "{}/../golden/diffcalc_inverse.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect(&path);
    serde_json::from_str(&text.replace(": NaN", ": null")).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn vec3(v: &Value) -> Vector3<f64> {
    Vector3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}

fn mat(v: &Value) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| f(&v[i][j]))
}

fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(std::f64::consts::TAU);
    if r > std::f64::consts::PI {
        r - std::f64::consts::TAU
    } else {
        r
    }
}

fn distance(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| wrap(x - y).abs())
        .fold(0.0, f64::max)
}

/// The constraint family of a golden case, by the names it constrains.
fn family(case: &Value) -> &'static str {
    let names: Vec<&str> = case["constraints"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let count = |set: &[&str]| names.iter().filter(|n| set.contains(n)).count();
    let naz = names.contains(&"naz");
    if names.contains(&"bisect") {
        return if naz { "naz+bisect" } else { "det+bisect" };
    }
    match (count(&DET), count(&REF), count(&SAMP)) {
        (1, 1, 1) => "det+ref+samp",
        (1, 0, 2) if naz => "naz+2samp",
        (1, 0, 2) => "det+2samp",
        (0, 1, 2) => "ref+2samp",
        (0, 0, 3) => "3samp",
        _ => "other",
    }
}

/// A golden `{name: degrees}` position in hklrs E6C order, radians.
fn position(named: &Value) -> [f64; 6] {
    ORDER.map(|n| f(&named[n]).to_radians())
}

fn constraints(case: &Value) -> Result<Constraints, Error> {
    let named: Vec<(&str, Option<f64>)> = case["constraints"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_f64().map(f64::to_radians)))
        .collect();
    Constraints::from_names(&named)
}

/// The golden's virtual angles of hklrs position `values`, from hklrs poses
/// turned into You's lab.
struct Virtual {
    angles: [f64; 6],
    n: Vector3<f64>,
    s: Vector3<f64>,
    kout: Vector3<f64>,
}

impl Virtual {
    fn new(geo: &Geometry, values: &[f64; 6], n_phi: &Vector3<f64>, s_phi: &Vector3<f64>) -> Self {
        let r = analytic::frame();
        let poses = geo.poses(values).unwrap();
        let z = r * poses.sample.rotation.to_rotation_matrix().into_inner() * r.transpose();
        let kf = poses.detector.rotation * geo.beam_direction().into_inner();
        Self {
            angles: *values,
            n: z * n_phi.normalize(),
            s: z * s_phi.normalize(),
            kout: r * kf,
        }
    }

    /// The golden's residual of constraint `name` at `value` (rad); arcsine
    /// angles compare sines.
    fn residual(&self, name: &str, value: Option<f64>) -> f64 {
        let (n, s, kout) = (self.n, self.s, self.kout);
        let q = kout - Vector3::y();
        let qh = q.normalize();
        let sine = |x: f64| (x.clamp(-1.0, 1.0) - value.unwrap().sin()).abs();
        let angle = |x: f64| wrap(x - value.unwrap()).abs();
        match name {
            "qaz" => angle(qh.x.atan2(qh.z)),
            "naz" if n.x.hypot(n.z) < 1e-12 => f64::INFINITY,
            "naz" => angle(n.x.atan2(n.z)),
            "alpha" => sine(-n.y),
            "beta" => sine(n.dot(&kout)),
            "a_eq_b" => (-n.y - n.dot(&kout)).abs(),
            "psi" => {
                let tc = n.dot(&qh);
                angle(n.cross(&qh).y.atan2(n.y - tc * qh.y))
            }
            "betain" => sine(-s.y),
            "betaout" => sine(s.dot(&kout)),
            "bin_eq_bout" => (-s.y - s.dot(&kout)).abs(),
            "bisect" | "omega" => {
                let a = Vector3::new(
                    self.angles[1].sin(),
                    self.angles[1].cos() * self.angles[0].cos(),
                    self.angles[1].cos() * self.angles[0].sin(),
                );
                let eq = Vector3::new(qh.x, 0.0, qh.z).normalize();
                if name == "bisect" {
                    a.dot(&Vector3::y().cross(&eq)).abs()
                } else {
                    let theta = (q.norm() / 2.0).asin();
                    angle(a.dot(&eq).atan2(a.y) - theta)
                }
            }
            _ => angle(self.angles[ORDER.iter().position(|o| *o == name).unwrap()]),
        }
    }
}

/// What one case came to, for the family summary.
#[derive(Default)]
struct Tally {
    cases: usize,
    failed: Vec<String>,
    solutions: usize,
    diffcalc_matched: usize,
    numeric_matched: usize,
    extra: usize,
    degenerate: usize,
    empty: usize,
    errors: BTreeMap<String, usize>,
    dc: BTreeMap<String, usize>,
}

/// Checks one case; `Err` names what is wrong.
fn check(case: &Value, tally: &mut Tally) -> Result<(), String> {
    let geo = preset("E6C").unwrap();
    let r = analytic::frame();
    let ub = mat(&case["UB"]);
    let hkl = vec3(&case["hkl"]);
    let wavelength = Wavelength::new(f(&case["wavelength"])).unwrap();
    let (n_phi, s_phi) = (vec3(&case["n_phi"]), vec3(&case["surf_nphi"]));
    let numeric = &case["numeric"];
    let family_found = numeric["family"].as_bool().unwrap_or(false);
    let isolated: Vec<[f64; 6]> = if family_found {
        Vec::new()
    } else {
        numeric["solutions"]
            .as_array()
            .map(|s| s.iter().map(position).collect())
            .unwrap_or_default()
    };
    let valid: Vec<[f64; 6]> = case["diffcalc"]["solutions"]
        .as_array()
        .map(|s| {
            s.iter()
                .filter(|s| s["valid"].as_bool() == Some(true))
                .map(|s| position(&s["angles"]))
                .collect()
        })
        .unwrap_or_default();
    let names = case["constraints"].as_object().unwrap();
    let constraints = constraints(case).map_err(|e| format!("from_names: {e}"))?;
    let result = analytic::e6c(
        &constraints,
        &(r.transpose() * ub * hkl),
        wavelength,
        &(r.transpose() * n_phi),
        &(r.transpose() * s_phi),
    );
    for id in case["dc_ids"].as_array().unwrap() {
        *tally.dc.entry(id.as_str().unwrap().to_owned()).or_default() += 1;
    }
    let solutions = match result {
        Err(Error::Unreachable { .. }) if case["category"] == "unreachable" => return Ok(()),
        Err(Error::Degenerate { .. }) if family_found => {
            tally.degenerate += 1;
            return Ok(());
        }
        Err(Error::PsiUndefined)
            if names.contains_key("psi") && isolated.is_empty() && valid.is_empty() =>
        {
            *tally.errors.entry("PsiUndefined".into()).or_default() += 1;
            return Ok(());
        }
        Err(e) => return Err(format!("{e:?}")),
        Ok(s) if family_found => return Err(format!("{} solutions, numeric family", s.len())),
        Ok(s) if case["category"] == "unreachable" => {
            return Err(format!("{} solutions, unreachable", s.len()));
        }
        Ok(s) => s,
    };
    if solutions.is_empty() {
        tally.empty += 1;
    }
    tally.solutions += solutions.len();
    for values in &solutions {
        let point = Point3::from(geo.beam_direction().into_inner());
        let sc = scattering(&geo, values, wavelength, &point).unwrap();
        let back = ub.lu().solve(&(r * sc.q_sample)).unwrap();
        let miss = (back - hkl).amax();
        if miss > HKL {
            return Err(format!("{values:?} misses hkl by {miss:e}"));
        }
        let v = Virtual::new(&geo, values, &n_phi, &s_phi);
        for (name, value) in names {
            let res = v.residual(name, value.as_f64().map(f64::to_radians));
            if res.is_nan() || res > CONSTRAINT {
                return Err(format!("{values:?} misses {name} by {res:e}"));
            }
        }
    }
    let found = |want: &[f64; 6]| solutions.iter().any(|s| distance(s, want) < MATCH);
    for (what, want) in [("diffcalc", &valid), ("numeric", &isolated)] {
        if let Some(missing) = want.iter().find(|w| !found(w)) {
            return Err(format!("{what} solution {missing:?} missing"));
        }
    }
    tally.diffcalc_matched += valid.len();
    tally.numeric_matched += isolated.len();
    tally.extra += solutions
        .iter()
        .filter(|s| {
            !valid
                .iter()
                .chain(&isolated)
                .any(|w| distance(s, w) < MATCH)
        })
        .count();
    Ok(())
}

fn run(family_name: &str) {
    let golden = load();
    let mut tally = Tally::default();
    for case in golden["cases"].as_array().unwrap() {
        if case["category"] == "unsupported" || family(case) != family_name {
            continue;
        }
        tally.cases += 1;
        if let Err(e) = check(case, &mut tally) {
            tally.failed.push(format!(
                "{} {} {} {:?}: {e}",
                case["id"], case["category"], case["tag"], case["dc_ids"]
            ));
        }
    }
    eprintln!(
        "{family_name}: {} cases, {} failed; {} solutions ({} diffcalc-valid and {} numeric matched, {} extra), {} degenerate, {} empty, errors {:?}, dc cases {:?}",
        tally.cases,
        tally.failed.len(),
        tally.solutions,
        tally.diffcalc_matched,
        tally.numeric_matched,
        tally.extra,
        tally.degenerate,
        tally.empty,
        tally.errors,
        tally.dc,
    );
    assert!(tally.cases > 0);
    assert!(tally.failed.is_empty(), "{}", tally.failed.join("\n"));
}

#[test]
fn det_ref_samp() {
    run("det+ref+samp");
}

#[test]
fn det_2samp() {
    run("det+2samp");
}

#[test]
fn det_bisect() {
    run("det+bisect");
}

#[test]
fn ref_2samp() {
    run("ref+2samp");
}

#[test]
fn samp_3() {
    run("3samp");
}

/// The golden cases citing defect `id`, each checked; `count` of them.
fn dc_cases(id: &str, count: usize) -> Vec<(Value, Tally)> {
    let golden = load();
    let cases: Vec<(Value, Tally)> = golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["dc_ids"].as_array().unwrap().iter().any(|i| i == id))
        .map(|case| {
            let mut tally = Tally::default();
            check(case, &mut tally).unwrap_or_else(|e| panic!("{id} {}: {e}", case["id"]));
            (case.clone(), tally)
        })
        .collect();
    assert_eq!(cases.len(), count, "{id}");
    cases
}

/// DC-2: diffcalc solves `naz` on the surface normal and filters on the
/// reference vector, so it returns nothing; `naz` is on the reference
/// vector here and the numeric solve's full set comes back.
#[test]
fn dc2_naz_is_on_the_reference_vector() {
    for (case, tally) in dc_cases("DC-2", 10) {
        assert_eq!(case["counts"]["returned"], 0);
        assert!(tally.numeric_matched > 0, "{}", case["id"]);
    }
}

/// DC-3: half of diffcalc's bisect positions put the χ axis off the plane;
/// every position here meets bisect.
#[test]
fn dc3_bisect_holds_for_every_position() {
    for (case, tally) in dc_cases("DC-3", 14) {
        assert!(case["counts"]["invalid"].as_u64().unwrap() > 0);
        assert!(tally.solutions > 0, "{}", case["id"]);
    }
}

/// DC-4: with η and φ held and cos η < 0 diffcalc aborts and returns
/// nothing; every isolated position of the numeric solve comes back here.
#[test]
fn dc4_eta_with_negative_cosine_is_reached() {
    for (case, tally) in dc_cases("DC-4", 4) {
        assert_eq!(case["counts"]["returned"], 0);
        let numeric = case["counts"]["numeric_isolated"].as_u64().unwrap();
        assert!(
            numeric > 0 && tally.numeric_matched as u64 == numeric,
            "{}",
            case["id"]
        );
    }
}

/// DC-5: hkl = 0 is a continuous family, not a division by zero.
#[test]
fn dc5_zero_q_is_degenerate() {
    for (_, tally) in dc_cases("DC-5", 1) {
        assert_eq!(tally.degenerate, 1);
    }
}

/// DC-6: at backscatter diffcalc rewrites the held μ and repeats positions;
/// the positions form a family and are reported as one.
#[test]
fn dc6_backscatter_is_degenerate() {
    for (_, tally) in dc_cases("DC-6", 1) {
        assert_eq!(tally.degenerate, 1);
    }
}

/// Every golden "unsupported" constraint set is refused by name.
#[test]
fn unsupported_sets_are_refused() {
    let golden = load();
    for case in golden["cases"].as_array().unwrap() {
        if case["category"] == "unsupported" {
            assert!(
                matches!(constraints(case), Err(Error::Unsupported { .. })),
                "{}",
                case["id"]
            );
        }
    }
}
