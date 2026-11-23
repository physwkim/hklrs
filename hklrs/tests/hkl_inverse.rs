//! hkl 5.1.9 inverse golden (golden/hkl_inverse.py), E4CV, E4CH, E6C,
//! ESRF BM28 PSIC, ZAXIS, ESRF ID01 PSIC, K4CV and K6C.
//!
//! Per case: the mode writes hkl's axes; every hkl solution within VIOLATION
//! of hkl's own mode equations is in the hklrs set within MATCH rad, directly
//! or after one Newton step on the independent residuals where hkl's own
//! precision is coarser (counted); one beyond it (HKL-18: the Ewald step of
//! hkl_mode_set_hkl_real does not re-test them; HKL-16: the kappa twin hkl
//! appends moves a held axis) still reaches hkl, is not in the hklrs set and
//! is counted under its defect;
//! every hklrs solution passes the independent check (extras beyond hkl's
//! are counted); |Q| > 4π/λ is `Unreachable`; `Degenerate` holds only where
//! the independent Jacobian is singular at every hkl solution; a case hkl
//! fails with |Q| <= 2k is an hkl miss only where hklrs returns a verified
//! solution (held axes can still leave the target out of reach).

mod common;
mod inverse_check;

use hklrs::inverse::{e4c, e6c, esrf_bm28_psic, k4c, k6c, zaxis};
use hklrs::nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use hklrs::presets::preset;
use hklrs::{Geometry, InverseError, InverseRequest, Mode, Target, Wavelength};
use inverse_check::{largest, wrap};
use serde_json::Value;

/// hkl's own residual leaves its angles within ~1e-7 rad of the exact
/// solution where the Jacobian is well conditioned.
const MATCH: f64 = 1e-6;
/// Largest σ_min of the independent Jacobian at an hkl solution that
/// confirms `Degenerate`.
const SINGULAR: f64 = 1e-6;
/// Smallest miss of hkl's own mode equations (the golden's
/// constraint_resid: deg, or Å⁻²) that rules an hkl solution out; hkl's
/// precision there is ~1e-5.
const VIOLATION: f64 = 1e-3;

fn angular_distance(a: &[f64], b: &[f64]) -> f64 {
    largest(a.iter().zip(b).map(|(x, y)| wrap(x - y).abs()))
}

fn mode<'g>(geo: &'g Geometry, case: &Value) -> Mode<'g> {
    let names = case["parameter_names"].as_array().unwrap();
    let param = |n: &str| {
        let i = names.iter().position(|p| p == n).unwrap();
        common::f(&case["parameters"][i])
    };
    let rad = |n: &str| param(n).to_radians();
    let hkl2 = || Vector3::new(param("h2"), param("k2"), param("l2"));
    let psi = || param("psi").to_radians();
    let kappa = case["geometry"] == "K4CV";
    let k6c = case["geometry"] == "K6C";
    match case["mode"].as_str().unwrap() {
        "bissector_vertical" if k6c => k6c::bissector_vertical(geo),
        "constant_omega_vertical" if k6c => k6c::constant_omega_vertical(geo, rad("omega")),
        "constant_chi_vertical" if k6c => k6c::constant_chi_vertical(geo, rad("chi")),
        "constant_phi_vertical" if k6c => k6c::constant_phi_vertical(geo, rad("phi")),
        "lifting_detector_kphi" if k6c => k6c::lifting_detector_kphi(geo),
        "lifting_detector_komega" if k6c => k6c::lifting_detector_komega(geo),
        "lifting_detector_mu" if k6c => k6c::lifting_detector_mu(geo),
        "double_diffraction_vertical" if k6c => k6c::double_diffraction_vertical(geo, hkl2()),
        "bissector_horizontal" if k6c => k6c::bissector_horizontal(geo),
        "constant_phi_horizontal" if k6c => k6c::constant_phi_horizontal(geo, rad("phi")),
        "constant_kphi_horizontal" if k6c => k6c::constant_kphi_horizontal(geo),
        "double_diffraction_horizontal" if k6c => k6c::double_diffraction_horizontal(geo, hkl2()),
        "psi_constant_vertical" if k6c => k6c::psi_constant_vertical(geo, hkl2(), psi()),
        // hkl's init takes the normal as a lab direction at the start
        // geometry, where the sample rotation carries the holder frame.
        "constant_incidence" if k6c => {
            let start = geo.poses(&axes(geo, &case["start_axes"])).unwrap();
            let normal =
                start.sample.rotation.inverse() * Vector3::new(param("x"), param("y"), param("z"));
            k6c::constant_incidence(geo, normal, rad("incidence"), rad("azimuth"))
        }
        "bissector" if kappa => k4c::bissector(geo),
        "constant_omega" if kappa => k4c::constant_omega(geo, rad("omega")),
        "constant_chi" if kappa => k4c::constant_chi(geo, rad("chi")),
        "constant_phi" if kappa => k4c::constant_phi(geo, rad("phi")),
        "double_diffraction" if kappa => k4c::double_diffraction(geo, hkl2()),
        "psi_constant" if kappa => k4c::psi_constant(geo, hkl2(), psi()),
        "bissector" => e4c::bissector(geo),
        "constant_omega" => e4c::constant_omega(geo),
        "constant_chi" => e4c::constant_chi(geo),
        "constant_phi" => e4c::constant_phi(geo),
        "double_diffraction" => e4c::double_diffraction(geo, hkl2()),
        "psi_constant" => e4c::psi_constant(geo, hkl2(), psi()),
        "bissector_vertical" => e6c::bissector_vertical(geo),
        "constant_omega_vertical" => e6c::constant_omega_vertical(geo),
        "constant_chi_vertical" => e6c::constant_chi_vertical(geo),
        "constant_phi_vertical" => e6c::constant_phi_vertical(geo),
        "lifting_detector_phi" => e6c::lifting_detector_phi(geo),
        "lifting_detector_omega" => e6c::lifting_detector_omega(geo),
        "lifting_detector_mu" => e6c::lifting_detector_mu(geo),
        "double_diffraction_vertical" => e6c::double_diffraction_vertical(geo, hkl2()),
        "bissector_horizontal" => e6c::bissector_horizontal(geo),
        "double_diffraction_horizontal" => e6c::double_diffraction_horizontal(geo, hkl2()),
        "psi_constant_vertical" => e6c::psi_constant_vertical(geo, hkl2(), psi()),
        "psi_constant_horizontal" => e6c::psi_constant_horizontal(geo, hkl2(), psi()),
        "constant_mu_horizontal" => e6c::constant_mu_horizontal(geo),
        "constant_nu_coplanar" => esrf_bm28_psic::constant_nu_coplanar(geo),
        "constant_delta_coplanar" => esrf_bm28_psic::constant_delta_coplanar(geo),
        "constant_eta_noncoplanar" => esrf_bm28_psic::constant_eta_noncoplanar(geo),
        "zaxis" => zaxis::zaxis(geo),
        "reflectivity" => zaxis::reflectivity(geo),
        m => panic!("mode {m}"),
    }
    .unwrap()
}

/// Jacobian of [`inverse_check::residuals`] over the written axes by
/// central differences.
fn jacobian(
    mode: &Mode,
    ub: &Matrix3<f64>,
    wavelength: Wavelength,
    hkl: &Vector3<f64>,
    values: &[f64],
) -> DMatrix<f64> {
    const H: f64 = 1e-6;
    let n = mode.written().len();
    let at = |axis: usize, d: f64| {
        let mut v = values.to_vec();
        v[axis] += d;
        inverse_check::residuals(mode, ub, wavelength, hkl, &v).unwrap()
    };
    let mut j = DMatrix::zeros(n, n);
    for (c, &axis) in mode.written().iter().enumerate() {
        let (plus, minus) = (at(axis, H), at(axis, -H));
        for r in 0..n {
            j[(r, c)] = (plus[r] - minus[r]) / (2.0 * H);
        }
    }
    j
}

/// One Newton step from `values` on the independent residuals: hkl's
/// solution polished without the solver under test.
fn newton(
    mode: &Mode,
    ub: &Matrix3<f64>,
    wavelength: Wavelength,
    hkl: &Vector3<f64>,
    values: &[f64],
) -> Option<Vec<f64>> {
    let r = inverse_check::residuals(mode, ub, wavelength, hkl, values).unwrap();
    let step = jacobian(mode, ub, wavelength, hkl, values)
        .lu()
        .solve(&DVector::from_vec(r))?;
    let mut x = values.to_vec();
    for (d, &axis) in step.iter().zip(mode.written()) {
        x[axis] = wrap(x[axis] - d);
    }
    Some(x)
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
    hkl16: usize,
    hkl18: usize,
    newton: usize,
    degenerate: usize,
    singular: usize,
    hklrs_solutions: usize,
    extras: usize,
    hkl_failed: usize,
    hkl_misses: usize,
    hkl_miss_solutions: usize,
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
        let (theirs, violators): (Vec<_>, Vec<_>) =
            case["solutions"].as_array().unwrap().iter().partition(|s| {
                let r = s["constraint_resid"].as_object().unwrap();
                r.values().all(|v| common::f(v).abs() <= VIOLATION)
            });
        for v in &violators {
            // A held axis off its value is the appended twin's.
            let held = v["constraint_resid"]
                .as_object()
                .unwrap()
                .iter()
                .any(|(k, r)| k.starts_with("fixed:") && common::f(r).abs() > VIOLATION);
            *if held { &mut n.hkl16 } else { &mut n.hkl18 } += 1;
        }
        let wrapped =
            |s: &Value| -> Vec<f64> { axes(&geo, &s["axes"]).into_iter().map(wrap).collect() };
        let theirs: Vec<Vec<f64>> = theirs.into_iter().map(wrapped).collect();
        let violators: Vec<Vec<f64>> = violators.into_iter().map(wrapped).collect();
        for v in &violators {
            let r = inverse_check::residuals(&mode, &ub, wavelength, &hkl, v).unwrap();
            let miss = largest(r[..3].iter().map(|d| d.abs()));
            if miss > MATCH {
                fail(format!(
                    "hkl solution {v:?} violates the mode and misses hkl by {miss:e}"
                ));
            }
        }
        let ours = match result {
            Ok(s) => s,
            Err(InverseError::NoSolution) => Vec::new(),
            Err(InverseError::Degenerate) => {
                n.degenerate += 1;
                n.singular += theirs.len();
                if theirs.is_empty() {
                    fail("Degenerate, no hkl solution to confirm it".into());
                }
                for t in &theirs {
                    let sigma = jacobian(&mode, &ub, wavelength, &hkl, t)
                        .singular_values()
                        .min();
                    if sigma > SINGULAR {
                        fail(format!(
                            "Degenerate, but hkl solution {t:?} has σ_min {sigma:e}"
                        ));
                    }
                }
                continue;
            }
            Err(e) => {
                fail(format!("{e}"));
                continue;
            }
        };
        let mut verified = 0;
        for s in &ours {
            match inverse_check::check(&mode, &ub, wavelength, &hkl, &current, s) {
                Ok(()) => verified += 1,
                Err(e) => fail(e),
            }
        }
        n.hklrs_solutions += ours.len();
        n.hkl_solutions += theirs.len();
        let nearest = |t: &[f64]| {
            ours.iter()
                .map(|s| angular_distance(s, t))
                .fold(f64::INFINITY, f64::min)
        };
        for t in &theirs {
            if nearest(t) <= MATCH {
                continue;
            }
            match newton(&mode, &ub, wavelength, &hkl, t) {
                Some(p) if nearest(&p) <= MATCH => n.newton += 1,
                p => fail(format!(
                    "hkl solution {t:?} not found, nearest {:e}, after a Newton step {:?}",
                    nearest(t),
                    p.map(|p| nearest(&p))
                )),
            }
        }
        for v in &violators {
            if ours.iter().any(|s| angular_distance(s, v) <= MATCH) {
                fail(format!(
                    "hkl solution {v:?} violates the mode, yet hklrs returns it"
                ));
            }
        }
        n.extras += ours
            .iter()
            .filter(|s| !theirs.iter().any(|t| angular_distance(s, t) <= MATCH))
            .count();
        if case.get("error").is_some() {
            n.hkl_failed += 1;
            n.hkl_misses += usize::from(verified > 0);
            n.hkl_miss_solutions += verified;
        }
    }
    println!(
        "{geometry} {mode_name}: {} cases, {} unreachable, {} degenerate (hkl's {} solutions \
         there singular); hkl {} solutions elsewhere, all found ({} after a Newton step), {} \
         moving a held axis (HKL-16) and {} violating an equation (HKL-18); hklrs {} solutions, {} beyond hkl's; hkl failed {} cases with \
         |Q| <= 2k, {} of them hkl misses (hklrs {} verified solutions there)",
        n.cases,
        n.unreachable,
        n.degenerate,
        n.singular,
        n.hkl_solutions,
        n.newton,
        n.hkl16,
        n.hkl18,
        n.hklrs_solutions,
        n.extras,
        n.hkl_failed,
        n.hkl_misses,
        n.hkl_miss_solutions,
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
    e6c_bissector_vertical: "E6C" "bissector_vertical",
    e6c_constant_omega_vertical: "E6C" "constant_omega_vertical",
    e6c_constant_chi_vertical: "E6C" "constant_chi_vertical",
    e6c_constant_phi_vertical: "E6C" "constant_phi_vertical",
    e6c_lifting_detector_phi: "E6C" "lifting_detector_phi",
    e6c_lifting_detector_omega: "E6C" "lifting_detector_omega",
    e6c_lifting_detector_mu: "E6C" "lifting_detector_mu",
    e6c_double_diffraction_vertical: "E6C" "double_diffraction_vertical",
    e6c_bissector_horizontal: "E6C" "bissector_horizontal",
    e6c_double_diffraction_horizontal: "E6C" "double_diffraction_horizontal",
    e6c_psi_constant_vertical: "E6C" "psi_constant_vertical",
    e6c_psi_constant_horizontal: "E6C" "psi_constant_horizontal",
    e6c_constant_mu_horizontal: "E6C" "constant_mu_horizontal",
    bm28_constant_nu_coplanar: "ESRF BM28 PSIC" "constant_nu_coplanar",
    bm28_constant_delta_coplanar: "ESRF BM28 PSIC" "constant_delta_coplanar",
    bm28_constant_eta_noncoplanar: "ESRF BM28 PSIC" "constant_eta_noncoplanar",
    zaxis_zaxis: "ZAXIS" "zaxis",
    zaxis_reflectivity: "ZAXIS" "reflectivity",
    id01_constant_nu_coplanar: "ESRF ID01 PSIC" "constant_nu_coplanar",
    id01_constant_delta_coplanar: "ESRF ID01 PSIC" "constant_delta_coplanar",
    id01_constant_eta_noncoplanar: "ESRF ID01 PSIC" "constant_eta_noncoplanar",
    k4cv_bissector: "K4CV" "bissector",
    k4cv_constant_omega: "K4CV" "constant_omega",
    k4cv_constant_phi: "K4CV" "constant_phi",
    k4cv_double_diffraction: "K4CV" "double_diffraction",
    k4cv_psi_constant: "K4CV" "psi_constant",
    k6c_bissector_vertical: "K6C" "bissector_vertical",
    k6c_constant_omega_vertical: "K6C" "constant_omega_vertical",
    k6c_constant_phi_vertical: "K6C" "constant_phi_vertical",
    k6c_lifting_detector_komega: "K6C" "lifting_detector_komega",
    k6c_double_diffraction_vertical: "K6C" "double_diffraction_vertical",
    k6c_bissector_horizontal: "K6C" "bissector_horizontal",
    k6c_constant_phi_horizontal: "K6C" "constant_phi_horizontal",
    k6c_constant_kphi_horizontal: "K6C" "constant_kphi_horizontal",
    k6c_double_diffraction_horizontal: "K6C" "double_diffraction_horizontal",
    k6c_psi_constant_vertical: "K6C" "psi_constant_vertical",
    k6c_constant_incidence: "K6C" "constant_incidence",
    k4cv_constant_chi: "K4CV" "constant_chi",
    k6c_constant_chi_vertical: "K6C" "constant_chi_vertical",
    k6c_lifting_detector_kphi: "K6C" "lifting_detector_kphi",
    k6c_lifting_detector_mu: "K6C" "lifting_detector_mu",
}
