//! Inverse invariants on E4CV and E4CH: every solution forwards back to the
//! target and satisfies its mode, held axes never move, the search is
//! deterministic, axis limits only filter the solutions after the cut-point
//! wrap, out-of-sphere targets and bad inputs are errors.

mod inverse_check;

use std::f64::consts::{PI, TAU};

use hklrs::inverse::{self, e4c, e6c};
use hklrs::nalgebra::{Matrix3, UnitQuaternion, Vector3};
use hklrs::presets::preset;
use hklrs::{
    Equation, Geometry, InverseError, InverseRequest, Lattice, Mode, Picker, Target, Wavelength,
};
use inverse_check::wrap;

fn ub() -> Matrix3<f64> {
    let u = UnitQuaternion::from_euler_angles(0.3, -0.7, 1.1);
    u.to_rotation_matrix().into_inner() * Lattice::orthorhombic(4.1, 5.3, 6.7).unwrap().b_matrix()
}

fn wavelength() -> Wavelength {
    Wavelength::new(1.54).unwrap()
}

fn assert_solution(mode: &Mode, hkl: &Vector3<f64>, current: &[f64], values: &[f64]) {
    if let Err(e) = inverse_check::check(mode, &ub(), wavelength(), hkl, current, values) {
        panic!("{}: {e}", mode.name());
    }
}

fn modes(geo: &Geometry) -> Vec<Mode<'_>> {
    vec![
        e4c::bissector(geo).unwrap(),
        e4c::constant_omega(geo).unwrap(),
        e4c::constant_chi(geo).unwrap(),
        e4c::constant_phi(geo).unwrap(),
        e4c::double_diffraction(geo, Vector3::new(1.0, 1.0, 0.0)).unwrap(),
        e4c::psi_constant(geo, Vector3::new(0.0, 0.0, 1.0), 0.4).unwrap(),
    ]
}

const TARGETS: [[f64; 3]; 4] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [-1.0, 1.0, 2.0],
];

fn current(geo: &Geometry) -> Vec<f64> {
    geo.values_from_degrees(&[10.0, 20.0, 30.0, 40.0]).unwrap()
}

fn request<'a>(mode: &'a Mode<'a>, hkl: Vector3<f64>, current: &'a [f64]) -> InverseRequest<'a> {
    InverseRequest {
        mode,
        ub: ub(),
        wavelength: wavelength(),
        target: Target::Hkl(hkl),
        current,
    }
}

#[test]
fn every_solution_reaches_target_and_satisfies_mode() {
    for name in ["E4CV", "E4CH"] {
        let geo = preset(name).unwrap();
        let current = current(&geo);
        for mode in modes(&geo) {
            for t in TARGETS {
                let hkl = Vector3::from(t);
                match request(&mode, hkl, &current).solve() {
                    Ok(solutions) => {
                        for s in &solutions {
                            assert_solution(&mode, &hkl, &current, s);
                        }
                    }
                    Err(InverseError::NoSolution) => {}
                    Err(e) => panic!("{name} {} {hkl}: {e}", mode.name()),
                }
            }
        }
    }
}

#[test]
fn bissector_finds_all_eight_branches() {
    // omega ↔ tth/2 up to π, chi and phi each with two branches: 2 (sign of
    // tth) × 2 (omega or omega + π) × 2 (chi, phi pair) — hkl drops the tth < 0,
    // omega + π branch.
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let mode = e4c::bissector(&geo).unwrap();
    let solutions = request(&mode, Vector3::new(1.0, 2.0, 1.0), &current)
        .solve()
        .unwrap();
    assert_eq!(solutions.len(), 8, "{solutions:?}");
}

#[test]
fn same_request_gives_identical_list() {
    for name in ["E4CV", "E4CH"] {
        let geo = preset(name).unwrap();
        let current = current(&geo);
        for mode in modes(&geo) {
            let req = request(&mode, Vector3::new(1.0, 2.0, 1.0), &current);
            let bits = |r: Result<Vec<Vec<f64>>, InverseError>| {
                r.map(|s| {
                    s.iter()
                        .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
            };
            assert_eq!(
                bits(req.solve()),
                bits(req.solve()),
                "{name} {}",
                mode.name()
            );
        }
    }
}

#[test]
fn sorted_by_distance_and_picker() {
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let mode = e4c::bissector(&geo).unwrap();
    let req = request(&mode, Vector3::new(1.0, 2.0, 1.0), &current);
    let solutions = req.solve().unwrap();
    let distance = |s: &[f64]| -> f64 {
        s.iter()
            .zip(&current)
            .map(|(a, b)| wrap(a - b).powi(2))
            .sum()
    };
    assert!(
        solutions
            .windows(2)
            .all(|w| distance(&w[0]) <= distance(&w[1]))
    );
    assert_eq!(req.pick(Picker::Closest).unwrap(), solutions[0]);
    assert!(solutions.contains(&req.pick(Picker::First).unwrap()));
}

#[test]
fn held_axis_never_moves() {
    // A held value outside (−π, π] stays exactly as given.
    let geo = preset("E4CH").unwrap();
    let current = vec![7.0, -4.0, 9.5, 0.3];
    for mode in modes(&geo).iter().skip(1).take(3) {
        let solutions = request(mode, Vector3::new(0.0, 1.0, 1.0), &current)
            .solve()
            .unwrap();
        for s in &solutions {
            assert_solution(mode, &Vector3::new(0.0, 1.0, 1.0), &current, s);
        }
    }
}

#[test]
fn unreachable_target_is_err_for_every_mode() {
    for name in ["E4CV", "E4CH"] {
        let geo = preset(name).unwrap();
        let current = current(&geo);
        let hkl = Vector3::new(6.0, 0.0, 0.0);
        let q = (ub() * hkl).norm();
        let q_max = 2.0 * wavelength().wavenumber();
        for mode in modes(&geo) {
            assert_eq!(
                request(&mode, hkl, &current).solve(),
                Err(InverseError::Unreachable { q, q_max }),
                "{name} {}",
                mode.name()
            );
        }
    }
}

#[test]
fn q_sample_target_matches_hkl_target() {
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let mode = e4c::constant_phi(&geo).unwrap();
    let hkl = Vector3::new(0.0, 1.0, 1.0);
    let by_hkl = request(&mode, hkl, &current).solve().unwrap();
    let by_q = InverseRequest {
        target: Target::QSample(ub() * hkl),
        ..request(&mode, hkl, &current)
    }
    .solve()
    .unwrap();
    assert_eq!(by_hkl, by_q);
}

#[test]
fn psi_of_a_psi_constant_solution_is_its_parameter() {
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let reference = Vector3::new(0.0, 0.0, 1.0);
    let mode = e4c::psi_constant(&geo, reference, 0.4).unwrap();
    for s in request(&mode, Vector3::new(1.0, 2.0, 1.0), &current)
        .solve()
        .unwrap()
    {
        let psi = inverse::psi(&geo, &ub(), &reference, &s).unwrap();
        assert!((psi - 0.4).abs() <= 1e-10, "{psi}");
    }
}

#[test]
fn mode_construction_errors() {
    let geo = preset("E4CV").unwrap();
    let tth = geo.axis_index("tth").unwrap();
    let omega = geo.axis_index("omega").unwrap();
    assert_eq!(
        Mode::new(&geo, "m", &[0, 1, 9], vec![]),
        Err(InverseError::NoSuchAxis { index: 9, count: 4 })
    );
    assert_eq!(
        Mode::new(&geo, "m", &[0, 1, 1], vec![]),
        Err(InverseError::WrittenTwice { name: "chi".into() })
    );
    assert_eq!(
        Mode::new(&geo, "m", &[0, 1, 2, 3], vec![]),
        Err(InverseError::EquationCount {
            written: 4,
            equations: 3
        })
    );
    assert_eq!(
        Mode::new(
            &geo,
            "m",
            &[0, 1, 2, 3],
            vec![Equation::Bisector {
                sample: tth,
                detector: omega
            }]
        ),
        Err(InverseError::NotInStack {
            name: "tth".into(),
            holder: "sample"
        })
    );
    assert_eq!(
        e4c::double_diffraction(&geo, Vector3::zeros()),
        Err(InverseError::ZeroReference { what: "hkl2" })
    );
    assert_eq!(
        e4c::psi_constant(&geo, Vector3::new(0.0, f64::NAN, 1.0), 0.0),
        Err(InverseError::NonFinite {
            what: "psi reference"
        })
    );
    assert_eq!(
        e4c::psi_constant(&geo, Vector3::x(), f64::INFINITY),
        Err(InverseError::NonFinite { what: "psi" })
    );
    let translated = Geometry::builder("t")
        .sample_rotation("omega", [0.0, -1.0, 0.0])
        .sample_translation("tz", [0.0, 0.0, 1.0])
        .sample_rotation("phi", [0.0, -1.0, 0.0])
        .detector_rotation("tth", [0.0, -1.0, 0.0])
        .build()
        .unwrap();
    assert_eq!(
        Mode::new(&translated, "m", &[0, 1, 3], vec![]),
        Err(InverseError::NotRotation { name: "tz".into() })
    );
    assert_eq!(
        e4c::bissector(&translated),
        Err(InverseError::Geometry(hklrs::Error::UnknownAxis {
            name: "chi".into()
        }))
    );
}

#[test]
fn bad_inputs_are_errors() {
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let mode = e4c::bissector(&geo).unwrap();
    let hkl = Vector3::new(1.0, 0.0, 0.0);
    assert!(matches!(
        request(&mode, hkl, &current[..3]).solve(),
        Err(InverseError::Geometry(hklrs::Error::AxisCount { .. }))
    ));
    let nan = [current[0], f64::NAN, current[2], current[3]];
    assert!(matches!(
        request(&mode, hkl, &nan).solve(),
        Err(InverseError::Geometry(hklrs::Error::NonFinite { index: 1 }))
    ));
    assert_eq!(
        request(&mode, Vector3::new(f64::NAN, 0.0, 0.0), &current).solve(),
        Err(InverseError::NonFinite { what: "target hkl" })
    );
    let mut bad = request(&mode, hkl, &current);
    bad.ub[(1, 2)] = f64::INFINITY;
    assert_eq!(bad.solve(), Err(InverseError::NonFinite { what: "UB" }));
}

#[test]
fn zero_q_is_degenerate_in_every_mode() {
    let degenerate = |mode: &Mode, current: &[f64]| {
        for target in [
            Target::Hkl(Vector3::zeros()),
            Target::QSample(Vector3::zeros()),
        ] {
            let req = InverseRequest {
                target,
                ..request(mode, Vector3::zeros(), current)
            };
            assert_eq!(
                req.solve(),
                Err(InverseError::Degenerate),
                "{}",
                mode.name()
            );
        }
    };
    for name in ["E4CV", "E4CH"] {
        let geo = preset(name).unwrap();
        let current = current(&geo);
        for mode in modes(&geo) {
            degenerate(&mode, &current);
        }
    }
    let geo = preset("E6C").unwrap();
    let current = geo
        .values_from_degrees(&[5.0, 10.0, 20.0, 30.0, 15.0, 40.0])
        .unwrap();
    let (hkl2, reference) = (Vector3::new(1.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
    for mode in [
        e6c::bissector_vertical(&geo),
        e6c::constant_omega_vertical(&geo),
        e6c::constant_chi_vertical(&geo),
        e6c::constant_phi_vertical(&geo),
        e6c::lifting_detector_phi(&geo),
        e6c::lifting_detector_omega(&geo),
        e6c::lifting_detector_mu(&geo),
        e6c::double_diffraction_vertical(&geo, hkl2),
        e6c::bissector_horizontal(&geo),
        e6c::double_diffraction_horizontal(&geo, hkl2),
        e6c::psi_constant_vertical(&geo, reference, 0.4),
        e6c::psi_constant_horizontal(&geo, reference, 0.4),
        e6c::constant_mu_horizontal(&geo),
    ] {
        degenerate(&mode.unwrap(), &current);
    }
}

type Solutions = Result<Vec<Vec<f64>>, InverseError>;

fn solve_with(
    geo: &Geometry,
    mode: fn(&Geometry) -> Result<Mode<'_>, InverseError>,
    hkl: Vector3<f64>,
) -> Solutions {
    let current = current(geo);
    let mode = mode(geo).unwrap();
    request(&mode, hkl, &current).solve()
}

fn bissector_121(geo: &Geometry) -> Solutions {
    solve_with(geo, e4c::bissector, Vector3::new(1.0, 2.0, 1.0))
}

/// What limits [min, max] on `axis` must leave of the unlimited `all`.
fn filtered(all: &[Vec<f64>], axis: usize, min: f64, max: f64) -> Solutions {
    let kept: Vec<_> = all
        .iter()
        .filter(|s| min <= s[axis] && s[axis] <= max)
        .cloned()
        .collect();
    if kept.is_empty() {
        Err(InverseError::NoSolutionInLimits(all.to_vec()))
    } else {
        Ok(kept)
    }
}

/// The f64 next to the non-zero `x`, away from zero.
fn next_out(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

/// The f64 next to the non-zero `x`, toward zero.
fn next_in(x: f64) -> f64 {
    f64::from_bits(x.to_bits() - 1)
}

#[test]
fn solution_exactly_on_a_limit_is_kept() {
    let mut geo = preset("E4CV").unwrap();
    let all = bissector_121(&geo).unwrap();
    let tth = geo.axis_index("tth").unwrap();
    let on = all[0][tth];
    geo.set_limits("tth", on, on).unwrap();
    let kept = bissector_121(&geo);
    assert!(kept.as_ref().unwrap().contains(&all[0]));
    assert_eq!(kept, filtered(&all, tth, on, on));
}

#[test]
fn solution_just_outside_a_limit_is_rejected() {
    let mut geo = preset("E4CV").unwrap();
    let all = bissector_121(&geo).unwrap();
    let tth = geo.axis_index("tth").unwrap();
    let on = all[0][tth];
    let (above, below) = if on > 0.0 {
        (next_out(on), next_in(on))
    } else {
        (next_in(on), next_out(on))
    };
    for (min, max) in [(above, f64::INFINITY), (f64::NEG_INFINITY, below)] {
        geo.set_limits("tth", min, max).unwrap();
        let result = bissector_121(&geo);
        assert_eq!(result, filtered(&all, tth, min, max), "[{min}, {max}]");
    }
}

#[test]
fn cut_point_moves_a_solution_into_range() {
    let mut geo = preset("E4CV").unwrap();
    let all = bissector_121(&geo).unwrap();
    let phi = geo.axis_index("phi").unwrap();
    let negative: Vec<_> = all.iter().filter(|s| s[phi] < 0.0).collect();
    assert!(!negative.is_empty() && negative.len() < all.len());
    // Disjoint from the default window [−π, π) but for its open end.
    geo.set_limits("phi", PI, TAU).unwrap();
    assert_eq!(
        bissector_121(&geo),
        Err(InverseError::NoSolutionInLimits(all.clone()))
    );
    geo.set_cut_point("phi", 0.0).unwrap();
    let moved = bissector_121(&geo).unwrap();
    assert_eq!(moved.len(), negative.len());
    for (m, s) in moved.iter().zip(&negative) {
        assert!((PI..TAU).contains(&m[phi]), "{m:?}");
        assert!((m[phi] - (s[phi] + TAU)).abs() <= 1e-15, "{m:?} {s:?}");
        for i in (0..m.len()).filter(|&i| i != phi) {
            assert_eq!(m[i].to_bits(), s[i].to_bits(), "{m:?} {s:?}");
        }
    }
}

#[test]
fn every_candidate_rejected_is_carried_in_the_error() {
    let mut geo = preset("E4CV").unwrap();
    let all = bissector_121(&geo).unwrap();
    let tth = geo.axis_index("tth").unwrap();
    let largest = all.iter().map(|s| s[tth].abs()).fold(0.0, f64::max);
    geo.set_limits("tth", next_out(largest), f64::INFINITY)
        .unwrap();
    assert_eq!(
        bissector_121(&geo),
        Err(InverseError::NoSolutionInLimits(all))
    );
}

#[test]
fn held_axis_outside_its_limits_is_not_filtered() {
    // phi is held at current's 40°.
    let mut geo = preset("E4CV").unwrap();
    let hkl = Vector3::new(0.0, 1.0, 1.0);
    let all = solve_with(&geo, e4c::constant_phi, hkl).unwrap();
    geo.set_limits("phi", -0.1, 0.1).unwrap();
    assert_eq!(solve_with(&geo, e4c::constant_phi, hkl), Ok(all));
}
