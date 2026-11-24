//! Inverse invariants on E4CV, E4CH and K4CV: every solution forwards back
//! to the target and satisfies its mode, held axes never move, the search is
//! deterministic, axis limits only filter the solutions after the cut-point
//! wrap, out-of-sphere targets and bad inputs are errors.

mod inverse_check;

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use hklrs::inverse::kappa::Branch;
use hklrs::inverse::{
    self, aps_04id_polar, e4c, e6c, esrf_bm28_psic, esrf_id01_psic, k6c, petra3_p09_eh2,
    soleil_sixs_med, zaxis,
};
use hklrs::nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use hklrs::presets::{KAPPA_ALPHA_DEG, PRESETS, kappa_axis, preset};
use hklrs::{
    Deg, Equation, Eulerian, Geometry, InverseError, InverseRequest, Lattice, Mode, ModeSpec,
    ModeTable, Picker, Target, Wavelength, scattering,
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

/// Mode `name` of `geo`'s preset table, its parameters arbitrary but for a
/// psi mode's reference and ψ: a second reflection (1, 1, 0), a surface
/// normal (0, 0, 1).
fn table_mode<'g>(
    geo: &'g Geometry,
    name: &str,
    reference: Vector3<f64>,
    psi: f64,
) -> Result<Mode<'g>, InverseError> {
    let parameters = ModeTable::of(geo.name()).unwrap().parameters(name).unwrap();
    let hkl2 = if parameters.iter().any(|p| p.name == "psi") {
        reference
    } else {
        Vector3::new(1.0, 1.0, 0.0)
    };
    let value = |p: &str| match p {
        "h2" => hkl2.x,
        "k2" => hkl2.y,
        "l2" => hkl2.z,
        "x" | "y" => 0.0,
        "z" => 1.0,
        "psi" => psi,
        "omega" => 0.2,
        "chi" => 0.3,
        "phi" => 0.4,
        "incidence" => 0.1,
        "azimuth" => 0.2,
        other => panic!("no value for {other}"),
    };
    let spec = parameters
        .iter()
        .fold(ModeSpec::new(name), |s, p| s.with(p.name, value(p.name)));
    spec.build(geo)
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

/// HKL-6: hkl restarts its solver from the process-global unseeded
/// `rand()`, so a result depends on what solved before it, in any thread.
#[test]
fn hkl_6_same_request_gives_identical_list() {
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
            let first = bits(req.solve());
            let _ = request(&mode, Vector3::new(-1.0, 1.0, 2.0), &current).solve();
            let again = std::thread::scope(|s| s.spawn(|| bits(req.solve())).join().unwrap());
            assert_eq!(first, bits(req.solve()), "{name} {}", mode.name());
            assert_eq!(first, again, "{name} {} in another thread", mode.name());
        }
    }
}

/// HKL-7: hkl's zero-angle kf is +x whatever the source direction, so its
/// inverse is wrong for any other beam. E4CV turned so the beam runs along
/// z, with UB turned alike, has E4CV's solutions.
#[test]
fn hkl_7_inverse_follows_the_beam_direction() {
    // x → z, z → −x, y kept
    let turn = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -FRAC_PI_2);
    let [omega, chi, phi, tth] = [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let turned = Geometry::builder("E4CV along z")
        .sample_rotation("omega", omega)
        .sample_rotation("chi", chi)
        .sample_rotation("phi", phi)
        .detector_rotation("tth", tth)
        .beam_direction([0.0, 0.0, 1.0])
        .build()
        .unwrap();
    assert!((turn * Vector3::x() - Vector3::z()).norm() < 1e-15);
    assert!((turn * Vector3::z() + Vector3::x()).norm() < 1e-15);
    let e4cv = preset("E4CV").unwrap();
    let current = current(&e4cv);
    let mut solved = 0;
    for (mode, turned_mode) in modes(&e4cv).iter().zip(modes(&turned)) {
        for t in TARGETS {
            let hkl = Vector3::from(t);
            let want = request(mode, hkl, &current).solve();
            let got = InverseRequest {
                ub: turn.to_rotation_matrix().into_inner() * ub(),
                ..request(&turned_mode, hkl, &current)
            }
            .solve();
            let close = match (&want, &got) {
                (Ok(w), Ok(g)) => {
                    solved += 1;
                    w.len() == g.len()
                        && w.iter()
                            .zip(g)
                            .all(|(a, b)| a.iter().zip(b).all(|(x, y)| wrap(x - y).abs() < 1e-9))
                }
                (w, g) => w == g,
            };
            assert!(close, "{} {hkl}: {want:?} vs {got:?}", mode.name());
        }
    }
    // 21 of the 24 solve in E4CV
    assert!(solved >= 21, "{solved}");
}

/// HKL-8: hkl's K4CV eulerians list items keep the sample quaternion from
/// before the solve, 1.50 and 2.15 away from the solution's for (10, 100,
/// 30) and (10, −60, 200). A solution's sample rotation is the E4CV
/// rotation of the target eulerians. (10, 100, 30) has χ = 2α, where the
/// two kappa images meet: a multiple root, `Degenerate` here.
#[test]
fn hkl_8_eulerians_solutions_turn_the_sample_to_the_target() {
    let geo = preset("K4CV").unwrap();
    let e4cv = preset("E4CV").unwrap();
    let arm = ["komega", "kappa", "kphi"].map(|n| geo.axis_index(n).unwrap());
    let mode = Mode::new_kappa(&geo, "eulerians", arm, &arm, vec![]).unwrap();
    let current = current(&geo);
    for target in [[10.0, 80.0, 30.0], [10.0, -60.0, 200.0]] {
        let eulerians = target.map(f64::to_radians);
        let want = e4cv
            .poses(&[eulerians[0], eulerians[1], eulerians[2], 0.0])
            .unwrap()
            .sample
            .rotation;
        let solutions = InverseRequest {
            target: Target::Eulerians(eulerians),
            ..request(&mode, Vector3::x(), &current)
        }
        .solve()
        .unwrap();
        assert!(!solutions.is_empty());
        for s in &solutions {
            let got = geo.poses(s).unwrap().sample.rotation;
            assert!(got.angle_to(&want) < 1e-12, "{target:?} {s:?}");
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

/// HKL-17: hkl checks |UB·h| ≤ 2k only in modes whose operations table
/// has `set`, so (7 0 0) for a = 4.1 at 1.54 Å is "unreachable" in E4C but
/// "none of the functions were solved" in 10 E6C, 5 K4CV and 12 K6C modes.
#[test]
fn hkl_17_unreachable_target_is_err_for_every_mode() {
    let hkl = Vector3::new(7.0, 0.0, 0.0);
    let q = (ub() * hkl).norm();
    let q_max = 2.0 * wavelength().wavenumber();
    let unreachable = |mode: Result<Mode, InverseError>, current: &[f64]| {
        let mode = mode.unwrap();
        assert_eq!(
            request(&mode, hkl, current).solve(),
            Err(InverseError::Unreachable { q, q_max }),
            "{}",
            mode.name()
        );
    };
    let four = [10.0, 20.0, 30.0, 40.0];
    let six = [5.0, 10.0, 20.0, 30.0, 15.0, 40.0];
    for (name, at) in [
        ("E4CV", &four[..]),
        ("E4CH", &four),
        ("E6C", &six),
        ("K4CV", &four),
        ("K6C", &six),
    ] {
        let geo = preset(name).unwrap();
        let at = geo.values_from_degrees(at).unwrap();
        for mode in ModeTable::of(name).unwrap().names() {
            unreachable(table_mode(&geo, mode, Vector3::z(), 0.4), &at);
        }
    }
}

/// PY2-7: hklpy2's th_tth backend takes asin(qλ/4π) unchecked, so q = 13 at
/// λ = 1 escapes as `ValueError: math domain error` instead of "no
/// solution". hklrs has no th_tth geometry (TwoC has no modes); a Q-sample
/// target beyond 2k is unreachable in every E4C mode.
#[test]
fn py2_7_q_beyond_two_k_is_unreachable() {
    let wavelength = Wavelength::new(1.0).unwrap();
    let q_max = 2.0 * wavelength.wavenumber();
    for name in ["E4CV", "E4CH"] {
        let geo = preset(name).unwrap();
        let current = current(&geo);
        for mode in modes(&geo) {
            let req = InverseRequest {
                wavelength,
                target: Target::QSample(Vector3::new(13.0, 0.0, 0.0)),
                ..request(&mode, Vector3::x(), &current)
            };
            assert_eq!(
                req.solve(),
                Err(InverseError::Unreachable { q: 13.0, q_max }),
                "{name} {}",
                mode.name()
            );
        }
    }
}

/// |Q| within 1e-12 of 2k, relative, is solved as |Q| = 2k: a double root
/// in the detector angle there, so Degenerate; beyond it is Unreachable.
#[test]
fn backscatter_limit_has_a_relative_tolerance() {
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let k = wavelength().wavenumber();
    let direction = Vector3::new(0.3, -0.5, 0.8).normalize();
    for mode in [
        e4c::bissector(&geo).unwrap(),
        e4c::constant_phi(&geo).unwrap(),
    ] {
        let at = |excess: f64| {
            InverseRequest {
                target: Target::QSample(direction * (2.0 * k * (1.0 + excess))),
                ..request(&mode, Vector3::zeros(), &current)
            }
            .solve()
        };
        for excess in [0.0, 1e-15, 1e-13] {
            assert_eq!(
                at(excess),
                Err(InverseError::Degenerate),
                "{} {excess}",
                mode.name()
            );
        }
        assert!(
            matches!(at(1e-11), Err(InverseError::Unreachable { .. })),
            "{}",
            mode.name()
        );
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
    // The count is the request's: the target gives three equations, or
    // none for a detector target.
    let four = Mode::new(&geo, "m", &[0, 1, 2, 3], vec![]).unwrap();
    let values = current(&geo);
    let counted = |target| {
        InverseRequest {
            target,
            ..request(&four, Vector3::new(1.0, 0.0, 0.0), &values)
        }
        .solve()
    };
    assert_eq!(
        counted(Target::Hkl(Vector3::new(1.0, 0.0, 0.0))),
        Err(InverseError::EquationCount {
            written: 4,
            equations: 3
        })
    );
    assert_eq!(
        counted(Target::Detector),
        Err(InverseError::EquationCount {
            written: 4,
            equations: 0
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

/// With the whole arm written, every solution's kappa twin is one too;
/// with kphi held, no twin is (it would move kphi) and kphi never moves.
#[test]
fn kappa_twin_is_admitted_unless_it_moves_a_held_axis() {
    let geo = preset("K4CV").unwrap();
    let current = current(&geo);
    let hkl = Vector3::new(1.0, 2.0, 1.0);
    let held = [
        Eulerian::Omega(0.3),
        Eulerian::Chi(-0.6),
        Eulerian::Phi(-0.4),
        Eulerian::Bisector { detector: 3 },
    ];
    for e in held {
        let mode = Mode::new_kappa(
            &geo,
            "m",
            [0, 1, 2],
            &[0, 1, 2, 3],
            vec![Equation::Kappa(e)],
        )
        .unwrap();
        let arm = mode.kappa().unwrap().arm();
        let solutions = request(&mode, hkl, &current).solve().unwrap();
        for s in &solutions {
            assert_solution(&mode, &hkl, &current, s);
            let [ko, ka, kp] = arm.twin([s[0], s[1], s[2]]);
            let twin = [ko, ka, kp, s[3]];
            assert!(
                solutions
                    .iter()
                    .any(|t| t.iter().zip(&twin).all(|(a, b)| wrap(a - b).abs() <= 1e-9)),
                "{e:?}: twin of {s:?} missing"
            );
        }
    }
    let mode = Mode::new_kappa(&geo, "m", [0, 1, 2], &[0, 1, 3], vec![]).unwrap();
    let solutions = request(&mode, hkl, &current).solve().unwrap();
    for s in &solutions {
        assert_solution(&mode, &hkl, &current, s);
    }
}

/// The arm alone written, an eulerians target turns it to both kappa
/// images of the target's rotation: each reads back to the target on its
/// branch and every other axis stays. chi counts modulo 2π (HKL-19: hkl
/// takes 350°, −300° and 450° as they are).
#[test]
fn eulerians_target_turns_the_arm_to_both_kappa_images() {
    for name in ["K4CV", "K6C"] {
        let geo = preset(name).unwrap();
        let arm = ["komega", "kappa", "kphi"].map(|n| geo.axis_index(n).unwrap());
        let mode = Mode::new_kappa(&geo, "eulerians", arm, &arm, vec![]).unwrap();
        let kappa = mode.kappa().unwrap().arm();
        let current: Vec<f64> = (0..geo.axes().len())
            .map(|i| 0.2 + 0.1 * i as f64)
            .collect();
        let solve = |eulerians| {
            InverseRequest {
                target: Target::Eulerians(eulerians),
                ..request(&mode, Vector3::x(), &current)
            }
            .solve()
        };
        let (omega, phi) = (0.7, -2.2);
        for chi_deg in [-80.0, 35.0, 350.0, -300.0, 450.0] {
            let chi = f64::to_radians(chi_deg);
            let solutions = solve([omega, chi, phi]).unwrap();
            assert_eq!(solutions.len(), 2, "{name} {chi_deg}: {solutions:?}");
            let mut branches = Vec::new();
            for s in &solutions {
                let at = arm.map(|i| s[i]);
                let branch = [Branch::Same, Branch::Opposite]
                    .into_iter()
                    .find(|&b| {
                        let back = kappa.to_eulerian(at, b);
                        back.iter()
                            .zip([omega, chi, phi])
                            .all(|(a, b)| wrap(a - b).abs() <= 1e-12)
                    })
                    .unwrap_or_else(|| panic!("{name} {chi_deg}: {s:?} reads back to neither"));
                branches.push(branch);
                for (i, (v, c)) in s.iter().zip(&current).enumerate() {
                    assert!(arm.contains(&i) || v == c, "{name}: axis {i} moved");
                }
            }
            branches.sort_by_key(|b| *b == Branch::Opposite);
            assert_eq!(
                branches,
                [Branch::Same, Branch::Opposite],
                "{name} {chi_deg}"
            );
        }
        // HKL-19 at the request: chi modulo 2π is one target.
        let (a, b) = [350.0, -10.0]
            .map(|c| solve([omega, f64::to_radians(c), phi]).unwrap())
            .into();
        assert!(
            a.len() == b.len()
                && a.iter()
                    .zip(&b)
                    .all(|(x, y)| x.iter().zip(y).all(|(u, v)| (u - v).abs() <= 1e-12)),
            "{a:?} vs {b:?}"
        );
        assert_eq!(
            solve([omega, 2.0, phi]),
            Err(InverseError::ChiOutOfReach {
                chi: 2.0,
                reach: 2.0 * KAPPA_ALPHA_DEG.to_radians()
            })
        );
        let plain = Mode::new(&geo, "m", &arm, vec![]).unwrap();
        assert_eq!(
            InverseRequest {
                target: Target::Eulerians([omega, 0.3, phi]),
                ..request(&plain, Vector3::x(), &current)
            }
            .solve(),
            Err(InverseError::NoKappaArm)
        );
    }
}

#[test]
fn kappa_mode_construction_errors() {
    let geo = preset("K4CV").unwrap();
    let all = [0, 1, 2, 3];
    let held = |e| vec![Equation::Kappa(e)];
    let chi = |v| held(Eulerian::Chi(v));
    let not_arm = |names: [&str; 3]| {
        Err(InverseError::NotKappaArm {
            names: names.map(String::from),
        })
    };
    assert_eq!(
        Mode::new_kappa(&geo, "m", [2, 1, 0], &all, chi(0.0)),
        not_arm(["kphi", "kappa", "komega"])
    );
    assert_eq!(
        Mode::new_kappa(&geo, "m", [1, 2, 3], &all, chi(0.0)),
        not_arm(["kappa", "kphi", "tth"])
    );
    assert_eq!(
        Mode::new(&geo, "m", &all, chi(0.0)),
        Err(InverseError::NoKappaArm)
    );
    let reach = 2.0 * KAPPA_ALPHA_DEG.to_radians();
    assert!(matches!(
        Mode::new_kappa(&geo, "m", [0, 1, 2], &all, chi(-1.8)),
        Err(InverseError::ChiOutOfReach { chi: -1.8, reach: r }) if (r - reach).abs() < 1e-15
    ));
    assert_eq!(
        Mode::new_kappa(&geo, "m", [0, 1, 2], &all, held(Eulerian::Omega(f64::NAN))),
        Err(InverseError::NonFinite {
            what: "eulerian angle"
        })
    );
    assert_eq!(
        Mode::new_kappa(
            &geo,
            "m",
            [0, 1, 2],
            &all,
            held(Eulerian::Bisector { detector: 0 })
        ),
        Err(InverseError::NotInStack {
            name: "komega".into(),
            holder: "detector"
        })
    );
    let arm = |komega: [f64; 3], kappa: [f64; 3]| {
        Geometry::builder("k")
            .sample_rotation("komega", komega)
            .sample_rotation("kappa", kappa)
            .sample_rotation("kphi", [0.0, -1.0, 0.0])
            .detector_rotation("tth", [0.0, -1.0, 0.0])
            .build()
            .unwrap()
    };
    let tilted = kappa_axis(Deg(KAPPA_ALPHA_DEG).into());
    let crossed = arm([0.0, 0.0, 1.0], tilted);
    assert_eq!(
        Mode::new_kappa(&crossed, "m", [0, 1, 2], &all, chi(0.0)),
        not_arm(["komega", "kappa", "kphi"])
    );
    // A kappa turning the other way (XU-3) is tilted by π − α.
    let reversed = arm([0.0, -1.0, 0.0], tilted.map(|c| -c));
    assert!(matches!(
        Mode::new_kappa(&reversed, "m", [0, 1, 2], &all, chi(0.0)),
        Err(InverseError::KappaAlpha { .. })
    ));
}

/// An E6C mode on mu, omega, gamma, delta holding the emergence: every
/// solution leaves the surface at it; ±π/2 are emergences, beyond is an
/// error.
#[test]
fn emergence_is_held_and_bounded() {
    let geo = preset("E6C").unwrap();
    let written = ["mu", "omega", "gamma", "delta"].map(|n| geo.axis_index(n).unwrap());
    let mode = |surface: Vector3<f64>, emergence: f64| {
        Mode::new(
            &geo,
            "emergence",
            &written,
            vec![Equation::Emergence { surface, emergence }],
        )
    };
    let surface = Vector3::new(0.2, 1.0, -0.3);
    let current = geo
        .values_from_degrees(&[5.0, 10.0, 20.0, 30.0, 15.0, 40.0])
        .unwrap();
    let mut solved = 0;
    for e in [-0.4, 0.1, 0.5] {
        let m = mode(surface, e).unwrap();
        for t in TARGETS {
            let hkl = Vector3::from(t);
            match request(&m, hkl, &current).solve() {
                Ok(solutions) => {
                    solved += solutions.len();
                    for s in &solutions {
                        assert_solution(&m, &hkl, &current, s);
                    }
                }
                Err(InverseError::NoSolution) => {}
                Err(e) => panic!("{hkl}: {e}"),
            }
        }
    }
    assert!(solved > 0);
    assert!(mode(surface, FRAC_PI_2).is_ok());
    assert!(mode(surface, -FRAC_PI_2).is_ok());
    assert_eq!(
        mode(surface, next_out(FRAC_PI_2)),
        Err(InverseError::EmergenceRange {
            emergence: next_out(FRAC_PI_2)
        })
    );
    assert_eq!(
        mode(surface, f64::NAN),
        Err(InverseError::NonFinite { what: "emergence" })
    );
    assert_eq!(
        mode(Vector3::zeros(), 0.1),
        Err(InverseError::ZeroReference {
            what: "emergence surface"
        })
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

/// HKL-25: the solve's thresholds are dimensionless, so one problem in
/// units of k (UB scaled with k) gives the same solutions at λ = 0.1 Å and
/// 50 Å.
#[test]
fn solutions_depend_on_q_in_units_of_k_only() {
    let geo = preset("E4CV").unwrap();
    let current = current(&geo);
    let mut solved = 0;
    for mode in modes(&geo) {
        let [a, b] = [0.1, 50.0].map(|lambda| {
            let wavelength = Wavelength::new(lambda).unwrap();
            InverseRequest {
                wavelength,
                ub: ub() * (wavelength.wavenumber() / self::wavelength().wavenumber()),
                ..request(&mode, Vector3::new(1.0, 2.0, 1.0), &current)
            }
            .solve()
        });
        let (Ok(a), Ok(b)) = (&a, &b) else {
            assert_eq!(a, b, "{}", mode.name());
            continue;
        };
        solved += 1;
        assert_eq!(a.len(), b.len(), "{}", mode.name());
        for (x, y) in a.iter().zip(b) {
            let d = inverse_check::largest(x.iter().zip(y).map(|(x, y)| (x - y).abs()));
            assert!(d <= 1e-12, "{}: {x:?} vs {y:?}", mode.name());
        }
    }
    assert!(solved >= 4, "{solved}");
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
    let geo = preset("ESRF BM28 PSIC").unwrap();
    let current = geo
        .values_from_degrees(&[5.0, 10.0, 20.0, 30.0, 15.0, 40.0])
        .unwrap();
    for mode in [
        esrf_bm28_psic::constant_nu_coplanar(&geo),
        esrf_bm28_psic::constant_delta_coplanar(&geo),
        esrf_bm28_psic::constant_eta_noncoplanar(&geo),
    ] {
        degenerate(&mode.unwrap(), &current);
    }
    let geo = preset("ZAXIS").unwrap();
    let current = geo.values_from_degrees(&[5.0, 10.0, 20.0, 30.0]).unwrap();
    for mode in [zaxis::zaxis(&geo), zaxis::reflectivity(&geo)] {
        degenerate(&mode.unwrap(), &current);
    }
    let geo = preset("ESRF ID01 PSIC").unwrap();
    let current = geo
        .values_from_degrees(&[5.0, 10.0, 20.0, 30.0, 15.0])
        .unwrap();
    for mode in [
        esrf_id01_psic::constant_nu_coplanar(&geo),
        esrf_id01_psic::constant_delta_coplanar(&geo),
        esrf_id01_psic::constant_eta_noncoplanar(&geo),
    ] {
        degenerate(&mode.unwrap(), &current);
    }
    let geo = preset("SOLEIL SIXS MED2+2").unwrap();
    let current = geo
        .values_from_degrees(&[5.0, 10.0, 20.0, 30.0, 15.0])
        .unwrap();
    for mode in [
        soleil_sixs_med::mu_fixed(&geo),
        soleil_sixs_med::reflectivity(&geo),
        soleil_sixs_med::emergence_fixed(&geo, Vector3::y(), 0.2),
    ] {
        degenerate(&mode.unwrap(), &current);
    }
    let geo = preset("SOLEIL SIXS MED1+2").unwrap();
    let current = geo.values_from_degrees(&[5.0, 10.0, 20.0, 30.0]).unwrap();
    for mode in [
        soleil_sixs_med::pitch_fixed(&geo),
        soleil_sixs_med::delta_fixed(&geo),
    ] {
        degenerate(&mode.unwrap(), &current);
    }
    for (name, degrees) in med_2_3() {
        let geo = preset(name).unwrap();
        let current = geo.values_from_degrees(degrees).unwrap();
        for mode in med_2_3_modes(&geo) {
            degenerate(&mode, &current);
            degenerate(&soleil_sixs_med::slits(mode).unwrap(), &current);
        }
    }
}

/// MED2+3 and its v2 with a current position.
fn med_2_3() -> [(&'static str, &'static [f64]); 2] {
    [
        ("SOLEIL SIXS MED2+3", &[7.0, 10.0, 20.0, 30.0, 15.0, 40.0]),
        ("SOLEIL SIXS MED2+3 v2", &[10.0, 20.0, 30.0, 15.0, 40.0]),
    ]
}

fn med_2_3_modes(geo: &Geometry) -> Vec<Mode<'_>> {
    vec![
        soleil_sixs_med::mu_fixed(geo).unwrap(),
        soleil_sixs_med::gamma_fixed(geo).unwrap(),
        soleil_sixs_med::emergence_fixed(geo, Vector3::y(), 0.2).unwrap(),
    ]
}

/// The slit fit doubles every solution of the mode without it: the same
/// other axes, eta_a at two roots π apart, each turning the slits into the
/// surface.
#[test]
fn slits_double_every_solution() {
    for (name, degrees) in med_2_3() {
        let geo = preset(name).unwrap();
        let current = geo.values_from_degrees(degrees).unwrap();
        let eta_a = geo.axis_index("eta_a").unwrap();
        let mut solved = 0;
        for base in med_2_3_modes(&geo) {
            let slits = soleil_sixs_med::slits(base.clone()).unwrap();
            for t in TARGETS {
                let hkl = Vector3::from(t);
                let (held, fitted) = match (
                    request(&base, hkl, &current).solve(),
                    request(&slits, hkl, &current).solve(),
                ) {
                    (Ok(held), Ok(fitted)) => (held, fitted),
                    (Err(InverseError::NoSolution), Err(InverseError::NoSolution)) => continue,
                    other => panic!("{name} {} {hkl}: {other:?}", base.name()),
                };
                assert_eq!(fitted.len(), 2 * held.len(), "{name} {}", base.name());
                for s in &fitted {
                    assert_solution(&slits, &hkl, &current, s);
                    let other_axes = |b: &Vec<f64>| {
                        (0..s.len())
                            .filter(|&i| i != eta_a)
                            .all(|i| (b[i] - s[i]).abs() <= 1e-9)
                    };
                    assert_eq!(held.iter().filter(|b| other_axes(b)).count(), 1);
                    let twins = fitted.iter().filter(|f| {
                        other_axes(f) && (wrap(f[eta_a] - s[eta_a]).abs() - PI).abs() <= 1e-9
                    });
                    assert_eq!(twins.count(), 1, "{name} {} {s:?}", base.name());
                }
                solved += fitted.len();
            }
        }
        assert!(solved > 0, "{name}");
    }
}

/// With the surface normal along k̂f, so along eta_a, every eta_a turns the
/// slits into the surface: the solution there keeps eta_a at `current`
/// rather than making the whole solve degenerate, and the other solutions
/// still come in root pairs.
#[test]
fn slits_along_the_surface_hold_at_current() {
    let geo = preset("SOLEIL SIXS MED2+3").unwrap();
    let mode = soleil_sixs_med::slits(soleil_sixs_med::gamma_fixed(&geo).unwrap()).unwrap();
    let eta_a = geo.axis_index("eta_a").unwrap();
    // mu - gamma = 90°, delta = 0: R_s·(0, -1, 0) is k̂f.
    let current = geo
        .values_from_degrees(&[7.0, 30.0, 20.0, -60.0, 0.0, 40.0])
        .unwrap();
    let miss = |values: &[f64]| {
        inverse_check::residuals(&mode, &ub(), wavelength(), &Vector3::zeros(), values).unwrap()
    };
    let hkl = Vector3::from_iterator(miss(&current)[..3].iter().copied());
    for eta in [-2.0, 0.5, 3.0] {
        let mut v = current.clone();
        v[eta_a] = eta;
        assert!(miss(&v)[3].abs() <= 1e-15);
    }
    let solutions = request(&mode, hkl, &current).solve().unwrap();
    let at_current = |s: &Vec<f64>| (0..s.len()).all(|i| (s[i] - current[i]).abs() <= 1e-9);
    let (here, others): (Vec<_>, Vec<_>) = solutions.iter().partition(|s| at_current(s));
    assert_eq!(here.len(), 1, "{solutions:?}");
    assert_eq!(here[0][eta_a], current[eta_a]);
    assert!(!others.is_empty());
    for s in &solutions {
        assert_solution(&mode, &hkl, &current, s);
    }
    for s in &others {
        let twins = others.iter().filter(|f| {
            (0..s.len()).all(|i| i == eta_a || (f[i] - s[i]).abs() <= 1e-9)
                && (wrap(f[eta_a] - s[eta_a]).abs() - PI).abs() <= 1e-9
        });
        assert_eq!(twins.count(), 1, "{s:?}");
    }
}

#[test]
fn slits_construction_errors() {
    let geo = preset("SOLEIL SIXS MED2+3").unwrap();
    let index = |n| geo.axis_index(n).unwrap();
    let mode = soleil_sixs_med::mu_fixed(&geo).unwrap();
    let with = |axis, surface, normal| mode.clone().with_slits(axis, surface, normal);
    let (y, z) = (Vector3::y(), Vector3::z());
    assert_eq!(
        with(index("mu"), y, z),
        Err(InverseError::NotInStack {
            name: "mu".into(),
            holder: "detector"
        })
    );
    assert_eq!(
        with(index("gamma"), y, z),
        Err(InverseError::SlitsMoveDetector {
            name: "gamma".into()
        })
    );
    assert_eq!(
        with(index("eta_a"), Vector3::zeros(), z),
        Err(InverseError::ZeroReference {
            what: "slits surface"
        })
    );
    assert_eq!(
        with(index("eta_a"), y, Vector3::new(0.0, 0.0, f64::NAN)),
        Err(InverseError::NonFinite {
            what: "slits normal"
        })
    );
    let written = Mode::new(
        &geo,
        "m",
        &[index("omega"), index("delta"), index("eta_a")],
        vec![],
    );
    assert_eq!(
        written.unwrap().with_slits(index("eta_a"), y, z),
        Err(InverseError::WrittenTwice {
            name: "eta_a".into()
        })
    );
    // MED2+2's innermost detector axis, delta, turns about -y, not the beam.
    let geo = preset("SOLEIL SIXS MED2+2").unwrap();
    let mode = soleil_sixs_med::mu_fixed(&geo).unwrap();
    let delta = geo.axis_index("delta").unwrap();
    assert_eq!(
        mode.with_slits(delta, y, z),
        Err(InverseError::SlitsMoveDetector {
            name: "delta".into()
        })
    );
}

/// ESRF BM28 PSIC constant_nu_coplanar from all-zero axes, hkl's fresh
/// geometry: chi = 0 makes eta and phi coaxial and nu = 0 keeps Q in the
/// lab xz plane, so only Q ⟂ y (the phi axis) is reached, with eta − phi
/// free. hkl's failures there are this geometry, not its solver.
#[test]
fn bm28_coplanar_at_chi_zero_reaches_only_q_normal_to_phi() {
    let geo = preset("ESRF BM28 PSIC").unwrap();
    let mode = esrf_bm28_psic::constant_nu_coplanar(&geo).unwrap();
    // Q in Å⁻¹ straight from the target.
    let ub = Matrix3::identity();
    let solve = |q: &Vector3<f64>, current: &[f64]| {
        InverseRequest {
            ub,
            ..request(&mode, *q, current)
        }
        .solve()
    };
    let (normal, oblique) = (Vector3::new(1.5, 0.0, 1.0), Vector3::new(1.5, 0.5, 1.0));
    let zero = [0.0; 6];
    assert_eq!(solve(&normal, &zero), Err(InverseError::Degenerate));
    assert_eq!(solve(&oblique, &zero), Err(InverseError::NoSolution));
    let chi = geo
        .values_from_degrees(&[0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
        .unwrap();
    let solutions = solve(&oblique, &chi).unwrap();
    assert!(!solutions.is_empty());
    for s in &solutions {
        inverse_check::check(&mode, &ub, wavelength(), &oblique, &chi, s).unwrap();
    }
}

/// zaxis + beta-fixed of `name`, whose axes 0 and 5 are α and β with the
/// surface normal along z of the second sample axis (chi = phi = 0), from a
/// position hkl's mode cannot hold: every solution keeps β and moves α, and
/// the position itself is among them.
fn beta_fixed_holds_the_exit_angle(
    name: &str,
    mode: fn(&Geometry) -> Result<Mode<'_>, InverseError>,
) {
    let geo = preset(name).unwrap();
    let mode = mode(&geo).unwrap();
    let beam = geo.beam_direction().into_inner();
    let at = geo
        .values_from_degrees(&[3.0, 40.0, 0.0, 0.0, 25.0, 7.0])
        .unwrap();
    let q = scattering(&geo, &at, wavelength(), &Point3::from(beam))
        .unwrap()
        .q_sample;
    let hkl = ub().lu().solve(&q).unwrap();
    let current = geo
        .values_from_degrees(&[10.0, 20.0, 0.0, 0.0, 10.0, 7.0])
        .unwrap();
    let solutions = request(&mode, hkl, &current).solve().unwrap();
    assert!(
        solutions
            .iter()
            .any(|s| s.iter().zip(&at).all(|(a, b)| wrap(a - b).abs() <= 1e-9)),
        "{solutions:?}"
    );
    for s in &solutions {
        assert_solution(&mode, &hkl, &current, s);
        let poses = geo.poses(s).unwrap();
        let n = poses.sample.rotation * Vector3::z();
        let (ki, kf) = (poses.source * beam, poses.detector.rotation * beam);
        assert!((n.dot(&kf) - current[5].sin()).abs() <= 1e-12, "{s:?}");
        assert!((-n.dot(&ki) - s[0].sin()).abs() <= 1e-12, "{s:?}");
    }
}

/// tau, mu, chi, phi, gamma, delta: β = delta, which hkl 5.1.9 writes.
#[test]
fn polar_beta_fixed_holds_the_exit_angle() {
    beta_fixed_holds_the_exit_angle("APS POLAR", aps_04id_polar::zaxis_beta_fixed);
}

/// mu, omega, chi, phi, delta, gamma: β = gamma, which hkl 5.1.9 writes.
#[test]
fn p09_beta_fixed_holds_the_exit_angle() {
    beta_fixed_holds_the_exit_angle("PETRA3 P09 EH2", petra3_p09_eh2::zaxis_beta_fixed);
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

/// phi, which constant_phi holds at current's 30°, against limits that
/// keep it exactly or miss it by one ulp.
#[test]
fn held_axis_is_bound_by_its_limits() {
    let mut geo = preset("E4CV").unwrap();
    let hkl = Vector3::new(0.0, 1.0, 1.0);
    let all = solve_with(&geo, e4c::constant_phi, hkl).unwrap();
    let on = current(&geo)[geo.axis_index("phi").unwrap()];
    geo.set_limits("phi", on, on).unwrap();
    assert_eq!(solve_with(&geo, e4c::constant_phi, hkl), Ok(all.clone()));
    for (min, max) in [
        (next_out(on), f64::INFINITY),
        (f64::NEG_INFINITY, next_in(on)),
    ] {
        geo.set_limits("phi", min, max).unwrap();
        assert_eq!(
            solve_with(&geo, e4c::constant_phi, hkl),
            Err(InverseError::NoSolutionInLimits(all.clone())),
            "[{min}, {max}]"
        );
    }
}

/// phi held at 270°, outside the cut window [−180°, 180°): it comes back
/// as 270°, and its limits are checked there, not at −90°.
#[test]
fn held_axis_is_not_wrapped_to_its_cut_point() {
    let mut geo = preset("E4CV").unwrap();
    let phi = geo.axis_index("phi").unwrap();
    geo.set_cut_point("phi", -PI).unwrap();
    let hkl = Vector3::new(0.0, 1.0, 1.0);
    let current = geo.values_from_degrees(&[10.0, 20.0, 270.0, 40.0]).unwrap();
    let solve = |geo: &Geometry| {
        let mode = e4c::constant_phi(geo).unwrap();
        request(&mode, hkl, &current).solve()
    };
    geo.set_limits("phi", PI, TAU).unwrap();
    let solutions = solve(&geo).unwrap();
    assert!(!solutions.is_empty());
    let mode = e4c::constant_phi(&geo).unwrap();
    for s in &solutions {
        assert_eq!(s[phi].to_bits(), current[phi].to_bits(), "{s:?}");
        assert_solution(&mode, &hkl, &current, s);
    }
    geo.set_limits("phi", -PI, 0.0).unwrap();
    assert_eq!(
        solve(&geo),
        Err(InverseError::NoSolutionInLimits(solutions))
    );
}

type Verdict = Result<Vec<Vec<f64>>, InverseError>;

/// `mode` solved for `hkl` from all zero, where Q is zero or along the beam
/// at every sector image of it, and from every written axis turned.
fn from_zero_and_turned(mode: &Mode, hkl: [f64; 3]) -> [Verdict; 2] {
    let zero = vec![0.0; mode.geometry().axes().len()];
    let mut turned = zero.clone();
    for (j, &w) in mode.written().iter().enumerate() {
        turned[w] = 0.3 - 0.37 * j as f64;
    }
    [zero, turned].map(|current| request(mode, Vector3::from(hkl), &current).solve())
}

/// The same set up to 2π per axis, or the same error.
fn same_verdict(a: &Verdict, b: &Verdict) -> bool {
    let within = |x: &[Vec<f64>], y: &[Vec<f64>]| {
        x.iter().all(|x| {
            y.iter()
                .any(|y| x.iter().zip(y).all(|(u, v)| wrap(u - v).abs() < 1e-6))
        })
    };
    match (a, b) {
        (Ok(a), Ok(b)) => a.len() == b.len() && within(a, b) && within(b, a),
        (a, b) => a == b,
    }
}

/// The ψ modes of every preset, hkl engine and psi engine, reach the same
/// set from all zero, where ψ is undefined, as from a turned current.
#[test]
fn psi_modes_reach_the_same_set_from_zero() {
    let mut failed = Vec::new();
    let (mut modes, mut solved) = (0, 0);
    for &name in PRESETS {
        let geo = preset(name).unwrap();
        let table = ModeTable::of(name).unwrap();
        let psi_constant: Vec<_> = table.psi_modes().collect();
        for reference in [Vector3::z(), Vector3::new(1.0, 1.0, 0.0)] {
            for psi in [-2.5, 0.4] {
                let built = psi_constant
                    .iter()
                    .map(|m| table_mode(&geo, m, reference, psi))
                    .chain(table.psi().map(|engine| engine(&geo, reference, psi)));
                for mode in built {
                    let mode = mode.unwrap();
                    modes += 1;
                    for hkl in TARGETS {
                        let [zero, turned] = from_zero_and_turned(&mode, hkl);
                        if !same_verdict(&zero, &turned) {
                            failed.push(format!(
                                "{name} {} {reference:?} {psi} {hkl:?}:\nzero {zero:?}\nturned {turned:?}",
                                mode.name()
                            ));
                        }
                        solved += usize::from(zero.is_ok());
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    // 23 modes: the psi_constant* of 11 presets and 9 psi engines
    assert_eq!(modes, 23 * 4);
    assert!(solved > 0);
}

/// A surface normal along the beam has no azimuth about it at any sector
/// image of all zero; K6C constant_incidence reaches the same set from there
/// as from a turned current.
#[test]
fn incidence_along_the_beam_reaches_the_same_set_from_zero() {
    let geo = preset("K6C").unwrap();
    let normal = geo.beam_direction().into_inner();
    let mut solved = 0;
    for (incidence, azimuth) in [(0.2, 0.5), (-0.3, 2.0), (0.05, -1.0)] {
        let mode = k6c::constant_incidence(&geo, k6c::K6C, normal, incidence, azimuth).unwrap();
        for hkl in TARGETS {
            let [zero, turned] = from_zero_and_turned(&mode, hkl);
            assert!(
                same_verdict(&zero, &turned),
                "{incidence} {azimuth} {hkl:?}:\nzero {zero:?}\nturned {turned:?}"
            );
            solved += usize::from(zero.is_ok());
        }
    }
    assert!(solved > 0);
}
