//! Inverse invariants on E4CV, E4CH and K4CV: every solution forwards back
//! to the target and satisfies its mode, held axes never move, the search is
//! deterministic, axis limits only filter the solutions after the cut-point
//! wrap, out-of-sphere targets and bad inputs are errors.

mod inverse_check;

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use hklrs::inverse::{
    self, aps_04id_polar, e4c, e6c, esrf_bm28_psic, esrf_id01_psic, petra3_p09_eh2,
    soleil_sixs_med, zaxis,
};
use hklrs::nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use hklrs::presets::{KAPPA_ALPHA_DEG, kappa_axis, preset};
use hklrs::{
    Deg, Equation, Eulerian, Geometry, InverseError, InverseRequest, Lattice, Mode, Picker, Target,
    Wavelength, scattering,
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
