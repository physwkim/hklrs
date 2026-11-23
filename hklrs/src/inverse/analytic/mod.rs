//! Closed-form candidates. For a Q goal, on the geometries that are You
//! (1999) 4S+2D axis for axis: the constraint families solved as
//! diffcalc-core (`059ec8a`, `src/diffcalc/hkl/calc*.py`) solves them, with
//! its defects corrected (upstream-defects.md DC-1..DC-6). For a detector
//! goal, the detector arms of [`detector`].
//!
//! [`search`] is the one boundary: it maps the mode's [`System`] onto a
//! closed form (You's constraints through [`frame`]) and offers what it
//! finds to the validator, as every generator does. Where an equation
//! leaves an angle free the solver samples it, and the validator finds the
//! family singular.

mod detector;
mod sample;
mod solve;
mod trig;
mod you;

use nalgebra::{Matrix3, Vector3};

use super::Equation;
use super::kappa::Branch;
use super::system::{Goal, System};
use super::validate::SolutionSet;
use crate::{AxisKind, Geometry};
use sample::Known;

/// hklrs's E6C lab (beam +x, `mu` and `gamma` about z) as You's lab (beam
/// +y, μ and ν about x): x → y, y → z, z → x. Lab and sample-frame vectors
/// map by `frame() * v`.
fn frame() -> Matrix3<f64> {
    Matrix3::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
}

/// Fixes the detector, or with `Naz` the azimuth about the beam of a unit
/// vector in the φ frame.
#[derive(Clone, Copy, Debug)]
enum Detector {
    Delta(f64),
    Nu(f64),
    /// Azimuth of Q about the beam, atan2(Q_x, Q_z).
    Qaz(f64),
    /// atan2(v_x, v_z) of the vector in the lab.
    Naz(Vector3<f64>, f64),
}

/// Fixes a unit vector v in the φ frame relative to the beams: You's
/// reference vector (alpha, beta, a_eq_b, psi) or surface normal (betain,
/// betaout, bin_eq_bout).
#[derive(Clone, Copy, Debug)]
enum Reference {
    /// asin(−v·k̂_in).
    Incidence(Vector3<f64>, f64),
    /// asin(v·k̂_out).
    Emergence(Vector3<f64>, f64),
    /// Incidence equals emergence.
    Equal(Vector3<f64>),
    /// Azimuth of v about Q̂, zero when v is in the plane of Q̂ and the beam.
    Psi(Vector3<f64>, f64),
}

impl Reference {
    fn vector(&self) -> Vector3<f64> {
        match *self {
            Reference::Incidence(v, _)
            | Reference::Emergence(v, _)
            | Reference::Equal(v)
            | Reference::Psi(v, _) => v,
        }
    }
}

/// With `bisect`, the χ axis MU·ETA·ŷ lies in the plane of the beam and Q's
/// azimuth (sin qaz, 0, cos qaz); this fixes where in that plane.
#[derive(Clone, Copy, Debug)]
enum Bisect {
    Mu(f64),
    Eta(f64),
    /// Angle of the χ axis from the beam in that plane, minus θ.
    Omega(f64),
}

/// One You (1999) constraint family with its values, radians.
#[derive(Clone, Copy, Debug)]
enum Constraints {
    DetRefSample(Detector, Reference, Known),
    DetSamples(Detector, Known),
    DetBisect(Detector, Bisect),
    RefSamples(Reference, Known),
    Samples(Known),
}

/// You's circle of each axis in geometry order, where the geometry is
/// You's under [`frame`]: four sample rotations about MU, ETA, CHI, PHI's
/// axes and two detector rotations about NU, DELTA's, outermost first, no
/// source axis, and the beam along +y.
fn layout(geometry: &Geometry) -> Option<[usize; 6]> {
    let r = frame();
    let (sample, detector) = (geometry.sample_stack(), geometry.detector_stack());
    if geometry.axes().len() != 6
        || sample.len() != 4
        || detector.len() != 2
        || !geometry.source_stack().is_empty()
        || r * geometry.beam_direction().into_inner() != Vector3::y()
    {
        return None;
    }
    let mut out = [None; 6];
    for (c, &i) in sample.iter().chain(detector).enumerate() {
        let AxisKind::Rotation { axis } = geometry.axes()[i].kind() else {
            return None;
        };
        if r * axis.into_inner() != you::axis(c) || out[i].is_some() {
            return None;
        }
        out[i] = Some(c);
    }
    Some(out.map(Option::unwrap))
}

/// The mode as You's constraints: the axes it holds at `current` and its
/// equations, which must make one family.
fn constraints(system: &System, layout: &[usize; 6]) -> Option<Constraints> {
    let unit = |v: &Vector3<f64>| (frame() * v).try_normalize(0.0);
    let (mut det, mut refs, mut known) = (Vec::new(), Vec::new(), [None; 4]);
    let (mut bisect, mut omega) = (false, None);
    for (i, &c) in layout.iter().enumerate() {
        if system.is_written(i) {
            continue;
        }
        let v = system.current()[i];
        match c {
            you::NU => det.push(Detector::Nu(v)),
            you::DELTA => det.push(Detector::Delta(v)),
            c => known[c] = Some(v),
        }
    }
    for e in system.extra() {
        match *e {
            Equation::Qaz { qaz } => det.push(Detector::Qaz(qaz)),
            Equation::Naz { normal, naz } => det.push(Detector::Naz(unit(&normal)?, naz)),
            Equation::Incidence { normal, incidence } => {
                refs.push(Reference::Incidence(unit(&normal)?, incidence))
            }
            Equation::Emergence { surface, emergence } => {
                refs.push(Reference::Emergence(unit(&surface)?, emergence))
            }
            Equation::IncidenceEqualsEmergence { normal } => {
                refs.push(Reference::Equal(unit(&normal)?))
            }
            Equation::PsiConstant { reference, psi } => {
                refs.push(Reference::Psi(unit(&(system.ub() * reference))?, psi))
            }
            Equation::InScatteringPlane { axis } if layout[axis] == you::CHI && !bisect => {
                bisect = true
            }
            Equation::VirtualOmega { axis, omega: w }
                if layout[axis] == you::CHI && omega.is_none() =>
            {
                omega = Some(w)
            }
            _ => return None,
        }
    }
    let held: Vec<usize> = (you::MU..=you::PHI)
        .filter(|&c| known[c].is_some())
        .collect();
    match (&det[..], &refs[..], held.len(), bisect, omega) {
        (&[d], &[r], 1, false, None) => Some(Constraints::DetRefSample(d, r, known)),
        (&[], &[r], 2, false, None) => Some(Constraints::RefSamples(r, known)),
        (&[], &[], 3, false, None) => Some(Constraints::Samples(known)),
        (&[d], &[], 2, false, None) => Some(Constraints::DetSamples(d, known)),
        (&[d], &[], _, true, _) => {
            let b = match (&held[..], omega) {
                (&[], Some(w)) => Bisect::Omega(w),
                (&[you::MU], None) => Bisect::Mu(known[you::MU]?),
                (&[you::ETA], None) => Bisect::Eta(known[you::ETA]?),
                _ => return None,
            };
            Some(Constraints::DetBisect(d, b))
        }
        _ => None,
    }
}

/// The set of `system` from a closed form of its mode: every candidate
/// offered, written axes as solved and the others exactly `current`, until
/// one is singular. `None` where no closed form applies.
pub(super) fn search<'a>(system: &'a System<'a>) -> Option<SolutionSet<'a>> {
    match system.goal() {
        Goal::Q(q) => you_search(system, q),
        Goal::Detector => detector::search(system),
        Goal::Arm { eulerians, .. } => arm_search(system, eulerians),
    }
}

/// [`search`] where the mode writes the kappa arm alone: both kappa images
/// of `eulerians`.
fn arm_search<'a>(system: &'a System<'a>, eulerians: &[f64; 3]) -> Option<SolutionSet<'a>> {
    let kappa = system.kappa()?;
    let axes = kappa.axes();
    let mut written = system.written().to_vec();
    written.sort_unstable();
    let mut sorted = axes;
    sorted.sort_unstable();
    if written != sorted || !system.extra().is_empty() {
        return None;
    }
    let mut set = SolutionSet::new(system);
    for branch in [Branch::Same, Branch::Opposite] {
        let mut values = system.current().to_vec();
        for (&i, v) in axes
            .iter()
            .zip(kappa.arm().to_kappa(*eulerians, branch).ok()?)
        {
            values[i] = v;
        }
        for image in system.slit_images(values) {
            set.offer(&image);
        }
    }
    Some(set)
}

/// [`search`] by You's solution, where the geometry and the mode are
/// You's. The reach gate has passed: `q` is nonzero and no longer than 2k.
fn you_search<'a>(system: &'a System<'a>, q: &Vector3<f64>) -> Option<SolutionSet<'a>> {
    let layout = layout(system.geometry())?;
    let constraints = constraints(system, &layout)?;
    let mut set = SolutionSet::new(system);
    for a in solve::Problem::new(&(frame() * q)).candidates(&constraints) {
        if set.is_degenerate() {
            break;
        }
        let values: Vec<f64> = (0..6)
            .map(|i| {
                if system.is_written(i) {
                    a[layout[i]]
                } else {
                    system.current()[i]
                }
            })
            .collect();
        set.offer(&values);
    }
    Some(set)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::super::validate::Found;
    use super::super::{
        InverseError, InverseRequest, Mode, Target, e6c, esrf_bm28_psic, solve as numeric, wrap,
    };
    use super::*;
    use crate::Wavelength;
    use crate::presets::{PRESETS, preset};

    /// Of every preset, E6C and ESRF BM28 PSIC are You's geometry, each axis
    /// on its circle in stack order.
    #[test]
    fn layout_maps_e6c_and_bm28_psic() {
        assert_eq!(frame().determinant(), 1.0);
        let mapped: Vec<_> = PRESETS
            .iter()
            .filter_map(|&name| layout(&preset(name).unwrap()).map(|l| (name, l)))
            .collect();
        let you = [you::MU, you::ETA, you::CHI, you::PHI, you::NU, you::DELTA];
        assert_eq!(mapped, [("E6C", you), ("ESRF BM28 PSIC", you)]);
    }

    /// Every hkl mode of the two presets is one of You's families but the
    /// bissector and double-diffraction modes.
    #[test]
    fn preset_modes_map_but_bissector_and_double_diffraction() {
        let (e, b) = (preset("E6C").unwrap(), preset("ESRF BM28 PSIC").unwrap());
        let (h, psi) = (Vector3::new(0.3, -0.2, 1.0), 0.4);
        let modes = [
            (e6c::bissector_vertical(&e), false),
            (e6c::constant_omega_vertical(&e), true),
            (e6c::constant_chi_vertical(&e), true),
            (e6c::constant_phi_vertical(&e), true),
            (e6c::lifting_detector_phi(&e), true),
            (e6c::lifting_detector_omega(&e), true),
            (e6c::lifting_detector_mu(&e), true),
            (e6c::double_diffraction_vertical(&e, h), false),
            (e6c::bissector_horizontal(&e), false),
            (e6c::double_diffraction_horizontal(&e, h), false),
            (e6c::psi_constant_vertical(&e, h, psi), true),
            (e6c::psi_constant_horizontal(&e, h, psi), true),
            (e6c::constant_mu_horizontal(&e), true),
            (esrf_bm28_psic::constant_nu_coplanar(&b), true),
            (esrf_bm28_psic::constant_delta_coplanar(&b), true),
            (esrf_bm28_psic::constant_eta_noncoplanar(&b), true),
        ];
        let current = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        for (mode, maps) in modes {
            let mode = mode.unwrap();
            let system = System::new(
                &mode,
                Matrix3::identity(),
                Goal::Q(Vector3::z()),
                1.0,
                &current,
            );
            let layout = layout(mode.geometry()).unwrap();
            assert_eq!(
                constraints(&system, &layout).is_some(),
                maps,
                "{}",
                mode.name()
            );
        }
    }

    /// On the four eulerians presets, from a held position with every axis
    /// turned, the two kappa images are the numeric search's set; at chi = 0
    /// komega and kphi are one axis and both find the set degenerate.
    #[test]
    fn arm_agrees_with_numeric() {
        for name in ["K4CV", "K6C", "SOLEIL SIRIUS KAPPA", "SOLEIL CRISTAL K4C"] {
            let geometry = preset(name).unwrap();
            let arm = ["komega", "kappa", "kphi"];
            let mode = Mode::named_kappa(&geometry, "eulerians", &arm, |_| vec![]).unwrap();
            let current: Vec<f64> = (0..geometry.axes().len())
                .map(|i| 0.3 - 0.17 * i as f64)
                .collect();
            for eulerians in [[0.4, 0.9, -1.2], [-2.0, -1.5, 3.0], [0.1, 0.0, 0.2]] {
                let request = InverseRequest {
                    mode: &mode,
                    ub: Matrix3::identity(),
                    wavelength: Wavelength::new(1.54).unwrap(),
                    target: Target::Eulerians(eulerians),
                    current: &current,
                };
                let system = request.system().unwrap();
                let ours = search(&system).unwrap().into_sorted();
                let theirs = numeric::search(&system).into_sorted();
                if eulerians[1] == 0.0 {
                    assert!(matches!(ours, Err(InverseError::Degenerate)), "{name}");
                    assert!(matches!(theirs, Err(InverseError::Degenerate)), "{name}");
                    continue;
                }
                let (a, b) = (ours.unwrap().solutions, theirs.unwrap().solutions);
                assert_eq!(a.len(), 2, "{name} {eulerians:?}: {a:?}");
                assert!(
                    a.len() == b.len()
                        && a.iter().zip(&b).all(|(x, y)| {
                            x.iter().zip(y).all(|(u, v)| wrap(u - v).abs() < 1e-9)
                        }),
                    "{name} {eulerians:?}: analytic {a:?}\nnumeric {b:?}"
                );
            }
        }
    }

    fn golden() -> Value {
        let path = format!(
            "{}/../golden/diffcalc_inverse.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).expect(&path);
        serde_json::from_str(&text.replace(": NaN", ": null")).unwrap()
    }

    fn vec3(v: &Value) -> Vector3<f64> {
        Vector3::from_fn(|i, _| v[i].as_f64().unwrap())
    }

    /// A diffcalc golden case (You's names, φ frame, degrees) as a mode of
    /// `geometry`, whose axes are You's circles in order, and its `current`:
    /// the held circles at their values, the written ones at zero, where the
    /// detector sits on the beam; each other constraint is its equation.
    fn pose<'g>(
        geometry: &'g Geometry,
        case: &Value,
    ) -> (Result<Mode<'g>, InverseError>, [f64; 6]) {
        const NAMES: [&str; 6] = ["mu", "eta", "chi", "phi", "nu", "delta"];
        let (r, chi) = (frame().transpose(), you::CHI);
        let ub = Matrix3::from_fn(|i, j| case["UB"][i][j].as_f64().unwrap());
        let n_phi = vec3(&case["n_phi"]);
        let (n, s) = (r * n_phi, r * vec3(&case["surf_nphi"]));
        let (mut current, mut extra, mut written) = ([0.0; 6], Vec::new(), vec![0, 1, 2, 3, 4, 5]);
        for (name, value) in case["constraints"].as_object().unwrap() {
            let v = || value.as_f64().unwrap().to_radians();
            if let Some(i) = NAMES.iter().position(|n| n == name) {
                current[i] = v();
                written.retain(|&w| w != i);
                continue;
            }
            extra.push(match name.as_str() {
                "qaz" => Equation::Qaz { qaz: v() },
                "naz" => Equation::Naz {
                    normal: n,
                    naz: v(),
                },
                "alpha" => Equation::Incidence {
                    normal: n,
                    incidence: v(),
                },
                "beta" => Equation::Emergence {
                    surface: n,
                    emergence: v(),
                },
                "a_eq_b" => Equation::IncidenceEqualsEmergence { normal: n },
                "psi" => Equation::PsiConstant {
                    reference: ub.lu().solve(&n_phi).unwrap(),
                    psi: v(),
                },
                "betain" => Equation::Incidence {
                    normal: s,
                    incidence: v(),
                },
                "betaout" => Equation::Emergence {
                    surface: s,
                    emergence: v(),
                },
                "bin_eq_bout" => Equation::IncidenceEqualsEmergence { normal: s },
                "bisect" => Equation::InScatteringPlane { axis: chi },
                "omega" => Equation::VirtualOmega {
                    axis: chi,
                    omega: v(),
                },
                other => panic!("constraint {other}"),
            });
        }
        (Mode::new(geometry, "golden", &written, extra), current)
    }

    /// On both presets every diffcalc golden case that maps comes to the same
    /// set from You's candidates as from the numeric search, each offered to
    /// the validator: the same error, or the same positions within 1e-6. The
    /// sets diffcalc refuses do not map.
    #[test]
    fn analytic_agrees_with_numeric() {
        let golden = golden();
        let mut agreed = std::collections::BTreeMap::new();
        let mut failed = Vec::new();
        for name in ["E6C", "ESRF BM28 PSIC"] {
            let geometry = preset(name).unwrap();
            for case in golden["cases"].as_array().unwrap() {
                let (id, category) = (&case["id"], case["category"].as_str().unwrap());
                let (mode, current) = pose(&geometry, case);
                let Ok(mode) = mode else {
                    assert_eq!(category, "unsupported", "{id}");
                    continue;
                };
                let request = InverseRequest {
                    mode: &mode,
                    ub: frame().transpose()
                        * Matrix3::from_fn(|i, j| case["UB"][i][j].as_f64().unwrap()),
                    wavelength: Wavelength::new(case["wavelength"].as_f64().unwrap()).unwrap(),
                    target: Target::Hkl(vec3(&case["hkl"])),
                    current: &current,
                };
                let system = match request.system() {
                    Ok(system) => system,
                    // Q = 0 is Degenerate before any solver.
                    Err(e) => {
                        let zero = vec3(&case["hkl"]) == Vector3::zeros();
                        assert!(
                            match e {
                                InverseError::Unreachable { .. } => category == "unreachable",
                                InverseError::EquationCount { .. } => category == "unsupported",
                                InverseError::Degenerate => zero,
                                _ => false,
                            },
                            "{id}: {e}"
                        );
                        continue;
                    }
                };
                let Some(ours) = search(&system) else {
                    assert_eq!(category, "unsupported", "{id}");
                    continue;
                };
                assert_ne!(category, "unsupported", "{id}");
                let (ours, theirs) = (ours.into_sorted(), numeric::search(&system).into_sorted());
                let kind = match (&ours, &theirs) {
                    (Ok(a), Ok(b)) => {
                        let within = |x: &[Vec<f64>], y: &[Vec<f64>]| {
                            x.iter().all(|x| {
                                y.iter()
                                    .any(|y| x.iter().zip(y).all(|(u, v)| wrap(u - v).abs() < 1e-6))
                            })
                        };
                        let (a, b) = (&a.solutions, &b.solutions);
                        if a.len() != b.len() || !within(a, b) || !within(b, a) {
                            failed.push(format!("{name} {id}: analytic {a:?}\nnumeric {b:?}"));
                            continue;
                        }
                        "solutions"
                    }
                    (Err(a), Err(b)) if a == b => match a {
                        InverseError::Degenerate => "degenerate",
                        InverseError::NoSolution => "none",
                        _ => "other error",
                    },
                    _ => {
                        let n = |r: &Result<Found, InverseError>| {
                            r.as_ref().map(|f| f.solutions.len()).map_err(Clone::clone)
                        };
                        failed.push(format!(
                            "{name} {id} {category}: analytic {:?}, numeric {:?}",
                            n(&ours),
                            n(&theirs)
                        ));
                        continue;
                    }
                };
                *agreed.entry((name, kind)).or_insert(0) += 1;
            }
        }
        eprintln!("analytic vs numeric: {agreed:?}, failed {}", failed.len());
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
