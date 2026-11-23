//! Closed-form candidates. For a Q goal, on every geometry [`map`] lays
//! onto You (1999) 4S+2D, a kappa arm through its eulerians: the constraint
//! families solved as diffcalc-core (`059ec8a`, `src/diffcalc/hkl/calc*.py`)
//! solves them, with its defects corrected (upstream-defects.md
//! DC-1..DC-6). For a detector goal, the detector arms of [`detector`].
//!
//! [`search`] is the one boundary: it maps the mode's [`System`] onto a
//! closed form (You's constraints through a [`Map`]) and offers what it
//! finds to the validator, as every generator does. Where an equation
//! leaves an angle free the solver samples it, and the validator finds the
//! family singular.

mod detector;
mod map;
mod sample;
mod solve;
mod trig;
mod you;

use std::f64::consts::PI;

use nalgebra::Vector3;

use super::kappa::{Branch, KappaAxes};
use super::system::{Goal, System};
use super::validate::SolutionSet;
use super::{Equation, Eulerian};
use map::{Arm, Circle, Map, Source};
use sample::Known;

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

/// The kappa arm as its eulerian cradle (omega, chi, phi), as the mode
/// moves it.
#[derive(Clone, Copy, Debug)]
struct Cradle {
    kappa: KappaAxes,
    /// With kappa held, the eulerians at `current`: chi stays, and omega
    /// and phi turn with komega and kphi. `None` with all three written,
    /// where each eulerian is free but as an equation holds it and either
    /// kappa image of a solution's is one.
    held: Option<[f64; 3]>,
}

impl Cradle {
    /// `None` where kappa is written and komega or kphi held, which turns
    /// omega or phi with kappa: no circle of You's.
    fn new(system: &System, kappa: KappaAxes) -> Option<Self> {
        let axes = kappa.axes();
        let held = match axes.map(|i| system.is_written(i)) {
            [true, true, true] => None,
            [_, false, _] => {
                let at = axes.map(|i| system.current()[i]);
                Some(kappa.arm().to_eulerian(at, Branch::Same))
            }
            _ => return None,
        };
        Some(Self { kappa, held })
    }

    /// Eulerian `e` where kappa is held and so is its axis.
    fn at(&self, system: &System, e: usize) -> Option<f64> {
        let held = self.held?;
        (!system.is_written(self.kappa.axes()[e])).then_some(held[e])
    }

    /// `values` with the arm at `eulerians`: both kappa images, those
    /// within reach, or komega and kphi turned as far from `current` as
    /// the eulerians are from `held`.
    fn arms(&self, system: &System, eulerians: [f64; 3], values: Vec<f64>) -> Vec<Vec<f64>> {
        let axes = self.kappa.axes();
        let Some(held) = self.held else {
            return [Branch::Same, Branch::Opposite]
                .into_iter()
                .filter_map(|b| self.kappa.arm().to_kappa(eulerians, b).ok())
                .map(|k| {
                    let mut values = values.clone();
                    for (&i, v) in axes.iter().zip(k) {
                        values[i] = v;
                    }
                    values
                })
                .collect();
        };
        let mut values = values;
        for e in [0, 2].into_iter().filter(|&e| system.is_written(axes[e])) {
            values[axes[e]] += eulerians[e] - held[e];
        }
        vec![values]
    }
}

/// The eulerians `held` stands for, up to the image (omega + π, −chi,
/// phi + π) of each, which turns the sample alike: every held one after
/// the first also at its other value mod π (omega, phi) or sign (chi).
fn images(held: [Option<f64>; 3]) -> Vec<[Option<f64>; 3]> {
    let mut out = vec![held];
    for e in (0..3).filter(|&e| held[e].is_some()).skip(1) {
        let other = |v: f64| if e == 1 { -v } else { v + PI };
        let turned: Vec<_> = out
            .iter()
            .map(|h| {
                let mut h = *h;
                h[e] = h[e].map(other);
                h
            })
            .collect();
        out.extend(turned);
    }
    out
}

/// The mode as You's constraints on `map`: the circles it holds, at
/// `current` through their axes, the kappa arm's eulerians (each of their
/// [`images`]) or fixed, and its equations, which must make one family.
/// Every written axis must be the map's.
fn constraints(system: &System, map: &Map, cradle: Option<&Cradle>) -> Option<Vec<Constraints>> {
    if !system.written().iter().all(|&i| map.carries(i)) {
        return None;
    }
    let unit = |v: &Vector3<f64>| (map.phi * v).try_normalize(0.0);
    let (mut det, mut refs, mut eulerians) = (Vec::new(), Vec::new(), [None; 3]);
    let (mut bisect, mut omega) = (false, None);
    let chi = |axis: usize| map.sample[you::CHI].source == Some(Source::Axis(axis));
    for e in system.extra() {
        match *e {
            Equation::Kappa(held) if cradle?.held.is_none() => {
                let (e, v) = match held {
                    Eulerian::Omega(v) => (0, v),
                    Eulerian::Chi(v) => (1, v),
                    Eulerian::Phi(v) => (2, v),
                    Eulerian::Bisector { .. } => return None,
                };
                if eulerians[e].replace(v).is_some() {
                    return None;
                }
            }
            Equation::Qaz { qaz } => det.push(Detector::Qaz(qaz + map.azimuth?)),
            Equation::Naz { normal, naz } => {
                det.push(Detector::Naz(unit(&normal)?, naz + map.azimuth?))
            }
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
            Equation::InScatteringPlane { axis } if chi(axis) && !bisect => bisect = true,
            Equation::VirtualOmega { axis, omega: w } if chi(axis) && omega.is_none() => {
                // The χ axis turned the other way is π from the axis's.
                omega = Some(if map.sample[you::CHI].sign < 0.0 {
                    w + PI
                } else {
                    w
                })
            }
            _ => return None,
        }
    }
    let fixed = |c: &Circle, eulerians: &[Option<f64>; 3]| match c.source {
        Some(Source::Axis(i)) if system.is_written(i) => None,
        Some(Source::Axis(i)) => Some(c.angle(system.current()[i])),
        Some(Source::Eulerian(e)) => {
            let at = cradle.and_then(|cradle| cradle.at(system, e));
            at.or(eulerians[e]).map(|v| c.angle(v))
        }
        None => Some(c.offset),
    };
    match &map.arm {
        Arm::Circles([nu, delta]) => {
            det.extend(fixed(nu, &eulerians).map(Detector::Nu));
            det.extend(fixed(delta, &eulerians).map(Detector::Delta));
        }
        // You's detector constraints are on NU and DELTA alone.
        Arm::Own(own) if own.iter().all(|&i| system.is_written(i)) => {}
        Arm::Own(_) => return None,
    }
    let family = |eulerians: [Option<f64>; 3]| {
        let known = map.sample.map(|c| fixed(&c, &eulerians));
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
    };
    images(eulerians).into_iter().map(family).collect()
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
/// You's under a [`Map`]. The reach gate has passed: `q` is nonzero and no
/// longer than 2k.
fn you_search<'a>(system: &'a System<'a>, q: &Vector3<f64>) -> Option<SolutionSet<'a>> {
    let map = Map::new(system.geometry(), system.kappa())?;
    let cradle = match map.kappa {
        Some(kappa) => Some(Cradle::new(system, kappa)?),
        None => None,
    };
    let families = constraints(system, &map, cradle.as_ref())?;
    let own = match map.arm {
        Arm::Own(own) => Some((detector::Pair::new(system.geometry(), own)?, own)),
        Arm::Circles(_) => None,
    };
    let mut set = SolutionSet::new(system);
    let problem = solve::Problem::new(&(map.phi * q));
    for a in families.iter().flat_map(|c| problem.candidates(c)) {
        let (mut values, mut eulerians) = (system.current().to_vec(), [0.0; 3]);
        for (c, circle) in map.circles() {
            match circle.source {
                Some(Source::Axis(i)) if system.is_written(i) => values[i] = circle.value(a[c]),
                Some(Source::Eulerian(e)) => eulerians[e] = circle.value(a[c]),
                _ => {}
            }
        }
        let arms = match &cradle {
            Some(cradle) => cradle.arms(system, eulerians, values),
            None => vec![values],
        };
        let positions = arms.into_iter().flat_map(|values| match &own {
            None => vec![values],
            Some((pair, [outer, inner])) => {
                let kf = map.lab.transpose() * you::kout(&a);
                pair.reach(&values, &kf)
                    .into_iter()
                    .map(|[v1, v2]| {
                        let mut values = values.clone();
                        values[*outer] = v1;
                        values[*inner] = v2;
                        values
                    })
                    .collect()
            }
        });
        for image in positions.flat_map(|v| system.slit_images(v)) {
            if set.is_degenerate() {
                return Some(set);
            }
            set.offer(&image);
        }
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
    use crate::presets::{PRESETS, preset};
    use crate::{Geometry, Wavelength};
    use nalgebra::Matrix3;
    use std::f64::consts::FRAC_PI_2;

    fn same_sets(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
        let within = |x: &[Vec<f64>], y: &[Vec<f64>]| {
            x.iter().all(|x| {
                y.iter()
                    .any(|y| x.iter().zip(y).all(|(u, v)| wrap(u - v).abs() < 1e-6))
            })
        };
        a.len() == b.len() && within(a, b) && within(b, a)
    }

    /// hklrs's E6C lab (beam +x, `mu` and `gamma` about z) as You's lab
    /// (beam +y, μ and ν about x): x → y, y → z, z → x.
    fn frame() -> Matrix3<f64> {
        Matrix3::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// The map of `geo`, its arm komega, kappa, kphi where it has one.
    fn map_of(geo: &Geometry) -> Option<Map> {
        let arm = ["komega", "kappa", "kphi"].map(|n| geo.axis_index(n));
        let kappa = match arm {
            [Ok(o), Ok(k), Ok(p)] => Some(KappaAxes::new(geo, [o, k, p]).unwrap()),
            _ => None,
        };
        Map::new(geo, kappa.as_ref())
    }

    /// Every preset lays onto You's circles but P23 6C (five sample
    /// rotations), PYFAI (three detector rotations) and LISA (a source
    /// axis): E6C and BM28 PSIC as they are, E4CH turned, ID01 with χ fixed
    /// at π/2, ZAXIS through a detector arm of its own, K6C's kappa arm as
    /// E6C's cradle.
    #[test]
    fn map_lays_presets_on_you() {
        let unmapped: Vec<_> = PRESETS
            .iter()
            .filter(|&&name| map_of(&preset(name).unwrap()).is_none())
            .collect();
        assert_eq!(unmapped, [&"PETRA3 P23 6C", &"PYFAI", &"PETRA3 P08 LISA"]);
        let fixed = |offset| Circle {
            source: None,
            sign: 1.0,
            offset,
        };
        let map = |name| {
            let geo = preset(name).unwrap();
            let on = move |axis, sign| Circle {
                source: Some(Source::Axis(geo.axis_index(axis).unwrap())),
                sign,
                offset: 0.0,
            };
            (map_of(&preset(name).unwrap()).unwrap(), on)
        };
        let you = [0, 1, 2, 3, 4, 5].map(|i| Circle {
            source: Some(Source::Axis(i)),
            sign: 1.0,
            offset: 0.0,
        });
        for name in ["E6C", "ESRF BM28 PSIC"] {
            let (map, _) = map(name);
            assert_eq!(map.lab, frame());
            assert_eq!(map.phi, frame());
            assert_eq!(map.sample, you[..4]);
            assert_eq!(map.arm, Arm::Circles([you[4], you[5]]));
            assert_eq!(map.azimuth, Some(0.0));
        }
        let (k6c, _) = map("K6C");
        assert_eq!((k6c.lab, k6c.phi), (frame(), frame()));
        let eulerian = |e| Circle {
            source: Some(Source::Eulerian(e)),
            ..you[0]
        };
        assert_eq!(k6c.sample, [you[0], eulerian(0), eulerian(1), eulerian(2)]);
        assert_eq!(k6c.arm, Arm::Circles([you[4], you[5]]));
        let (e4ch, on) = map("E4CH");
        let turn = Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
        assert_eq!(e4ch.lab, turn);
        let sample = [fixed(0.0), on("omega", 1.0), on("chi", 1.0), on("phi", 1.0)];
        assert_eq!(e4ch.sample, sample);
        assert_eq!(e4ch.arm, Arm::Circles([fixed(0.0), on("tth", 1.0)]));
        assert_eq!(e4ch.azimuth, Some(FRAC_PI_2));
        let (id01, on) = map("ESRF ID01 PSIC");
        assert_eq!(id01.lab, frame());
        let sample = [
            on("mu", -1.0),
            on("eta", 1.0),
            fixed(FRAC_PI_2),
            on("phi", 1.0),
        ];
        assert_eq!(id01.sample, sample);
        assert_eq!(id01.arm, Arm::Circles([on("nu", -1.0), on("delta", 1.0)]));
        let (zaxis, on) = map("ZAXIS");
        let sample = [on("mu", 1.0), on("omega", 1.0), fixed(0.0), fixed(0.0)];
        assert_eq!(zaxis.sample, sample);
        let own = ["delta", "gamma"].map(|a| preset("ZAXIS").unwrap().axis_index(a).unwrap());
        assert_eq!(zaxis.arm, Arm::Own(own));
    }

    /// Every hkl "hkl" engine mode of preset `name`, labelled, its
    /// parameters arbitrary; MED2+3's also with the slit fit.
    fn hkl_modes<'g>(name: &str, geo: &'g Geometry) -> Vec<(String, Mode<'g>)> {
        use super::super::{
            aps_04id_polar as polar, e4c, k4c, k6c, petra3_p09_eh2 as p09, petra3_p23_4c as p23,
            soleil_nanoscopium_robot as robot, soleil_sirius_turret as turret,
            soleil_sixs_med as med, zaxis,
        };
        let (h2, reference, psi) = (
            Vector3::new(1.0, -1.0, 0.5),
            Vector3::new(0.2, 0.3, 1.0),
            0.4,
        );
        let surface = Vector3::new(0.1, 1.0, 0.2);
        let modes = match name {
            "TwoC" => vec![],
            "K4CV" | "SOLEIL CRISTAL K4C" => vec![
                k4c::bissector(geo),
                k4c::constant_omega(geo, 0.3),
                k4c::constant_chi(geo, 0.5),
                k4c::constant_phi(geo, -0.4),
                k4c::double_diffraction(geo, h2),
                k4c::psi_constant(geo, reference, psi),
            ],
            "K6C" | "SOLEIL SIRIUS KAPPA" => vec![
                k6c::bissector_vertical(geo),
                k6c::constant_omega_vertical(geo, 0.3),
                k6c::constant_chi_vertical(geo, 0.5),
                k6c::constant_phi_vertical(geo, -0.4),
                k6c::lifting_detector_kphi(geo),
                k6c::lifting_detector_komega(geo),
                k6c::lifting_detector_mu(geo),
                k6c::double_diffraction_vertical(geo, h2),
                k6c::bissector_horizontal(geo),
                k6c::constant_phi_horizontal(geo, -0.4),
                k6c::constant_kphi_horizontal(geo),
                k6c::double_diffraction_horizontal(geo, h2),
                k6c::psi_constant_vertical(geo, reference, psi),
                k6c::constant_incidence(geo, surface, 0.2, 0.7),
                k6c::psi_vertical_soleil_sirius_kappa(geo, reference, psi),
            ],
            "E4CV" | "E4CH" | "SOLEIL MARS" => vec![
                e4c::bissector(geo),
                e4c::constant_omega(geo),
                e4c::constant_chi(geo),
                e4c::constant_phi(geo),
                e4c::double_diffraction(geo, h2),
                e4c::psi_constant(geo, reference, psi),
            ],
            "E6C" => vec![
                e6c::bissector_vertical(geo),
                e6c::constant_omega_vertical(geo),
                e6c::constant_chi_vertical(geo),
                e6c::constant_phi_vertical(geo),
                e6c::lifting_detector_phi(geo),
                e6c::lifting_detector_omega(geo),
                e6c::lifting_detector_mu(geo),
                e6c::double_diffraction_vertical(geo, h2),
                e6c::bissector_horizontal(geo),
                e6c::double_diffraction_horizontal(geo, h2),
                e6c::psi_constant_vertical(geo, reference, psi),
                e6c::psi_constant_horizontal(geo, reference, psi),
                e6c::constant_mu_horizontal(geo),
            ],
            "ESRF BM28 PSIC" | "ESRF ID01 PSIC" => vec![
                esrf_bm28_psic::constant_nu_coplanar(geo),
                esrf_bm28_psic::constant_delta_coplanar(geo),
                esrf_bm28_psic::constant_eta_noncoplanar(geo),
            ],
            "ZAXIS" => vec![zaxis::zaxis(geo), zaxis::reflectivity(geo)],
            "SOLEIL SIXS MED1+2" => vec![med::pitch_fixed(geo), med::delta_fixed(geo)],
            "SOLEIL SIXS MED2+2" => vec![
                med::mu_fixed(geo),
                med::reflectivity(geo),
                med::emergence_fixed(geo, surface, -0.3),
            ],
            "SOLEIL SIXS MED2+3" | "SOLEIL SIXS MED2+3 v2" => [
                med::mu_fixed(geo),
                med::gamma_fixed(geo),
                med::emergence_fixed(geo, surface, -0.3),
            ]
            .into_iter()
            .flat_map(|m| {
                let m = m.unwrap();
                [Ok(m.clone()), med::slits(m)]
            })
            .collect(),
            "APS POLAR" => vec![
                polar::constant_phi_horizontal(geo),
                polar::zaxis_alpha_fixed(geo),
                polar::zaxis_beta_fixed(geo),
                polar::zaxis_alpha_eq_beta(geo),
                polar::bissecting_horizontal(geo),
                polar::constant_mu_horizontal(geo),
                polar::constant_chi_horizontal(geo),
                polar::lifting_detector_tau(geo),
                polar::lifting_detector_mu(geo),
                polar::lifting_detector_chi(geo),
                polar::lifting_detector_phi(geo),
                polar::psi_constant_horizontal(geo, reference, psi),
                polar::psi_constant_vertical(geo, reference, psi),
            ],
            "PETRA3 P09 EH2" => vec![
                p09::zaxis_alpha_fixed(geo),
                p09::zaxis_beta_fixed(geo),
                p09::zaxis_alpha_eq_beta(geo),
                p09::bissecting_horizontal(geo),
                p09::constant_omega_horizontal(geo),
                p09::constant_chi_horizontal(geo),
                p09::constant_phi_horizontal(geo),
                p09::lifting_detector_mu(geo),
                p09::lifting_detector_omega(geo),
                p09::lifting_detector_chi(geo),
                p09::lifting_detector_phi(geo),
            ],
            "PETRA3 P23 4C" => vec![
                p23::bissector_vertical(geo),
                p23::lifting_detector_omega_t(geo),
                p23::lifting_detector_mu(geo),
                p23::bissector_horizontal(geo),
                p23::psi_constant(geo, reference, psi),
            ],
            "SOLEIL SIRIUS TURRET" => vec![turret::lifting_detector_thetah(geo)],
            "SOLEIL NANOSCOPIUM ROBOT" => vec![
                robot::lifting_detector_rz(geo),
                robot::lifting_detector_rs(geo),
                robot::lifting_detector_rx(geo),
            ],
            other => panic!("{other} maps but has no modes here"),
        };
        modes
            .into_iter()
            .map(Result::unwrap)
            .map(|m| {
                let slits = if m.slits().is_some() { " slits" } else { "" };
                (format!("{}{slits}", m.name()), m)
            })
            .collect()
    }

    /// The closed form's set against the numeric search's for `mode`, from
    /// three held positions and five targets, the last at 2θ near 1.3°:
    /// per case `None` where the closed form refuses the mode, else whether
    /// a position solves it (the same positions within 1e-6) or the same
    /// error, or the disagreement.
    fn cases(mode: &Mode) -> Vec<Option<Result<bool, String>>> {
        let ub = nalgebra::Rotation3::from_euler_angles(0.3, -0.2, 0.5).into_inner() * 1.5;
        let targets = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.3, -0.4, 1.1),
            Vector3::new(1.2, 0.8, -0.5),
            Vector3::new(2.5, 1.0, 1.5),
            Vector3::new(0.002, -0.012, -0.019),
        ];
        let n = mode.geometry().axes().len();
        let turned: Vec<f64> = (0..n).map(|i| 0.3 - 0.17 * i as f64).collect();
        // Held χ, ν or δ far from zero, where the modes holding them reach.
        let far = [0.6, 1.2, -0.9, 0.8, 1.0, -0.7, 0.4];
        let far: Vec<f64> = (0..n).map(|i| far[i % far.len()]).collect();
        let mut out = Vec::new();
        for current in [vec![0.0; n], turned, far] {
            for h in targets {
                let request = InverseRequest {
                    mode,
                    ub,
                    wavelength: Wavelength::new(1.54).unwrap(),
                    target: Target::Hkl(h),
                    current: &current,
                };
                let system = request.system().unwrap();
                out.push(search(&system).map(|ours| {
                    let [ours, theirs] = [ours, numeric::search(&system)]
                        .map(|s| s.into_sorted().map(|f| f.solutions));
                    match (&ours, &theirs) {
                        (Ok(a), Ok(b)) if same_sets(a, b) => Ok(true),
                        (Err(a), Err(b)) if a == b => Ok(false),
                        _ => Err(format!(
                            "{current:?} {h:?}:\nanalytic {ours:?}\nnumeric {theirs:?}"
                        )),
                    }
                }));
            }
        }
        out
    }

    /// On every preset that maps, every hkl mode that is one of You's
    /// families agrees with the numeric search in every case of [`cases`]
    /// and solves at least one. The modes that are not tie a sample axis to
    /// a detector one (`Bisector`, `Equal`, the kappa arm's), add a second
    /// reflection or an azimuth to the incidence, hold one of a detector
    /// arm's own two axes, which no NU or DELTA constraint is, or hold kphi
    /// with kappa written, which no eulerian is.
    #[test]
    fn preset_modes_agree_with_numeric() {
        let mut refused = std::collections::BTreeSet::new();
        let (mut mapped, mut failed, mut unsolved) = (0, Vec::new(), Vec::new());
        for &name in PRESETS {
            let geo = preset(name).unwrap();
            if map_of(&geo).is_none() {
                continue;
            }
            for (label, mode) in hkl_modes(name, &geo) {
                let label = format!("{name}: {label}");
                let cases = cases(&mode);
                if cases.iter().any(Option::is_none) {
                    refused.insert(label);
                    continue;
                }
                mapped += 1;
                let cases = cases.into_iter().flatten();
                let (solved, errors): (Vec<_>, Vec<_>) = cases.partition(Result::is_ok);
                failed.extend(
                    errors
                        .into_iter()
                        .map(|e| format!("{label} {}", e.unwrap_err())),
                );
                if !solved.into_iter().any(|s| s == Ok(true)) {
                    unsolved.push(label);
                }
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
        assert!(unsolved.is_empty(), "{unsolved:?}");
        assert_eq!(mapped, 86);
        let refused: Vec<_> = refused.iter().map(String::as_str).collect();
        assert_eq!(
            refused,
            [
                "APS POLAR: 4-circles bissecting horizontal",
                "APS POLAR: 4-circles constant chi horizontal",
                "APS POLAR: 4-circles constant mu horizontal",
                "APS POLAR: 4-circles constant phi horizontal",
                "APS POLAR: psi constant horizontal",
                "APS POLAR: psi constant vertical",
                "APS POLAR: zaxis + alpha=beta",
                "APS POLAR: zaxis + beta-fixed",
                "E4CH: bissector",
                "E4CH: double_diffraction",
                "E4CV: bissector",
                "E4CV: double_diffraction",
                "E6C: bissector_horizontal",
                "E6C: bissector_vertical",
                "E6C: double_diffraction_horizontal",
                "E6C: double_diffraction_vertical",
                "K4CV: bissector",
                "K4CV: double_diffraction",
                "K6C: bissector_horizontal",
                "K6C: bissector_vertical",
                "K6C: constant_incidence",
                "K6C: constant_kphi_horizontal",
                "K6C: double_diffraction_horizontal",
                "K6C: double_diffraction_vertical",
                "PETRA3 P09 EH2: 4-circles bissecting horizontal",
                "PETRA3 P09 EH2: 4-circles constant chi horizontal",
                "PETRA3 P09 EH2: 4-circles constant omega horizontal",
                "PETRA3 P09 EH2: 4-circles constant phi horizontal",
                "PETRA3 P09 EH2: zaxis + alpha=beta",
                "PETRA3 P09 EH2: zaxis + beta-fixed",
                "SOLEIL CRISTAL K4C: bissector",
                "SOLEIL CRISTAL K4C: double_diffraction",
                "SOLEIL MARS: bissector",
                "SOLEIL MARS: double_diffraction",
                "SOLEIL SIRIUS KAPPA: bissector_horizontal",
                "SOLEIL SIRIUS KAPPA: bissector_vertical",
                "SOLEIL SIRIUS KAPPA: constant_incidence",
                "SOLEIL SIRIUS KAPPA: constant_kphi_horizontal",
                "SOLEIL SIRIUS KAPPA: double_diffraction_horizontal",
                "SOLEIL SIRIUS KAPPA: double_diffraction_vertical",
                "SOLEIL SIXS MED1+2: delta_fixed",
                "SOLEIL SIXS MED2+2: reflectivity",
                "SOLEIL SIXS MED2+3: gamma_fixed",
                "SOLEIL SIXS MED2+3: gamma_fixed slits",
                "ZAXIS: reflectivity",
            ]
        );
    }

    /// The equations no hkl mode of a mapped preset gives, through maps that
    /// turn them: qaz and naz on each preset whose azimuth is not zero, the
    /// arm and the last two sample circles written; bisect with virtual
    /// omega on E6C with χ reversed, which maps χ turned the other way.
    #[test]
    fn turned_equations_agree_with_numeric() {
        let reversed = Geometry::builder("E6C, chi reversed")
            .sample_rotation("mu", [0.0, 0.0, 1.0])
            .sample_rotation("omega", [0.0, -1.0, 0.0])
            .sample_rotation("chi", [-1.0, 0.0, 0.0])
            .sample_rotation("phi", [0.0, -1.0, 0.0])
            .detector_rotation("gamma", [0.0, 0.0, 1.0])
            .detector_rotation("delta", [0.0, -1.0, 0.0])
            .build()
            .unwrap();
        assert_eq!(map_of(&reversed).unwrap().sample[you::CHI].sign, -1.0);
        let azimuths = [
            ("SOLEIL NANOSCOPIUM ROBOT", ["rs", "rx", "delta", "gamma"]),
            ("APS POLAR", ["chi", "phi", "gamma", "delta"]),
            ("PETRA3 P09 EH2", ["chi", "phi", "delta", "gamma"]),
            ("SOLEIL SIXS MED1+2", ["pitch", "mu", "gamma", "delta"]),
            (
                "SOLEIL SIRIUS TURRET",
                ["alphay", "alphax", "delta", "gamma"],
            ),
        ]
        .map(|(name, written)| (preset(name).unwrap(), written));
        let normal = Vector3::new(0.2, 0.3, 1.0);
        let mut modes = Vec::new();
        for (geo, written) in &azimuths {
            assert_ne!(map_of(geo).unwrap().azimuth, Some(0.0));
            modes.push(Mode::named(geo, "qaz", written, |_| {
                vec![Equation::Qaz { qaz: 0.4 }]
            }));
            modes.push(Mode::named(geo, "naz", written, |_| {
                vec![Equation::Naz { normal, naz: -2.1 }]
            }));
        }
        let bisect = ["mu", "omega", "chi", "phi", "gamma"];
        modes.push(Mode::named(&reversed, "bisect omega", &bisect, |w| {
            vec![
                Equation::InScatteringPlane { axis: w[2] },
                Equation::VirtualOmega {
                    axis: w[2],
                    omega: 0.3,
                },
            ]
        }));
        for mode in modes.into_iter().map(Result::unwrap) {
            let label = format!("{} {}", mode.geometry().name(), mode.name());
            let cases: Vec<_> = cases(&mode).into_iter().map(|c| c.expect(&label)).collect();
            let failed: Vec<_> = cases.iter().filter_map(|c| c.clone().err()).collect();
            assert!(failed.is_empty(), "{label}: {}", failed.join("\n"));
            assert!(cases.contains(&Ok(true)), "{label}");
        }
    }

    /// On the four eulerians presets, from a held position with every axis
    /// turned, the two kappa images are the numeric search's set; at chi = 0
    /// komega and kphi are one axis and both find the set degenerate.
    #[test]
    fn arm_agrees_with_numeric() {
        for name in ["K4CV", "K6C", "SOLEIL SIRIUS KAPPA", "SOLEIL CRISTAL K4C"] {
            let geometry = preset(name).unwrap();
            let mode = super::super::pseudo::eulerians(&geometry).unwrap();
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

    /// With the arm written, eulerians within |chi| ≤ 2α come back as both
    /// kappa images, each reading back on its branch; past 2α, as none.
    #[test]
    fn cradle_drops_chi_past_reach() {
        let geometry = preset("K4CV").unwrap();
        let mode = super::super::k4c::psi_constant(&geometry, Vector3::z(), 0.4).unwrap();
        let current = vec![0.0; geometry.axes().len()];
        let request = InverseRequest {
            mode: &mode,
            ub: Matrix3::identity(),
            wavelength: Wavelength::new(1.54).unwrap(),
            target: Target::Hkl(Vector3::x()),
            current: &current,
        };
        let system = request.system().unwrap();
        let kappa = map_of(&geometry).unwrap().kappa.unwrap();
        let cradle = Cradle::new(&system, kappa).unwrap();
        let (reach, arm, axes) = (kappa.arm().reach(), kappa.arm(), kappa.axes());
        for chi in [reach - 1e-9, 1e-9 - reach] {
            let eulerians = [0.3, chi, -0.2];
            let images = cradle.arms(&system, eulerians, current.clone());
            assert_eq!(images.len(), 2, "{chi}");
            for (values, branch) in images.iter().zip([Branch::Same, Branch::Opposite]) {
                let back = arm.to_eulerian(axes.map(|i| values[i]), branch);
                assert!(
                    back.iter()
                        .zip(eulerians)
                        .all(|(a, b)| wrap(a - b).abs() < 1e-9)
                );
            }
        }
        for chi in [reach + 1e-9, -reach - 1e-9] {
            assert!(
                cradle
                    .arms(&system, [0.3, chi, -0.2], current.clone())
                    .is_empty()
            );
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
