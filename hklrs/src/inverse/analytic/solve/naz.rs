//! `naz` with two sample circles or with `bisect`, which diffcalc-core
//! documents and rejects (DC-1).

use std::f64::consts::{PI, TAU};

use nalgebra::{Matrix3, Vector3};

use super::super::super::validate::ACCEPT;
use super::super::Bisect;
use super::super::sample::{self, Known};
use super::super::you::{Angles, CHI, EPS, PHI, Roots, axis, circle, rot, solve_component, spin};
use super::Problem;

/// Grid points per period where φ is scanned.
const SCAN: usize = 1440;

impl Problem {
    /// `naz` of unit vector `n` (φ frame) with two sample circles held.
    pub(super) fn naz_samples(&self, n: &Vector3<f64>, naz: f64, known: &Known) -> Vec<Angles> {
        let level = Vector3::new(naz.cos(), 0.0, -naz.sin());
        let eqs = [(self.p0, Vector3::y(), -self.st), (*n, level, 0.0)];
        self.free_detector(sample::two_held_heights(known, &eqs))
    }

    /// With bisect the χ axis lies at an angle t from the beam in the plane
    /// of the beam and Q̂, so Q̂ sits at height sin(t − θ) along it. ω fixes
    /// t = θ + ω and that height fixes φ; a held μ or η leaves φ to a scan.
    pub(super) fn naz_bisect(&self, n: &Vector3<f64>, naz: f64, b: &Bisect) -> Vec<Angles> {
        match *b {
            Bisect::Omega(omega) => {
                let t = self.theta() + omega;
                let phis = solve_component(&axis(PHI), &self.p0, &Vector3::y(), omega.sin());
                let mut out = Vec::new();
                for phi in phis.values() {
                    out.extend(self.positions(n, naz, t, phi));
                }
                out
            }
            Bisect::Mu(mu) => self.naz_bisect_scan(n, naz, |a| a.z * mu.cos() - a.y * mu.sin()),
            Bisect::Eta(eta) => self.naz_bisect_scan(n, naz, |a| a.x - eta.sin()),
        }
    }

    /// The χ turns s that put Q̂ into place about the χ axis at tilt t.
    fn chi(&self, t: f64, phi: f64) -> Roots {
        let q = rot(&Vector3::x(), t).transpose() * Vector3::new(0.0, -self.st, self.ct);
        spin(&Vector3::y(), &(circle(PHI, phi) * self.p0), &q)
    }

    /// Z = R_y(qaz)·Rₓ(t)·CHI(s)·PHI(φ): at qaz = 0 the χ axis is Rₓ(t)·ŷ and
    /// Q̂ is (0, −sin θ, cos θ), s from [`Self::chi`], and naz fixes qaz.
    fn lab(n: &Vector3<f64>, naz: f64, t: f64, phi: f64, s: f64) -> Option<Matrix3<f64>> {
        let z = rot(&Vector3::x(), t) * circle(CHI, s) * circle(PHI, phi);
        let n = z * n;
        (n.x.hypot(n.z) >= EPS).then(|| rot(&Vector3::y(), naz - n.x.atan2(n.z)) * z)
    }

    fn positions(&self, n: &Vector3<f64>, naz: f64, t: f64, phi: f64) -> Vec<Angles> {
        let mut out = Vec::new();
        for s in self.chi(t, phi).values() {
            if let Some(z) = Self::lab(n, naz, t, phi, s) {
                let samples = sample::one_held(&[None, None, None, Some(phi)], &z);
                out.extend(self.free_detector(samples));
            }
        }
        out
    }

    /// Roots of `miss` of the χ axis over φ. The height h of Q̂ puts the axis
    /// at t = θ + asin h or θ + π − asin h, one branch each, both defined
    /// for every φ, so neither ends in a fold where a root could hide.
    fn naz_bisect_scan(
        &self,
        n: &Vector3<f64>,
        naz: f64,
        miss: impl Fn(&Vector3<f64>) -> f64,
    ) -> Vec<Angles> {
        let residual =
            |phi: f64, branch: usize| self.chi_lab(n, naz, phi, branch).map(|a| miss(&a));
        let mut out = Vec::new();
        for (phi, branch) in scan(residual) {
            out.extend(self.positions(n, naz, self.tilt(phi, branch), phi));
        }
        out
    }

    /// t of the χ axis at φ on `branch` of [`Self::naz_bisect_scan`].
    fn tilt(&self, phi: f64, branch: usize) -> f64 {
        let lift = (circle(PHI, phi) * self.p0).y.clamp(-1.0, 1.0).asin();
        self.theta() + if branch == 0 { lift } else { PI - lift }
    }

    /// The χ axis in the lab at φ on `branch`; `None` where χ has no one
    /// value or n no azimuth.
    fn chi_lab(&self, n: &Vector3<f64>, naz: f64, phi: f64, branch: usize) -> Option<Vector3<f64>> {
        let t = self.tilt(phi, branch);
        let Roots::One(s) = self.chi(t, phi) else {
            return None;
        };
        let z = Self::lab(n, naz, t, phi, s)?;
        Some(z * circle(PHI, phi).transpose() * Vector3::y())
    }
}

/// Roots of `f(u, branch)` over one period for branches 0 and 1. A point
/// of a grid of [`SCAN`] where |f| is least among same-signed neighbours
/// brackets an extremum turning f back toward zero; refined, it joins the
/// grid as a knot. A root is a sign change between knots, bisected, a knot
/// where f is zero, or an extremum that stays on its side within [`ACCEPT`]
/// of zero: f touches zero there, a double root, for the validator to judge.
fn scan(f: impl Fn(f64, usize) -> Option<f64>) -> Vec<(f64, usize)> {
    let step = TAU / SCAN as f64;
    let at = |i: usize| -PI + step * i as f64;
    let mut out = Vec::new();
    for branch in 0..2 {
        let f = |u: f64| f(u, branch);
        let grid: Vec<Option<f64>> = (0..SCAN).map(|i| f(at(i))).collect();
        // (u, f(u), whether f touches zero at u), in order over a period.
        let mut knots: Vec<(f64, Option<f64>, bool)> = Vec::with_capacity(SCAN);
        for (i, &v) in grid.iter().enumerate() {
            let near = [grid[(i + SCAN - 1) % SCAN], grid[(i + 1) % SCAN]];
            let extremum = match (near, v) {
                ([Some(a), Some(b)], Some(v)) if v * a > 0.0 && v * b > 0.0 => {
                    let s = v.signum();
                    // Strict on one side: of two equal points only the first.
                    (s * v < s * a && s * v <= s * b)
                        .then(|| least(|u| f(u).map(|x| s * x), at(i) - step, at(i) + step))
                        .flatten()
                        .and_then(|e| Some((e, f(e)?)))
                        .map(|(e, m)| (e, m, s * m > 0.0 && m.abs() <= ACCEPT))
                }
                _ => None,
            };
            match extremum {
                Some((e, m, touch)) if e < at(i) => {
                    knots.extend([(e, Some(m), touch), (at(i), v, false)])
                }
                Some((e, m, touch)) => knots.extend([(at(i), v, false), (e, Some(m), touch)]),
                None => knots.push((at(i), v, false)),
            }
        }
        for (k, &(u, fu, touch)) in knots.iter().enumerate() {
            let Some(a) = fu else {
                continue;
            };
            if a == 0.0 || touch {
                out.push((u, branch));
            }
            let (v, fv, _) = knots[(k + 1) % knots.len()];
            let v = if k + 1 == knots.len() { v + TAU } else { v };
            if let Some(b) = fv
                && a * b < 0.0
            {
                out.push((bisect(f, u, v, a), branch));
            }
        }
    }
    out
}

/// The root of `f` between `lo` and `hi`, where f is `at_lo` and of the
/// other sign at `hi`.
fn bisect(f: impl Fn(f64) -> Option<f64>, mut lo: f64, mut hi: f64, at_lo: f64) -> f64 {
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        match f(mid) {
            Some(m) if (m > 0.0) == (at_lo > 0.0) => lo = mid,
            Some(_) => hi = mid,
            None => break,
        }
    }
    0.5 * (lo + hi)
}

/// Golden-section minimum of `g` in [`a`, `b`], `None` where g is not
/// defined at a point it tries.
fn least(g: impl Fn(f64) -> Option<f64>, mut a: f64, mut b: f64) -> Option<f64> {
    let r = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
    let (mut gc, mut gd) = (g(c)?, g(d)?);
    for _ in 0..60 {
        if gc < gd {
            (b, d, gd) = (d, c, gc);
            c = b - r * (b - a);
            gc = g(c)?;
        } else {
            (a, c, gc) = (c, d, gd);
            d = a + r * (b - a);
            gd = g(d)?;
        }
    }
    Some(if gc < gd { c } else { d })
}

#[cfg(test)]
mod tests {
    use super::super::super::super::system::Goal;
    use super::super::super::super::{
        InverseError, InverseRequest, Mode, Target, solve as numeric, wrap,
    };
    use super::super::super::map::Map;
    use super::super::super::search;
    use super::super::super::you::ETA;
    use super::*;
    use crate::presets::preset;
    use crate::{Equation, Wavelength, scattering};
    use nalgebra::{Point3, Rotation3};

    /// f = 1 − cos(u − x0) − c on branch 0, least at x0, which sits mid-cell
    /// or in the cell that closes the period.
    #[test]
    fn scan_finds_roots_inside_one_cell() {
        let step = TAU / SCAN as f64;
        for x0 in [-PI + 100.5 * step, PI - 0.5 * step] {
            let roots = |c: f64| {
                let mut r: Vec<f64> = scan(|u, b| (b == 0).then(|| 1.0 - (u - x0).cos() - c))
                    .into_iter()
                    .map(|(u, b)| {
                        assert_eq!(b, 0);
                        wrap(u - x0)
                    })
                    .collect();
                r.sort_by(f64::total_cmp);
                r
            };
            let pair = roots(1.0 - (0.3 * step).cos());
            assert_eq!(pair.len(), 2, "pair at {x0}: {pair:?}");
            for (r, want) in pair.iter().zip([-0.3 * step, 0.3 * step]) {
                assert!((r - want).abs() < 1e-12, "pair at {x0}: {pair:?}");
            }
            for c in [0.0, -0.5 * ACCEPT] {
                let touch = roots(c);
                assert_eq!(touch.len(), 1, "touch {c} at {x0}: {touch:?}");
                assert!(touch[0].abs() < 1e-5, "touch {c} at {x0}: {touch:?}");
            }
            assert!(roots(-2.0 * ACCEPT).is_empty(), "miss at {x0}");
        }
    }

    /// E6C naz+bisect with omega (You's η) held at sin η = g − d, where g is
    /// the maximum of the χ axis's x over φ: d = 0 is a double root,
    /// singular, and a small d two roots a quarter cell either side of it.
    #[test]
    fn naz_bisect_finds_roots_inside_one_cell() {
        let geo = preset("E6C").unwrap();
        let (normal, naz) = (Vector3::new(0.2, 0.3, 1.0), 0.7);
        let extra = vec![
            Equation::Naz { normal, naz },
            Equation::InScatteringPlane { axis: 2 },
        ];
        let mode = Mode::new(&geo, "naz bisect", &[0, 2, 3, 4, 5], extra).unwrap();
        let ub = Rotation3::from_euler_angles(0.3, -0.2, 0.5).into_inner() * 1.5;
        let hkl = Vector3::new(0.3, -0.4, 1.1);
        let wavelength = Wavelength::new(1.54).unwrap();
        let request = |current: &[f64; 6]| {
            let request = InverseRequest {
                mode: &mode,
                ub,
                wavelength,
                target: Target::Hkl(hkl),
                current,
            };
            let system = request.system().unwrap();
            let theirs = numeric::search(&system).into_sorted().map(|f| f.solutions);
            (
                search(&system).unwrap().into_sorted().map(|f| f.solutions),
                theirs,
            )
        };
        let map = Map::new(&geo, None).unwrap();
        let zero = [0.0; 6];
        let at_zero = InverseRequest {
            mode: &mode,
            ub,
            wavelength,
            target: Target::Hkl(hkl),
            current: &zero,
        };
        let Goal::Q(q) = *at_zero.system().unwrap().goal() else {
            panic!("hkl goal");
        };
        let problem = Problem::new(&(map.phi * q));
        let (n, naz) = ((map.phi * normal).normalize(), naz + map.azimuth.unwrap());
        let g = |phi: f64| problem.chi_lab(&n, naz, phi, 0).unwrap().x;
        let slope = |phi: f64| g(phi + 1e-6) - g(phi - 1e-6);
        // The maximum of g on branch 0, where its slope turns in (−0.9, −0.8).
        let (mut lo, mut hi) = (-0.9, -0.8);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if slope(mid) > 0.0 { lo = mid } else { hi = mid }
        }
        let top = 0.5 * (lo + hi);
        let step = TAU / SCAN as f64;
        let bend = (g(top + 1e-4) - 2.0 * g(top) + g(top - 1e-4)) / 1e-8;
        let held = |d: f64| {
            let mut current = [0.0; 6];
            current[1] = map.sample[ETA].value((g(top) - d).asin());
            current
        };

        let (ours, theirs) = request(&held(0.0));
        assert_eq!(ours, Err(InverseError::Degenerate));
        assert_eq!(theirs, Err(InverseError::Degenerate));

        let current = held(0.5 * bend.abs() * (0.25 * step).powi(2));
        let (ours, theirs) = request(&current);
        let (ours, theirs) = (ours.unwrap(), theirs.unwrap());
        let near = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(u, v)| wrap(u - v).abs() < 1e-6);
        assert_eq!(ours.len(), theirs.len(), "{ours:?}\n{theirs:?}");
        assert!(
            ours.iter().all(|x| theirs.iter().any(|y| near(x, y))),
            "{ours:?}\n{theirs:?}"
        );
        let point = Point3::from(geo.beam_direction().into_inner());
        let mut inside = 0;
        for values in &ours {
            let back = ub.lu().solve(
                &scattering(&geo, values, wavelength, &point)
                    .unwrap()
                    .q_sample,
            );
            assert!((back.unwrap() - hkl).amax() < 1e-9, "{values:?}");
            assert_eq!(values[1], current[1]);
            let phi = map.sample[PHI].angle(values[3]);
            inside += usize::from(wrap(phi - top).abs() < step);
        }
        assert_eq!(inside, 4, "{ours:?}");
    }
}
