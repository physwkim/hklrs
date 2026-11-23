//! `naz` with two sample circles or with `bisect`, which diffcalc-core
//! documents and rejects (DC-1).

use std::f64::consts::{PI, TAU};

use nalgebra::{Matrix3, Vector3};

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
        let tilt = |phi: f64, branch: usize| {
            let lift = (circle(PHI, phi) * self.p0).y.clamp(-1.0, 1.0).asin();
            self.theta() + if branch == 0 { lift } else { PI - lift }
        };
        // Where χ is free the residual has no one value; the scan steps over.
        let residual = |phi: f64, branch: usize| {
            let t = tilt(phi, branch);
            let Roots::One(s) = self.chi(t, phi) else {
                return None;
            };
            let z = Self::lab(n, naz, t, phi, s)?;
            Some(miss(&(z * circle(PHI, phi).transpose() * Vector3::y())))
        };
        let mut out = Vec::new();
        for (phi, branch) in scan(residual) {
            out.extend(self.positions(n, naz, tilt(phi, branch), phi));
        }
        out
    }
}

/// Roots of `f(u, branch)` over one period for branches 0 and 1: sign
/// changes on a grid of [`SCAN`] points, bisected. A root where f touches
/// zero without changing sign is not found.
fn scan(f: impl Fn(f64, usize) -> Option<f64>) -> Vec<(f64, usize)> {
    let at = |i: usize| -PI + TAU * i as f64 / SCAN as f64;
    let mut out = Vec::new();
    for branch in 0..2 {
        let values: Vec<Option<f64>> = (0..=SCAN).map(|i| f(at(i), branch)).collect();
        for (i, pair) in values.windows(2).enumerate() {
            let [Some(u), Some(v)] = [pair[0], pair[1]] else {
                continue;
            };
            if u == 0.0 {
                out.push((at(i), branch));
                continue;
            }
            if (u > 0.0) == (v > 0.0) {
                continue;
            }
            let (mut lo, mut hi) = (at(i), at(i + 1));
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                match f(mid, branch) {
                    Some(m) if (m > 0.0) == (u > 0.0) => lo = mid,
                    Some(_) => hi = mid,
                    None => break,
                }
            }
            out.push((0.5 * (lo + hi), branch));
        }
    }
    out
}
