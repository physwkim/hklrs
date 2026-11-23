//! `naz` with two sample circles or with `bisect`, which diffcalc-core
//! documents and rejects (DC-1).

use std::f64::consts::{PI, TAU};

use nalgebra::{Matrix3, Vector3};

use super::super::Bisect;
use super::super::sample::{self, Known};
use super::super::you::{Angles, CHI, EPS, Free, PHI, axis, circle, rot, solve_component, spin};
use super::Problem;

/// Grid points per period where φ is scanned.
const SCAN: usize = 1440;

impl Problem {
    pub(super) fn naz_samples(&self, naz: f64, known: &Known) -> Vec<(Angles, Free)> {
        let level = Vector3::new(naz.cos(), 0.0, -naz.sin());
        let eqs = [(self.p0, Vector3::y(), -self.st), (self.n, level, 0.0)];
        self.free_detector(sample::two_held_heights(known, &eqs), None)
    }

    /// With bisect the χ axis lies at an angle t from the beam in the plane
    /// of the beam and Q̂, so Q̂ sits at height sin(t − θ) along it. ω fixes
    /// t = θ + ω and that height fixes φ; a held μ or η leaves φ to a scan.
    pub(super) fn naz_bisect(&self, naz: f64, b: &Bisect) -> Vec<(Angles, Free)> {
        match *b {
            Bisect::Omega(omega) => {
                let t = self.theta() + omega;
                let phis = solve_component(&axis(PHI), &self.p0, &Vector3::y(), omega.sin());
                let mut out = Vec::new();
                for (phi, f) in phis.tagged("phi") {
                    out.extend(self.positions(naz, t, phi, f));
                }
                out
            }
            Bisect::Mu(mu) => self.naz_bisect_scan(naz, |a| a.z * mu.cos() - a.y * mu.sin()),
            Bisect::Eta(eta) => self.naz_bisect_scan(naz, |a| a.x - eta.sin()),
        }
    }

    /// Z = R_y(qaz)·Rₓ(t)·CHI(s)·PHI(φ): at qaz = 0 the χ axis is Rₓ(t)·ŷ and
    /// Q̂ is (0, −sin θ, cos θ), s turns Q̂ into place about the axis, and
    /// naz fixes qaz.
    fn lab(&self, naz: f64, t: f64, phi: f64) -> Vec<(Matrix3<f64>, Free)> {
        let tilt = rot(&Vector3::x(), t);
        let q = tilt.transpose() * Vector3::new(0.0, -self.st, self.ct);
        spin(&Vector3::y(), &(circle(PHI, phi) * self.p0), &q)
            .tagged("chi")
            .into_iter()
            .filter_map(|(s, f)| {
                let z = tilt * circle(CHI, s) * circle(PHI, phi);
                let n = z * self.n;
                (n.x.hypot(n.z) >= EPS).then(|| (rot(&Vector3::y(), naz - n.x.atan2(n.z)) * z, f))
            })
            .collect()
    }

    fn positions(&self, naz: f64, t: f64, phi: f64, free: Free) -> Vec<(Angles, Free)> {
        let mut out = Vec::new();
        for (z, f) in self.lab(naz, t, phi) {
            let samples = sample::one_held(&[None, None, None, Some(phi)], &z);
            out.extend(self.free_detector(samples, free.or(f)));
        }
        out
    }

    /// Roots of `miss` of the χ axis over φ. The height h of Q̂ puts the axis
    /// at t = θ + asin h or θ + π − asin h, one branch each, both defined
    /// for every φ, so neither ends in a fold where a root could hide.
    fn naz_bisect_scan(
        &self,
        naz: f64,
        miss: impl Fn(&Vector3<f64>) -> f64,
    ) -> Vec<(Angles, Free)> {
        let tilt = |phi: f64, branch: usize| {
            let lift = (circle(PHI, phi) * self.p0).y.clamp(-1.0, 1.0).asin();
            self.theta() + if branch == 0 { lift } else { PI - lift }
        };
        let residual = |phi: f64, branch: usize| {
            let lab = self.lab(naz, tilt(phi, branch), phi);
            let (z, _) = lab.into_iter().find(|(_, f)| f.is_none())?;
            Some(miss(&(z * circle(PHI, phi).transpose() * Vector3::y())))
        };
        let mut out = Vec::new();
        for (phi, branch) in scan(residual) {
            out.extend(self.positions(naz, tilt(phi, branch), phi, None));
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
