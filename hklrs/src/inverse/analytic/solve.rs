//! The target in You's φ frame and the constraint families built on it.

use nalgebra::{Matrix3, Vector3};

use super::sample::{self, Known};
use super::you::{
    self, Angles, EPS, ETA, Free, MU, Roots, arms, axis, circle, cos_eq, normal, rot,
    solve_component,
};
use super::{Bisect, Constraints, Detector, Error, Reference, TOL};

mod naz;

/// The reflection to reach: Q̂ and θ, with the reference vector n and the
/// surface normal s, all unit vectors in the φ frame.
pub(super) struct Problem {
    p0: Vector3<f64>,
    st: f64,
    ct: f64,
    pub(super) n: Vector3<f64>,
    pub(super) s: Vector3<f64>,
}

fn unit(v: &Vector3<f64>, what: &'static str) -> Result<Vector3<f64>, Error> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite { what });
    }
    v.try_normalize(0.0).ok_or(Error::ZeroReference { what })
}

impl Problem {
    /// `q` in Å⁻¹ and `k` = 2π/λ; `reference` and `surface` any length.
    pub(super) fn new(
        q: &Vector3<f64>,
        k: f64,
        reference: &Vector3<f64>,
        surface: &Vector3<f64>,
    ) -> Result<Self, Error> {
        let (n, s) = (unit(reference, "reference")?, unit(surface, "surface")?);
        if !q.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite { what: "Q" });
        }
        let (q_norm, q_max) = (q.norm(), 2.0 * k);
        // Q past 2k by less than the round-trip tolerance is backscatter.
        if q_norm - q_max > TOL * k {
            return Err(Error::Unreachable { q: q_norm, q_max });
        }
        if q_norm == 0.0 {
            return Err(Error::Degenerate {
                what: "every sample circle (Q = 0)",
            });
        }
        let st = (q_norm / q_max).min(1.0);
        Ok(Self {
            p0: q / q_norm,
            st,
            ct: (1.0 - st * st).sqrt(),
            n,
            s,
        })
    }

    /// Q in the φ frame, units of k.
    pub(super) fn q(&self) -> Vector3<f64> {
        2.0 * self.st * self.p0
    }

    pub(super) fn theta(&self) -> f64 {
        self.st.atan2(self.ct)
    }

    /// Whether ψ of n has a value: n is not along Q̂ and Q̂ not along the beam.
    pub(super) fn psi_defined(&self) -> bool {
        self.p0.cross(&self.n).norm() * self.ct >= EPS
    }

    pub(super) fn candidates(&self, constraints: &Constraints) -> Vec<(Angles, Free)> {
        match constraints {
            Constraints::DetRefSample(d, r, h) => self.det_ref_sample(d, r, &h.known()),
            Constraints::DetSamples(Detector::Naz(naz), h) => self.naz_samples(*naz, &h.known()),
            Constraints::DetBisect(Detector::Naz(naz), b) => self.naz_bisect(*naz, b),
            Constraints::DetSamples(d, h) => self.det_samples(d, &h.known()),
            Constraints::DetBisect(d, b) => self.det_bisect(d, b),
            Constraints::RefSamples(r, h) => self.ref_samples(r, &h.known()),
            Constraints::Samples(h) => self.samples(&h.known()),
        }
    }

    /// Q̂ in the lab with the detector arm at azimuth `qaz` about the beam.
    fn at_qaz(&self, qaz: f64) -> Vector3<f64> {
        Vector3::new(self.ct * qaz.sin(), -self.st, self.ct * qaz.cos())
    }

    /// Every Q̂ in the lab the detector constraint allows. `Naz` leaves Q̂ free
    /// about the beam, so every caller routes it apart.
    fn detector(&self, d: &Detector) -> Vec<(Vector3<f64>, Free)> {
        let c2 = 1.0 - 2.0 * self.st * self.st;
        let q = |nu: f64, delta: f64| {
            (you::kout(&[0.0, 0.0, 0.0, 0.0, nu, delta]) - Vector3::y()) / (2.0 * self.st)
        };
        match *d {
            Detector::Delta(delta) => cos_eq(delta.cos(), c2)
                .tagged("nu")
                .into_iter()
                .map(|(nu, f)| (q(nu, delta), f))
                .collect(),
            Detector::Nu(nu) => cos_eq(nu.cos(), c2)
                .tagged("delta")
                .into_iter()
                .map(|(delta, f)| (q(nu, delta), f))
                .collect(),
            Detector::Qaz(qaz) => vec![(self.at_qaz(qaz), None)],
            Detector::Naz(_) => unreachable!("naz fixes no lab Q̂"),
        }
    }

    /// Lab frame [Q̂, e₁, e₂] for Q̂ along `q`, e₁ the beam's part normal to
    /// Q̂ (any normal when Q̂ is along the beam).
    fn q_frame(q: &Vector3<f64>) -> Matrix3<f64> {
        let y = Vector3::y();
        let e1 = (y - q * q.dot(&y))
            .try_normalize(EPS)
            .unwrap_or_else(|| normal(q));
        Matrix3::from_columns(&[*q, e1, q.cross(&e1)])
    }

    /// φ frame [Q̂, u₂, u₃] for vector `r`, u₂ along r's part normal to Q̂
    /// (any normal when r is along Q̂), with cos τ and sin τ of r from Q̂.
    fn ref_frame(&self, r: &Vector3<f64>) -> (Matrix3<f64>, f64, f64) {
        let c = r.dot(&self.p0);
        let perp = r - self.p0 * c;
        let u2 = perp.try_normalize(EPS).unwrap_or_else(|| normal(&self.p0));
        let u = Matrix3::from_columns(&[self.p0, u2, self.p0.cross(&u2)]);
        (u, c, perp.norm())
    }

    /// ψ, the turn about Q̂ from the [`Self::q_frame`] to the
    /// [`Self::ref_frame`] that meets `r`. With Z = L·R_x(ψ)·Uᵀ the lab
    /// height of the vector is (Z·v)_y = −sin θ cos τ + sin τ cos θ cos ψ,
    /// and each reference constraint fixes that height.
    fn psi(&self, r: &Reference, cos_tau: f64, sin_tau: f64) -> Roots {
        let height = match *r {
            Reference::Alpha(v) | Reference::Betain(v) => -v.sin(),
            Reference::Beta(v) | Reference::Betaout(v) => v.sin() - 2.0 * self.st * cos_tau,
            Reference::AEqB | Reference::BinEqBout => -self.st * cos_tau,
            Reference::Psi(v) => return Roots::One(v),
        };
        cos_eq(sin_tau * self.ct, height + self.st * cos_tau)
    }

    /// Every lab rotation Z and lab Q̂ that meet the detector and reference
    /// constraints; `Naz` turns the qaz = 0 solutions about the beam, which
    /// keeps every reference angle and moves naz and qaz together.
    fn orientations(&self, d: &Detector, r: &Reference) -> Vec<(Matrix3<f64>, Vector3<f64>, Free)> {
        let v = if r.on_surface() { self.s } else { self.n };
        let (u, cos_tau, sin_tau) = self.ref_frame(&v);
        let psis = self.psi(r, cos_tau, sin_tau).tagged("psi");
        let orient = |q: &Vector3<f64>, free: Free| {
            let l = Self::q_frame(q);
            psis.iter()
                .map(move |&(psi, f)| (l * rot(&Vector3::x(), psi) * u.transpose(), *q, free.or(f)))
                .collect::<Vec<_>>()
        };
        match *d {
            Detector::Naz(naz) => orient(&self.at_qaz(0.0), None)
                .into_iter()
                .filter_map(|(z, q, f)| {
                    let n = z * self.n;
                    (n.x.hypot(n.z) >= EPS).then(|| {
                        let turn = rot(&Vector3::y(), naz - n.x.atan2(n.z));
                        (turn * z, turn * q, f)
                    })
                })
                .collect(),
            _ => self
                .detector(d)
                .iter()
                .flat_map(|(q, f)| orient(q, *f))
                .collect(),
        }
    }

    fn det_ref_sample(&self, d: &Detector, r: &Reference, known: &Known) -> Vec<(Angles, Free)> {
        self.orientations(d, r)
            .into_iter()
            .flat_map(|(z, q, f)| self.finish(Some(d), sample::one_held(known, &z), &q, f))
            .collect()
    }

    fn det_samples(&self, d: &Detector, known: &Known) -> Vec<(Angles, Free)> {
        self.detector(d)
            .iter()
            .flat_map(|(q, f)| self.finish(Some(d), sample::two_held(known, &self.p0, q), q, *f))
            .collect()
    }

    fn det_bisect(&self, d: &Detector, b: &Bisect) -> Vec<(Angles, Free)> {
        let mut out = Vec::new();
        for (q, fq) in self.detector(d) {
            let Some(eq) = Vector3::new(q.x, 0.0, q.z).try_normalize(EPS) else {
                continue;
            };
            for (mu, eta, f) in self.chi_axis(b, &eq) {
                let known = [Some(mu), Some(eta), None, None];
                let samples = sample::two_held(&known, &self.p0, &q);
                out.extend(self.finish(Some(d), samples, &q, fq.or(f)));
            }
        }
        out
    }

    /// With Z = L·R_x(ψ)·Uᵀ the beam in the φ frame is
    /// Zᵀŷ = U·R_x(ψ)ᵀ·(−sin θ, cos θ, 0) whatever the detector, so each ψ
    /// fixes it and the two free circles turn it onto ŷ.
    fn ref_samples(&self, r: &Reference, known: &Known) -> Vec<(Angles, Free)> {
        let v = if r.on_surface() { self.s } else { self.n };
        let (u, cos_tau, sin_tau) = self.ref_frame(&v);
        let beam = Vector3::new(-self.st, self.ct, 0.0);
        let mut out = Vec::new();
        for (psi, f) in self.psi(r, cos_tau, sin_tau).tagged("psi") {
            let w = u * rot(&Vector3::x(), psi).transpose() * beam;
            out.extend(self.free_detector(sample::two_held(known, &w, &Vector3::y()), f));
        }
        out
    }

    /// The one free circle puts Q̂ at the lab height −sin θ.
    fn samples(&self, known: &Known) -> Vec<(Angles, Free)> {
        let found = sample::three_held(known, &self.p0, &Vector3::y(), -self.st);
        self.free_detector(found, None)
    }

    /// [`Self::finish`] for sample positions that fix lab Q̂ themselves.
    fn free_detector(&self, samples: Vec<([f64; 4], Free)>, free: Free) -> Vec<(Angles, Free)> {
        samples
            .into_iter()
            .flat_map(|(s, fs)| {
                let q = you::z(&[s[0], s[1], s[2], s[3], 0.0, 0.0]) * self.p0;
                self.finish(None, vec![(s, fs)], &q, free)
            })
            .collect()
    }

    /// (μ, η) that put the χ axis MU·ETA·ŷ in the plane of the beam and the
    /// unit azimuth `eq`, where `b` asks. Both signs of cos η are kept: the
    /// branch diffcalc adds by π − asin η leaves the plane (DC-3).
    fn chi_axis(&self, b: &Bisect, eq: &Vector3<f64>) -> Vec<(f64, f64, Free)> {
        let normal = Vector3::new(eq.z, 0.0, -eq.x);
        match *b {
            Bisect::Omega(omega) => {
                let t = self.theta() + omega;
                arms(&(t.cos() * Vector3::y() + t.sin() * eq), "mu")
            }
            Bisect::Mu(mu) => {
                let w = circle(MU, mu).transpose() * normal;
                solve_component(&axis(ETA), &Vector3::y(), &w, 0.0)
                    .tagged("eta")
                    .into_iter()
                    .map(|(eta, f)| (mu, eta, f))
                    .collect()
            }
            Bisect::Eta(eta) => {
                let chi = circle(ETA, eta) * Vector3::y();
                solve_component(&axis(MU), &chi, &normal, 0.0)
                    .tagged("mu")
                    .into_iter()
                    .map(|(mu, f)| (mu, eta, f))
                    .collect()
            }
        }
    }

    /// Full positions from sample circles that turn Q̂ onto lab `q`, with the
    /// detector arm that diffracts there: a held arm circle stays at its
    /// value, otherwise both arm branches.
    fn finish(
        &self,
        d: Option<&Detector>,
        samples: Vec<([f64; 4], Free)>,
        q: &Vector3<f64>,
        free: Free,
    ) -> Vec<(Angles, Free)> {
        let kout = Vector3::y() + 2.0 * self.st * q;
        let arm: Vec<(f64, f64, Free)> = match d.copied() {
            Some(Detector::Delta(delta)) if delta.cos().abs() >= EPS => {
                let c = delta.cos().signum();
                vec![((c * kout.z).atan2(c * kout.y), delta, None)]
            }
            Some(Detector::Delta(delta)) => Roots::Free
                .tagged("nu")
                .into_iter()
                .map(|(nu, f)| (nu, delta, f))
                .collect(),
            Some(Detector::Nu(nu)) => {
                vec![(
                    nu,
                    kout.x.atan2(kout.y * nu.cos() + kout.z * nu.sin()),
                    None,
                )]
            }
            None | Some(Detector::Qaz(_) | Detector::Naz(_)) => arms(&kout, "nu"),
        };
        let mut out = Vec::new();
        for (s, fs) in samples {
            for &(nu, delta, fa) in &arm {
                out.push(([s[0], s[1], s[2], s[3], nu, delta], free.or(fs).or(fa)));
            }
        }
        out
    }
}
