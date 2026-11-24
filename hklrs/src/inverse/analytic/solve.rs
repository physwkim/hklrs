//! The target in You's φ frame and the constraint families built on it.

use nalgebra::{Matrix3, Vector3};

use super::sample::{self, Known};
use super::you::{
    self, Angles, EPS, ETA, MU, Roots, arms, axis, circle, cos_eq, normal, rot, solve_component,
};
use super::{Bisect, Constraints, Detector, Reference};

mod naz;

/// The reflection to reach: Q̂ in the φ frame and θ.
pub(super) struct Problem {
    p0: Vector3<f64>,
    st: f64,
    ct: f64,
}

impl Problem {
    /// `q` in units of k, nonzero and no longer than 2.
    pub(super) fn new(q: &Vector3<f64>) -> Self {
        let q_norm = q.norm();
        // The reach gate pulls |q| onto 2 at most; this absorbs its rounding.
        let st = (q_norm / 2.0).min(1.0);
        Self {
            p0: q / q_norm,
            st,
            ct: (1.0 - st * st).sqrt(),
        }
    }

    pub(super) fn theta(&self) -> f64 {
        self.st.atan2(self.ct)
    }

    pub(super) fn candidates(&self, constraints: &Constraints) -> Vec<Angles> {
        match constraints {
            Constraints::DetRefSample(d, r, known) => self.det_ref_sample(d, r, known),
            Constraints::DetSamples(Detector::Naz(n, naz), known) => {
                self.naz_samples(n, *naz, known)
            }
            Constraints::DetBisect(Detector::Naz(n, naz), b) => self.naz_bisect(n, *naz, b),
            Constraints::DetSamples(d, known) => self.det_samples(d, known),
            Constraints::DetBisect(d, b) => self.det_bisect(d, b),
            Constraints::RefSamples(r, known) => self.ref_samples(r, known),
            Constraints::Samples(known) => self.samples(known),
        }
    }

    /// Q̂ in the lab with the detector arm at azimuth `qaz` about the beam.
    fn at_qaz(&self, qaz: f64) -> Vector3<f64> {
        Vector3::new(self.ct * qaz.sin(), -self.st, self.ct * qaz.cos())
    }

    /// Every Q̂ in the lab the detector constraint allows. `Naz` leaves Q̂ free
    /// about the beam, so every caller routes it apart.
    fn detector(&self, d: &Detector) -> Vec<Vector3<f64>> {
        // k̂f = (sin δ, cos δ cos ν, cos δ sin ν) has k̂f·ŷ = cos 2θ, so
        // Q̂ = (k̂f − ŷ)/(2 sin θ) has y = −sin θ, taken as is: k̂f − ŷ
        // cancels there.
        let q = |nu: f64, delta: f64| {
            let s = 2.0 * self.st;
            Vector3::new(delta.sin() / s, -self.st, delta.cos() * nu.sin() / s)
        };
        match *d {
            Detector::Delta(delta) => self
                .other_arm(delta)
                .values()
                .into_iter()
                .map(|nu| q(nu, delta))
                .collect(),
            Detector::Nu(nu) => self
                .other_arm(nu)
                .values()
                .into_iter()
                .map(|delta| q(nu, delta))
                .collect(),
            Detector::Qaz(qaz) => vec![self.at_qaz(qaz)],
            Detector::Naz(..) => unreachable!("naz fixes no lab Q̂"),
        }
    }

    /// x with cos x · cos `a` = cos 2θ: the other detector circle where one
    /// is at `a`. acos of cos 2θ / cos a would lose a root near 0 or π to
    /// rounding, so x comes from sin²(x/2) and cos²(x/2), each a product
    /// without cancellation over cos a.
    fn other_arm(&self, a: f64) -> Roots {
        let cos_a = a.cos();
        if cos_a.abs() < EPS {
            return cos_eq(cos_a, 1.0 - 2.0 * self.st * self.st);
        }
        let (h, c, st) = ((a / 2.0).sin(), (a / 2.0).cos(), self.st);
        let [sin2, cos2] = [(st - h) * (st + h), (c - st) * (c + st)].map(|p| p / cos_a);
        // cos_eq's tolerance on cos x, 1e-9, on its half angles.
        if sin2.min(cos2) < -0.5e-9 {
            return Roots::Empty;
        }
        let x = 2.0 * sin2.max(0.0).sqrt().atan2(cos2.max(0.0).sqrt());
        Roots::Two([x, -x])
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
            Reference::Incidence(_, v) => -v.sin(),
            Reference::Emergence(_, v) => v.sin() - 2.0 * self.st * cos_tau,
            Reference::Equal(_) => -self.st * cos_tau,
            Reference::Psi(_, v) => return Roots::One(v),
        };
        cos_eq(sin_tau * self.ct, height + self.st * cos_tau)
    }

    /// Every lab rotation Z and lab Q̂ that meet the detector and reference
    /// constraints; `Naz` turns the qaz = 0 solutions about the beam, which
    /// keeps every reference angle and moves naz and qaz together.
    fn orientations(&self, d: &Detector, r: &Reference) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
        let (u, cos_tau, sin_tau) = self.ref_frame(&r.vector());
        let psis = self.psi(r, cos_tau, sin_tau).values();
        let orient = |q: &Vector3<f64>| {
            let l = Self::q_frame(q);
            psis.iter()
                .map(move |&psi| (l * rot(&Vector3::x(), psi) * u.transpose(), *q))
                .collect::<Vec<_>>()
        };
        match *d {
            Detector::Naz(n, naz) => orient(&self.at_qaz(0.0))
                .into_iter()
                .filter_map(|(z, q)| {
                    let n = z * n;
                    (n.x.hypot(n.z) >= EPS).then(|| {
                        let turn = rot(&Vector3::y(), naz - n.x.atan2(n.z));
                        (turn * z, turn * q)
                    })
                })
                .collect(),
            _ => self.detector(d).iter().flat_map(orient).collect(),
        }
    }

    fn det_ref_sample(&self, d: &Detector, r: &Reference, known: &Known) -> Vec<Angles> {
        self.orientations(d, r)
            .into_iter()
            .flat_map(|(z, q)| self.finish(Some(d), sample::one_held(known, &z), &q))
            .collect()
    }

    fn det_samples(&self, d: &Detector, known: &Known) -> Vec<Angles> {
        self.detector(d)
            .iter()
            .flat_map(|q| self.finish(Some(d), sample::two_held(known, &self.p0, q), q))
            .collect()
    }

    fn det_bisect(&self, d: &Detector, b: &Bisect) -> Vec<Angles> {
        let mut out = Vec::new();
        for q in self.detector(d) {
            let Some(eq) = Vector3::new(q.x, 0.0, q.z).try_normalize(EPS) else {
                continue;
            };
            for (mu, eta) in self.chi_axis(b, &eq) {
                let known = [Some(mu), Some(eta), None, None];
                let samples = sample::two_held(&known, &self.p0, &q);
                out.extend(self.finish(Some(d), samples, &q));
            }
        }
        out
    }

    /// With Z = L·R_x(ψ)·Uᵀ the beam in the φ frame is
    /// Zᵀŷ = U·R_x(ψ)ᵀ·(−sin θ, cos θ, 0) whatever the detector, so each ψ
    /// fixes it and the two free circles turn it onto ŷ.
    fn ref_samples(&self, r: &Reference, known: &Known) -> Vec<Angles> {
        let (u, cos_tau, sin_tau) = self.ref_frame(&r.vector());
        let beam = Vector3::new(-self.st, self.ct, 0.0);
        let mut out = Vec::new();
        for psi in self.psi(r, cos_tau, sin_tau).values() {
            let w = u * rot(&Vector3::x(), psi).transpose() * beam;
            out.extend(self.free_detector(sample::two_held(known, &w, &Vector3::y())));
        }
        out
    }

    /// The one free circle puts Q̂ at the lab height −sin θ.
    fn samples(&self, known: &Known) -> Vec<Angles> {
        self.free_detector(sample::three_held(known, &self.p0, &Vector3::y(), -self.st))
    }

    /// [`Self::finish`] for sample positions that fix lab Q̂ themselves.
    fn free_detector(&self, samples: Vec<[f64; 4]>) -> Vec<Angles> {
        samples
            .into_iter()
            .flat_map(|s| {
                let q = you::z(&[s[0], s[1], s[2], s[3], 0.0, 0.0]) * self.p0;
                self.finish(None, vec![s], &q)
            })
            .collect()
    }

    /// (μ, η) that put the χ axis MU·ETA·ŷ in the plane of the beam and the
    /// unit azimuth `eq`, where `b` asks. Both signs of cos η are kept: the
    /// branch diffcalc adds by π − asin η leaves the plane (DC-3).
    fn chi_axis(&self, b: &Bisect, eq: &Vector3<f64>) -> Vec<(f64, f64)> {
        let normal = Vector3::new(eq.z, 0.0, -eq.x);
        match *b {
            Bisect::Omega(omega) => {
                let t = self.theta() + omega;
                arms(&(t.cos() * Vector3::y() + t.sin() * eq))
            }
            Bisect::Mu(mu) => {
                let w = circle(MU, mu).transpose() * normal;
                solve_component(&axis(ETA), &Vector3::y(), &w, 0.0)
                    .values()
                    .into_iter()
                    .map(|eta| (mu, eta))
                    .collect()
            }
            Bisect::Eta(eta) => {
                let chi = circle(ETA, eta) * Vector3::y();
                solve_component(&axis(MU), &chi, &normal, 0.0)
                    .values()
                    .into_iter()
                    .map(|mu| (mu, eta))
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
        samples: Vec<[f64; 4]>,
        q: &Vector3<f64>,
    ) -> Vec<Angles> {
        let kout = Vector3::y() + 2.0 * self.st * q;
        let arm: Vec<(f64, f64)> = match d.copied() {
            Some(Detector::Delta(delta)) if delta.cos().abs() >= EPS => {
                let c = delta.cos().signum();
                vec![((c * kout.z).atan2(c * kout.y), delta)]
            }
            Some(Detector::Delta(delta)) => Roots::Free
                .values()
                .into_iter()
                .map(|nu| (nu, delta))
                .collect(),
            Some(Detector::Nu(nu)) => {
                vec![(nu, kout.x.atan2(kout.y * nu.cos() + kout.z * nu.sin()))]
            }
            None | Some(Detector::Qaz(_) | Detector::Naz(..)) => arms(&kout),
        };
        let mut out = Vec::new();
        for s in samples {
            for &(nu, delta) in &arm {
                out.push([s[0], s[1], s[2], s[3], nu, delta]);
            }
        }
        out
    }
}
