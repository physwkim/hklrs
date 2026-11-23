//! The target in You's φ frame and the constraint families built on it.

use nalgebra::{Matrix3, Vector3};

use super::sample::{self, Known};
use super::you::{self, Angles, EPS, Free, Roots, arms, cos_eq, normal, rot};
use super::{Constraints, Detector, Error, Reference};

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
        if q_norm > q_max {
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

    /// Whether ψ of n has a value: n is not along Q̂ and Q̂ not along the beam.
    pub(super) fn psi_defined(&self) -> bool {
        self.p0.cross(&self.n).norm() * self.ct >= EPS
    }

    pub(super) fn candidates(&self, constraints: &Constraints) -> Vec<(Angles, Free)> {
        match constraints {
            Constraints::DetRefSample(d, r, h) => self.det_ref_sample(d, r, &h.known()),
        }
    }

    /// Q̂ in the lab with the detector arm at azimuth `qaz` about the beam.
    fn at_qaz(&self, qaz: f64) -> Vector3<f64> {
        Vector3::new(self.ct * qaz.sin(), -self.st, self.ct * qaz.cos())
    }

    /// Every Q̂ in the lab the detector constraint allows; `Naz` does not fix
    /// the detector and has none.
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
            Detector::Naz(_) => Vec::new(),
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
            .flat_map(|(z, q, f)| self.finish(d, sample::one_held(known, &z), &q, f))
            .collect()
    }

    /// Full positions from sample circles that turn Q̂ onto lab `q`, with the
    /// detector arm that diffracts there: a held arm circle stays at its
    /// value, otherwise both arm branches.
    fn finish(
        &self,
        d: &Detector,
        samples: Vec<([f64; 4], Free)>,
        q: &Vector3<f64>,
        free: Free,
    ) -> Vec<(Angles, Free)> {
        let kout = Vector3::y() + 2.0 * self.st * q;
        let arm: Vec<(f64, f64, Free)> = match *d {
            Detector::Delta(delta) if delta.cos().abs() >= EPS => {
                let c = delta.cos().signum();
                vec![((c * kout.z).atan2(c * kout.y), delta, None)]
            }
            Detector::Delta(delta) => Roots::Free
                .tagged("nu")
                .into_iter()
                .map(|(nu, f)| (nu, delta, f))
                .collect(),
            Detector::Nu(nu) => {
                vec![(
                    nu,
                    kout.x.atan2(kout.y * nu.cos() + kout.z * nu.sin()),
                    None,
                )]
            }
            Detector::Qaz(_) | Detector::Naz(_) => arms(&kout, "nu"),
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
