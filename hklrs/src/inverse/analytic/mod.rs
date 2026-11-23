//! Closed-form inverse for E6C: the You (1999) 4S+2D constraint families,
//! solved as diffcalc-core (`059ec8a`, `src/diffcalc/hkl/calc*.py`) solves
//! them, with its defects corrected (upstream-defects.md DC-1..DC-6).
//!
//! The solver works in You's frame and circle names; [`e6c`] is the only
//! boundary, where [`frame`] and [`E6C`] map hklrs's E6C onto it. Every
//! candidate is checked against the Q round trip and every constraint before
//! it is returned. Where an equation leaves an angle free the solver samples
//! it, and a sample that passes the check is reported as
//! [`Error::Degenerate`].

mod sample;
mod solve;
mod you;

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use nalgebra::{Matrix3, Vector3};

use crate::Wavelength;
use you::{Angles, EPS};

/// Largest Q miss (units of k) and constraint miss (rad, or sine for the
/// angles You defines by their sine) a returned solution may have.
const TOL: f64 = 1e-9;

/// Solutions closer than this in every circle (rad) are one.
const SAME: f64 = 1e-9;

/// hklrs's E6C lab (beam +x, `mu` and `gamma` about z) as You's lab (beam
/// +y, μ and ν about x): x → y, y → z, z → x. Lab and sample-frame vectors
/// map by `frame() * v`, rotations by `frame() * r * frame().transpose()`.
pub fn frame() -> Matrix3<f64> {
    Matrix3::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
}

/// Each hklrs E6C axis in geometry order, the You circle it is under
/// [`frame`], and the sign from the hklrs value to You's.
pub const E6C: [(&str, &str, f64); 6] = [
    ("mu", "mu", 1.0),
    ("omega", "eta", 1.0),
    ("chi", "chi", 1.0),
    ("phi", "phi", 1.0),
    ("gamma", "nu", 1.0),
    ("delta", "delta", 1.0),
];

/// Fixes the detector, or with `Naz` the azimuth about the beam of the
/// reference vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Detector {
    Delta(f64),
    Nu(f64),
    /// Azimuth of Q about the beam, atan2(Q_x, Q_z).
    Qaz(f64),
    /// Azimuth of the reference vector about the beam, atan2(n_x, n_z).
    Naz(f64),
}

/// Fixes the reference vector n (`Alpha`, `Beta`, `AEqB`, `Psi`) or the
/// surface normal s (`Betain`, `Betaout`, `BinEqBout`) relative to the beams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference {
    /// asin(−n·k̂_in).
    Alpha(f64),
    /// asin(n·k̂_out).
    Beta(f64),
    AEqB,
    /// Azimuth of n about Q̂, zero when n is in the plane of Q̂ and the beam.
    Psi(f64),
    /// asin(−s·k̂_in).
    Betain(f64),
    /// asin(s·k̂_out).
    Betaout(f64),
    BinEqBout,
}

impl Reference {
    fn on_surface(&self) -> bool {
        matches!(
            self,
            Reference::Betain(_) | Reference::Betaout(_) | Reference::BinEqBout
        )
    }
}

/// A sample circle of Z = MU·ETA·CHI·PHI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Circle {
    Mu,
    Eta,
    Chi,
    Phi,
}

/// `N` distinct sample circles held at values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Held<const N: usize>([(Circle, f64); N]);

impl<const N: usize> Held<N> {
    pub fn new(held: [(Circle, f64); N]) -> Result<Self, Error> {
        for (i, (c, _)) in held.iter().enumerate() {
            if held[..i].iter().any(|(d, _)| d == c) {
                return Err(Error::Unsupported {
                    names: format!("{c:?} held twice"),
                });
            }
        }
        Ok(Self(held))
    }

    fn known(&self) -> sample::Known {
        let mut known = [None; 4];
        for &(c, v) in &self.0 {
            known[c as usize] = Some(v);
        }
        known
    }
}

/// One You (1999) constraint family with its values, radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Constraints {
    DetRefSample(Detector, Reference, Held<1>),
}

impl Constraints {
    /// From You's names as diffcalc's `Constraints` keys them, values in
    /// radians; `a_eq_b` and `bin_eq_bout` take `None`.
    pub fn from_names(named: &[(&str, Option<f64>)]) -> Result<Self, Error> {
        let unsupported = || Error::Unsupported {
            names: named.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("+"),
        };
        let (mut det, mut refs, mut held) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &(name, value)) in named.iter().enumerate() {
            if named[..i].iter().any(|(n, _)| *n == name) {
                return Err(unsupported());
            }
            let v = || match value {
                Some(v) if v.is_finite() => Ok(v),
                Some(_) => Err(Error::NonFinite {
                    what: "constraint value",
                }),
                None => Err(Error::Value {
                    name: name.to_owned(),
                    problem: "needs a value",
                }),
            };
            let flag = || match value {
                None => Ok(()),
                Some(_) => Err(Error::Value {
                    name: name.to_owned(),
                    problem: "takes no value",
                }),
            };
            match name {
                "delta" => det.push(Detector::Delta(v()?)),
                "nu" => det.push(Detector::Nu(v()?)),
                "qaz" => det.push(Detector::Qaz(v()?)),
                "naz" => det.push(Detector::Naz(v()?)),
                "alpha" => refs.push(Reference::Alpha(v()?)),
                "beta" => refs.push(Reference::Beta(v()?)),
                "psi" => refs.push(Reference::Psi(v()?)),
                "betain" => refs.push(Reference::Betain(v()?)),
                "betaout" => refs.push(Reference::Betaout(v()?)),
                "a_eq_b" => refs.push(flag().map(|()| Reference::AEqB)?),
                "bin_eq_bout" => refs.push(flag().map(|()| Reference::BinEqBout)?),
                "mu" => held.push((Circle::Mu, v()?)),
                "eta" => held.push((Circle::Eta, v()?)),
                "chi" => held.push((Circle::Chi, v()?)),
                "phi" => held.push((Circle::Phi, v()?)),
                _ => return Err(unsupported()),
            }
        }
        match (&det[..], &refs[..], &held[..]) {
            (&[d], &[r], &[h]) => Ok(Constraints::DetRefSample(d, r, Held::new([h])?)),
            _ => Err(unsupported()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("|Q| = {q} Å⁻¹ exceeds 4π/λ = {q_max} Å⁻¹")]
    Unreachable { q: f64, q_max: f64 },
    #[error("the solutions form a continuous family in {what}")]
    Degenerate { what: &'static str },
    #[error("psi is undefined: the reference is parallel to Q or Q is along the beam")]
    PsiUndefined,
    #[error("{what} is not finite")]
    NonFinite { what: &'static str },
    #[error("{what} is zero")]
    ZeroReference { what: &'static str },
    #[error("{names} is not a You (1999) constraint set")]
    Unsupported { names: String },
    #[error("constraint {name:?} {problem}")]
    Value { name: String, problem: &'static str },
}

/// Every E6C position, hklrs axis order (mu, omega, chi, phi, gamma, delta),
/// radians in (−π, π], that brings `q_sample` (Å⁻¹ with 2π, hklrs sample
/// frame) into reflection and meets `constraints`. `reference` and `surface`
/// are the reference vector and surface normal in the same frame as
/// `q_sample`; only their directions matter (DC-7).
pub fn e6c(
    constraints: &Constraints,
    q_sample: &Vector3<f64>,
    wavelength: Wavelength,
    reference: &Vector3<f64>,
    surface: &Vector3<f64>,
) -> Result<Vec<[f64; 6]>, Error> {
    let r = frame();
    let problem = solve::Problem::new(
        &(r * q_sample),
        wavelength.wavenumber(),
        &(r * reference),
        &(r * surface),
    )?;
    if let Constraints::DetRefSample(_, Reference::Psi(_), _) = constraints {
        if !problem.psi_defined() {
            return Err(Error::PsiUndefined);
        }
    }
    let to_e6c = E6C.map(|(_, you, sign)| {
        let c = sample::NAMES.iter().position(|n| *n == you).unwrap();
        (c, sign)
    });
    let mut out: Vec<[f64; 6]> = Vec::new();
    for (a, free) in problem.candidates(constraints) {
        if !satisfied(constraints, &problem, &a) {
            continue;
        }
        if let Some(what) = free {
            return Err(Error::Degenerate { what });
        }
        let v = to_e6c.map(|(c, sign)| wrap(sign * a[c]));
        if !out
            .iter()
            .any(|u| u.iter().zip(&v).all(|(x, y)| wrap(x - y).abs() < SAME))
        {
            out.push(v);
        }
    }
    Ok(out)
}

/// Residue of `a` modulo 2π in (−π, π].
fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r > PI { r - TAU } else { r }
}

fn near(angle: f64) -> bool {
    wrap(angle).abs() <= TOL
}

/// `sine` is sin(value) for an angle You defines as an arcsine.
fn arcsine(sine: f64, value: f64) -> bool {
    value.abs() <= FRAC_PI_2 + TOL && (sine - value.sin()).abs() <= TOL
}

/// The final check every candidate passes: Q round trip and each constraint
/// by its definition, independent of how the candidate was found.
fn satisfied(constraints: &Constraints, p: &solve::Problem, a: &Angles) -> bool {
    let z = you::z(a);
    let kout = you::kout(a);
    let q = kout - Vector3::y();
    let (n, s) = (z * p.n, z * p.s);
    if (z.transpose() * q - p.q()).norm() > TOL {
        return false;
    }
    let detector = |d: &Detector| match *d {
        Detector::Delta(v) => near(a[you::DELTA] - v),
        Detector::Nu(v) => near(a[you::NU] - v),
        Detector::Qaz(v) => q.x.hypot(q.z) > EPS && near(q.x.atan2(q.z) - v),
        Detector::Naz(v) => n.x.hypot(n.z) > EPS && near(n.x.atan2(n.z) - v),
    };
    let reference = |r: &Reference| match *r {
        Reference::Alpha(v) => arcsine(-n.y, v),
        Reference::Beta(v) => arcsine(n.dot(&kout), v),
        Reference::AEqB => (n.y + n.dot(&kout)).abs() <= TOL,
        Reference::Psi(v) => {
            let qh = q.normalize();
            let (y, x) = (n.cross(&qh).y, n.y - n.dot(&qh) * qh.y);
            x.hypot(y) > EPS && near(y.atan2(x) - v)
        }
        Reference::Betain(v) => arcsine(-s.y, v),
        Reference::Betaout(v) => arcsine(s.dot(&kout), v),
        Reference::BinEqBout => (s.y + s.dot(&kout)).abs() <= TOL,
    };
    let held = |h: &[(Circle, f64)]| h.iter().all(|&(c, v)| near(a[c as usize] - v));
    match constraints {
        Constraints::DetRefSample(d, r, h) => detector(d) && reference(r) && held(&h.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AxisKind, presets::preset};

    /// [`frame`] and [`E6C`] against the E6C preset: each hklrs axis maps to
    /// its You circle's axis with its sign, and the beam to +y.
    #[test]
    fn frame_maps_e6c_onto_you() {
        let geo = preset("E6C").unwrap();
        let r = frame();
        assert_eq!(r.determinant(), 1.0);
        assert_eq!(r * geo.beam_direction().into_inner(), Vector3::y());
        for (axis, (name, you, sign)) in geo.axes().iter().zip(E6C) {
            assert_eq!(axis.name(), name);
            let AxisKind::Rotation { axis: v } = axis.kind() else {
                panic!("{name} is not a rotation");
            };
            let c = sample::NAMES.iter().position(|n| *n == you).unwrap();
            assert_eq!(r * v.into_inner(), sign * you::axis(c), "{name}");
        }
    }
}
