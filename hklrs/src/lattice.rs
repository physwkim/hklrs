//! Direct lattice and the Busing-Levy B matrix.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use nalgebra::{Matrix3, Vector3};

use crate::{Error, Rad, units::Deg};

/// Lengths in Å, angles in radians. Every value satisfies a, b, c > 0,
/// angles in (0, π) and V² > 0, so every value has a B matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl Lattice {
    pub fn new(a: f64, b: f64, c: f64, alpha: Rad, beta: Rad, gamma: Rad) -> Result<Self, Error> {
        for (what, v) in [("a", a), ("b", b), ("c", c)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::NotPositive { what, value: v });
            }
        }
        for (name, v) in [("alpha", alpha.0), ("beta", beta.0), ("gamma", gamma.0)] {
            if !(v.is_finite() && v > 0.0 && v < PI) {
                return Err(Error::LatticeAngle { name, value: v });
            }
        }
        let l = Self {
            a,
            b,
            c,
            alpha: alpha.0,
            beta: beta.0,
            gamma: gamma.0,
        };
        let v2 = l.volume_factor_squared();
        if v2 > 0.0 {
            Ok(l)
        } else {
            Err(Error::LatticeVolume { v2 })
        }
    }

    /// Same as `new` with angles in degrees.
    pub fn from_degrees(
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<Self, Error> {
        Self::new(
            a,
            b,
            c,
            Deg(alpha).into(),
            Deg(beta).into(),
            Deg(gamma).into(),
        )
    }

    pub fn cubic(a: f64) -> Result<Self, Error> {
        Self::new(a, a, a, Rad(FRAC_PI_2), Rad(FRAC_PI_2), Rad(FRAC_PI_2))
    }

    pub fn tetragonal(a: f64, c: f64) -> Result<Self, Error> {
        Self::new(a, a, c, Rad(FRAC_PI_2), Rad(FRAC_PI_2), Rad(FRAC_PI_2))
    }

    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Result<Self, Error> {
        Self::new(a, b, c, Rad(FRAC_PI_2), Rad(FRAC_PI_2), Rad(FRAC_PI_2))
    }

    pub fn hexagonal(a: f64, c: f64) -> Result<Self, Error> {
        Self::new(a, a, c, Rad(FRAC_PI_2), Rad(FRAC_PI_2), Rad(2.0 * PI / 3.0))
    }

    pub fn rhombohedral(a: f64, alpha: Rad) -> Result<Self, Error> {
        Self::new(a, a, a, alpha, alpha, alpha)
    }

    pub fn monoclinic(a: f64, b: f64, c: f64, beta: Rad) -> Result<Self, Error> {
        Self::new(a, b, c, Rad(FRAC_PI_2), beta, Rad(FRAC_PI_2))
    }

    pub fn a(&self) -> f64 {
        self.a
    }
    pub fn b(&self) -> f64 {
        self.b
    }
    pub fn c(&self) -> f64 {
        self.c
    }
    pub fn alpha(&self) -> Rad {
        Rad(self.alpha)
    }
    pub fn beta(&self) -> Rad {
        Rad(self.beta)
    }
    pub fn gamma(&self) -> Rad {
        Rad(self.gamma)
    }

    /// V² / (abc)².
    fn volume_factor_squared(&self) -> f64 {
        let (ca, cb, cg) = (self.alpha.cos(), self.beta.cos(), self.gamma.cos());
        1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    }

    /// Cell volume in Å³.
    pub fn volume(&self) -> f64 {
        self.a * self.b * self.c * self.volume_factor_squared().sqrt()
    }

    /// Direct metric tensor G (Å²).
    pub fn metric(&self) -> Matrix3<f64> {
        let (a, b, c) = (self.a, self.b, self.c);
        let (ca, cb, cg) = (self.alpha.cos(), self.beta.cos(), self.gamma.cos());
        Matrix3::new(
            a * a,
            a * b * cg,
            a * c * cb,
            a * b * cg,
            b * b,
            b * c * ca,
            a * c * cb,
            b * c * ca,
            c * c,
        )
    }

    /// Crystallographic reciprocal lattice, without 2π: a* = 1/d(100) for
    /// orthogonal axes. Lengths in Å⁻¹.
    pub fn reciprocal(&self) -> Lattice {
        let (sa, sb, sg) = (self.alpha.sin(), self.beta.sin(), self.gamma.sin());
        let (ca, cb, cg) = (self.alpha.cos(), self.beta.cos(), self.gamma.cos());
        let v = self.volume();
        // sin α* = (V/abc)/(sin β sin γ) and cos α* = (cos β cos γ − cos α)/(sin β sin γ);
        // the common positive divisor drops out of atan2.
        let s = self.volume_factor_squared().sqrt();
        let angle = |c1: f64, c2: f64, c3: f64| f64::atan2(s, c2 * c3 - c1);
        // V* = 1/V > 0 and atan2 with s > 0 lies in (0, π), so the reciprocal
        // of a valid lattice is valid without re-checking.
        Lattice {
            a: self.b * self.c * sa / v,
            b: self.a * self.c * sb / v,
            c: self.a * self.b * sg / v,
            alpha: angle(ca, cb, cg),
            beta: angle(cb, ca, cg),
            gamma: angle(cg, ca, cb),
        }
    }

    /// Busing-Levy B (Å⁻¹, includes 2π): upper triangular, B·h is the
    /// scattering vector of reflection h in the crystal Cartesian frame.
    pub fn b_matrix(&self) -> Matrix3<f64> {
        let r = self.reciprocal();
        let (sb, cb) = r.beta.sin_cos();
        let (sg, cg) = r.gamma.sin_cos();
        TAU * Matrix3::new(
            r.a,
            r.b * cg,
            r.c * cb,
            0.0,
            r.b * sg,
            -r.c * sb * self.alpha.cos(),
            0.0,
            0.0,
            1.0 / self.c,
        )
    }

    /// d-spacing of reflection `hkl` in Å, from the same B as every other path.
    pub fn d_spacing(&self, hkl: &Vector3<f64>) -> f64 {
        TAU / (self.b_matrix() * hkl).norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn rejects_invalid() {
        assert!(Lattice::cubic(0.0).is_err());
        assert!(Lattice::cubic(f64::NAN).is_err());
        assert!(Lattice::from_degrees(1.0, 1.0, 1.0, 0.0, 90.0, 90.0).is_err());
        assert!(Lattice::from_degrees(1.0, 1.0, 1.0, 180.0, 90.0, 90.0).is_err());
        // alpha + beta < gamma: no cell, V² < 0
        assert!(matches!(
            Lattice::from_degrees(1.0, 1.0, 1.0, 30.0, 30.0, 120.0),
            Err(Error::LatticeVolume { .. })
        ));
    }

    #[test]
    fn cubic_b_is_diagonal() {
        let b = Lattice::cubic(1.54).unwrap().b_matrix();
        let k = TAU / 1.54;
        assert!((b - Matrix3::from_diagonal_element(k)).abs().max() < 1e-15);
    }

    #[test]
    fn b_reproduces_metric() {
        // Bᵀ B = (2π)² G⁻¹ for any cell.
        let l = Lattice::from_degrees(4.1, 5.3, 6.7, 82.0, 97.0, 101.0).unwrap();
        let b = l.b_matrix();
        let g_inv = l.metric().try_inverse().unwrap() * TAU * TAU;
        assert!((b.transpose() * b - g_inv).abs().max() < 1e-12);
    }

    #[test]
    fn reciprocal_of_reciprocal_is_direct() {
        let l = Lattice::from_degrees(4.1, 5.3, 6.7, 82.0, 97.0, 101.0).unwrap();
        let rr = l.reciprocal().reciprocal();
        for (x, y) in [
            (rr.a, l.a),
            (rr.b, l.b),
            (rr.c, l.c),
            (rr.alpha, l.alpha),
            (rr.beta, l.beta),
            (rr.gamma, l.gamma),
        ] {
            close(x, y, 1e-12);
        }
        close(l.volume() * l.reciprocal().volume(), 1.0, 1e-14);
    }

    #[test]
    fn hexagonal_d_spacing() {
        // d(100) = a·√3/2 for hexagonal.
        let l = Lattice::hexagonal(3.0, 5.0).unwrap();
        close(
            l.d_spacing(&Vector3::new(1.0, 0.0, 0.0)),
            3.0 * 3f64.sqrt() / 2.0,
            1e-14,
        );
        close(l.d_spacing(&Vector3::new(0.0, 0.0, 1.0)), 5.0, 1e-14);
    }
}
