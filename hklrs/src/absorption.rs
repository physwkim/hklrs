//! Absorption weights (design §5.1). The correction divides the intensity
//! by the weight.

use nalgebra::{Unit, UnitQuaternion, Vector3};

use crate::units::positive;
use crate::{Energy, Error, PerMetre};

/// Linear attenuation of a material at a photon energy. The core holds no
/// material tables; a constant μ implements it directly.
pub trait Attenuation {
    fn mu(&self, energy: Energy) -> PerMetre;
}

impl Attenuation for PerMetre {
    fn mu(&self, _energy: Energy) -> PerMetre {
        *self
    }
}

/// Why a pixel has no sample weight. `s_i = −k̂i·n̂`, `s_f = k̂f·n̂`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Masked {
    /// `s_i ≤ 0`: the beam does not enter through the face.
    Unlit,
    /// `s_f = 0`: the scattered beam leaves along the face.
    GrazingExit,
    /// `s_f < 0` on a bulk sample: no beam leaves through the back.
    ThroughBulk,
}

fn unit(v: Vector3<f64>, what: &str) -> Result<Unit<Vector3<f64>>, Error> {
    if v.iter().all(|x| x.is_finite()) {
        if let Some(u) = Unit::try_new(v, 0.0) {
            return Ok(u);
        }
    }
    Err(Error::ZeroVector {
        what: what.to_owned(),
    })
}

/// The rectangular face the beam enters, sample frame, centred on the
/// sample position (the beam axis passes through its centre), metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    normal: Unit<Vector3<f64>>,
    along: Unit<Vector3<f64>>,
    length: f64,
    width: f64,
}

impl Face {
    /// `normal` points out of the sample; `length` runs along the part of
    /// `axis` in the face plane, `width` across it.
    pub fn new(
        normal: Vector3<f64>,
        axis: Vector3<f64>,
        length: f64,
        width: f64,
    ) -> Result<Self, Error> {
        let normal = unit(normal, "face normal")?;
        let along = unit(
            axis - axis.dot(&normal) * normal.into_inner(),
            "face axis in the face plane",
        )?;
        Ok(Self {
            normal,
            along,
            length: positive("face length", length)?,
            width: positive("face width", width)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Thickness {
    Bulk,
    Finite(f64),
}

/// A flat sample: its entrance face and its thickness along the normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slab {
    face: Face,
    thickness: Thickness,
}

impl Slab {
    /// A slab `thickness` metres thick.
    pub fn new(face: Face, thickness: f64) -> Result<Self, Error> {
        Ok(Self {
            face,
            thickness: Thickness::Finite(positive("slab thickness", thickness)?),
        })
    }

    /// A crystal too thick for the beam to leave through the back.
    pub fn bulk(face: Face) -> Self {
        Self {
            face,
            thickness: Thickness::Bulk,
        }
    }
}

/// Beam profile across k̂i with unit total flux, centred on the sample
/// position. Its first size lies along `axis` projected across k̂i, the
/// second along k̂i × that, metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beam {
    axis: Unit<Vector3<f64>>,
    profile: Profile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Profile {
    /// Full widths.
    TopHat([f64; 2]),
    /// Standard deviations.
    Gaussian([f64; 2]),
}

fn sizes(what: &'static str, size: [f64; 2]) -> Result<[f64; 2], Error> {
    Ok([positive(what, size[0])?, positive(what, size[1])?])
}

impl Beam {
    /// Uniform over a `width[0] × width[1]` rectangle.
    pub fn top_hat(axis: Vector3<f64>, width: [f64; 2]) -> Result<Self, Error> {
        Ok(Self {
            axis: unit(axis, "beam profile axis")?,
            profile: Profile::TopHat(sizes("beam width", width)?),
        })
    }

    /// Gaussian with full widths at half maximum `fwhm`.
    pub fn gaussian_fwhm(axis: Vector3<f64>, fwhm: [f64; 2]) -> Result<Self, Error> {
        let to_sigma = 1.0 / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt());
        let [f0, f1] = sizes("beam FWHM", fwhm)?;
        Ok(Self {
            axis: unit(axis, "beam profile axis")?,
            profile: Profile::Gaussian([f0 * to_sigma, f1 * to_sigma]),
        })
    }

    /// Fraction of the flux that falls inside `poly`, a convex polygon
    /// around the origin in the profile's own coordinates.
    fn fraction(&self, poly: &[[f64; 2]]) -> f64 {
        match self.profile {
            Profile::TopHat([w0, w1]) => area(&clip_to_box(poly, [w0 / 2.0, w1 / 2.0])) / (w0 * w1),
            Profile::Gaussian([s0, s1]) => {
                let z: Vec<_> = poly.iter().map(|p| [p[0] / s0, p[1] / s1]).collect();
                normal_mass(&clip_to_box(&z, [REACH; 2]))
            }
        }
    }
}

/// Half-width in σ of the box a Gaussian is integrated over; the flux
/// outside it, about 4e-17, is below half an ulp of 1.
const REACH: f64 = 8.5;

/// Positive nodes and weights of 8-point Gauss-Legendre on [−1, 1].
const GAUSS_LEGENDRE_8: [(f64, f64); 4] = [
    (0.18343464249564978, 0.36268378337836166),
    (0.525532409916329, 0.3137066458778869),
    (0.7966664774136267, 0.22238103445337443),
    (0.9602898564975362, 0.10122853629037706),
];

/// ∫ f over [lo, hi], 8-point Gauss-Legendre on each piece between
/// consecutive integers.
fn integrate(lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let mut sum = 0.0;
    let mut a = lo;
    while a < hi {
        let b = (a.floor() + 1.0).min(hi);
        let (c, h) = ((a + b) / 2.0, (b - a) / 2.0);
        sum += h * GAUSS_LEGENDRE_8
            .iter()
            .map(|&(x, w)| w * (f(c - h * x) + f(c + h * x)))
            .sum::<f64>();
        a = b;
    }
    sum
}

/// Standard bivariate normal mass inside a convex polygon around the
/// origin, as the sum over edges AB of the triangle 0AB. Its radial
/// integral is closed, leaving (d/2π) ∫ g(d² + t²) dt along the edge, with
/// d the edge line's signed distance from the origin, t the position along
/// it, and g(s) = (1 − e^(−s/2))/s entire, so fixed-order quadrature
/// converges at any edge slope; around the origin no two terms cancel.
fn normal_mass(poly: &[[f64; 2]]) -> f64 {
    let g = |s: f64| {
        if s == 0.0 {
            0.5
        } else {
            -(-s / 2.0).exp_m1() / s
        }
    };
    let mut sum = 0.0;
    for (i, a) in poly.iter().enumerate() {
        let b = poly[(i + 1) % poly.len()];
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy);
        if len == 0.0 {
            continue;
        }
        let d = (a[0] * dy - a[1] * dx) / len;
        let t0 = (a[0] * dx + a[1] * dy) / len;
        sum += d * integrate(t0, t0 + len, |t| g(d * d + t * t));
    }
    sum.abs() / std::f64::consts::TAU
}

/// Sample-side state of one frame: everything that depends on k̂i and R_s
/// alone, so a pixel costs one dot product and two exponentials.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlabFrame {
    /// n̂ in the lab frame.
    normal: Unit<Vector3<f64>>,
    mu: f64,
    thickness: Thickness,
    lit: Result<Lit, Masked>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Lit {
    s_i: f64,
    footprint: f64,
}

impl SlabFrame {
    /// `mu` is the sample's [`Attenuation::mu`] at the frame energy, `ki`
    /// the incident direction and `r_s` the sample rotation, lab frame.
    /// Fails only when `beam`'s axis is parallel to `ki`.
    pub fn new(
        slab: &Slab,
        beam: &Beam,
        mu: PerMetre,
        ki: &Unit<Vector3<f64>>,
        r_s: &UnitQuaternion<f64>,
    ) -> Result<Self, Error> {
        let u = unit(
            beam.axis.into_inner() - beam.axis.dot(ki) * ki.into_inner(),
            "beam profile axis across k_i",
        )?;
        let v = ki.cross(&u);
        let face = &slab.face;
        let normal = r_s * face.normal;
        let s_i = -ki.dot(&normal);
        let lit = if s_i > 0.0 {
            // The projection along k̂i scales face areas by s_i, so F is the
            // intercepted flux fraction over s_i.
            let a = r_s * (face.along.into_inner() * (face.length / 2.0));
            let b = r_s * (face.normal.cross(&face.along) * (face.width / 2.0));
            let (au, av, bu, bv) = (a.dot(&u), a.dot(&v), b.dot(&u), b.dot(&v));
            let poly = [
                [au + bu, av + bv],
                [bu - au, bv - av],
                [-au - bu, -av - bv],
                [au - bu, av - bv],
            ];
            Ok(Lit {
                s_i,
                footprint: beam.fraction(&poly) / s_i,
            })
        } else {
            Err(Masked::Unlit)
        };
        Ok(Self {
            normal,
            mu: mu.per_metre(),
            thickness: slab.thickness,
            lit,
        })
    }

    /// F, the beam flux per unit face area integrated over the face:
    /// 1/s_i with the whole footprint on the face, less when it spills.
    pub fn footprint(&self) -> Result<f64, Masked> {
        self.lit.map(|l| l.footprint)
    }

    /// A = F ∫₀ᵗ exp(−μz/s_i − μ L_f(z)) dz in metres for the scattered
    /// direction `kf`; L_f = z/s_f in reflection, (t − z)/|s_f| in
    /// transmission.
    pub fn weight(&self, kf: &Unit<Vector3<f64>>) -> Result<f64, Masked> {
        let lit = self.lit?;
        let s_f = kf.dot(&self.normal);
        let b = self.mu / lit.s_i;
        let path = if s_f > 0.0 {
            let a = b + self.mu / s_f;
            match self.thickness {
                Thickness::Bulk => 1.0 / a,
                Thickness::Finite(t) => t * phi(-a * t),
            }
        } else if s_f < 0.0 {
            let Thickness::Finite(t) = self.thickness else {
                return Err(Masked::ThroughBulk);
            };
            // ∫₀ᵗ e^(−bz − c(t−z)) dz = t e^(−min(b,c) t) φ(−|b − c| t):
            // φ's argument is never positive, so nothing overflows, and
            // |s_f| = s_i is φ(0) = 1.
            let c = self.mu / -s_f;
            t * (-b.min(c) * t).exp() * phi(-(b - c).abs() * t)
        } else {
            return Err(Masked::GrazingExit);
        };
        Ok(lit.footprint * path)
    }
}

/// A flat layer on the detector, crossed by the scattered beam.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    mu: PerMetre,
    thickness: f64,
}

impl Layer {
    /// `thickness` along the detector normal, metres.
    pub fn new(mu: PerMetre, thickness: f64) -> Result<Self, Error> {
        Ok(Self {
            mu,
            thickness: positive("layer thickness", thickness)?,
        })
    }

    /// Optical depth along a ray at `cos θ` to the normal.
    fn depth(&self, cos: f64) -> f64 {
        self.mu.per_metre() * self.thickness / cos
    }
}

/// Detector-side terms at one energy; an absent term weighs 1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DetectorPath {
    /// Air or He between the sample and the pixel.
    pub air: Option<PerMetre>,
    /// Window or filter, transmitted.
    pub window: Option<Layer>,
    /// Sensor, absorbed.
    pub sensor: Option<Layer>,
}

impl DetectorPath {
    /// Weight of the pixel at `p` from the sample, on a detector with
    /// `normal` of either sense, lab frame.
    pub fn weight(&self, p: &Vector3<f64>, normal: &Unit<Vector3<f64>>) -> Result<f64, Error> {
        let (kf, distance) = Unit::try_new_and_get(*p, 0.0).ok_or(Error::DetectorAtSample)?;
        let cos = kf.dot(normal).abs();
        let air = self
            .air
            .map_or(1.0, |mu| (-mu.per_metre() * distance).exp());
        let window = self.window.map_or(1.0, |w| (-w.depth(cos)).exp());
        let sensor = self.sensor.map_or(1.0, |s| -(-s.depth(cos)).exp_m1());
        Ok(air * window * sensor)
    }
}

/// φ(x) = expm1(x)/x with φ(0) = 1, so t φ(−at) = ∫₀ᵗ e^(−az) dz holds at
/// a = 0 instead of being 0/0.
fn phi(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { x.exp_m1() / x }
}

/// Sutherland-Hodgman clip of a convex polygon to |x| ≤ half[0],
/// |y| ≤ half[1]. The clipped coordinate is set to the bound exactly, so a
/// box inside the polygon comes back as its own corners.
fn clip_to_box(poly: &[[f64; 2]], half: [f64; 2]) -> Vec<[f64; 2]> {
    let mut poly = poly.to_vec();
    for (axis, sign) in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)] {
        let inside = |p: &[f64; 2]| sign * p[axis] <= half[axis];
        let bound = sign * half[axis];
        let mut out = Vec::with_capacity(poly.len() + 1);
        for (i, p) in poly.iter().enumerate() {
            let q = &poly[(i + 1) % poly.len()];
            if inside(p) {
                out.push(*p);
            }
            if inside(p) != inside(q) {
                let t = (bound - p[axis]) / (q[axis] - p[axis]);
                let mut r = [bound; 2];
                r[1 - axis] = p[1 - axis] + t * (q[1 - axis] - p[1 - axis]);
                out.push(r);
            }
        }
        poly = out;
    }
    poly
}

/// Unsigned area of a simple polygon (shoelace).
fn area(poly: &[[f64; 2]]) -> f64 {
    let twice: f64 = (0..poly.len())
        .map(|i| {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum();
    twice.abs() / 2.0
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, SQRT_2, TAU};

    use super::*;

    /// Relative closeness that fails on NaN.
    fn close(got: f64, want: f64, rel: f64) -> bool {
        (got - want).abs() <= rel * want.abs()
    }

    fn mu(v: f64) -> PerMetre {
        PerMetre::new(v).unwrap()
    }

    fn ki() -> Unit<Vector3<f64>> {
        Vector3::x_axis()
    }

    /// Sample frame: face normal ẑ, length along x̂; the beam along x̂ meets
    /// it at incidence `alpha` after R_s, a rotation about ŷ by −alpha.
    fn tilt(alpha: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -alpha)
    }

    fn face(length: f64, width: f64) -> Face {
        Face::new(Vector3::z(), Vector3::x(), length, width).unwrap()
    }

    /// A beam far smaller than the face: F = 1/s_i.
    fn pencil() -> Beam {
        Beam::top_hat(Vector3::z(), [1e-6, 1e-6]).unwrap()
    }

    fn frame(slab: &Slab, beam: &Beam, m: f64, alpha: f64) -> SlabFrame {
        SlabFrame::new(slab, beam, mu(m), &ki(), &tilt(alpha)).unwrap()
    }

    /// Exit direction in the incidence plane at 2θ from k̂i.
    fn kf(tth: f64) -> Unit<Vector3<f64>> {
        Unit::new_normalize(Vector3::new(tth.cos(), 0.0, tth.sin()))
    }

    #[test]
    fn bulk_symmetric_reflection_is_one_over_two_mu() {
        let slab = Slab::bulk(face(1.0, 1.0));
        for m in [0.5, 1e3, 4e5] {
            for alpha in [0.01, 0.1, 0.5, 1.0, 1.5] {
                let f = frame(&slab, &pencil(), m, alpha);
                let a = f.weight(&kf(2.0 * alpha)).unwrap();
                assert!(close(a, 1.0 / (2.0 * m), 1e-13), "μ {m} α {alpha}: {a}");
            }
        }
    }

    #[test]
    fn normal_transmission_is_t_exp_minus_mu_t() {
        let t = 2e-3;
        let slab = Slab::new(face(1.0, 1.0), t).unwrap();
        for mt in [1e-3, 0.5, 3.0, 20.0] {
            let f = frame(&slab, &pencil(), mt / t, FRAC_PI_2);
            let a = f.weight(&ki()).unwrap();
            assert!(close(a, t * (-mt).exp(), 1e-14), "μt {mt}: {a}");
        }
    }

    #[test]
    fn thin_slab_tends_to_t_over_s_i() {
        let t = 1e-4;
        let slab = Slab::new(face(1.0, 1.0), t).unwrap();
        let alpha = 0.3;
        let s_i = -ki().dot(&(tilt(alpha) * Vector3::z_axis()));
        // reflection, symmetric transmission, oblique transmission
        for tth in [2.0 * alpha, 0.0, -0.2] {
            for mt in [1e-6, 1e-12, 1e-300] {
                let f = frame(&slab, &pencil(), mt / t, alpha);
                let a = f.weight(&kf(tth)).unwrap();
                let s_f = kf(tth).dot(&(tilt(alpha) * Vector3::z_axis()));
                let bound = mt * (1.0 / s_i + 1.0 / s_f.abs()) + 1e-15;
                assert!(close(a, t / s_i, bound), "2θ {tth} μt {mt}: {a}");
            }
        }
    }

    #[test]
    fn transmission_at_equal_obliquity_is_exact() {
        let (t, m) = (1e-3, 800.0);
        let slab = Slab::new(face(1.0, 1.0), t).unwrap();
        for alpha in [0.05, 0.4, 1.2] {
            let f = frame(&slab, &pencil(), m, alpha);
            let s_i = -ki().dot(&(tilt(alpha) * Vector3::z_axis()));
            // k̂f = k̂i gives s_f = −s_i bit for bit: b = c, φ(0) = 1
            let a = f.weight(&ki()).unwrap();
            let want = f.footprint().unwrap() * t * (-m * t / s_i).exp();
            assert!(close(a, want, 4.0 * f64::EPSILON), "α {alpha}: {a}");
            // and the neighbours on either side agree with it to first
            // order: δA/A ≈ (μt/2) δ|s_f| / s_i²
            for d in [-1e-9, 1e-9] {
                let near = f.weight(&kf(d)).unwrap();
                let bound = m * t * d.abs() / (s_i * s_i);
                assert!(close(near, a, bound), "α {alpha} 2θ {d}: {near}");
            }
        }
    }

    /// Simpson's rule on the defining integral, for the branches of the
    /// closed form that the named limits do not reach.
    #[test]
    fn closed_form_matches_the_defining_integral() {
        let (t, m) = (3e-4, 4e3);
        let slab = Slab::new(face(1.0, 1.0), t).unwrap();
        for alpha in [0.1, 0.7] {
            let n = tilt(alpha) * Vector3::z_axis();
            let s_i = -ki().dot(&n);
            for tth in [0.5, 2.0, 0.05, -0.1, -0.6] {
                let k = kf(tth);
                let s_f = k.dot(&n);
                let l_f = |z: f64| if s_f > 0.0 { z / s_f } else { (t - z) / -s_f };
                let g = |z: f64| (-m * z / s_i - m * l_f(z)).exp();
                let steps = 20000;
                let h = t / steps as f64;
                let simpson: f64 = (0..=steps)
                    .map(|j| {
                        let w = if j == 0 || j == steps {
                            1.0
                        } else {
                            [2.0, 4.0][j % 2]
                        };
                        w * g(j as f64 * h)
                    })
                    .sum::<f64>()
                    * h
                    / 3.0;
                let f = frame(&slab, &pencil(), m, alpha);
                let a = f.weight(&k).unwrap();
                let want = f.footprint().unwrap() * simpson;
                assert!(close(a, want, 1e-11), "α {alpha} 2θ {tth}: {a} vs {want}");
            }
        }
    }

    #[test]
    fn top_hat_footprint_is_min_of_l_over_w_and_one_over_s_i() {
        let (l, w) = (0.01, 1e-4);
        // the face is wider than the beam across the incidence plane
        let slab = Slab::bulk(face(l, 1e-3));
        let beam = Beam::top_hat(Vector3::z(), [w, 5e-4]).unwrap();
        for alpha in [1e-3, 3e-3, 0.005, 0.01, 0.0125, 0.1, 1.0, 1.5] {
            let f = frame(&slab, &beam, 1.0, alpha).footprint().unwrap();
            let s_i = -ki().dot(&(tilt(alpha) * Vector3::z_axis()));
            let want = (l / w).min(1.0 / s_i);
            assert!(close(f, want, 1e-12), "α {alpha}: {f} vs {want}");
        }
    }

    /// erf(x) from scipy.special.erf.
    const ERF: [(f64, f64); 4] = [
        (0.5, 0.5204998778130465),
        (1.0, 0.8427007929497148),
        (2.0, 0.9953222650189527),
        (3.0, 0.9999779095030014),
    ];

    /// FWHM = 2√(2 ln 2) σ.
    const FWHM_PER_SIGMA: f64 = 2.3548200450309493;

    fn gaussian(sigma: [f64; 2]) -> Beam {
        Beam::gaussian_fwhm(Vector3::z(), sigma.map(|s| s * FWHM_PER_SIGMA)).unwrap()
    }

    /// Degree 0 holds the weights to a sum of exactly 1 on [0, 1]; above
    /// it, powi rounds once per multiply.
    #[test]
    fn gauss_legendre_8_is_exact_to_degree_15() {
        assert_eq!(integrate(0.0, 1.0, |_| 1.0), 1.0);
        for k in 1..=15 {
            let got = integrate(0.0, 1.0, |x| x.powi(k));
            let tol = 2.0 * f64::from(k + 1) * f64::EPSILON;
            assert!(close(got, 1.0 / f64::from(k + 1), tol), "{k}: {got}");
        }
        // pieces split at the integers
        assert!(close(
            integrate(-2.5, 1.75, |x| x.powi(15)),
            (1.75f64.powi(16) - 2.5f64.powi(16)) / 16.0,
            1e-14
        ));
    }

    /// At normal incidence the face projects onto itself, so half-sizes of
    /// √2·x0·σ0 and √2·x1·σ1 intercept erf(x0)·erf(x1) of the flux.
    #[test]
    fn gaussian_footprint_is_the_erf_product_at_normal_incidence() {
        let sigma = [1e-4, 3e-4];
        let beam = gaussian(sigma);
        for (x0, e0) in ERF {
            for (x1, e1) in ERF {
                let half = [SQRT_2 * x0 * sigma[0], SQRT_2 * x1 * sigma[1]];
                let slab = Slab::bulk(face(2.0 * half[0], 2.0 * half[1]));
                let f = frame(&slab, &beam, 1.0, FRAC_PI_2).footprint().unwrap();
                assert!(close(f, e0 * e1, 1e-14), "x {x0} {x1}: {f}");
            }
        }
    }

    /// A round beam on a square face: the intercepted flux does not depend
    /// on how the square is turned, at any edge slope.
    #[test]
    fn gaussian_footprint_of_a_square_ignores_its_turn() {
        let s = 1e-4;
        let (x, e) = ERF[1];
        let side = 2.0 * SQRT_2 * x * s;
        for turn in [0.0, 1e-9, 0.3, FRAC_PI_4, 1.1] {
            let axis = Vector3::new(f64::cos(turn), f64::sin(turn), 0.0);
            let face = Face::new(Vector3::z(), axis, side, side).unwrap();
            let f = frame(&Slab::bulk(face), &gaussian([s, s]), 1.0, FRAC_PI_2)
                .footprint()
                .unwrap();
            assert!(close(f, e * e, 1e-14), "turn {turn}: {f}");
        }
    }

    /// With the whole footprint on the face the top-hat gives 1/s_i; the
    /// Gaussian reaches it as the face outgrows its tails, through
    /// erf(x)²/s_i for a projected half-size of √2·x·σ.
    #[test]
    fn gaussian_footprint_converges_to_the_full_footprint() {
        let s = 1e-4;
        let alpha = 0.2;
        let s_i = -ki().dot(&(tilt(alpha) * Vector3::z_axis()));
        let face_for = |x: f64| face(2.0 * SQRT_2 * x * s / s_i, 2.0 * SQRT_2 * x * s);
        let hat = Beam::top_hat(Vector3::z(), [s, s]).unwrap();
        let full = frame(&Slab::bulk(face_for(1.0)), &hat, 1.0, alpha);
        assert!(close(full.footprint().unwrap(), 1.0 / s_i, 1e-15));
        for (x, e) in ERF.into_iter().chain([(REACH / SQRT_2, 1.0), (20.0, 1.0)]) {
            let f = frame(&Slab::bulk(face_for(x)), &gaussian([s, s]), 1.0, alpha);
            let f = f.footprint().unwrap();
            assert!(close(f * s_i, e * e, 1e-14), "x {x}: {f}");
        }
    }

    /// At grazing incidence the footprint outgrows the face along the beam
    /// and F saturates at L times the peak line density across the face:
    /// L/w for a top-hat, L/(√(2π)σ) for a Gaussian, one value at
    /// w = √(2π)σ. The Gaussian's gap is y²/6 − O(y⁴), y = L s_i/(2σ).
    #[test]
    fn gaussian_footprint_saturates_at_the_top_hat_limit() {
        let (l, s) = (0.01, 1e-4);
        let w = TAU.sqrt() * s;
        // 1 m across: no spill sideways
        let slab = Slab::bulk(face(l, 1.0));
        let hat = Beam::top_hat(Vector3::z(), [w, s]).unwrap();
        let limit = l / w;
        let mut gap = f64::INFINITY;
        for alpha in [1e-2, 1e-3, 1e-4, 1e-5] {
            let s_i = -ki().dot(&(tilt(alpha) * Vector3::z_axis()));
            let h = frame(&slab, &hat, 1.0, alpha).footprint().unwrap();
            assert!(close(h, limit, 1e-12), "α {alpha}: top-hat {h}");
            let g = frame(&slab, &gaussian([s, s]), 1.0, alpha);
            let next = 1.0 - g.footprint().unwrap() / limit;
            let y = l * s_i / (2.0 * s);
            assert!(
                next > 0.0 && next <= y * y / 5.0 && next < gap,
                "α {alpha}: {next}"
            );
            gap = next;
        }
        assert!(gap <= 5e-8, "{gap}");
    }

    #[test]
    fn clip_to_box_gives_the_exact_overlap() {
        // a square of inradius 1 turned 45° over the box of inradius 1: a
        // regular octagon of inradius 1
        let r = SQRT_2;
        let diamond = [[r, 0.0], [0.0, r], [-r, 0.0], [0.0, -r]];
        let oct = area(&clip_to_box(&diamond, [1.0, 1.0]));
        assert!(close(oct, 8.0 * (r - 1.0), 1e-15), "{oct}");
        // a box inside the polygon comes back as its own corners
        let big = [[5.0, 5.0], [-5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]];
        let mut got = clip_to_box(&big, [0.3, 0.7]);
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            [[-0.3, -0.7], [-0.3, 0.7], [0.3, -0.7], [0.3, 0.7]].to_vec()
        );
        // disjoint: empty
        let far = [[3.0, 3.0], [2.0, 3.0], [2.0, 2.0], [3.0, 2.0]];
        assert_eq!(area(&clip_to_box(&far, [1.0, 1.0])), 0.0);
    }

    #[test]
    fn mask_unlit_when_s_i_is_not_positive() {
        let slab = Slab::new(face(1.0, 1.0), 1e-3).unwrap();
        // s_i = 0 exactly (beam along the face), and s_i < 0 (from behind)
        for r_s in [
            UnitQuaternion::identity(),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.2),
        ] {
            let f = SlabFrame::new(&slab, &pencil(), mu(10.0), &ki(), &r_s).unwrap();
            assert_eq!(f.footprint(), Err(Masked::Unlit));
            assert_eq!(f.weight(&kf(0.4)), Err(Masked::Unlit));
        }
    }

    #[test]
    fn mask_grazing_exit_when_s_f_is_zero() {
        // k̂f = ŷ lies in the face plane for every tilt about ŷ: s_f = 0
        let kf = Vector3::y_axis();
        for slab in [
            Slab::bulk(face(1.0, 1.0)),
            Slab::new(face(1.0, 1.0), 1e-3).unwrap(),
        ] {
            let f = frame(&slab, &pencil(), 10.0, 0.3);
            assert_eq!(f.weight(&kf), Err(Masked::GrazingExit));
        }
    }

    #[test]
    fn mask_through_bulk_when_s_f_is_negative() {
        let exit = kf(-0.1);
        let f = frame(&Slab::bulk(face(1.0, 1.0)), &pencil(), 10.0, 0.3);
        assert_eq!(f.weight(&exit), Err(Masked::ThroughBulk));
        // the same exit through a slab has a weight
        let slab = Slab::new(face(1.0, 1.0), 1e-3).unwrap();
        let a = frame(&slab, &pencil(), 10.0, 0.3).weight(&exit).unwrap();
        assert!(a > 0.0, "{a}");
    }

    #[test]
    fn rejects_degenerate_configuration() {
        assert!(Face::new(Vector3::z(), Vector3::z(), 1.0, 1.0).is_err());
        assert!(Face::new(Vector3::zeros(), Vector3::x(), 1.0, 1.0).is_err());
        assert!(Face::new(Vector3::z(), Vector3::x(), 0.0, 1.0).is_err());
        assert!(Slab::new(face(1.0, 1.0), f64::INFINITY).is_err());
        assert!(Beam::top_hat(Vector3::z(), [1.0, f64::NAN]).is_err());
        let along_ki = Beam::top_hat(Vector3::x(), [1.0, 1.0]).unwrap();
        let slab = Slab::bulk(face(1.0, 1.0));
        let r_s = tilt(0.3);
        assert!(SlabFrame::new(&slab, &along_ki, mu(1.0), &ki(), &r_s).is_err());
    }

    #[test]
    fn constant_mu_ignores_energy() {
        let mu = PerMetre::new(1234.5).unwrap();
        for kev in [1.0, 8.048, 100.0] {
            assert_eq!(mu.mu(Energy::new(kev).unwrap()), mu);
        }
    }

    fn layer(m: f64, t: f64) -> Layer {
        Layer::new(mu(m), t).unwrap()
    }

    #[test]
    fn detector_terms_at_normal_incidence() {
        let (p, n) = (Vector3::new(0.0, 0.0, 0.5), Vector3::z_axis());
        let alone = |path: DetectorPath| path.weight(&p, &n).unwrap();
        assert_eq!(alone(DetectorPath::default()), 1.0);
        let air = alone(DetectorPath {
            air: Some(mu(1.2)),
            ..Default::default()
        });
        assert!(close(air, (-0.6f64).exp(), 2.0 * f64::EPSILON), "{air}");
        let window = alone(DetectorPath {
            window: Some(layer(1e4, 25e-6)),
            ..Default::default()
        });
        assert!(
            close(window, (-0.25f64).exp(), 2.0 * f64::EPSILON),
            "{window}"
        );
        let sensor = alone(DetectorPath {
            sensor: Some(layer(3e3, 450e-6)),
            ..Default::default()
        });
        assert!(
            close(sensor, 1.0 - (-1.35f64).exp(), 4.0 * f64::EPSILON),
            "{sensor}"
        );
        let all = DetectorPath {
            air: Some(mu(1.2)),
            window: Some(layer(1e4, 25e-6)),
            sensor: Some(layer(3e3, 450e-6)),
        };
        let w = all.weight(&p, &n).unwrap();
        assert!(close(w, air * window * sensor, 2.0 * f64::EPSILON), "{w}");
    }

    /// At 60° to the normal, of either sense, the layers are crossed over
    /// twice their thickness; the air path is |p| at any angle.
    #[test]
    fn detector_layers_at_60_degrees_double_the_path() {
        let p = Vector3::new(3f64.sqrt(), 0.0, 1.0) * 0.25;
        let path = DetectorPath {
            air: Some(mu(1.2)),
            window: Some(layer(1e4, 25e-6)),
            sensor: Some(layer(3e3, 450e-6)),
        };
        let want = (-0.6f64).exp() * (-0.5f64).exp() * -(-2.7f64).exp_m1();
        for n in [Vector3::z_axis(), -Vector3::z_axis()] {
            let w = path.weight(&p, &n).unwrap();
            assert!(close(w, want, 1e-15), "{n:?}: {w}");
        }
    }

    /// 1 − e^(−x) is 0 below x ≈ 1e-16; −expm1(−x) keeps x.
    #[test]
    fn thin_sensor_absorbs_its_optical_depth() {
        let path = DetectorPath {
            sensor: Some(layer(1e-14, 1e-6)),
            ..Default::default()
        };
        let w = path.weight(&Vector3::z(), &Vector3::z_axis()).unwrap();
        assert!(close(w, 1e-20, 4.0 * f64::EPSILON), "{w}");
    }

    #[test]
    fn detector_path_rejects_bad_input() {
        let path = DetectorPath::default();
        let got = path.weight(&Vector3::zeros(), &Vector3::z_axis());
        assert!(matches!(got, Err(Error::DetectorAtSample)), "{got:?}");
        for t in [0.0, -1e-6, f64::NAN, f64::INFINITY] {
            assert!(Layer::new(mu(1.0), t).is_err(), "{t}");
        }
    }
}
