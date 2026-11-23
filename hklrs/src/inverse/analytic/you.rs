//! You (1999) frame: lab x, y, z with the beam along +y, Z = MU·ETA·CHI·PHI
//! takes the φ frame to the lab, D = NU·DELTA turns the detector arm, and
//! vectors are in units of k, so Q = D·ŷ − ŷ.

use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

/// Below this a sine, cosine or projected length counts as zero.
pub(super) const EPS: f64 = 1e-10;

/// How many values a free angle is sampled at when an equation does not
/// depend on it; a sample that passes the final check proves a continuous
/// family. They are spread over the circle, so a family over an interval of
/// the angle is hit, and none is a round number of degrees, so a constraint
/// value does not coincide with one.
const PROBES: usize = 12;

fn probes() -> impl Iterator<Item = f64> {
    (0..PROBES)
        .map(|i| (i as f64 + 0.37) * std::f64::consts::TAU / PROBES as f64 - std::f64::consts::PI)
}

/// Circle indices into [`Angles`]: sample circles in Z order, then NU, DELTA.
pub(super) const MU: usize = 0;
pub(super) const ETA: usize = 1;
pub(super) const CHI: usize = 2;
pub(super) const PHI: usize = 3;
pub(super) const NU: usize = 4;
pub(super) const DELTA: usize = 5;

/// Right-handed rotation axis of each circle.
pub(super) fn axis(circle: usize) -> Vector3<f64> {
    match circle {
        MU | NU => Vector3::x(),
        CHI => Vector3::y(),
        _ => -Vector3::z(),
    }
}

pub(super) fn rot(axis: &Vector3<f64>, angle: f64) -> Matrix3<f64> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*axis), angle).into_inner()
}

pub(super) fn circle(c: usize, angle: f64) -> Matrix3<f64> {
    rot(&axis(c), angle)
}

/// Tag of a candidate built from a probed free angle: what is free.
pub(super) type Free = Option<&'static str>;

/// One E6C position in You's circles, radians, indexed by [`MU`]..[`DELTA`].
pub(super) type Angles = [f64; 6];

/// Z of the sample circles `s` (MU, ETA, CHI, PHI).
pub(super) fn z(s: &[f64]) -> Matrix3<f64> {
    (MU..=PHI).fold(Matrix3::identity(), |acc, c| acc * circle(c, s[c]))
}

pub(super) fn kout(a: &Angles) -> Vector3<f64> {
    circle(NU, a[NU]) * circle(DELTA, a[DELTA]) * Vector3::y()
}

/// The values one angle can take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Roots {
    Two([f64; 2]),
    One(f64),
    /// The equation holds for every value.
    Free,
    Empty,
}

impl Roots {
    /// Isolated roots untagged, a free angle as [`PROBES`] tagged `what`.
    pub(super) fn tagged(self, what: &'static str) -> Vec<(f64, Free)> {
        match self {
            Roots::Two(r) => r.iter().map(|&x| (x, None)).collect(),
            Roots::One(x) => vec![(x, None)],
            Roots::Free => probes().map(|x| (x, Some(what))).collect(),
            Roots::Empty => Vec::new(),
        }
    }
}

/// A cos x = c.
pub(super) fn cos_eq(a: f64, c: f64) -> Roots {
    if a.abs() < EPS {
        return if c.abs() < EPS {
            Roots::Free
        } else {
            Roots::Empty
        };
    }
    let r = c / a;
    if r.abs() > 1.0 + 1e-9 {
        return Roots::Empty;
    }
    let x = r.clamp(-1.0, 1.0).acos();
    Roots::Two([x, -x])
}

/// x with α + β cos x + γ sin x = 0.
pub(super) fn harmonic([alpha, beta, gamma]: [f64; 3]) -> Roots {
    let base = gamma.atan2(beta);
    match cos_eq(beta.hypot(gamma), -alpha) {
        Roots::Two([x, y]) => Roots::Two([base + x, base + y]),
        other => other,
    }
}

/// α, β, γ of (R_axis(x)·p)·w − target = α + β cos x + γ sin x.
pub(super) fn component_terms(
    axis: &Vector3<f64>,
    p: &Vector3<f64>,
    w: &Vector3<f64>,
    target: f64,
) -> [f64; 3] {
    let along = axis * axis.dot(p);
    [
        along.dot(w) - target,
        (p - along).dot(w),
        axis.cross(p).dot(w),
    ]
}

/// x with (R_axis(x)·p)·w = target.
pub(super) fn solve_component(
    axis: &Vector3<f64>,
    p: &Vector3<f64>,
    w: &Vector3<f64>,
    target: f64,
) -> Roots {
    harmonic(component_terms(axis, p, w, target))
}

/// x with R_axis(x)·from along `to` in the plane normal to `axis`; free when
/// both lie on the axis.
pub(super) fn spin(axis: &Vector3<f64>, from: &Vector3<f64>, to: &Vector3<f64>) -> Roots {
    let f = from - axis * axis.dot(from);
    let t = to - axis * axis.dot(to);
    match (f.norm() < EPS, t.norm() < EPS) {
        (true, true) => Roots::Free,
        (false, false) => Roots::One(axis.dot(&f.cross(&t)).atan2(f.dot(&t))),
        _ => Roots::Empty,
    }
}

/// (outer, inner) with R_x(outer)·R_−z(inner)·ŷ = v, i.e.
/// v = (sin inner, cos inner cos outer, cos inner sin outer): NU, DELTA for
/// k_out and MU, ETA for the χ axis.
pub(super) fn arms(v: &Vector3<f64>, what: &'static str) -> Vec<(f64, f64, Free)> {
    let c = v.y.hypot(v.z);
    let inner = v.x.atan2(c);
    if c < EPS {
        return Roots::Free
            .tagged(what)
            .into_iter()
            .map(|(outer, free)| (outer, inner, free))
            .collect();
    }
    let outer = v.z.atan2(v.y);
    vec![
        (outer, inner, None),
        (
            outer + std::f64::consts::PI,
            std::f64::consts::PI - inner,
            None,
        ),
    ]
}

/// A unit vector normal to unit `v`.
pub(super) fn normal(v: &Vector3<f64>) -> Vector3<f64> {
    let i = v.iamin();
    let mut e = Vector3::zeros();
    e[i] = 1.0;
    v.cross(&e).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_component_hits_target() {
        let axis = Vector3::new(0.3, -0.4, 0.5).normalize();
        let (p, w) = (Vector3::new(0.2, 0.9, -0.1), Vector3::new(-0.7, 0.1, 0.6));
        let Roots::Two(xs) = solve_component(&axis, &p, &w, 0.25) else {
            panic!("no roots");
        };
        for x in xs {
            assert!(((rot(&axis, x) * p).dot(&w) - 0.25).abs() < 1e-14);
        }
    }

    #[test]
    fn arms_rebuild_their_vector() {
        let v = Vector3::new(0.3, -0.5, 0.2).normalize();
        for (outer, inner, free) in arms(&v, "") {
            assert_eq!(free, None);
            let a = [0.0, 0.0, 0.0, 0.0, outer, inner];
            assert!((kout(&a) - v).norm() < 1e-15);
        }
    }
}
