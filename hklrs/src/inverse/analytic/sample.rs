//! Z = MU·ETA·CHI·PHI solved for its free circles, whichever are held.

use nalgebra::{Matrix3, Vector3};

use super::trig;
use super::you::{
    Free, Roots, axis, circle, component_terms, harmonic, normal, solve_component, spin,
};

pub(super) const NAMES: [&str; 6] = ["mu", "eta", "chi", "phi", "nu", "delta"];

/// Sample circles (MU, ETA, CHI, PHI); `None` where free.
pub(super) type Known = [Option<f64>; 4];

/// The free circles of `known` in Z order and the products of the held
/// rotations before, between and after them (one more than the free).
fn split(known: &Known) -> (Vec<usize>, Vec<Matrix3<f64>>) {
    let mut free = Vec::new();
    let mut seg = vec![Matrix3::identity()];
    for (c, held) in known.iter().enumerate() {
        match *held {
            Some(v) => {
                let last = seg.len() - 1;
                seg[last] *= circle(c, v);
            }
            None => {
                free.push(c);
                seg.push(Matrix3::identity());
            }
        }
    }
    (free, seg)
}

fn fill(known: &Known, free: &[usize], values: &[f64]) -> [f64; 4] {
    let mut s = known.map(|v| v.unwrap_or(0.0));
    for (&c, &v) in free.iter().zip(values) {
        s[c] = v;
    }
    s
}

/// (x, y) with left·R_a(x)·mid·R_b(y)·right·p = q for circles a, b.
fn two_free(
    left: &Matrix3<f64>,
    a: usize,
    mid: &Matrix3<f64>,
    b: usize,
    right: &Matrix3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
) -> Vec<(f64, f64, Free)> {
    let (q, p, ax) = (left.transpose() * q, right * p, axis(a));
    let mut out = Vec::new();
    let ys = solve_component(&axis(b), &p, &(mid.transpose() * ax), ax.dot(&q));
    for (y, fy) in ys.tagged(NAMES[b]) {
        for (x, fx) in spin(&ax, &(mid * circle(b, y) * p), &q).tagged(NAMES[a]) {
            out.push((x, y, fy.or(fx)));
        }
    }
    out
}

/// Every pair of free circles that turns `p` onto `q` with two held.
pub(super) fn two_held(known: &Known, p: &Vector3<f64>, q: &Vector3<f64>) -> Vec<([f64; 4], Free)> {
    let (free, seg) = split(known);
    let [a, b] = free[..] else {
        unreachable!("two held circles leave two free")
    };
    two_free(&seg[0], a, &seg[1], b, &seg[2], p, q)
        .into_iter()
        .map(|(x, y, f)| (fill(known, &free, &[x, y]), f))
        .collect()
}

/// Every pair of free circles with (Z·p)·w = h for both `(p, w, h)`. Each
/// equation is α + β cos y + γ sin y in the second free circle y, with α, β,
/// γ of degree 1 in the first, x; y exists where (α₁γ₂ − α₂γ₁)² +
/// (β₁α₂ − β₂α₁)² = (β₁γ₂ − β₂γ₁)², of degree 4 in x.
pub(super) fn two_held_heights(
    known: &Known,
    eqs: &[(Vector3<f64>, Vector3<f64>, f64); 2],
) -> Vec<([f64; 4], Free)> {
    let (free, seg) = split(known);
    let [a, b] = free[..] else {
        unreachable!("two held circles leave two free")
    };
    let terms = |x: f64| {
        let m = (seg[0] * circle(a, x) * seg[1]).transpose();
        eqs.map(|(p, w, h)| component_terms(&axis(b), &(seg[2] * p), &(m * w), h))
    };
    let g = |x: f64| {
        let [[a1, b1, c1], [a2, b2, c2]] = terms(x);
        (a1 * c2 - a2 * c1).powi(2) + (b1 * a2 - b2 * a1).powi(2) - (b1 * c2 - b2 * c1).powi(2)
    };
    let xs = match trig::roots(g) {
        Some(xs) => xs.into_iter().map(|x| (x, None)).collect(),
        None => Roots::Free.tagged(NAMES[a]),
    };
    let mut out = Vec::new();
    for (x, fx) in xs {
        // y from the equation that depends on it more; the final check holds
        // the candidate to the other.
        let [t1, t2] = terms(x);
        let ys = if t1[1].hypot(t1[2]) >= t2[1].hypot(t2[2]) {
            t1
        } else {
            t2
        };
        for (y, fy) in harmonic(ys).tagged(NAMES[b]) {
            out.push((fill(known, &free, &[x, y]), fx.or(fy)));
        }
    }
    out
}

/// Every value of the one free circle with (Z·p)·y = `height`.
pub(super) fn three_held(
    known: &Known,
    p: &Vector3<f64>,
    y: &Vector3<f64>,
    height: f64,
) -> Vec<([f64; 4], Free)> {
    let (free, seg) = split(known);
    let [a] = free[..] else {
        unreachable!("three held circles leave one free")
    };
    solve_component(&axis(a), &(seg[1] * p), &(seg[0].transpose() * y), height)
        .tagged(NAMES[a])
        .into_iter()
        .map(|(x, f)| (fill(known, &free, &[x]), f))
        .collect()
}

/// Every decomposition of the lab rotation `z` with one circle held.
pub(super) fn one_held(known: &Known, z: &Matrix3<f64>) -> Vec<([f64; 4], Free)> {
    let (free, seg) = split(known);
    let [a, b, c] = free[..] else {
        unreachable!("one held circle leaves three free")
    };
    let w = seg[0].transpose() * z * seg[3].transpose();
    let (ac, u) = (axis(c), normal(&axis(c)));
    let mut out = Vec::new();
    for (x, y, fxy) in two_free(&Matrix3::identity(), a, &seg[1], b, &seg[2], &ac, &(w * ac)) {
        let m = circle(a, x) * seg[1] * circle(b, y) * seg[2];
        for (zv, fz) in spin(&ac, &u, &(m.transpose() * w * u)).tagged(NAMES[c]) {
            out.push((fill(known, &free, &[x, y, zv]), fxy.or(fz)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::you::{MU, PHI, z};
    use super::*;

    #[test]
    fn one_held_recovers_every_held_circle() {
        let s = [0.4, -1.1, 2.3, 0.7];
        let zs = z(&s);
        for held in MU..=PHI {
            let mut known = [None; 4];
            known[held] = Some(s[held]);
            let found = one_held(&known, &zs);
            assert!(
                found
                    .iter()
                    .all(|(f, free)| free.is_none() && (z(f) - zs).norm() < 1e-13)
            );
            assert!(found.iter().any(|(f, _)| {
                (0..4).all(|i| (f[i] - s[i]).sin().abs() < 1e-12 && (f[i] - s[i]).cos() > 0.0)
            }));
        }
    }
}
