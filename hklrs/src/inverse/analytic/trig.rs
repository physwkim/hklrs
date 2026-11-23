//! Real roots of a trigonometric polynomial of low degree.

use std::f64::consts::TAU;

use nalgebra::{Complex, DMatrix};

use super::you::EPS;

/// Samples of one period: more than twice the degree, so the DFT gives the
/// coefficients exactly.
const SAMPLES: usize = 16;

/// Eigenvalue arguments closer than this are one multiple root.
const CLUSTER: f64 = 1e-6;

/// Every x in (−π, π] with g(x) = 0, for g a trigonometric polynomial of
/// degree below `SAMPLES / 2` with terms of order one; `None` when g
/// vanishes identically.
pub(super) fn roots(g: impl Fn(f64) -> f64) -> Option<Vec<f64>> {
    let n = SAMPLES as f64;
    let values: Vec<f64> = (0..SAMPLES).map(|j| g(TAU * j as f64 / n)).collect();
    // g(x) = Σ c_k e^{ikx} over |k| < SAMPLES / 2, c_−k = conj(c_k).
    let c: Vec<Complex<f64>> = (0..SAMPLES / 2)
        .map(|k| {
            let turn = |j: usize| Complex::from_polar(1.0, -TAU * (k * j) as f64 / n);
            values
                .iter()
                .enumerate()
                .map(|(j, &v)| turn(j) * v)
                .sum::<Complex<f64>>()
                / n
        })
        .collect();
    let size = c.iter().map(|z| z.norm()).fold(0.0, f64::max);
    if size < EPS {
        return None;
    }
    let m = (1..c.len())
        .rev()
        .find(|&k| c[k].norm() > size * EPS)
        .unwrap_or(0);
    if m == 0 {
        return Some(Vec::new());
    }
    // e^{imx}·g(x) is a polynomial of degree 2m in z = e^{ix}; its roots on
    // the unit circle are the real roots of g.
    let coef = |i: usize| if i >= m { c[i - m] } else { c[m - i].conj() };
    let mut companion = DMatrix::<Complex<f64>>::zeros(2 * m, 2 * m);
    for i in 0..2 * m {
        if i > 0 {
            companion[(i, i - 1)] = Complex::new(1.0, 0.0);
        }
        companion[(i, 2 * m - 1)] = -coef(i) / coef(2 * m);
    }
    let t = companion.schur().unpack().1;
    // The `order`-th derivative of g from its coefficients.
    let deriv = |x: f64, order: u32| -> f64 {
        (1..=m)
            .map(|k| {
                let turn = Complex::new(0.0, k as f64).powu(order);
                2.0 * (turn * c[k] * Complex::from_polar(1.0, k as f64 * x)).re
            })
            .sum::<f64>()
            + if order == 0 { c[0].re } else { 0.0 }
    };
    // Newton on the `order`-th derivative while it shrinks.
    let polish = |mut x: f64, order: u32| {
        for _ in 0..8 {
            let v = deriv(x, order);
            let next = x - v / deriv(x, order + 1);
            // NaN compares false and stops too.
            if deriv(next, order).abs() < v.abs() {
                x = next;
            } else {
                break;
            }
        }
        x
    };
    // A double root splits into two eigenvalues about √ε apart, off the
    // circle or along it; such a cluster is one root, where g is stationary.
    let mut args: Vec<f64> = (0..2 * m)
        .map(|i| t[(i, i)])
        .filter(|z| (z.norm() - 1.0).abs() < 1e-3)
        .map(|z| z.arg())
        .collect();
    args.sort_by(f64::total_cmp);
    let mut clusters: Vec<Vec<f64>> = Vec::new();
    for x in args {
        match clusters.last_mut() {
            Some(last) if x - last[last.len() - 1] < CLUSTER => last.push(x),
            _ => clusters.push(vec![x]),
        }
    }
    if clusters.len() > 1
        && clusters[0][0] + TAU - clusters[clusters.len() - 1].last().unwrap() < CLUSTER
    {
        let last = clusters.pop().unwrap();
        clusters[0].extend(last.iter().map(|x| x - TAU));
    }
    Some(
        clusters
            .into_iter()
            .map(|xs| {
                let mean = xs.iter().sum::<f64>() / xs.len() as f64;
                polish(mean, if xs.len() == 1 { 0 } else { 1 })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_simple_and_double() {
        let want = [-2.9, -1.3, 0.2, 0.25, 1.7, 3.1];
        // Six half-angle sines make a trigonometric polynomial of degree 3.
        let g = |x: f64| want.iter().map(|r| ((x - r) / 2.0).sin()).product::<f64>();
        let mut found = roots(g).unwrap();
        found.sort_by(f64::total_cmp);
        assert_eq!(found.len(), want.len());
        for (f, w) in found.iter().zip(want) {
            assert!((f - w).abs() < 1e-12, "{f} vs {w}");
        }
        // A double root at 0.4 between simple ones at −1 and 2.
        let g =
            |x: f64| ((x + 1.0) / 2.0).sin() * ((x - 2.0) / 2.0).sin() * (1.0 - (x - 0.4).cos());
        let mut found = roots(g).unwrap();
        found.sort_by(f64::total_cmp);
        assert_eq!(found.len(), 3, "{found:?}");
        for (f, w) in found.iter().zip([-1.0, 0.4, 2.0]) {
            assert!((f - w).abs() < 1e-10, "{f} vs {w}");
        }
        assert_eq!(roots(|_| 0.0), None);
        assert_eq!(roots(|x| 2.0 + x.cos()), Some(Vec::new()));
    }
}
