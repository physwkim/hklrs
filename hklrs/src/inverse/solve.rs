//! Deterministic search: Levenberg-Marquardt from the sector images
//! {x, π−x, π+x, −x} (per written axis) of every regular solution admitted,
//! starting from `current`'s images until the first one is.

use std::collections::VecDeque;
use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use super::system::System;
use super::validate::{SolutionSet, same};
use super::wrap;

const MAX_ITERATIONS: usize = 100;
/// Initial damping relative to the largest diagonal element of JᵀJ.
const TAU: f64 = 1e-3;
/// Relative step length at which the iteration has stalled.
const STEP: f64 = 1e-15;
/// Smallest singular value of the Jacobian at a solution that is seeded
/// from: such a solution is isolated and its angles are known to about
/// 1e-16 / REGULAR rad, far inside the dedupe distance. Points of a
/// solution continuum are admitted but not seeded from, or the search
/// would walk the continuum.
const REGULAR: f64 = 1e-6;
/// Solutions seeded from, at most; bounds the search whatever the mode.
const MAX_ROOTS: usize = 64;

struct Search<'a> {
    system: &'a System<'a>,
    set: SolutionSet<'a>,
    tried: Vec<Vec<f64>>,
}

impl<'a> Search<'a> {
    /// Solves from `seed` unless it was tried; returns the written values of
    /// the solution reached when the set admits it and it is regular.
    fn seed(&mut self, seed: Vec<f64>) -> Option<Vec<f64>> {
        if self.tried.iter().any(|t| same(t, &seed)) {
            return None;
        }
        let x = levenberg_marquardt(self.system, &seed);
        self.tried.push(seed);
        let new = self.set.offer(&self.system.values(&x))?;
        let regular = self
            .system
            .eval(new)
            .is_some_and(|e| e.j.singular_values().min() >= REGULAR);
        regular.then(|| self.system.written_values(new))
    }
}

pub(super) fn search<'a>(system: &'a System<'a>) -> SolutionSet<'a> {
    let mut s = Search {
        system,
        set: SolutionSet::new(system),
        tried: Vec::new(),
    };
    // `current`'s images are only a deterministic multi-start, needed until
    // a regular solution's own images take over.
    let start = system.written_values(system.current());
    let mut roots: VecDeque<_> = images(&start)
        .find_map(|seed| s.seed(seed))
        .into_iter()
        .collect();
    let mut seeded = 0;
    while let Some(root) = roots.pop_front() {
        if seeded == MAX_ROOTS {
            break;
        }
        seeded += 1;
        for seed in images(&root) {
            roots.extend(s.seed(seed));
        }
    }
    s.set
}

/// The 4ⁿ sector images of `x`, `x` itself first.
fn images(x: &[f64]) -> impl Iterator<Item = Vec<f64>> + '_ {
    (0..4usize.pow(x.len() as u32)).map(|image| sector(x, image))
}

/// Image number `image` of `x`: base-4 digit j picks the sector of axis j.
fn sector(x: &[f64], image: usize) -> Vec<f64> {
    let mut rest = image;
    x.iter()
        .map(|&v| {
            let s = rest % 4;
            rest /= 4;
            wrap(match s {
                0 => v,
                1 => PI - v,
                2 => PI + v,
                _ => -v,
            })
        })
        .collect()
}

/// Damped Gauss-Newton (Madsen, Nielsen & Tingleff 2004, alg. 3.16) on the
/// written axes from `seed`; returns where it stopped, converged or not.
fn levenberg_marquardt(system: &System, seed: &[f64]) -> Vec<f64> {
    let mut x = DVector::from_column_slice(seed);
    let Some(mut e) = system.eval(&system.values(seed)) else {
        return seed.to_vec();
    };
    let n = x.len();
    let mut cost = e.f.norm_squared();
    let mut a = e.j.tr_mul(&e.j);
    let mut g = e.j.tr_mul(&e.f);
    let mut mu = TAU * a.diagonal().max().max(1.0);
    let mut nu = 2.0;
    for _ in 0..MAX_ITERATIONS {
        if !(cost > 0.0 && mu.is_finite()) {
            break;
        }
        let Some(h) = (&a + DMatrix::identity(n, n) * mu)
            .cholesky()
            .map(|c| c.solve(&-&g))
        else {
            mu *= nu;
            nu *= 2.0;
            continue;
        };
        // A decrease the model predicts below the rounding of `cost` could
        // not be confirmed: x is a stationary point, root or not.
        if h.norm() <= STEP * (x.norm() + STEP) || h.dot(&(&h * mu - &g)) <= f64::EPSILON * cost {
            break;
        }
        let x_new = &x + &h;
        match system.eval(&system.values(x_new.as_slice())) {
            Some(e_new) if e_new.f.norm_squared() < cost => {
                let cost_new = e_new.f.norm_squared();
                let rho = (cost - cost_new) / h.dot(&(&h * mu - &g));
                mu *= (1.0 / 3.0f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                (x, e, cost) = (x_new, e_new, cost_new);
                a = e.j.tr_mul(&e.j);
                g = e.j.tr_mul(&e.f);
            }
            _ => {
                mu *= nu;
                nu *= 2.0;
            }
        }
    }
    x.as_slice().to_vec()
}
