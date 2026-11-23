//! Deterministic search: Levenberg-Marquardt from the sector images
//! {x, π−x, π+x, −x} (per written axis) of every solution admitted,
//! starting from the images of `current` and then of the sector midpoints,
//! every axis at π/4, until the first one is, and stopping at the first
//! singular one. Only a [`Start`], where every equation is defined, is
//! solved from. A kappa mode offers the kappa twin of every solution
//! reached as well; the validator rejects it when it moves a held axis.

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_4, PI};

use nalgebra::{DMatrix, DVector};

use super::system::{Eval, System};
use super::validate::{SolutionSet, same};
use super::wrap;

const MAX_ITERATIONS: usize = 100;
/// Initial damping relative to the largest diagonal element of JᵀJ.
const TAU: f64 = 1e-3;
/// Relative step length at which the iteration has stalled.
const STEP: f64 = 1e-15;
/// Solutions seeded from, at most; bounds the search whatever the mode.
const MAX_ROOTS: usize = 64;

/// Written values where every equation of the system is defined, with the
/// residual and Jacobian there: the only point a solve starts from.
struct Start {
    x: Vec<f64>,
    eval: Eval,
}

impl Start {
    fn at(system: &System, x: Vec<f64>) -> Option<Self> {
        let eval = system.eval(&system.values(&x))?;
        Some(Self { x, eval })
    }
}

struct Search<'a> {
    system: &'a System<'a>,
    set: SolutionSet<'a>,
    tried: Vec<Vec<f64>>,
}

impl<'a> Search<'a> {
    /// Solves from `seed` unless it was tried, the set is degenerate or an
    /// equation is undefined there;
    /// returns the written values of the solution reached and of its kappa
    /// twin, those the set admits at one of their slit roots.
    fn seed(&mut self, seed: Vec<f64>) -> Vec<Vec<f64>> {
        if self.set.is_degenerate() || self.tried.iter().any(|t| same(t, &seed)) {
            return Vec::new();
        }
        self.tried.push(seed.clone());
        let Some(start) = Start::at(self.system, seed) else {
            return Vec::new();
        };
        let x = levenberg_marquardt(self.system, start);
        let reached = self.system.values(&x);
        let twin = self.system.kappa().map(|k| k.twin(&reached));
        [Some(reached), twin]
            .into_iter()
            .flatten()
            .filter_map(|c| {
                // Every slit root is offered; they share the written values.
                self.system
                    .slit_images(c)
                    .into_iter()
                    .filter_map(|s| {
                        self.set
                            .offer(&s)
                            .map(|new| self.system.written_values(new))
                    })
                    .reduce(|first, _| first)
            })
            .collect()
    }
}

pub(super) fn search<'a>(system: &'a System<'a>) -> SolutionSet<'a> {
    let mut s = Search {
        system,
        set: SolutionSet::new(system),
        tried: Vec::new(),
    };
    // The starts' images are only a deterministic multi-start, needed until
    // a solution's own images take over. The images of a `current` on a set
    // where an equation is undefined (k̂f = ±k̂i, a normal along k̂i) all stay
    // on it; the midpoints' images put no axis at a multiple of π/2.
    let start = system.written_values(system.current());
    let midpoints = vec![FRAC_PI_4; start.len()];
    let mut roots: VecDeque<_> = [start, midpoints]
        .into_iter()
        .flat_map(images)
        .map(|seed| s.seed(seed))
        .find(|found| !found.is_empty())
        .unwrap_or_default()
        .into();
    let mut seeded = 0;
    while let Some(root) = roots.pop_front() {
        if seeded == MAX_ROOTS {
            break;
        }
        seeded += 1;
        for seed in images(root) {
            roots.extend(s.seed(seed));
        }
    }
    s.set
}

/// The 4ⁿ sector images of `x`, `x` itself first.
fn images(x: Vec<f64>) -> impl Iterator<Item = Vec<f64>> {
    (0..4usize.pow(x.len() as u32)).map(move |image| sector(&x, image))
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
/// written axes from `start`; returns where it stopped, converged or not.
fn levenberg_marquardt(system: &System, start: Start) -> Vec<f64> {
    let mut x = DVector::from_vec(start.x);
    let mut e = start.eval;
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

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3};

    use super::super::system::Goal;
    use super::*;
    use crate::inverse::{Equation, Eulerian, Mode};
    use crate::presets::preset;

    /// The seed that reaches the first solution also yields its kappa twin,
    /// whether or not the sector images would reach it.
    #[test]
    fn a_seed_offers_the_kappa_twin() {
        let geo = preset("K4CV").unwrap();
        let held = vec![Equation::Kappa(Eulerian::Chi(-0.6))];
        let mode = Mode::new_kappa(&geo, "m", [0, 1, 2], &[0, 1, 2, 3], held).unwrap();
        let current = [0.2, -0.9, 0.5, 0.7];
        let q = Vector3::new(0.3, -0.2, 0.4);
        let system = System::new(&mode, Matrix3::identity(), Goal::Q(q), 1.0, &current);
        let mut s = Search {
            system: &system,
            set: SolutionSet::new(&system),
            tried: Vec::new(),
        };
        let roots = images(current.to_vec())
            .map(|seed| s.seed(seed))
            .find(|found| !found.is_empty())
            .unwrap();
        let twin = mode.kappa().unwrap().twin(&system.values(&roots[0]));
        assert_eq!(roots.len(), 2);
        assert!(same(&roots[1], &system.written_values(&twin)));
    }
}
