//! The only way into a solution set. Generators (the solver from each seed,
//! the kappa twin and slit roots of its solution, and any later Ewald
//! construction) only offer candidates; a candidate is a solution when every
//! axis the mode does not write is exactly `current` and all of the mode's
//! equations, the slit equation included, hold at its values, written axes
//! wrapped to their cut points. A regular solution is admitted; a singular
//! one makes the set degenerate. Regularity is that of the solved axes: the
//! slits move no other equation, and where they are free the generator holds
//! them. Axis limits filter the admitted set once the search is over, so a
//! solution outside them still seeds it. They bind every axis, a held one at
//! its unwrapped value: a held motor is never asked to turn a full circle.

use super::system::System;
use super::{InverseError, wrap};

/// Largest residual admitted (dimensionless: Q in units of k, angles in rad).
pub(super) const ACCEPT: f64 = 1e-12;
/// Axis values closer than this (rad, on every axis) are one solution.
const SAME: f64 = 1e-9;
/// Smallest singular value of the Jacobian at a regular solution, whose
/// angles are then known to about 1e-16 / REGULAR rad, far inside SAME.
/// Below it the solution is not isolated at that precision: a point of a
/// continuum, which no finite list can stand for, or a multiple root.
const REGULAR: f64 = 1e-6;

pub(super) fn same(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| wrap(x - y).abs() <= SAME)
}

pub(super) struct SolutionSet<'a> {
    system: &'a System<'a>,
    admitted: Vec<Vec<f64>>,
    /// A singular solution was offered.
    degenerate: bool,
}

pub(super) struct Found {
    /// Sorted by distance to `current`.
    pub solutions: Vec<Vec<f64>>,
    /// Index in `solutions` of the first one admitted.
    pub first: usize,
}

impl<'a> SolutionSet<'a> {
    pub fn new(system: &'a System<'a>) -> Self {
        Self {
            system,
            admitted: Vec::new(),
            degenerate: false,
        }
    }

    pub fn is_degenerate(&self) -> bool {
        self.degenerate
    }

    /// Admits `candidate` (every axis, geometry order) with its written
    /// axes wrapped to their cut points; returns the admitted values when it
    /// is a regular solution not already in the set.
    pub fn offer(&mut self, candidate: &[f64]) -> Option<&[f64]> {
        let current = self.system.current();
        if candidate.len() != current.len() {
            return None;
        }
        let mut values = Vec::with_capacity(current.len());
        for (i, (&c, &held)) in candidate.iter().zip(current).enumerate() {
            if self.system.is_written(i) {
                values.push(self.system.axis(i).wrap(c));
            } else if c.to_bits() == held.to_bits() {
                values.push(c);
            } else {
                return None;
            }
        }
        let e = self.system.eval(&values)?;
        let slit = self.system.slit_miss(&values);
        if !e.f.iter().chain(&slit).all(|r| r.abs() <= ACCEPT) {
            return None;
        }
        if e.j.singular_values().min() < REGULAR {
            self.degenerate = true;
            return None;
        }
        if self.admitted.iter().any(|s| same(s, &values)) {
            return None;
        }
        self.admitted.push(values);
        self.admitted.last().map(Vec::as_slice)
    }

    /// The admitted solutions whose axes are all within their limits,
    /// sorted; when there is none, the others, sorted, as
    /// `NoSolutionInLimits`.
    pub fn into_sorted(self) -> Result<Found, InverseError> {
        if self.degenerate {
            return Err(InverseError::Degenerate);
        }
        let system = self.system;
        let (inside, outside): (Vec<_>, Vec<_>) = self.admitted.into_iter().partition(|s| {
            s.iter()
                .enumerate()
                .all(|(i, &v)| system.axis(i).within_limits(v))
        });
        match sorted(system.current(), inside) {
            (solutions, Some(first)) => Ok(Found { solutions, first }),
            _ if outside.is_empty() => Err(InverseError::NoSolution),
            _ => Err(InverseError::NoSolutionInLimits(
                sorted(system.current(), outside).0,
            )),
        }
    }
}

/// `set` (in the order offered) sorted by distance to `current`, and where
/// its first went. Distances within SAME are one: an exact tie comes out
/// of a solver apart by noise, and the values, near a cut, are no better,
/// so the order offered keeps it.
fn sorted(current: &[f64], set: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Option<usize>) {
    let distance = |s: &[f64]| -> f64 {
        s.iter()
            .zip(current)
            .map(|(a, b)| wrap(a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut indexed: Vec<(f64, usize, Vec<f64>)> = set
        .into_iter()
        .enumerate()
        .map(|(i, s)| (distance(&s), i, s))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));
    for tied in indexed.chunk_by_mut(|a, b| b.0 - a.0 <= SAME) {
        tied.sort_by_key(|&(_, i, _)| i);
    }
    let first = indexed.iter().position(|&(_, i, _)| i == 0);
    (indexed.into_iter().map(|(_, _, s)| s).collect(), first)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two solutions a tie apart by noise either way keep the order offered;
    /// apart by more than SAME, the nearer comes first.
    #[test]
    fn sorted_ties_keep_the_order_offered() {
        let current = [0.0, 0.0];
        for noise in [1e-13, -1e-13] {
            let (a, b) = (vec![1.0 + noise, 0.0], vec![-1.0, 0.0]);
            let set = vec![a.clone(), b.clone()];
            assert_eq!(sorted(&current, set), (vec![a.clone(), b.clone()], Some(0)));
            let set = vec![b.clone(), a.clone()];
            assert_eq!(sorted(&current, set), (vec![b, a], Some(0)));
        }
        let (a, b) = (vec![1.0 + 1e-8, 0.0], vec![-1.0, 0.0]);
        let set = vec![a.clone(), b.clone()];
        assert_eq!(sorted(&current, set), (vec![b, a], Some(1)));
    }
}
