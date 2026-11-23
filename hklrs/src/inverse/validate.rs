//! The only way into a solution set. Generators (the solver from each seed,
//! the kappa twin and slit roots of its solution, and any later Ewald
//! construction) only offer candidates; a candidate is a solution when every
//! axis the mode does not write is exactly `current` and all of the mode's
//! equations, the slit equation included, hold at its values, written axes
//! wrapped to their cut points. A regular solution is admitted; a singular
//! one makes the set degenerate. Regularity is that of the solved axes: the
//! slits move no other equation, and where they are free the generator holds
//! them. Axis limits filter the admitted set once the search is over, so a
//! solution outside them still seeds it.

use std::cmp::Ordering;

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

    /// The admitted solutions whose written axes are all within their
    /// limits, sorted; when there is none, the others, sorted, as
    /// `NoSolutionInLimits`.
    pub fn into_sorted(self) -> Result<Found, InverseError> {
        if self.degenerate {
            return Err(InverseError::Degenerate);
        }
        let system = self.system;
        let (inside, outside): (Vec<_>, Vec<_>) = self.admitted.into_iter().partition(|s| {
            s.iter()
                .enumerate()
                .all(|(i, &v)| !system.is_written(i) || system.axis(i).within_limits(v))
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

/// `set` sorted by distance to `current`, and where its first went.
fn sorted(current: &[f64], set: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Option<usize>) {
    let distance = |s: &[f64]| -> f64 {
        s.iter()
            .zip(current)
            .map(|(a, b)| wrap(a - b).powi(2))
            .sum()
    };
    let mut indexed: Vec<(usize, Vec<f64>)> = set.into_iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        distance(a).total_cmp(&distance(b)).then_with(|| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or(Ordering::Equal)
        })
    });
    let first = indexed.iter().position(|&(i, _)| i == 0);
    (indexed.into_iter().map(|(_, s)| s).collect(), first)
}
