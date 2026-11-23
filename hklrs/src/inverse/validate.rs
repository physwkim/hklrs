//! The only way into a solution set. Generators (the solver from each seed,
//! and any later Ewald or kappa-twin construction) only offer candidates; a
//! candidate is admitted when every axis the mode does not write is exactly
//! `current` and all of the mode's equations hold at its wrapped values.

use std::cmp::Ordering;

use super::system::System;
use super::wrap;

/// Largest residual admitted (dimensionless: Q in units of k, angles in rad).
const ACCEPT: f64 = 1e-12;
/// Axis values closer than this (rad, on every axis) are one solution.
const SAME: f64 = 1e-9;

pub(super) fn same(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| wrap(x - y).abs() <= SAME)
}

pub(super) struct SolutionSet<'a> {
    system: &'a System<'a>,
    admitted: Vec<Vec<f64>>,
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
        }
    }

    /// Admits `candidate` (every axis, geometry order) with its written
    /// axes wrapped to (−π, π]; returns the admitted values when it is
    /// valid and not already in the set.
    pub fn offer(&mut self, candidate: &[f64]) -> Option<&[f64]> {
        let current = self.system.current();
        if candidate.len() != current.len() {
            return None;
        }
        let mut values = Vec::with_capacity(current.len());
        for (i, (&c, &held)) in candidate.iter().zip(current).enumerate() {
            if self.system.is_written(i) {
                values.push(wrap(c));
            } else if c.to_bits() == held.to_bits() {
                values.push(c);
            } else {
                return None;
            }
        }
        let e = self.system.eval(&values)?;
        if !e.f.iter().all(|r| r.abs() <= ACCEPT) || self.admitted.iter().any(|s| same(s, &values))
        {
            return None;
        }
        self.admitted.push(values);
        self.admitted.last().map(Vec::as_slice)
    }

    /// `None` when nothing was admitted.
    pub fn into_sorted(self) -> Option<Found> {
        let current = self.system.current();
        let distance = |s: &[f64]| -> f64 {
            s.iter()
                .zip(current)
                .map(|(a, b)| wrap(a - b).powi(2))
                .sum()
        };
        let mut order: Vec<usize> = (0..self.admitted.len()).collect();
        order.sort_by(|&a, &b| {
            let (sa, sb) = (&self.admitted[a], &self.admitted[b]);
            distance(sa).total_cmp(&distance(sb)).then_with(|| {
                sa.iter()
                    .zip(sb)
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(Ordering::Equal)
            })
        });
        let first = order.iter().position(|&i| i == 0)?;
        Some(Found {
            solutions: order.iter().map(|&i| self.admitted[i].clone()).collect(),
            first,
        })
    }
}
