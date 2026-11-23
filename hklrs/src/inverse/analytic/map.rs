//! A geometry as You's 4S+2D: a rotation L of the lab with L·beam = ŷ,
//! each sample rotation on a circle of Z in stack order as R_c(v + σx) of
//! its value x, the circles left over fixed at v, and k̂f turned by NU and
//! DELTA the same way (v = 0) or by two detector axes of its own. It is
//! derived from the axes: every L, v ∈ {0, ±π/2, π} and σ = ±1 under which
//! the rotations agree, the one with the fewest offsets and flipped senses
//! kept, so a geometry that is You's maps onto itself.

use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::{Matrix3, Vector3};

use super::you::{self, DELTA, EPS, NU, PHI};
use crate::{AxisKind, Geometry};

/// A circle's angle: `offset + sign·x` of geometry axis `axis`, or
/// `offset` without one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Circle {
    pub(super) axis: Option<usize>,
    pub(super) sign: f64,
    pub(super) offset: f64,
}

impl Circle {
    const fn fixed(offset: f64) -> Self {
        Self {
            axis: None,
            sign: 1.0,
            offset,
        }
    }

    /// The angle at axis value `x`.
    pub(super) fn angle(&self, x: f64) -> f64 {
        self.offset + self.sign * x
    }

    /// The axis value at `angle`.
    pub(super) fn value(&self, angle: f64) -> f64 {
        self.sign * (angle - self.offset)
    }
}

/// What turns k̂f.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Arm {
    /// NU and DELTA, each at offset 0.
    Circles([Circle; 2]),
    /// The detector stack, through its two rotations not on the sample
    /// (outer first), which no circle carries.
    Own([usize; 2]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Map {
    /// L: the lab into You's.
    pub(super) lab: Matrix3<f64>,
    /// M = P⁻¹·L, P the turn of the sample circles at their offsets: the
    /// sample frame into You's φ frame.
    pub(super) phi: Matrix3<f64>,
    /// MU, ETA, CHI, PHI.
    pub(super) sample: [Circle; 4],
    pub(super) arm: Arm,
    /// You's azimuth atan2(x, z) about the beam of lab y, which hklrs's qaz
    /// and naz start from; `None` with the beam along y.
    pub(super) azimuth: Option<f64>,
}

impl Map {
    /// `None` with a source axis, more sample or detector rotations than
    /// You's, or axes no L, offsets and senses lay on You's circles.
    pub(super) fn new(geometry: &Geometry) -> Option<Self> {
        if !geometry.source_stack().is_empty() {
            return None;
        }
        let beam = geometry.beam_direction().into_inner();
        let sample = rotations(geometry, geometry.sample_stack());
        let up = (Vector3::y() - beam * beam.y).try_normalize(EPS);
        let mut best: Option<((bool, usize, usize), Self)> = None;
        for lab in labs(&beam) {
            let Some((arm, flipped)) = arm(geometry, &lab, &beam) else {
                continue;
            };
            let azimuth = up.map(|r| lab * r).map(|r| r.x.atan2(r.z));
            let mut placed = Vec::new();
            let mut circles = [Circle::fixed(0.0); 4];
            place(
                &sample,
                &lab,
                0,
                &Matrix3::identity(),
                &mut circles,
                &mut placed,
            );
            for (sample, p) in placed {
                let cost = (
                    matches!(arm, Arm::Own(_)),
                    sample.iter().filter(|c| c.offset != 0.0).count(),
                    flipped + sample.iter().filter(|c| c.sign < 0.0).count(),
                );
                if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                    let phi = p.transpose() * lab;
                    let map = Self {
                        lab,
                        phi,
                        sample,
                        arm,
                        azimuth,
                    };
                    best = Some((cost, map));
                }
            }
        }
        best.map(|(_, map)| map)
    }

    /// The circles, each with its index into You's angles.
    pub(super) fn circles(&self) -> impl Iterator<Item = (usize, Circle)> {
        let arm = match self.arm {
            Arm::Circles(arm) => arm.to_vec(),
            Arm::Own(_) => Vec::new(),
        };
        self.sample.into_iter().chain(arm).enumerate()
    }

    /// Whether `axis` is on a circle or is one of the arm's own.
    pub(super) fn carries(&self, axis: usize) -> bool {
        self.circles().any(|(_, c)| c.axis == Some(axis))
            || matches!(self.arm, Arm::Own(own) if own.contains(&axis))
    }
}

/// The rotations of `stack`, outermost first, with their lab axes at zero.
fn rotations(geometry: &Geometry, stack: &[usize]) -> Vec<(usize, Vector3<f64>)> {
    stack
        .iter()
        .filter_map(|&i| match geometry.axes()[i].kind() {
            AxisKind::Rotation { axis } | AxisKind::RotationAbout { axis, .. } => {
                Some((i, axis.into_inner()))
            }
            AxisKind::Translation { .. } | AxisKind::Parameter => None,
        })
        .collect()
}

/// The proper signed permutations that take `beam` to ŷ.
fn labs(beam: &Vector3<f64>) -> impl Iterator<Item = Matrix3<f64>> + '_ {
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    ORDERS
        .into_iter()
        .flat_map(|order| {
            (0..8).map(move |signs| {
                Matrix3::from_fn(|i, j| match (order[i] == j, signs >> i & 1) {
                    (false, _) => 0.0,
                    (true, 0) => 1.0,
                    (true, _) => -1.0,
                })
            })
        })
        .filter(move |l| l.determinant() > 0.0 && (l * beam - Vector3::y()).norm() < EPS)
}

/// Circle `c` turned by `v`, a multiple of π/2, exactly.
fn quarter(c: usize, v: f64) -> Matrix3<f64> {
    you::circle(c, v).map(f64::round)
}

/// Every way the sample rotations `axes` (outermost first) lie on circles
/// `c..=PHI` once the circles before turn by `p` at their offsets, pushed
/// with that turn of all four.
fn place(
    axes: &[(usize, Vector3<f64>)],
    lab: &Matrix3<f64>,
    c: usize,
    p: &Matrix3<f64>,
    circles: &mut [Circle; 4],
    out: &mut Vec<([Circle; 4], Matrix3<f64>)>,
) {
    if c > PHI {
        if axes.is_empty() {
            out.push((*circles, *p));
        }
        return;
    }
    let want = p * you::axis(c);
    for offset in [0.0, FRAC_PI_2, -FRAC_PI_2, PI] {
        let next = p * quarter(c, offset);
        if let Some(&(i, a)) = axes.first()
            && let Some(sign) = sense(&(lab * a), &want)
        {
            circles[c] = Circle {
                axis: Some(i),
                sign,
                offset,
            };
            place(&axes[1..], lab, c + 1, &next, circles, out);
        }
        if axes.len() <= PHI - c {
            circles[c] = Circle::fixed(offset);
            place(axes, lab, c + 1, &next, circles, out);
        }
    }
}

/// The arm under `lab` and how many of its circles turn the other way:
/// NU and DELTA where the detector's rotations are its own and lie on them
/// in order, else its two own rotations. An innermost rotation about the
/// beam leaves k̂f where it is and is no part of the arm.
fn arm(geometry: &Geometry, lab: &Matrix3<f64>, beam: &Vector3<f64>) -> Option<(Arm, usize)> {
    let mut turning = rotations(geometry, geometry.detector_stack());
    while turning
        .last()
        .is_some_and(|(_, a)| a.cross(beam).norm() < EPS)
    {
        turning.pop();
    }
    let sample = geometry.sample_stack();
    let own: Vec<usize> = turning
        .iter()
        .map(|&(i, _)| i)
        .filter(|i| !sample.contains(i))
        .collect();
    let slots: &[&[usize]] = match turning.len() {
        0 => &[&[]],
        1 => &[&[NU], &[DELTA]],
        2 => &[&[NU, DELTA]],
        _ => &[],
    };
    if own.len() == turning.len() {
        for slots in slots {
            let circles = turning.iter().zip(*slots).try_fold(
                [Circle::fixed(0.0); 2],
                |mut circles, (&(i, a), &c)| {
                    circles[c - NU] = Circle {
                        axis: Some(i),
                        sign: sense(&(lab * a), &you::axis(c))?,
                        offset: 0.0,
                    };
                    Some(circles)
                },
            );
            if let Some(circles) = circles {
                let flipped = circles.iter().filter(|c| c.sign < 0.0).count();
                return Some((Arm::Circles(circles), flipped));
            }
        }
    }
    Some((Arm::Own(own.try_into().ok()?), 0))
}

/// ±1 where `a` is ±`b`.
fn sense(a: &Vector3<f64>, b: &Vector3<f64>) -> Option<f64> {
    [1.0, -1.0].into_iter().find(|s| (a - b * *s).norm() < EPS)
}
