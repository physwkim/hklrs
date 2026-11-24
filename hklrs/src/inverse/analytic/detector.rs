//! A detector goal whose equations each hold k̂f to a plane k̂f·a = α of the
//! lab, its rotations at `current`: one written detector axis turns k̂f onto
//! the plane; two take it to where two planes meet the unit sphere, the
//! inner axis by k̂f's component along the outer one and the outer by
//! spinning onto it. The validator keeps the side of a plane each equation
//! means (the sign of a signed q or qpar, the half-plane of qaz).

use nalgebra::{UnitQuaternion, Vector3};

use super::super::Equation;
use super::super::mode::rotation;
use super::super::system::System;
use super::super::validate::SolutionSet;
use super::you::{EPS, solve_component, spin};
use crate::Geometry;

/// (a, α): k̂f·a = α.
type Plane = (Vector3<f64>, f64);

/// The set of a detector goal; `None` unless the mode writes one or two
/// detector axes and each equation is a plane.
pub(super) fn search<'a>(system: &'a System<'a>) -> Option<SolutionSet<'a>> {
    let geometry = system.geometry();
    let current = system.current();
    let stack = geometry.detector_stack();
    let mut at = system
        .written()
        .iter()
        .map(|w| stack.iter().position(|s| s == w))
        .collect::<Option<Vec<_>>>()?;
    at.sort_unstable();
    let poses = geometry.poses(current).ok()?;
    let beam = geometry.beam_direction().into_inner();
    let ki = poses.source * beam;
    let planes = system
        .extra()
        .iter()
        .map(|e| plane(system, e, &ki, &poses.sample.rotation))
        .collect::<Option<Vec<_>>>()?;
    let written: Vec<Vec<f64>> = match (&at[..], &planes[..]) {
        (&[p], &[(a, alpha)]) => {
            let (outer, inner) = (
                turn(geometry, current, &stack[..p]),
                turn(geometry, current, &stack[p + 1..]),
            );
            let axis = rotation(geometry, stack[p]).ok()?.1;
            solve_component(&axis, &(inner * beam), &(outer.inverse() * a), alpha)
                .values()
                .into_iter()
                .map(|v| vec![v])
                .collect()
        }
        (&[p1, p2], &[first, second]) => {
            let pair = Pair::new(geometry, [stack[p1], stack[p2]])?;
            let mut out = Vec::new();
            for t in meet(first, second)? {
                out.extend(pair.reach(current, &t).into_iter().map(Vec::from));
            }
            out
        }
        _ => return None,
    };
    let mut set = SolutionSet::new(system);
    for x in written {
        if set.is_degenerate() {
            break;
        }
        let mut values = current.to_vec();
        for (&p, v) in at.iter().zip(x) {
            values[stack[p]] = v;
        }
        for image in system.slit_images(values) {
            set.offer(&image);
        }
    }
    Some(set)
}

/// The rotation of the stack's axes `range` at `values`.
fn turn(geometry: &Geometry, values: &[f64], range: &[usize]) -> UnitQuaternion<f64> {
    range.iter().fold(UnitQuaternion::identity(), |r, &i| {
        r * geometry.axes()[i].transform(values[i]).rotation
    })
}

/// Two detector rotations that take k̂f where it must go: the inner by
/// k̂f's component along the outer, the outer by spinning onto it.
pub(super) struct Pair<'g> {
    geometry: &'g Geometry,
    /// Stack positions, outer first.
    at: [usize; 2],
    axes: [Vector3<f64>; 2],
}

impl<'g> Pair<'g> {
    /// `None` unless `axes`, outer first, are rotations of the detector
    /// stack.
    pub(super) fn new(geometry: &'g Geometry, axes: [usize; 2]) -> Option<Self> {
        let stack = geometry.detector_stack();
        let [p1, p2] = axes.map(|i| stack.iter().position(|&s| s == i));
        let [a1, a2] = axes.map(|i| rotation(geometry, i).map(|r| r.1));
        Some(Self {
            geometry,
            at: [p1?, p2?],
            axes: [a1.ok()?, a2.ok()?],
        })
    }

    /// The pair's values, outer first, that turn k̂f to unit `kf`, the
    /// stack's other axes at `values`.
    pub(super) fn reach(&self, values: &[f64], kf: &Vector3<f64>) -> Vec<[f64; 2]> {
        let (geometry, [p1, p2], [a1, a2]) = (self.geometry, self.at, self.axes);
        let stack = geometry.detector_stack();
        let outer = turn(geometry, values, &stack[..p1]);
        let between = turn(geometry, values, &stack[p1 + 1..p2]);
        let b = turn(geometry, values, &stack[p2 + 1..]) * geometry.beam_direction().into_inner();
        let w = outer.inverse() * kf;
        let mut out = Vec::new();
        for v2 in solve_component(&a2, &b, &(between.inverse() * a1), w.dot(&a1)).values() {
            let turned = between * (geometry.axes()[stack[p2]].transform(v2).rotation * b);
            for v1 in spin(&a1, &turned, &w).values() {
                out.push([v1, v2]);
            }
        }
        out
    }
}

/// `e` of `system` as a plane of k̂f, with k̂i and the sample rotation: the
/// plane of k̂i and qaz's direction about it, the plane of a qper, or the
/// 2θ cone about k̂i of an equation that fixes 2θ.
fn plane(
    system: &System,
    e: &Equation,
    ki: &Vector3<f64>,
    sample: &UnitQuaternion<f64>,
) -> Option<Plane> {
    match *e {
        Equation::Qaz { qaz } => {
            let y = (Vector3::y() - ki * ki.y).try_normalize(EPS)?;
            let toward = y * qaz.cos() + ki.cross(&y) * qaz.sin();
            Some((ki.cross(&toward), 0.0))
        }
        Equation::Qper { surface, qper } => {
            let n = (sample * surface).normalize();
            Some((n, qper / system.k() + ki.dot(&n)))
        }
        _ => Some((*ki, system.cos_two_theta(e)?)),
    }
}

/// The unit vectors on both planes: two mirror images, one where the line
/// of the planes touches the sphere, or none; `None` for parallel planes.
fn meet((a1, c1): Plane, (a2, c2): Plane) -> Option<Vec<Vector3<f64>>> {
    let m = a1.cross(&a2);
    let m2 = m.norm_squared();
    if m2 < EPS * EPS {
        return None;
    }
    let g = a1.dot(&a2);
    let p = (a1 * (c1 * a2.norm_squared() - c2 * g) + a2 * (c2 * a1.norm_squared() - c1 * g)) / m2;
    let r = 1.0 - p.norm_squared();
    Some(if r < -EPS {
        Vec::new()
    } else if r <= 0.0 {
        vec![p]
    } else {
        let t = m * (r / m2).sqrt();
        vec![p + t, p - t]
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::super::super::{InverseError, InverseRequest, Mode, Target, solve as numeric, wrap};
    use super::*;
    use crate::Wavelength;
    use crate::presets::preset;
    use nalgebra::Matrix3;

    /// The q2, tth2 and qper_qpar presets: two written detector axes, gamma
    /// and delta.
    const TWO_AXES: [&str; 11] = [
        "E6C",
        "K6C",
        "SOLEIL SIRIUS KAPPA",
        "SOLEIL SIRIUS TURRET",
        "PETRA3 P23 4C",
        "PETRA3 P23 6C",
        "ZAXIS",
        "SOLEIL SIXS MED1+2",
        "SOLEIL SIXS MED2+2",
        "SOLEIL SIXS MED2+3",
        "SOLEIL SIXS MED2+3 v2",
    ];
    /// The q presets: tth alone.
    const ONE_AXIS: [&str; 4] = ["E4CV", "E4CH", "K4CV", "SOLEIL MARS"];
    const LAMBDA: f64 = 1.54;

    fn k() -> f64 {
        Wavelength::new(LAMBDA).unwrap().wavenumber()
    }

    /// The closed form's set and the numeric search's, of `written` closed by
    /// `extra` from `current`, under a detector target.
    fn both(
        geo: &Geometry,
        written: &[&str],
        extra: Vec<Equation>,
        current: &[f64],
    ) -> [Result<Vec<Vec<f64>>, InverseError>; 2] {
        let mode = Mode::named(geo, "m", written, |_| extra).unwrap();
        let request = InverseRequest {
            mode: &mode,
            ub: Matrix3::identity(),
            wavelength: Wavelength::new(LAMBDA).unwrap(),
            target: Target::Detector,
            current,
        };
        let system = request.system().unwrap();
        let ours = search(&system).expect("a closed form");
        [ours, numeric::search(&system)].map(|s| s.into_sorted().map(|f| f.solutions))
    }

    fn same_sets(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
        let within = |x: &[Vec<f64>], y: &[Vec<f64>]| {
            x.iter().all(|x| {
                y.iter()
                    .any(|y| x.iter().zip(y).all(|(u, v)| wrap(u - v).abs() < 1e-6))
            })
        };
        a.len() == b.len() && within(a, b) && within(b, a)
    }

    /// Two held positions of `geo`: all zero, and every axis turned.
    fn currents(geo: &Geometry) -> [Vec<f64>; 2] {
        let n = geo.axes().len();
        [
            vec![0.0; n],
            (0..n).map(|i| 0.3 - 0.17 * i as f64).collect(),
        ]
    }

    /// On every q2, tth2 and qper_qpar preset, from two held positions, the
    /// detector arms come to the numeric search's set, or both fail alike.
    #[test]
    fn two_axis_arms_agree_with_numeric() {
        let k = k();
        let surface = Vector3::y();
        let goals = [
            vec![Equation::QNorm { q: 3.0 }, Equation::Qaz { qaz: 0.4 }],
            vec![Equation::QNorm { q: 5.5 }, Equation::Qaz { qaz: -2.9 }],
            vec![Equation::Tth { tth: 1.1 }, Equation::Qaz { qaz: 1.3 }],
            vec![Equation::Tth { tth: 2.8 }, Equation::Qaz { qaz: PI }],
            vec![
                Equation::Qper { surface, qper: 1.5 },
                Equation::Qpar { surface, qpar: 2.0 },
            ],
            vec![
                Equation::Qper {
                    surface,
                    qper: -0.8,
                },
                Equation::Qpar {
                    surface,
                    qpar: -1.2 * k,
                },
            ],
        ];
        let mut failed = Vec::new();
        let mut solved = 0;
        for name in TWO_AXES {
            let geo = preset(name).unwrap();
            for current in currents(&geo) {
                for extra in &goals {
                    let [ours, theirs] = both(&geo, &["gamma", "delta"], extra.clone(), &current);
                    match (&ours, &theirs) {
                        (Ok(a), Ok(b)) if same_sets(a, b) => solved += a.len(),
                        (Err(a), Err(b)) if a == b => {}
                        _ => failed.push(format!(
                            "{name} {current:?} {extra:?}:\nclosed {ours:?}\nnumeric {theirs:?}"
                        )),
                    }
                }
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
        assert!(solved > 0);
    }

    /// The q engine's tth from asin on the four-circle presets, at both
    /// signs and at the cone points 2θ = 0 and π, which are regular roots.
    #[test]
    fn one_axis_tth_agrees_with_numeric() {
        let q_max = 2.0 * k();
        for name in ONE_AXIS {
            let geo = preset(name).unwrap();
            let tth = geo.axis_index("tth").unwrap();
            for current in currents(&geo) {
                for q in [2.1, -2.1, 0.0, q_max, -q_max] {
                    let extra = vec![Equation::QSigned { q }];
                    let [ours, theirs] = both(&geo, &["tth"], extra, &current);
                    let ours = ours.unwrap();
                    assert!(same_sets(&ours, &theirs.unwrap()), "{name} {q}");
                    let want = 2.0 * (q / q_max).asin();
                    assert_eq!(ours.len(), 1, "{name} {q}: {ours:?}");
                    assert!(
                        wrap(ours[0][tth] - want).abs() < 1e-12,
                        "{name} {q}: {ours:?}"
                    );
                }
            }
        }
    }
}
