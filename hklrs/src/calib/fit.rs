//! `calibrate_area` and `calibrate_linear`: the residual model of
//! [`model`](super::model) driven by the in-house Levenberg–Marquardt solver.

use nalgebra::Vector3;

use super::model::{Rig, Values, residual_count};
use super::{CalFrame, CalibError, Failure, Fit, Param, ParamId, Reject, lm};
use crate::xu::{Area, Direction, Linear};
use crate::{Deg, Error, Geometry, PixelDetector, Sample};

/// Relative eigenvalue floor of `JᵀJ` below which a parameter combination is
/// not determined.
const RANK_TOL: f64 = 1e-10;
/// How many times the set of locatable frames may change between solves.
const ACTIVE_ROUNDS: usize = 4;
/// Two direction sets whose per-residual cost differs by less than this
/// fraction are ambiguous.
const AMBIGUITY: f64 = 0.10;

/// What an area-detector calibration holds fixed: the goniometer, the sample
/// for reflection frames, and the chip.
#[derive(Clone, Debug)]
pub struct AreaModel {
    pub geometry: Geometry,
    /// Needed by `FrameKind::Reflection` frames.
    pub sample: Option<Sample>,
    /// Name of the rotation axis `AngleOffset` shifts; `None` when the
    /// offset is held at its start value of zero.
    pub offset_axis: Option<String>,
    pub nch: [usize; 2],
    /// The xu detector directions; `None` searches the eight combinations
    /// perpendicular to the beam.
    pub directions: Option<[Direction; 2]>,
}

/// One explicit start of an area calibration. Every parameter is free or
/// held; the wavelength is fitted when its `Param` is free (XU-7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaStart {
    pub cch1: Param,
    pub cch2: Param,
    pub pwidth1: Param,
    pub pwidth2: Param,
    pub distance: Param,
    pub tiltazimuth: Param,
    pub tilt: Param,
    pub detrot: Param,
    pub angle_offset: Param,
    pub wavelength: Param,
}

#[derive(Clone, Debug)]
pub struct LinearModel {
    pub geometry: Geometry,
    pub sample: Option<Sample>,
    pub offset_axis: Option<String>,
    pub nch: usize,
    /// The xu detector direction; `None` searches the four perpendicular to
    /// the beam.
    pub direction: Option<Direction>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearStart {
    pub cch: Param,
    pub pwidth: Param,
    pub distance: Param,
    pub tilt: Param,
    pub angle_offset: Param,
    pub wavelength: Param,
}

impl AreaStart {
    fn params(&self) -> Vec<(ParamId, Param)> {
        vec![
            (ParamId::Cch1, self.cch1),
            (ParamId::Cch2, self.cch2),
            (ParamId::Pwidth1, self.pwidth1),
            (ParamId::Pwidth2, self.pwidth2),
            (ParamId::Distance, self.distance),
            (ParamId::TiltAzimuth, self.tiltazimuth),
            (ParamId::Tilt, self.tilt),
            (ParamId::Detrot, self.detrot),
            (ParamId::AngleOffset, self.angle_offset),
            (ParamId::Wavelength, self.wavelength),
        ]
    }
}

impl LinearStart {
    fn params(&self) -> Vec<(ParamId, Param)> {
        vec![
            (ParamId::Cch2, self.cch),
            (ParamId::Pwidth2, self.pwidth),
            (ParamId::Distance, self.distance),
            (ParamId::Tilt, self.tilt),
            (ParamId::AngleOffset, self.angle_offset),
            (ParamId::Wavelength, self.wavelength),
        ]
    }
}

/// A converged run with the numbers that rank it against other runs.
struct Run<T> {
    fit: Fit<T>,
    /// `½ Σ r²` per residual.
    cost: f64,
}

/// Typical size of a parameter, for the solver's scaling.
fn scale(id: ParamId, start: f64) -> f64 {
    let relative = |floor: f64| (start.abs() * 0.01).max(floor);
    match id {
        ParamId::Cch1 | ParamId::Cch2 => 10.0,
        ParamId::Pwidth1 | ParamId::Pwidth2 => relative(1e-12),
        ParamId::Distance => relative(1e-9),
        ParamId::Wavelength => relative(1e-6),
        ParamId::TiltAzimuth | ParamId::Tilt | ParamId::Detrot | ParamId::AngleOffset => 1.0,
    }
}

type Build<'a> = dyn Fn(&Values) -> Result<PixelDetector, Error> + 'a;

/// Why each frame is out of the fit at one parameter point, `None` for a
/// frame that can be located.
fn classify(
    rig: &Rig,
    frames: &[CalFrame],
    build: &Build,
    v: &Values,
) -> Result<Vec<Option<Reject>>, CalibError> {
    let det = build(v).map_err(CalibError::Model)?;
    Ok(frames
        .iter()
        .enumerate()
        .map(|(i, f)| rig.residual(&det, v, i, f).err())
        .collect())
}

/// Fit the free members of `params` to `frames`. `build` makes the detector
/// from the parameter values, `finish` the calibrated object.
fn run<T>(
    rig: &Rig,
    frames: &[CalFrame],
    params: &[(ParamId, Param)],
    build: &Build,
    finish: &dyn Fn(&Values) -> T,
) -> Result<Run<T>, CalibError> {
    for (id, p) in params {
        p.check(*id)?;
    }
    let mut base = Values([0.0; ParamId::COUNT]);
    for (id, p) in params {
        base.0[id.index()] = p.start;
    }
    let free: Vec<(ParamId, Param)> = params.iter().copied().filter(|(_, p)| p.free).collect();
    if free.is_empty() {
        return Err(CalibError::BadProblem);
    }
    if free.iter().any(|(id, _)| *id == ParamId::AngleOffset) && !rig.has_offset_axis() {
        return Err(CalibError::BadProblem);
    }
    let ids: Vec<ParamId> = free.iter().map(|(id, _)| *id).collect();
    let scales: Vec<f64> = free.iter().map(|(id, p)| scale(*id, p.start)).collect();
    let values = |x: &[f64]| {
        let mut v = base;
        for (id, &xi) in ids.iter().zip(x) {
            v.0[id.index()] = xi;
        }
        v
    };

    let mut x: Vec<f64> = free.iter().map(|(_, p)| p.start).collect();
    let mut out = classify(rig, frames, build, &values(&x))?;
    for round in 0..ACTIVE_ROUNDS {
        let active: Vec<usize> = (0..frames.len()).filter(|&i| out[i].is_none()).collect();
        let m: usize = active.iter().map(|&i| residual_count(&frames[i])).sum();
        if m <= ids.len() {
            return Err(CalibError::TooFewFrames {
                have: m,
                need: ids.len() + 1,
            });
        }
        let sol = lm::solve(
            |x: &[f64]| {
                let v = values(x);
                let Ok(det) = build(&v) else {
                    return vec![f64::NAN; m];
                };
                active
                    .iter()
                    .flat_map(|&i| {
                        rig.residual(&det, &v, i, &frames[i])
                            .unwrap_or_else(|_| vec![f64::NAN; residual_count(&frames[i])])
                    })
                    .collect()
            },
            &x,
            &scales,
            &lm::Options::default(),
        )
        .map_err(|failed| CalibError::from_failed(&ids, failed))?;
        x = sol.x.clone();
        let now = classify(rig, frames, build, &values(&x))?;
        let stable = (0..frames.len()).all(|i| now[i].is_none() == out[i].is_none());
        if stable || round + 1 == ACTIVE_ROUNDS {
            if !stable {
                // frames keep entering and leaving the detector as the
                // parameters move: no settled set of frames to report
                return Err(CalibError::from_failed(
                    &ids,
                    lm::Failed {
                        reason: Failure::StalledStep,
                        x,
                        cost: sol.cost,
                        iters: sol.iters,
                    },
                ));
            }
            let rejected = out
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.map(|r| (i, r)))
                .collect();
            let cost = sol.cost / sol.m as f64;
            let fit = Fit::from_solution(finish(&values(&x)), &ids, &sol, rejected, RANK_TOL)?;
            return Ok(Run { fit, cost });
        }
        out = now;
    }
    unreachable!("the last round returns")
}

/// The direction strings perpendicular to the beam, when it lies along an
/// axis.
fn perpendicular(beam: Vector3<f64>) -> Result<Vec<Direction>, CalibError> {
    let along = (0..3)
        .find(|&a| beam[a].abs() > 1.0 - 1e-9)
        .ok_or(CalibError::BadProblem)?;
    let mut out = Vec::new();
    for (a, name) in ["x", "y", "z"].into_iter().enumerate() {
        if a == along {
            continue;
        }
        for sign in ["+", "-"] {
            out.push(format!("{name}{sign}").parse().expect("a direction string"));
        }
    }
    Ok(out)
}

/// Most informative of several failures: a parameter that cannot be
/// determined, then the run that got closest.
fn worse(a: CalibError, b: CalibError) -> CalibError {
    use CalibError::*;
    match (&a, &b) {
        (Unidentifiable { .. }, _) => a,
        (_, Unidentifiable { .. }) => b,
        (NotConverged { cost: ca, .. }, NotConverged { cost: cb, .. }) => {
            if cb < ca {
                b
            } else {
                a
            }
        }
        (NotConverged { .. }, _) => a,
        (_, NotConverged { .. }) => b,
        _ => a,
    }
}

/// The lowest-cost converged run of `starts`, or the most informative failure.
fn best_start<T>(
    starts: usize,
    mut one: impl FnMut(usize) -> Result<Run<T>, CalibError>,
) -> Result<Run<T>, CalibError> {
    let mut best: Option<Result<Run<T>, CalibError>> = None;
    for s in 0..starts {
        let r = one(s);
        best = Some(match (best, r) {
            (None, r) => r,
            (Some(Ok(a)), Ok(b)) => Ok(if b.cost < a.cost { b } else { a }),
            (Some(Ok(a)), Err(_)) | (Some(Err(_)), Ok(a)) => Ok(a),
            (Some(Err(a)), Err(b)) => Err(worse(a, b)),
        });
    }
    best.unwrap_or(Err(CalibError::BadProblem))
}

/// The best candidate, or `AmbiguousDirections` when the runner-up is within
/// 10 % of it. `one` fits one candidate.
fn best_candidate<T>(
    candidates: &[Vec<Direction>],
    mut one: impl FnMut(&[Direction]) -> Result<Run<T>, CalibError>,
) -> Result<Fit<T>, CalibError> {
    let mut ok: Vec<(usize, Run<T>)> = Vec::new();
    let mut failure: Option<CalibError> = None;
    for (i, c) in candidates.iter().enumerate() {
        match one(c) {
            Ok(r) => ok.push((i, r)),
            Err(e) => failure = Some(failure.map_or(e.clone(), |f| worse(f, e))),
        }
    }
    ok.sort_by(|a, b| a.1.cost.total_cmp(&b.1.cost));
    let mut it = ok.into_iter();
    let Some((bi, best)) = it.next() else {
        return Err(failure.unwrap_or(CalibError::BadProblem));
    };
    if let Some((ri, runner)) = it.next()
        && runner.cost <= best.cost * (1.0 + AMBIGUITY)
    {
        return Err(CalibError::AmbiguousDirections {
            best: candidates[bi].clone(),
            runner_up: candidates[ri].clone(),
            best_cost: best.cost,
            runner_up_cost: runner.cost,
        });
    }
    Ok(best.fit)
}

/// Fit an xu `Area` to `frames`. `starts` is an explicit list; the
/// lowest-cost converged run wins. Frames whose ray misses the detector are
/// listed in `Fit::rejected`, never silently left out.
pub fn calibrate_area(
    m: &AreaModel,
    frames: &[CalFrame],
    starts: &[AreaStart],
) -> Result<Fit<Area>, CalibError> {
    if starts.is_empty() {
        return Err(CalibError::BadProblem);
    }
    let rig = Rig::new(
        &m.geometry,
        m.sample.as_ref(),
        m.offset_axis.as_deref(),
        frames,
    )?;
    let candidates: Vec<Vec<Direction>> = match m.directions {
        Some(d) => vec![d.to_vec()],
        None => {
            let perp = perpendicular(rig.beam().into_inner())?;
            let mut all = Vec::new();
            for &d1 in &perp {
                for &d2 in &perp {
                    if d1.vector().dot(&d2.vector()) == 0.0 {
                        all.push(vec![d1, d2]);
                    }
                }
            }
            all
        }
    };
    best_candidate(&candidates, |dirs| {
        let area = |v: &Values| {
            let mut a = Area::new(
                dirs[0],
                dirs[1],
                [v.get(ParamId::Cch1), v.get(ParamId::Cch2)],
                m.nch,
                v.get(ParamId::Distance),
                [v.get(ParamId::Pwidth1), v.get(ParamId::Pwidth2)],
            );
            a.tilt = Deg(v.get(ParamId::Tilt));
            a.tiltazimuth = Deg(v.get(ParamId::TiltAzimuth));
            a.detrot = Deg(v.get(ParamId::Detrot));
            a
        };
        let beam = rig.beam();
        best_start(starts.len(), |s| {
            run(
                &rig,
                frames,
                &starts[s].params(),
                &|v| area(v).detector(beam),
                &|v| area(v),
            )
        })
    })
}

/// Fit an xu `Linear` to `frames`, whose peaks carry no slow-axis
/// information (`From<Peak1> for Peak`).
pub fn calibrate_linear(
    m: &LinearModel,
    frames: &[CalFrame],
    starts: &[LinearStart],
) -> Result<Fit<Linear>, CalibError> {
    if starts.is_empty() {
        return Err(CalibError::BadProblem);
    }
    let rig = Rig::new(
        &m.geometry,
        m.sample.as_ref(),
        m.offset_axis.as_deref(),
        frames,
    )?;
    let candidates: Vec<Vec<Direction>> = match m.direction {
        Some(d) => vec![vec![d]],
        None => perpendicular(rig.beam().into_inner())?
            .into_iter()
            .map(|d| vec![d])
            .collect(),
    };
    best_candidate(&candidates, |dirs| {
        let line = |v: &Values| {
            let mut l = Linear::new(
                dirs[0],
                v.get(ParamId::Cch2),
                m.nch,
                v.get(ParamId::Distance),
                v.get(ParamId::Pwidth2),
            );
            l.tilt = Deg(v.get(ParamId::Tilt));
            l
        };
        let beam = rig.beam();
        best_start(starts.len(), |s| {
            run(
                &rig,
                frames,
                &starts[s].params(),
                &|v| line(v).detector(beam),
                &|v| line(v),
            )
        })
    })
}
