//! `calibrate_area` and `calibrate_linear`: the residual model of
//! [`model`](super::model) driven by the in-house Levenberg–Marquardt solver.

use nalgebra::Vector3;

use super::model::{Rig, Values, cell, residual_count};
use super::{CalFrame, CalibError, Fit, FrameKind, Param, ParamId, lm};
use crate::xu::{Area, Direction, Linear};
use crate::{Deg, Error, Geometry, PixelDetector, Sample};

/// Relative eigenvalue floor of `JᵀJ` below which a parameter combination is
/// not determined.
const RANK_TOL: f64 = 1e-10;
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

/// What [`calibrate_area_hkl`] fits: the detector and what the reflection
/// frames add.
#[derive(Clone, Debug, PartialEq)]
pub struct AreaHkl {
    pub area: Area,
    pub wavelength: f64,
    pub angle_offset: Deg,
    /// `Rz(sample_azimuth) · Ry(sample_tilt)` applied to the sample's UB.
    pub sample_tilt: Deg,
    pub sample_azimuth: Deg,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearHkl {
    pub line: Linear,
    pub wavelength: f64,
    pub angle_offset: Deg,
    pub sample_tilt: Deg,
    pub sample_azimuth: Deg,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaHklStart {
    pub area: AreaStart,
    pub sample_tilt: Param,
    pub sample_azimuth: Param,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearHklStart {
    pub line: LinearStart,
    pub sample_tilt: Param,
    pub sample_azimuth: Param,
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
        ParamId::TiltAzimuth
        | ParamId::Tilt
        | ParamId::Detrot
        | ParamId::AngleOffset
        | ParamId::SampleTilt
        | ParamId::SampleAzimuth => 1.0,
    }
}

type Build<'a> = dyn Fn(&Values) -> Result<PixelDetector, Error> + 'a;

/// Fit the free members of `params` to `frames`. `build` makes the detector
/// from the parameter values, `finish` the calibrated object.
///
/// A frame is out of the fit only when its ray has no index on its module's
/// plane (behind or parallel) at the start; it is listed, never dropped. A
/// frame predicted off the chip keeps its residual.
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

    let x0: Vec<f64> = free.iter().map(|(_, p)| p.start).collect();
    let v0 = values(&x0);
    let det = build(&v0).map_err(CalibError::Model)?;
    let mut rejected = Vec::new();
    let mut active = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let on_image = cell(f.peak().pos).is_some_and(|(s, c)| det.module(s, c).is_ok());
        if !on_image {
            return Err(CalibError::BadFrame {
                index: i,
                what: "peak lies outside the detector image",
            });
        }
        match rig.residual(&det, &v0, i, f) {
            Ok(_) => active.push(i),
            Err(r) => rejected.push((i, r)),
        }
    }
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
        &x0,
        &scales,
        &lm::Options::default(),
    )
    .map_err(|failed| CalibError::from_failed(&ids, failed))?;
    let cost = sol.cost / sol.m as f64;
    let fit = Fit::from_solution(finish(&values(&sol.x)), &ids, &sol, rejected, RANK_TOL)?;
    Ok(Run { fit, cost })
}

/// Whether two detectors of one image shape put every pixel at the same
/// place (within a hundredth of the pixel pitch), so they predict the same
/// pixel for every ray whatever their parameters.
fn same_pixels(a: &PixelDetector, b: &PixelDetector) -> bool {
    let (n, k) = a.shape();
    if a.shape() != b.shape() {
        return false;
    }
    let p = |d: &PixelDetector, i, j| d.pixel(i, j).expect("inside the shape");
    let mut pitch = f64::INFINITY;
    if n > 1 {
        pitch = pitch.min((p(a, 1, 0) - p(a, 0, 0)).norm());
    }
    if k > 1 {
        pitch = pitch.min((p(a, 0, 1) - p(a, 0, 0)).norm());
    }
    let tol = 0.01 * pitch;
    (0..n).all(|i| (0..k).all(|j| (p(a, i, j) - p(b, i, j)).norm() <= tol))
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

/// The best candidate, or `AmbiguousDirections` when a runner-up within 10 %
/// of it puts the pixels elsewhere. Candidates whose detectors put every
/// pixel in the same place (`same`) are one fit: the one with the lowest
/// `rank` is returned. `one` fits one candidate.
fn best_candidate<T>(
    candidates: &[Vec<Direction>],
    mut one: impl FnMut(&[Direction]) -> Result<Run<T>, CalibError>,
    same: impl Fn(&T, &T) -> bool,
    rank: impl Fn(&T) -> f64,
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
    let (mut chosen, mut others) = (best, Vec::new());
    let mut ci = bi;
    for (i, r) in it {
        if same(chosen.fit.value(), r.fit.value()) {
            if rank(r.fit.value()) < rank(chosen.fit.value()) {
                (chosen, ci) = (r, i);
            }
        } else {
            others.push((i, r));
        }
    }
    if let Some((ri, runner)) = others.into_iter().next()
        && runner.cost <= chosen.cost * (1.0 + AMBIGUITY)
    {
        return Err(CalibError::AmbiguousDirections {
            best: candidates[ci].clone(),
            runner_up: candidates[ri].clone(),
            best_cost: chosen.cost,
            runner_up_cost: runner.cost,
        });
    }
    Ok(chosen.fit)
}

/// Fit an xu `Area` to `frames`. `starts` is an explicit list; the
/// lowest-cost converged run wins. Frames whose ray misses the detector are
/// listed in `Fit::rejected`, never silently left out.
pub fn calibrate_area(
    m: &AreaModel,
    frames: &[CalFrame],
    starts: &[AreaStart],
) -> Result<Fit<Area>, CalibError> {
    let params: Vec<_> = starts.iter().map(AreaStart::params).collect();
    area_fit(m, frames, &params, &|a, _| a, &|a| a)
}

/// Like [`calibrate_area`], also fitting the sample's orientation and, when
/// its `Param` is free, the wavelength from the reflection frames, as xu's
/// `area_detector_calib_hkl` does. Primary-beam frames fix the detector,
/// reflection frames (selected by [`FrameKind`], whatever their indices) tie
/// it to the sample; at least one of each is required.
pub fn calibrate_area_hkl(
    m: &AreaModel,
    frames: &[CalFrame],
    starts: &[AreaHklStart],
) -> Result<Fit<AreaHkl>, CalibError> {
    needs_both_kinds(frames)?;
    let params: Vec<_> = starts
        .iter()
        .map(|s| {
            let mut p = s.area.params();
            p.push((ParamId::SampleTilt, s.sample_tilt));
            p.push((ParamId::SampleAzimuth, s.sample_azimuth));
            p
        })
        .collect();
    area_fit(
        m,
        frames,
        &params,
        &|area, v| AreaHkl {
            area,
            wavelength: v.get(ParamId::Wavelength),
            angle_offset: Deg(v.get(ParamId::AngleOffset)),
            sample_tilt: Deg(v.get(ParamId::SampleTilt)),
            sample_azimuth: Deg(v.get(ParamId::SampleAzimuth)),
        },
        &|h| &h.area,
    )
}

fn needs_both_kinds(frames: &[CalFrame]) -> Result<(), CalibError> {
    let reflections = frames
        .iter()
        .filter(|f| matches!(f.kind(), FrameKind::Reflection(_)))
        .count();
    if reflections == 0 || reflections == frames.len() {
        return Err(CalibError::BadProblem);
    }
    Ok(())
}

fn area_fit<T>(
    m: &AreaModel,
    frames: &[CalFrame],
    params: &[Vec<(ParamId, Param)>],
    make: &dyn Fn(Area, &Values) -> T,
    area_of: &dyn Fn(&T) -> &Area,
) -> Result<Fit<T>, CalibError> {
    if params.is_empty() {
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
    let beam = rig.beam();
    best_candidate(
        &candidates,
        |dirs| {
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
            best_start(params.len(), |s| {
                run(
                    &rig,
                    frames,
                    &params[s],
                    &|v| area(v).detector(beam),
                    &|v| make(area(v), v),
                )
            })
        },
        |a, b| same_detector(area_of(a).detector(beam), area_of(b).detector(beam)),
        |a| {
            // detrot turns the detector about its normal, so a quarter turn
            // with the directions swapped is the same detector: prefer the
            // turn in (-45°, 45°]
            let w = (area_of(a).detrot.0 + 180.0).rem_euclid(360.0) - 180.0;
            if w > -45.0 && w <= 45.0 {
                0.0
            } else {
                1.0 + w.abs()
            }
        },
    )
}

fn same_detector(a: Result<PixelDetector, Error>, b: Result<PixelDetector, Error>) -> bool {
    matches!((a, b), (Ok(a), Ok(b)) if same_pixels(&a, &b))
}

/// Fit an xu `Linear` to `frames`, whose peaks carry no slow-axis
/// information (`From<Peak1> for Peak`).
pub fn calibrate_linear(
    m: &LinearModel,
    frames: &[CalFrame],
    starts: &[LinearStart],
) -> Result<Fit<Linear>, CalibError> {
    let params: Vec<_> = starts.iter().map(LinearStart::params).collect();
    linear_fit(m, frames, &params, &|l, _| l, &|l| l)
}

/// [`calibrate_area_hkl`] for a linear detector.
pub fn calibrate_linear_hkl(
    m: &LinearModel,
    frames: &[CalFrame],
    starts: &[LinearHklStart],
) -> Result<Fit<LinearHkl>, CalibError> {
    needs_both_kinds(frames)?;
    let params: Vec<_> = starts
        .iter()
        .map(|s| {
            let mut p = s.line.params();
            p.push((ParamId::SampleTilt, s.sample_tilt));
            p.push((ParamId::SampleAzimuth, s.sample_azimuth));
            p
        })
        .collect();
    linear_fit(
        m,
        frames,
        &params,
        &|line, v| LinearHkl {
            line,
            wavelength: v.get(ParamId::Wavelength),
            angle_offset: Deg(v.get(ParamId::AngleOffset)),
            sample_tilt: Deg(v.get(ParamId::SampleTilt)),
            sample_azimuth: Deg(v.get(ParamId::SampleAzimuth)),
        },
        &|h| &h.line,
    )
}

fn linear_fit<T>(
    m: &LinearModel,
    frames: &[CalFrame],
    params: &[Vec<(ParamId, Param)>],
    make: &dyn Fn(Linear, &Values) -> T,
    line_of: &dyn Fn(&T) -> &Linear,
) -> Result<Fit<T>, CalibError> {
    if params.is_empty() {
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
    let beam = rig.beam();
    best_candidate(
        &candidates,
        |dirs| {
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
            best_start(params.len(), |s| {
                run(
                    &rig,
                    frames,
                    &params[s],
                    &|v| line(v).detector(beam),
                    &|v| make(line(v), v),
                )
            })
        },
        |a, b| same_detector(line_of(a).detector(beam), line_of(b).detector(beam)),
        // a tilt past a half turn is the opposite direction at the supplement
        |l| ((line_of(l).tilt.0 + 180.0).rem_euclid(360.0) - 180.0).abs(),
    )
}
