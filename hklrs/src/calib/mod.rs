//! Detector calibration: fit the parameters of an xrayutilities `Area` or
//! `Linear` detector to peak positions found in frames taken at known angles.
//!
//! A fit is a [`Fit`] only if it converged. Anything else is a
//! [`CalibError`], whose `NotConverged` carries the best parameter values as
//! `(ParamId, f64)` pairs for diagnosis and nothing a caller could mistake
//! for a calibrated detector (xrayutilities returned its start values after
//! a failed fit, XU-6). Uncertainties are keyed by [`ParamId`]; fixed
//! parameters have none.

use nalgebra::{DMatrix, Vector3};

use crate::xu::Direction;

pub mod lm;
pub mod peak;

pub use lm::{Failure, Stop};
pub use peak::{Peak, Peak1, Reject};

/// A calibration parameter. A linear detector uses `Cch2`, `Pwidth2`,
/// `Distance`, `Tilt` and `AngleOffset`: its channel index is the fast axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ParamId {
    Cch1,
    Cch2,
    Pwidth1,
    Pwidth2,
    Distance,
    TiltAzimuth,
    Tilt,
    Detrot,
    /// Zero offset of the outer detector circle.
    AngleOffset,
}

/// A start value, fitted or held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Param {
    pub start: f64,
    pub free: bool,
}

impl Param {
    pub fn free(start: f64) -> Self {
        Self { start, free: true }
    }

    pub fn fixed(value: f64) -> Self {
        Self {
            start: value,
            free: false,
        }
    }

    /// `Err(BadStart)` unless the start value is finite.
    pub fn check(&self, id: ParamId) -> Result<(), CalibError> {
        if self.start.is_finite() {
            Ok(())
        } else {
            Err(CalibError::BadStart {
                id,
                value: self.start,
            })
        }
    }
}

/// What a frame shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameKind {
    /// The direct beam, where k_f = k_i.
    PrimaryBeam,
    /// A Bragg peak with these Miller indices.
    Reflection(Vector3<f64>),
}

/// One frame reduced to a peak: the axis values it was taken at, the peak
/// position and error in pixels, and what the peak is. A `sigma` component
/// of infinity says the frame carries no information on that pixel axis
/// (a row detector's slow axis).
#[derive(Clone, Debug, PartialEq)]
pub struct CalFrame {
    angles: Vec<f64>,
    peak: Peak,
    kind: FrameKind,
}

impl CalFrame {
    /// `index` is the frame's place in the caller's list, for the error.
    pub fn new(
        index: usize,
        angles: Vec<f64>,
        peak: Peak,
        kind: FrameKind,
    ) -> Result<Self, CalibError> {
        let bad = |what| Err(CalibError::BadFrame { index, what });
        if angles.iter().any(|a| !a.is_finite()) {
            return bad("angle is not finite");
        }
        if peak.pos.iter().any(|p| !p.is_finite()) {
            return bad("peak position is not finite");
        }
        if peak.sigma.iter().any(|s| s.is_nan() || *s <= 0.0) {
            return bad("peak sigma is not positive");
        }
        if peak.sigma.iter().all(|s| s.is_infinite()) {
            return bad("peak sigma is infinite on both axes");
        }
        if let FrameKind::Reflection(h) = kind
            && h.iter().any(|v| !v.is_finite())
        {
            return bad("hkl is not finite");
        }
        Ok(Self { angles, peak, kind })
    }

    pub fn angles(&self) -> &[f64] {
        &self.angles
    }

    pub fn peak(&self) -> Peak {
        self.peak
    }

    pub fn kind(&self) -> FrameKind {
        self.kind
    }
}

impl From<Peak1> for Peak {
    /// A row detector's peak: slow index 0, no information along it.
    fn from(p: Peak1) -> Self {
        Self {
            pos: [0.0, p.pos],
            sigma: [f64::INFINITY, p.sigma],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CalibError {
    /// The fit did not converge; `best` is the best point reached.
    NotConverged {
        reason: Failure,
        best: Vec<(ParamId, f64)>,
        cost: f64,
        iters: usize,
    },
    /// No combination of the data pins these free parameters down.
    Unidentifiable {
        params: Vec<ParamId>,
    },
    TooFewFrames {
        have: usize,
        need: usize,
    },
    /// Two direction pairs fit within 10 % of each other.
    AmbiguousDirections {
        best: (Direction, Direction),
        runner_up: (Direction, Direction),
        best_cost: f64,
        runner_up_cost: f64,
    },
    BadStart {
        id: ParamId,
        value: f64,
    },
    BadFrame {
        index: usize,
        what: &'static str,
    },
    /// No free parameters, or fewer residuals than free parameters.
    BadProblem,
}

impl std::fmt::Display for CalibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConverged {
                reason,
                cost,
                iters,
                ..
            } => {
                write!(
                    f,
                    "calibration did not converge ({reason:?}) after {iters} iterations, cost {cost}"
                )
            }
            Self::Unidentifiable { params } => {
                write!(
                    f,
                    "the frames do not determine {params:?}; fix them or add frames"
                )
            }
            Self::TooFewFrames { have, need } => {
                write!(f, "{have} residuals, at least {need} needed")
            }
            Self::AmbiguousDirections {
                best,
                runner_up,
                best_cost,
                runner_up_cost,
            } => write!(
                f,
                "detector directions {best:?} (cost {best_cost}) and {runner_up:?} (cost {runner_up_cost}) fit equally well; give them"
            ),
            Self::BadStart { id, value } => write!(f, "start value of {id:?} is {value}"),
            Self::BadFrame { index, what } => write!(f, "frame {index}: {what}"),
            Self::BadProblem => write!(
                f,
                "no free parameters, or fewer residuals than free parameters"
            ),
        }
    }
}

impl std::error::Error for CalibError {}

impl CalibError {
    /// The error for a solver run that did not converge; `ids` names the
    /// solver's parameters in order.
    pub fn from_failed(ids: &[ParamId], failed: lm::Failed) -> Self {
        if failed.reason == Failure::BadInput {
            return Self::BadProblem;
        }
        Self::NotConverged {
            reason: failed.reason,
            best: ids.iter().copied().zip(failed.x).collect(),
            cost: failed.cost,
            iters: failed.iters,
        }
    }
}

/// A fitted free parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fitted {
    pub id: ParamId,
    pub value: f64,
    pub sigma: f64,
}

/// A converged calibration of `T` (an `Area` or `Linear`).
#[derive(Clone, Debug, PartialEq)]
pub struct Fit<T> {
    value: T,
    free: Vec<Fitted>,
    cov: DMatrix<f64>,
    rms: f64,
    dof: usize,
    rejected: Vec<(usize, Reject)>,
    iters: usize,
    stop: Stop,
}

impl<T> Fit<T> {
    /// Covariance, errors and rms of a converged solver run whose parameters
    /// are `ids` (the free ones, in solver order); `value` is the calibrated
    /// object built from `sol.x` and the held parameters. The residuals must
    /// be in units of their σ: `cov = s² (JᵀJ)⁻¹` with `s² = 2·cost/dof`
    /// then has its stated coverage, and `rms` is about 1 for a good model.
    ///
    /// # Panics
    /// If `ids.len() != sol.x.len()`.
    pub fn from_solution(
        value: T,
        ids: &[ParamId],
        sol: &lm::Solution,
        rejected: Vec<(usize, Reject)>,
        rank_tol: f64,
    ) -> Result<Self, CalibError> {
        assert_eq!(ids.len(), sol.x.len(), "one ParamId per solver parameter");
        let n = ids.len();
        if sol.m <= n {
            return Err(CalibError::TooFewFrames {
                have: sol.m,
                need: n + 1,
            });
        }
        let dof = sol.m - n;
        let inv = sol
            .covariance(rank_tol)
            .map_err(|bad| CalibError::Unidentifiable {
                params: bad.into_iter().map(|j| ids[j]).collect(),
            })?;
        let s2 = 2.0 * sol.cost / dof as f64;
        let cov = inv * s2;
        let free = (0..n)
            .map(|j| Fitted {
                id: ids[j],
                value: sol.x[j],
                sigma: cov[(j, j)].sqrt(),
            })
            .collect();
        Ok(Self {
            value,
            free,
            cov,
            rms: (2.0 * sol.cost / sol.m as f64).sqrt(),
            dof,
            rejected,
            iters: sol.iters,
            stop: sol.stop,
        })
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    /// The free parameters, in the order of [`cov`](Self::cov).
    pub fn free(&self) -> &[Fitted] {
        &self.free
    }

    /// σ of a parameter; `None` if it was held fixed.
    pub fn sigma(&self, id: ParamId) -> Option<f64> {
        self.free.iter().find(|f| f.id == id).map(|f| f.sigma)
    }

    pub fn cov(&self) -> &DMatrix<f64> {
        &self.cov
    }

    /// Root mean square of the residuals, in σ units.
    pub fn rms(&self) -> f64 {
        self.rms
    }

    pub fn dof(&self) -> usize {
        self.dof
    }

    /// Frames left out, by index, with the reason.
    pub fn rejected(&self) -> &[(usize, Reject)] {
        &self.rejected
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn stop(&self) -> Stop {
        self.stop
    }

    pub fn into_value(self) -> T {
        self.value
    }
}
