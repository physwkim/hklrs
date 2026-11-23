//! The residual model: where a frame's peak should sit for given detector
//! parameters, through the one forward path (`Geometry::poses`, the xu
//! importer's `detector`, `PixelDetector::locate`). There is no second pixel
//! model in the fit, so the detector circle order is the geometry's alone
//! (XU-1).

use std::f64::consts::TAU;

use nalgebra::{Matrix3, Point3, Unit, Vector3};

use super::{CalFrame, CalibError, FrameKind, ParamId, Reject};
use crate::{Error, Geometry, PixelDetector, Poses, Sample};

/// The parameter values of one evaluation, indexed by [`ParamId`]. Degrees
/// for the angles, as in the xu importer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Values(pub [f64; ParamId::COUNT]);

impl Values {
    pub fn get(&self, id: ParamId) -> f64 {
        self.0[id.index()]
    }
}

/// What every frame is placed through: the geometry, the sample for
/// reflection frames, and which axis the fitted `AngleOffset` shifts.
pub(crate) struct Rig<'a> {
    geometry: &'a Geometry,
    /// UB of the sample, `None` when there is none.
    ub: Option<Matrix3<f64>>,
    offset_axis: Option<usize>,
    /// Per frame, its axis values in the units `Geometry::poses` takes.
    axes: Vec<Vec<f64>>,
}

impl<'a> Rig<'a> {
    pub fn new(
        geometry: &'a Geometry,
        sample: Option<&Sample>,
        offset_axis: Option<&str>,
        frames: &[CalFrame],
    ) -> Result<Self, CalibError> {
        let offset_axis = offset_axis
            .map(|name| geometry.axis_index(name))
            .transpose()
            .map_err(CalibError::Model)?;
        if let Some(i) = offset_axis
            && !geometry.axes()[i].is_rotation()
        {
            return Err(CalibError::OffsetAxisNotRotation(
                geometry.axes()[i].name().to_owned(),
            ));
        }
        let mut axes = Vec::with_capacity(frames.len());
        for (index, f) in frames.iter().enumerate() {
            if matches!(f.kind(), FrameKind::Reflection(_)) && sample.is_none() {
                return Err(CalibError::BadFrame {
                    index,
                    what: "reflection frame but the model has no sample",
                });
            }
            axes.push(geometry.values_from_degrees(f.angles()).map_err(|_| {
                CalibError::BadFrame {
                    index,
                    what: "angles do not match the geometry's axes",
                }
            })?);
        }
        Ok(Self {
            geometry,
            ub: sample.map(Sample::ub),
            offset_axis,
            axes,
        })
    }

    pub fn has_offset_axis(&self) -> bool {
        self.offset_axis.is_some()
    }

    pub fn beam(&self) -> Unit<Vector3<f64>> {
        self.geometry.beam_direction()
    }

    fn poses(&self, frame: usize, offset_deg: f64) -> Result<Poses, Error> {
        let mut values = self.axes[frame].clone();
        if let Some(i) = self.offset_axis {
            values[i] += offset_deg.to_radians();
        }
        self.geometry.poses(&values)
    }

    /// The fractional pixel (slow, fast) where `frame` lands on the plane of
    /// the module that holds its observed peak, whether or not that is on
    /// the chip: the residual geometry is the plane's linear index mapping,
    /// not the chip's extent. `None` when the peak is outside the image.
    pub fn locate(
        &self,
        detector: &PixelDetector,
        v: &Values,
        index: usize,
        frame: &CalFrame,
    ) -> Result<[f64; 2], Reject> {
        let poses = self
            .poses(index, v.get(ParamId::AngleOffset))
            .map_err(|_| Reject::NoRay)?;
        let ki = poses.source * self.geometry.beam_direction();
        let direction = match frame.kind() {
            FrameKind::PrimaryBeam => ki.into_inner(),
            FrameKind::Reflection(hkl) => {
                let lambda = v.get(ParamId::Wavelength);
                if !(lambda.is_finite() && lambda > 0.0) {
                    return Err(Reject::NoRay);
                }
                // the sample frame is where UB·hkl lives; the sample holder
                // carries it into the lab
                let ub = self.ub.expect("checked in Rig::new");
                let q_lab = poses.sample.rotation * (ub * hkl);
                (TAU / lambda) * ki.into_inner() + q_lab
            }
        };
        let to_arm = poses.detector.inverse();
        let origin: Point3<f64> = to_arm * poses.sample_position();
        let (slow, fast) = cell(frame.peak().pos).ok_or(Reject::NoRay)?;
        let module = detector.module(slow, fast).map_err(|_| Reject::NoRay)?;
        detector
            .locate_on(module, origin, to_arm.rotation * direction)
            .map_err(|e| match e {
                Error::RayBehind => Reject::RayBehind,
                Error::RayParallel => Reject::RayParallel,
                _ => Reject::NoRay,
            })
    }

    /// `(locate − peak)/σ` of one frame on the axes with a finite σ. A
    /// frame that cannot be located is `Err`, not a missing entry.
    pub fn residual(
        &self,
        detector: &PixelDetector,
        v: &Values,
        index: usize,
        frame: &CalFrame,
    ) -> Result<Vec<f64>, Reject> {
        let at = self.locate(detector, v, index, frame)?;
        let peak = frame.peak();
        Ok((0..2)
            .filter(|&a| peak.sigma[a].is_finite())
            .map(|a| (at[a] - peak.pos[a]) / peak.sigma[a])
            .collect())
    }
}

/// The image pixel (slow, fast) a peak position rounds to; `None` for a
/// position below the first pixel.
pub(crate) fn cell(pos: [f64; 2]) -> Option<(usize, usize)> {
    let r = |x: f64| (x >= -0.5).then(|| x.round() as usize);
    Some((r(pos[0])?, r(pos[1])?))
}

/// Residuals per axis with a finite σ in `frame`.
pub(crate) fn residual_count(frame: &CalFrame) -> usize {
    frame.peak().sigma.iter().filter(|s| s.is_finite()).count()
}
