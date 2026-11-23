//! pyFAI PONI geometry to a [`PixelDetector`].
//!
//! pyFAI (2026.9.0, `Geometry.calc_pos_zyx`) puts pixel (d1, d2) at
//! `R · (p1 − poni1, p2 − poni2, dist)`, components along its axes 1, 2, 3,
//! with `p = pixel · (physical index + ½)` and `R = R3(rot3)·R2(rot2)·R1(rot1)`
//! (`Geometry.rotation_matrix`). The orientation maps the image index to
//! the physical index (`Detector._reorder_indexes_from_orientation`).

use nalgebra::{Matrix3, Unit, Vector3};
use thiserror::Error;

use crate::{Error, PixelDetector, Rad};

/// Smallest sine of the angle between `up` and the beam for them to define
/// pyFAI's axis 1.
const UP_BEAM_MIN_SIN: f64 = 1e-6;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PoniError {
    #[error("{what} is not finite: {value}")]
    NonFinite { what: &'static str, value: f64 },
    #[error("pyFAI orientation {0} is not one of 1-4")]
    Orientation(u32),
    #[error("the up direction is along the beam (sin of their angle = {sin:e})")]
    UpAlongBeam { sin: f64 },
    #[error(transparent)]
    Core(#[from] Error),
}

/// Image orientation (pyFAI `detectors.orientation.Orientation`, EXIF
/// numbering); pyFAI supports 1-4 and calls 3 its native one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    TopLeft = 1,
    TopRight = 2,
    BottomRight = 3,
    BottomLeft = 4,
}

impl Orientation {
    /// Whether the slow and the fast image index run against pyFAI's
    /// physical axes 1 and 2.
    fn flips(self) -> (bool, bool) {
        match self {
            Orientation::TopLeft => (true, true),
            Orientation::TopRight => (true, false),
            Orientation::BottomRight => (false, false),
            Orientation::BottomLeft => (false, true),
        }
    }
}

impl TryFrom<u32> for Orientation {
    type Error = PoniError;

    fn try_from(value: u32) -> Result<Self, PoniError> {
        match value {
            1 => Ok(Orientation::TopLeft),
            2 => Ok(Orientation::TopRight),
            3 => Ok(Orientation::BottomRight),
            4 => Ok(Orientation::BottomLeft),
            _ => Err(PoniError::Orientation(value)),
        }
    }
}

/// Lab directions of pyFAI's axes: axis 3 along the incident beam, axis 1
/// up (pyFAI `doc/source/geometry.rst`, "Default geometry") and axis 2 =
/// axis 1 × axis 3, because pyFAI's (1, 2, 3) is left-handed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Columns: the lab directions of pyFAI axes 1, 2, 3.
    axes: Matrix3<f64>,
}

impl Frame {
    /// Axis 1 is the part of `up` normal to `beam`, so `up` need not be
    /// exactly perpendicular to a tilted beam.
    pub fn new(beam: Unit<Vector3<f64>>, up: Unit<Vector3<f64>>) -> Result<Self, PoniError> {
        let sin = beam.cross(&up).norm();
        // NaN when a component is not finite
        if sin.is_nan() || sin < UP_BEAM_MIN_SIN {
            return Err(PoniError::UpAlongBeam { sin });
        }
        let a1 = (up.into_inner() - beam.into_inner() * beam.dot(&up)).normalize();
        let a3 = beam.into_inner();
        Ok(Self {
            axes: Matrix3::from_columns(&[a1, a1.cross(&a3), a3]),
        })
    }
}

/// pyFAI's rotations as `Geometry.rotation_matrix` writes them, in its axis
/// components.
fn rot1(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c)
}

fn rot2(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c)
}

fn rot3(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// A pyFAI detector geometry. Lengths in metres; `shape` is (slow, fast),
/// pyFAI's (dim1, dim2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poni {
    pub dist: f64,
    pub poni1: f64,
    pub poni2: f64,
    pub rot1: Rad,
    pub rot2: Rad,
    pub rot3: Rad,
    pub pixel1: f64,
    pub pixel2: f64,
    pub shape: (usize, usize),
    pub orientation: Orientation,
}

fn finite(what: &'static str, value: f64) -> Result<(), PoniError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PoniError::NonFinite { what, value })
    }
}

fn positive(what: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::NotPositive { what, value })
    }
}

impl Poni {
    /// The detector in the lab frame that `frame` places pyFAI's axes in,
    /// positions relative to the sample.
    pub fn detector(&self, frame: &Frame) -> Result<PixelDetector, PoniError> {
        for (what, v) in [
            ("dist", self.dist),
            ("poni1", self.poni1),
            ("poni2", self.poni2),
            ("rot1", self.rot1.0),
            ("rot2", self.rot2.0),
            ("rot3", self.rot3.0),
        ] {
            finite(what, v)?;
        }
        positive("pixel1", self.pixel1)?;
        positive("pixel2", self.pixel2)?;
        let r = frame.axes * rot3(self.rot3.0) * rot2(self.rot2.0) * rot1(self.rot1.0);
        let (flip1, flip2) = self.orientation.flips();
        // image index 0 sits at physical index n − 1 on a flipped axis; f64
        // so that a zero shape reaches PixelDetector::new's check
        let first = |flip: bool, n: usize| if flip { n as f64 - 1.0 } else { 0.0 };
        let step = |flip: bool| if flip { -1.0 } else { 1.0 };
        let origin = r * Vector3::new(
            self.pixel1 * (first(flip1, self.shape.0) + 0.5) - self.poni1,
            self.pixel2 * (first(flip2, self.shape.1) + 0.5) - self.poni2,
            self.dist,
        );
        Ok(PixelDetector::new(
            self.shape,
            origin,
            r.column(0) * (step(flip1) * self.pixel1),
            r.column(1) * (step(flip2) * self.pixel2),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_numbers_are_1_to_4() {
        for n in 1..=4 {
            assert_eq!(Orientation::try_from(n).unwrap() as u32, n);
        }
        for n in [0, 5, 8] {
            assert_eq!(Orientation::try_from(n), Err(PoniError::Orientation(n)));
        }
    }

    #[test]
    fn frame_takes_the_part_of_up_normal_to_the_beam_and_is_left_handed() {
        let x = Vector3::x_axis();
        let tilted = Unit::new_normalize(Vector3::new(0.3, 0.0, 1.0));
        let f = Frame::new(x, tilted).unwrap();
        assert_eq!(f.axes.column(0), Vector3::z());
        assert_eq!(f.axes.column(1), Vector3::y());
        assert_eq!(f.axes.column(2), Vector3::x());
        assert!(matches!(
            Frame::new(x, x),
            Err(PoniError::UpAlongBeam { .. })
        ));
    }
}
