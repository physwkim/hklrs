//! pyFAI PONI geometry to and from a [`PixelDetector`].
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

/// Largest |cos| between a detector's pixel axes whose shear a PONI, which
/// has none, may drop.
const MAX_SKEW_COS: f64 = 1e-9;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PoniError {
    #[error("{what} is not finite: {value}")]
    NonFinite { what: &'static str, value: f64 },
    #[error("pyFAI orientation {0} is not one of 1-4")]
    Orientation(u32),
    #[error("the up direction is along the beam (sin of their angle = {sin:e})")]
    UpAlongBeam { sin: f64 },
    #[error("a PONI needs two pixels along each axis to read the pitch, got {slow}x{fast}")]
    TooFewPixels { slow: usize, fast: usize },
    #[error("the pixel axes are not perpendicular (cos of their angle = {cos:e})")]
    Skewed { cos: f64 },
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
    /// For the slow and the fast axis: the physical index step per image
    /// index (−1 where the image runs against pyFAI's axis) and the physical
    /// index of image index 0. In f64 so that a zero shape reaches
    /// `PixelDetector::new`'s check.
    fn index_map(self, shape: (usize, usize)) -> [(f64, f64); 2] {
        let (flip1, flip2) = match self {
            Orientation::TopLeft => (true, true),
            Orientation::TopRight => (true, false),
            Orientation::BottomRight => (false, false),
            Orientation::BottomLeft => (false, true),
        };
        let map = |flip: bool, n: usize| {
            if flip {
                (-1.0, n as f64 - 1.0)
            } else {
                (1.0, 0.0)
            }
        };
        [map(flip1, shape.0), map(flip2, shape.1)]
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

fn rotation(r1: f64, r2: f64, r3: f64) -> Matrix3<f64> {
    rot3(r3) * rot2(r2) * rot1(r1)
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
        let r = frame.axes * rotation(self.rot1.0, self.rot2.0, self.rot3.0);
        let [(step1, first1), (step2, first2)] = self.orientation.index_map(self.shape);
        let origin = r * Vector3::new(
            self.pixel1 * (first1 + 0.5) - self.poni1,
            self.pixel2 * (first2 + 0.5) - self.poni2,
            self.dist,
        );
        Ok(PixelDetector::new(
            self.shape,
            origin,
            r.column(0) * (step1 * self.pixel1),
            r.column(1) * (step2 * self.pixel2),
        )?)
    }

    /// The PONI of `detector` read with `orientation`; [`Poni::detector`]
    /// of it gives `detector` back. rot2 comes out in [−π/2, π/2] and rot1,
    /// rot3 in [−π, π]; at rot2 = ±π/2, where rot1 and rot3 turn about one
    /// axis, rot3 takes what rot1 leaves.
    pub fn from_detector(
        detector: &PixelDetector,
        orientation: Orientation,
        frame: &Frame,
    ) -> Result<Poni, PoniError> {
        let shape = detector.shape();
        if shape.0 < 2 || shape.1 < 2 {
            return Err(PoniError::TooFewPixels {
                slow: shape.0,
                fast: shape.1,
            });
        }
        let p00 = detector.pixel(0, 0)?.coords;
        // across the whole detector, so the rounding of the positions is
        // divided by n − 1
        let e_slow = (detector.pixel(shape.0 - 1, 0)?.coords - p00) / (shape.0 - 1) as f64;
        let e_fast = (detector.pixel(0, shape.1 - 1)?.coords - p00) / (shape.1 - 1) as f64;
        let (pixel1, pixel2) = (e_slow.norm(), e_fast.norm());
        positive("pixel1", pixel1)?;
        positive("pixel2", pixel2)?;
        let [(step1, first1), (step2, first2)] = orientation.index_map(shape);
        let u1 = e_slow * (step1 / pixel1);
        let u2 = e_fast * (step2 / pixel2);
        let cos = u1.dot(&u2);
        if cos.is_nan() || cos.abs() > MAX_SKEW_COS {
            return Err(PoniError::Skewed { cos });
        }
        // the frame is left-handed, so axis 3 is u2 × u1
        let r = frame.axes.transpose() * Matrix3::from_columns(&[u1, u2, u2.cross(&u1)]);
        let rot2 = r[(2, 0)].atan2(r[(2, 1)].hypot(r[(2, 2)]));
        let rot1 = (-r[(2, 1)]).atan2(r[(2, 2)]);
        let r3 = r * self::rot1(rot1).transpose() * self::rot2(rot2).transpose();
        let rot3 = r3[(1, 0)].atan2(r3[(0, 0)]);
        // through the rotation the PONI holds, so the round trip closes on it
        let local = (frame.axes * rotation(rot1, rot2, rot3)).transpose() * p00;
        Ok(Poni {
            dist: local.z,
            poni1: pixel1 * (first1 + 0.5) - local.x,
            poni2: pixel2 * (first2 + 0.5) - local.y,
            rot1: Rad(rot1),
            rot2: Rad(rot2),
            rot3: Rad(rot3),
            pixel1,
            pixel2,
            shape,
            orientation,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

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

    const ORIENTATIONS: [Orientation; 4] = [
        Orientation::TopLeft,
        Orientation::TopRight,
        Orientation::BottomRight,
        Orientation::BottomLeft,
    ];

    fn oblique() -> Frame {
        let beam = Unit::new_normalize(Vector3::new(0.3, -0.5, 0.81));
        let up = Unit::new_normalize(beam.cross(&Vector3::new(0.2, 0.7, -0.4)));
        Frame::new(beam, up).unwrap()
    }

    fn poni(rot: (f64, f64, f64), orientation: Orientation) -> Poni {
        Poni {
            dist: 0.31,
            poni1: 0.0071,
            poni2: -0.0253,
            rot1: Rad(rot.0),
            rot2: Rad(rot.1),
            rot3: Rad(rot.2),
            pixel1: 7.5e-5,
            pixel2: 1.72e-4,
            shape: (213, 347),
            orientation,
        }
    }

    /// Largest distance between the same pixel of `a` and `b`; NaN if any is.
    fn gap(a: &PixelDetector, b: &PixelDetector) -> f64 {
        assert_eq!(a.shape(), b.shape());
        let (n1, n2) = a.shape();
        let mut worst = 0.0f64;
        for (i, j) in [
            (0, 0),
            (n1 - 1, 0),
            (0, n2 - 1),
            (n1 - 1, n2 - 1),
            (n1 / 3, n2 / 2),
        ] {
            let d = (a.pixel(i, j).unwrap() - b.pixel(i, j).unwrap()).norm();
            if d > worst || d.is_nan() {
                worst = d;
            }
        }
        worst
    }

    /// Largest field difference, lengths in metres and angles in radians;
    /// NaN if any is.
    fn field_gap(a: &Poni, b: &Poni) -> f64 {
        assert_eq!((a.shape, a.orientation), (b.shape, b.orientation));
        let d = [
            a.dist - b.dist,
            a.poni1 - b.poni1,
            a.poni2 - b.poni2,
            a.rot1.0 - b.rot1.0,
            a.rot2.0 - b.rot2.0,
            a.rot3.0 - b.rot3.0,
            a.pixel1 - b.pixel1,
            a.pixel2 - b.pixel2,
        ];
        d.iter()
            .map(|x| x.abs())
            .fold(0.0, |m, x| if x > m || x.is_nan() { x } else { m })
    }

    #[test]
    fn export_gives_back_the_fields_in_the_principal_ranges() {
        for frame in [
            oblique(),
            Frame::new(Vector3::x_axis(), Vector3::z_axis()).unwrap(),
        ] {
            for o in ORIENTATIONS {
                for rot in [
                    (0.0, 0.0, 0.0),
                    (0.21, 0.0, 0.0),
                    (0.0, -0.13, 0.0),
                    (0.0, 0.0, 0.37),
                    (0.21, -0.13, 0.37),
                    (2.5, -1.2, -2.9),
                    (-3.1, 1.5, 3.1),
                ] {
                    let p = poni(rot, o);
                    let back =
                        Poni::from_detector(&p.detector(&frame).unwrap(), o, &frame).unwrap();
                    let d = field_gap(&p, &back);
                    assert!(d <= 1e-12, "{o:?} {rot:?}: {d}");
                }
            }
        }
    }

    /// Angles outside the principal ranges and at rot2 = ±π/2, where rot1
    /// and rot3 share an axis, come back as another PONI of the same detector.
    #[test]
    fn import_export_import_is_the_identity() {
        let frame = oblique();
        for o in ORIENTATIONS {
            for read_as in ORIENTATIONS {
                for rot in [
                    (0.21, -0.13, 0.37),
                    (4.0, 2.0, -3.5),
                    (-7.0, 3.3, 9.0),
                    (0.4, FRAC_PI_2, -0.9),
                    (0.4, -FRAC_PI_2, -0.9),
                    (PI, 0.0, -PI),
                ] {
                    let det = poni(rot, o).detector(&frame).unwrap();
                    let back = Poni::from_detector(&det, read_as, &frame).unwrap();
                    assert_eq!(back.orientation, read_as);
                    let d = gap(&det, &back.detector(&frame).unwrap());
                    assert!(d <= 1e-14, "{o:?} read as {read_as:?}, {rot:?}: {d} m");
                }
            }
        }
    }

    #[test]
    fn export_needs_two_pixels_per_axis_and_perpendicular_axes() {
        let frame = oblique();
        let o = Orientation::BottomRight;
        let x = Vector3::new(1e-4, 0.0, 0.0);
        let y = Vector3::new(0.0, 1e-4, 0.0);
        let at = Vector3::new(0.3, 0.0, 0.0);
        for shape in [(1, 5), (5, 1)] {
            let det = PixelDetector::new(shape, at, x, y).unwrap();
            assert_eq!(
                Poni::from_detector(&det, o, &frame),
                Err(PoniError::TooFewPixels {
                    slow: shape.0,
                    fast: shape.1
                })
            );
        }
        let skewed = PixelDetector::new((5, 5), at, x, y + x * 1e-6).unwrap();
        assert!(matches!(
            Poni::from_detector(&skewed, o, &frame),
            Err(PoniError::Skewed { .. })
        ));
    }
}
