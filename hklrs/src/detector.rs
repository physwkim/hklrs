//! Pixel detector in the detector-arm frame (all axes at zero), metres.
//! A point detector is one pixel.

use nalgebra::{Point3, Unit, Vector3};

use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct PixelDetector {
    shape: (usize, usize),
    origin: Point3<f64>,
    e_slow: Vector3<f64>,
    e_fast: Vector3<f64>,
}

fn finite(v: &Vector3<f64>, what: &str) -> Result<(), Error> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::ZeroVector {
            what: what.to_owned(),
        })
    }
}

impl PixelDetector {
    /// `origin` is the centre of pixel (0, 0); `e_slow`/`e_fast` are one-pixel
    /// steps, so their lengths are the pixel pitches.
    pub fn new(
        shape: (usize, usize),
        origin: Vector3<f64>,
        e_slow: Vector3<f64>,
        e_fast: Vector3<f64>,
    ) -> Result<Self, Error> {
        if shape.0 == 0 || shape.1 == 0 {
            return Err(Error::DetectorShape {
                slow: shape.0,
                fast: shape.1,
            });
        }
        finite(&origin, "detector origin")?;
        finite(&e_slow, "e_slow")?;
        finite(&e_fast, "e_fast")?;
        Ok(Self {
            shape,
            origin: Point3::from(origin),
            e_slow,
            e_fast,
        })
    }

    /// One pixel at `distance` metres along `direction`.
    pub fn point(direction: Unit<Vector3<f64>>, distance: f64) -> Result<Self, Error> {
        if !(distance.is_finite() && distance > 0.0) {
            return Err(Error::NotPositive {
                what: "detector distance",
                value: distance,
            });
        }
        Self::new(
            (1, 1),
            direction.into_inner() * distance,
            Vector3::zeros(),
            Vector3::zeros(),
        )
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Centre of pixel (slow, fast) in the detector-arm frame.
    pub fn pixel(&self, slow: usize, fast: usize) -> Result<Point3<f64>, Error> {
        if slow >= self.shape.0 || fast >= self.shape.1 {
            return Err(Error::PixelOutside {
                slow,
                fast,
                shape: self.shape,
            });
        }
        Ok(self.origin + self.e_slow * slow as f64 + self.e_fast * fast as f64)
    }
}
