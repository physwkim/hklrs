//! Pixel detector in the detector-arm frame (all axes at zero), metres.
//! A point detector is one pixel and a linear detector is one row.

use std::ops::Range;

use nalgebra::{Point3, Unit, Vector3};

use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct PixelDetector {
    shape: (usize, usize),
    origin: Point3<f64>,
    e_slow: Vector3<f64>,
    e_fast: Vector3<f64>,
    /// Row-major over `shape`, `Some` only when a pixel is masked, so an
    /// all-clear mask and no mask are one value.
    mask: Option<Vec<bool>>,
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

fn any_masked(masked: Vec<bool>) -> Option<Vec<bool>> {
    masked.contains(&true).then_some(masked)
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
            mask: None,
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

    /// `masked[slow * shape.1 + fast]` excludes pixel (slow, fast); replaces
    /// any earlier mask.
    pub fn with_mask(self, masked: Vec<bool>) -> Result<Self, Error> {
        if self.shape.0.checked_mul(self.shape.1) != Some(masked.len()) {
            return Err(Error::MaskShape {
                len: masked.len(),
                shape: self.shape,
            });
        }
        Ok(Self {
            mask: any_masked(masked),
            ..self
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Row-major mask over the shape, `None` when no pixel is masked.
    pub fn mask(&self) -> Option<&[bool]> {
        self.mask.as_deref()
    }

    fn index(&self, slow: usize, fast: usize) -> Result<usize, Error> {
        if slow >= self.shape.0 || fast >= self.shape.1 {
            return Err(Error::PixelOutside {
                slow,
                fast,
                shape: self.shape,
            });
        }
        Ok(slow * self.shape.1 + fast)
    }

    /// Centre of pixel (slow, fast) in the detector-arm frame.
    pub fn pixel(&self, slow: usize, fast: usize) -> Result<Point3<f64>, Error> {
        self.index(slow, fast)?;
        Ok(self.origin + self.e_slow * slow as f64 + self.e_fast * fast as f64)
    }

    pub fn is_masked(&self, slow: usize, fast: usize) -> Result<bool, Error> {
        let i = self.index(slow, fast)?;
        Ok(self.mask.as_ref().is_some_and(|m| m[i]))
    }

    /// Pixels `slow` x `fast` as a detector of their own, whose pixel (0, 0)
    /// is this detector's (slow.start, fast.start).
    pub fn roi(&self, slow: Range<usize>, fast: Range<usize>) -> Result<Self, Error> {
        if slow.is_empty() || fast.is_empty() || slow.end > self.shape.0 || fast.end > self.shape.1
        {
            return Err(Error::Roi {
                slow,
                fast,
                shape: self.shape,
            });
        }
        let det = Self::new(
            (slow.len(), fast.len()),
            self.pixel(slow.start, fast.start)?.coords,
            self.e_slow,
            self.e_fast,
        )?;
        let width = self.shape.1;
        let mask = self.mask.as_ref().map(|m| {
            slow.flat_map(|s| fast.clone().map(move |f| m[s * width + f]))
                .collect()
        });
        Ok(Self {
            mask: mask.and_then(any_masked),
            ..det
        })
    }

    /// Blocks of `slow` x `fast` pixels as one pixel each, centred on the
    /// block. Trailing pixels that fill no block are dropped; a block is
    /// masked when any of its pixels is.
    pub fn bin(&self, slow: usize, fast: usize) -> Result<Self, Error> {
        if slow == 0 || fast == 0 || slow > self.shape.0 || fast > self.shape.1 {
            return Err(Error::Bin {
                slow,
                fast,
                shape: self.shape,
            });
        }
        let shape = (self.shape.0 / slow, self.shape.1 / fast);
        let half = |n: usize| (n - 1) as f64 / 2.0;
        let det = Self::new(
            shape,
            self.origin.coords + self.e_slow * half(slow) + self.e_fast * half(fast),
            self.e_slow * slow as f64,
            self.e_fast * fast as f64,
        )?;
        let width = self.shape.1;
        let mask = self.mask.as_ref().map(|m| {
            (0..shape.0)
                .flat_map(|bs| {
                    (0..shape.1).map(move |bf| {
                        (bs * slow..(bs + 1) * slow)
                            .any(|s| (bf * fast..(bf + 1) * fast).any(|f| m[s * width + f]))
                    })
                })
                .collect()
        });
        Ok(Self {
            mask: mask.and_then(any_masked),
            ..det
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> PixelDetector {
        PixelDetector::new(
            (5, 7),
            Vector3::new(0.3, -0.02, 0.01),
            Vector3::new(1e-3, 2e-4, -3e-4),
            Vector3::new(-1e-4, 7e-4, 5e-4),
        )
        .unwrap()
    }

    fn close(a: Point3<f64>, b: Point3<f64>) -> bool {
        (a - b).norm() <= 1e-15
    }

    #[test]
    fn roi_pixel_is_the_parent_pixel_offset_by_the_roi_start() {
        let det = area();
        let r = det.roi(1..4, 2..7).unwrap();
        assert_eq!(r.shape(), (3, 5));
        for s in 0..3 {
            for f in 0..5 {
                assert!(close(
                    r.pixel(s, f).unwrap(),
                    det.pixel(s + 1, f + 2).unwrap()
                ));
            }
        }
    }

    #[test]
    fn roi_rejects_empty_and_outside_ranges() {
        let det = area();
        for (slow, fast) in [(2..2, 0..7), (0..5, 3..3), (0..6, 0..7), (0..5, 1..8)] {
            assert!(matches!(det.roi(slow, fast), Err(Error::Roi { .. })));
        }
    }

    #[test]
    fn bin_pixel_is_its_block_centre_and_drops_the_remainder() {
        let det = area();
        let b = det.bin(2, 3).unwrap();
        assert_eq!(b.shape(), (2, 2));
        for bs in 0..2 {
            for bf in 0..2 {
                let sum = (0..2)
                    .flat_map(|s| (0..3).map(move |f| (2 * bs + s, 3 * bf + f)))
                    .map(|(s, f)| det.pixel(s, f).unwrap().coords)
                    .sum::<Vector3<f64>>();
                assert!(close(b.pixel(bs, bf).unwrap(), Point3::from(sum / 6.0)));
            }
        }
        assert_eq!(det.bin(1, 1).unwrap(), det);
    }

    #[test]
    fn bin_rejects_zero_and_oversized_factors() {
        let det = area();
        for (s, f) in [(0, 1), (1, 0), (6, 1), (1, 8)] {
            assert!(matches!(det.bin(s, f), Err(Error::Bin { .. })));
        }
    }

    #[test]
    fn a_linear_detector_is_one_row() {
        let det = PixelDetector::new(
            (1, 9),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 1e-3),
        )
        .unwrap();
        let b = det.roi(0..1, 1..9).unwrap().bin(1, 3).unwrap();
        assert_eq!(b.shape(), (1, 2));
        assert!(close(b.pixel(0, 1).unwrap(), det.pixel(0, 5).unwrap()));
        assert!(matches!(det.bin(2, 1), Err(Error::Bin { .. })));
    }

    #[test]
    fn mask_follows_roi_and_bin() {
        let mut m = vec![false; 35];
        m[2 * 7 + 3] = true;
        let det = area().with_mask(m).unwrap();
        assert!(det.is_masked(2, 3).unwrap());
        assert!(!det.is_masked(2, 4).unwrap());
        assert!(matches!(
            det.is_masked(5, 0),
            Err(Error::PixelOutside { .. })
        ));

        let r = det.roi(1..5, 2..6).unwrap();
        assert!(r.is_masked(1, 1).unwrap());
        assert_eq!(r.mask().unwrap().iter().filter(|&&x| x).count(), 1);
        // the roi leaves the masked pixel out: no mask remains
        assert_eq!(det.roi(3..5, 0..7).unwrap().mask(), None);

        // block (1, 1) of 2x3 holds pixel (2, 3); every other block is clear
        let b = det.bin(2, 3).unwrap();
        assert_eq!(b.mask().unwrap(), &[false, false, false, true]);
    }

    #[test]
    fn an_all_clear_mask_is_no_mask() {
        let det = area();
        assert_eq!(det.clone().with_mask(vec![false; 35]).unwrap(), det);
        assert!(matches!(
            det.with_mask(vec![true; 34]),
            Err(Error::MaskShape { len: 34, .. })
        ));
    }
}
