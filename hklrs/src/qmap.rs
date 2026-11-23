//! Batch forward: one frame's axis values over a whole detector.
//!
//! Every pixel goes through [`scattering_at`], the kernel [`crate::forward`]
//! uses; the map only decides which part of its result to store.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::{Error, Geometry, PixelDetector, Poses, Scattering, Wavelength, scattering_at};

/// Axis values (axis order, radians / metres) and wavelength of one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub values: Vec<f64>,
    pub wavelength: Wavelength,
}

/// Three contiguous f64 planes (x, y, z), row-major (slow, fast).
#[derive(Clone, Debug, PartialEq)]
pub struct Planes {
    shape: (usize, usize),
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// Q in the sample frame per pixel, Å⁻¹ with 2π.
pub type QMap = Planes;

impl Planes {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The vector at pixel (slow, fast); panics outside the shape.
    pub fn get(&self, slow: usize, fast: usize) -> Vector3<f64> {
        assert!(slow < self.shape.0 && fast < self.shape.1);
        let i = slow * self.shape.1 + fast;
        Vector3::new(self.x[i], self.y[i], self.z[i])
    }
}

/// Fills `N` planes of `detector`'s shape, one rayon task per row. `f` sees
/// the frame's poses, the lab-frame sample-to-pixel vector and the kernel's
/// [`Scattering`] for each pixel.
fn fill<const N: usize>(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
    f: impl Fn(&Poses, &Vector3<f64>, &Scattering) -> [f64; N] + Sync,
) -> Result<[Vec<f64>; N], Error> {
    let (slow, fast) = detector.shape();
    let poses = geometry.poses(&frame.values)?;
    let mut planes: [Vec<f64>; N] = std::array::from_fn(|_| vec![0.0; slow * fast]);
    let mut chunks: Vec<_> = planes.iter_mut().map(|p| p.chunks_mut(fast)).collect();
    let rows: Vec<[&mut [f64]; N]> = (0..slow)
        .map(|_| std::array::from_fn(|k| chunks[k].next().expect("one chunk per row")))
        .collect();
    rows.into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..fast {
                let point: Point3<f64> = detector.pixel(i, j)?;
                let (p, s) = scattering_at(geometry, &poses, frame.wavelength, &point)?;
                for (plane, v) in row.iter_mut().zip(f(&poses, &p, &s)) {
                    plane[j] = v;
                }
            }
            Ok(())
        })?;
    Ok(planes)
}

fn planes3(shape: (usize, usize), [x, y, z]: [Vec<f64>; 3]) -> Planes {
    Planes { shape, x, y, z }
}

/// Q in the sample frame for every pixel.
pub fn q_map(geometry: &Geometry, frame: &Frame, detector: &PixelDetector) -> Result<QMap, Error> {
    let planes = fill(geometry, frame, detector, |_, _, s| s.q_sample.into())?;
    Ok(planes3(detector.shape(), planes))
}

/// Pixel positions relative to the sample, lab frame, metres: the `p` of
/// the forward kernel.
pub fn detector_positions(
    geometry: &Geometry,
    frame: &Frame,
    detector: &PixelDetector,
) -> Result<Planes, Error> {
    let planes = fill(geometry, frame, detector, |_, p, _| (*p).into())?;
    Ok(planes3(detector.shape(), planes))
}

#[cfg(test)]
mod tests {
    use nalgebra::Unit;

    use super::*;
    use crate::{Lattice, Orientation, Sample, forward};

    /// Sample ω about z, detector 2θ about z and ν about y, a detector
    /// translation along x: every pose component enters p.
    fn geometry() -> Geometry {
        Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .detector_translation("dx", [1.0, 0.0, 0.0])
            .detector_rotation("tt", [0.0, 0.0, 1.0])
            .detector_rotation("nu", [0.0, 1.0, 0.0])
            .build()
            .unwrap()
    }

    fn frame() -> Frame {
        Frame {
            values: vec![0.31, 0.002, 0.57, -0.23],
            wavelength: Wavelength::new(1.54).unwrap(),
        }
    }

    fn detector() -> PixelDetector {
        PixelDetector::new(
            (5, 7),
            Vector3::new(0.3, -0.01, -0.02),
            Vector3::new(0.0, 0.0, 1e-3),
            Vector3::new(0.0, 2e-3, 0.0),
        )
        .unwrap()
    }

    fn sample() -> Sample {
        Sample::new(
            "s",
            Lattice::cubic(4.0).unwrap(),
            Orientation::from_u_matrix(&nalgebra::Matrix3::identity()).unwrap(),
        )
    }

    #[test]
    fn batch_equals_per_pixel_forward_exactly() {
        let (g, f, d) = (geometry(), frame(), detector());
        let map = q_map(&g, &f, &d).unwrap();
        assert_eq!(map.shape(), (5, 7));
        for i in 0..5 {
            for j in 0..7 {
                let one = forward(
                    &g,
                    &sample(),
                    &f.values,
                    f.wavelength,
                    &d.pixel(i, j).unwrap(),
                )
                .unwrap();
                let got = map.get(i, j);
                assert!(got.iter().all(|v| v.is_finite()));
                assert_eq!(got, one.scattering.q_sample, "pixel ({i}, {j})");
            }
        }
    }

    #[test]
    fn position_direction_is_kf_and_p_is_the_kernels() {
        let (g, f, d) = (geometry(), frame(), detector());
        let pos = detector_positions(&g, &f, &d).unwrap();
        let poses = g.poses(&f.values).unwrap();
        for i in 0..5 {
            for j in 0..7 {
                let one = forward(
                    &g,
                    &sample(),
                    &f.values,
                    f.wavelength,
                    &d.pixel(i, j).unwrap(),
                )
                .unwrap();
                let p = pos.get(i, j);
                let want = poses.detector * d.pixel(i, j).unwrap() - poses.sample_position();
                assert_eq!(p, want);
                let dir = Unit::new_normalize(p);
                let diff = (dir.into_inner() - one.scattering.kf.into_inner()).norm();
                assert!(diff < 1e-15, "pixel ({i}, {j}): {diff:e}");
            }
        }
    }

    #[test]
    fn a_pixel_on_the_sample_is_an_error() {
        let g = Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        let d = PixelDetector::new(
            (2, 2),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert_eq!(q_map(&g, &f, &d), Err(Error::DetectorAtSample));
    }
}
