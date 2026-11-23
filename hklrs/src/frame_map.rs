//! One detector's frames into a [`Cube`] (design §8.5): the step a live map
//! and a batch reduction share. A frame's axis readbacks, wavelength and
//! raw image become per-pixel coordinates and normalised pixels on the
//! region's binned detector, which are added to a cube that grows to hold
//! them.

use crate::grid::{Binning, Cube, Projected};
use crate::normalise::{FrameScalars, NormalisedFrame, Normaliser, PixelFields, RawFrame, Region};
use crate::projection::Projection;
use crate::{Error, Geometry, PixelDetector, Wavelength};

/// What every frame of one scan shares.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameMap {
    geometry: Geometry,
    projection: Projection,
    /// The region's binned detector: both the projection and the normaliser
    /// work on its pixels.
    detector: PixelDetector,
    normaliser: Normaliser,
    /// One per dimension of `projection`.
    binnings: Vec<Binning>,
}

/// One frame on the binned detector: each pixel of [`MappedFrame::pixels`]
/// is at the same (slow, fast) pixel of [`MappedFrame::points`].
#[derive(Clone, Debug, PartialEq)]
pub struct MappedFrame {
    points: Projected,
    pixels: NormalisedFrame,
}

impl MappedFrame {
    pub fn points(&self) -> &Projected {
        &self.points
    }

    pub fn pixels(&self) -> &NormalisedFrame {
        &self.pixels
    }
}

impl FrameMap {
    /// `detector` and `fields` are the whole raw detector, which `region`
    /// cuts and bins; `binnings` has one binning per dimension of
    /// `projection`. Fields of another shape than `detector` are refused
    /// here, by [`Normaliser::new`].
    pub fn new(
        geometry: Geometry,
        projection: Projection,
        detector: &PixelDetector,
        region: &Region,
        fields: PixelFields,
        binnings: Vec<Binning>,
    ) -> Result<Self, Error> {
        let dimensions = projection.names().len();
        if binnings.len() != dimensions {
            return Err(Error::Dimensions {
                expected: dimensions,
                got: binnings.len(),
            });
        }
        let (detector, reducer) = region.apply(detector)?;
        Ok(Self {
            geometry,
            projection,
            detector,
            normaliser: Normaliser::new(fields, reducer)?,
            binnings,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The binned detector.
    pub fn detector(&self) -> &PixelDetector {
        &self.detector
    }

    /// The raw frame's (slow, fast) shape.
    pub fn raw_shape(&self) -> (usize, usize) {
        self.normaliser.raw_shape()
    }

    pub fn binnings(&self) -> &[Binning] {
        &self.binnings
    }

    /// One frame: `readbacks` in the geometry's axis order, degrees for
    /// rotations and metres for translations; `raw` the whole raw image, of
    /// the raw shape.
    pub fn map(
        &self,
        readbacks: &[f64],
        wavelength: Wavelength,
        raw: &RawFrame,
        scalars: &FrameScalars,
    ) -> Result<MappedFrame, Error> {
        let values = self.geometry.values_from_degrees(readbacks)?;
        let pixels = self.normaliser.normalise(raw, scalars, &[])?;
        let points =
            self.projection
                .project(&self.geometry, &values, wavelength, &self.detector)?;
        Ok(MappedFrame { points, pixels })
    }

    /// Adds every unmasked pixel of `frame` to `cube`, growing it to hold
    /// them.
    pub fn add(&self, cube: &mut Cube, frame: &MappedFrame) -> Result<(), Error> {
        cube.add_growing(&self.binnings, &frame.points, &frame.pixels)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::presets::preset;

    /// K6C mu, komega, kappa, kphi, gamma, delta, degrees.
    const READBACKS: [f64; 6] = [3.0, 20.0, -30.0, 40.0, 12.0, 25.0];

    fn raw_detector() -> PixelDetector {
        PixelDetector::new(
            (4, 6),
            Vector3::new(0.5, -0.003, -0.002),
            Vector3::new(0.0, 0.0, 0.001),
            Vector3::new(0.0, 0.001, 0.0),
        )
        .unwrap()
    }

    fn binnings(n: usize) -> Vec<Binning> {
        vec![Binning::new(0.01, 0.0).unwrap(); n]
    }

    #[test]
    fn binnings_must_match_the_projection() {
        let e = FrameMap::new(
            preset("K6C").unwrap(),
            Projection::QparQper {
                normal: Vector3::z_axis(),
            },
            &raw_detector(),
            &Region::new(0..4, 0..6, (1, 1)),
            PixelFields::new((4, 6)),
            binnings(3),
        )
        .unwrap_err();
        assert!(
            matches!(
                e,
                Error::Dimensions {
                    expected: 2,
                    got: 3
                }
            ),
            "{e:?}"
        );
    }

    /// A 2×2-binned roi: the coordinates are those of the binned detector's
    /// pixels and the counts its blocks' sums, so the cube holds the roi's
    /// raw counts and no trailing pixel.
    #[test]
    fn a_frame_is_mapped_on_the_binned_detector() {
        let geometry = preset("K6C").unwrap();
        let wavelength = Wavelength::new(1.54).unwrap();
        let region = Region::new(1..4, 1..6, (2, 2));
        let map = FrameMap::new(
            geometry.clone(),
            Projection::QSample,
            &raw_detector(),
            &region,
            PixelFields::new((4, 6)),
            binnings(3),
        )
        .unwrap();
        let raw: Vec<f64> = (0..24).map(f64::from).collect();
        let frame = map
            .map(
                &READBACKS,
                wavelength,
                &RawFrame::new((4, 6), raw.clone()).unwrap(),
                &FrameScalars::new(),
            )
            .unwrap();

        let (binned, _) = region.apply(&raw_detector()).unwrap();
        assert_eq!(map.detector(), &binned);
        let values = geometry.values_from_degrees(&READBACKS).unwrap();
        let points = Projection::QSample
            .project(&geometry, &values, wavelength, &binned)
            .unwrap();
        assert_eq!(frame.points(), &points);

        // blocks of rows 1-2 at columns 1-2 and 3-4; row 3 and column 5
        // fill none
        let block = |s: usize, f: usize| raw[s * 6 + f] + raw[s * 6 + f + 1];
        let sums = [block(1, 1) + block(2, 1), block(1, 3) + block(2, 3)];
        let counts: Vec<f64> = frame
            .pixels()
            .data()
            .iter()
            .map(|p| p.unwrap().counts())
            .collect();
        assert_eq!(counts, sums);

        let mut cube = Cube::default();
        map.add(&mut cube, &frame).unwrap();
        assert_eq!(cube.photons().iter().sum::<f64>(), sums[0] + sums[1]);
        assert_eq!(cube.contributions().iter().sum::<u64>(), 2);
    }

    /// Fields of another raw shape, which each frame used to refuse, are
    /// refused before the first: another pixel count, and the transposed
    /// shape of the same count.
    #[test]
    fn fields_of_another_raw_shape_are_refused_at_setup() {
        for shape in [(3, 6), (6, 4)] {
            let e = FrameMap::new(
                preset("K6C").unwrap(),
                Projection::QSample,
                &raw_detector(),
                &Region::new(1..4, 1..6, (2, 2)),
                PixelFields::new(shape),
                binnings(3),
            )
            .unwrap_err();
            assert_eq!(
                e,
                Error::ShapeMismatch {
                    what: "pixel fields",
                    shape,
                    expected: (4, 6),
                }
            );
        }
    }

    /// One readback per axis, and a raw frame of the raw shape: the
    /// transposed shape of the same pixel count is refused as much as
    /// another count.
    #[test]
    fn readbacks_are_one_per_axis_and_raw_the_whole_frame() {
        let map = FrameMap::new(
            preset("K6C").unwrap(),
            Projection::QSample,
            &raw_detector(),
            &Region::new(0..4, 0..6, (1, 1)),
            PixelFields::new((4, 6)),
            binnings(3),
        )
        .unwrap();
        assert_eq!(map.raw_shape(), (4, 6));
        let wavelength = Wavelength::new(1.54).unwrap();
        let scalars = FrameScalars::new();
        let frame = |shape: (usize, usize)| RawFrame::new(shape, vec![0.0; 24]).unwrap();
        let e = map
            .map(&READBACKS[..5], wavelength, &frame((4, 6)), &scalars)
            .unwrap_err();
        assert!(
            matches!(
                e,
                Error::AxisCount {
                    expected: 6,
                    got: 5
                }
            ),
            "{e:?}"
        );
        assert!(
            map.map(&READBACKS, wavelength, &frame((4, 6)), &scalars)
                .is_ok()
        );
        for shape in [(6, 4), (2, 12)] {
            let e = map
                .map(&READBACKS, wavelength, &frame(shape), &scalars)
                .unwrap_err();
            assert_eq!(
                e,
                Error::ShapeMismatch {
                    what: "raw frame",
                    shape,
                    expected: (4, 6)
                }
            );
        }
    }
}
