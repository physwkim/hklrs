//! Intensity normalisation (design §5, §7): per-frame scalars and per-pixel
//! fields as a `(counts, weight)` pair per pixel, so a consumer chooses how
//! a bin estimates intensity (`Σcounts / Σweight`, or the mean of
//! `counts / weight`). The corrected value of a pixel is `counts / weight`.
//!
//! `counts = raw − dark_rate · count_time`, signed: xrayutilities clamps at
//! 0, which biases low counts upwards. `weight` is the product of every
//! factor that scales the intensity: `monitor / reference`, `count_time`,
//! the absorber's transmission, the flat field relative to its mean, and
//! the geometric weights the caller passes (solid angle, polarisation,
//! absorption).
//!
//! Dark, flat, monitor, count time and transmission are independent stages
//! that are applied when given and skipped when not, for any detector
//! shape; there is no branch on the data's dimension.
//!
//! A bin of `bn.0 × bn.1` raw pixels sums the counts of its pixels and the
//! weights they would have: the caller's geometric weights are those of the
//! binned pixel, which spans the block, so each raw pixel carries `1/n` of
//! it, and the flat field enters per raw pixel. `Σcounts / Σweight` of the
//! block is then the ratio of sums over its pixels, and a consumer that
//! sums bins reaches the ratio of sums over any set of pixels.
//!
//! Where a pixel is masked is decided in [`Normaliser::normalise`] and only
//! there, in this order: the detector's mask, a geometric weight the caller
//! passed without a value, a flat field without a usable value, a count
//! that is not finite, a total weight that is not positive and finite.

use std::ops::Range;

use crate::absorption::{Masked, Weight};
use crate::{Error, PixelDetector, Weights};

/// A flat-field pixel at or below this, or not finite, has no gain to
/// divide by: the pixel is masked [`Masked::BadFlat`] (xrayutilities sets
/// it to 1).
pub const FLAT_MIN: f64 = 1e-5;

/// The transmitted fraction `T > 0` of the beam through an absorber
/// (attenuator foils) for one frame's record `D`. Any `Fn(&D) -> f64`
/// is one, so closures, function items and partial applications all work;
/// xrayutilities's `absfun` is `1 / T`.
pub trait Transmission<D: ?Sized> {
    fn transmission(&self, data: &D) -> f64;
}

impl<D: ?Sized, F: Fn(&D) -> f64> Transmission<D> for F {
    fn transmission(&self, data: &D) -> f64 {
        self(data)
    }
}

fn positive(what: &'static str, value: f64) -> Result<f64, Error> {
    if value > 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err(Error::NotPositive { what, value })
    }
}

/// The mean of a scan's monitor, xrayutilities's default `av_mon`.
pub fn mean_monitor(monitor: &[f64]) -> Result<f64, Error> {
    let mut sum = 0.0;
    for &m in monitor {
        sum += positive("monitor", m)?;
    }
    positive("monitor mean", sum / monitor.len() as f64)
}

/// What one frame contributes to every pixel's weight. A value that is not
/// positive and finite is an [`Error`] where it is given, never replaced
/// by 1 (xrayutilities does that for a zero or NaN factor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameScalars {
    monitor: Option<f64>,
    count_time: Option<f64>,
    transmission: f64,
}

impl Default for FrameScalars {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameScalars {
    /// No monitor, count time or absorber: all three factors are 1.
    pub fn new() -> Self {
        Self {
            monitor: None,
            count_time: None,
            transmission: 1.0,
        }
    }

    /// The frame's monitor counts against the `reference` they are scaled
    /// to ([`mean_monitor`] of the scan, or a fixed value).
    pub fn monitor(self, value: f64, reference: f64) -> Result<Self, Error> {
        let ratio = positive("monitor", value)? / positive("monitor reference", reference)?;
        Ok(Self {
            monitor: Some(positive("monitor / reference", ratio)?),
            ..self
        })
    }

    /// The count time, s; the dark field is a rate per second.
    pub fn count_time(self, seconds: f64) -> Result<Self, Error> {
        Ok(Self {
            count_time: Some(positive("count time", seconds)?),
            ..self
        })
    }

    pub fn transmission(self, t: f64) -> Result<Self, Error> {
        Ok(Self {
            transmission: positive("transmission", t)?,
            ..self
        })
    }

    /// The transmission of `absorber` on this frame's `data`.
    pub fn absorber<D: ?Sized>(
        self,
        absorber: &impl Transmission<D>,
        data: &D,
    ) -> Result<Self, Error> {
        self.transmission(absorber.transmission(data))
    }

    /// `monitor / reference · count_time · transmission`.
    fn factor(&self) -> f64 {
        self.monitor.unwrap_or(1.0) * self.count_time.unwrap_or(1.0) * self.transmission
    }
}

/// The detector's per-pixel fields, on the whole raw frame, before any
/// roi or binning.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelFields {
    shape: (usize, usize),
    dark_rate: Option<Vec<f64>>,
    /// The flat field over its mean, `None` where it has no usable value.
    flat: Option<Vec<Option<f64>>>,
}

impl PixelFields {
    /// No dark and no flat field.
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            shape,
            dark_rate: None,
            flat: None,
        }
    }

    fn check(&self, what: &'static str, len: usize) -> Result<(), Error> {
        let expected = self.shape.0 * self.shape.1;
        if len == expected {
            Ok(())
        } else {
            Err(Error::PixelCount {
                what,
                len,
                expected,
            })
        }
    }

    /// Dark counts per second, row-major over the raw shape.
    pub fn dark_rate(self, rate: Vec<f64>) -> Result<Self, Error> {
        self.check("dark field", rate.len())?;
        if let Some(index) = rate.iter().position(|r| !r.is_finite()) {
            return Err(Error::NonFinite { index });
        }
        Ok(Self {
            dark_rate: Some(rate),
            ..self
        })
    }

    /// Relative gain, row-major over the raw shape. The mean is taken over
    /// the usable pixels (finite and above [`FLAT_MIN`]); the others are
    /// masked.
    pub fn flat(self, gain: Vec<f64>) -> Result<Self, Error> {
        self.check("flat field", gain.len())?;
        let usable = |g: f64| g.is_finite() && g > FLAT_MIN;
        let (sum, n) = gain
            .iter()
            .filter(|&&g| usable(g))
            .fold((0.0, 0usize), |(s, n), &g| (s + g, n + 1));
        if n == 0 {
            return Err(Error::NotPositive {
                what: "flat field mean",
                value: 0.0,
            });
        }
        let mean = sum / n as f64;
        Ok(Self {
            flat: Some(
                gain.iter()
                    .map(|&g| usable(g).then_some(g / mean))
                    .collect(),
            ),
            ..self
        })
    }
}

/// A roi and a block size on the raw frame. [`Region::apply`] cuts the
/// detector and the data with the same numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub slow: Range<usize>,
    pub fast: Range<usize>,
    /// Block size (slow, fast).
    pub bin: (usize, usize),
}

impl Region {
    pub fn new(slow: Range<usize>, fast: Range<usize>, bin: (usize, usize)) -> Self {
        Self { slow, fast, bin }
    }

    /// `detector.roi(..).bin(..)` and the [`Reducer`] of the same blocks:
    /// binned pixel `(i, j)` is the block of raw pixels starting at
    /// `(slow.start + i·bin.0, fast.start + j·bin.1)`, so its centre is the
    /// mean of their centres and the data it holds is theirs. Trailing
    /// pixels that fill no block are in neither.
    pub fn apply(&self, detector: &PixelDetector) -> Result<(PixelDetector, Reducer), Error> {
        let binned = detector
            .roi(self.slow.clone(), self.fast.clone())?
            .bin(self.bin.0, self.bin.1)?;
        let reducer = Reducer {
            raw_shape: detector.shape(),
            start: (self.slow.start, self.fast.start),
            bin: self.bin,
            shape: binned.shape(),
            mask: binned.mask().map(<[bool]>::to_vec),
        };
        Ok((binned, reducer))
    }
}

/// The blocks of one [`Region`]; only [`Region::apply`] makes one.
#[derive(Clone, Debug, PartialEq)]
pub struct Reducer {
    raw_shape: (usize, usize),
    start: (usize, usize),
    bin: (usize, usize),
    shape: (usize, usize),
    mask: Option<Vec<bool>>,
}

impl Reducer {
    /// The binned detector's shape.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The raw pixels of binned pixel `(i, j)`, as flat indices into the
    /// raw frame.
    fn block(&self, i: usize, j: usize) -> impl Iterator<Item = usize> {
        let (s0, f0) = (self.start.0 + i * self.bin.0, self.start.1 + j * self.bin.1);
        let (bs, bf, width) = (self.bin.0, self.bin.1, self.raw_shape.1);
        (s0..s0 + bs).flat_map(move |s| (f0..f0 + bf).map(move |f| s * width + f))
    }
}

/// One pixel's counts and weight; `counts / weight` is the corrected value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalised {
    pub counts: f64,
    pub weight: Weight,
}

impl Normalised {
    pub fn value(&self) -> f64 {
        self.counts / self.weight.get()
    }
}

/// A frame's pixels over the binned shape, row-major (slow, fast).
#[derive(Clone, Debug, PartialEq)]
pub struct NormalisedFrame {
    shape: (usize, usize),
    pub data: Vec<Result<Normalised, Masked>>,
}

impl NormalisedFrame {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn get(&self, slow: usize, fast: usize) -> Result<Normalised, Masked> {
        assert!(slow < self.shape.0 && fast < self.shape.1);
        self.data[slow * self.shape.1 + fast]
    }
}

/// The per-pixel stages of one detector.
#[derive(Clone, Debug, PartialEq)]
pub struct Normaliser {
    fields: PixelFields,
}

impl Normaliser {
    pub fn new(fields: PixelFields) -> Self {
        Self { fields }
    }

    /// One frame. `raw` is the whole raw frame, row-major over the shape
    /// of `fields`; `weights` are geometric weights of the binned
    /// detector, each of `reducer`'s shape (empty for none).
    ///
    /// A dark field with no count time in `scalars` is an error: its rate
    /// has no counts to subtract.
    pub fn normalise(
        &self,
        raw: &[f64],
        scalars: &FrameScalars,
        reducer: &Reducer,
        weights: &[&Weights],
    ) -> Result<NormalisedFrame, Error> {
        let f = &self.fields;
        f.check("raw frame", raw.len())?;
        if f.shape != reducer.raw_shape {
            return Err(Error::PixelCount {
                what: "region of another raw shape",
                len: reducer.raw_shape.0 * reducer.raw_shape.1,
                expected: f.shape.0 * f.shape.1,
            });
        }
        if f.dark_rate.is_some() && scalars.count_time.is_none() {
            return Err(Error::DarkNeedsTime);
        }
        let out = reducer.shape.0 * reducer.shape.1;
        for w in weights {
            if w.shape() != reducer.shape {
                return Err(Error::PixelCount {
                    what: "weights",
                    len: w.shape().0 * w.shape().1,
                    expected: out,
                });
            }
        }
        let n = (reducer.bin.0 * reducer.bin.1) as f64;
        let time = scalars.count_time.unwrap_or(0.0);
        let mut data = Vec::with_capacity(out);
        for i in 0..reducer.shape.0 {
            for j in 0..reducer.shape.1 {
                // The one place a pixel is masked, in precedence order.
                data.push((|| -> Result<Normalised, Masked> {
                    let at = i * reducer.shape.1 + j;
                    if reducer.mask.as_ref().is_some_and(|m| m[at]) {
                        return Err(Masked::Detector);
                    }
                    let mut geometric = scalars.factor();
                    for w in weights {
                        geometric *= w.get(i, j)?.get();
                    }
                    let mut gain = 0.0;
                    for p in reducer.block(i, j) {
                        gain += match &f.flat {
                            Some(flat) => flat[p].ok_or(Masked::BadFlat)?,
                            None => 1.0,
                        };
                    }
                    let mut counts = 0.0;
                    for p in reducer.block(i, j) {
                        if !raw[p].is_finite() {
                            return Err(Masked::BadCount);
                        }
                        counts += raw[p] - f.dark_rate.as_ref().map_or(0.0, |d| d[p] * time);
                    }
                    if !counts.is_finite() {
                        return Err(Masked::BadCount);
                    }
                    let weight = Weight::new(geometric * gain / n, Masked::Opaque)?;
                    Ok(Normalised { counts, weight })
                })());
            }
        }
        Ok(NormalisedFrame {
            shape: reducer.shape,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::{Frame, Geometry, Wavelength, solid_angle};

    fn detector(shape: (usize, usize)) -> PixelDetector {
        PixelDetector::new(
            shape,
            Vector3::new(0.4, -0.01, -0.012),
            Vector3::new(0.0, 0.0, 1e-3),
            Vector3::new(0.0, 1.5e-3, 0.0),
        )
        .unwrap()
    }

    fn geometry_frame() -> (Geometry, Frame) {
        let g = Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let f = Frame {
            values: vec![0.0],
            wavelength: Wavelength::new(1.0).unwrap(),
        };
        (g, f)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    /// XU-13: a binned pixel sits at the mean of the centres of the raw
    /// pixels whose counts it sums, whatever the roi start, and a trailing
    /// partial block is in neither.
    #[test]
    fn binned_pixel_is_where_its_block_is() {
        let raw_det = detector((7, 11));
        let g = Vector3::new(0.3, -2.1, 5.7);
        let value = |p: nalgebra::Point3<f64>| 1000.0 + g.dot(&p.coords);
        let frame: Vec<f64> = (0..7)
            .flat_map(|s| (0..11).map(move |f| (s, f)))
            .map(|(s, f)| value(raw_det.pixel(s, f).unwrap()))
            .collect();
        let n = Normaliser::new(PixelFields::new((7, 11)));
        for (slow, fast, bin) in [
            (0..7, 0..11, (3, 3)),
            (0..6, 1..10, (3, 3)),
            (1..7, 4..11, (2, 3)),
            (2..7, 5..11, (5, 4)),
            (0..7, 0..11, (1, 1)),
        ] {
            let (binned, r) = Region::new(slow, fast, bin).apply(&raw_det).unwrap();
            let out = n.normalise(&frame, &FrameScalars::new(), &r, &[]).unwrap();
            assert_eq!(out.shape(), binned.shape());
            let size = (bin.0 * bin.1) as f64;
            for i in 0..out.shape().0 {
                for j in 0..out.shape().1 {
                    let p = out.get(i, j).unwrap();
                    assert!(close(p.weight.get(), 1.0));
                    let at = value(binned.pixel(i, j).unwrap());
                    assert!(close(p.counts / size, at), "{bin:?} ({i}, {j})");
                }
            }
        }
    }

    /// `Σcounts / Σweight` over the pixels of a block is the same as over
    /// the block's raw pixels, flat field per raw pixel and the binned
    /// pixel's geometric weight shared out `1/n` each.
    #[test]
    fn block_weight_is_the_sum_of_the_pixels_weights() {
        let det = detector((4, 6));
        let (g, f) = geometry_frame();
        let (binned, r) = Region::new(0..4, 0..6, (2, 3)).apply(&det).unwrap();
        let sa = solid_angle(&g, &f, &binned).unwrap();
        let flat: Vec<f64> = (0..24).map(|i| 0.8 + 0.01 * i as f64).collect();
        let mean = flat.iter().sum::<f64>() / 24.0;
        let dark: Vec<f64> = (0..24).map(|i| 0.1 * i as f64).collect();
        let raw: Vec<f64> = (0..24).map(|i| 500.0 + 7.0 * i as f64).collect();
        let fields = PixelFields::new((4, 6))
            .dark_rate(dark.clone())
            .unwrap()
            .flat(flat.clone())
            .unwrap();
        let s = FrameScalars::new()
            .monitor(900.0, 1000.0)
            .unwrap()
            .count_time(2.0)
            .unwrap()
            .transmission(0.25)
            .unwrap();
        let out = Normaliser::new(fields)
            .normalise(&raw, &s, &r, &[&sa])
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let (mut counts, mut weight) = (0.0, 0.0);
                for p in (0..24).filter(|p| (p / 6) / 2 == i && (p % 6) / 3 == j) {
                    counts += raw[p] - dark[p] * 2.0;
                    weight +=
                        0.9 * 2.0 * 0.25 * (flat[p] / mean) * sa.get(i, j).unwrap().get() / 6.0;
                }
                let got = out.get(i, j).unwrap();
                assert!(close(got.counts, counts), "({i}, {j})");
                assert!(close(got.weight.get(), weight), "({i}, {j})");
                assert!(close(got.value(), counts / weight));
            }
        }
    }

    /// Dark-subtracted counts keep their sign.
    #[test]
    fn counts_are_not_clamped_at_zero() {
        let det = detector((1, 3));
        let (_, r) = Region::new(0..1, 0..3, (1, 1)).apply(&det).unwrap();
        let fields = PixelFields::new((1, 3))
            .dark_rate(vec![10.0, 10.0, 10.0])
            .unwrap();
        let s = FrameScalars::new().count_time(2.0).unwrap();
        let out = Normaliser::new(fields)
            .normalise(&[5.0, 20.0, 50.0], &s, &r, &[])
            .unwrap();
        let counts: Vec<f64> = out.data.iter().map(|p| p.unwrap().counts).collect();
        assert_eq!(counts, [-15.0, 0.0, 30.0]);
    }

    /// One boundary per test of the precedence: detector mask, a caller's
    /// weight, the flat field, the count, the total weight. Each pixel
    /// below has the reason named and every later one.
    fn five() -> (Reducer, PixelDetector) {
        let det = detector((1, 5));
        let masked = det
            .clone()
            .with_mask(vec![true, false, false, false, false]);
        let (binned, r) = Region::new(0..1, 0..5, (1, 1))
            .apply(&masked.unwrap())
            .unwrap();
        (r, binned)
    }

    fn weights_with(binned: &PixelDetector, errs: &[usize]) -> Weights {
        let (g, f) = geometry_frame();
        let mut w = solid_angle(&g, &f, binned).unwrap();
        for &k in errs {
            w.data[k] = Err(Masked::EdgeOn);
        }
        w
    }

    fn run(
        flat: &[f64],
        raw: &[f64],
        scalars: &FrameScalars,
        w: &Weights,
        r: &Reducer,
    ) -> Vec<Result<Normalised, Masked>> {
        let fields = PixelFields::new((1, 5)).flat(flat.to_vec()).unwrap();
        Normaliser::new(fields)
            .normalise(raw, scalars, r, &[w])
            .unwrap()
            .data
    }

    #[test]
    fn detector_mask_precedes_the_weights() {
        let (r, binned) = five();
        let w = weights_with(&binned, &[0, 1]);
        let out = run(&[1.0; 5], &[1.0; 5], &FrameScalars::new(), &w, &r);
        assert_eq!(out[0], Err(Masked::Detector));
        assert_eq!(out[1], Err(Masked::EdgeOn));
        assert!(out[2].is_ok());
    }

    #[test]
    fn weights_precede_the_flat_field() {
        let (r, binned) = five();
        let w = weights_with(&binned, &[1]);
        let out = run(
            &[1.0, 0.0, 0.0, 1.0, 1.0],
            &[1.0; 5],
            &FrameScalars::new(),
            &w,
            &r,
        );
        assert_eq!(out[1], Err(Masked::EdgeOn));
        assert_eq!(out[2], Err(Masked::BadFlat));
    }

    #[test]
    fn flat_field_precedes_the_count() {
        let (r, binned) = five();
        let w = weights_with(&binned, &[]);
        let raw = [1.0, 1.0, f64::NAN, f64::INFINITY, 1.0];
        let out = run(
            &[1.0, 1.0, 0.0, 1.0, 1.0],
            &raw,
            &FrameScalars::new(),
            &w,
            &r,
        );
        assert_eq!(out[2], Err(Masked::BadFlat));
        assert_eq!(out[3], Err(Masked::BadCount));
        assert!(out[4].is_ok());
    }

    /// A weight that underflows is `Opaque`, after a non-finite count.
    #[test]
    fn count_precedes_the_total_weight() {
        let (r, binned) = five();
        let w = weights_with(&binned, &[]);
        let tiny = FrameScalars::new()
            .monitor(1e-200, 1.0)
            .unwrap()
            .transmission(1e-200)
            .unwrap();
        let raw = [1.0, 1.0, f64::NAN, 1.0, 1.0];
        let out = run(&[1.0; 5], &raw, &tiny, &w, &r);
        assert_eq!(out[2], Err(Masked::BadCount));
        assert_eq!(out[3], Err(Masked::Opaque));
    }

    /// A block is masked by any one of its pixels, for each reason that
    /// looks at raw pixels.
    #[test]
    fn one_bad_pixel_masks_its_block() {
        let det = detector((2, 4))
            .with_mask(vec![false, false, false, false, false, false, true, false])
            .unwrap();
        let (_, r) = Region::new(0..2, 0..4, (2, 2)).apply(&det).unwrap();
        let fields = PixelFields::new((2, 4))
            .flat(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
            .unwrap();
        // block 0 is clean; block 1 has a masked pixel (also a bad flat)
        let n = Normaliser::new(fields.clone());
        let out = n
            .normalise(&[1.0; 8], &FrameScalars::new(), &r, &[])
            .unwrap();
        assert!(out.data[0].is_ok());
        assert_eq!(out.data[1], Err(Masked::Detector));
        let det = detector((2, 4));
        let (_, r) = Region::new(0..2, 0..4, (2, 2)).apply(&det).unwrap();
        let out = n
            .normalise(&[1.0; 8], &FrameScalars::new(), &r, &[])
            .unwrap();
        assert_eq!(out.data[1], Err(Masked::BadFlat));
        let n = Normaliser::new(PixelFields::new((2, 4)));
        let raw = [1.0, 1.0, 1.0, f64::NAN, 1.0, 1.0, 1.0, 1.0];
        let out = n.normalise(&raw, &FrameScalars::new(), &r, &[]).unwrap();
        assert!(out.data[0].is_ok());
        assert_eq!(out.data[1], Err(Masked::BadCount));
    }

    #[test]
    fn bad_frame_scalars_are_errors_not_ones() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(FrameScalars::new().monitor(bad, 1.0).is_err(), "{bad}");
            assert!(FrameScalars::new().monitor(1.0, bad).is_err(), "{bad}");
            assert!(FrameScalars::new().count_time(bad).is_err(), "{bad}");
            assert!(FrameScalars::new().transmission(bad).is_err(), "{bad}");
        }
        assert!(mean_monitor(&[]).is_err());
        assert!(mean_monitor(&[1.0, 0.0]).is_err());
        assert_eq!(mean_monitor(&[1.0, 3.0]), Ok(2.0));
    }

    #[test]
    fn a_dark_field_needs_a_count_time_and_shapes_must_agree() {
        let det = detector((1, 3));
        let (_, r) = Region::new(0..1, 0..3, (1, 1)).apply(&det).unwrap();
        let dark = Normaliser::new(PixelFields::new((1, 3)).dark_rate(vec![0.0; 3]).unwrap());
        assert_eq!(
            dark.normalise(&[1.0; 3], &FrameScalars::new(), &r, &[]),
            Err(Error::DarkNeedsTime)
        );
        assert!(PixelFields::new((1, 3)).flat(vec![1.0; 2]).is_err());
        assert!(
            PixelFields::new((1, 3))
                .dark_rate(vec![f64::NAN; 3])
                .is_err()
        );
        assert!(PixelFields::new((1, 3)).flat(vec![0.0; 3]).is_err());
        let n = Normaliser::new(PixelFields::new((1, 3)));
        assert!(
            n.normalise(&[1.0; 2], &FrameScalars::new(), &r, &[])
                .is_err()
        );
        let (g, f) = geometry_frame();
        let other = solid_angle(&g, &f, &detector((1, 2))).unwrap();
        assert!(
            n.normalise(&[1.0; 3], &FrameScalars::new(), &r, &[&other])
                .is_err()
        );
        let (_, r6) = Region::new(0..1, 0..3, (1, 1))
            .apply(&detector((1, 3)))
            .unwrap();
        let n6 = Normaliser::new(PixelFields::new((2, 3)));
        assert!(
            n6.normalise(&[1.0; 6], &FrameScalars::new(), &r6, &[])
                .is_err()
        );
    }

    /// XU-11: a closure over a method, a function item and a closure with
    /// captured state are all absorbers; xu's arity check refuses the first
    /// and crashes on the others.
    #[test]
    fn any_callable_is_an_absorber() {
        struct Foils(Vec<f64>);
        impl Foils {
            fn at(&self, i: &usize) -> f64 {
                self.0[*i]
            }
        }
        fn half(_: &usize) -> f64 {
            0.5
        }
        let foils = Foils(vec![1.0, 0.1]);
        let s = FrameScalars::new()
            .absorber(&|i: &usize| foils.at(i), &1)
            .unwrap();
        assert_eq!(s.factor(), 0.1);
        let s = FrameScalars::new().absorber(&half, &0).unwrap();
        assert_eq!(s.factor(), 0.5);
        let k = 0.25;
        let s = FrameScalars::new().absorber(&move |_: &()| k, &()).unwrap();
        assert_eq!(s.factor(), 0.25);
        assert!(FrameScalars::new().absorber(&|_: &()| 0.0, &()).is_err());
    }
}
