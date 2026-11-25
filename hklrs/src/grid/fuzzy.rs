//! Pixel footprints: how a pixel's counts and weight are laid on the bins
//! of a [`Binning`]. `Footprint::reach` is the one place a pixel's share
//! of a bin is computed.

use super::{Binning, Coord, CoordKind, Projected};
use crate::Error;
use crate::normalise::Normalised;

/// How a pixel's weight is laid on a bin lattice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Footprint {
    /// All of it in the bin of the pixel's coordinate.
    Point,
    /// A pixel spans `width`, centred on its coordinate (same units as the
    /// coordinate), and its counts and weight are shared out over the bins
    /// it overlaps in proportion to the overlap: xu's `FuzzyGridder1D`,
    /// whose fractions these are. A bin of the global lattice exists
    /// wherever the footprint reaches, so unlike xu nothing is clamped at a
    /// grid end.
    Width(f64),
}

impl Footprint {
    /// Refuses a width that is not positive and finite, on a coordinate of
    /// `kind`.
    pub(crate) fn check(self, kind: CoordKind) -> Result<Self, Error> {
        match self {
            Footprint::Width(w) if !(w.is_finite() && w > 0.0) => Err(Error::Width {
                what: "footprint width",
                value: w,
                kind,
            }),
            _ => Ok(self),
        }
    }

    /// The bins of `binning` a pixel at `v`, on a coordinate of `kind`,
    /// reaches; `Err` when an end of the footprint has no bin.
    pub(crate) fn reach(self, binning: Binning, kind: CoordKind, v: f64) -> Result<Reach, Error> {
        let (lo, hi, width) = match self {
            Footprint::Point => (v, v, 0.0),
            Footprint::Width(w) => (v - w / 2.0, v + w / 2.0, w),
        };
        let bin = |value: f64| {
            binning.bin(value).ok_or(Error::NoBin {
                value,
                step: binning.step(),
                kind,
            })
        };
        Ok(Reach {
            binning,
            lo,
            hi,
            width,
            k0: bin(lo)?,
            k1: bin(hi)?,
        })
    }
}

/// One pixel's footprint, [lo, hi], on one lattice: bins `k0..=k1`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reach {
    binning: Binning,
    lo: f64,
    hi: f64,
    width: f64,
    k0: i64,
    k1: i64,
}

impl Reach {
    /// The share of the footprint in bin `k`: all of it when the footprint
    /// lies in one bin, else the overlap over the width.
    fn share(&self, k: i64) -> f64 {
        if self.k0 == self.k1 {
            1.0
        } else {
            let b = self.binning;
            (self.hi.min(b.edge(k + 1)) - self.lo.max(b.edge(k))) / self.width
        }
    }

    /// (bin, share) of every bin with a positive share, ascending; a
    /// footprint ending exactly on an edge has no part in the bin above.
    pub(crate) fn shares(&self) -> impl Iterator<Item = (i64, f64)> + '_ {
        (self.k0..=self.k1).filter_map(|k| {
            let f = self.share(k);
            (f > 0.0).then_some((k, f))
        })
    }
}

impl Projected {
    /// The frame with `footprints`, one per dimension in order; every
    /// width must be positive and finite.
    pub fn with_footprint(mut self, footprints: Vec<Footprint>) -> Result<Self, Error> {
        if footprints.len() != self.planes.len() {
            return Err(Error::Dimensions {
                expected: self.planes.len(),
                got: footprints.len(),
            });
        }
        self.footprints = footprints
            .into_iter()
            .zip(&self.coords)
            .map(|(f, c)| f.check(c.kind()))
            .collect::<Result<_, _>>()?;
        Ok(self)
    }

    pub fn footprints(&self) -> &[Footprint] {
        &self.footprints
    }
}

/// The pieces of a frame's unmasked pixels: each pixel's value and, for
/// every bin its footprint reaches with a positive share, the bin's global
/// indices and the share, the product of the per-axis shares. A pixel's
/// pieces run with the last axis fastest; a point footprint is one piece of
/// share exactly 1.0.
pub(crate) struct Pieces {
    dims: usize,
    values: Vec<Normalised>,
    /// pixel p's pieces are `ends[p - 1]..ends[p]`
    ends: Vec<usize>,
    bins: Vec<i64>,
    shares: Vec<f64>,
    /// (bin, share) per axis of the pixel being pushed, and the piece's
    /// index into each
    axis: Vec<Vec<(i64, f64)>>,
    at: Vec<usize>,
}

impl Pieces {
    pub(crate) fn with_capacity(dims: usize, pixels: usize) -> Self {
        Self {
            dims,
            values: Vec::with_capacity(pixels),
            ends: Vec::with_capacity(pixels),
            bins: Vec::with_capacity(pixels * dims),
            shares: Vec::with_capacity(pixels),
            axis: vec![Vec::new(); dims],
            at: vec![0; dims],
        }
    }

    /// Adds a pixel of value `n` at `coords`, one per axis of `axes`, laid
    /// by `footprints` on `binnings`. `Err`, and nothing added, when an end
    /// of its footprint has no bin.
    pub(crate) fn push(
        &mut self,
        n: Normalised,
        axes: &[Coord],
        footprints: &[Footprint],
        binnings: &[Binning],
        coords: impl IntoIterator<Item = f64>,
    ) -> Result<(), Error> {
        let Self {
            dims,
            values,
            ends,
            bins,
            shares,
            axis,
            at,
        } = self;
        debug_assert!(axes.len() == *dims && footprints.len() == *dims && binnings.len() == *dims);
        let laid = axes.iter().zip(footprints).zip(binnings).zip(coords);
        for (a, (((c, f), &b), v)) in axis.iter_mut().zip(laid) {
            a.clear();
            a.extend(f.reach(b, c.kind(), v)?.shares());
        }
        if axis.iter().all(|a| !a.is_empty()) {
            at.fill(0);
            loop {
                let mut share = 1.0;
                for (a, &j) in axis.iter().zip(at.iter()) {
                    let (k, s) = a[j];
                    bins.push(k);
                    share *= s;
                }
                shares.push(share);
                // the next piece, the last axis fastest
                let Some(d) = (0..*dims).rev().find(|&d| at[d] + 1 < axis[d].len()) else {
                    break;
                };
                at[d] += 1;
                at[d + 1..].fill(0);
            }
        }
        values.push(n);
        ends.push(shares.len());
        Ok(())
    }

    /// Global bins of every piece, flattened piece-major.
    pub(crate) fn bins(&self) -> &[i64] {
        &self.bins
    }

    pub(crate) fn len(&self) -> usize {
        self.shares.len()
    }

    /// Each pixel's value and its pieces, (global bins, share), in push
    /// order.
    pub(crate) fn pixels(
        &self,
    ) -> impl Iterator<Item = (Normalised, impl Iterator<Item = (&[i64], f64)> + '_)> + '_ {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        self.values
            .iter()
            .zip(starts.zip(&self.ends))
            .map(move |(&n, (s, &e))| {
                let bins = self.bins[s * self.dims..e * self.dims].chunks_exact(self.dims);
                (n, bins.zip(self.shares[s..e].iter().copied()))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Image;
    use crate::absorption::{Masked, Weight};
    use crate::grid::{Bounds, Coord, Cube};
    use crate::normalise::NormalisedFrame;

    const ONE: &[&str] = &["x"];
    const TWO: &[&str] = &["x", "y"];

    fn linear(names: &[&'static str]) -> Vec<Coord> {
        names.iter().map(|&n| Coord::linear(n)).collect()
    }

    fn n(counts: f64) -> Normalised {
        Normalised::new(counts, Weight::new(1.0, Masked::Opaque)).unwrap()
    }

    fn unit() -> Binning {
        Binning::new(CoordKind::Linear, 1.0, 0.0).unwrap()
    }

    fn pieces(footprints: &[Footprint], coords: &[f64]) -> Result<Vec<(Vec<i64>, f64)>, Error> {
        let axes = vec![Coord::linear("x"); coords.len()];
        let binnings = vec![unit(); coords.len()];
        let mut p = Pieces::with_capacity(coords.len(), 1);
        p.push(n(1.0), &axes, footprints, &binnings, coords.iter().copied())?;
        let (_, ps) = p.pixels().next().unwrap();
        Ok(ps.map(|(k, f)| (k.to_vec(), f)).collect())
    }

    #[test]
    fn a_point_is_one_piece_of_share_exactly_one() {
        let got = pieces(&[Footprint::Point; 2], &[0.75, -2.5]).unwrap();
        assert_eq!(got, [(vec![0, -3], 1.0)]);
    }

    #[test]
    fn shares_multiply_across_axes_the_last_axis_fastest() {
        // x [0.25, 1.25]: 3/4 in bin 0, 1/4 in 1; y [2.75, 3.75]: 1/4 in 2, 3/4 in 3
        let got = pieces(&[Footprint::Width(1.0); 2], &[0.75, 3.25]).unwrap();
        assert_eq!(
            got,
            [
                (vec![0, 2], 0.1875),
                (vec![0, 3], 0.5625),
                (vec![1, 2], 0.0625),
                (vec![1, 3], 0.1875),
            ]
        );
    }

    #[test]
    fn a_footprint_ending_on_an_edge_has_no_piece_above() {
        let got = pieces(&[Footprint::Width(1.0), Footprint::Point], &[0.5, 0.5]).unwrap();
        assert_eq!(got, [(vec![0, 0], 1.0)]);
    }

    #[test]
    fn a_footprint_end_without_a_bin_adds_nothing() {
        let axes = [Coord::linear("x"), Coord::angle("tth")];
        let binnings = [unit(); 2];
        let mut p = Pieces::with_capacity(2, 1);
        let w = [Footprint::Point, Footprint::Width(1.0)];
        for bad in [f64::NAN, f64::INFINITY, 1e300] {
            let err = p
                .push(n(1.0), &axes, &w, &binnings, [0.5, bad])
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::NoBin {
                        kind: CoordKind::Angle,
                        ..
                    }
                ),
                "{err:?}"
            );
        }
        assert_eq!((p.pixels().count(), p.len(), p.bins().len()), (0, 0, 0));
    }

    /// One row of pixels at `coords` (per pixel, one value per dimension)
    /// of `counts` and weight 1, laid by `footprints`.
    fn frame(
        names: &[&'static str],
        coords: &[&[f64]],
        counts: &[f64],
        footprints: &[Footprint],
    ) -> (Projected, NormalisedFrame) {
        let shape = (1, coords.len());
        let planes = (0..names.len())
            .map(|d| Image::from_fn(shape, |_, i| coords[i][d]))
            .collect();
        let points = Projected::new(linear(names), planes)
            .unwrap()
            .with_footprint(footprints.to_vec())
            .unwrap();
        (points, Image::from_fn(shape, |_, i| Ok(n(counts[i]))))
    }

    fn grown(points: &Projected, pixels: &NormalisedFrame) -> Cube {
        let mut c = Cube::default();
        let binnings = vec![unit(); points.planes().len()];
        c.add_growing(&binnings, points, pixels).unwrap();
        c
    }

    /// A zero cube over bins `first..=last` of the unit lattice.
    fn fixed(first: f64, last: f64) -> Cube {
        let (points, pixels) = frame(ONE, &[&[first], &[last]], &[1.0, 1.0], &[Footprint::Point]);
        let mut b = Bounds::new(vec![unit()]);
        b.include(&points, &pixels).unwrap();
        b.cube().unwrap()
    }

    fn on_threads<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(f)
    }

    #[test]
    fn the_axes_span_the_bins_a_footprint_reaches_in_both_passes() {
        let w = Footprint::Width(1.0);
        // [-1, 0] reaches bin -1 only, [3, 4] bin 3 only; y [0.25, 1.25] two
        let (points, pixels) = frame(
            TWO,
            &[&[-0.5, 0.75], &[3.5, 0.75], &[1.0, 0.75]],
            &[1.0, 2.0, 4.0],
            &[w, w],
        );
        let one = grown(&points, &pixels);
        let spans: Vec<_> = one.axes().iter().map(|a| (a.first(), a.size())).collect();
        assert_eq!(spans, [(-1, 5), (0, 2)]);

        let mut b = Bounds::new(vec![unit(); 2]);
        b.include(&points, &pixels).unwrap();
        let mut two = b.cube().unwrap();
        assert_eq!(two.add(&points, &pixels).unwrap(), 0);
        assert_eq!(two, one);
    }

    #[test]
    fn fixed_axes_keep_the_share_inside_and_count_pixels_not_wholly_inside() {
        let mut c = fixed(0.5, 2.5);
        let (points, pixels) = frame(
            ONE,
            &[&[1.5], &[0.25], &[10.0]],
            &[4.0, 8.0, 1.0],
            &[Footprint::Width(1.0)],
        );
        // [1, 2] is bin 1; [-0.25, 0.75] is 1/4 in bin -1, outside; [9.5, 10.5]
        assert_eq!(c.add(&points, &pixels).unwrap(), 2);
        assert_eq!(c.photons(), [6.0, 4.0, 0.0]);
        assert_eq!(c.weight(), [0.75, 1.0, 0.0]);
        assert_eq!(c.contributions(), [1, 1, 0]);
    }

    #[test]
    fn contributions_count_the_pixels_that_reach_a_bin() {
        // [-1, 3]: a quarter in each of bins -1..=2, none in 3; then a point
        let (wide, px) = frame(ONE, &[&[1.0]], &[8.0], &[Footprint::Width(4.0)]);
        let mut c = grown(&wide, &px);
        let (point, px) = frame(ONE, &[&[0.5]], &[1.0], &[Footprint::Point]);
        c.add_growing(&[unit()], &point, &px).unwrap();
        assert_eq!(c.axes()[0].first(), -1);
        assert_eq!(c.photons(), [2.0, 3.0, 2.0, 2.0]);
        assert_eq!(c.weight(), [0.25, 1.25, 0.25, 0.25]);
        assert_eq!(c.contributions(), [1, 2, 1, 1]);
    }

    #[test]
    fn add_frames_with_footprints_sums_as_frame_by_frame_on_any_thread_count() {
        // 1e16 + 1 rounds back to 1e16: bin 0 ends at 1 and bin 1 at 4 only
        // if they take +1e16, the ones and -1e16 in this order
        let w = [Footprint::Width(1.0)];
        let frames = [
            frame(ONE, &[&[1.0], &[2.5]], &[2e16, 3.0], &w),
            frame(
                ONE,
                &[&[0.5], &[1.5], &[1.25]],
                &[1.0, 1.0, 4.0],
                &[Footprint::Point],
            ),
            frame(ONE, &[&[1.0], &[0.5]], &[-2e16, 1.0], &w),
        ];
        let start = fixed(0.0, 3.0);
        let mut expected = start.clone();
        for (p, px) in &frames {
            expected.add(p, px).unwrap();
        }
        assert_eq!(expected.photons(), [1.0, 4.0, 3.0, 0.0]);
        let frames: Vec<_> = frames.iter().map(|(p, px)| (p, px)).collect();
        for threads in [1, 2, 3, 5, 12] {
            let mut c = start.clone();
            assert_eq!(on_threads(threads, || c.add_frames(&frames)).unwrap(), 0);
            assert_eq!(c, expected, "{threads} threads");
        }
    }

    #[test]
    fn cubes_laid_by_different_footprints_merge() {
        let (a, apx) = frame(ONE, &[&[0.5]], &[1.0], &[Footprint::Point]);
        let (b, bpx) = frame(ONE, &[&[1.0]], &[2.0], &[Footprint::Width(1.0)]);
        let mut both = grown(&a, &apx);
        both.add_growing(&[unit()], &b, &bpx).unwrap();
        assert_eq!(grown(&a, &apx).merge(grown(&b, &bpx)).unwrap(), both);
        assert_eq!(both.weight(), [1.5, 0.5]);
    }

    #[test]
    fn a_footprint_is_given_per_dimension() {
        let planes = vec![Image::from_fn((1, 1), |_, _| 0.5); 2];
        let p = Projected::new(linear(TWO), planes).unwrap();
        assert_eq!(p.footprints(), [Footprint::Point; 2]);
        assert_eq!(
            p.clone().with_footprint(vec![Footprint::Width(1.0)]),
            Err(Error::Dimensions {
                expected: 2,
                got: 1
            })
        );
        assert!(matches!(
            p.with_footprint(vec![Footprint::Point, Footprint::Width(-1.0)]),
            Err(Error::Width {
                kind: CoordKind::Linear,
                ..
            })
        ));
    }

    #[test]
    fn a_width_must_be_positive_and_finite() {
        for w in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                Footprint::Width(w).check(CoordKind::Angle),
                Err(Error::Width {
                    what: "footprint width",
                    kind: CoordKind::Angle,
                    ..
                })
            ));
        }
        let ok = |f: Footprint| f.check(CoordKind::Linear);
        assert_eq!(ok(Footprint::Point), Ok(Footprint::Point));
        assert_eq!(ok(Footprint::Width(0.5)), Ok(Footprint::Width(0.5)));
    }
}
