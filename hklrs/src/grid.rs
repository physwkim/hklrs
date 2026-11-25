//! Gridding (design §7): per-pixel projection coordinates and normalised
//! pixels summed into a cube of Σcounts, Σweight and contributions per bin,
//! the sums a bin of a cut holds ([`crate::cut::Bin`]). A bin's intensity
//! is their ratio Σcounts/Σweight ([`Cube::intensity`]).
//!
//! The one bin rule: on a dimension binned with width `step` from `anchor`,
//! a value v falls in global bin k = ⌊(v − anchor)/step⌋, the half-open
//! interval [anchor + k·step, anchor + (k+1)·step). [`Binning::bin`] is the
//! only place it is computed. binoculars 7be4f79 takes rint(v/resolution)
//! instead (hkl-binoculars.c:375, :842-844), bins centred on multiples of the
//! resolution whose edges go to the even neighbour.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::normalise::{Normalised, NormalisedFrame};
use crate::{Error, Image};

pub mod fuzzy;
use fuzzy::{Footprint, Pieces};
#[cfg(feature = "hdf5")]
pub mod hdf5;

/// The bin lattice of one dimension: width and the edge every other edge
/// is an integer number of widths from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binning {
    step: f64,
    anchor: f64,
}

impl Binning {
    pub fn new(step: f64, anchor: f64) -> Result<Self, Error> {
        if !(step.is_finite() && step > 0.0) {
            return Err(Error::NotPositive {
                what: "bin step",
                value: step,
            });
        }
        if !anchor.is_finite() {
            return Err(Error::NotFinite {
                what: "bin anchor",
                value: anchor,
            });
        }
        Ok(Self { step, anchor })
    }

    pub fn step(&self) -> f64 {
        self.step
    }

    pub fn anchor(&self) -> f64 {
        self.anchor
    }

    /// The bin rule: global bin ⌊(v − anchor)/step⌋.
    pub fn bin(&self, v: f64) -> Result<i64, Error> {
        let k = ((v - self.anchor) / self.step).floor();
        // 2⁶³ is exact in f64; NaN fails both comparisons
        const LIMIT: f64 = 9_223_372_036_854_775_808.0;
        if (-LIMIT..LIMIT).contains(&k) {
            Ok(k as i64)
        } else {
            Err(Error::NoBin {
                value: v,
                step: self.step,
            })
        }
    }

    /// Lower edge of global bin `k`.
    pub fn edge(&self, k: i64) -> f64 {
        self.anchor + k as f64 * self.step
    }
}

/// What a projection coordinate measures, which fixes the unit it is
/// shown in: an angle is radians in hklrs and degrees in Python; a linear
/// coordinate (Å⁻¹, r.l.u., an index) is shown as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordKind {
    Angle,
    Linear,
}

/// One coordinate of a projection: the name of the cube axis it is binned
/// on, and its kind. Two coordinates of one name but another kind are
/// different axes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coord {
    name: Cow<'static, str>,
    kind: CoordKind,
}

impl Coord {
    pub fn new(name: impl Into<Cow<'static, str>>, kind: CoordKind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }

    pub const fn angle(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            kind: CoordKind::Angle,
        }
    }

    pub const fn linear(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            kind: CoordKind::Linear,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> CoordKind {
        self.kind
    }
}

/// One cube axis: `size` consecutive global bins of `binning` from `first`.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    coord: Coord,
    binning: Binning,
    first: i64,
    size: usize,
}

impl Axis {
    pub fn name(&self) -> &str {
        self.coord.name()
    }

    pub fn kind(&self) -> CoordKind {
        self.coord.kind
    }

    pub fn coord(&self) -> &Coord {
        &self.coord
    }

    pub fn binning(&self) -> Binning {
        self.binning
    }

    /// Global index of the first bin.
    pub fn first(&self) -> i64 {
        self.first
    }

    /// Number of bins, at least one.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Lower edge of the first bin.
    pub fn origin(&self) -> f64 {
        self.binning.edge(self.first)
    }

    fn last(&self) -> i64 {
        self.first + (self.size as i64 - 1)
    }

    /// Position of global bin `k` on this axis.
    fn position(&self, k: i64) -> Option<usize> {
        let i = usize::try_from(k.checked_sub(self.first)?).ok()?;
        (i < self.size).then_some(i)
    }

    /// The axis over global bins first..=last; its size fits both usize
    /// and i64, so `last()` is exact.
    fn spanning(coord: &Coord, binning: Binning, (first, last): (i64, i64)) -> Result<Self, Error> {
        let size = i64::try_from(i128::from(last) - i128::from(first) + 1)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::CubeSize)?;
        Ok(Self {
            coord: coord.clone(),
            binning,
            first,
            size,
        })
    }

    fn describe(&self) -> (Coord, Binning) {
        (self.coord.clone(), self.binning)
    }
}

/// One frame in projection coordinates: per dimension its coordinate,
/// which the projection gives, its values over the shape of the pixels it
/// was projected from, and the footprint that lays a pixel on its bins (a
/// point unless [`with_footprint`](Self::with_footprint) says otherwise).
#[derive(Clone, Debug, PartialEq)]
pub struct Projected {
    coords: Vec<Coord>,
    planes: Vec<Image<f64>>,
    footprints: Vec<Footprint>,
}

impl Projected {
    pub fn new(coords: Vec<Coord>, planes: Vec<Image<f64>>) -> Result<Self, Error> {
        if coords.is_empty() || planes.len() != coords.len() {
            return Err(Error::Dimensions {
                expected: coords.len(),
                got: planes.len(),
            });
        }
        let shape = planes[0].shape();
        for p in &planes {
            p.check("projection plane", shape)?;
        }
        let footprints = vec![Footprint::Point; coords.len()];
        Ok(Self {
            coords,
            planes,
            footprints,
        })
    }

    pub fn coords(&self) -> &[Coord] {
        &self.coords
    }

    pub fn planes(&self) -> &[Image<f64>] {
        &self.planes
    }

    /// The (slow, fast) shape of the pixels projected.
    pub fn shape(&self) -> (usize, usize) {
        self.planes[0].shape()
    }

    pub fn pixels(&self) -> usize {
        self.planes[0].data().len()
    }

    fn check_binnings(&self, binnings: &[Binning]) -> Result<(), Error> {
        if binnings.len() == self.coords.len() {
            Ok(())
        } else {
            Err(Error::Dimensions {
                expected: self.coords.len(),
                got: binnings.len(),
            })
        }
    }
}

/// The unmasked pixels of one frame with their positions. A masked pixel
/// adds nothing.
fn unmasked(
    points: &Projected,
    pixels: &NormalisedFrame,
) -> Result<Vec<(usize, Normalised)>, Error> {
    pixels.check("normalised frame", points.shape())?;
    Ok(pixels
        .data()
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().ok().map(|n| (i, *n)))
        .collect())
}

/// The unmasked pixels of one frame laid by their footprints on
/// `binnings`. Every pixel is checked here, before any state changes.
fn binned(
    binnings: &[Binning],
    points: &Projected,
    pixels: &NormalisedFrame,
) -> Result<Pieces, Error> {
    points.check_binnings(binnings)?;
    let taken = unmasked(points, pixels)?;
    let mut pieces = Pieces::with_capacity(binnings.len(), taken.len());
    for (i, n) in taken {
        let coords = points.planes.iter().map(|p| p.data()[i]);
        pieces.push(n, &points.footprints, binnings, coords)?;
    }
    Ok(pieces)
}

/// (first, last) per dimension of flattened `bins`, `None` when empty.
fn span(dims: usize, bins: &[i64]) -> Option<Vec<(i64, i64)>> {
    let mut rows = bins.chunks_exact(dims);
    let mut span: Vec<(i64, i64)> = rows.next()?.iter().map(|&k| (k, k)).collect();
    for row in rows {
        for ((lo, hi), &k) in span.iter_mut().zip(row) {
            *lo = (*lo).min(k);
            *hi = (*hi).max(k);
        }
    }
    Some(span)
}

/// First pass of two-pass filling: the bins the unmasked pixels of every
/// frame seen so far fall in. [`Bounds::cube`] then allocates a cube every
/// one of those pixels lands in.
#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    binnings: Vec<Binning>,
    seen: Option<Seen>,
}

#[derive(Clone, Debug, PartialEq)]
struct Seen {
    coords: Vec<Coord>,
    span: Vec<(i64, i64)>,
}

impl Bounds {
    pub fn new(binnings: Vec<Binning>) -> Self {
        Self {
            binnings,
            seen: None,
        }
    }

    pub fn include(&mut self, points: &Projected, pixels: &NormalisedFrame) -> Result<(), Error> {
        if let Some(seen) = &self.seen
            && seen.coords != points.coords
        {
            return Err(Error::GridMismatch {
                left: describe_coords(&seen.coords, &self.binnings),
                right: describe_coords(&points.coords, &self.binnings),
            });
        }
        let pieces = binned(&self.binnings, points, pixels)?;
        let Some(frame) = span(self.binnings.len(), pieces.bins()) else {
            return Ok(());
        };
        match &mut self.seen {
            None => {
                self.seen = Some(Seen {
                    coords: points.coords.clone(),
                    span: frame,
                })
            }
            Some(seen) => {
                for ((lo, hi), (flo, fhi)) in seen.span.iter_mut().zip(frame) {
                    *lo = (*lo).min(flo);
                    *hi = (*hi).max(fhi);
                }
            }
        }
        Ok(())
    }

    /// A zero cube over the bins seen; the empty cube when no unmasked
    /// pixel was included.
    pub fn cube(self) -> Result<Cube, Error> {
        match self.seen {
            None => Ok(Cube::default()),
            Some(seen) => Cube::zeros(axes_over(&seen.coords, &self.binnings, &seen.span)?),
        }
    }
}

fn describe_coords(coords: &[Coord], binnings: &[Binning]) -> Vec<(Coord, Binning)> {
    coords
        .iter()
        .cloned()
        .zip(binnings.iter().copied())
        .collect()
}

fn axes_over(
    coords: &[Coord],
    binnings: &[Binning],
    span: &[(i64, i64)],
) -> Result<Vec<Axis>, Error> {
    coords
        .iter()
        .zip(binnings)
        .zip(span)
        .map(|((c, b), &s)| Axis::spanning(c, *b, s))
        .collect()
}

/// Σcounts (photons), Σweight and contributions per bin, row-major over
/// `axes` (the last axis fastest). A cube without axes is the empty cube,
/// the identity of [`Cube::merge`]; every other cube has at least one bin
/// per axis.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cube {
    axes: Vec<Axis>,
    photons: Vec<f64>,
    weight: Vec<f64>,
    contributions: Vec<u64>,
}

impl Cube {
    fn zeros(axes: Vec<Axis>) -> Result<Self, Error> {
        let n = axes
            .iter()
            .try_fold(1usize, |n, a| n.checked_mul(a.size))
            .ok_or(Error::CubeSize)?;
        let mut photons = Vec::new();
        let mut weight = Vec::new();
        let mut contributions = Vec::new();
        photons.try_reserve_exact(n).map_err(|_| Error::CubeSize)?;
        weight.try_reserve_exact(n).map_err(|_| Error::CubeSize)?;
        contributions
            .try_reserve_exact(n)
            .map_err(|_| Error::CubeSize)?;
        photons.resize(n, 0.0);
        weight.resize(n, 0.0);
        contributions.resize(n, 0);
        Ok(Self {
            axes,
            photons,
            weight,
            contributions,
        })
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// Σcounts per bin.
    pub fn photons(&self) -> &[f64] {
        &self.photons
    }

    /// Σweight per bin.
    pub fn weight(&self) -> &[f64] {
        &self.weight
    }

    pub fn contributions(&self) -> &[u64] {
        &self.contributions
    }

    /// Σcounts/Σweight per bin; `None` where Σweight is 0, a bin no pixel
    /// reached.
    pub fn intensity(&self) -> impl ExactSizeIterator<Item = Option<f64>> + '_ {
        self.photons
            .iter()
            .zip(&self.weight)
            .map(|(&c, &w)| (w != 0.0).then(|| c / w))
    }

    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    /// Linear position of the bin with global indices `bins`, worked out
    /// from the axes on every call; `None` outside the cube.
    pub fn offset(&self, bins: &[i64]) -> Option<usize> {
        if bins.len() != self.axes.len() || self.axes.is_empty() {
            return None;
        }
        self.axes
            .iter()
            .zip(bins)
            .try_fold(0, |off, (a, &k)| Some(off * a.size + a.position(k)?))
    }

    /// Global indices of the bin at linear position `offset`.
    fn bins_at(&self, mut offset: usize, out: &mut [i64]) {
        for (k, a) in out.iter_mut().zip(&self.axes).rev() {
            *k = a.first + (offset % a.size) as i64;
            offset /= a.size;
        }
    }

    fn check_grid(&self, coords: &[Coord], binnings: &[Binning]) -> Result<(), Error> {
        let same = self.axes.len() == coords.len()
            && self
                .axes
                .iter()
                .zip(coords.iter().zip(binnings))
                .all(|(a, (c, b))| a.coord == *c && a.binning == *b);
        if same {
            Ok(())
        } else {
            Err(Error::GridMismatch {
                left: self.axes.iter().map(Axis::describe).collect(),
                right: describe_coords(coords, binnings),
            })
        }
    }

    /// Adds every unmasked pixel of `pixels`, the normalised frame of
    /// `points`, to the bins its footprint reaches. The axes stay as they
    /// are: the part of a footprint outside them is not added. Returns how
    /// many unmasked pixels were not wholly inside. On error the cube is
    /// unchanged.
    pub fn add(&mut self, points: &Projected, pixels: &NormalisedFrame) -> Result<usize, Error> {
        let (rows, outside) = self.placed(points, pixels)?;
        self.sums().deposit(&rows);
        Ok(outside)
    }

    /// [`add`](Self::add) of each of `frames`, (points, pixels), in order,
    /// in parallel: every bin takes its pixels in frame then pixel order, so
    /// the sums are those of adding the frames one by one. Every frame is
    /// checked before any state changes; the error is the first failing
    /// frame's and leaves the cube unchanged. Returns how many unmasked
    /// pixels fell outside the axes in all.
    pub fn add_frames(
        &mut self,
        frames: &[(&Projected, &NormalisedFrame)],
    ) -> Result<usize, Error> {
        let placed: Vec<_> = frames
            .par_iter()
            .map(|&(points, pixels)| {
                let (mut rows, outside) = self.placed(points, pixels)?;
                // stable: a bin keeps its pixels in pixel order
                rows.sort_by_key(|r| r.0);
                Ok((rows, outside))
            })
            .collect();
        let placed = placed.into_iter().collect::<Result<Vec<_>, Error>>()?;
        let outside = placed.iter().map(|p| p.1).sum();
        let rows: Vec<_> = placed.into_iter().map(|p| p.0).collect();

        // the cube cut at quantiles of a sample of the offsets; each piece is
        // summed by one task, frame after frame
        let tasks = 4 * rayon::current_num_threads();
        let total: usize = rows.iter().map(Vec::len).sum();
        let stride = (total / (16 * tasks)).max(1);
        let mut sample: Vec<usize> = rows
            .iter()
            .flat_map(|r| r.iter().step_by(stride).map(|r| r.0))
            .collect();
        sample.sort_unstable();
        let cuts = (1..tasks).filter_map(|s| sample.get(s * sample.len() / tasks).copied());
        let mut rest = self.sums();
        let mut pieces: Vec<_> = cuts.map(|hi| rest.split_below(hi)).collect();
        pieces.push(rest);
        pieces.into_par_iter().for_each(|mut piece| {
            let (lo, hi) = (piece.lo, piece.lo + piece.photons.len());
            for rows in &rows {
                let first = rows.partition_point(|r| r.0 < lo);
                let end = rows.partition_point(|r| r.0 < hi);
                piece.deposit(&rows[first..end]);
            }
        });
        Ok(outside)
    }

    fn sums(&mut self) -> Sums<'_> {
        Sums {
            lo: 0,
            photons: &mut self.photons,
            weight: &mut self.weight,
            contributions: &mut self.contributions,
        }
    }

    /// The pieces of one frame's unmasked pixels inside the axes in pixel
    /// order, and how many pixels were not wholly inside.
    fn placed(
        &self,
        points: &Projected,
        pixels: &NormalisedFrame,
    ) -> Result<(Vec<Row>, usize), Error> {
        if self.is_empty() {
            return Ok((Vec::new(), unmasked(points, pixels)?.len()));
        }
        let binnings: Vec<Binning> = self.axes.iter().map(|a| a.binning).collect();
        self.check_grid(&points.coords, &binnings)?;
        Ok(self.rows(&binned(&binnings, points, pixels)?))
    }

    /// One-pass filling: widens the axes to take every unmasked pixel of
    /// this frame, then adds them. An empty cube takes its axes from
    /// `binnings` and the projection's coordinates. On error the cube is
    /// unchanged.
    pub fn add_growing(
        &mut self,
        binnings: &[Binning],
        points: &Projected,
        pixels: &NormalisedFrame,
    ) -> Result<(), Error> {
        if !self.is_empty() {
            self.check_grid(&points.coords, binnings)?;
        }
        let pieces = binned(binnings, points, pixels)?;
        let Some(span) = span(binnings.len(), pieces.bins()) else {
            return Ok(());
        };
        let axes = union(&self.axes, &axes_over(&points.coords, binnings, &span)?)?;
        if axes != self.axes {
            *self = Self::zeros(axes)?.with(self);
        }
        let (rows, outside) = self.rows(&pieces);
        debug_assert_eq!(outside, 0, "the axes span every unmasked pixel");
        self.sums().deposit(&rows);
        Ok(())
    }

    /// The pieces inside the axes as rows, in piece order, and how many
    /// pixels had a piece outside.
    fn rows(&self, pieces: &Pieces) -> (Vec<Row>, usize) {
        let mut rows = Vec::with_capacity(pieces.len());
        let mut outside = 0;
        for (n, ps) in pieces.pixels() {
            let mut inside = true;
            for (k, share) in ps {
                match self.offset(k) {
                    Some(off) => rows.push((off, share, n)),
                    None => inside = false,
                }
            }
            outside += usize::from(!inside);
        }
        (rows, outside)
    }

    /// `self` plus `other` bin by bin; `other`'s axes lie inside `self`'s.
    fn with(mut self, other: &Self) -> Self {
        let mut k = vec![0; other.axes.len()];
        let sums = other.photons.iter().zip(&other.weight);
        for (j, ((&p, &w), &c)) in sums.zip(&other.contributions).enumerate() {
            other.bins_at(j, &mut k);
            let off = self.offset(&k).expect("the axes contain the other cube");
            self.photons[off] += p;
            self.weight[off] += w;
            self.contributions[off] += c;
        }
        self
    }

    /// Sum of two cubes over the union of their axes, bin by bin: Σcounts,
    /// Σweight and contributions. The empty cube is the
    /// identity; otherwise both need the same coordinates (name and kind)
    /// and binnings in the same order.
    pub fn merge(self, other: Self) -> Result<Self, Error> {
        if other.is_empty() {
            return Ok(self);
        }
        if self.is_empty() {
            return Ok(other);
        }
        let axes = union(&self.axes, &other.axes)?;
        let out = if axes == self.axes {
            self
        } else {
            Self::zeros(axes)?.with(&self)
        };
        Ok(out.with(&other))
    }
}

/// One piece of a pixel: (offset, share, value).
type Row = (usize, f64, Normalised);

/// The sums of a cube's bins from offset `lo` on, as many as the slices
/// hold.
struct Sums<'a> {
    lo: usize,
    photons: &'a mut [f64],
    weight: &'a mut [f64],
    contributions: &'a mut [u64],
}

impl<'a> Sums<'a> {
    /// Splits off the bins below offset `hi`.
    fn split_below(&mut self, hi: usize) -> Sums<'a> {
        let n = hi - self.lo;
        let (photons, rest) = std::mem::take(&mut self.photons).split_at_mut(n);
        self.photons = rest;
        let (weight, rest) = std::mem::take(&mut self.weight).split_at_mut(n);
        self.weight = rest;
        let (contributions, rest) = std::mem::take(&mut self.contributions).split_at_mut(n);
        self.contributions = rest;
        let lo = std::mem::replace(&mut self.lo, hi);
        Sums {
            lo,
            photons,
            weight,
            contributions,
        }
    }

    /// Adds each piece to the bin at its offset: its share of the counts
    /// and of the weight, and one contribution, as a bin of a cut takes a
    /// pixel.
    fn deposit(&mut self, rows: &[Row]) {
        for &(off, share, n) in rows {
            let i = off - self.lo;
            self.photons[i] += share * n.counts();
            self.weight[i] += share * n.weight().get();
            self.contributions[i] += 1;
        }
    }
}

/// Axes spanning both `a` and `b`; `a` may be empty.
fn union(a: &[Axis], b: &[Axis]) -> Result<Vec<Axis>, Error> {
    if a.is_empty() {
        return Ok(b.to_vec());
    }
    let same = a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.coord == y.coord && x.binning == y.binning);
    if !same {
        return Err(Error::GridMismatch {
            left: a.iter().map(Axis::describe).collect(),
            right: b.iter().map(Axis::describe).collect(),
        });
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            Axis::spanning(
                &x.coord,
                x.binning,
                (x.first.min(y.first), x.last().max(y.last())),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absorption::{Masked, Weight};

    const ONE: &[&str] = &["x"];
    const TWO: &[&str] = &["x", "y"];

    fn unit() -> Binning {
        Binning::new(1.0, 0.0).unwrap()
    }

    fn linear(names: &[&'static str]) -> Vec<Coord> {
        names.iter().map(|&n| Coord::linear(n)).collect()
    }

    /// Points of `kinds` as one row of pixels, shape (1, n).
    fn points_of(kinds: Vec<Coord>, coords: &[&[f64]]) -> Projected {
        let planes = (0..kinds.len())
            .map(|d| Image::from_fn((1, coords.len()), |_, i| coords[i][d]))
            .collect();
        Projected::new(kinds, planes).unwrap()
    }

    /// Linear points as one row of pixels, shape (1, n).
    fn points(names: &[&'static str], coords: &[&[f64]]) -> Projected {
        points_of(linear(names), coords)
    }

    /// `pixels` as one row, the shape [`points`] gives.
    fn row(pixels: &[Result<Normalised, Masked>]) -> NormalisedFrame {
        Image::from_fn((1, pixels.len()), |_, i| pixels[i])
    }

    fn pixel(counts: f64, weight: f64) -> Result<Normalised, Masked> {
        Normalised::new(counts, Weight::new(weight, Masked::Opaque))
    }

    /// Pixels of `counts`, each of weight 1.
    fn unweighted(counts: &[f64]) -> NormalisedFrame {
        Image::from_fn((1, counts.len()), |_, i| pixel(counts[i], 1.0))
    }

    /// A cube filled in one pass from points and normalised pixels.
    fn filled(
        names: &'static [&'static str],
        coords: &[&[f64]],
        pixels: &[Result<Normalised, Masked>],
    ) -> Cube {
        let mut c = Cube::default();
        c.add_growing(
            &vec![unit(); names.len()],
            &points(names, coords),
            &row(pixels),
        )
        .unwrap();
        c
    }

    fn cube(names: &'static [&'static str], coords: &[&[f64]], counts: &[f64]) -> Cube {
        filled(names, coords, unweighted(counts).data())
    }

    /// binoculars' rule (hkl-binoculars.c:375): rint, ties to even.
    fn rint_bin(v: f64, resolution: f64) -> i64 {
        (v / resolution).round_ties_even() as i64
    }

    #[test]
    fn floor_takes_the_lower_edge_on_every_bin() {
        let b = unit();
        for k in -3..=3 {
            let edge = k as f64;
            assert_eq!(b.bin(edge).unwrap(), k, "edge {edge}");
            assert_eq!(b.bin(edge - 1e-12).unwrap(), k - 1, "below {edge}");
            assert_eq!(b.edge(k), edge);
        }
        let shifted = Binning::new(0.5, 0.25).unwrap();
        assert_eq!(shifted.bin(0.25).unwrap(), 0);
        assert_eq!(shifted.bin(0.2499).unwrap(), -1);
        assert_eq!(shifted.bin(0.75).unwrap(), 1);
    }

    #[test]
    fn floor_diverges_from_binoculars_rint() {
        // anchor −½ gives binoculars' bins centred on the integers; the two
        // rules then differ only on the edges, which rint sends to the even
        // neighbour (0.5 down, 1.5 up) and floor always up.
        let centred = Binning::new(1.0, -0.5).unwrap();
        let mut differ = Vec::new();
        for v in [-1.5, -0.5, 0.5, 1.5, 2.5, 0.49, 1.51, -0.7, 3.0] {
            let (f, r) = (centred.bin(v).unwrap(), rint_bin(v, 1.0));
            if f != r {
                differ.push((v, f, r));
            }
        }
        assert_eq!(differ, [(-1.5, -1, -2), (0.5, 1, 0), (2.5, 3, 2)]);
        // with the default anchor every bin is shifted by half a step
        assert_eq!((unit().bin(0.7).unwrap(), rint_bin(0.7, 1.0)), (0, 1));
    }

    #[test]
    fn values_without_a_bin_are_errors() {
        let b = unit();
        for v in [f64::NAN, f64::INFINITY, -f64::INFINITY, 1e300] {
            assert!(matches!(b.bin(v), Err(Error::NoBin { .. })), "{v}");
        }
        assert!(Binning::new(0.0, 0.0).is_err());
        assert!(Binning::new(f64::NAN, 0.0).is_err());
        assert!(Binning::new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn axis_origin_is_the_lower_edge_of_its_first_bin() {
        let c = cube(ONE, &[&[-2.5], &[4.0]], &[1.0, 1.0]);
        let a = &c.axes()[0];
        assert_eq!((a.first(), a.size(), a.origin()), (-3, 8, -3.0));
        assert_eq!(c.contributions()[0], 1);
        assert_eq!(c.contributions()[7], 1);
    }

    fn sample_cubes() -> [Cube; 4] {
        // dyadic counts and weights: every sum is exact, so associativity
        // holds bit for bit and the test does not depend on the order of
        // additions
        let px = |a: (f64, f64), b: (f64, f64)| [pixel(a.0, a.1), pixel(b.0, b.1)];
        [
            filled(
                TWO,
                &[&[0.5, 0.5], &[2.5, 1.5]],
                &px((1.25, 0.5), (2.5, 2.0)),
            ),
            // overlaps the first
            filled(
                TWO,
                &[&[2.0, 1.0], &[3.5, -1.5]],
                &px((0.5, 4.0), (4.0, 1.0)),
            ),
            // disjoint from both
            filled(
                TWO,
                &[&[10.0, 7.0], &[11.0, 9.0]],
                &px((8.0, 0.25), (0.125, 8.0)),
            ),
            // same span as the first
            filled(
                TWO,
                &[&[0.25, 1.75], &[2.0, 0.0]],
                &px((3.0, 1.0), (0.75, 0.5)),
            ),
        ]
    }

    #[test]
    fn merge_identity() {
        for c in sample_cubes() {
            assert_eq!(Cube::default().merge(c.clone()).unwrap(), c);
            assert_eq!(c.clone().merge(Cube::default()).unwrap(), c);
        }
        assert_eq!(
            Cube::default().merge(Cube::default()).unwrap(),
            Cube::default()
        );
    }

    #[test]
    fn merge_is_commutative_on_overlapping_and_disjoint_axes() {
        let cs = sample_cubes();
        for a in &cs {
            for b in &cs {
                let ab = a.clone().merge(b.clone()).unwrap();
                let ba = b.clone().merge(a.clone()).unwrap();
                assert_eq!(ab, ba);
                let sum = |c: &Cube, f: fn(&Cube) -> &[f64]| f(c).iter().sum::<f64>();
                for f in [Cube::photons, Cube::weight] {
                    assert_eq!(sum(&ab, f), sum(a, f) + sum(b, f));
                }
                let n = |c: &Cube| c.contributions().iter().sum::<u64>();
                assert_eq!(n(&ab), n(a) + n(b));
            }
        }
    }

    #[test]
    fn merge_is_associative_on_overlapping_and_disjoint_axes() {
        let cs = sample_cubes();
        for a in &cs {
            for b in &cs {
                for c in &cs {
                    let left = a.clone().merge(b.clone()).unwrap().merge(c.clone());
                    let right = a.clone().merge(b.clone().merge(c.clone()).unwrap());
                    assert_eq!(left.unwrap(), right.unwrap());
                }
            }
        }
    }

    #[test]
    fn merge_spans_the_gap_between_disjoint_cubes() {
        let [a, _, c, _] = sample_cubes();
        let m = a.merge(c).unwrap();
        let spans: Vec<_> = m.axes().iter().map(|a| (a.first(), a.size())).collect();
        assert_eq!(spans, [(0, 12), (0, 10)]);
        assert_eq!(m.contributions().iter().sum::<u64>(), 4);
        assert_eq!(m.offset(&[5, 5]).map(|o| m.contributions()[o]), Some(0));
    }

    #[test]
    fn merge_of_different_grids_is_an_error() {
        let a = cube(TWO, &[&[0.0, 0.0]], &[1.0]);
        let names = cube(&["x", "z"], &[&[0.0, 0.0]], &[1.0]);
        let dims = cube(ONE, &[&[0.0]], &[1.0]);
        let mut step = Cube::default();
        step.add_growing(
            &[unit(), Binning::new(0.5, 0.0).unwrap()],
            &points(TWO, &[&[0.0, 0.0]]),
            &unweighted(&[1.0]),
        )
        .unwrap();
        let mut anchor = Cube::default();
        anchor
            .add_growing(
                &[unit(), Binning::new(1.0, 0.5).unwrap()],
                &points(TWO, &[&[0.0, 0.0]]),
                &unweighted(&[1.0]),
            )
            .unwrap();
        for other in [names, dims, step, anchor] {
            assert!(matches!(
                a.clone().merge(other),
                Err(Error::GridMismatch { .. })
            ));
        }
    }

    /// Grids that differ only in a coordinate's kind name it in the error,
    /// at every site that refuses them.
    #[test]
    fn a_grid_mismatch_names_the_coordinate_kind() {
        let angle = points_of(vec![Coord::angle("x")], &[&[0.5]]);
        let line = points(ONE, &[&[0.5]]);
        let one = unweighted(&[1.0]);
        let mismatch = Error::GridMismatch {
            left: vec![(Coord::linear("x"), unit())],
            right: vec![(Coord::angle("x"), unit())],
        };
        let a = cube(ONE, &[&[0.5]], &[1.0]);
        let mut b = Cube::default();
        b.add_growing(&[unit()], &angle, &one).unwrap();
        assert_eq!(a.clone().merge(b), Err(mismatch.clone()));
        assert_eq!(a.clone().add(&angle, &one), Err(mismatch.clone()));
        assert_eq!(
            a.clone().add_growing(&[unit()], &angle, &one),
            Err(mismatch.clone())
        );
        let mut bounds = Bounds::new(vec![unit()]);
        bounds.include(&line, &one).unwrap();
        assert_eq!(bounds.include(&angle, &one), Err(mismatch.clone()));
        let text = mismatch.to_string();
        assert!(text.contains("Linear") && text.contains("Angle"), "{text}");
    }

    /// An angle and a linear coordinate of one name and binning are two
    /// axes: every way of meeting both refuses and changes nothing.
    #[test]
    fn a_coordinate_kind_is_part_of_the_grid() {
        let angle = points_of(vec![Coord::angle("x")], &[&[0.5]]);
        let line = points(ONE, &[&[0.5]]);
        let one = unweighted(&[1.0]);
        let a = cube(ONE, &[&[0.5]], &[1.0]);
        assert_eq!(a.axes()[0].kind(), CoordKind::Linear);
        let mut b = Cube::default();
        b.add_growing(&[unit()], &angle, &one).unwrap();
        assert_eq!(b.axes()[0].coord(), &Coord::angle("x"));
        assert!(matches!(
            a.clone().merge(b.clone()),
            Err(Error::GridMismatch { .. })
        ));
        for mut c in [a.clone(), b.clone()] {
            let before = c.clone();
            let other = if c == a { &angle } else { &line };
            assert!(matches!(
                c.add(other, &one),
                Err(Error::GridMismatch { .. })
            ));
            assert!(matches!(
                c.add_growing(&[unit()], other, &one),
                Err(Error::GridMismatch { .. })
            ));
            assert_eq!(c, before);
        }
        let mut bounds = Bounds::new(vec![unit()]);
        bounds.include(&line, &one).unwrap();
        assert!(matches!(
            bounds.include(&angle, &one),
            Err(Error::GridMismatch { .. })
        ));
    }

    #[test]
    fn two_pass_equals_one_pass() {
        let frames = [
            points(TWO, &[&[0.5, 0.5], &[2.5, 1.5], &[-4.0, 3.0]]),
            points(TWO, &[&[9.0, -2.0], &[2.0, 1.0], &[0.1, 0.2]]),
        ];
        let pixels = [
            row(&[pixel(1.0, 0.5), Err(Masked::Unlit), pixel(3.0, 2.0)]),
            row(&[pixel(0.1, 0.5), Err(Masked::Unlit), pixel(0.3, 2.0)]),
        ];
        let binnings = vec![Binning::new(0.75, 0.1).unwrap(), unit()];
        let mut bounds = Bounds::new(binnings.clone());
        for (f, px) in frames.iter().zip(&pixels) {
            bounds.include(f, px).unwrap();
        }
        let mut two = bounds.cube().unwrap();
        let mut one = Cube::default();
        for (f, px) in frames.iter().zip(&pixels) {
            assert_eq!(two.add(f, px).unwrap(), 0);
            one.add_growing(&binnings, f, px).unwrap();
        }
        assert_eq!(one, two);
        assert_eq!(two.contributions().iter().sum::<u64>(), 4);
    }

    fn on_threads<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(f)
    }

    #[test]
    fn add_frames_sums_each_bin_in_frame_then_pixel_order_on_any_thread_count() {
        // ±1e16 + 1 rounds back to ±1e16: bin 1 ends at 1 only if it takes
        // 1e16, the ones, -1e16 and the last one in this order
        let frames = [
            (
                points(ONE, &[&[0.5], &[1.5], &[2.5]]),
                [3.0, 1e16, 4.0].as_slice(),
            ),
            (
                points(ONE, &[&[1.5], &[1.5], &[1.5], &[7.0]]),
                &[1.0, 1.0, 1.0, 5.0],
            ),
            (points(ONE, &[&[1.5], &[2.5], &[1.5]]), &[-1e16, 2.0, 1.0]),
        ]
        .map(|(p, c)| (p, unweighted(c)));
        let start = cube(ONE, &[&[0.0], &[2.0]], &[0.0, 0.0]);
        let mut expected = start.clone();
        for (p, px) in &frames {
            expected.add(p, px).unwrap();
        }
        assert_eq!(expected.photons, [3.0, 1.0, 6.0]);
        let frames: Vec<_> = frames.iter().map(|(p, px)| (p, px)).collect();
        for threads in [1, 2, 3, 5, 12] {
            let mut c = start.clone();
            assert_eq!(on_threads(threads, || c.add_frames(&frames)).unwrap(), 1);
            assert_eq!(c, expected, "{threads} threads");
        }
    }

    #[test]
    fn add_frames_fails_on_the_first_failing_frame_on_any_thread_count() {
        let frames: Vec<_> = (0..40)
            .map(|f| {
                let x = match f {
                    17 => 1e300,
                    30 => f64::NAN,
                    35 => -1e300,
                    _ => 0.5,
                };
                (points(ONE, &[&[0.5], &[x]]), unweighted(&[1.0, 1.0]))
            })
            .collect();
        let frames: Vec<_> = frames.iter().map(|(p, px)| (p, px)).collect();
        let start = cube(ONE, &[&[0.0]], &[0.0]);
        for threads in [1, 4, 16] {
            let mut c = start.clone();
            let e = on_threads(threads, || c.add_frames(&frames));
            assert!(
                matches!(e, Err(Error::NoBin { value, .. }) if value == 1e300),
                "{threads} threads: {e:?}"
            );
            assert_eq!(c, start);
        }
    }

    #[test]
    fn fixed_axes_report_pixels_outside() {
        let mut c = cube(ONE, &[&[0.0], &[2.0]], &[1.0, 1.0]);
        let out = c
            .add(
                &points(ONE, &[&[1.0], &[5.0], &[-1.0]]),
                &unweighted(&[1.0; 3]),
            )
            .unwrap();
        assert_eq!(out, 2);
        assert_eq!(c.contributions(), [1, 1, 1]);
        assert_eq!(c.weight(), [1.0, 1.0, 1.0]);
        assert_eq!(
            Cube::default()
                .add(&points(ONE, &[&[1.0]]), &unweighted(&[1.0]))
                .unwrap(),
            1
        );
    }

    #[test]
    fn masked_pixels_contribute_nothing_and_do_not_widen_the_axes() {
        let p = points(ONE, &[&[0.0], &[100.0], &[1.0]]);
        let px = row(&[pixel(1.0, 0.5), Err(Masked::GrazingExit), pixel(2.0, 4.0)]);
        let mut c = Cube::default();
        c.add_growing(&[unit()], &p, &px).unwrap();
        assert_eq!(c.axes()[0].size(), 2);
        assert_eq!(c.photons(), [1.0, 2.0]);
        assert_eq!(c.weight(), [0.5, 4.0]);
        assert_eq!(c.contributions(), [1, 1]);
        let mut b = Bounds::new(vec![unit()]);
        b.include(&p, &row(&[Err(Masked::Unlit); 3])).unwrap();
        assert_eq!(b.cube().unwrap(), Cube::default());
    }

    #[test]
    fn a_nan_count_is_masked_as_bad_count_and_adds_nothing() {
        let p = points(ONE, &[&[0.0], &[5.0]]);
        let nan = unweighted(&[1.0, f64::NAN]);
        assert_eq!(nan.data()[1], Err(Masked::BadCount));
        let want = cube(ONE, &[&[0.0]], &[1.0]);
        assert_eq!(filled(ONE, &[&[0.0], &[5.0]], nan.data()), want);
        let mut b = Bounds::new(vec![unit()]);
        b.include(&p, &nan).unwrap();
        let mut c = b.cube().unwrap();
        assert_eq!(c.add(&p, &nan).unwrap(), 0);
        assert_eq!(c, want);
    }

    #[test]
    fn shape_and_dimension_mismatches_are_errors() {
        let p = points(ONE, &[&[0.0], &[1.0]]);
        let mut c = Cube::default();
        assert_eq!(
            c.add_growing(&[unit()], &p, &unweighted(&[1.0])),
            Err(Error::ShapeMismatch {
                what: "normalised frame",
                shape: (1, 1),
                expected: (1, 2)
            })
        );
        assert!(matches!(
            c.add_growing(&[unit(), unit()], &p, &unweighted(&[1.0, 1.0])),
            Err(Error::Dimensions { .. })
        ));
        let column = |v: Vec<f64>| Image::new((v.len(), 1), v).unwrap();
        assert_eq!(
            Projected::new(
                linear(TWO),
                vec![
                    column(vec![0.0, 1.0]),
                    points(ONE, &[&[0.0], &[1.0]]).planes()[0].clone()
                ]
            ),
            Err(Error::ShapeMismatch {
                what: "projection plane",
                shape: (1, 2),
                expected: (2, 1)
            })
        );
        assert_eq!(c, Cube::default());
        // a pixel with no bin after good ones leaves a filled cube as it was
        let mut c = cube(ONE, &[&[0.0]], &[1.0]);
        let before = c.clone();
        let far = points(ONE, &[&[0.0], &[f64::NAN]]);
        let two = unweighted(&[1.0, 1.0]);
        assert!(matches!(c.add(&far, &two), Err(Error::NoBin { .. })));
        assert!(matches!(
            c.add_growing(&[unit()], &far, &two),
            Err(Error::NoBin { .. })
        ));
        assert_eq!(c, before);
    }

    /// Points over a (2, 1) column and pixels over a (1, 2) row have the
    /// same count; every way in refuses the pair and changes nothing.
    #[test]
    fn a_transposed_frame_is_refused() {
        let column = Projected::new(
            linear(ONE),
            vec![Image::new((2, 1), vec![0.0, 1.0]).unwrap()],
        )
        .unwrap();
        let row = unweighted(&[1.0, 1.0]);
        let refused = Err(Error::ShapeMismatch {
            what: "normalised frame",
            shape: (1, 2),
            expected: (2, 1),
        });
        let mut empty = Cube::default();
        assert_eq!(empty.add(&column, &row).map(drop), refused);
        assert_eq!(empty.add_growing(&[unit()], &column, &row), refused);
        assert_eq!(empty, Cube::default());
        let mut c = cube(ONE, &[&[0.0], &[1.0]], &[1.0, 1.0]);
        let before = c.clone();
        assert_eq!(c.add(&column, &row).map(drop), refused);
        assert_eq!(c.add_growing(&[unit()], &column, &row), refused);
        assert_eq!(c, before);
        let mut b = Bounds::new(vec![unit()]);
        assert_eq!(b.include(&column, &row), refused);
        assert_eq!(b, Bounds::new(vec![unit()]));
    }

    #[test]
    fn intensity_is_the_ratio_of_sums_and_absent_where_nothing_fell() {
        // bin 0 takes 3 of weight 1 and 1 of weight 3: the ratio of sums is
        // 1, the mean of their corrected values 5/3
        let coords: &[&[f64]] = &[&[0.0], &[0.5], &[2.0]];
        let c = filled(
            ONE,
            coords,
            &[pixel(3.0, 1.0), pixel(1.0, 3.0), pixel(6.0, 4.0)],
        );
        assert_eq!(c.photons(), [4.0, 0.0, 6.0]);
        assert_eq!(c.weight(), [4.0, 0.0, 4.0]);
        assert_eq!(c.contributions(), [2, 0, 1]);
        let ratio = |c: &Cube| c.intensity().collect::<Vec<_>>();
        assert_eq!(ratio(&c), [Some(1.0), None, Some(1.5)]);
        // merged cubes sum before the ratio
        let m = c.merge(filled(ONE, &[&[2.0]], &[pixel(2.0, 4.0)])).unwrap();
        assert_eq!(ratio(&m), [Some(1.0), None, Some(1.0)]);
    }

    #[test]
    fn bin4_intensity_is_never_rounded() {
        // BIN-4: binoculars' angles projection stores rint(image·w/flat)
        // (hkl-binoculars.c:378): image 1 at 0.2, 0.4, 0.6, 0.8 gives
        // 0, 0, 1, 1 (repro d3). Here those are the counts.
        let c = cube(
            ONE,
            &[&[0.0], &[1.0], &[2.0], &[3.0]],
            &[0.2, 0.4, 0.6, 0.8],
        );
        assert_eq!(c.photons(), [0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn bin5_photons_count_past_two_to_the_24() {
        // BIN-5: float photons stay at 16777216 after +1 while contributions
        // reach 2 (repro d3)
        let mut c = cube(ONE, &[&[0.0]], &[16_777_216.0]);
        c.add(&points(ONE, &[&[0.0]]), &unweighted(&[1.0])).unwrap();
        assert_eq!(c.photons(), [16_777_217.0]);
        assert_eq!(c.contributions(), [2]);
    }

    #[test]
    fn bin6_merged_cube_adds_at_the_right_bin() {
        // BIN-6: after new_merge binoculars keeps the first cube's offset0,
        // so adding A's pixel again lands in bin 10 instead of bin 0 (repro
        // d4: merge+sa [0]=1 [10]=1 [131]=1, new+sa [0]=2 [131]=1)
        let pa: &[&[f64]] = &[&[0.0, 1.0, 2.0]];
        let pb: &[&[f64]] = &[&[10.0, 11.0, 13.0]];
        let names: &'static [&'static str] = &["a", "b", "c"];
        let a = cube(names, pa, &[1.0]);
        let b = cube(names, pb, &[1.0]);
        let mut merged = a.merge(b).unwrap();
        let mut bounds = Bounds::new(vec![unit(); 3]);
        let one = unweighted(&[1.0]);
        bounds.include(&points(names, pa), &one).unwrap();
        bounds.include(&points(names, pb), &one).unwrap();
        let mut fresh = bounds.cube().unwrap();
        fresh.add(&points(names, pa), &one).unwrap();
        fresh.add(&points(names, pb), &one).unwrap();
        assert_eq!(merged, fresh);
        for c in [&mut merged, &mut fresh] {
            assert_eq!(c.add(&points(names, pa), &one).unwrap(), 0);
        }
        assert_eq!(merged, fresh);
        let at_a = merged.offset(&[0, 1, 2]).unwrap();
        assert_eq!(at_a, 0);
        assert_eq!(merged.contributions()[at_a], 2);
        assert_eq!(merged.contributions().iter().sum::<u64>(), 3);
    }

    #[test]
    fn bin7_merging_the_empty_cube_returns_the_other() {
        // BIN-7: binoculars' new_merge(empty, B) returns NULL (repro d4)
        let b = cube(TWO, &[&[0.98, 0.2]], &[1.0]);
        assert_eq!(Cube::default().merge(b.clone()).unwrap(), b);
    }
}
