//! Gridding (design §7): per-pixel projection coordinates and normalised
//! pixels summed into a cube of Σcounts, Σweight and contributions per bin,
//! the sums a bin of a cut holds ([`crate::cut::Bin`]).
//!
//! The one bin rule: on a dimension binned with width `step` from `anchor`,
//! a value v falls in global bin k = ⌊(v − anchor)/step⌋, the half-open
//! interval [anchor + k·step, anchor + (k+1)·step). [`Binning::bin`] is the
//! only place it is computed. binoculars 7be4f79 takes rint(v/resolution)
//! instead (hkl-binoculars.c:375, :842-844), bins centred on multiples of the
//! resolution whose edges go to the even neighbour.

use crate::Error;
use crate::absorption::Masked;
use crate::normalise::Normalised;

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

/// One cube axis: `size` consecutive global bins of `binning` from `first`.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    name: String,
    binning: Binning,
    first: i64,
    size: usize,
}

impl Axis {
    pub fn name(&self) -> &str {
        &self.name
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
    fn spanning(name: &str, binning: Binning, (first, last): (i64, i64)) -> Result<Self, Error> {
        let size = i64::try_from(i128::from(last) - i128::from(first) + 1)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::CubeSize)?;
        Ok(Self {
            name: name.to_owned(),
            binning,
            first,
            size,
        })
    }

    fn describe(&self) -> (String, f64, f64) {
        (self.name.clone(), self.binning.step, self.binning.anchor)
    }
}

/// One frame in projection coordinates: per dimension its name, which the
/// projection gives, and one value per pixel in the detector's row-major
/// order.
#[derive(Clone, Debug, PartialEq)]
pub struct Projected {
    names: &'static [&'static str],
    planes: Vec<Vec<f64>>,
}

impl Projected {
    pub fn new(names: &'static [&'static str], planes: Vec<Vec<f64>>) -> Result<Self, Error> {
        if names.is_empty() || planes.len() != names.len() {
            return Err(Error::Dimensions {
                expected: names.len(),
                got: planes.len(),
            });
        }
        let pixels = planes[0].len();
        for p in &planes {
            check_len("projection plane", p.len(), pixels)?;
        }
        Ok(Self { names, planes })
    }

    pub fn names(&self) -> &'static [&'static str] {
        self.names
    }

    pub fn planes(&self) -> &[Vec<f64>] {
        &self.planes
    }

    pub fn pixels(&self) -> usize {
        self.planes[0].len()
    }

    /// Global bins of pixel `i` into `out`.
    fn bins(&self, binnings: &[Binning], i: usize, out: &mut [i64]) -> Result<(), Error> {
        for ((k, b), plane) in out.iter_mut().zip(binnings).zip(&self.planes) {
            *k = b.bin(plane[i])?;
        }
        Ok(())
    }

    fn check_binnings(&self, binnings: &[Binning]) -> Result<(), Error> {
        if binnings.len() == self.names.len() {
            Ok(())
        } else {
            Err(Error::Dimensions {
                expected: self.names.len(),
                got: binnings.len(),
            })
        }
    }
}

fn check_len(what: &'static str, len: usize, expected: usize) -> Result<(), Error> {
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

/// The unmasked pixels of one frame with their positions, each with
/// finite counts. A masked pixel adds nothing.
fn unmasked(
    points: &Projected,
    pixels: &[Result<Normalised, Masked>],
) -> Result<Vec<(usize, Normalised)>, Error> {
    check_len("normalised frame", pixels.len(), points.pixels())?;
    pixels
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().ok().map(|n| (i, *n)))
        .map(|(i, n)| {
            if n.counts.is_finite() {
                Ok((i, n))
            } else {
                Err(Error::NotFinite {
                    what: "pixel counts",
                    value: n.counts,
                })
            }
        })
        .collect()
}

/// The unmasked pixels of one frame and their global bins, flattened
/// pixel-major. Every pixel is checked here, before any state changes.
fn binned(
    binnings: &[Binning],
    points: &Projected,
    pixels: &[Result<Normalised, Masked>],
) -> Result<(Vec<Normalised>, Vec<i64>), Error> {
    points.check_binnings(binnings)?;
    let taken = unmasked(points, pixels)?;
    let mut values = Vec::with_capacity(taken.len());
    let mut bins = Vec::with_capacity(taken.len() * binnings.len());
    let mut k = vec![0; binnings.len()];
    for (i, n) in taken {
        points.bins(binnings, i, &mut k)?;
        values.push(n);
        bins.extend_from_slice(&k);
    }
    Ok((values, bins))
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
    names: &'static [&'static str],
    span: Vec<(i64, i64)>,
}

impl Bounds {
    pub fn new(binnings: Vec<Binning>) -> Self {
        Self {
            binnings,
            seen: None,
        }
    }

    pub fn include(
        &mut self,
        points: &Projected,
        pixels: &[Result<Normalised, Masked>],
    ) -> Result<(), Error> {
        if let Some(seen) = &self.seen
            && seen.names != points.names
        {
            return Err(Error::GridMismatch {
                left: describe_names(seen.names, &self.binnings),
                right: describe_names(points.names, &self.binnings),
            });
        }
        let (_, bins) = binned(&self.binnings, points, pixels)?;
        let Some(frame) = span(self.binnings.len(), &bins) else {
            return Ok(());
        };
        match &mut self.seen {
            None => {
                self.seen = Some(Seen {
                    names: points.names,
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
            Some(seen) => Cube::zeros(axes_over(seen.names, &self.binnings, &seen.span)?),
        }
    }
}

fn describe_names(names: &[&str], binnings: &[Binning]) -> Vec<(String, f64, f64)> {
    names
        .iter()
        .zip(binnings)
        .map(|(n, b)| ((*n).to_owned(), b.step, b.anchor))
        .collect()
}

fn axes_over(
    names: &[&str],
    binnings: &[Binning],
    span: &[(i64, i64)],
) -> Result<Vec<Axis>, Error> {
    names
        .iter()
        .zip(binnings)
        .zip(span)
        .map(|((n, b), &s)| Axis::spanning(n, *b, s))
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

    fn check_grid(&self, names: &[&str], binnings: &[Binning]) -> Result<(), Error> {
        let same = self.axes.len() == names.len()
            && self
                .axes
                .iter()
                .zip(names.iter().zip(binnings))
                .all(|(a, (n, b))| a.name == *n && a.binning == *b);
        if same {
            Ok(())
        } else {
            Err(Error::GridMismatch {
                left: self.axes.iter().map(Axis::describe).collect(),
                right: describe_names(names, binnings),
            })
        }
    }

    /// Adds every unmasked pixel of `pixels`, the normalised frame of
    /// `points`, to the bin it falls in. The axes stay as they are; returns
    /// how many unmasked pixels fell outside them and were not added. On
    /// error the cube is unchanged.
    pub fn add(
        &mut self,
        points: &Projected,
        pixels: &[Result<Normalised, Masked>],
    ) -> Result<usize, Error> {
        if self.is_empty() {
            return Ok(unmasked(points, pixels)?.len());
        }
        let binnings: Vec<Binning> = self.axes.iter().map(|a| a.binning).collect();
        self.check_grid(points.names, &binnings)?;
        let (values, bins) = binned(&binnings, points, pixels)?;
        Ok(self.deposit(&bins, &values))
    }

    /// One-pass filling: widens the axes to take every unmasked pixel of
    /// this frame, then adds them. An empty cube takes its axes from
    /// `binnings` and the projection's names. On error the cube is
    /// unchanged.
    pub fn add_growing(
        &mut self,
        binnings: &[Binning],
        points: &Projected,
        pixels: &[Result<Normalised, Masked>],
    ) -> Result<(), Error> {
        if !self.is_empty() {
            self.check_grid(points.names, binnings)?;
        }
        let (values, bins) = binned(binnings, points, pixels)?;
        let Some(span) = span(binnings.len(), &bins) else {
            return Ok(());
        };
        let axes = union(&self.axes, &axes_over(points.names, binnings, &span)?)?;
        if axes != self.axes {
            *self = Self::zeros(axes)?.with(self);
        }
        let outside = self.deposit(&bins, &values);
        debug_assert_eq!(outside, 0, "the axes span every unmasked pixel");
        Ok(())
    }

    /// Adds each of `values` to the bin of its row of flattened `bins`: its
    /// counts, its weight and one contribution, as a bin of a cut takes a
    /// pixel. Returns how many rows fell outside the axes.
    fn deposit(&mut self, bins: &[i64], values: &[Normalised]) -> usize {
        let mut outside = 0;
        for (k, n) in bins.chunks_exact(self.axes.len()).zip(values) {
            match self.offset(k) {
                Some(off) => {
                    self.photons[off] += n.counts;
                    self.weight[off] += n.weight.get();
                    self.contributions[off] += 1;
                }
                None => outside += 1,
            }
        }
        outside
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
    /// identity; otherwise both need the same names and binnings in the
    /// same order.
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

/// Axes spanning both `a` and `b`; `a` may be empty.
fn union(a: &[Axis], b: &[Axis]) -> Result<Vec<Axis>, Error> {
    if a.is_empty() {
        return Ok(b.to_vec());
    }
    let same = a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.name == y.name && x.binning == y.binning);
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
                &x.name,
                x.binning,
                (x.first.min(y.first), x.last().max(y.last())),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absorption::Weight;

    const ONE: &[&str] = &["x"];
    const TWO: &[&str] = &["x", "y"];

    fn unit() -> Binning {
        Binning::new(1.0, 0.0).unwrap()
    }

    fn points(names: &'static [&'static str], coords: &[&[f64]]) -> Projected {
        let planes = (0..names.len())
            .map(|d| coords.iter().map(|c| c[d]).collect())
            .collect();
        Projected::new(names, planes).unwrap()
    }

    fn pixel(counts: f64, weight: f64) -> Result<Normalised, Masked> {
        Ok(Normalised {
            counts,
            weight: Weight::new(weight, Masked::Opaque)?,
        })
    }

    /// Pixels of `counts`, each of weight 1.
    fn unweighted(counts: &[f64]) -> Vec<Result<Normalised, Masked>> {
        counts.iter().map(|&c| pixel(c, 1.0)).collect()
    }

    /// A cube filled in one pass from points and normalised pixels.
    fn filled(
        names: &'static [&'static str],
        coords: &[&[f64]],
        pixels: &[Result<Normalised, Masked>],
    ) -> Cube {
        let mut c = Cube::default();
        c.add_growing(&vec![unit(); names.len()], &points(names, coords), pixels)
            .unwrap();
        c
    }

    fn cube(names: &'static [&'static str], coords: &[&[f64]], counts: &[f64]) -> Cube {
        filled(names, coords, &unweighted(counts))
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

    #[test]
    fn two_pass_equals_one_pass() {
        let frames = [
            points(TWO, &[&[0.5, 0.5], &[2.5, 1.5], &[-4.0, 3.0]]),
            points(TWO, &[&[9.0, -2.0], &[2.0, 1.0], &[0.1, 0.2]]),
        ];
        let pixels = [
            [pixel(1.0, 0.5), Err(Masked::Unlit), pixel(3.0, 2.0)],
            [pixel(0.1, 0.5), Err(Masked::Unlit), pixel(0.3, 2.0)],
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
        let px = [pixel(1.0, 0.5), Err(Masked::GrazingExit), pixel(2.0, 4.0)];
        let mut c = Cube::default();
        c.add_growing(&[unit()], &p, &px).unwrap();
        assert_eq!(c.axes()[0].size(), 2);
        assert_eq!(c.photons(), [1.0, 2.0]);
        assert_eq!(c.weight(), [0.5, 4.0]);
        assert_eq!(c.contributions(), [1, 1]);
        let mut b = Bounds::new(vec![unit()]);
        b.include(&p, &[Err(Masked::Unlit); 3]).unwrap();
        assert_eq!(b.cube().unwrap(), Cube::default());
    }

    #[test]
    fn non_finite_counts_and_length_mismatch_are_errors() {
        let p = points(ONE, &[&[0.0], &[1.0]]);
        let mut c = Cube::default();
        let nan = unweighted(&[1.0, f64::NAN]);
        assert!(matches!(
            c.add_growing(&[unit()], &p, &nan),
            Err(Error::NotFinite { .. })
        ));
        let mut b = Bounds::new(vec![unit()]);
        assert!(matches!(b.include(&p, &nan), Err(Error::NotFinite { .. })));
        assert!(matches!(c.add(&p, &nan), Err(Error::NotFinite { .. })));
        assert!(matches!(
            c.add_growing(&[unit()], &p, &unweighted(&[1.0])),
            Err(Error::PixelCount {
                what: "normalised frame",
                ..
            })
        ));
        assert!(matches!(
            c.add_growing(&[unit(), unit()], &p, &unweighted(&[1.0, 1.0])),
            Err(Error::Dimensions { .. })
        ));
        assert!(matches!(
            Projected::new(TWO, vec![vec![0.0], vec![]]),
            Err(Error::PixelCount { .. })
        ));
        assert_eq!(c, Cube::default());
        // a bad pixel after good ones leaves a filled cube as it was
        let mut c = cube(ONE, &[&[0.0]], &[1.0]);
        let before = c.clone();
        let far = points(ONE, &[&[0.0], &[5.0]]);
        assert!(c.add(&far, &unweighted(&[1.0, f64::INFINITY])).is_err());
        assert!(c.add_growing(&[unit()], &far, &nan).is_err());
        assert_eq!(c, before);
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
