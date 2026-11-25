//! Pixel footprints: how a pixel's counts and weight are laid on the bins
//! of a [`Binning`]. `Footprint::reach` is the one place a pixel's share
//! of a bin is computed.

use super::Binning;
use crate::Error;

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
    /// Refuses a width that is not positive and finite.
    pub(crate) fn check(self) -> Result<Self, Error> {
        match self {
            Footprint::Width(w) if !(w.is_finite() && w > 0.0) => Err(Error::NotPositive {
                what: "footprint width",
                value: w,
            }),
            _ => Ok(self),
        }
    }

    /// The bins of `binning` a pixel at `v` reaches; `Err` when an end of
    /// the footprint has no bin.
    pub(crate) fn reach(self, binning: Binning, v: f64) -> Result<Reach, Error> {
        let (lo, hi, width) = match self {
            Footprint::Point => (v, v, 0.0),
            Footprint::Width(w) => (v - w / 2.0, v + w / 2.0, w),
        };
        Ok(Reach {
            binning,
            lo,
            hi,
            width,
            k0: binning.bin(lo)?,
            k1: binning.bin(hi)?,
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
