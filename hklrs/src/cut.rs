//! Line cuts through reciprocal space (design §5, §7): per pixel Q, the
//! coordinate along the cut and the distance from it, the bin and selection
//! those give, and the sums of counts and weights per bin
//! ([`Cut::accumulate`]). A bin holds Σcounts and Σweight apart; its value
//! is their ratio ([`Bin::intensity`], [`CutResult::intensity`]).
//!
//! Q is in the cut frame of xrayutilities' `HXRD([1,0,0],[0,0,1])` with
//! `trans=False`: x normal to the scattering plane, y along the in-plane
//! reference (the beam side), z the surface normal. Angles are radians; Q is
//! in Å⁻¹ with the 2π. Every kind follows xrayutilities deff08f
//! `analysis/line_cuts.py` (:117-:663) except for the choices listed on
//! [`Cut`]: one half-width, the bin rule of [`Binning`], and a defined
//! outcome for a pixel without coordinates.

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use nalgebra::Vector3;

use crate::absorption::Masked;
pub use crate::grid::fuzzy::Footprint;
use crate::grid::{Binning, Coord, CoordKind};
use crate::normalise::Normalised;
use crate::{Error, Wavelength};

/// What the cut runs along.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CutKind {
    Qz,
    Qy,
    Qx,
    /// Sample rocking angle ω.
    Omega,
    /// ω–2θ direction; the coordinate is 2θ.
    Radial,
    TwoTheta,
    /// Arbitrary line through `point` along `dir`; the coordinate is Q·d̂.
    /// A [`Cut`] holds `dir` divided by its largest |component| (the first
    /// on a tie) and `point` moved along the line to where that component
    /// is zero, with no -0.0 in either.
    Line {
        point: [f64; 3],
        dir: [f64; 3],
    },
}

/// The direction the window is taken in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntDir {
    /// The plane perpendicular to the cut in Q (Å⁻¹).
    Q,
    Omega,
    TwoTheta,
    Radial,
}

impl IntDir {
    /// What the distance from the cut in this direction measures, and so
    /// the half-width: Å⁻¹ in Q, an angle in the others.
    pub fn perp_kind(&self) -> CoordKind {
        match self {
            IntDir::Q => CoordKind::Linear,
            IntDir::Omega | IntDir::TwoTheta | IntDir::Radial => CoordKind::Angle,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CutError {
    #[error("a {kind} cut has no {intdir:?} integration direction")]
    IntDir { kind: &'static str, intdir: IntDir },
    /// Footprints of results of one cut, whose coordinate along is of
    /// `kind`.
    #[error("cut results have different footprints: {left:?} vs {right:?}")]
    Footprint {
        left: Footprint,
        right: Footprint,
        kind: CoordKind,
    },
    #[error("cut results are of different cuts on one axis: {left:?} vs {right:?}")]
    Cut { left: Box<Cut>, right: Box<Cut> },
    /// |at|² of a cut that reads |at| (Omega, Radial, TwoTheta).
    #[error("{kind} cut position has |at|² = {squared:e}: zero, subnormal or infinite")]
    Position { kind: &'static str, squared: f64 },
    #[error(transparent)]
    Value(#[from] Error),
}

/// Where one pixel falls on a cut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cutting {
    /// Coordinate along the cut (Å⁻¹, or radians for an angular cut).
    pub along: f64,
    /// Signed distance from the cut in the integration direction (Å⁻¹ or
    /// radians). NaN when the pixel has no such coordinate (|Q| > 2k, or
    /// the cut position is out of reach).
    pub perp: f64,
}

/// A pixel inside the window and its bin on the cut axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Located {
    pub cutting: Cutting,
    pub bin: i64,
}

/// A line cut: kind, integration direction, position, window, axis.
///
/// Differences from xrayutilities, each with its defect id in
/// `upstream-defects.md`:
/// - `half_width` is the half-width of the window for every kind; xu's
///   `intrange` is the full width in five docstrings that say half
///   (XU-17).
/// - A window that holds no pixel is an ordinary result; xu raises
///   (XU-18, `line_cuts.py:59`).
/// - The window is `|perp| < half_width`, strict, as xu; NaN is outside.
#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    kind: CutKind,
    intdir: IntDir,
    at: [f64; 3],
    half_width: f64,
    binning: Binning,
}

impl Cut {
    /// A cut through `at`, a point in Q of which the kind reads the
    /// components off its own axis: Qz (x, y), Qy (x, z), Qx (y, z), and
    /// with an angular integration direction Qz y alone and Qy z alone;
    /// Omega reads |at|, Radial and TwoTheta y and ρ = √(x² + z²).
    /// [`CutKind::Line`] carries its own point and ignores `at`. The cut
    /// stores what it reads, as the point [`Cut::at`] returns, and a line
    /// as described on [`CutKind::Line`], so inputs that make one cut make
    /// equal cuts. Equality is exact: a last-bit difference refuses a merge.
    /// Omega, Radial and TwoTheta refuse an `at` whose |at|² is zero,
    /// subnormal or infinite.
    pub fn new(
        kind: CutKind,
        intdir: IntDir,
        at: [f64; 3],
        half_width: f64,
        binning: Binning,
    ) -> Result<Self, CutError> {
        if !(half_width.is_finite() && half_width > 0.0) {
            return Err(Error::Width {
                what: "cut half-width",
                value: half_width,
                kind: intdir.perp_kind(),
            }
            .into());
        }
        if at.iter().any(|v| !v.is_finite()) {
            return Err(Error::NotFinite {
                what: "cut position",
                value: at.into_iter().find(|v| !v.is_finite()).unwrap(),
            }
            .into());
        }
        if let CutKind::Line { point, dir } = kind {
            if point.iter().any(|v| !v.is_finite()) {
                return Err(Error::NotFinite {
                    what: "line point",
                    value: point.into_iter().find(|v| !v.is_finite()).unwrap(),
                }
                .into());
            }
            let d = Vector3::from(dir);
            if !(d.iter().all(|v| v.is_finite()) && d.amax() > 0.0) {
                return Err(Error::ZeroVector {
                    what: "line direction".into(),
                }
                .into());
            }
        }
        if !kind.allows(intdir) {
            return Err(CutError::IntDir {
                kind: kind.name(),
                intdir,
            });
        }
        // √ of a normal fl(r²) is r again, so the stored |at| reads back
        let reach = |squared: f64| {
            if squared.is_normal() {
                Ok(squared)
            } else {
                Err(CutError::Position {
                    kind: kind.name(),
                    squared,
                })
            }
        };
        let [x, y, z] = at;
        let (kind, at) = match (kind, intdir) {
            (CutKind::Qz, IntDir::Q) => (kind, [x, y, 0.0]),
            (CutKind::Qz, _) => (kind, [0.0, y, 0.0]),
            (CutKind::Qy, IntDir::Q) => (kind, [x, 0.0, z]),
            (CutKind::Qy, _) => (kind, [0.0, 0.0, z]),
            (CutKind::Qx, _) => (kind, [0.0, y, z]),
            (CutKind::Line { point, dir }, _) => {
                let (p, d) = (Vector3::from(point), Vector3::from(dir));
                let m = (1..3).fold(0, |m, i| if d[i].abs() > d[m].abs() { i } else { m });
                // d[m] becomes ±1 and p[m] 0 exactly, and no smaller |d[i]|
                // rounds up to 1, so the form is its own canonical form;
                // + 0.0 drops a -0.0, which remaking could turn into 0.0
                let d = d / d[m].abs() + Vector3::zeros();
                let point: [f64; 3] = (p - d * (p[m] / d[m]) + Vector3::zeros()).into();
                if let Some(&value) = point.iter().find(|v| !v.is_finite()) {
                    return Err(Error::NotFinite {
                        what: "line point where the direction's largest component is zero",
                        value,
                    }
                    .into());
                }
                (
                    CutKind::Line {
                        point,
                        dir: d.into(),
                    },
                    [0.0; 3],
                )
            }
            (CutKind::Omega, _) => {
                let r = reach(Vector3::from(at).norm_squared())?.sqrt();
                (kind, [0.0, 0.0, r])
            }
            (CutKind::Radial | CutKind::TwoTheta, _) => {
                let rho = z.hypot(x);
                reach(y * y + rho * rho)?;
                (kind, [0.0, y, rho])
            }
        };
        Ok(Self {
            kind,
            intdir,
            at,
            half_width,
            binning,
        })
    }

    pub fn kind(&self) -> CutKind {
        self.kind
    }

    pub fn intdir(&self) -> IntDir {
        self.intdir
    }

    /// What the cut reads of its position, as a point: zero where it reads
    /// nothing, Omega's |at| as z, Radial's and TwoTheta's ρ as z.
    pub fn at(&self) -> [f64; 3] {
        self.at
    }

    pub fn half_width(&self) -> f64 {
        self.half_width
    }

    pub fn binning(&self) -> Binning {
        self.binning
    }

    /// The coordinate along the cut and the distance from it for one Q.
    pub fn cutting(&self, q: &Vector3<f64>, wavelength: Wavelength) -> Cutting {
        let lam = wavelength.angstrom();
        let [ax, ay, az] = self.at;
        let at = Vector3::new(ax, ay, az);
        let (along, perp) = match (self.kind, self.intdir) {
            (CutKind::Qz, IntDir::Q) => (q.z, (q.x - ax).hypot(q.y - ay)),
            (CutKind::Qz, IntDir::Omega) => {
                let (om, tt) = coplanar_om_tt(q, wavelength);
                let qq = 4.0 * PI / lam * (tt / 2.0).sin();
                let ocut = (ay / qq).asin() + tt / 2.0;
                (qq * (ocut - tt / 2.0).cos(), om - ocut)
            }
            (CutKind::Qz, _) => {
                let (om, tt) = coplanar_om_tt(q, wavelength);
                let qq = 4.0 * PI / lam * (tt / 2.0).sin();
                let ttcut = 2.0 * (om - (ay / qq).asin());
                let along = 4.0 * PI / lam * (ttcut / 2.0).sin() * (om - ttcut / 2.0).cos();
                (along, tt - ttcut)
            }
            (CutKind::Qy, IntDir::Q) => (q.y, (q.x - ax).hypot(q.z - az)),
            (CutKind::Qy, IntDir::Omega) => {
                let (om, tt) = coplanar_om_tt_real(q, wavelength);
                let qq = 4.0 * PI / lam * (tt / 2.0).sin();
                let ocut = tt / 2.0 + np_sign(q.y) * (az / qq).acos();
                (qq * (ocut - tt / 2.0).sin(), om - ocut)
            }
            (CutKind::Qy, _) => {
                let (om, tt) = coplanar_om_tt_real(q, wavelength);
                let s = (om.sin() - az * lam / TAU).asin();
                let mut ttcut = (om - s).rem_euclid(TAU);
                if (tt - om).abs() > PI / 2.0 {
                    ttcut = (om + s + PI).rem_euclid(TAU);
                }
                let qq = 4.0 * PI / lam * (ttcut / 2.0).sin();
                (qq * (om - ttcut / 2.0).sin(), tt - ttcut)
            }
            (CutKind::Qx, IntDir::Q) => (q.x, (q.y - ay).hypot(q.z - az)),
            (CutKind::Qx, dir) => {
                let (om, tt) = coplanar_om_tt(q, wavelength);
                let (ocut, ttcut) = coplanar_om_tt(&Vector3::new(q.x, ay, az), wavelength);
                let perp = if dir == IntDir::Omega {
                    om - ocut
                } else {
                    tt - ttcut
                };
                (q.x, perp)
            }
            (CutKind::Omega, dir) => {
                let (om, tt) = coplanar_om_tt(q, wavelength);
                let d = tt - bragg_two_theta(az, wavelength);
                if dir == IntDir::Radial {
                    (om - d / 2.0, d)
                } else {
                    (om, d)
                }
            }
            (CutKind::Radial, dir) => {
                let (om, tt) = coplanar_om_tt(q, wavelength);
                let (ocut, ttcut) = coplanar_om_tt(&at, wavelength);
                if dir == IntDir::Omega {
                    (tt, om - ((tt - ttcut) / 2.0 + ocut))
                } else {
                    let d = 2.0 * (tt / 2.0 - om - (ttcut / 2.0 - ocut));
                    (tt - d, d)
                }
            }
            (CutKind::TwoTheta, dir) => {
                let (om, tt) = coplanar_om_tt(q, wavelength);
                let (ocut, _) = coplanar_om_tt(&at, wavelength);
                if dir == IntDir::Omega {
                    (tt, om - ocut)
                } else {
                    let d = 2.0 * (om - ocut);
                    (tt - d, d)
                }
            }
            (CutKind::Line { point, dir }, _) => {
                let d = Vector3::from(dir).normalize();
                (q.dot(&d), (q - Vector3::from(point)).cross(&d).norm())
            }
        };
        Cutting { along, perp }
    }

    /// Whether a distance from the cut is inside the window: `|perp| <
    /// half_width`, so a pixel exactly at the bound is out and NaN is out.
    pub fn selects(&self, perp: f64) -> bool {
        perp.abs() < self.half_width
    }

    /// The pixel's place on the cut: `None` when outside the window, else
    /// its coordinate and bin on the cut axis, `Err` when the coordinate
    /// has no bin (not finite).
    pub fn locate(
        &self,
        q: &Vector3<f64>,
        wavelength: Wavelength,
    ) -> Result<Option<Located>, Error> {
        let cutting = self.cutting(q, wavelength);
        if !self.selects(cutting.perp) {
            return Ok(None);
        }
        let bin = self.binning.bin(cutting.along).ok_or(Error::NoBin {
            value: cutting.along,
            step: self.binning.step(),
            kind: self.kind.coord().kind(),
        })?;
        Ok(Some(Located { cutting, bin }))
    }
}

/// What one bin of a cut holds. The two sums stay apart.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bin {
    /// Σ counts (× overlap fraction under [`Footprint::Width`]).
    pub counts: f64,
    /// Σ weight (× overlap fraction).
    pub weight: f64,
    /// Pixels that put anything in the bin.
    pub contributions: u64,
}

impl Bin {
    /// Σcounts / Σweight; `None` where Σweight is 0.
    pub fn intensity(&self) -> Option<f64> {
        (self.weight > 0.0).then(|| self.counts / self.weight)
    }
}

/// Pixels that were not accumulated, by cause. Each pixel is counted once,
/// in the first cause that applies, in the order of the fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Skipped {
    /// By reason; counts that are NaN or infinite are `BadCount`, as
    /// `Normalised::new` refuses them. xu reports such a pixel as used
    /// while its gridder drops it (XU-19).
    masked: [u64; MASKED.len()],
    /// No usable coordinate: the Q is beyond 2k or out of reach of the cut
    /// so the distance is NaN, or the coordinate has no bin.
    pub no_coordinate: u64,
    /// A coordinate, but the distance is not `< half_width`.
    pub outside: u64,
}

const MASKED: [Masked; 9] = [
    Masked::Unlit,
    Masked::GrazingExit,
    Masked::ThroughBulk,
    Masked::Opaque,
    Masked::EdgeOn,
    Masked::Extinct,
    Masked::Detector,
    Masked::BadFlat,
    Masked::BadCount,
];

/// Position in [`MASKED`]; exhaustive, so a new reason does not compile
/// until it has a counter.
fn masked_index(m: Masked) -> usize {
    match m {
        Masked::Unlit => 0,
        Masked::GrazingExit => 1,
        Masked::ThroughBulk => 2,
        Masked::Opaque => 3,
        Masked::EdgeOn => 4,
        Masked::Extinct => 5,
        Masked::Detector => 6,
        Masked::BadFlat => 7,
        Masked::BadCount => 8,
    }
}

impl Skipped {
    /// Pixels skipped because their weight was masked for `reason`.
    pub fn masked(&self, reason: Masked) -> u64 {
        self.masked[masked_index(reason)]
    }

    pub fn total(&self) -> u64 {
        self.masked.iter().sum::<u64>() + self.no_coordinate + self.outside
    }

    fn add(&mut self, other: &Self) {
        for (a, b) in self.masked.iter_mut().zip(other.masked) {
            *a += b;
        }
        self.no_coordinate += other.no_coordinate;
        self.outside += other.outside;
    }
}

/// The sums of one cut, per global bin of its [`Binning`], sparse and
/// ascending, with the cut and footprint that made them.
#[derive(Clone, Debug, PartialEq)]
pub struct CutResult {
    cut: Cut,
    footprint: Footprint,
    bins: BTreeMap<i64, Bin>,
    skipped: Skipped,
}

impl CutResult {
    pub fn cut(&self) -> &Cut {
        &self.cut
    }

    pub fn footprint(&self) -> Footprint {
        self.footprint
    }

    /// The bins that received anything, as (global bin, sums), ascending.
    pub fn bins(&self) -> impl Iterator<Item = (i64, &Bin)> {
        self.bins.iter().map(|(k, b)| (*k, b))
    }

    pub fn get(&self, bin: i64) -> Option<&Bin> {
        self.bins.get(&bin)
    }

    pub fn skipped(&self) -> &Skipped {
        &self.skipped
    }

    /// The value of each bin that received anything, as (global bin,
    /// Σcounts / Σweight), ascending; `None` where Σweight is 0.
    pub fn intensity(&self) -> impl Iterator<Item = (i64, Option<f64>)> {
        self.bins.iter().map(|(k, b)| (*k, b.intensity()))
    }

    /// The sum of two results of the same cut and footprint, for frames
    /// accumulated apart. A result with no pixel at all is the identity.
    /// Cuts on different axes (coordinate or binning) are a
    /// [`Error::GridMismatch`]; on one axis but another position, window
    /// or integration direction, a [`CutError::Cut`].
    pub fn merge(mut self, other: Self) -> Result<Self, CutError> {
        if self.cut != other.cut {
            let axis = |c: &Cut| (c.kind.coord(), c.binning);
            let (left, right) = (axis(&self.cut), axis(&other.cut));
            if left != right {
                return Err(Error::GridMismatch {
                    left: vec![left],
                    right: vec![right],
                }
                .into());
            }
            return Err(CutError::Cut {
                left: Box::new(self.cut),
                right: Box::new(other.cut),
            });
        }
        if self.footprint != other.footprint {
            return Err(CutError::Footprint {
                left: self.footprint,
                right: other.footprint,
                kind: self.cut.kind.coord().kind(),
            });
        }
        for (k, b) in other.bins {
            let into = self.bins.entry(k).or_default();
            into.counts += b.counts;
            into.weight += b.weight;
            into.contributions += b.contributions;
        }
        self.skipped.add(&other.skipped);
        Ok(self)
    }

    fn put(&mut self, bin: i64, fraction: f64, n: &Normalised) {
        let b = self.bins.entry(bin).or_default();
        b.counts += fraction * n.counts();
        b.weight += fraction * n.weight().get();
        b.contributions += 1;
    }
}

impl Cut {
    /// Sums `pixels` (Q and normalised value, or why there is none) over
    /// the cut. A pixel is skipped, and counted once under its cause in
    /// [`Skipped`], when it is masked, when its counts are not finite, when
    /// it has no coordinate, or when it is outside the window; all else is
    /// laid on the axis by `footprint`. The one error is a footprint width
    /// that is not positive and finite.
    pub fn accumulate(
        &self,
        wavelength: Wavelength,
        footprint: Footprint,
        pixels: impl IntoIterator<Item = (Vector3<f64>, Result<Normalised, Masked>)>,
    ) -> Result<CutResult, Error> {
        let along = self.kind.coord().kind();
        footprint.check(along)?;
        let mut out = CutResult {
            cut: self.clone(),
            footprint,
            bins: BTreeMap::new(),
            skipped: Skipped::default(),
        };
        for (q, value) in pixels {
            let n = match value {
                Ok(n) => n,
                Err(m) => {
                    out.skipped.masked[masked_index(m)] += 1;
                    continue;
                }
            };
            let c = self.cutting(&q, wavelength);
            if c.perp.is_nan() || !c.along.is_finite() {
                out.skipped.no_coordinate += 1;
                continue;
            }
            if !self.selects(c.perp) {
                out.skipped.outside += 1;
                continue;
            }
            match footprint.reach(self.binning, along, c.along) {
                Ok(reach) => {
                    for (k, f) in reach.shares() {
                        out.put(k, f, &n);
                    }
                }
                Err(_) => out.skipped.no_coordinate += 1,
            }
        }
        Ok(out)
    }
}

impl CutKind {
    /// The coordinate along a cut of this kind: an angle for Omega, Radial
    /// and TwoTheta, a length in Q for the others.
    pub fn coord(&self) -> Coord {
        match self {
            CutKind::Omega | CutKind::Radial | CutKind::TwoTheta => Coord::angle(self.name()),
            _ => Coord::linear(self.name()),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CutKind::Qz => "Qz",
            CutKind::Qy => "Qy",
            CutKind::Qx => "Qx",
            CutKind::Omega => "Omega",
            CutKind::Radial => "Radial",
            CutKind::TwoTheta => "TwoTheta",
            CutKind::Line { .. } => "Line",
        }
    }

    /// The integration directions xu implements for the kind.
    fn allows(&self, d: IntDir) -> bool {
        use IntDir::*;
        match self {
            CutKind::Qz | CutKind::Qy | CutKind::Qx => matches!(d, Q | Omega | TwoTheta),
            CutKind::Omega => matches!(d, TwoTheta | Radial),
            CutKind::Radial => matches!(d, Omega | TwoTheta),
            CutKind::TwoTheta => matches!(d, Omega | Radial),
            CutKind::Line { .. } => d == Q,
        }
    }
}

/// numpy `sign`: 0 at 0 and NaN at NaN.
fn np_sign(v: f64) -> f64 {
    if v.is_nan() || v == 0.0 {
        v * 0.0
    } else {
        v.signum()
    }
}

/// Angle between q and z with the cosine clamped to [-1, 1], as xu's
/// `VecAngle` (`math/transforms.py:170`); NaN for q = 0.
fn angle_to_z(q: &Vector3<f64>) -> f64 {
    let s = q.z / q.norm();
    (if s.abs() > 1.0 { s.signum() } else { s }).acos()
}

fn two_theta(q: &Vector3<f64>, wavelength: Wavelength) -> f64 {
    bragg_two_theta(q.norm(), wavelength)
}

fn bragg_two_theta(q: f64, wavelength: Wavelength) -> f64 {
    2.0 * (q / 2.0 / wavelength.wavenumber()).asin()
}

/// ω and 2θ of xu's coplanar `Q2Ang(geometry="realTilt")`:
/// `2θ = 2 asin(|Q|/2k)`, `ω = θ + atan2(qy, √(qz² + qx²))`. NaN where
/// |Q| > 2k.
pub fn coplanar_om_tt(q: &Vector3<f64>, wavelength: Wavelength) -> (f64, f64) {
    let tt = two_theta(q, wavelength);
    (tt / 2.0 + q.y.atan2(q.z.hypot(q.x)), tt)
}

/// ω and 2θ of `Q2Ang(geometry="real")`: ω = θ ± ∠(Q, z) with the minus sign
/// where the azimuth `−atan2(qx, qy)` is beyond ±π/2.
pub fn coplanar_om_tt_real(q: &Vector3<f64>, wavelength: Wavelength) -> (f64, f64) {
    let tt = two_theta(q, wavelength);
    let phi = -q.x.atan2(q.y);
    let sign = if phi.abs() > PI / 2.0 { -1.0 } else { 1.0 };
    (tt / 2.0 + sign * angle_to_z(q), tt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absorption::Weight;

    fn lam() -> Wavelength {
        Wavelength::new(1.5406).unwrap()
    }

    fn binning() -> Binning {
        Binning::new(CoordKind::Linear, 0.5, 0.0).unwrap()
    }

    #[test]
    fn intensity_is_the_ratio_of_the_sums() {
        let b = |counts, weight| Bin {
            counts,
            weight,
            contributions: 1,
        };
        assert_eq!(b(6.0, 4.0).intensity(), Some(1.5));
        assert_eq!(b(6.0, 0.0).intensity(), None);
        assert_eq!(b(0.0, 2.0).intensity(), Some(0.0));
    }

    /// Every `Weight` is positive, so a bin's Σweight is 0 only when
    /// fraction · weight underflows: a quarter of the smallest subnormal
    /// rounds to 0 while its counts do not.
    #[test]
    fn a_bin_whose_weight_underflows_has_no_intensity() {
        let c = Cut::new(
            CutKind::Qz,
            IntDir::Q,
            [0.0; 3],
            1.0,
            Binning::new(CoordKind::Linear, 1.0, 0.0).unwrap(),
        )
        .unwrap();
        let weight = Weight::new(5e-324, Masked::Opaque).unwrap();
        let px = [(
            Vector3::new(0.0, 0.0, 1.25),
            Normalised::new(8.0, Ok(weight)),
        )];
        // [0.75, 1.75]: a quarter in bin 0, three quarters in bin 1
        let r = c.accumulate(lam(), Footprint::Width(1.0), px).unwrap();
        let b0 = r.get(0).unwrap();
        assert_eq!((b0.counts, b0.weight, b0.contributions), (2.0, 0.0, 1));
        assert_eq!(r.get(1).unwrap().weight, 5e-324);
        let v: Vec<_> = r.intensity().collect();
        assert_eq!(v[0], (0, None));
        assert!(matches!(v[1], (1, Some(_))));
    }

    #[test]
    fn a_nan_count_is_masked_as_bad_count_and_adds_nothing() {
        let c = Cut::new(
            CutKind::Qz,
            IntDir::Q,
            [0.0; 3],
            1.0,
            Binning::new(CoordKind::Linear, 1.0, 0.0).unwrap(),
        )
        .unwrap();
        let one = Weight::new(1.0, Masked::Opaque);
        let q = Vector3::new(0.0, 0.0, 0.5);
        let px = [
            (q, Normalised::new(2.0, one)),
            (q, Normalised::new(f64::NAN, one)),
        ];
        let r = c.accumulate(lam(), Footprint::Point, px).unwrap();
        assert_eq!(r.skipped().masked(Masked::BadCount), 1);
        assert_eq!(r.skipped().total(), 1);
        let b = r.get(0).unwrap();
        assert_eq!((b.counts, b.weight, b.contributions), (2.0, 1.0, 1));
    }

    #[test]
    fn symmetric_reflection_has_omega_half_two_theta() {
        let k = lam().wavenumber();
        let theta = 0.3_f64;
        let q = Vector3::new(0.0, 0.0, 2.0 * k * theta.sin());
        let (om, tt) = coplanar_om_tt(&q, lam());
        assert!((om - theta).abs() < 1e-15 && (tt - 2.0 * theta).abs() < 1e-15);
        let (om, tt) = coplanar_om_tt_real(&q, lam());
        assert!((om - theta).abs() < 1e-15 && (tt - 2.0 * theta).abs() < 1e-15);
    }

    #[test]
    fn beyond_the_ewald_sphere_has_no_angles() {
        let q = Vector3::new(0.0, 0.0, 2.0 * lam().wavenumber() * 1.01);
        let (om, tt) = coplanar_om_tt(&q, lam());
        assert!(om.is_nan() && tt.is_nan());
    }

    #[test]
    fn window_is_strict_and_nan_is_outside() {
        let c = Cut::new(CutKind::Qz, IntDir::Q, [0.0; 3], 0.1, binning()).unwrap();
        assert!(c.selects(0.0999999) && c.selects(-0.0999999));
        assert!(!c.selects(0.1) && !c.selects(-0.1) && !c.selects(f64::NAN));
    }

    #[test]
    fn qz_q_cut_coordinates() {
        let c = Cut::new(CutKind::Qz, IntDir::Q, [1.0, 2.0, 0.0], 0.1, binning()).unwrap();
        let got = c.cutting(&Vector3::new(1.03, 2.04, 3.0), lam());
        assert_eq!(got.along, 3.0);
        assert!((got.perp - 0.05).abs() < 1e-15);
    }

    #[test]
    fn bin_edge_goes_to_the_upper_bin() {
        let c = Cut::new(CutKind::Qz, IntDir::Q, [0.0; 3], 0.1, binning()).unwrap();
        let at = |z| {
            c.locate(&Vector3::new(0.0, 0.0, z), lam())
                .unwrap()
                .unwrap()
                .bin
        };
        assert_eq!((at(0.49999), at(0.5), at(-0.0), at(-0.01)), (0, 1, 0, -1));
    }

    #[test]
    fn empty_window_is_none_not_an_error() {
        let c = Cut::new(CutKind::Qz, IntDir::Q, [5.0, 0.0, 0.0], 0.1, binning()).unwrap();
        assert_eq!(c.locate(&Vector3::new(0.0, 0.0, 1.0), lam()), Ok(None));
    }

    #[test]
    fn non_finite_coordinate_in_the_window_is_an_error() {
        let c = Cut::new(CutKind::Qz, IntDir::Q, [0.0; 3], 0.1, binning()).unwrap();
        assert!(matches!(
            c.locate(&Vector3::new(0.0, 0.0, f64::NAN), lam()),
            Err(Error::NoBin {
                kind: CoordKind::Linear,
                ..
            })
        ));
    }

    #[test]
    fn a_refused_width_carries_the_kind_of_its_coordinate() {
        for (intdir, perp) in [
            (IntDir::Q, CoordKind::Linear),
            (IntDir::Omega, CoordKind::Angle),
        ] {
            assert!(matches!(
                Cut::new(CutKind::Qz, intdir, [0.0; 3], 0.0, binning()),
                Err(CutError::Value(Error::Width { what: "cut half-width", kind, .. })) if kind == perp
            ));
        }
        let c = Cut::new(
            CutKind::Omega,
            IntDir::TwoTheta,
            [0.0, 0.0, 2.0],
            0.1,
            binning(),
        )
        .unwrap();
        let sum = |f| c.accumulate(lam(), f, std::iter::empty());
        assert!(matches!(
            sum(Footprint::Width(0.0)),
            Err(Error::Width {
                what: "footprint width",
                kind: CoordKind::Angle,
                ..
            })
        ));
        let (point, wide) = (Footprint::Point, Footprint::Width(0.1));
        assert_eq!(
            sum(point).unwrap().merge(sum(wide).unwrap()).unwrap_err(),
            CutError::Footprint {
                left: point,
                right: wide,
                kind: CoordKind::Angle
            }
        );
    }

    #[test]
    fn line_distance_and_coordinate() {
        let kind = CutKind::Line {
            point: [0.0, 1.0, 0.0],
            dir: [0.0, 0.0, 2.0],
        };
        let c = Cut::new(kind, IntDir::Q, [0.0; 3], 1.0, binning()).unwrap();
        let got = c.cutting(&Vector3::new(0.3, 1.4, 5.0), lam());
        assert!((got.along - 5.0).abs() < 1e-15 && (got.perp - 0.5).abs() < 1e-15);
    }

    /// A result records its cut, and merges only with a result of the
    /// same cut: another kind is another axis, another position or window
    /// alone is another cut on that axis.
    #[test]
    fn a_merge_refuses_another_kind_position_or_window_alone() {
        let cut = |kind, at, half_width| {
            Cut::new(kind, IntDir::TwoTheta, at, half_width, binning()).unwrap()
        };
        let sums = |c: &Cut| {
            c.accumulate(lam(), Footprint::Point, std::iter::empty())
                .unwrap()
        };
        let base = cut(CutKind::Qz, [0.0, 2.0, 0.0], 0.1);
        let a = sums(&base);
        assert_eq!(a.cut(), &base);
        assert_eq!(a.clone().merge(sums(&base)), Ok(a.clone()));
        for kind in [CutKind::Qx, CutKind::Omega] {
            let other = cut(kind, [0.0, 2.0, 0.0], 0.1);
            assert_eq!(
                a.clone().merge(sums(&other)),
                Err(Error::GridMismatch {
                    left: vec![(Coord::linear("Qz"), binning())],
                    right: vec![(kind.coord(), binning())],
                }
                .into())
            );
        }
        for other in [
            cut(CutKind::Qz, [0.0, 2.5, 0.0], 0.1),
            cut(CutKind::Qz, [0.0, 2.0, 0.0], 0.2),
        ] {
            assert_eq!(
                a.clone().merge(sums(&other)),
                Err(CutError::Cut {
                    left: Box::new(base.clone()),
                    right: Box::new(other),
                })
            );
        }
    }

    /// Inputs that differ only in what the cut does not read make one cut,
    /// whose results merge; what it reads still tells cuts apart.
    #[test]
    fn differently_written_inputs_of_one_cut_merge() {
        let cut = |kind, intdir, at| Cut::new(kind, intdir, at, 0.1, binning()).unwrap();
        let line = |point, dir| cut(CutKind::Line { point, dir }, IntDir::Q, [0.0; 3]);
        let sums = |c: &Cut| {
            let weight = Weight::new(1.0, Masked::Opaque).unwrap();
            let px = [(
                Vector3::new(1.02, 2.0, 0.6),
                Normalised::new(3.0, Ok(weight)),
            )];
            let r = c.accumulate(lam(), Footprint::Point, px).unwrap();
            assert_eq!((r.bins().count(), r.skipped().total()), (1, 0));
            r
        };
        let one = |a: Cut, b: Cut| {
            assert_eq!(a, b);
            let (ra, rb) = (sums(&a), sums(&b));
            let doubled: Vec<_> = ra.bins().map(|(k, b)| (k, 2.0 * b.counts)).collect();
            let merged = ra.merge(rb).unwrap();
            assert_eq!(
                merged
                    .bins()
                    .map(|(k, b)| (k, b.counts))
                    .collect::<Vec<_>>(),
                doubled
            );
        };
        // the line along z through (1, 2, ·), from any of its points and any
        // positive length of its direction
        let z = line([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]);
        assert_eq!(
            z.kind(),
            CutKind::Line {
                point: [1.0, 2.0, 0.0],
                dir: [0.0, 0.0, 1.0]
            }
        );
        one(z, line([1.0, 2.0, -7.0], [0.0, 0.0, 2.5]));
        // as d̂, 5·[1, 2, 3] was another direction in the last bit
        for k in [3.0, 5.0] {
            one(
                line([1.0, 2.0, 0.6], [1.0, 2.0, 3.0]),
                line([1.0, 2.0, 0.6], [k, 2.0 * k, 3.0 * k]),
            );
        }
        // a Qz cut reads x and y of `at` with its window in Q, y alone with
        // it in an angle
        one(
            cut(CutKind::Qz, IntDir::Q, [1.0, 2.0, 0.0]),
            cut(CutKind::Qz, IntDir::Q, [1.0, 2.0, 5.0]),
        );
        one(
            cut(CutKind::Qz, IntDir::Omega, [1.0, 2.0, 0.0]),
            cut(CutKind::Qz, IntDir::Omega, [0.3, 2.0, 5.0]),
        );
        // Omega reads |at|, Radial and TwoTheta y and √(x² + z²)
        for d in [IntDir::TwoTheta, IntDir::Radial] {
            one(
                cut(CutKind::Omega, d, [0.0, 0.0, 2.5]),
                cut(CutKind::Omega, d, [1.5, 2.0, 0.0]),
            );
        }
        for (kind, d) in [
            (CutKind::Radial, IntDir::Omega),
            (CutKind::Radial, IntDir::TwoTheta),
            (CutKind::TwoTheta, IntDir::Omega),
            (CutKind::TwoTheta, IntDir::Radial),
        ] {
            one(
                cut(kind, d, [0.0, 2.0, 1.25]),
                cut(kind, d, [0.75, 2.0, 1.0]),
            );
        }
        let z = [0.0, 0.0, 1.0];
        assert_ne!(line([1.0, 2.0, 3.0], z), line([1.0, 2.5, 3.0], z));
        // the sign of the direction is the sign of the coordinate
        assert_ne!(
            line([1.0, 2.0, 3.0], z),
            line([1.0, 2.0, 3.0], [0.0, 0.0, -1.0])
        );
        assert_ne!(
            cut(CutKind::Qz, IntDir::Q, [1.0, 2.0, 0.0]),
            cut(CutKind::Qz, IntDir::Q, [1.1, 2.0, 0.0])
        );
    }

    /// xorshift64.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// ±0, any finite f64 by its bits, or uniform in [-3, 3).
    fn value(s: &mut u64) -> f64 {
        match next(s) % 8 {
            0 => [0.0, -0.0][(next(s) % 2) as usize],
            1..=3 => Some(f64::from_bits(next(s)))
                .filter(|v| v.is_finite())
                .unwrap_or(0.0),
            _ => (next(s) >> 11) as f64 / (1_u64 << 53) as f64 * 6.0 - 3.0,
        }
    }

    /// Every f64 a cut stores, by its bits, so -0.0 and 0.0 differ.
    fn bits(c: &Cut) -> Vec<u64> {
        let line = match c.kind() {
            CutKind::Line { point, dir } => [point, dir].concat(),
            _ => vec![],
        };
        line.into_iter()
            .chain(c.at())
            .chain([c.half_width()])
            .map(f64::to_bits)
            .collect()
    }

    /// `c` made again from what it stores, asserted the same to the last bit.
    fn remade(c: &Cut) -> Cut {
        let again = Cut::new(c.kind(), c.intdir(), c.at(), c.half_width(), c.binning()).unwrap();
        assert_eq!(bits(&again), bits(c), "{c:?}");
        assert_eq!(&again, c);
        again
    }

    /// The stored position is canonical: a cut made again from it is the
    /// same cut, to the last bit, for positions from ±0 to near overflow.
    /// The kinds that read |at| refuse one whose |at|² is not normal, and
    /// every kind keeps cuts near both ends of the normal range.
    #[test]
    fn a_cut_made_from_its_own_position_is_the_same_cut() {
        let combos = [
            (CutKind::Qz, IntDir::Q),
            (CutKind::Qz, IntDir::Omega),
            (CutKind::Qy, IntDir::Q),
            (CutKind::Qy, IntDir::TwoTheta),
            (CutKind::Qx, IntDir::Q),
            (CutKind::Omega, IntDir::TwoTheta),
            (CutKind::Radial, IntDir::Omega),
            (CutKind::TwoTheta, IntDir::Radial),
        ];
        let mut s = 0x9e37_79b9_7f4a_7c15_u64;
        // per combo: refused, kept with |at|² below 2^-900, above 2^900,
        // kept from an `at` with a -0.0
        let mut seen = [[0; 4]; 8];
        for _ in 0..10000 {
            let at = [value(&mut s), value(&mut s), value(&mut s)];
            for (k, &(kind, d)) in combos.iter().enumerate() {
                match Cut::new(kind, d, at, 0.1, binning()) {
                    Ok(c) => {
                        remade(&c);
                        let squared = Vector3::from(c.at()).norm_squared();
                        seen[k][1] += usize::from(squared < 2f64.powi(-900));
                        seen[k][2] += usize::from(squared > 2f64.powi(900));
                        seen[k][3] +=
                            usize::from(at.iter().any(|v| *v == 0.0 && v.is_sign_negative()));
                    }
                    Err(CutError::Position { squared, .. }) => {
                        assert!(!squared.is_normal(), "{at:?}");
                        seen[k][0] += 1;
                    }
                    Err(e) => panic!("{at:?}: {e}"),
                }
            }
        }
        for (&(kind, _), [refused, low, high, negative_zero]) in combos.iter().zip(seen) {
            let reads_length = matches!(kind, CutKind::Omega | CutKind::Radial | CutKind::TwoTheta);
            assert_eq!(refused > 0, reads_length, "{kind:?}");
            assert!(
                low > 0 && high > 0 && negative_zero > 0,
                "{kind:?} {low} {high} {negative_zero}"
            );
        }
    }

    #[test]
    fn a_cut_that_reads_its_length_refuses_one_whose_square_is_not_normal() {
        // low² is f64::MIN_POSITIVE; high² rounds below f64::MAX
        let (low, high) = (2f64.powi(-511), 2f64.powi(512).next_down());
        for (kind, d) in [
            (CutKind::Omega, IntDir::TwoTheta),
            (CutKind::Omega, IntDir::Radial),
            (CutKind::Radial, IntDir::Omega),
            (CutKind::Radial, IntDir::TwoTheta),
            (CutKind::TwoTheta, IntDir::Omega),
            (CutKind::TwoTheta, IntDir::Radial),
        ] {
            for axis in 0..3 {
                let at = |v| {
                    let mut at = [0.0; 3];
                    at[axis] = v;
                    at
                };
                for v in [low, high, -high] {
                    let c = Cut::new(kind, d, at(v), 0.1, binning()).unwrap();
                    assert_eq!(Vector3::from(c.at()).amax(), v.abs(), "{kind:?} {axis} {v}");
                    remade(&c);
                }
                for (v, squared) in [
                    (0.0, 0.0),
                    (-0.0, 0.0),
                    (low.next_down(), f64::MIN_POSITIVE.next_down()),
                    (2f64.powi(512), f64::INFINITY),
                ] {
                    assert_eq!(
                        Cut::new(kind, d, at(v), 0.1, binning()),
                        Err(CutError::Position {
                            kind: kind.name(),
                            squared
                        }),
                        "{kind:?} {axis} {v}"
                    );
                }
            }
        }
    }

    /// As above for a line, whichever component of the direction is the
    /// largest, of either sign, tied or not, with ±0 components, and with
    /// magnitudes from subnormal to near overflow.
    #[test]
    fn a_line_cut_made_from_its_own_line_is_the_same_cut() {
        let largest =
            |d: [f64; 3]| (1..3).fold(0, |m, i| if d[i].abs() > d[m].abs() { i } else { m });
        let line = |point, dir| {
            Cut::new(
                CutKind::Line { point, dir },
                IntDir::Q,
                [0.0; 3],
                0.1,
                binning(),
            )
        };
        let mut s = 0x2545_f491_4f6c_dd1d_u64;
        // [largest][negative][tied with a later component]
        let mut seen = [[[0; 2]; 2]; 3];
        let (mut zero, mut negative_zero, mut subnormal, mut huge) = (0, 0, 0, 0);
        for _ in 0..20000 {
            let mut d = [value(&mut s), value(&mut s), value(&mut s)];
            if next(&mut s).is_multiple_of(3) {
                let m = largest(d);
                let j = (m + 1 + (next(&mut s) % 2) as usize) % 3;
                d[j] = if next(&mut s).is_multiple_of(2) {
                    d[m]
                } else {
                    -d[m]
                };
            }
            let scale = match next(&mut s) % 8 {
                0 => f64::MIN_POSITIVE * 2f64.powi(-40),
                1 => 2f64.powi(1020),
                _ => 1.0,
            };
            let d = d.map(|v| v * scale);
            let p = [value(&mut s), value(&mut s), value(&mut s)];
            let c = match line(p, d) {
                Ok(c) => c,
                Err(CutError::Value(Error::ZeroVector { .. } | Error::NotFinite { .. })) => {
                    continue;
                }
                Err(e) => panic!("{p:?} {d:?}: {e}"),
            };
            let CutKind::Line { point, dir } = c.kind() else {
                unreachable!()
            };
            let m = largest(d);
            assert_eq!(largest(dir), m, "{d:?}");
            assert_eq!(dir[m], d[m].signum(), "{d:?}");
            assert_eq!(point[m], 0.0, "{p:?} {d:?}");
            assert!(
                point
                    .iter()
                    .chain(&dir)
                    .all(|v| v.to_bits() != (-0.0f64).to_bits()),
                "{p:?} {d:?}"
            );
            remade(&c);
            let tie = (m + 1..3).any(|j| d[j].abs() == d[m].abs());
            seen[m][usize::from(d[m] < 0.0)][usize::from(tie)] += 1;
            zero += usize::from(d.contains(&0.0));
            negative_zero += usize::from(
                p.iter()
                    .chain(&d)
                    .any(|v| v.to_bits() == (-0.0f64).to_bits()),
            );
            subnormal += usize::from(d[m].is_subnormal());
            huge += usize::from(
                d[m].abs() > 2f64.powi(1000) || p.iter().any(|v| v.abs() > 2f64.powi(1000)),
            );
        }
        for (m, by_sign) in seen.iter().enumerate() {
            for (negative, by_tie) in by_sign.iter().enumerate() {
                // the first of tied components is the largest, so the last never ties
                let cases = if m == 2 { &by_tie[..1] } else { &by_tie[..] };
                assert!(cases.iter().all(|&n| n > 0), "{m} {negative} {by_tie:?}");
            }
        }
        assert!(
            zero > 0 && negative_zero > 0 && subnormal > 0 && huge > 0,
            "{zero} {negative_zero} {subnormal} {huge}"
        );
        // a -0.0 the move along the line keeps would come back as 0.0
        let c = line([-0.0, 0.0, -5.0], [-0.0, 0.0, 1.0]).unwrap();
        assert_eq!(
            bits(&remade(&c))[..6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0].map(f64::to_bits)
        );
        // a direction too short or too long for |d| to be finite and positive
        for (d, held) in [
            ([1e-200, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1e308, -1e308, 0.0], [1.0, -1.0, 0.0]),
        ] {
            let c = line([0.0, 2.0, 3.0], d).unwrap();
            assert_eq!(
                c.kind(),
                CutKind::Line {
                    point: [0.0, 2.0, 3.0],
                    dir: held
                }
            );
        }
    }

    #[test]
    fn construction_rejects_what_xu_has_no_branch_for() {
        let at = [0.0; 3];
        let bad = |kind, d| Cut::new(kind, d, at, 0.1, binning()).unwrap_err();
        assert!(matches!(
            bad(CutKind::Omega, IntDir::Q),
            CutError::IntDir { .. }
        ));
        assert!(matches!(
            bad(CutKind::Qz, IntDir::Radial),
            CutError::IntDir { .. }
        ));
        let line = |dir| CutKind::Line {
            point: [0.0; 3],
            dir,
        };
        assert!(matches!(
            bad(line([1.0; 3]), IntDir::Omega),
            CutError::IntDir { .. }
        ));
        assert!(matches!(
            bad(line([0.0; 3]), IntDir::Q),
            CutError::Value(Error::ZeroVector { .. })
        ));
        // a finite point whose move along the line overflows
        let far = CutKind::Line {
            point: [1e308, 0.0, 1e308],
            dir: [-1.0, 0.0, 1.0],
        };
        assert!(matches!(
            Cut::new(far, IntDir::Q, at, 0.1, binning()),
            Err(CutError::Value(Error::NotFinite { value, .. })) if value == f64::INFINITY
        ));
        for w in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Cut::new(CutKind::Qz, IntDir::Q, at, w, binning()).is_err());
        }
    }
}
