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
use crate::grid::Binning;
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

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CutError {
    #[error("a {kind} cut has no {intdir:?} integration direction")]
    IntDir { kind: &'static str, intdir: IntDir },
    #[error("cut results have different footprints: {left:?} vs {right:?}")]
    Footprint { left: Footprint, right: Footprint },
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
    /// A cut through `at`, a point in Q of which the kind uses the
    /// components off its own axis: Qz (x, y), Qy (x, z), Qx (y, z);
    /// Omega, Radial and TwoTheta use all three, through their ω and 2θ.
    /// [`CutKind::Line`] carries its own point and ignores `at`.
    pub fn new(
        kind: CutKind,
        intdir: IntDir,
        at: [f64; 3],
        half_width: f64,
        binning: Binning,
    ) -> Result<Self, CutError> {
        if !(half_width.is_finite() && half_width > 0.0) {
            return Err(Error::NotPositive {
                what: "cut half-width",
                value: half_width,
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
            if !(d.iter().all(|v| v.is_finite()) && d.norm() > 0.0) {
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
                let (_, ttcut) = coplanar_om_tt(&at, wavelength);
                let d = tt - ttcut;
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
                let d = Vector3::from(dir);
                let along = q.dot(&d.normalize());
                (
                    along,
                    (q - Vector3::from(point)).cross(&d).norm() / d.norm(),
                )
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
        let bin = self.binning.bin(cutting.along)?;
        Ok(Some(Located { cutting, bin }))
    }
}

/// How a pixel's weight is laid on the cut axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Footprint {
    /// All of it in the bin of the pixel's coordinate.
    Point,
    /// A pixel spans `width` along the cut, centred on its coordinate (same
    /// units as the coordinate), and its counts and weight are shared out
    /// over the bins it overlaps in proportion to the overlap: xu's
    /// `FuzzyGridder1D`, whose fractions these are. A bin of the global
    /// lattice exists wherever the footprint reaches, so unlike xu nothing
    /// is clamped at a grid end.
    Width(f64),
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
    masked: [u64; MASKED.len()],
    /// An accumulated `Normalised` whose counts are NaN or infinite. xu
    /// reports such a pixel as used while its gridder drops it (XU-19).
    pub non_finite: u64,
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
        self.masked.iter().sum::<u64>() + self.non_finite + self.no_coordinate + self.outside
    }

    fn add(&mut self, other: &Self) {
        for (a, b) in self.masked.iter_mut().zip(other.masked) {
            *a += b;
        }
        self.non_finite += other.non_finite;
        self.no_coordinate += other.no_coordinate;
        self.outside += other.outside;
    }
}

/// The sums of one cut, per global bin of its [`Binning`], sparse and
/// ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct CutResult {
    binning: Binning,
    footprint: Footprint,
    bins: BTreeMap<i64, Bin>,
    skipped: Skipped,
}

impl CutResult {
    pub fn binning(&self) -> Binning {
        self.binning
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

    /// The sum of two results of the same binning and footprint, for
    /// frames accumulated apart. A result with no pixel at all is the
    /// identity.
    pub fn merge(mut self, other: Self) -> Result<Self, CutError> {
        if self.binning != other.binning {
            let name = |b: Binning| vec![("cut".to_owned(), b.step(), b.anchor())];
            return Err(Error::GridMismatch {
                left: name(self.binning),
                right: name(other.binning),
            }
            .into());
        }
        if self.footprint != other.footprint {
            return Err(CutError::Footprint {
                left: self.footprint,
                right: other.footprint,
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
        b.counts += fraction * n.counts;
        b.weight += fraction * n.weight.get();
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
        if let Footprint::Width(w) = footprint
            && !(w.is_finite() && w > 0.0)
        {
            return Err(Error::NotPositive {
                what: "footprint width",
                value: w,
            });
        }
        let mut out = CutResult {
            binning: self.binning,
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
            if !n.counts.is_finite() {
                out.skipped.non_finite += 1;
                continue;
            }
            let c = self.cutting(&q, wavelength);
            if c.perp.is_nan() || !c.along.is_finite() {
                out.skipped.no_coordinate += 1;
                continue;
            }
            if !self.selects(c.perp) {
                out.skipped.outside += 1;
                continue;
            }
            let b = self.binning;
            match footprint {
                Footprint::Point => match b.bin(c.along) {
                    Ok(k) => out.put(k, 1.0, &n),
                    Err(_) => out.skipped.no_coordinate += 1,
                },
                Footprint::Width(w) => {
                    let (lo, hi) = (c.along - w / 2.0, c.along + w / 2.0);
                    let (Ok(k0), Ok(k1)) = (b.bin(lo), b.bin(hi)) else {
                        out.skipped.no_coordinate += 1;
                        continue;
                    };
                    for k in k0..=k1 {
                        let f = if k0 == k1 {
                            1.0
                        } else {
                            (hi.min(b.edge(k + 1)) - lo.max(b.edge(k))) / w
                        };
                        // a footprint ending exactly on an edge has no part in the bin above
                        if f > 0.0 {
                            out.put(k, f, &n);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

impl CutKind {
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
    2.0 * (q.norm() / 2.0 / wavelength.wavenumber()).asin()
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
        Binning::new(0.5, 0.0).unwrap()
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
            Binning::new(1.0, 0.0).unwrap(),
        )
        .unwrap();
        let weight = Weight::new(5e-324, Masked::Opaque).unwrap();
        let px = [(
            Vector3::new(0.0, 0.0, 1.25),
            Ok(Normalised {
                counts: 8.0,
                weight,
            }),
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
        assert!(c.locate(&Vector3::new(0.0, 0.0, f64::NAN), lam()).is_err());
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
        for w in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Cut::new(CutKind::Qz, IntDir::Q, at, w, binning()).is_err());
        }
    }
}
