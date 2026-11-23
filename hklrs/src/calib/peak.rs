//! Peak positions in detector frames: a windowed centre of mass for images,
//! and the same followed by a pseudo-Voigt fit for one-dimensional rows.
//!
//! Positions are fractional pixel indices with pixel centres at integers, as
//! [`crate::PixelDetector::pixel`] counts. Pixel values must be counts: the
//! position error is the Poisson error of the centroid, `√(Σ dᵢ (xᵢ − c)²) / N`
//! over the window, with `dᵢ` the pixel counts (at least 1) and `N` the
//! background-subtracted sum. Background pixels count, so a wide window
//! costs precision; for a pure peak it reduces to second moment over √N.
//!
//! The search starts at the window position of largest background-subtracted
//! sum above the mean of its surrounding ring (a summed-area table), never at
//! the whole-frame centre of mass: a background that is off by δ over N
//! pixels adds `Nδ` to that, which swamps a peak at δ ≈ 0.1 on a detector of
//! 640000 pixels (xrayutilities' `_peak_position`, XU-20). The starting
//! window and the settled one are peaks only if that excess is at least
//! `min_snr` σ of its own Poisson noise, else [`Reject::NoSignal`].

use thiserror::Error;

use super::lm;

/// A located peak: fractional index along (slow, fast) and its standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub pos: [f64; 2],
    pub sigma: [f64; 2],
}

/// A peak along one row of channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak1 {
    pub pos: f64,
    pub sigma: f64,
}

/// Why a frame yields no peak. These are findings about the frame, not
/// failures of the caller's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    /// No window of the frame holds a sum of `min_snr` σ above the border
    /// background.
    NoSignal,
    /// An unmasked pixel in the peak window is at or above the saturation level.
    Saturated { slow: usize, fast: usize },
    /// Signal of at least `edge_fraction` of the window maximum lies within
    /// `border` pixels of the image edge, so the peak is cut off.
    TouchesEdge,
    /// The window did not settle within `max_passes` re-centrings.
    NotConverged { passes: usize },
    /// The pseudo-Voigt fit did not converge, left its window, or has an
    /// unidentifiable position.
    FitFailed,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PeakError {
    #[error("image has {len} values, shape ({slow}, {fast}) needs {}", slow * fast)]
    Shape {
        len: usize,
        slow: usize,
        fast: usize,
    },
    #[error("mask has {mask} values, image has {data}")]
    MaskLength { mask: usize, data: usize },
    #[error("unmasked pixel ({slow}, {fast}) is not finite")]
    NonFinite { slow: usize, fast: usize },
    #[error("image of {slow} x {fast} has no pixels left outside the {border}-pixel border")]
    TooSmall {
        slow: usize,
        fast: usize,
        border: usize,
    },
    #[error("frame rejected: {0:?}")]
    Rejected(Reject),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakOptions {
    /// Side of the centre-of-mass window, pixels (xu `nwindow`).
    pub window: usize,
    /// The window is settled when the centre moves less than this, pixels.
    pub tol: f64,
    pub max_passes: usize,
    /// Width of the image-edge ring that gives the background and the edge test.
    pub border: usize,
    pub edge_fraction: f64,
    /// A window is a peak only if its background-subtracted sum exceeds
    /// this many σ of its own Poisson noise. Default 6: the largest of the
    /// ~10⁶ window sums of a signal-free frame reaches about √(2 ln 10⁶) ≈
    /// 5.3 σ, so 6 leaves the chance of a false peak per frame near 10⁻³
    /// at the largest detectors and far below for the usual ones.
    pub min_snr: f64,
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self {
            window: 50,
            tol: 0.01,
            max_passes: 20,
            border: 3,
            edge_fraction: 0.1,
            min_snr: 6.0,
        }
    }
}

/// A frame: row-major counts, `shape = (slow, fast)`. `mask[i]` true marks a
/// bad pixel, which carries no weight.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub data: &'a [f64],
    pub shape: (usize, usize),
    pub mask: Option<&'a [bool]>,
    pub saturation: Option<f64>,
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

fn validate(img: &Image, ns: usize, nf: usize) -> Result<(), PeakError> {
    if ns == 0 || nf == 0 || img.data.len() != ns * nf {
        return Err(PeakError::Shape {
            len: img.data.len(),
            slow: ns,
            fast: nf,
        });
    }
    if let Some(m) = img.mask
        && m.len() != img.data.len()
    {
        return Err(PeakError::MaskLength {
            mask: m.len(),
            data: img.data.len(),
        });
    }
    for (i, v) in img.data.iter().enumerate() {
        if !v.is_finite() && !img.mask.is_some_and(|m| m[i]) {
            return Err(PeakError::NonFinite {
                slow: i / nf,
                fast: i % nf,
            });
        }
    }
    Ok(())
}

/// Weight of pixel `i` in a window of half-width `half` about `c`: 1 inside,
/// falling linearly to 0 across the edge pixel, so the centre of mass is a
/// continuous function of `c`. A hard window makes it jump as `c` crosses
/// a pixel boundary, and re-centring then cycles between two windows
/// whenever the noise at the window edge exceeds the tolerance.
fn omega(i: usize, c: f64, half: f64) -> f64 {
    (half + 0.5 - (i as f64 - c).abs()).clamp(0.0, 1.0)
}

fn span(c: f64, half: f64, n: usize) -> std::ops::RangeInclusive<usize> {
    let lo = (c - half - 1.0).floor().max(0.0) as usize;
    let hi = ((c + half + 1.0).ceil().min((n - 1) as f64)) as usize;
    lo..=hi
}

/// Pixels of a window with their weight, which is non-zero.
fn pixels(
    c: [f64; 2],
    half: [f64; 2],
    ns: usize,
    nf: usize,
) -> impl Iterator<Item = (usize, usize, f64)> {
    span(c[0], half[0], ns).flat_map(move |i| {
        span(c[1], half[1], nf).filter_map(move |j| {
            let om = omega(i, c[0], half[0]) * omega(j, c[1], half[1]);
            (om > 0.0).then_some((i, j, om))
        })
    })
}

/// Windowed centre of mass of `w` and its weight sum.
fn com(w: &[f64], ns: usize, nf: usize, c: [f64; 2], half: [f64; 2]) -> ([f64; 2], f64) {
    let (mut s, mut a, mut b) = (0.0, 0.0, 0.0);
    for (i, j, om) in pixels(c, half, ns, nf) {
        let v = om * w[i * nf + j];
        s += v;
        a += v * i as f64;
        b += v * j as f64;
    }
    ([a / s, b / s], s)
}

/// Sum over every `bs × bf` box of a frame, by summed-area table, O(N).
struct Boxes {
    nf: usize,
    sat: Vec<f64>,
}

impl Boxes {
    fn new(v: impl Fn(usize) -> f64, ns: usize, nf: usize) -> Self {
        let mut sat = vec![0.0; (ns + 1) * (nf + 1)];
        for i in 0..ns {
            let mut row = 0.0;
            for j in 0..nf {
                row += v(i * nf + j);
                sat[(i + 1) * (nf + 1) + j + 1] = sat[i * (nf + 1) + j + 1] + row;
            }
        }
        Self { nf, sat }
    }

    fn sum(&self, i: usize, j: usize, bs: usize, bf: usize) -> f64 {
        let at = |a: usize, b: usize| self.sat[a * (self.nf + 1) + b];
        at(i + bs, j + bf) - at(i, j + bf) - at(i + bs, j) + at(i, j)
    }
}

/// Window sum, its Poisson variance, and the position error terms
/// `Σ var·(x−c)²` along each axis, over the tapered window about `c`.
/// A pixel's variance is its raw count, at least 1.
fn window_stats(img: &Image, w: &[f64], c: [f64; 2], half: [f64; 2]) -> (f64, f64, [f64; 2]) {
    let (ns, nf) = img.shape;
    let (mut n, mut var, mut e) = (0.0, 0.0, [0.0; 2]);
    for (i, j, om) in pixels(c, half, ns, nf) {
        let k = i * nf + j;
        if img.mask.is_some_and(|m| m[k]) {
            continue;
        }
        let vk = om * om * img.data[k].max(1.0);
        n += om * w[k];
        var += vk;
        e[0] += vk * (i as f64 - c[0]).powi(2);
        e[1] += vk * (j as f64 - c[1]).powi(2);
    }
    (n, var, e)
}

/// The window sum must clear `min_snr` σ, else there is no peak.
fn significant(n: f64, var: f64, opts: &PeakOptions) -> Result<(), Reject> {
    if n.is_nan() || n <= 0.0 || n < opts.min_snr * var.sqrt() {
        Err(Reject::NoSignal)
    } else {
        Ok(())
    }
}

/// Box sums of the background-subtracted counts and of their Poisson
/// variance, each box measured against the ring of pixels around it.
///
/// The border median fixes the background only at the border. Whatever the
/// interior sits above it (a level, a gradient) adds `Nδ` to a sum over the
/// frame, and still `Bδ` to a box of `B` pixels, which would count as signal
/// in a signal-free frame. The ring mean is the local background: a level
/// cancels and so does a gradient that is linear across the ring.
struct Scan {
    sums: Boxes,
    vars: Boxes,
    ns: usize,
    nf: usize,
    bs: usize,
    bf: usize,
    margin: [usize; 2],
}

impl Scan {
    fn new(img: &Image, w: &[f64], window: usize) -> Self {
        let (ns, nf) = img.shape;
        let (bs, bf) = (window.clamp(1, ns), window.clamp(1, nf));
        let var = |k: usize| {
            if img.mask.is_some_and(|m| m[k]) {
                0.0
            } else {
                img.data[k].max(1.0)
            }
        };
        Self {
            sums: Boxes::new(|k| w[k], ns, nf),
            vars: Boxes::new(var, ns, nf),
            ns,
            nf,
            bs,
            bf,
            // A one-row frame has no ring across the row.
            margin: [if ns == 1 { 0 } else { window }, window],
        }
    }

    /// Box with top-left pixel `(i, j)`: its sum above the ring mean, and
    /// the variance of that. With no ring (the box is the frame) the
    /// plain sum.
    fn excess(&self, i: usize, j: usize) -> (f64, f64) {
        let (bs, bf) = (self.bs, self.bf);
        let (i0, j0) = (
            i.saturating_sub(self.margin[0]),
            j.saturating_sub(self.margin[1]),
        );
        let i1 = (i + bs + self.margin[0]).min(self.ns);
        let j1 = (j + bf + self.margin[1]).min(self.nf);
        let (rs, rv) = (
            self.sums.sum(i0, j0, i1 - i0, j1 - j0),
            self.vars.sum(i0, j0, i1 - i0, j1 - j0),
        );
        let (bsum, bvar) = (self.sums.sum(i, j, bs, bf), self.vars.sum(i, j, bs, bf));
        let ring = ((i1 - i0) * (j1 - j0) - bs * bf) as f64;
        if ring <= 0.0 {
            return (bsum, bvar);
        }
        let f = (bs * bf) as f64 / ring;
        (bsum - f * (rs - bsum), bvar + f * f * (rv - bvar))
    }

    /// The box of largest excess: its centre, excess and variance.
    fn best(&self) -> ([f64; 2], f64, f64) {
        let mut best = (f64::NEG_INFINITY, 0, 0, 0.0);
        for i in 0..=self.ns - self.bs {
            for j in 0..=self.nf - self.bf {
                let (e, v) = self.excess(i, j);
                if e > best.0 {
                    best = (e, i, j, v);
                }
            }
        }
        (
            [
                best.1 as f64 + (self.bs - 1) as f64 / 2.0,
                best.2 as f64 + (self.bf - 1) as f64 / 2.0,
            ],
            best.0,
            best.3,
        )
    }

    /// Excess and variance of the box whose centre is nearest `c`.
    fn at(&self, c: [f64; 2]) -> (f64, f64) {
        let origin = |c: f64, b: usize, n: usize| {
            ((c - (b - 1) as f64 / 2.0).round().max(0.0) as usize).min(n - b)
        };
        self.excess(
            origin(c[0], self.bs, self.ns),
            origin(c[1], self.bf, self.nf),
        )
    }
}

/// Re-centre a window of half-width `half` from `c` until the centre moves
/// less than `tol`.
fn settle(
    w: &[f64],
    ns: usize,
    nf: usize,
    mut c: [f64; 2],
    half: [f64; 2],
    opts: &PeakOptions,
) -> Result<[f64; 2], Reject> {
    for _ in 0..opts.max_passes {
        let (next, sum) = com(w, ns, nf, c, half);
        if sum.is_nan() || sum <= 0.0 {
            return Err(Reject::NoSignal);
        }
        let shift = (next[0] - c[0]).hypot(next[1] - c[1]);
        c = next;
        if shift < opts.tol {
            return Ok(c);
        }
    }
    Err(Reject::NotConverged {
        passes: opts.max_passes,
    })
}

/// Saturation and edge tests over the final window.
fn check_window(
    img: &Image,
    w: &[f64],
    c: [f64; 2],
    half: [f64; 2],
    opts: &PeakOptions,
) -> Result<(), Reject> {
    let (ns, nf) = img.shape;
    let b = opts.border;
    let bad = |k: usize| img.mask.is_some_and(|m| m[k]);
    let wmax = pixels(c, half, ns, nf).fold(f64::MIN, |m, (i, j, _)| m.max(w[i * nf + j]));
    for (i, j, _) in pixels(c, half, ns, nf) {
        let k = i * nf + j;
        if let Some(sat) = img.saturation
            && !bad(k)
            && img.data[k] >= sat
        {
            return Err(Reject::Saturated { slow: i, fast: j });
        }
        let edge = (ns > 1 && (i < b || i >= ns - b)) || j < b || j >= nf - b;
        if edge && w[k] >= opts.edge_fraction * wmax {
            return Err(Reject::TouchesEdge);
        }
    }
    Ok(())
}

/// Background: the lowest median among the border strips, so a peak that
/// reaches one side does not raise the estimate.
fn lowest_median<const N: usize>(strips: [Vec<f64>; N]) -> Option<f64> {
    strips.into_iter().filter_map(median).min_by(f64::total_cmp)
}

/// Centre of mass of a frame in a window re-centred until it settles.
pub fn find_peak(img: &Image, opts: &PeakOptions) -> Result<Peak, PeakError> {
    let (ns, nf) = img.shape;
    validate(img, ns, nf)?;
    let bad = |i: usize| img.mask.is_some_and(|m| m[i]);
    let b = opts.border;
    let too_small = || PeakError::TooSmall {
        slow: ns,
        fast: nf,
        border: b,
    };
    if ns < 2 * b + 1 || nf < 2 * b + 1 {
        return Err(too_small());
    }
    let strip = |rows: (usize, usize), cols: (usize, usize)| -> Vec<f64> {
        let mut v = Vec::new();
        for i in rows.0..rows.1 {
            for j in cols.0..cols.1 {
                if !bad(i * nf + j) {
                    v.push(img.data[i * nf + j]);
                }
            }
        }
        v
    };
    let bg = lowest_median([
        strip((0, b), (0, nf)),
        strip((ns - b, ns), (0, nf)),
        strip((0, ns), (0, b)),
        strip((0, ns), (nf - b, nf)),
    ])
    .ok_or_else(too_small)?;
    // Signed weights: clipping noise at zero would pull the centre toward
    // the window centre by roughly (noise sum / peak sum) × offset.
    let w: Vec<f64> = (0..ns * nf)
        .map(|i| if bad(i) { 0.0 } else { img.data[i] - bg })
        .collect();
    let half = [(opts.window / 2) as f64; 2];
    let scan = Scan::new(img, &w, opts.window);
    let (c0, e0, v0) = scan.best();
    significant(e0, v0, opts).map_err(PeakError::Rejected)?;
    let c = settle(&w, ns, nf, c0, half, opts).map_err(PeakError::Rejected)?;
    check_window(img, &w, c, half, opts).map_err(PeakError::Rejected)?;
    let (ex, vx) = scan.at(c);
    significant(ex, vx, opts).map_err(PeakError::Rejected)?;
    let (n, _, e) = window_stats(img, &w, c, half);
    Ok(Peak {
        pos: c,
        sigma: [e[0].sqrt() / n, e[1].sqrt() / n],
    })
}

fn pseudo_voigt(x: f64, p: &[f64]) -> f64 {
    let t = (x - p[0]) / p[1].abs();
    let gauss = (-4.0 * std::f64::consts::LN_2 * t * t).exp();
    let lorentz = 1.0 / (1.0 + 4.0 * t * t);
    let eta = 0.5 * (1.0 + p[4].sin());
    p[3] + p[2] * (eta * lorentz + (1.0 - eta) * gauss)
}

/// A peak along one row of channels: [`find_peak`]'s window on one axis,
/// then a pseudo-Voigt fit (centre, FWHM, height, background, Lorentz share)
/// to counts weighted by their Poisson error, over ±3.5 FWHM of the window
/// centre. `sigma` is the fit's error of the centre with the other
/// parameters held, scaled by `max(1, χ²/dof)^½`.
pub fn find_peak_1d(
    row: &[f64],
    mask: Option<&[bool]>,
    saturation: Option<f64>,
    opts: &PeakOptions,
) -> Result<Peak1, PeakError> {
    let n = row.len();
    let img = Image {
        data: row,
        shape: (1, n),
        mask,
        saturation,
    };
    validate(&img, 1, n)?;
    let bad = |i: usize| mask.is_some_and(|m| m[i]);
    let b = opts.border;
    let too_small = || PeakError::TooSmall {
        slow: 1,
        fast: n,
        border: b,
    };
    if n < 2 * b + 1 {
        return Err(too_small());
    }
    let end = |range: std::ops::Range<usize>| -> Vec<f64> {
        range.filter(|&i| !bad(i)).map(|i| row[i]).collect()
    };
    let bg = lowest_median([end(0..b), end(n - b..n)]).ok_or_else(too_small)?;
    let w: Vec<f64> = (0..n)
        .map(|i| if bad(i) { 0.0 } else { row[i] - bg })
        .collect();
    let half = [f64::INFINITY, (opts.window / 2) as f64];
    let scan = Scan::new(&img, &w, opts.window);
    let (c0, e0, v0) = scan.best();
    significant(e0, v0, opts).map_err(PeakError::Rejected)?;
    let c = settle(&w, 1, n, c0, half, opts).map_err(PeakError::Rejected)?;
    check_window(&img, &w, c, half, opts).map_err(PeakError::Rejected)?;
    let (ex, vx) = scan.at(c);
    significant(ex, vx, opts).map_err(PeakError::Rejected)?;
    let c = c[1];
    let (kmax, wmax) =
        pixels([0.0, c], half, 1, n).fold(
            (0, f64::MIN),
            |m, (_, j, _)| if w[j] > m.1 { (j, w[j]) } else { m },
        );
    if wmax <= 0.0 {
        return Err(PeakError::Rejected(Reject::NoSignal));
    }
    let above = |v: &&f64| **v >= 0.5 * wmax;
    let below = w[..kmax].iter().rev().take_while(above).count();
    let past = w[kmax + 1..].iter().take_while(above).count();
    let fwhm0 = (1 + below + past) as f64;
    let reach = (3.5 * fwhm0).ceil().max(4.0) as usize;
    let (flo, fhi) = (
        (c.round() as usize).saturating_sub(reach),
        ((c.round() as usize) + reach).min(n - 1),
    );
    let xs: Vec<usize> = (flo..=fhi).filter(|&j| !bad(j)).collect();
    let failed = || Err(PeakError::Rejected(Reject::FitFailed));
    if xs.len() < 7 {
        return failed();
    }
    let start = [c, fwhm0, wmax, bg, 0.0];
    let scale = [1.0, fwhm0, wmax.abs(), bg.abs().max(1.0), 1.0];
    let model = |p: &[f64]| -> Vec<f64> {
        xs.iter()
            .map(|&j| (pseudo_voigt(j as f64, p) - row[j]) / row[j].max(1.0).sqrt())
            .collect()
    };
    let Ok(sol) = lm::solve(model, &start, &scale, &lm::Options::default()) else {
        return failed();
    };
    if sol.x[0] < flo as f64 || sol.x[0] > fhi as f64 || sol.x[1] == 0.0 {
        return failed();
    }
    let Ok(cov) = sol.covariance_of(&[0, 1, 2, 3], 1e-12) else {
        return failed();
    };
    let s2 = (2.0 * sol.cost / (xs.len() - 5) as f64).max(1.0);
    Ok(Peak1 {
        pos: sol.x[0],
        sigma: (s2 * cov[(0, 0)]).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    /// Gaussian blob plus flat background, counts with Poisson-sized noise.
    fn blob(
        shape: (usize, usize),
        c: [f64; 2],
        sg: f64,
        amp: f64,
        bg: f64,
        rng: &mut Rng,
    ) -> Vec<f64> {
        (0..shape.0 * shape.1)
            .map(|k| {
                let (i, j) = ((k / shape.1) as f64, (k % shape.1) as f64);
                let r2 = (i - c[0]).powi(2) + (j - c[1]).powi(2);
                let mean = bg + amp * (-0.5 * r2 / (sg * sg)).exp();
                mean + mean.sqrt() * rng.normal()
            })
            .collect()
    }

    fn img(data: &[f64], shape: (usize, usize)) -> Image<'_> {
        Image {
            data,
            shape,
            mask: None,
            saturation: None,
        }
    }

    #[test]
    fn subpixel_position_and_error_coverage() {
        let shape = (64, 64);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut sum2, mut count) = (0.0, 0.0);
        for _ in 0..300 {
            let c = [30.0 + rng.uniform() * 4.0, 28.0 + rng.uniform() * 4.0];
            let data = blob(shape, c, 3.0, 300.0, 5.0, &mut rng);
            let p = find_peak(&img(&data, shape), &PeakOptions::default()).unwrap();
            for a in 0..2 {
                assert!((p.pos[a] - c[a]).abs() < 6.0 * p.sigma[a], "{p:?} vs {c:?}");
                sum2 += ((p.pos[a] - c[a]) / p.sigma[a]).powi(2);
                count += 1.0;
            }
        }
        let rms = (sum2 / count).sqrt();
        assert!((0.7..1.4).contains(&rms), "rms of error/sigma = {rms}");
    }

    #[test]
    fn hot_pixel_outside_the_window_is_ignored() {
        let shape = (64, 64);
        let mut rng = Rng(7);
        let mut data = blob(shape, [20.3, 21.6], 3.0, 300.0, 5.0, &mut rng);
        data[60 * 64 + 60] = 3000.0;
        let opts = PeakOptions {
            window: 20,
            ..PeakOptions::default()
        };
        let p = find_peak(&img(&data, shape), &opts).unwrap();
        assert!(
            (p.pos[0] - 20.3).abs() < 0.1 && (p.pos[1] - 21.6).abs() < 0.1,
            "{p:?}"
        );
        // The same frame with one pass cannot settle: the starting window is
        // the box with the largest sum, a pixel or so from the true centre.
        let one = PeakOptions {
            max_passes: 1,
            ..opts
        };
        assert_eq!(
            find_peak(&img(&data, shape), &one),
            Err(PeakError::Rejected(Reject::NotConverged { passes: 1 }))
        );
    }

    #[test]
    fn a_masked_hot_pixel_does_not_count() {
        let shape = (64, 64);
        let mut rng = Rng(11);
        let mut data = blob(shape, [30.0, 30.0], 3.0, 300.0, 5.0, &mut rng);
        let mut mask = vec![false; 64 * 64];
        data[40 * 64 + 40] = 1e9;
        mask[40 * 64 + 40] = true;
        let image = Image {
            mask: Some(&mask),
            ..img(&data, shape)
        };
        let p = find_peak(&image, &PeakOptions::default()).unwrap();
        assert!((p.pos[0] - 30.0).abs() < 0.1 && (p.pos[1] - 30.0).abs() < 0.1);
    }

    #[test]
    fn peak_on_the_edge_is_rejected() {
        let shape = (64, 64);
        let mut rng = Rng(3);
        let data = blob(shape, [1.5, 30.0], 3.0, 300.0, 5.0, &mut rng);
        assert_eq!(
            find_peak(&img(&data, shape), &PeakOptions::default()),
            Err(PeakError::Rejected(Reject::TouchesEdge))
        );
    }

    #[test]
    fn saturated_pixel_is_rejected_with_its_place() {
        let shape = (64, 64);
        let mut rng = Rng(5);
        let data = blob(shape, [30.0, 30.0], 3.0, 300.0, 5.0, &mut rng);
        let image = Image {
            saturation: Some(250.0),
            ..img(&data, shape)
        };
        let Err(PeakError::Rejected(Reject::Saturated { slow, fast })) =
            find_peak(&image, &PeakOptions::default())
        else {
            panic!("not rejected as saturated");
        };
        assert!((28..=32).contains(&slow) && (28..=32).contains(&fast));
    }

    #[test]
    fn flat_frame_has_no_signal_and_bad_input_is_an_error() {
        let flat = vec![5.0; 32 * 32];
        assert_eq!(
            find_peak(&img(&flat, (32, 32)), &PeakOptions::default()),
            Err(PeakError::Rejected(Reject::NoSignal))
        );
        assert!(matches!(
            find_peak(&img(&flat, (32, 31)), &PeakOptions::default()),
            Err(PeakError::Shape { .. })
        ));
        let mut nan = flat.clone();
        nan[40] = f64::NAN;
        assert_eq!(
            find_peak(&img(&nan, (32, 32)), &PeakOptions::default()),
            Err(PeakError::NonFinite { slow: 1, fast: 8 })
        );
        assert!(matches!(
            find_peak(&img(&flat[..4], (2, 2)), &PeakOptions::default()),
            Err(PeakError::TooSmall { .. })
        ));
    }

    /// 800 x 800 frame, 2 px Gaussian of amplitude 2000 at (530, 540), the
    /// border strips at level 5 and the interior `lift(i, j)` above it.
    fn big_frame(lift: impl Fn(usize, usize) -> f64) -> Vec<f64> {
        let mut rng = Rng(99);
        let n = 800;
        (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                let edge = i < 3 || i >= n - 3 || j < 3 || j >= n - 3;
                let r2 = (i as f64 - 530.0).powi(2) + (j as f64 - 540.0).powi(2);
                let mean = 5.0 + if edge { 0.0 } else { lift(i, j) } + 2000.0 * (-r2 / 8.0).exp();
                mean + mean.sqrt() * rng.normal()
            })
            .collect()
    }

    #[test]
    fn leftover_background_does_not_pull_the_peak_off() {
        let shape = (800, 800);
        // Interior 0.1 above the border median: 640000 x 0.1 outweighs the
        // peak's 50000 counts in a whole-frame centre of mass.
        let flat = big_frame(|_, _| 0.1);
        // Background rising across the frame, border strip at the low side.
        let ramp = big_frame(|_, j| 0.0004 * j as f64);
        for data in [flat, ramp] {
            let p = find_peak(&img(&data, shape), &PeakOptions::default())
                .expect("a clear peak must be found");
            assert!(
                (p.pos[0] - 530.0).abs() < 6.0 * p.sigma[0]
                    && (p.pos[1] - 540.0).abs() < 6.0 * p.sigma[1],
                "{p:?}"
            );
        }
    }

    #[test]
    fn background_alone_is_no_signal() {
        // Frames without a peak, interior above the border by a level or a
        // ramp: a window that settles on background must be rejected.
        let shape = (800, 800);
        for seed in 1..=4 {
            for ramp in [false, true] {
                let mut rng = Rng(seed);
                let data: Vec<f64> = (0..800 * 800)
                    .map(|k| {
                        let (i, j) = (k / 800, k % 800);
                        let edge = !(3..797).contains(&i) || !(3..797).contains(&j);
                        let lift = if ramp { 0.0004 * j as f64 } else { 0.1 };
                        let mean: f64 = 5.0 + if edge { 0.0 } else { lift };
                        mean + mean.sqrt() * rng.normal()
                    })
                    .collect();
                assert_eq!(
                    find_peak(&img(&data, shape), &PeakOptions::default()),
                    Err(PeakError::Rejected(Reject::NoSignal)),
                    "seed {seed} ramp {ramp}"
                );
            }
        }
        let mut rng = Rng(5);
        let data: Vec<f64> = (0..800)
            .map(|j| {
                let mean: f64 = if (3..797).contains(&j) { 5.3 } else { 5.0 };
                mean + mean.sqrt() * rng.normal()
            })
            .collect();
        let lifted = data;
        assert_eq!(
            find_peak_1d(&lifted, None, None, &PeakOptions::default()),
            Err(PeakError::Rejected(Reject::NoSignal))
        );
    }

    fn row_pv(n: usize, c: f64, fwhm: f64, eta: f64, amp: f64, bg: f64, rng: &mut Rng) -> Vec<f64> {
        (0..n)
            .map(|j| {
                let t = (j as f64 - c) / fwhm;
                let g = (-4.0 * std::f64::consts::LN_2 * t * t).exp();
                let l = 1.0 / (1.0 + 4.0 * t * t);
                let mean = bg + amp * (eta * l + (1.0 - eta) * g);
                mean + mean.sqrt() * rng.normal()
            })
            .collect()
    }

    #[test]
    fn pseudo_voigt_row_position_within_its_error() {
        let mut rng = Rng(0xDEAD_BEEF_1234);
        let (mut sum2, mut count) = (0.0, 0.0);
        for _ in 0..200 {
            let c = 100.0 + rng.uniform() * 3.0;
            let row = row_pv(200, c, 6.0, 0.4, 2000.0, 20.0, &mut rng);
            let p = find_peak_1d(&row, None, None, &PeakOptions::default()).unwrap();
            assert!((p.pos - c).abs() < 0.1, "{p:?} vs {c}");
            sum2 += ((p.pos - c) / p.sigma).powi(2);
            count += 1.0;
        }
        let rms = (sum2 / count).sqrt();
        assert!((0.7..1.4).contains(&rms), "rms of error/sigma = {rms}");
    }

    #[test]
    fn row_rejections_name_their_reason() {
        let mut rng = Rng(21);
        let edge = row_pv(200, 2.0, 6.0, 0.4, 2000.0, 20.0, &mut rng);
        assert_eq!(
            find_peak_1d(&edge, None, None, &PeakOptions::default()),
            Err(PeakError::Rejected(Reject::TouchesEdge))
        );
        let row = row_pv(200, 100.0, 6.0, 0.4, 2000.0, 20.0, &mut rng);
        assert!(matches!(
            find_peak_1d(&row, None, Some(1500.0), &PeakOptions::default()),
            Err(PeakError::Rejected(Reject::Saturated { slow: 0, .. }))
        ));
        let flat = vec![5.0; 100];
        assert_eq!(
            find_peak_1d(&flat, None, None, &PeakOptions::default()),
            Err(PeakError::Rejected(Reject::NoSignal))
        );
    }
}
