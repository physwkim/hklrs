//! Peak positions in detector frames: a windowed centre of mass for images,
//! and the same followed by a pseudo-Voigt fit for one-dimensional rows.
//!
//! Positions are fractional pixel indices with pixel centres at integers, as
//! [`crate::PixelDetector::pixel`] counts. Pixel values must be counts: the
//! position error is the Poisson error of the centroid, `√(Σ dᵢ (xᵢ − c)²) / N`
//! over the window, with `dᵢ` the pixel counts (at least 1) and `N` the
//! background-subtracted sum. Background pixels count, so a wide window
//! costs precision; for a pure peak it reduces to second moment over √N.

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
    /// No positive signal above the border background in the window.
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
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self {
            window: 50,
            tol: 0.01,
            max_passes: 20,
            border: 3,
            edge_fraction: 0.1,
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

/// Re-centre a window of half-width `half` until the centre moves less than
/// `tol`; the whole image is the first window.
fn settle(
    w: &[f64],
    ns: usize,
    nf: usize,
    half: [f64; 2],
    opts: &PeakOptions,
) -> Result<[f64; 2], Reject> {
    let inf = [f64::INFINITY; 2];
    let (mut c, total) = com(w, ns, nf, [0.0; 2], inf);
    if total.is_nan() || total <= 0.0 {
        return Err(Reject::NoSignal);
    }
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
    let c = settle(&w, ns, nf, half, opts).map_err(PeakError::Rejected)?;
    check_window(img, &w, c, half, opts).map_err(PeakError::Rejected)?;
    let (_, n) = com(&w, ns, nf, c, half);
    let (mut va, mut vb) = (0.0, 0.0);
    for (i, j, om) in pixels(c, half, ns, nf) {
        let k = i * nf + j;
        if !bad(k) {
            let var = om * om * img.data[k].max(1.0);
            va += var * (i as f64 - c[0]).powi(2);
            vb += var * (j as f64 - c[1]).powi(2);
        }
    }
    Ok(Peak {
        pos: c,
        sigma: [va.sqrt() / n, vb.sqrt() / n],
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
    let c = settle(&w, 1, n, half, opts).map_err(PeakError::Rejected)?;
    check_window(&img, &w, c, half, opts).map_err(PeakError::Rejected)?;
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
        // The same frame with one pass cannot settle: the whole-image centre
        // of mass sits between the peak and the hot pixel.
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
