//! The normaliser against xrayutilities 1.8.0 deff08f IntensityNormalizer
//! (golden/xu_normalise.py), on counts / weight per pixel.
//!
//! Where xu is correct the values agree to 1e-12. Where xu is wrong, the
//! test asserts what xu returned and that hklrs differs, citing the
//! upstream-defects entry.

pub mod common;

use common::{f, load};
use hklrs::nalgebra::Vector3;
use hklrs::normalise::{FrameScalars, Normalised, Normaliser, PixelFields, Region};
use hklrs::{Error, absorption::Masked};
use hklrs::{Image, PixelDetector};
use serde_json::Value;

const TOL: f64 = 1e-12;

fn numbers(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| numbers(x, out)),
        Value::Null => {}
        x => out.push(f(x)),
    }
}

fn flat(v: &Value) -> Vec<f64> {
    let mut out = vec![];
    numbers(v, &mut out);
    out
}

/// The detector's shape (slow, fast) for a frame of xu shape `shape`: a
/// point is one pixel, a PSD one row.
fn shape(case: &Value) -> (usize, usize) {
    let dims: Vec<usize> = case["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    match dims[..] {
        [] => (1, 1),
        [n] => (1, n),
        [a, b] => (a, b),
        _ => panic!("{dims:?}"),
    }
}

fn detector((h, w): (usize, usize)) -> PixelDetector {
    PixelDetector::new(
        (h, w),
        Vector3::new(0.4, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1e-3),
        Vector3::new(0.0, 1e-3, 0.0),
    )
    .unwrap()
}

/// What xu is given, rebuilt for hklrs: per frame, `(raw, scalars)` and
/// the normaliser. `Err` is the first frame-level error.
struct Case {
    frames: Vec<Image<f64>>,
    scalars: Vec<Result<FrameScalars, Error>>,
    normaliser: Normaliser,
}

fn build(case: &Value) -> Case {
    let shape = shape(case);
    let size = shape.0 * shape.1;
    let raw = flat(&case["raw"]);
    let frames: Vec<Image<f64>> = raw
        .chunks(size)
        .map(|f| Image::new(shape, f.to_vec()).unwrap())
        .collect();
    let n = frames.len();
    let per_frame = |key: &str| case[key].as_array().map(|_| flat(&case[key]));
    let mon = per_frame("mon");
    let absfun = per_frame("absfun");
    let time_array = per_frame("time");
    let time_float = case["time"].as_f64();
    let reference = match (&mon, case["av_mon"].as_f64()) {
        (_, Some(v)) => Some(v),
        (Some(m), None) => Some(m.iter().sum::<f64>() / m.len() as f64),
        (None, None) => None,
    };
    let scalars = (0..n)
        .map(|k| {
            let mut s = FrameScalars::new();
            if let Some(m) = &mon {
                s = s.monitor(m[k], reference.unwrap_or(f64::NAN))?;
            }
            if let Some(t) = &time_array {
                s = s.count_time(t[k])?;
            } else if let Some(t) = time_float {
                s = s.count_time(t)?;
            }
            if let Some(a) = &absfun {
                // xu multiplies by absfun; the absorber transmits its inverse
                s = s.transmission(1.0 / a[k])?;
            }
            Ok(s)
        })
        .collect();
    let mut fields = PixelFields::new(shape);
    if case["dark"].is_array() {
        fields = fields
            .dark_rate(Image::new(shape, flat(&case["dark"])).unwrap())
            .unwrap();
    }
    if case["flat"].is_array() {
        fields = fields
            .flat(Image::new(shape, flat(&case["flat"])).unwrap())
            .unwrap();
    }
    let (_, r) = Region::new(0..shape.0, 0..shape.1, (1, 1))
        .apply(&detector(shape))
        .unwrap();
    Case {
        frames,
        scalars,
        normaliser: Normaliser::new(fields, r).unwrap(),
    }
}

impl Case {
    fn shape(&self) -> (usize, usize) {
        self.normaliser.raw_shape()
    }

    /// Frame `k` whole, unbinned: its pixels.
    fn pixels(&self, k: usize) -> Vec<Result<Normalised, Masked>> {
        self.normaliser
            .normalise(&self.frames[k], self.scalars[k].as_ref().unwrap(), &[])
            .unwrap()
            .into_data()
    }
}

fn xu(case: &Value) -> Vec<f64> {
    flat(&case["xu"])
}

fn cases<'a>(doc: &'a Value, group: &str) -> &'a serde_json::Map<String, Value> {
    doc[group].as_object().unwrap()
}

#[test]
fn counts_over_weight_is_xus_intensity() {
    let doc = load("xu_normalise.json");
    for (name, case) in cases(&doc, "compare") {
        let c = build(case);
        let want = xu(case);
        let size = c.shape().0 * c.shape().1;
        for k in 0..c.frames.len() {
            for (i, p) in c.pixels(k).iter().enumerate() {
                let got = p.as_ref().unwrap().value();
                let w = want[k * size + i];
                assert!(
                    (got - w).abs() <= TOL * w.abs(),
                    "{name} frame {k} pixel {i}: {got} vs {w}"
                );
            }
        }
    }
}

/// XU-9: with no monitor, no time array and no absfun xu's factor is a
/// float and its 2D branch returns `raw * c`, ignoring dark and flat.
#[test]
fn xu9_flat_and_dark_apply_without_a_monitor() {
    let doc = load("xu_normalise.json");
    let div = cases(&doc, "divergence");
    let case = &div["xu9_psd_flat_no_monitor"];
    let c = build(case);
    let raw = c.frames[0].data();
    assert_eq!(xu(case), raw, "xu skips the flat field");
    let gain = flat(&case["flat"]);
    let mean = gain.iter().sum::<f64>() / gain.len() as f64;
    for (i, p) in c.pixels(0).iter().enumerate() {
        let want = raw[i] / (gain[i] / mean);
        assert!(close(p.as_ref().unwrap().value(), want), "{i}");
    }

    let case = &div["xu9_psd_dark_flat_time_float"];
    let c = build(case);
    let raw = c.frames[0].data();
    for (x, r) in xu(case).iter().zip(raw) {
        assert!(close(*x, r / 2.5), "xu applies the time only");
    }
    let (dark, gain) = (flat(&case["dark"]), flat(&case["flat"]));
    let mean = gain.iter().sum::<f64>() / gain.len() as f64;
    for (i, p) in c.pixels(0).iter().enumerate() {
        let want = (raw[i] - dark[i] * 2.5) / (2.5 * gain[i] / mean);
        assert!(close(p.as_ref().unwrap().value(), want), "{i}");
    }
}

/// XU-10: a CCD stack with no monitor raises in xu.
#[test]
fn xu10_ccd_without_a_monitor_normalises() {
    let doc = load("xu_normalise.json");
    let case = &cases(&doc, "divergence")["xu10_ccd_no_monitor"];
    assert!(case["xu_error"].as_str().unwrap().starts_with("IndexError"));
    let c = build(case);
    for k in 0..c.frames.len() {
        for (i, p) in c.pixels(k).iter().enumerate() {
            let p = p.as_ref().unwrap();
            assert_eq!((p.counts(), p.weight().get()), (c.frames[k].data()[i], 1.0));
        }
    }
}

/// XU-21: xu's 3D branch ignores dark and flat field arguments
/// ("still missing"), so a CCD stack is corrected for monitor, time and
/// absfun only.
#[test]
fn xu21_ccd_applies_dark_and_flat() {
    let doc = load("xu_normalise.json");
    let case = &cases(&doc, "divergence")["ccd_dark_flat_ignored"];
    let with_fields = build(case);
    let mut without = case.clone();
    without["dark"] = Value::Null;
    without["flat"] = Value::Null;
    let without = build(&without);
    let want = xu(case);
    let size = with_fields.shape().0 * with_fields.shape().1;
    let (mut differs, mut matches_plain) = (0, 0);
    for k in 0..with_fields.frames.len() {
        let (a, b) = (with_fields.pixels(k), without.pixels(k));
        for i in 0..size {
            let (a, b) = (
                a[i].as_ref().unwrap().value(),
                b[i].as_ref().unwrap().value(),
            );
            let x = want[k * size + i];
            matches_plain += close(b, x) as usize;
            differs += !close(a, x) as usize;
        }
    }
    assert_eq!(
        matches_plain,
        with_fields.frames.len() * size,
        "xu equals the uncorrected value"
    );
    assert_eq!(
        differs,
        with_fields.frames.len() * size,
        "hklrs applies the fields"
    );
}

/// XU-22: xu sets a dark-subtracted pixel below 0 to 0; hklrs
/// keeps the sign, which an unbiased mean of low counts needs.
#[test]
fn xu22_dark_subtraction_is_signed() {
    let doc = load("xu_normalise.json");
    let case = &cases(&doc, "divergence")["dark_clamped"];
    let c = build(case);
    let (want, dark) = (xu(case), flat(&case["dark"]));
    let (w, t) = (c.shape().1, flat(&case["time"]));
    let mut negative = 0;
    for k in 0..c.frames.len() {
        for (i, p) in c.pixels(k).iter().enumerate() {
            let p = p.as_ref().unwrap();
            let sub = c.frames[k].data()[i] - dark[i] * t[k];
            assert!(close(p.counts(), sub));
            if sub < 0.0 {
                negative += 1;
                assert_eq!(want[k * w + i], 0.0, "xu clamps");
                assert!(p.value() < 0.0);
            }
        }
    }
    assert!(negative > 0, "the case must have a pixel below the dark");
}

/// XU-23: a monitor of 0 makes xu's factor infinite, and xu
/// replaces it by 1 without a word, returning the raw counts as the
/// normalised intensity; hklrs refuses the frame.
#[test]
fn xu23_a_zero_monitor_is_an_error() {
    let doc = load("xu_normalise.json");
    let case = &cases(&doc, "divergence")["zero_monitor"];
    let c = build(case);
    let (want, raw) = (xu(case), flat(&case["raw"]));
    assert_eq!(want[1], raw[1], "xu returns the raw counts");
    assert!(matches!(c.scalars[1], Err(Error::NotPositive { .. })));
    assert!(c.scalars[0].is_ok() && c.scalars[2].is_ok());
}

/// XU-24: xu sets a flat-field pixel below 1e-5 to 1 and
/// normalises with it; hklrs masks the pixel.
#[test]
fn xu24_a_dead_flat_pixel_is_masked() {
    let doc = load("xu_normalise.json");
    let case = &cases(&doc, "divergence")["flat_below_1e-5"];
    let c = build(case);
    let (want, last) = (xu(case), c.shape().1 - 1);
    for k in 0..c.frames.len() {
        let out = c.pixels(k);
        assert!(
            want[k * c.shape().1 + last].is_finite(),
            "xu returns a number"
        );
        assert_eq!(out[last], Err(Masked::BadFlat));
        assert!(out[..last].iter().all(Result::is_ok));
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL * a.abs().max(b.abs())
}
