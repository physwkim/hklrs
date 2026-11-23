//! Line-cut accumulation against xrayutilities deff08f (golden/xu_cuts.py).
//! Weight 1 and counts = xu's intensity, so Σcounts / contributions per bin is
//! xu's `Gridder1D` output and Σcounts / Σweight is its `FuzzyGridder1D` output
//! (`_get_cut`). xu's node-centred grid of `nx` points from `xmin` to `xmax`
//! is the lattice of step `(xmax - xmin) / (nx - 1)` anchored half a step
//! below `xmin`: its node `j` is bin `j`. XU-19 is asserted with its ID.

pub mod common;

use common::{f, load};
use hklrs::PixelDetector;
use hklrs::Wavelength;
use hklrs::absorption::Masked;
use hklrs::cut::{Cut, CutError, CutKind, CutResult, Footprint, IntDir};
use hklrs::grid::Binning;
use hklrs::nalgebra::Vector3;
use hklrs::normalise::{FrameScalars, Normalised, Normaliser, PixelFields, Region};
use serde_json::Value;

const NPOINTS: f64 = 40.0;

fn points(doc: &Value, cloud: &str) -> Vec<Vector3<f64>> {
    let c = &doc["clouds"][cloud];
    (0..c[0].as_array().unwrap().len())
        .map(|i| Vector3::new(f(&c[0][i]), f(&c[1][i]), f(&c[2][i])))
        .collect()
}

fn intdir(s: &str) -> IntDir {
    match s {
        "q" => IntDir::Q,
        "omega" => IntDir::Omega,
        "2theta" => IntDir::TwoTheta,
        "radial" => IntDir::Radial,
        _ => panic!("{s}"),
    }
}

fn vec3(v: &Value) -> [f64; 3] {
    [f(&v[0]), f(&v[1]), f(&v[2])]
}

fn at3(kind: &str, a: &Value) -> [f64; 3] {
    let v = |i: usize| f(&a[i]);
    match kind {
        "qz" => [v(0), v(1), 0.0],
        "qy" => [v(0), 0.0, v(1)],
        "qx" => [0.0, v(0), v(1)],
        _ => [v(0), v(1), v(2)],
    }
}

fn kind(c: &Value) -> CutKind {
    match c["kind"].as_str().unwrap() {
        "qz" => CutKind::Qz,
        "qy" => CutKind::Qy,
        "qx" => CutKind::Qx,
        "omega" => CutKind::Omega,
        "radial" => CutKind::Radial,
        "twotheta" => CutKind::TwoTheta,
        "line" => CutKind::Line {
            point: vec3(&c["point"]),
            dir: vec3(&c["dir"]),
        },
        k => panic!("{k}"),
    }
}

/// Unit of xu's cut coordinate: radians per golden unit.
fn along_scale(c: &Value) -> f64 {
    if matches!(c["kind"].as_str().unwrap(), "omega" | "radial" | "twotheta") {
        1f64.to_radians()
    } else {
        1.0
    }
}

fn perp_scale(c: &Value) -> f64 {
    if c["intdir"] == "q" {
        1.0
    } else {
        1f64.to_radians()
    }
}

/// `n` pixels of weight exactly 1 and counts `counts[i]` (a NaN count is not
/// possible here: the normaliser masks it as `BadCount`, tested below).
fn unit_pixels(counts: &[f64]) -> Vec<Result<Normalised, Masked>> {
    let n = counts.len();
    let det = PixelDetector::new(
        (1, n),
        Vector3::new(0.4, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1e-3),
        Vector3::new(0.0, 1e-3, 0.0),
    )
    .unwrap();
    let (_, r) = Region::new(0..1, 0..n, (1, 1)).apply(&det).unwrap();
    let frame = Normaliser::new(PixelFields::new((1, n)), r)
        .unwrap()
        .normalise(counts, &FrameScalars::new(), &[])
        .unwrap();
    frame.data
}

struct Setup {
    cut: Cut,
    wl: Wavelength,
    q: Vec<Vector3<f64>>,
    width: f64,
}

fn setup(doc: &Value, c: &Value) -> Setup {
    let k = c["kind"].as_str().unwrap();
    let g = &c["grid"];
    let dx = (f(&g["xmax"]) - f(&g["xmin"])) / (NPOINTS - 1.0) * along_scale(c);
    let anchor = f(&g["xmin"]) * along_scale(c) - dx / 2.0;
    let dis = f(&c["intrange"]) / if k == "line" { 1.0 } else { 2.0 } * perp_scale(c);
    let at = if k == "line" {
        [0.0; 3]
    } else {
        at3(k, &c["at"])
    };
    Setup {
        cut: Cut::new(
            kind(c),
            intdir(c["intdir"].as_str().unwrap()),
            at,
            dis,
            Binning::new(dx, anchor).unwrap(),
        )
        .unwrap(),
        wl: Wavelength::new(f(&doc["wavelength"])).unwrap(),
        q: points(doc, c["cloud"].as_str().unwrap()),
        width: f(&g["width"]) * along_scale(c),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + b.abs())
}

fn zip<'a>(
    q: &'a [Vector3<f64>],
    px: Vec<Result<Normalised, Masked>>,
) -> impl Iterator<Item = (Vector3<f64>, Result<Normalised, Masked>)> + 'a {
    q.iter().copied().zip(px)
}

fn intensity(doc: &Value, c: &Value) -> Vec<f64> {
    doc["intensity"][c["cloud"].as_str().unwrap()]
        .as_array()
        .unwrap()
        .iter()
        .map(f)
        .collect()
}

#[test]
fn point_footprint_is_xu_gridder1d() {
    let doc = load("xu_cuts.json");
    let mut checked = 0;
    for c in doc["cases"].as_array().unwrap() {
        if c.get("grid").is_none() || c["grid"].get("point").is_none() {
            continue;
        }
        let s = setup(&doc, c);
        let inten = intensity(&doc, c);
        let r = s
            .cut
            .accumulate(s.wl, Footprint::Point, zip(&s.q, unit_pixels(&inten)))
            .unwrap();
        let g = &c["grid"];
        let name = format!("{} {} {}", c["cloud"], c["kind"], c["intdir"]);
        let used: i64 = c["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .sum();
        assert_eq!(g["selected"].as_i64().unwrap(), used, "{name}");
        let mut total = 0;
        for (k, b) in r.bins() {
            assert!(
                (0..NPOINTS as i64).contains(&k),
                "{name}: bin {k} off xu's grid"
            );
            total += b.contributions;
            assert_eq!(b.weight, b.contributions as f64, "{name} bin {k}");
        }
        assert_eq!(total as i64, used, "{name}");
        for j in 0..NPOINTS as usize {
            let want_n = f(&g["point_norm"][j]);
            let b = r.get(j as i64).copied().unwrap_or_default();
            assert_eq!(b.contributions as f64, want_n, "{name} bin {j} count");
            if want_n > 0.0 {
                assert!(
                    close(b.counts / b.contributions as f64, f(&g["point"][j])),
                    "{name} bin {j}: {} vs xu {}",
                    b.counts / b.contributions as f64,
                    f(&g["point"][j])
                );
            }
        }
        assert_eq!(
            r.skipped().total() as usize,
            s.q.len() - used as usize,
            "{name}"
        );
        checked += 1;
    }
    assert_eq!(checked, 2 * (15 + 2) - 2 * 2);
}

fn assert_fuzzy(name: &str, r: &CutResult, g: &Value) {
    for j in 0..NPOINTS as usize {
        let want = f(&g["fuzzy"][j]);
        match r.get(j as i64) {
            Some(b) if b.weight > 1e-16 => assert!(
                close(b.counts / b.weight, want),
                "{name} bin {j}: {} vs xu {want}",
                b.counts / b.weight
            ),
            _ => assert_eq!(want, 0.0, "{name} bin {j}: xu has data, hklrs none"),
        }
    }
}

#[test]
fn width_footprint_is_xu_fuzzy_gridder1d() {
    let doc = load("xu_cuts.json");
    let mut checked = 0;
    for c in doc["cases"].as_array().unwrap() {
        if c.get("grid").is_none() || c.get("nan_at").is_some() {
            continue;
        }
        let s = setup(&doc, c);
        let r = s
            .cut
            .accumulate(
                s.wl,
                Footprint::Width(s.width),
                zip(&s.q, unit_pixels(&intensity(&doc, c))),
            )
            .unwrap();
        assert_fuzzy(
            &format!("{} {} {}", c["cloud"], c["kind"], c["intdir"]),
            &r,
            &c["grid"],
        );
        // spreading shares one pixel's weight out: Σweight over the bins is the pixel count
        let w: f64 = r.bins().map(|(_, b)| b.weight).sum();
        let used = c["mask"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v.as_i64() == Some(1))
            .count();
        assert!(close(w, used as f64), "{w} vs {used}");
        checked += 1;
    }
    assert_eq!(checked, 2 * (15 + 2) - 2 * 2);
}

/// XU-19: xu's used-mask counts a selected pixel whose intensity is NaN
/// although its gridder drops it; here `Normalised::new` masks it as
/// `BadCount`, it contributes to no bin, and the other pixels still give
/// xu's gridded values.
#[test]
fn xu_19_nan_intensity_is_not_a_contribution() {
    let doc = load("xu_cuts.json");
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c.get("nan_at").is_some())
        .unwrap();
    let s = setup(&doc, c);
    let hit: Vec<usize> = c["nan_at"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert!(hit.len() > 3);
    let mut px = unit_pixels(&intensity(&doc, c));
    let w = px[0].unwrap().weight();
    for &i in &hit {
        px[i] = Normalised::new(f64::NAN, Ok(w));
    }
    let xu_used: i64 = c["mask"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .sum();
    assert!(
        hit.iter().all(|&i| c["mask"][i].as_i64() == Some(1)),
        "xu marks the NaN pixels used"
    );
    let r = s
        .cut
        .accumulate(s.wl, Footprint::Width(s.width), zip(&s.q, px.clone()))
        .unwrap();
    assert_eq!(r.skipped().masked(Masked::BadCount) as usize, hit.len());
    // a point footprint puts one contribution per accepted pixel
    let p = s
        .cut
        .accumulate(s.wl, Footprint::Point, zip(&s.q, px))
        .unwrap();
    let contributing: u64 = p.bins().map(|(_, b)| b.contributions).sum();
    assert_eq!(contributing as i64, xu_used - hit.len() as i64);
    assert_eq!(p.skipped().masked(Masked::BadCount) as usize, hit.len());
    assert!(
        r.bins()
            .all(|(_, b)| b.counts.is_finite() && b.weight.is_finite())
    );
    assert_fuzzy("xu-19", &r, &c["grid"]);
}

#[test]
fn a_nan_raw_count_is_masked_by_the_normaliser_and_counted_by_reason() {
    let doc = load("xu_cuts.json");
    let c = &doc["cases"][0];
    let s = setup(&doc, c);
    let mut inten = intensity(&doc, c);
    inten[3] = f64::NAN;
    let px = unit_pixels(&inten);
    assert_eq!(px[3], Err(Masked::BadCount));
    let r = s
        .cut
        .accumulate(s.wl, Footprint::Point, zip(&s.q, px))
        .unwrap();
    assert_eq!(r.skipped().masked(Masked::BadCount), 1);
    assert_eq!(r.skipped().masked(Masked::Unlit), 0);
}

#[test]
fn merge_of_two_halves_is_the_whole_and_checks_binning() {
    let doc = load("xu_cuts.json");
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "radial" && c["grid"].get("point").is_some())
        .unwrap();
    let s = setup(&doc, c);
    let px = unit_pixels(&intensity(&doc, c));
    let mut px2 = px.clone();
    px2[5] = Err(Masked::Unlit);
    px2[9] = Err(Masked::Detector);
    for fp in [Footprint::Point, Footprint::Width(s.width)] {
        let whole = s.cut.accumulate(s.wl, fp, zip(&s.q, px2.clone())).unwrap();
        let (a, b) = (s.q.len() / 3, s.q.len());
        let lo = s
            .cut
            .accumulate(s.wl, fp, zip(&s.q[..a], px2[..a].to_vec()))
            .unwrap();
        let hi = s
            .cut
            .accumulate(s.wl, fp, zip(&s.q[a..b], px2[a..b].to_vec()))
            .unwrap();
        let both = lo.clone().merge(hi.clone()).unwrap();
        assert_eq!(both.skipped(), whole.skipped());
        assert_eq!(both.bins().count(), whole.bins().count());
        for ((k1, x), (k2, y)) in both.bins().zip(whole.bins()) {
            assert_eq!(k1, k2);
            assert_eq!(x.contributions, y.contributions);
            assert!(close(x.counts, y.counts) && close(x.weight, y.weight));
        }
        // commutes on the sums
        let rev = hi.merge(lo).unwrap();
        assert_eq!(rev.skipped(), whole.skipped());
        assert!(
            rev.bins()
                .zip(whole.bins())
                .all(|((_, x), (_, y))| close(x.counts, y.counts))
        );
    }
    let other = Cut::new(
        CutKind::Radial,
        IntDir::Omega,
        [0.0; 3],
        1.0,
        Binning::new(0.5, 0.0).unwrap(),
    )
    .unwrap();
    let a = s
        .cut
        .accumulate(s.wl, Footprint::Point, std::iter::empty())
        .unwrap();
    let b = other
        .accumulate(s.wl, Footprint::Point, std::iter::empty())
        .unwrap();
    assert!(a.clone().merge(b).is_err());
    let w = s
        .cut
        .accumulate(s.wl, Footprint::Width(0.1), std::iter::empty())
        .unwrap();
    assert!(matches!(a.merge(w), Err(CutError::Footprint { .. })));
}

#[test]
fn a_bad_footprint_width_is_an_error() {
    let doc = load("xu_cuts.json");
    let s = setup(&doc, &doc["cases"][0]);
    for w in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            s.cut
                .accumulate(s.wl, Footprint::Width(w), std::iter::empty())
                .is_err()
        );
    }
}
