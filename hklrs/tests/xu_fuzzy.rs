//! Cube pixel footprints against xrayutilities deff08f's FuzzyGridder1D, 2D
//! and 3D (golden/xu_fuzzy.py). Every pixel has weight 1 and counts = xu's
//! data, so photons, weight and intensity are xu's `_gdata`, `_gnorm` and
//! `.data`. xu's node-centred grid of `n` points from `min` to `max` is the
//! lattice of step `(max - min) / (n - 1)` anchored half a step below `min`,
//! its node `j` being bin `j`. The `xu_*` tests at the end assert where the
//! cube and xu part.

pub mod common;

use common::{f, load};
use hklrs::Image;
use hklrs::absorption::{Masked, Weight};
use hklrs::grid::fuzzy::Footprint;
use hklrs::grid::{Binning, Bounds, Coord, CoordKind, Cube, Projected};
use hklrs::normalise::{Normalised, NormalisedFrame};
use serde_json::Value;

/// Linear coordinates x, y, z of `dims` dimensions.
fn coords(dims: usize) -> Vec<Coord> {
    ["x", "y", "z"][..dims]
        .iter()
        .map(|&n| Coord::linear(n))
        .collect()
}

fn list(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

fn value(counts: f64) -> Result<Normalised, Masked> {
    Normalised::new(counts, Weight::new(1.0, Masked::Opaque))
}

/// One frame of `coords[axis][pixel]` with `data` as counts, laid by `footprint`.
fn frame(
    coords: &[Vec<f64>],
    data: &[f64],
    footprint: Vec<Footprint>,
) -> (Projected, NormalisedFrame) {
    let shape = (1, data.len());
    let planes = coords
        .iter()
        .map(|c| Image::from_fn(shape, |_, i| c[i]))
        .collect();
    let points = Projected::new(self::coords(coords.len()), planes)
        .unwrap()
        .with_footprint(footprint)
        .unwrap();
    (points, Image::from_fn(shape, |_, i| value(data[i])))
}

/// xu's grid of `bins` nodes over `ranges` per axis.
fn binnings(bins: &[usize], ranges: &[(f64, f64)]) -> Vec<Binning> {
    bins.iter()
        .zip(ranges)
        .map(|(&n, &(lo, hi))| {
            let d = (hi - lo) / (n - 1) as f64;
            Binning::new(CoordKind::Linear, d, lo - d / 2.0).unwrap()
        })
        .collect()
}

/// A zero cube on exactly xu's grid: bins 0..n per axis, from a point
/// frame on its first and last nodes.
fn xu_grid(bins: &[usize], ranges: &[(f64, f64)]) -> Cube {
    let corners: Vec<Vec<f64>> = ranges.iter().map(|&(lo, hi)| vec![lo, hi]).collect();
    let (points, pixels) = frame(&corners, &[1.0, 1.0], vec![Footprint::Point; bins.len()]);
    let mut b = Bounds::new(binnings(bins, ranges));
    b.include(&points, &pixels).unwrap();
    let cube = b.cube().unwrap();
    for (a, &n) in cube.axes().iter().zip(bins) {
        assert_eq!((a.first(), a.size()), (0, n));
    }
    cube
}

/// xu forms each share from bin-index arithmetic and multiplies it into
/// the data first; the sums agree to 1e-14.
fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-13 * want.abs().max(1.0)
}

/// The cube's sums at xu's flat (C-order) index `j` against xu's.
fn assert_bin(cube: &Cube, off: usize, xu: &Value, j: usize, what: &str) {
    let (gdata, gnorm) = (f(&xu["gdata"][j]), f(&xu["gnorm"][j]));
    let (p, w) = (cube.photons()[off], cube.weight()[off]);
    assert!(close(p, gdata), "{what} bin {j}: photons {p} vs {gdata}");
    assert!(close(w, gnorm), "{what} bin {j}: weight {w} vs {gnorm}");
    let i = cube.intensity().nth(off).unwrap();
    match i {
        None => assert_eq!(gnorm, 0.0, "{what} bin {j}"),
        Some(i) => assert!(close(i, f(&xu["normalised"][j])), "{what} bin {j}: {i}"),
    }
    // contributions count the pixels that reached the bin
    assert_eq!(cube.contributions()[off] > 0, w > 0.0, "{what} bin {j}");
}

struct Case {
    name: String,
    bins: Vec<usize>,
    ranges: Vec<(f64, f64)>,
    footprint: Vec<Footprint>,
    coords: Vec<Vec<f64>>,
    data: Vec<f64>,
}

fn cases(doc: &Value) -> Vec<(Case, &Value)> {
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let ranges = c["ranges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (f(&r[0]), f(&r[1])))
                .collect();
            let case = Case {
                name: c["name"].as_str().unwrap().to_owned(),
                bins: c["bins"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect(),
                ranges,
                footprint: list(&c["width"])
                    .into_iter()
                    .map(Footprint::Width)
                    .collect(),
                coords: c["coords"].as_array().unwrap().iter().map(list).collect(),
                data: list(&c["data"]),
            };
            (case, c)
        })
        .collect()
}

/// xu's grid holds every footprint: nothing is outside the fixed cube and
/// every bin matches xu's sums.
#[test]
fn fixed_axes_match_xu_fuzzy_gridders() {
    let doc = load("xu_fuzzy.json");
    let all = cases(&doc);
    assert_eq!(all.len(), 9);
    for (c, xu) in all {
        let mut cube = xu_grid(&c.bins, &c.ranges);
        let (points, pixels) = frame(&c.coords, &c.data, c.footprint.clone());
        assert_eq!(cube.add(&points, &pixels).unwrap(), 0, "{}", c.name);
        assert_eq!(cube.photons().len(), c.bins.iter().product::<usize>());
        for j in 0..cube.photons().len() {
            assert_bin(&cube, j, xu, j, &c.name);
        }
    }
}

/// One-pass filling spans the bins the footprints reach, each the xu bin
/// of the same global index.
#[test]
fn growing_axes_match_xu_fuzzy_gridders() {
    let doc = load("xu_fuzzy.json");
    for (c, xu) in cases(&doc) {
        let mut cube = Cube::default();
        let (points, pixels) = frame(&c.coords, &c.data, c.footprint.clone());
        cube.add_growing(&binnings(&c.bins, &c.ranges), &points, &pixels)
            .unwrap();
        let mut k = vec![0i64; c.bins.len()];
        for off in 0..cube.photons().len() {
            // global indices of `off`, the last axis fastest
            let mut rest = off;
            for (d, a) in cube.axes().iter().enumerate().rev() {
                k[d] = a.first() + (rest % a.size()) as i64;
                rest /= a.size();
            }
            assert!(
                k.iter()
                    .zip(&c.bins)
                    .all(|(&k, &n)| (0..n as i64).contains(&k)),
                "{}: bin {k:?} is beyond xu's grid",
                c.name
            );
            let j = k
                .iter()
                .zip(&c.bins)
                .fold(0, |j, (&k, &n)| j * n + k as usize);
            assert_bin(&cube, off, xu, j, &c.name);
        }
    }
}

const R: (f64, f64) = (0.0, 4.0);

fn defect(doc: &Value, name: &str) -> Value {
    doc["defects"][name].clone()
}

/// xu clamps the first and last bin a footprint reaches to the grid but
/// takes their shares from the unclamped ends (gridder1d.c:137-153,
/// gridder2d.c:156-163, gridder3d.c:177-184): the part beyond the grid lands
/// in the edge bin. The cube keeps the true overlap and drops the rest.
#[test]
fn xu_folds_a_footprint_beyond_the_grid_into_the_edge_bin() {
    let doc = load("xu_fuzzy.json");
    // [-1.4, 1.6]: 0.3 below bin 0, 1/3 in bins 0 and 1, 1/30 in bin 2
    let want = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 30.0, 0.0, 0.0];
    let xu = defect(&doc, "edge_overflow_1d");
    let mut cube = xu_grid(&[5], &[R]);
    let (points, pixels) = frame(&[vec![0.1]], &[1.0], vec![Footprint::Width(3.0)]);
    assert_eq!(cube.add(&points, &pixels).unwrap(), 1);
    for (j, &w) in want.iter().enumerate() {
        assert!(close(cube.weight()[j], w), "bin {j}: {}", cube.weight()[j]);
    }
    assert!(close(f(&xu["gnorm"][0]), want[0] + 0.3));
    assert!(close(f(&xu["gnorm"][1]), want[1]));

    let xu = defect(&doc, "edge_overflow_2d");
    let mut cube = xu_grid(&[5, 5], &[R, R]);
    let (points, pixels) = frame(
        &[vec![0.1], vec![2.1]],
        &[1.0],
        vec![Footprint::Width(3.0), Footprint::Width(0.4)],
    );
    assert_eq!(cube.add(&points, &pixels).unwrap(), 1);
    assert!(close(cube.weight()[2], want[0]));
    assert!(close(f(&xu["gnorm"][2]), want[0] + 0.3));
}

/// xu's 1D gridder keeps a pixel when its centre is within half a width of
/// the grid (gridder1d.c:132); 2D and 3D only when it is within [min, max]
/// (gridder2d.c:147, gridder3d.c:161), though bin 0 starts half a step
/// below min. A footprint inside bin 0 is in the cube either way.
#[test]
fn xu_range_checks_drop_footprints_inside_bin_zero() {
    let doc = load("xu_fuzzy.json");
    // [-0.35, 0.05]: xu 1D keeps it, 2D drops it
    let mut cube = xu_grid(&[5], &[R]);
    let (points, pixels) = frame(&[vec![-0.15]], &[1.0], vec![Footprint::Width(0.4)]);
    assert_eq!(cube.add(&points, &pixels).unwrap(), 0);
    assert_eq!(cube.weight()[0], 1.0);
    assert_eq!(f(&defect(&doc, "range_keep_1d")["gnorm"][0]), 1.0);

    let mut cube = xu_grid(&[5, 5], &[R, R]);
    let (points, pixels) = frame(
        &[vec![-0.15], vec![2.1]],
        &[1.0],
        vec![Footprint::Width(0.4); 2],
    );
    assert_eq!(cube.add(&points, &pixels).unwrap(), 0);
    assert_eq!(cube.weight()[2], 1.0);
    assert!(
        list(&defect(&doc, "range_drop_2d")["gnorm"])
            .iter()
            .all(|&w| w == 0.0)
    );

    // [-0.4, -0.2]: xu 1D drops it too
    let mut cube = xu_grid(&[5], &[R]);
    let (points, pixels) = frame(&[vec![-0.3]], &[1.0], vec![Footprint::Width(0.2)]);
    assert_eq!(cube.add(&points, &pixels).unwrap(), 0);
    assert_eq!(cube.weight()[0], 1.0);
    assert!(
        list(&defect(&doc, "range_drop_1d")["gnorm"])
            .iter()
            .all(|&w| w == 0.0)
    );
}

/// `if not width:` (gridder.py:354) makes a zero width half a step in xu's
/// 1D gridder only; 2D takes it as a point. A zero width is refused here and
/// a point is `Footprint::Point`, which is xu 2D's result.
#[test]
fn xu_1d_takes_a_zero_width_as_half_a_step() {
    let doc = load("xu_fuzzy.json");
    let shape = (1, 1);
    let points = Projected::new(coords(1), vec![Image::from_fn(shape, |_, _| 0.3)]).unwrap();
    assert!(matches!(
        points.with_footprint(vec![Footprint::Width(0.0)]),
        Err(hklrs::Error::Width { .. })
    ));

    let mut cube = xu_grid(&[5], &[R]);
    let (points, pixels) = frame(&[vec![0.3]], &[1.0], vec![Footprint::Point]);
    cube.add(&points, &pixels).unwrap();
    let xu2: Vec<f64> = (0..5)
        .map(|i| f(&defect(&doc, "zero_width_2d")["gnorm"][i * 5 + 2]))
        .collect();
    assert_eq!(cube.weight(), xu2);
    assert_eq!(cube.weight(), [1.0, 0.0, 0.0, 0.0, 0.0]);
    let xu1 = list(&defect(&doc, "zero_width_1d")["gnorm"]);
    assert!(close(xu1[0], 0.9) && close(xu1[1], 0.1));
}

/// A NaN coordinate passes xu's range checks and `gindex`
/// (gridder_utils.c:77) casts rint(NaN) to unsigned, undefined behaviour;
/// on x86-64 the pixel lands in bin 0. The cube refuses the frame.
#[test]
fn xu_bins_a_nan_coordinate() {
    let doc = load("xu_fuzzy.json");
    let mut cube = xu_grid(&[5], &[R]);
    let (points, pixels) = frame(
        &[vec![f64::NAN, 2.1]],
        &[7.0, 1.0],
        vec![Footprint::Width(0.4)],
    );
    let before = cube.clone();
    assert!(matches!(
        cube.add(&points, &pixels),
        Err(hklrs::Error::NoBin { .. })
    ));
    assert_eq!(cube, before);
    assert_eq!(f(&defect(&doc, "nan_coordinate_1d")["gdata"][0]), 7.0);
}
