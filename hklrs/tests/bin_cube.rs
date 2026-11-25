//! The hklrs Cube against binoculars-ng 7be4f79's own accumulation of a
//! synthetic 20-frame K6C komega scan in qx, qy, qz
//! (golden/bin_7be4f79_cube.c). Bins are `Binning::new(Linear, res,
//! -res/2)`, the bins binoculars centres on multiples of res. Images are
//! integers and weights powers of two, so photons compare exactly.
//!
//! The normaliser takes binoculars' corrected intensity, image·weight/flat
//! with flat 1, as each frame's raw counts and has nothing to weigh them
//! by, so every pixel has weight 1: Σcounts is binoculars' photons and
//! Σweight its contributions. binoculars' mask is the detector's.

pub mod common;
use common::{f, load};

use std::ops::Range;

use hklrs::frame_map::{FrameMap, MappedFrame};
use hklrs::grid::{Binning, Bounds, CoordKind, Cube};
use hklrs::nalgebra::Vector3;
use hklrs::normalise::{FrameScalars, PixelFields, Region};
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{Image, PixelDetector, Wavelength};
use serde_json::Value;

/// binoculars' single-precision Q against hklrs' (bin_projections.rs).
const Q_TOL: f64 = 1e-6;
/// Closest a pixel may come to a bin edge: ten times `Q_TOL`, so
/// binoculars' coordinate is in the same bin and floor and rint agree.
const MARGIN: f64 = 1e-5;

/// The scan's counts and per-frame weight, as the C driver makes them.
fn image(frame: usize, pixel: usize) -> f64 {
    ((7 * pixel + 13 * frame) % 17) as f64
}

fn weight(frame: usize) -> f64 {
    2f64.powi(frame as i32 % 5 - 2)
}

struct Scan {
    golden: Value,
    map: FrameMap,
    /// Every pixel of each frame, masked or not.
    frames: Vec<MappedFrame>,
}

fn scan() -> Scan {
    let golden = load("bin_7be4f79_cube.json");
    let list = |k: &str| -> Vec<f64> { golden[k].as_array().unwrap().iter().map(f).collect() };
    let v = |k: &str| {
        let a = list(k);
        Vector3::new(a[0], a[1], a[2])
    };
    let geometry = preset("K6C").unwrap();
    let scanned = geometry
        .axes()
        .iter()
        .position(|a| a.name() == golden["scanned"].as_str().unwrap())
        .unwrap();
    let wavelength = Wavelength::new(f(&golden["wavelength"])).unwrap();
    let shape = list("shape");
    let shape = (shape[0] as usize, shape[1] as usize);
    let masked: Vec<usize> = golden["masked"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.as_u64().unwrap() as usize)
        .collect();
    let detector = PixelDetector::new(shape, v("origin"), v("e_slow"), v("e_fast"))
        .unwrap()
        .with_mask(Image::from_fn(shape, |s, f| {
            masked.contains(&(s * shape.1 + f))
        }))
        .unwrap();
    let binnings = list("resolutions")
        .iter()
        .map(|&r| Binning::new(CoordKind::Linear, r, -r / 2.0).unwrap())
        .collect();
    let map = FrameMap::new(
        geometry,
        Projection::QSample,
        &detector,
        &Region::new(0..shape.0, 0..shape.1, (1, 1)),
        PixelFields::new(),
        binnings,
    )
    .unwrap();
    let frames = (0..golden["frames"].as_u64().unwrap() as usize)
        .map(|frame| {
            let mut deg = list("values_deg");
            deg[scanned] += frame as f64 * f(&golden["step_deg"]);
            let raw: Vec<f64> = (0..shape.0 * shape.1)
                .map(|i| image(frame, i) * weight(frame))
                .collect();
            let raw = Image::new(shape, raw).unwrap();
            map.map(&deg, wavelength, &raw, &FrameScalars::new())
                .unwrap()
        })
        .collect();
    Scan {
        golden,
        map,
        frames,
    }
}

impl Scan {
    /// One-pass filling of frames `range`, as binoculars' cube_add_space.
    fn grown(&self, range: Range<usize>) -> Cube {
        let mut cube = Cube::default();
        for frame in &self.frames[range] {
            self.map.add(&mut cube, frame).unwrap();
        }
        cube
    }

    /// `cube` has binoculars cube `key`'s axes and, bin by bin, its photons
    /// as Σcounts and its contributions as contributions and Σweight.
    fn check(&self, cube: &Cube, key: &str) {
        let golden = &self.golden["cubes"][key];
        let axes = golden["axes"].as_array().unwrap();
        assert_eq!(cube.axes().len(), axes.len(), "{key}");
        for (axis, (b, theirs)) in cube.axes().iter().zip(self.map.binnings().iter().zip(axes)) {
            let imin = theirs["imin"].as_i64().unwrap();
            let imax = theirs["imax"].as_i64().unwrap();
            assert_eq!(axis.name(), theirs["name"].as_str().unwrap(), "{key}");
            assert_eq!(b.step(), f(&theirs["resolution"]), "{key}");
            assert_eq!(axis.binning(), *b, "{key}");
            assert_eq!(
                (axis.first(), axis.size() as i64),
                (imin, imax - imin + 1),
                "{key} {}",
                axis.name()
            );
        }
        let mut photons = vec![0.0; cube.photons().len()];
        let mut contributions = vec![0; cube.contributions().len()];
        for bin in golden["bins"].as_array().unwrap() {
            let i = bin[0].as_u64().unwrap() as usize;
            photons[i] = f(&bin[1]);
            contributions[i] = bin[2].as_u64().unwrap();
        }
        assert_eq!(cube.contributions(), contributions, "{key}");
        assert_eq!(cube.photons(), photons, "{key}");
        let weight: Vec<f64> = contributions.iter().map(|&c| c as f64).collect();
        assert_eq!(cube.weight(), weight, "{key}");
    }
}

#[test]
fn every_pixel_is_far_from_a_bin_edge_so_floor_and_rint_agree() {
    let s = scan();
    let fine = f(&s.golden["fine"]);
    let coordinates = s.golden["coordinates"].as_array().unwrap();
    let mut closest = f64::INFINITY;
    for (frame, (m, rows)) in s.frames.iter().zip(coordinates).enumerate() {
        for (i, row) in rows.as_array().unwrap().iter().enumerate() {
            for (d, (plane, b)) in m.points().planes().iter().zip(s.map.binnings()).enumerate() {
                let ours = plane.data()[i];
                let theirs = row[d].as_i64().unwrap() as f64 * fine;
                assert!((ours - theirs).abs() < Q_TOL, "{frame} {i} {d}");
                let k = b.bin(ours).unwrap();
                closest = closest.min((ours - b.edge(k)).min(b.edge(k + 1) - ours));
                // binoculars' rule, rint(v/res), on binoculars' coordinate
                let rint = (theirs / b.step()).round_ties_even();
                assert_eq!(k, rint as i64, "{frame} {i} {d}");
            }
        }
    }
    assert!(closest > MARGIN, "{closest}");
}

#[test]
fn add_growing_matches_cube_add_space() {
    let s = scan();
    s.check(&s.grown(0..s.frames.len()), "one_pass");
}

#[test]
fn bounds_then_add_match_cube_new() {
    let s = scan();
    let mut bounds = Bounds::new(s.map.binnings().to_vec());
    for m in &s.frames {
        bounds.include(m.points(), m.pixels()).unwrap();
    }
    let mut cube = bounds.cube().unwrap();
    for m in &s.frames {
        assert_eq!(cube.add(m.points(), m.pixels()).unwrap(), 0);
    }
    s.check(&cube, "two_pass");
}

#[test]
fn merge_of_half_scans_matches_cube_new_merge() {
    let s = scan();
    let half = s.frames.len() / 2;
    let (a, b) = (s.grown(0..half), s.grown(half..s.frames.len()));
    s.check(&a, "first_half");
    s.check(&b, "second_half");
    s.check(&a.merge(b).unwrap(), "merged");
}

/// binoculars' HDF5 I/O on the merged cube (golden/bin_7be4f79_hdf5.c):
/// binoculars wrote bin_7be4f79_cube.h5 with hkl_binoculars_cube_save_hdf5,
/// h5repack gzipped it into bin_7be4f79_cube_gzip.h5, hklrs wrote
/// hklrs_7be4f79_cube.h5, and bin_7be4f79_hdf5.json is
/// hkl_binoculars_cube_new_from_file's verdict on all three.
/// `HKLRS_BLESS=1` rewrites hklrs' file (bin_7be4f79_hdf5.sh).
#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::{common::load, scan};
    use hklrs::grid::{Cube, hdf5};
    use rust_hdf5::H5File;
    use rust_hdf5::format::messages::filter::FILTER_DEFLATE;

    fn golden(file: &str) -> String {
        format!("{}/../golden/{file}", env!("CARGO_MANIFEST_DIR"))
    }

    fn merged() -> Cube {
        let s = scan();
        let half = s.frames.len() / 2;
        s.grown(0..half)
            .merge(s.grown(half..s.frames.len()))
            .unwrap()
    }

    #[test]
    fn binoculars_file_reads_as_the_merged_cube() {
        let (cube, config) = hdf5::read(golden("bin_7be4f79_cube.h5")).unwrap();
        // binoculars' f32 counts widen exactly; its axes read as anchored
        // at −step/2, the binnings of scan(); it has no weight, so Σweight
        // reads as its contributions
        assert_eq!(cube, merged());
        let verdict = load("bin_7be4f79_hdf5.json");
        assert_eq!(config, verdict["config"].as_str().unwrap());
    }

    /// binoculars loaded `file` as the merged cube.
    fn binoculars_loaded(file: &str) {
        let verdict = &load("bin_7be4f79_hdf5.json")["loads"][file];
        for key in ["loaded", "axes", "offset0", "photons", "contributions"] {
            assert_eq!(verdict[key], true, "{file}: {key}");
        }
        assert_eq!(verdict["cube_cmp"], 0, "{file}");
    }

    #[test]
    fn gzipped_binoculars_file_reads_as_the_merged_cube() {
        let file = golden("bin_7be4f79_cube_gzip.h5");
        let f = H5File::open(&file).unwrap();
        for name in [
            "axes/qx",
            "axes/qy",
            "axes/qz",
            "config",
            "counts",
            "contributions",
        ] {
            let ds = f.dataset(&format!("binoculars/{name}")).unwrap();
            let ids: Vec<u16> = ds.filters().unwrap().iter().map(|f| f.id).collect();
            assert_eq!(ids, [FILTER_DEFLATE], "{name}");
        }
        binoculars_loaded("bin_7be4f79_cube_gzip.h5");
        let config = load("bin_7be4f79_hdf5.json")["config"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(hdf5::read(&file).unwrap(), (merged(), config));
    }

    #[test]
    fn hklrs_writes_the_merged_cube() {
        let theirs = golden("bin_7be4f79_cube.h5");
        let (_, config) = hdf5::read(&theirs).unwrap();
        let ours = golden("hklrs_7be4f79_cube.h5");
        if std::env::var_os("HKLRS_BLESS").is_some() {
            hdf5::write(&ours, &merged(), &config).unwrap();
            return;
        }
        // the committed file is what binoculars loaded: this writer must
        // still produce it byte for byte
        let fresh = format!("{}/hklrs_7be4f79_cube.h5", env!("CARGO_TARGET_TMPDIR"));
        hdf5::write(&fresh, &merged(), &config).unwrap();
        assert!(std::fs::read(&fresh).unwrap() == std::fs::read(&ours).unwrap());

        binoculars_loaded("hklrs_7be4f79_cube.h5");

        // binoculars' tools read the axis rows: index, min, max, resolution,
        // imin, imax, the same doubles binoculars writes
        let (a, b) = (H5File::open(&ours).unwrap(), H5File::open(&theirs).unwrap());
        for name in ["qx", "qy", "qz"] {
            let row = |f: &H5File| {
                f.dataset(&format!("binoculars/axes/{name}"))
                    .unwrap()
                    .read_raw::<f64>()
                    .unwrap()
            };
            assert_eq!(row(&a), row(&b), "{name}");
        }
        // Σweight is a dataset binoculars' file lacks and loads past
        assert!(b.dataset("binoculars/weight").is_err());
        let weight = a.dataset("binoculars/weight").unwrap();
        assert_eq!(weight.read_raw::<f64>().unwrap(), merged().weight());
        assert_eq!(hdf5::read(&ours).unwrap(), (merged(), config));
    }
}
