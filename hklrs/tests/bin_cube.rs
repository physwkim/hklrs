//! The hklrs Cube against binoculars-ng 7be4f79's own accumulation of a
//! synthetic 20-frame K6C komega scan in qx, qy, qz
//! (golden/bin_7be4f79_cube.c). Bins are `Binning::new(res, -res/2)`, the
//! bins binoculars centres on multiples of res. Images are integers and
//! weights powers of two, so photons compare exactly.
//!
//! hklrs has no public per-frame weight or caller mask (`Weight::new` is
//! crate-private): each image is multiplied by its frame's weight, as
//! binoculars does, and the masked pixels are left out of the points.

pub mod common;
use common::{f, load};

use hklrs::grid::{Binning, Bounds, Cube, Projected};
use hklrs::nalgebra::Vector3;
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{PixelDetector, Wavelength};
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
    binnings: Vec<Binning>,
    /// Every pixel of each frame, masked or not.
    points: Vec<Projected>,
    /// Each frame's unmasked pixels and their weighted counts.
    frames: Vec<(Projected, Vec<f64>)>,
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
    let detector = PixelDetector::new(
        (shape[0] as usize, shape[1] as usize),
        v("origin"),
        v("e_slow"),
        v("e_fast"),
    )
    .unwrap();
    let masked: Vec<usize> = golden["masked"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.as_u64().unwrap() as usize)
        .collect();
    let binnings = list("resolutions")
        .iter()
        .map(|&r| Binning::new(r, -r / 2.0).unwrap())
        .collect();
    let mut points = Vec::new();
    let mut frames = Vec::new();
    for frame in 0..golden["frames"].as_u64().unwrap() as usize {
        let mut deg = list("values_deg");
        deg[scanned] += frame as f64 * f(&golden["step_deg"]);
        let values = geometry.values_from_degrees(&deg).unwrap();
        let p = Projection::QSample
            .project(&geometry, &values, wavelength, &detector)
            .unwrap();
        let lit: Vec<usize> = (0..p.pixels()).filter(|i| !masked.contains(i)).collect();
        let planes = p
            .planes()
            .iter()
            .map(|plane| lit.iter().map(|&i| plane[i]).collect())
            .collect();
        let counts = lit
            .iter()
            .map(|&i| image(frame, i) * weight(frame))
            .collect();
        frames.push((Projected::new(p.names(), planes).unwrap(), counts));
        points.push(p);
    }
    Scan {
        golden,
        binnings,
        points,
        frames,
    }
}

impl Scan {
    /// One-pass filling of `frames`, as binoculars' cube_add_space.
    fn grown(&self, frames: &[(Projected, Vec<f64>)]) -> Cube {
        let mut cube = Cube::default();
        for (p, counts) in frames {
            cube.add_growing(&self.binnings, p, counts, None).unwrap();
        }
        cube
    }

    /// `cube` has binoculars cube `key`'s axes and, bin by bin, its photons
    /// and contributions.
    fn check(&self, cube: &Cube, key: &str) {
        let golden = &self.golden["cubes"][key];
        let axes = golden["axes"].as_array().unwrap();
        assert_eq!(cube.axes().len(), axes.len(), "{key}");
        for (axis, (b, theirs)) in cube.axes().iter().zip(self.binnings.iter().zip(axes)) {
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
    }
}

#[test]
fn every_pixel_is_far_from_a_bin_edge_so_floor_and_rint_agree() {
    let s = scan();
    let fine = f(&s.golden["fine"]);
    let coordinates = s.golden["coordinates"].as_array().unwrap();
    let mut closest = f64::INFINITY;
    for (frame, (p, rows)) in s.points.iter().zip(coordinates).enumerate() {
        for (i, row) in rows.as_array().unwrap().iter().enumerate() {
            for (d, (plane, b)) in p.planes().iter().zip(&s.binnings).enumerate() {
                let ours = plane[i];
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
    s.check(&s.grown(&s.frames), "one_pass");
}

#[test]
fn bounds_then_add_match_cube_new() {
    let s = scan();
    let mut bounds = Bounds::new(s.binnings.clone());
    for (p, _) in &s.frames {
        bounds.include(p, None).unwrap();
    }
    let mut cube = bounds.cube().unwrap();
    for (p, counts) in &s.frames {
        assert_eq!(cube.add(p, counts, None).unwrap(), 0);
    }
    s.check(&cube, "two_pass");
}

#[test]
fn merge_of_half_scans_matches_cube_new_merge() {
    let s = scan();
    let (first, second) = s.frames.split_at(s.frames.len() / 2);
    let (a, b) = (s.grown(first), s.grown(second));
    s.check(&a, "first_half");
    s.check(&b, "second_half");
    s.check(&a.merge(b).unwrap(), "merged");
}

/// binoculars' HDF5 I/O on the merged cube (golden/bin_7be4f79_hdf5.c):
/// binoculars wrote bin_7be4f79_cube.h5 with hkl_binoculars_cube_save_hdf5,
/// hklrs wrote hklrs_7be4f79_cube.h5, and bin_7be4f79_hdf5.json is
/// hkl_binoculars_cube_new_from_file's verdict on both. `HKLRS_BLESS=1`
/// rewrites hklrs' file (bin_7be4f79_hdf5.sh).
#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::{common::load, scan};
    use hklrs::grid::{Cube, hdf5};
    use rust_hdf5::H5File;

    fn golden(file: &str) -> String {
        format!("{}/../golden/{file}", env!("CARGO_MANIFEST_DIR"))
    }

    fn merged() -> Cube {
        let s = scan();
        let (first, second) = s.frames.split_at(s.frames.len() / 2);
        s.grown(first).merge(s.grown(second)).unwrap()
    }

    #[test]
    fn binoculars_file_reads_as_the_merged_cube() {
        let (cube, config) = hdf5::read(golden("bin_7be4f79_cube.h5")).unwrap();
        // binoculars' f32 counts widen exactly; its axes read as anchored
        // at −step/2, the binnings of scan()
        assert_eq!(cube, merged());
        let verdict = load("bin_7be4f79_hdf5.json");
        assert_eq!(config, verdict["config"].as_str().unwrap());
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

        let verdict = &load("bin_7be4f79_hdf5.json")["loads"]["hklrs_7be4f79_cube.h5"];
        for key in ["loaded", "axes", "offset0", "photons", "contributions"] {
            assert_eq!(verdict[key], true, "{key}");
        }
        assert_eq!(verdict["cube_cmp"], 0);

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
        assert_eq!(hdf5::read(&ours).unwrap(), (merged(), config));
    }
}
