//! The detector table against binoculars-ng 7be4f79 (golden/bin_7be4f79_detectors.c):
//! shape, pixel positions and mask of every detector. Equal everywhere except
//! the Pilatus CdTe 2M centre strips, where binoculars is wrong (BIN-1).

pub mod common;

use common::{f, load};
use hklrs::{Detector, Error};
use serde_json::Value;

const POS_TOL: f64 = 1e-14;
/// Cirpad positions go through single-precision cglm matrices in binoculars.
const CIRPAD_TOL: f64 = 1e-6;

fn doc() -> Value {
    load("bin_7be4f79_detectors.json")
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn golden_mask(d: &Value, len: usize) -> Vec<bool> {
    let mut mask = vec![false; len];
    for run in d["mask_runs"].as_array().unwrap() {
        let (start, n) = (u(&run[0]), u(&run[1]));
        mask[start..start + n].fill(true);
    }
    mask
}

fn detector(d: &Value) -> Detector {
    Detector::try_from(d["id"].as_u64().unwrap() as u32).unwrap()
}

fn is_regular(det: Detector) -> bool {
    det.pixel_detector().is_ok()
}

#[test]
fn name_and_shape() {
    for d in doc()["detectors"].as_array().unwrap() {
        let det = detector(d);
        assert_eq!(det.name(), d["name"].as_str().unwrap());
        assert_eq!(
            det.shape(),
            (u(&d["height"]), u(&d["width"])),
            "{}",
            det.name()
        );
    }
}

#[test]
fn pixel_positions() {
    for d in doc()["detectors"].as_array().unwrap() {
        let det = detector(d);
        if let Some(samples) = d.get("samples") {
            for s in samples.as_array().unwrap() {
                let p = det.pixel(u(&s[0]), u(&s[1])).unwrap();
                for (k, got) in p.iter().enumerate() {
                    let err = (got - f(&s[2 + k])).abs();
                    assert!(
                        err < CIRPAD_TOL,
                        "{} {s}: axis {k} off by {err}",
                        det.name()
                    );
                }
            }
            continue;
        }
        let y: Vec<f64> = d["y"].as_array().unwrap().iter().map(f).collect();
        let z: Vec<f64> = d["z"].as_array().unwrap().iter().map(f).collect();
        let grid = det.pixel_detector().ok();
        for (j, &zj) in z.iter().enumerate() {
            for (i, &yi) in y.iter().enumerate() {
                let direct = det.pixel(j, i).unwrap();
                let via_grid = grid.as_ref().map(|g| g.pixel(j, i).unwrap());
                for p in std::iter::once(direct).chain(via_grid) {
                    let err = p.x.abs().max((p.y - yi).abs()).max((p.z - zj).abs());
                    assert!(
                        err < POS_TOL,
                        "{} pixel ({j}, {i}) off by {err}",
                        det.name()
                    );
                }
            }
        }
    }
}

/// Columns 729..732 and 1216..1219 are what binoculars masks as the strips of
/// modules 1 and 2; the right ones are 736..739 and 1230..1233.
#[test]
fn masks() {
    for d in doc()["detectors"].as_array().unwrap() {
        let det = detector(d);
        let (h, w) = det.shape();
        let want = golden_mask(d, h * w);
        let got = det.mask();
        let differ: Vec<(usize, usize)> = (0..h * w)
            .filter(|&k| got[k] != want[k])
            .map(|k| (k / w, k % w))
            .collect();
        if det != Detector::PilatusCdte2M {
            assert!(
                differ.is_empty(),
                "{}: {} pixels differ",
                det.name(),
                differ.len()
            );
            continue;
        }
        let strip_cols = [
            729, 730, 731, 736, 737, 738, 1216, 1217, 1218, 1230, 1231, 1232,
        ];
        let gap_row = |j: usize| j % 212 >= 195;
        let expected: Vec<(usize, usize)> = (0..h)
            .filter(|&j| !gap_row(j))
            .flat_map(|j| strip_cols.iter().map(move |&c| (j, c)))
            .collect();
        assert_eq!(
            differ, expected,
            "Pilatus differs from binoculars outside BIN-1"
        );
    }
}

/// BIN-1: binoculars steps the centre strip by the module width (487), not the
/// module pitch (487 + 7 gap), so modules 1 and 2 lose the wrong columns.
#[test]
fn bin_1_pilatus_centre_strip_on_every_module() {
    let mask = Detector::PilatusCdte2M.mask();
    let w = 1475;
    let masked = |j: usize, c: usize| mask[j * w + c];
    for k in 0..3 {
        let centre = (487 + 7) * k + 243;
        for c in [centre - 1, centre, centre + 1] {
            assert!(masked(5, c), "module {k}: centre column {c} not masked");
        }
        for c in [centre - 2, centre + 2] {
            assert!(!masked(5, c), "module {k}: column {c} masked");
        }
    }
    for c in [729, 730, 731, 1216, 1217, 1218] {
        assert!(!masked(5, c), "binoculars' wrong column {c} is masked");
    }
}

/// BIN-2: binoculars reads one entry past its table for id == count; here
/// every id outside the table is an error and none falls back to detector 0.
#[test]
fn bin_2_unknown_id_is_an_error() {
    let n = Detector::ALL.len() as u32;
    for id in [n, n + 1, u32::MAX] {
        assert_eq!(
            Detector::try_from(id),
            Err(Error::UnknownDetector {
                id,
                count: n as usize
            })
        );
    }
    for (id, det) in Detector::ALL.iter().enumerate() {
        assert_eq!(Detector::try_from(id as u32), Ok(*det));
        assert_eq!(*det as usize, id);
    }
}

#[test]
fn irregular_detectors_have_no_pixel_detector() {
    let irregular: Vec<_> = Detector::ALL
        .into_iter()
        .filter(|d| !is_regular(*d))
        .collect();
    assert_eq!(
        irregular,
        [
            Detector::ImxpadS140,
            Detector::ImxpadS70,
            Detector::MerlinMedipix3RxQuad512,
            Detector::Cirpad
        ]
    );
    assert_eq!(
        Detector::Cirpad.pixel_detector(),
        Err(Error::DetectorNotRegular { name: "Cirpad" })
    );
}
