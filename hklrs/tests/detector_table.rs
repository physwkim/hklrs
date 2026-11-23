//! The detector table against binoculars-ng 7be4f79 (golden/bin_7be4f79_detectors.c)
//! and pyFAI 2026.9.0 (golden/pyfai_imxpad.py): shape, pixel positions and
//! mask of every detector. Equal to binoculars except the Pilatus CdTe 2M
//! centre strips (BIN-1), the imXpad chip-end pixels (BIN-12) and the
//! single-precision Cirpad placement (BIN-11).

pub mod common;

use common::{f, load};
use hklrs::nalgebra::{Point3, Vector3};
use hklrs::{Detector, Error, Layout};
use serde_json::Value;

const POS_TOL: f64 = 1e-14;
/// BIN-11: binoculars stores the Cirpad translations and angles as float and
/// places the modules in float. The arithmetic gives up to 6.2e-8 m, the
/// stored values the rest: 1.106e-7 m at most over the golden samples.
const CIRPAD_TOL: f64 = 1.11e-7;

fn doc() -> Value {
    load("bin_7be4f79_detectors.json")
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Doubles a golden writes as decimal strings, read to the bit.
fn exact(v: &Value) -> Vec<f64> {
    let parse = |x: &Value| x.as_str().unwrap().parse().unwrap();
    v.as_array().unwrap().iter().map(parse).collect()
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

/// BIN-12: binoculars' minus pyFAI's centre of pixel `i` of `n` in chips of
/// `chip`, in pitches along the axis: binoculars puts the last pixel of a
/// chip at its far edge, and makes the detector's last pixel wide too.
fn bin_12_offset(i: usize, n: usize, chip: usize) -> f64 {
    if i == n - 1 {
        2.0
    } else if i % chip == chip - 1 {
        1.25
    } else {
        0.0
    }
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
        if det == Detector::Cirpad {
            continue;
        }
        let row = det.row();
        let (h, w) = row.shape;
        let chip = match row.layout {
            Layout::Chips { chip } => Some(chip),
            _ => None,
        };
        // binoculars is farther along -y and along the rows' z than hklrs
        let sz = if row.flip_z { -1.0 } else { 1.0 };
        let dy = |i| chip.map_or(0.0, |c| bin_12_offset(i, w, c.1) * row.pitch);
        let dz = |j| chip.map_or(0.0, |c| bin_12_offset(j, h, c.0) * row.pitch);
        let (y, z) = (floats(&d["y"]), floats(&d["z"]));
        let pd = det.pixel_detector();
        for (j, &zj) in z.iter().enumerate() {
            for (i, &yi) in y.iter().enumerate() {
                let p = pd.pixel(j, i).unwrap();
                let err =
                    p.x.abs()
                        .max((p.y - (yi + dy(i))).abs())
                        .max((p.z - (zj - sz * dz(j))).abs());
                assert!(
                    err < POS_TOL,
                    "{} pixel ({j}, {i}) off by {err}",
                    det.name()
                );
            }
        }
    }
}

/// The imXpad pixels are pyFAI's edges and centres to the bit.
#[test]
fn imxpad_pixels_are_pyfai() {
    for g in load("pyfai_imxpad.json")["detectors"].as_array().unwrap() {
        let det = Detector::ALL
            .into_iter()
            .find(|d| d.name() == g["name"])
            .unwrap();
        let row = det.row();
        assert_eq!(row.shape, (u(&g["shape"][0]), u(&g["shape"][1])));
        assert_eq!(
            row.layout,
            Layout::Chips {
                chip: (u(&g["module"][0]), u(&g["module"][1]))
            }
        );
        assert_eq!([row.pitch; 2], [f(&g["pixel"][0]), f(&g["pixel"][1])]);
        let pd = det.pixel_detector();
        let [m] = pd.modules() else {
            panic!("{}: one module", det.name())
        };
        let (c1, c2) = (exact(&g["centre1"]), exact(&g["centre2"]));
        for (axis, edges, centres) in [
            (m.slow(), exact(&g["edges1"]), &c1),
            (m.fast(), exact(&g["edges2"]), &c2),
        ] {
            assert_eq!(axis.pixel_count(), centres.len());
            for (i, &c) in centres.iter().enumerate() {
                assert_eq!(axis.centre(i), c, "{} pixel {i}", det.name());
                assert_eq!(axis.width(i), edges[i + 1] - edges[i]);
            }
        }
        for (j, &cj) in c1.iter().enumerate() {
            for (i, &ci) in c2.iter().enumerate() {
                assert_eq!(pd.pixel(j, i).unwrap(), Point3::new(0.0, -ci, cj));
            }
        }
    }
}

/// BIN-11: binoculars places the Cirpad modules in single precision; each
/// module is an S70 whose chip-end pixels are also BIN-12's.
#[test]
fn bin_11_cirpad_placed_in_single_precision() {
    let d = &doc()["detectors"][Detector::Cirpad as usize];
    assert_eq!(detector(d), Detector::Cirpad);
    let pd = Detector::Cirpad.pixel_detector();
    let pitch = Detector::Cirpad.row().pitch;
    let samples = d["samples"].as_array().unwrap();
    assert!(!samples.is_empty());
    let mut worst = 0.0f64;
    for s in samples {
        let (j, i) = (u(&s[0]), u(&s[1]));
        let m = pd
            .modules()
            .iter()
            .find(|m| (m.at().0..m.at().0 + m.shape().0).contains(&j))
            .unwrap();
        let (h, w) = m.shape();
        let (slow, fast) = (m.slow().dir().unwrap(), m.fast().dir().unwrap());
        let bin_12 = slow.into_inner() * (bin_12_offset(j - m.at().0, h, 120) * pitch)
            + fast.into_inner() * (bin_12_offset(i, w, 80) * pitch);
        let want = Vector3::new(f(&s[2]), f(&s[3]), f(&s[4]));
        let err = (want - pd.pixel(j, i).unwrap().coords - bin_12).abs().max();
        worst = worst.max(err);
    }
    assert!(
        worst > 1e-9 && worst <= CIRPAD_TOL,
        "largest Cirpad error {worst}"
    );
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
        assert_eq!(got.shape(), (h, w), "{}", det.name());
        let got = got.data();
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
    assert_eq!(mask.shape().1, w);
    let masked = |j: usize, c: usize| mask.data()[j * w + c];
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
fn every_row_is_a_pixel_detector_with_its_mask() {
    for det in Detector::ALL {
        let pd = det.pixel_detector();
        assert_eq!(pd.shape(), det.shape(), "{}", det.name());
        let mask = det.mask();
        let mask = mask.data();
        assert_eq!(
            pd.mask(),
            mask.contains(&true).then_some(mask),
            "{}",
            det.name()
        );
        let modules = match det {
            Detector::MerlinMedipix3RxQuad512 => 4,
            Detector::Cirpad => 20,
            _ => 1,
        };
        assert_eq!(pd.modules().len(), modules, "{}", det.name());
    }
}
