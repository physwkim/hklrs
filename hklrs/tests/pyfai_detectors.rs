//! pyFAI 2026.9.0's named detector classes from a PONI file against the
//! instances `Detector.factory` builds (golden/pyfai_detectors.py), default
//! or binned by the `binning` key: `calc_pos_zyx` at pixels of each
//! supported class, `Detector.mask`, and a typed error for every other
//! class and where pyFAI's binned mask raises. An ImXPad image is flipped once, as
//! pyFAI's whole-image `calc_cartesian_positions` flips it; pyFAI's
//! by-index path flips it twice (upstream defect FAI-2).

pub mod common;

use common::{f, load};
use hklrs::PixelDetector;
use hklrs::nalgebra::{Unit, Vector3};
use hklrs::poni::{LabAxes, PoniError, PoniFile};
use serde_json::Value;

/// Metres; pixel positions are about 0.3 m from the sample.
const TOL: f64 = 1e-14;

fn labs() -> [(&'static str, LabAxes, [Vector3<f64>; 3]); 2] {
    let beam = Vector3::new(0.3, -0.5, 0.81).normalize();
    let up = beam.cross(&Vector3::new(0.2, 0.7, -0.4)).normalize();
    [("hkl", Vector3::x(), Vector3::z()), ("oblique", beam, up)].map(|(name, beam, up)| {
        let lab = LabAxes::new(Unit::new_unchecked(beam), Unit::new_unchecked(up)).unwrap();
        // lab directions of pyFAI axes 1, 2, 3
        (name, lab, [up, up.cross(&beam), beam])
    })
}

fn class<'a>(g: &'a Value, name: &str) -> &'a Value {
    let classes = g["classes"].as_array().unwrap();
    classes.iter().find(|c| c["class"] == name).unwrap()
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect()
}

fn pair(v: &Value) -> (usize, usize) {
    let n = |k: usize| v[k].as_u64().unwrap() as usize;
    (n(0), n(1))
}

/// The class's instance binned `b` as the golden gives it.
fn binned<'a>(c: &'a Value, b: &Value) -> &'a Value {
    let binned = c["binned"].as_array().unwrap();
    binned.iter().find(|e| e["binning"] == *b).unwrap()
}

/// count, sum and sum of squares (mod 2^64) of the masked row-major indices.
fn fingerprint(det: &PixelDetector) -> Vec<u64> {
    let mask = det.mask().unwrap_or(&[]);
    let mut out = [0u64; 3];
    for (k, _) in mask.iter().enumerate().filter(|&(_, &m)| m) {
        let k = k as u64;
        out = [
            out[0] + 1,
            out[1].wrapping_add(k),
            out[2].wrapping_add(k.wrapping_mul(k)),
        ];
    }
    out.to_vec()
}

fn file(name: &str, config: &str) -> PoniFile {
    PoniFile::parse(&format!(
        "poni_version: 2.1\nDetector: {name}\nDetector_config: {config}\n\
         Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\nRot3: 0.3\n\
         Wavelength: 1e-10\n"
    ))
    .unwrap()
}

fn lab() -> LabAxes {
    labs()[0].1
}

fn unsupported(name: &str, config: &str) -> Result<&'static str, PoniError> {
    match file(name, config).detector(&lab()) {
        Err(PoniError::UnsupportedDetector { class, reason }) => {
            assert_eq!(class, name);
            Ok(reason)
        }
        Err(e) => Err(e),
        Ok(_) => panic!("{name} {config}: built"),
    }
}

#[test]
fn named_classes_match_pyfai_calc_pos_zyx() {
    let g = load("pyfai_detectors.json");
    let mut worst = 0.0f64;
    let mut built = std::collections::BTreeSet::new();
    for case in g["cases"].as_array().unwrap() {
        let name = case["class"].as_str().unwrap();
        let c = class(&g, name);
        let pf = PoniFile::parse(case["poni"].as_str().unwrap()).unwrap();
        assert_eq!(pf.poni(), Err(PoniError::NamedDetector(name.to_string())));
        let (shape, mask) = match &case["binning"] {
            Value::Null => (pair(&c["shape"]), u64s(&c["fingerprint"])),
            b => {
                let e = binned(c, b);
                (
                    pair(&e["shape"]),
                    e["mask"]
                        .as_array()
                        .map_or(vec![0; 3], |_| u64s(&e["mask"])),
                )
            }
        };
        for (lab_name, lab, [a1, a2, a3]) in labs() {
            let det = pf.detector(&lab).unwrap();
            assert_eq!(det.shape(), shape, "{name}");
            assert_eq!(fingerprint(&det), mask, "{name} mask");
            let pixels = case["pixels"].as_array().unwrap();
            for (px, zyx) in pixels.iter().zip(case["zyx"].as_array().unwrap()) {
                let (t3, t1, t2) = (f(&zyx[0]), f(&zyx[1]), f(&zyx[2]));
                let want = a1 * t1 + a2 * t2 + a3 * t3;
                let ix = |k: usize| px[k].as_u64().unwrap() as usize;
                let d = (det.pixel(ix(0), ix(1)).unwrap().coords - want).norm();
                assert!(
                    d <= TOL,
                    "{name}, {lab_name} lab, orientation {}, binning {}, pixel {px}: {d} m",
                    case["orientation"],
                    case["binning"]
                );
                worst = worst.max(d);
            }
        }
        built.insert(name);
    }
    eprintln!("named classes: worst pixel distance to pyFAI {worst} m");
    // every class a PONI file can name and hklrs builds has a case
    let supported: std::collections::BTreeSet<_> = g["classes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["layout"] == "uniform" || c["layout"] == "edges")
        .map(|c| c["class"].as_str().unwrap())
        .collect();
    assert_eq!(built, supported);
}

#[test]
fn other_classes_are_unsupported_by_name() {
    let g = load("pyfai_detectors.json");
    let (mut supported, mut refused) = (0, 0);
    for c in g["classes"].as_array().unwrap() {
        let name = c["class"].as_str().unwrap();
        match c["layout"].as_str().unwrap() {
            "generic" => continue,
            "uniform" | "edges" => supported += 1,
            layout => {
                assert_eq!(unsupported(name, "{}").unwrap(), layout, "{name}");
                refused += 1;
            }
        }
        assert_eq!(
            file(name, "{}").poni(),
            Err(PoniError::NamedDetector(name.to_string()))
        );
    }
    eprintln!("pyFAI classes: {supported} named supported, {refused} unsupported");
    assert_eq!(
        supported + refused + 1,
        g["classes"].as_array().unwrap().len()
    );
}

#[test]
fn names_resolve_through_the_factory_s_aliases() {
    let g = load("pyfai_detectors.json");
    for (name, class) in g["probes"].as_object().unwrap() {
        let pf = file(name, "{}");
        match class.as_str() {
            None => {
                assert_eq!(pf.poni(), Err(PoniError::UnknownDetector(name.clone())));
                assert!(matches!(
                    pf.detector(&lab()),
                    Err(PoniError::UnknownDetector(n)) if &n == name
                ));
            }
            Some("Detector") => {
                assert!(pf.poni().is_err_and(|e| e == PoniError::Missing("pixel1")))
            }
            Some(class) => {
                assert_eq!(pf.poni(), Err(PoniError::NamedDetector(name.clone())));
                if let Err(PoniError::UnsupportedDetector { class: c, .. }) = pf.detector(&lab()) {
                    assert_eq!(c, class, "{name}");
                }
            }
        }
    }
}

#[test]
fn config_other_than_the_default_instance_is_refused() {
    let lab = lab();
    let refused = |name: &str, config: &str| match file(name, config).detector(&lab) {
        Err(PoniError::DetectorConfig { class, key }) => {
            assert_eq!(class, name);
            key
        }
        other => panic!("{name} {config}: {other:?}"),
    };
    assert_eq!(refused("Pilatus1M", r#"{"pixel1": 1e-4}"#), "pixel1");
    assert_eq!(
        refused("Pilatus1M", r#"{"max_shape": [1043, 980]}"#),
        "max_shape"
    );
    // pyFAI's binning setter is not theirs: Rayonix bins by a pitch table,
    // ImXPad's edges span max_shape whatever the binning
    assert_eq!(refused("Rayonix133", r#"{"binning": [1, 1]}"#), "binning");
    assert_eq!(refused("ImXPadS10", r#"{"binning": [2, 2]}"#), "binning");
    assert_eq!(
        refused("Pilatus1M", r#"{"module_size": [97, 487]}"#),
        "module_size"
    );
    assert_eq!(
        refused("Pilatus1M", r#"{"x_offset_file": "x.edf"}"#),
        "x_offset_file"
    );
    // the default's own values, a zero pitch, a sensor and entries the class
    // does not take build the default instance
    for config in [
        r#"{"pixel1": 0.000172, "pixel2": 0, "max_shape": [1043, 981]}"#,
        r#"{"binning": [1, 1], "module_size": [195, 487], "orientation": 3}"#,
        r#"{"sensor": {"material": "Si", "thickness": 4.5e-4}, "splinefile": "a.spline"}"#,
        r#"{"x_offset_file": null, "unknown": 1}"#,
    ] {
        let det = file("Pilatus1M", config).detector(&lab).unwrap();
        assert_eq!(det.shape(), (1043, 981), "{config}");
    }
    // Perkin's default instance is binned 2x2
    for (b, shape) in [("[2, 2]", (2048, 2048)), ("[1, 1]", (4096, 4096))] {
        let config = format!(r#"{{"binning": {b}}}"#);
        let det = file("Perkin", &config).detector(&lab).unwrap();
        assert_eq!(det.shape(), shape, "{b}");
    }
    // 1043 x 981 does not bin 2x2, so pyFAI's Detector.mask raises
    assert_eq!(
        file("Pilatus1M", r#"{"binning": [2, 2]}"#).detector(&lab),
        Err(PoniError::MaskBinning {
            class: "Pilatus1M",
            binning: (2, 2),
            max_shape: (1043, 981)
        })
    );
    assert_eq!(
        file("Rayonix133", r#"{"binning": [2, 2]}"#)
            .detector(&lab)
            .unwrap()
            .shape(),
        (2048, 2048)
    );
    assert_eq!(
        file("Mythen", r#"{"splinefile": "a.spline"}"#).detector(&lab),
        Err(PoniError::Unsupported("spline file"))
    );
}

/// Every binning of the golden on every class pyFAI bins by its setter's
/// rule: the binned shape and `Detector.mask`, a typed error where pyFAI's
/// mask raises (the binning does not divide `max_shape`) or the binned
/// image is empty; the binning key on another class is refused.
#[test]
fn binned_instances_are_pyfai_s() {
    let g = load("pyfai_detectors.json");
    let lab = lab();
    let (mut built, mut raises, mut empty, mut refused) = (0, 0, 0, 0);
    for c in g["classes"].as_array().unwrap() {
        let name = c["class"].as_str().unwrap();
        if c["layout"] != "uniform" && c["layout"] != "edges" {
            continue;
        }
        let default = pair(&c["binning"]);
        let instances = c["binned"].as_array().map_or(&[][..], |v| v.as_slice());
        for e in instances {
            let b = pair(&e["binning"]);
            let config = format!(r#"{{"binning": {}}}"#, e["binning"]);
            let got = file(name, &config).detector(&lab);
            if c["binnable"] == false {
                if b == default {
                    assert_eq!(got.unwrap().shape(), pair(&c["shape"]), "{name}");
                } else {
                    assert!(
                        matches!(&got, Err(PoniError::DetectorConfig { key, .. }) if key == "binning"),
                        "{name} {b:?}: {got:?}"
                    );
                    refused += 1;
                }
                continue;
            }
            let shape = pair(&e["shape"]);
            if shape.0 == 0 || shape.1 == 0 {
                assert!(
                    matches!(
                        got,
                        Err(PoniError::Core(hklrs::Error::DetectorShape { .. }))
                    ),
                    "{name} {b:?}: {got:?}"
                );
                empty += 1;
            } else if e["mask_error"].is_string() {
                let Err(PoniError::MaskBinning {
                    class,
                    binning,
                    max_shape,
                }) = got
                else {
                    panic!("{name} {b:?}: {got:?}");
                };
                let want = (name, (b.0 as u64, b.1 as u64), pair(&c["max_shape"]));
                assert_eq!((class, binning, max_shape), want);
                raises += 1;
            } else {
                let det = got.unwrap();
                assert_eq!(det.shape(), shape, "{name} {b:?}");
                let mask = e["mask"]
                    .as_array()
                    .map_or(vec![0; 3], |_| u64s(&e["mask"]));
                assert_eq!(fingerprint(&det), mask, "{name} {b:?}");
                built += 1;
            }
        }
    }
    eprintln!("binned: {built} built, {raises} mask raises, {empty} empty, {refused} refused");
    assert!(built > 0 && raises > 0 && empty > 0 && refused > 0);
}

/// FAI-2: `ImXPadS10.calc_cartesian_positions(d1, d2)` reverses the
/// indices and then interpolates the edges `calc_pixels_edges` has already
/// reversed, so pyFAI's by-index positions are those of orientation 3 in
/// every orientation and part from the whole image's in 1, 2 and 4.
#[test]
fn imxpad_by_index_departs_from_the_whole_image_off_orientation_3() {
    let g = load("pyfai_detectors.json");
    let mut seen = 0;
    for case in g["cases"].as_array().unwrap() {
        let Some(by_index) = case["zyx_by_index"].as_array() else {
            continue;
        };
        let name = case["class"].as_str().unwrap();
        let o = case["orientation"].as_u64().unwrap();
        let apart = case["zyx"]
            .as_array()
            .unwrap()
            .iter()
            .zip(by_index)
            .map(|(w, b)| {
                (0..3)
                    .map(|k| (f(&w[k]) - f(&b[k])).abs())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        if o == 3 {
            assert_eq!(apart, 0.0, "{name}");
        } else {
            // more than a pixel
            assert!(apart > 1e-3, "{name} orientation {o}: {apart} m");
        }
        // the two flips cancel: the by-index positions are hklrs's in
        // orientation 3, the files differing only in the orientation
        let three = g["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["class"] == name && c["orientation"] == 3)
            .unwrap();
        let (_, lab, [a1, a2, a3]) = &labs()[0];
        let det = PoniFile::parse(three["poni"].as_str().unwrap())
            .unwrap()
            .detector(lab)
            .unwrap();
        for (px, zyx) in case["pixels"].as_array().unwrap().iter().zip(by_index) {
            let want = a1 * f(&zyx[1]) + a2 * f(&zyx[2]) + a3 * f(&zyx[0]);
            let ix = |k: usize| px[k].as_u64().unwrap() as usize;
            let d = (det.pixel(ix(0), ix(1)).unwrap().coords - want).norm();
            assert!(d <= TOL, "{name} orientation {o}, pixel {px}: {d} m");
        }
        seen += 1;
    }
    assert_eq!(seen, 16);
}
