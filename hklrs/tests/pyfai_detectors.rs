//! pyFAI 2026.9.0's named detector classes from a PONI file against the
//! instances `Geometry.load` builds (golden/pyfai_detectors.py) from the
//! files pyFAI writes and from files of other pitches and binnings:
//! `calc_pos_zyx` at pixels of each supported class, `Detector.mask`, and a
//! typed error for every other class, where pyFAI raises, and where its
//! instance's pitch, binning and shape disagree. An ImXPad image is flipped
//! once, as pyFAI's whole-image `calc_cartesian_positions` flips it;
//! pyFAI's by-index path flips it twice (upstream defect FAI-2).

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

fn ints(v: &Value) -> (i64, i64) {
    (v[0].as_i64().unwrap(), v[1].as_i64().unwrap())
}

/// A pitch from its repr.
fn pitch(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

/// The golden's mask fingerprint, none for no mask.
fn mask_of(e: &Value) -> Vec<u64> {
    e["mask"]
        .as_array()
        .map_or(vec![0; 3], |_| u64s(&e["mask"]))
}

/// pyFAI's instance `e` of class `c` has the class's pitch at its binning
/// and, for ImXPad, the default binning: a Rayonix's
/// table pitch, a Mar345 plate within the class's `max_shape`, else the
/// default instance's pitch scaled by the binning, to a relative 1e-9.
fn agrees(c: &Value, e: &Value) -> bool {
    let rule = &c["pitch"];
    (0..2).all(|k| {
        let Some(p) = e["pixel"][k].as_str().map(|p| p.parse::<f64>().unwrap()) else {
            return false;
        };
        let b = e["binning"][k].as_i64().unwrap();
        let own_b = c["binning"][k].as_i64().unwrap();
        let want = match rule["rule"].as_str().unwrap() {
            "table" => rule["table"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t[0].as_i64() == Some(b))
                .map(|t| t[1].as_f64().unwrap()),
            "plate" => {
                let max = e["max_shape"][k].as_i64().unwrap();
                (1..=rule["max_shape"][k].as_i64().unwrap())
                    .contains(&max)
                    .then_some(p)
            }
            _ => Some(pitch(&c["pixel"][k]) * b as f64 / own_b as f64),
        };
        b >= 1
            && p.is_finite()
            && p > 0.0
            && want.is_some_and(|w| (p - w).abs() <= 1e-9 * w)
            && (c["layout"] != "edges" || b == own_b)
    })
}

/// What hklrs makes of a file of class `c` that pyFAI reads as `e`, its
/// instance's facts, with `outcome` the exception the read or the mask
/// raises: the refusal, or None for a detector.
fn refusal(c: &Value, e: &Value, outcome: &Value) -> Option<PoniError> {
    let name: &'static str = c["class"].as_str().unwrap().to_string().leak();
    if outcome["raises"].is_string() {
        return Some(PoniError::Raises { class: name });
    }
    if e["class"] == "Detector" {
        // version 1 of one pitch or of a class without force_pixel: the
        // generic detector, which has no shape
        return Some(PoniError::Missing("max_shape"));
    }
    if let Some(k) = (0..2).find(|&k| e["pixel"][k].is_null()) {
        let key = ["pixel1", "pixel2"][k];
        return Some(PoniError::NoPitch { class: name, key });
    }
    if !agrees(c, e) {
        return Some(PoniError::Inconsistent {
            class: name,
            pixel: (pitch(&e["pixel"][0]), pitch(&e["pixel"][1])),
            binning: ints(&e["binning"]),
            shape: ints(&e["shape"]),
            class_pixel: (pitch(&c["pixel"][0]), pitch(&c["pixel"][1])),
            class_binning: (ints(&c["binning"]).0 as u64, ints(&c["binning"]).1 as u64),
        });
    }
    let (n0, n1) = pair(&e["shape"]);
    if n0 == 0 || n1 == 0 {
        // the binning setter floors max_shape to an empty image
        return Some(PoniError::Core(hklrs::Error::DetectorShape {
            slow: n0,
            fast: n1,
        }));
    }
    if outcome["mask_error"].is_string() {
        let (b0, b1) = ints(&e["binning"]);
        return Some(PoniError::MaskBinning {
            class: name,
            binning: (b0 as u64, b1 as u64),
            max_shape: pair(&e["max_shape"]),
        });
    }
    None
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
        let pf = PoniFile::parse(case["poni"].as_str().unwrap()).unwrap();
        assert_eq!(pf.poni(), Err(PoniError::NamedDetector(name.to_string())));
        let (shape, mask) = (pair(&case["shape"]), u64s(&case["fingerprint"]));
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
    assert_eq!(
        refused("Pilatus1M", r#"{"max_shape": [1043, 980]}"#),
        "max_shape"
    );
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
    // Perkin's default instance is binned 2x2; a Rayonix bins by its table
    for (name, b, shape) in [
        ("Perkin", "[2, 2]", (2048, 2048)),
        ("Perkin", "[1, 1]", (4096, 4096)),
        ("Rayonix133", "[1, 1]", (4096, 4096)),
        ("Rayonix133", "[2, 2]", (2048, 2048)),
    ] {
        let config = format!(r#"{{"binning": {b}}}"#);
        let det = file(name, &config).detector(&lab).unwrap();
        assert_eq!(det.shape(), shape, "{name} {b}");
    }
    // Detector.__init__ keeps a pitch Pilatus1M's is not, unbinned
    let refused = file("Pilatus1M", r#"{"pixel1": 1e-4}"#).detector(&lab);
    assert_eq!(
        refused,
        Err(PoniError::Inconsistent {
            class: "Pilatus1M",
            pixel: (1e-4, 172e-6),
            binning: (1, 1),
            shape: (1043, 981),
            class_pixel: (172e-6, 172e-6),
            class_binning: (1, 1),
        })
    );
    let text = refused.unwrap_err().to_string();
    assert!(
        text.ends_with(
            "Pilatus1M defines its own pixel, (0.000172, 0.000172) at binning (1, 1), \
             and another pitch belongs on the generic Detector"
        ),
        "{text}"
    );
    // ImXPad's chip edges span max_shape whatever the binning
    assert!(matches!(
        file("ImXPadS10", r#"{"binning": [2, 2]}"#).detector(&lab),
        Err(PoniError::Inconsistent {
            class: "ImXPadS10",
            binning: (2, 2),
            shape: (60, 40),
            ..
        })
    ));
    // the binning setter divides by the binning, and Rayonix's by 0.0
    for name in ["Pilatus1M", "Rayonix133"] {
        assert_eq!(
            file(name, r#"{"binning": [0, 0]}"#).detector(&lab),
            Err(PoniError::Raises { class: name })
        );
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
        file("Mythen", r#"{"splinefile": "a.spline"}"#).detector(&lab),
        Err(PoniError::Unsupported("spline file"))
    );
}

/// Every file the golden reads (Detector_config pitches and binnings,
/// version 1 pixel sizes): pyFAI's instance where its pitch, binning and
/// shape agree, at the pixels of the generic detector of that pitch, shape
/// and the class's orientation (the golden checks pyFAI's are), with
/// `Detector.mask`; else the typed refusal of what pyFAI does.
#[test]
fn reads_are_pyfai_s_instance_or_refused() {
    let g = load("pyfai_detectors.json");
    let lab = lab();
    let mut seen = std::collections::BTreeMap::new();
    for c in g["classes"].as_array().unwrap() {
        let Some(reads) = c["reads"].as_array().filter(|_| c["layout"] != "generic") else {
            continue;
        };
        for e in reads {
            let head = e["head"].as_str().unwrap();
            let text = format!(
                "{head}Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\n\
                 Rot3: 0.3\nWavelength: 1e-10\n"
            );
            let pf = PoniFile::parse(&text).unwrap();
            let got = pf.detector(&lab);
            let outcome = match (refusal(c, e, e), got) {
                (Some(want), got) => {
                    assert_eq!(got.err(), Some(want.clone()), "{head}");
                    format!("{want:?}")
                        .split([' ', '('])
                        .next()
                        .unwrap()
                        .to_string()
                }
                (None, got) => {
                    let det = got.unwrap_or_else(|err| panic!("{head}: {err:?}"));
                    assert_eq!(det.shape(), pair(&e["shape"]), "{head}");
                    assert_eq!(fingerprint(&det), mask_of(e), "{head}");
                    if c["layout"] == "uniform" {
                        let generic = format!(
                            "poni_version: 2.1\nDetector: Detector\nDetector_config: \
                             {{\"pixel1\": {}, \"pixel2\": {}, \"max_shape\": {}, \
                             \"orientation\": {}}}\n{}",
                            e["pixel"][0].as_str().unwrap(),
                            e["pixel"][1].as_str().unwrap(),
                            e["shape"],
                            c["orientation"],
                            &text[head.len()..]
                        );
                        let want = PoniFile::parse(&generic).unwrap().detector(&lab).unwrap();
                        let (n0, n1) = det.shape();
                        for (i, j) in [
                            (0, 0),
                            (n0 - 1, 0),
                            (0, n1 - 1),
                            (n0 - 1, n1 - 1),
                            (n0 / 3, n1 / 2),
                        ] {
                            let d = (det.pixel(i, j).unwrap() - want.pixel(i, j).unwrap()).norm();
                            assert!(d <= TOL, "{head} pixel ({i}, {j}): {d} m");
                        }
                    }
                    "built".to_string()
                }
            };
            *seen.entry(outcome).or_insert(0) += 1;
        }
    }
    eprintln!("reads: {seen:?}");
    for outcome in [
        "built",
        "Core",
        "Raises",
        "NoPitch",
        "Inconsistent",
        "MaskBinning",
        "Missing",
    ] {
        assert!(seen.contains_key(outcome), "{outcome}: {seen:?}");
    }
}

/// PoniFile.write of each modelled class binned by the factory at each
/// binning of the golden, read back: pyFAI's `calc_pos_zyx` and
/// `Detector.mask` of what `Geometry.load` builds, or the refusal where
/// that instance's pitch, binning and shape disagree (Mar555's pitch is
/// kept on the unbinned shape, a Rayonix's off-table binning divides its
/// pitch) or its mask raises. FReLoN, which takes no pitch, reads back as
/// the default instance, as pyFAI reads it.
#[test]
fn round_trips_are_pyfai_s_read_or_refused() {
    let g = load("pyfai_detectors.json");
    let mut seen = std::collections::BTreeMap::new();
    for c in g["classes"].as_array().unwrap() {
        let Some(trips) = c["round_trip"]
            .as_array()
            .filter(|_| c["layout"] != "generic")
        else {
            continue;
        };
        let name = c["class"].as_str().unwrap();
        for e in trips {
            let pf = PoniFile::parse(e["poni"].as_str().unwrap()).unwrap();
            let at = format!("{name} binned {}", e["binning"]);
            let read = if e["read"].is_object() { &e["read"] } else { e };
            if let Some(want) = refusal(c, read, e) {
                assert_eq!(pf.detector(&lab()).err(), Some(want.clone()), "{at}");
                let outcome = format!("{want:?}")
                    .split([' ', '('])
                    .next()
                    .unwrap()
                    .to_string();
                *seen.entry(outcome).or_insert(0) += 1;
                continue;
            }
            for (lab_name, lab, [a1, a2, a3]) in labs() {
                let det = pf
                    .detector(&lab)
                    .unwrap_or_else(|err| panic!("{at}: {err:?}"));
                assert_eq!(det.shape(), pair(&read["shape"]), "{at}");
                assert_eq!(fingerprint(&det), mask_of(e), "{at}");
                let pixels = e["pixels"].as_array().unwrap();
                for (px, zyx) in pixels.iter().zip(e["zyx"].as_array().unwrap()) {
                    let want = a1 * f(&zyx[1]) + a2 * f(&zyx[2]) + a3 * f(&zyx[0]);
                    let ix = |k: usize| px[k].as_u64().unwrap() as usize;
                    let d = (det.pixel(ix(0), ix(1)).unwrap().coords - want).norm();
                    assert!(d <= TOL, "{at}, {lab_name} lab, pixel {px}: {d} m");
                }
            }
            let kept = e["written"] == *read;
            *seen
                .entry(if kept { "kept" } else { "read otherwise" }.to_string())
                .or_insert(0) += 1;
        }
    }
    eprintln!("round trips: {seen:?}");
    for outcome in ["kept", "read otherwise", "Inconsistent", "MaskBinning"] {
        assert!(seen.contains_key(outcome), "{outcome}: {seen:?}");
    }
}

/// The generic Detector with a binning key (the golden's generic_files):
/// pyFAI's pitch times the binning over `max_shape` floor divided, at its
/// `calc_pos_zyx`, refused where `Detector.binning`'s setter raises (no
/// pitch, no `max_shape`, a 0) and as the core refuses an empty image or a
/// negative pitch. A round trip reads back as pyFAI reads it: the binned
/// pitch over the unbinned `max_shape` `PoniFile.write` gives.
#[test]
fn generic_binnings_are_pyfai_s() {
    let g = load("pyfai_detectors.json");
    let classes = g["classes"].as_array().unwrap();
    let c = classes.iter().find(|c| c["layout"] == "generic").unwrap();
    let geometry = "Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\n\
                    Rot3: 0.3\nWavelength: 1e-10\n";
    let reads = c["reads"].as_array().unwrap().iter();
    let reads = reads.map(|e| (format!("{}{geometry}", e["head"].as_str().unwrap()), e, e));
    let trips = c["round_trip"].as_array().unwrap().iter();
    let trips = trips.map(|e| (e["poni"].as_str().unwrap().to_string(), &e["read"], e));
    let mut seen = std::collections::BTreeMap::new();
    for (text, facts, outcome) in reads.chain(trips) {
        let pf = PoniFile::parse(&text).unwrap();
        let (n0, n1) = if facts["shape"].is_array() {
            pair(&facts["shape"])
        } else {
            (0, 0)
        };
        let want = if outcome["raises"].is_string() {
            Some(PoniError::Raises { class: "Detector" })
        } else if n0 == 0 || n1 == 0 {
            Some(PoniError::Core(hklrs::Error::DetectorShape {
                slow: n0,
                fast: n1,
            }))
        } else {
            let p = (pitch(&facts["pixel"][0]), pitch(&facts["pixel"][1]));
            (p.0 <= 0.0).then_some(PoniError::Core(hklrs::Error::NotPositive {
                what: "pixel1",
                value: p.0,
            }))
        };
        if let Some(want) = want {
            assert_eq!(pf.detector(&lab()).err(), Some(want.clone()), "{text}");
            let outcome = format!("{want:?}")
                .split([' ', '('])
                .next()
                .unwrap()
                .to_string();
            *seen.entry(outcome).or_insert(0) += 1;
            continue;
        }
        let poni = pf.poni().unwrap();
        let p = (pitch(&facts["pixel"][0]), pitch(&facts["pixel"][1]));
        assert_eq!(
            (poni.pixel1, poni.pixel2, poni.shape),
            (p.0, p.1, (n0, n1)),
            "{text}"
        );
        for (lab_name, lab, [a1, a2, a3]) in labs() {
            let det = pf.detector(&lab).unwrap();
            assert_eq!(det.mask(), None, "{text}");
            let pixels = outcome["pixels"].as_array().unwrap();
            for (px, zyx) in pixels.iter().zip(outcome["zyx"].as_array().unwrap()) {
                let want = a1 * f(&zyx[1]) + a2 * f(&zyx[2]) + a3 * f(&zyx[0]);
                let ix = |k: usize| px[k].as_u64().unwrap() as usize;
                let d = (det.pixel(ix(0), ix(1)).unwrap().coords - want).norm();
                assert!(d <= TOL, "{text}, {lab_name} lab, pixel {px}: {d} m");
            }
        }
        *seen.entry("built".to_string()).or_insert(0) += 1;
    }
    eprintln!("generic: {seen:?}");
    for outcome in ["built", "Raises", "Core"] {
        assert!(seen.contains_key(outcome), "{outcome}: {seen:?}");
    }
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
