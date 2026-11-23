//! Forward parity with hkl ("hkl" engine get): golden/hkl_forward.json (hkl
//! 5.1.9) and golden/hkl_7be4f79_forward.json (PYFAI, only in 7be4f79).
//!
//! hkl's kf is the detector rotation applied to +x and ignores detector
//! translations; hklrs moves the detector point with the full pose. A
//! translation that shifts the point off its ray from the sample (PYFAI
//! `poni1`, `poni2`) therefore changes the direction, so for those axes the
//! hkl comparison is made with the translation at 0 and the difference is
//! checked separately (`translations_move_the_detector_point`). A translation
//! along the ray (CRISTAL K4C `dtx`, PYFAI `sdd`) changes nothing in either.

use hklrs::nalgebra::{Matrix3, Vector3};
use hklrs::{AxisKind, Lattice, Orientation, PixelDetector, Sample, Wavelength, forward, presets};
use serde_json::Value;

/// Translations that move the detector point off its ray from the sample.
const OFF_RAY: &[(&str, &str)] = &[("PYFAI", "poni1"), ("PYFAI", "poni2")];

/// Presets hkl has no "hkl" engine for: the axis table only.
const NO_ENGINE: &[&str] = &["TwoC"];

fn load(file: &str) -> Value {
    let path = format!("{}/../golden/{file}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).unwrap()
}

/// Both golden files merged: geometries by name, cases concatenated.
fn golden() -> Value {
    let mut g = load("hkl_forward.json");
    let h = load("hkl_7be4f79_forward.json");
    for (k, v) in h["geometries"].as_object().unwrap() {
        g["geometries"][k] = v.clone();
    }
    g["cases"]
        .as_array_mut()
        .unwrap()
        .extend(h["cases"].as_array().unwrap().iter().cloned());
    g
}

fn is_off_ray(geometry: &str, axis: &str) -> bool {
    OFF_RAY.contains(&(geometry, axis))
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn mat(v: &Value) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| f(&v[i][j]))
}

fn vec3(v: &Value) -> Vector3<f64> {
    Vector3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}

#[test]
fn presets_match_hkl_axis_tables() {
    let g = golden();
    for name in presets::PRESETS {
        let geo = presets::preset(name).unwrap();
        let axes = g["geometries"][name]["axes"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} not in golden"));
        let ours: Vec<_> = geo.axes().iter().map(|a| a.name()).collect();
        let theirs: Vec<_> = axes.iter().map(|a| a["name"].as_str().unwrap()).collect();
        assert_eq!(ours, theirs, "{name} axis order");
        for (a, h) in geo.axes().iter().zip(axes) {
            let v = match a.kind() {
                AxisKind::Rotation { axis } | AxisKind::RotationAbout { axis, .. } => {
                    Some(axis.into_inner())
                }
                AxisKind::Translation { direction } => Some(direction.into_inner()),
                AxisKind::Parameter => None,
            };
            // hkl's unit label fixes the class: angle -> rotation, length with
            // an axis vector -> translation, length without -> parameter
            let angle = matches!(h["unit"].as_str().unwrap(), "Degree" | "Milli Radian");
            let class_ok = match a.kind() {
                AxisKind::Rotation { .. } | AxisKind::RotationAbout { .. } => angle,
                AxisKind::Translation { .. } => !angle && !h["axis_v"].is_null(),
                AxisKind::Parameter => !angle && h["axis_v"].is_null(),
            };
            assert!(class_ok, "{name} {}: unit {}", a.name(), h["unit"]);
            match (v, h["axis_v"].is_null()) {
                (None, true) => {}
                (Some(v), false) => assert!(
                    (v - vec3(&h["axis_v"])).norm() < 1e-15,
                    "{name} {}",
                    a.name()
                ),
                _ => panic!("{name} {}: axis vector presence differs", a.name()),
            }
        }
    }
}

#[test]
fn forward_hkl_matches_hkl_engine() {
    let g = golden();
    let mut checked = 0;
    for case in g["cases"].as_array().unwrap() {
        let name = case["geometry"].as_str().unwrap();
        if !presets::PRESETS.contains(&name) || NO_ENGINE.contains(&name) {
            continue;
        }
        assert!(case.get("error").is_none(), "{name}: {}", case["error"]);
        let geo = presets::preset(name).unwrap();
        let l = &case["lattice"];
        let lattice = Lattice::from_degrees(
            f(&l["a"]),
            f(&l["b"]),
            f(&l["c"]),
            f(&l["alpha"]),
            f(&l["beta"]),
            f(&l["gamma"]),
        )
        .unwrap();
        let sample = Sample::new(
            "golden",
            lattice,
            Orientation::from_u_matrix(&mat(&case["U"])).unwrap(),
        );
        let ctx = format!(
            "{name} {} λ={} #{}",
            case["sample"], case["wavelength"], case["index"]
        );

        // hkl's B is wrong unless beta = gamma = 90° (upstream-defects.md
        // HKL-12), so the full UB chain is compared only for those cells; the
        // geometry is compared for every cell through hkl's own UB below.
        let hkl_b_valid = f(&l["beta"]) == 90.0 && f(&l["gamma"]) == 90.0;
        let hkl_ub = mat(&case["UB"]);
        if hkl_b_valid {
            let ub_err = (sample.ub() - hkl_ub).abs().max();
            assert!(ub_err < 1e-12, "{ctx}: UB differs by {ub_err}");
        }

        // golden canonical values are degrees and millimetres
        let user: Vec<f64> = geo
            .axes()
            .iter()
            .map(|a| {
                let v = f(&case["axes_canonical_deg_mm"][a.name()]);
                if is_off_ray(name, a.name()) {
                    0.0
                } else if matches!(
                    a.kind(),
                    AxisKind::Rotation { .. } | AxisKind::RotationAbout { .. }
                ) {
                    v
                } else {
                    v * 1e-3 // mm -> m, translations and length parameters
                }
            })
            .collect();
        let values = geo.values_from_degrees(&user).unwrap();
        let det = PixelDetector::point(geo.beam_direction(), 1.0).unwrap();
        let wl = Wavelength::new(f(&case["wavelength"])).unwrap();
        let out = forward(&geo, &sample, &values, wl, &det.pixel(0, 0).unwrap()).unwrap();
        let expect = vec3(&case["hkl"]);
        let via_hkl_ub = hkl_ub.lu().solve(&out.q_sample).unwrap();
        let err = (via_hkl_ub - expect).abs().max();
        assert!(
            err < 1e-10,
            "{ctx}: geometry: hkl {via_hkl_ub:?} vs {expect:?}"
        );
        if hkl_b_valid {
            let err = (out.hkl - expect).abs().max();
            assert!(err < 1e-10, "{ctx}: hkl {:?} vs {expect:?}", out.hkl);
        }
        checked += 1;
    }
    assert_eq!(
        checked,
        (presets::PRESETS.len() - NO_ENGINE.len()) * 40,
        "every case of every ported preset that has an engine"
    );
}

/// The full pose moves the point by the translation; the angle between the
/// hklrs kf and the rotation-only kf is the angle between the point and the
/// translated point, whatever the rotations are.
#[test]
fn translations_move_the_detector_point() {
    let g = golden();
    let mut worst = 0.0f64;
    let mut n = 0;
    for case in g["cases"].as_array().unwrap() {
        let name = case["geometry"].as_str().unwrap();
        if name != "PYFAI" {
            continue;
        }
        let geo = presets::preset(name).unwrap();
        let canon = |ax: &str| f(&case["axes_canonical_deg_mm"][ax]);
        let l = &case["lattice"];
        let sample = Sample::new(
            "golden",
            Lattice::from_degrees(
                f(&l["a"]),
                f(&l["b"]),
                f(&l["c"]),
                f(&l["alpha"]),
                f(&l["beta"]),
                f(&l["gamma"]),
            )
            .unwrap(),
            Orientation::from_u_matrix(&mat(&case["U"])).unwrap(),
        );
        let make = |zero: bool| -> Vec<f64> {
            let user: Vec<f64> = geo
                .axes()
                .iter()
                .map(|a| match a.kind() {
                    AxisKind::Translation { .. } if zero => 0.0,
                    AxisKind::Translation { .. } => canon(a.name()) * 1e-3,
                    _ => canon(a.name()),
                })
                .collect();
            geo.values_from_degrees(&user).unwrap()
        };
        let det = PixelDetector::point(geo.beam_direction(), 1.0).unwrap();
        let wl = Wavelength::new(f(&case["wavelength"])).unwrap();
        let px = det.pixel(0, 0).unwrap();
        let full = forward(&geo, &sample, &make(false), wl, &px).unwrap();
        let rot_only = forward(&geo, &sample, &make(true), wl, &px).unwrap();
        // translation in the arm frame is (sdd, poni2, poni1) in metres
        let t = Vector3::new(
            canon("sdd") * 1e-3,
            canon("poni2") * 1e-3,
            canon("poni1") * 1e-3,
        );
        let moved = px.coords + t;
        let expect = hklrs::angle_between(&px.coords, &moved).0;
        let got = hklrs::angle_between(&full.kf, &rot_only.kf).0;
        assert!(
            (got - expect).abs() < 1e-12,
            "{name} #{}: {got} vs {expect}",
            case["index"]
        );
        worst = worst.max(got);
        n += 1;
    }
    assert_eq!(n, 40);
    eprintln!("PYFAI: worst kf difference to hkl {worst} rad at 1 m, |t| <= 0.05 m per axis");
}
