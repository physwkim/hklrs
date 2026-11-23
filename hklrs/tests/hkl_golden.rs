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
//!
//! PYFAI's rot3 turns about −x as in pyFAI and about +x in hkl (HKL-37), so
//! hkl at rot3 is asserted equal to hklrs at −rot3 and unequal at rot3.

mod common;
mod lattice_check;

use common::{canonical_to_user, f, load, mat, vec3};
use hklrs::nalgebra::Vector3;
use hklrs::{
    AxisKind, Orientation, PixelDetector, Sample, Wavelength, forward, presets, scattering,
};
use lattice_check::{hkl_b_correct, lattice};
use serde_json::Value;

/// Translations that move the detector point off its ray from the sample.
const OFF_RAY: &[(&str, &str)] = &[("PYFAI", "poni1"), ("PYFAI", "poni2")];

/// HKL-37: the axis hkl turns the other way from hklrs.
const HKL_37: (&str, &str) = ("PYFAI", "rot3");

/// Axes hklrs adds that hkl does not list: P08 LISA's `tilt`, the
/// wavelength-dependent monochromator deflection (`presets::lisa_tilt`).
const EXTRA_AXES: &[(&str, &str)] = &[("PETRA3 P08 LISA", "tilt")];

/// Detector point of a golden case, metres: hkl's kf for LISA is the full
/// detector transform of (D_SD, 0, 0); the others use 1 m along the beam.
fn detector_point(name: &str, geo: &hklrs::Geometry) -> hklrs::nalgebra::Point3<f64> {
    if name == "PETRA3 P08 LISA" {
        hklrs::nalgebra::Point3::new(presets::LISA_SD, 0.0, 0.0)
    } else {
        PixelDetector::point(geo.beam_direction(), 1.0)
            .unwrap()
            .pixel(0, 0)
            .unwrap()
    }
}

/// Presets hkl has no "hkl" engine for: the axis table only.
const NO_ENGINE: &[&str] = &["TwoC"];

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

#[test]
fn presets_match_hkl_axis_tables() {
    let g = golden();
    for name in presets::PRESETS {
        let geo = presets::preset(name).unwrap();
        let axes = g["geometries"][name]["axes"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} not in golden"));
        let ours: Vec<_> = geo
            .axes()
            .iter()
            .map(|a| a.name())
            .filter(|n| !EXTRA_AXES.contains(&(name, n)))
            .collect();
        let theirs: Vec<_> = axes.iter().map(|a| a["name"].as_str().unwrap()).collect();
        assert_eq!(ours, theirs, "{name} axis order");
        let listed = geo
            .axes()
            .iter()
            .filter(|a| !EXTRA_AXES.contains(&(name, a.name())));
        for (a, h) in listed.zip(axes) {
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
                (Some(v), false) => {
                    let sign = if (*name, a.name()) == HKL_37 {
                        -1.0
                    } else {
                        1.0
                    };
                    assert!(
                        (v - sign * vec3(&h["axis_v"])).norm() < 1e-15,
                        "{name} {}",
                        a.name()
                    )
                }
                _ => panic!("{name} {}: axis vector presence differs", a.name()),
            }
        }
    }
}

#[test]
fn forward_hkl_matches_hkl_engine() {
    let g = golden();
    let mut checked = 0;
    let mut divergent = 0;
    for case in g["cases"].as_array().unwrap() {
        let name = case["geometry"].as_str().unwrap();
        if !presets::PRESETS.contains(&name) || NO_ENGINE.contains(&name) {
            continue;
        }
        assert!(case.get("error").is_none(), "{name}: {}", case["error"]);
        let geo = presets::preset(name).unwrap();
        let lattice = lattice(&case["lattice"]);
        let sample = Sample::new(
            "golden",
            lattice,
            Orientation::from_u_matrix(&mat(&case["U"])).unwrap(),
        );
        let ctx = format!(
            "{name} {} λ={} #{}",
            case["sample"], case["wavelength"], case["index"]
        );

        // the full UB chain is compared only where hkl's B is right (HKL-12);
        // the geometry is compared for every cell through hkl's own UB below
        let hkl_b_valid = hkl_b_correct(&lattice);
        let hkl_ub = mat(&case["UB"]);
        if hkl_b_valid {
            let ub_err = (sample.ub() - hkl_ub).abs().max();
            assert!(ub_err < 1e-12, "{ctx}: UB differs by {ub_err}");
        }

        let wl = Wavelength::new(f(&case["wavelength"])).unwrap();
        // golden canonical values are degrees and millimetres; the HKL-37
        // axis is multiplied by `hkl_37`
        let values = |hkl_37: f64| {
            let user: Vec<f64> = geo
                .axes()
                .iter()
                .map(|a| {
                    if EXTRA_AXES.contains(&(name, a.name())) {
                        return hklrs::Deg::from(presets::lisa_tilt(wl).unwrap()).0;
                    }
                    if is_off_ray(name, a.name()) {
                        return 0.0;
                    }
                    let v = canonical_to_user(a, f(&case["axes_canonical_deg_mm"][a.name()]));
                    if (name, a.name()) == HKL_37 {
                        hkl_37 * v
                    } else {
                        v
                    }
                })
                .collect();
            geo.values_from_degrees(&user).unwrap()
        };
        let point = detector_point(name, &geo);
        let expect = vec3(&case["hkl"]);
        let geometry_gap = |out: &hklrs::Forward| {
            let via_hkl_ub = hkl_ub.lu().solve(&out.scattering.q_sample).unwrap();
            (via_hkl_ub - expect).abs().max()
        };
        let out = forward(&geo, &sample, &values(-1.0), wl, &point).unwrap();
        let err = geometry_gap(&out);
        assert!(err < 1e-10, "{ctx}: geometry: hkl differs by {err}");
        if hkl_b_valid {
            let err = (out.hkl - expect).abs().max();
            assert!(err < 1e-10, "{ctx}: hkl {:?} vs {expect:?}", out.hkl);
        }
        if name == HKL_37.0 {
            let unflipped = forward(&geo, &sample, &values(1.0), wl, &point).unwrap();
            let gap = geometry_gap(&unflipped);
            // 1e4 times the match tolerance; the smallest gap, 4.4e-4, is
            // cubic λ=1.54 #7, where rot3 is −0.013°
            assert!(gap > 1e-6, "{ctx}: HKL-37 not seen, hkl differs by {gap}");
            divergent += 1;
        }
        checked += 1;
    }
    assert_eq!(
        checked,
        (presets::PRESETS.len() - NO_ENGINE.len()) * 40,
        "every case of every ported preset that has an engine"
    );
    assert_eq!(divergent, 40, "every PYFAI case shows HKL-37");
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
        let make = |zero: bool| -> Vec<f64> {
            let user: Vec<f64> = geo
                .axes()
                .iter()
                .map(|a| match a.kind() {
                    AxisKind::Translation { .. } if zero => 0.0,
                    _ => canonical_to_user(a, canon(a.name())),
                })
                .collect();
            geo.values_from_degrees(&user).unwrap()
        };
        let det = PixelDetector::point(geo.beam_direction(), 1.0).unwrap();
        let wl = Wavelength::new(f(&case["wavelength"])).unwrap();
        let px = det.pixel(0, 0).unwrap();
        let full = scattering(&geo, &make(false), wl, &px).unwrap();
        let rot_only = scattering(&geo, &make(true), wl, &px).unwrap();
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
