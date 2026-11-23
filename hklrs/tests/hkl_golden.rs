//! Forward parity with hkl 5.1.9 ("hkl" engine get), golden/hkl_forward.json.

use hklrs::nalgebra::{Matrix3, Vector3};
use hklrs::{AxisKind, Lattice, Orientation, PixelDetector, Sample, Wavelength, forward, presets};
use serde_json::Value;

fn golden() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../golden/hkl_forward.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("golden/hkl_forward.json")).unwrap()
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
        if !presets::PRESETS.contains(&name) || case.get("error").is_some() {
            continue;
        }
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
                if matches!(a.kind(), AxisKind::Translation { .. }) {
                    v * 1e-3
                } else {
                    v
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
        presets::PRESETS.len() * 40,
        "every case of every ported preset"
    );
}
