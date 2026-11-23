//! Projections against binoculars-ng 7be4f79 on one K6C frame and a 3×4
//! detector (golden/bin_7be4f79_projections.c). binoculars computes Q in
//! single precision (cglm vec3s, mat4s), so Q-derived coordinates agree to
//! 1e-6; its angles space is double precision and in degrees.

pub mod common;
use common::{f, load, mat};

use hklrs::nalgebra::{Matrix3, Unit, Vector3};
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{
    Frame, Geometry, Lattice, Orientation, PixelDetector, Rad, Sample, Wavelength,
    detector_positions,
};
use serde_json::Value;

const Q_TOL: f64 = 1e-6;

struct Setup {
    golden: Value,
    geometry: Geometry,
    frame: Frame,
    detector: PixelDetector,
    sample: Sample,
}

fn setup() -> Setup {
    let golden = load("bin_7be4f79_projections.json");
    let list = |k: &str| -> Vec<f64> { golden[k].as_array().unwrap().iter().map(f).collect() };
    let v = |k: &str| {
        let a = list(k);
        Vector3::new(a[0], a[1], a[2])
    };
    let geometry = preset("K6C").unwrap();
    let frame = Frame {
        values: geometry.values_from_degrees(&list("values_deg")).unwrap(),
        wavelength: Wavelength::new(f(&golden["wavelength"])).unwrap(),
    };
    let shape = list("shape");
    let detector = PixelDetector::new(
        (shape[0] as usize, shape[1] as usize),
        v("origin"),
        v("e_slow"),
        v("e_fast"),
    )
    .unwrap();
    let l = list("lattice");
    let sample = Sample::new(
        "s",
        Lattice::new(l[0], l[1], l[2], Rad(l[3]), Rad(l[4]), Rad(l[5])).unwrap(),
        Orientation::from_u_matrix(&mat(&golden["u"])).unwrap(),
    );
    Setup {
        golden,
        geometry,
        frame,
        detector,
        sample,
    }
}

impl Setup {
    /// binoculars' coordinates of `key`, one row per pixel, after checking
    /// that it labels them with `names` and kept every pixel.
    fn space(&self, key: &str, names: &[&str]) -> Vec<Vec<f64>> {
        let s = &self.golden["spaces"][key];
        let got: Vec<&str> = s["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap())
            .collect();
        assert_eq!(got, names, "{key}");
        let res = f(&self.golden["resolution"]);
        let rows: Vec<Vec<f64>> = s["indexes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|i| f(i) * res).collect())
            .collect();
        let (slow, fast) = self.detector.shape();
        assert_eq!(rows.len(), slow * fast, "{key}");
        rows
    }

    /// hklrs' coordinates for `p`, one row per pixel.
    fn project(&self, p: &Projection) -> Vec<Vec<f64>> {
        let out = p
            .project(&self.geometry, &self.frame, &self.detector)
            .unwrap();
        (0..out.pixels())
            .map(|i| out.planes().iter().map(|plane| plane[i]).collect())
            .collect()
    }
}

fn max_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn q_sample_and_qpar_qper_match_qcustom() {
    let s = setup();
    for (key, p) in [
        ("qx_qy_qz", Projection::QSample),
        // binoculars' surface normals: HORIZONTAL ẑ, VERTICAL ŷ
        // (hkl-binoculars.c:801-812)
        (
            "qpar_qper_horizontal",
            Projection::QparQper {
                normal: Unit::new_normalize(Vector3::z()),
            },
        ),
        (
            "qpar_qper_vertical",
            Projection::QparQper {
                normal: Unit::new_normalize(Vector3::y()),
            },
        ),
    ] {
        let d = max_diff(&s.space(key, p.names()), &s.project(&p));
        assert!(d < Q_TOL, "{key}: {d}");
    }
}

#[test]
fn hkl_matches_binoculars_through_hkl_ub_but_not_its_b_hkl12() {
    // HKL-12: hkl's B[1][2] is wrong for a triclinic cell and the hkl space
    // takes UB from hkl_sample_UB_get (hkl-binoculars.c:1532), so binoculars
    // inherits it. Same U, same lattice: only UB's third column differs.
    let s = setup();
    let hkl_ub = mat(&s.golden["ub"]);
    let ub = s.sample.ub();
    for j in 0..2 {
        assert!((hkl_ub.column(j) - ub.column(j)).amax() < 1e-12, "{j}");
    }
    assert!((hkl_ub.column(2) - ub.column(2)).amax() > 1e-4);

    let bin = s.space("hkl", &["H", "K", "L"]);
    // binoculars' kernel is UB⁻¹ Q_sample: with hkl's UB it is hklrs' Q
    let inv: Matrix3<f64> = hkl_ub.try_inverse().unwrap();
    let through_hkl_ub: Vec<Vec<f64>> = s
        .project(&Projection::QSample)
        .iter()
        .map(|q| (inv * Vector3::new(q[0], q[1], q[2])).as_slice().to_vec())
        .collect();
    let d = max_diff(&bin, &through_hkl_ub);
    assert!(d < Q_TOL, "{d}");

    let d = max_diff(&bin, &s.project(&Projection::Hkl(s.sample.clone())));
    assert!(d > 1e-3, "{d}");
}

#[test]
fn angles_match_binoculars_except_tth_bin13() {
    let s = setup();
    let bin = s.space("angles", Projection::Angles.names());
    let ours = s.project(&Projection::Angles);
    let p = detector_positions(&s.geometry, &s.frame, &s.detector).unwrap();
    for (i, (b, r)) in bin.iter().zip(&ours).enumerate() {
        for k in 0..2 {
            assert!((b[k] - r[k].to_degrees()).abs() < 1e-9, "{i} {k}");
        }
        // BIN-13: binoculars' tth is acos of the pixel's x in metres
        assert!((b[2] - p.x[i].acos().to_degrees()).abs() < 1e-9, "{i}");
        assert!((b[2] - r[2].to_degrees()).abs() > 30.0, "{i}");
    }
}
