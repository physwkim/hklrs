//! Solid angle and polarisation against pyFAI 2026.9.0
//! `Geometry.solidAngleArray` and `Geometry.polarization`
//! (golden/pyfai_weights.py), on detectors imported from the PONI.

pub mod common;

use common::{f, load};
use hklrs::nalgebra::{Unit, Vector3};
use hklrs::poni::{LabAxes, Orientation, Poni};
use hklrs::{
    Frame, Geometry, PixelDetector, Polarisation, Rad, Wavelength, Weights, polarisation,
    solid_angle,
};
use serde_json::Value;

/// Relative; both sides are float64 to a few ulp.
const SOLID_ANGLE_TOL: f64 = 1e-14;

/// pyFAI casts its float64 polarisation to float32 (2⁻²⁴ relative); twice
/// that, plus the float64 rounding of its O(1) terms.
fn polarisation_tol(pyfai: f64) -> f64 {
    pyfai * f64::from(f32::EPSILON) + 4.0 * f64::EPSILON
}

fn poni(g: &Value, case: &Value) -> Poni {
    let n = |v: &Value| v.as_u64().unwrap() as usize;
    Poni {
        dist: f(&g["dist"]),
        poni1: f(&g["poni1"]),
        poni2: f(&g["poni2"]),
        rot1: Rad(f(&case["rot1"])),
        rot2: Rad(f(&case["rot2"])),
        rot3: Rad(f(&case["rot3"])),
        pixel1: f(&g["pixel1"]),
        pixel2: f(&g["pixel2"]),
        shape: (n(&g["shape"][0]), n(&g["shape"][1])),
        orientation: Orientation::try_from(case["orientation"].as_u64().unwrap() as u32).unwrap(),
    }
}

fn labs() -> [(&'static str, Vector3<f64>, Vector3<f64>); 2] {
    let beam = Vector3::new(0.3, -0.5, 0.81).normalize();
    let up = beam.cross(&Vector3::new(0.2, 0.7, -0.4)).normalize();
    [
        // hkl presets: x along the beam, z up
        ("hkl", Vector3::x(), Vector3::z()),
        ("oblique", beam, up),
    ]
}

/// One golden case seen from one lab.
struct Case<'a> {
    what: String,
    geometry: Geometry,
    detector: PixelDetector,
    /// pyFAI axes 1 (up) and 2 in the lab
    a1: Vector3<f64>,
    a2: Vector3<f64>,
    golden: &'a Value,
}

fn cases(g: &Value) -> Vec<Case<'_>> {
    let golden = g["cases"].as_array().unwrap();
    let mut seen: Vec<(String, u64)> = golden
        .iter()
        .map(|c| {
            (
                c["rots"].as_str().unwrap().to_owned(),
                c["orientation"].as_u64().unwrap(),
            )
        })
        .collect();
    seen.sort();
    let mut want: Vec<(String, u64)> = ["none", "rot1", "rot2", "rot3", "all", "all_large"]
        .into_iter()
        .map(|r| (r.to_owned(), 3))
        .chain([1, 2, 4].map(|o| ("all".to_owned(), o)))
        .collect();
    want.sort();
    assert_eq!(seen, want);
    let mut out = Vec::new();
    for (lab_name, beam, up) in labs() {
        let lab = LabAxes::new(Unit::new_unchecked(beam), Unit::new_unchecked(up)).unwrap();
        for case in golden {
            let p = poni(g, case);
            out.push(Case {
                what: format!(
                    "{lab_name} {} orientation {}",
                    case["rots"], case["orientation"]
                ),
                geometry: Geometry::builder("pyfai")
                    .beam_direction(beam.into())
                    .build()
                    .unwrap(),
                detector: p.detector(&lab).unwrap(),
                a1: up,
                a2: up.cross(&beam),
                golden: case,
            });
        }
    }
    out
}

/// Neither weight depends on the wavelength.
fn frame() -> Frame {
    Frame {
        values: vec![],
        wavelength: Wavelength::new(1.0).unwrap(),
    }
}

/// Every pixel of `w` times `scale` against `want[slow][fast]`; a masked
/// pixel or a NaN fails.
fn compare(w: &Weights, want: &Value, scale: f64, tol: impl Fn(f64) -> f64, what: &str) {
    let (slow, fast) = w.shape();
    let rows = want.as_array().unwrap();
    assert_eq!(rows.len(), slow, "{what}");
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), fast, "{what}");
        for (j, v) in row.iter().enumerate() {
            let p = f(v);
            let got = w
                .get(i, j)
                .unwrap_or_else(|m| panic!("{what} ({i}, {j}): {m:?}, pyFAI {p}"))
                .get()
                * scale;
            assert!(
                (got - p).abs() <= tol(p),
                "{what} ({i}, {j}): hklrs {got}, pyFAI {p}"
            );
        }
    }
}

/// solidAngleArray is cos³ of the incidence, dist³/|p|³; hklrs's solid
/// angle is pixel1·pixel2·dist/|p|³, so scaled by dist²/(pixel1·pixel2)
/// it is the same number.
#[test]
fn solid_angle_matches_pyfai_solid_angle_array() {
    let g = load("pyfai_weights.json");
    let scale = f(&g["dist"]).powi(2) / (f(&g["pixel1"]) * f(&g["pixel2"]));
    for c in cases(&g) {
        let w = solid_angle(&c.geometry, &frame(), &c.detector).unwrap();
        compare(
            &w,
            &c.golden["solid_angle"],
            scale,
            |p| p * SOLID_ANGLE_TOL,
            &c.what,
        );
    }
}

/// pyFAI's ½(1 + cos²2θ − f cos 2(χ+δ) sin²2θ), χ = atan2(t1, t2), is
/// hklrs's with ê = cos δ a2 − sin δ a1: then ê·k̂f = sin 2θ cos(χ+δ) and
/// (k̂i × ê)·k̂f = ±sin 2θ sin(χ+δ).
#[test]
fn polarisation_matches_pyfai_polarization() {
    let g = load("pyfai_weights.json");
    let mut factors = Vec::new();
    for c in cases(&g) {
        for pol in c.golden["polarization"].as_array().unwrap() {
            let (factor, offset) = (f(&pol["factor"]), f(&pol["axis_offset"]));
            let e = c.a2 * offset.cos() - c.a1 * offset.sin();
            let w = polarisation(
                &c.geometry,
                &frame(),
                &c.detector,
                Polarisation { e, factor },
            )
            .unwrap();
            let what = format!("{} factor {factor} axis_offset {offset}", c.what);
            compare(&w, &pol["values"], 1.0, polarisation_tol, &what);
            factors.push(factor);
        }
    }
    assert!(factors.contains(&1.0) && factors.contains(&-1.0));
}
