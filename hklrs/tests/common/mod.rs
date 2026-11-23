//! Readers for the golden JSON files shared by the parity tests.

use hklrs::nalgebra::{Matrix3, Vector3};
use hklrs::{Axis, Lattice};
use serde_json::Value;

pub fn load(file: &str) -> Value {
    let path = format!("{}/../golden/{file}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).unwrap()
}

pub fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

pub fn mat(v: &Value) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| f(&v[i][j]))
}

pub fn vec3(v: &Value) -> Vector3<f64> {
    Vector3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}

/// `{"a", "b", "c", "alpha", "beta", "gamma"}`, Å and degrees.
pub fn lattice(l: &Value) -> Lattice {
    Lattice::from_degrees(
        f(&l["a"]),
        f(&l["b"]),
        f(&l["c"]),
        f(&l["alpha"]),
        f(&l["beta"]),
        f(&l["gamma"]),
    )
    .unwrap()
}

/// Whether hkl's B[1][2] (∝ cos α*) equals the Busing-Levy one (∝ −cos α),
/// i.e. cβ cγ = cα (1 − sβ sγ); hkl's B and UB are wrong otherwise
/// (upstream-defects.md HKL-12).
pub fn hkl_b_correct(l: &Lattice) -> bool {
    let (ca, cb, cg) = (l.alpha().0.cos(), l.beta().0.cos(), l.gamma().0.cos());
    let (sb, sg) = (l.beta().0.sin(), l.gamma().0.sin());
    ((cb * cg - ca) / (sb * sg) + ca).abs() < 1e-14
}

/// A golden canonical value (degrees, millimetres) as the user value
/// `Geometry::values_from_degrees` takes (degrees, metres).
pub fn canonical_to_user(axis: &Axis, v: f64) -> f64 {
    if axis.is_rotation() { v } else { v * 1e-3 }
}
