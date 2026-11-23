//! Readers for the golden JSON files shared by the parity tests.

use hklrs::Axis;
use hklrs::nalgebra::{Matrix3, Vector3};
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

/// A golden canonical value (degrees, millimetres) as the user value
/// `Geometry::values_from_degrees` takes (degrees, metres).
pub fn canonical_to_user(axis: &Axis, v: f64) -> f64 {
    if axis.is_rotation() { v } else { v * 1e-3 }
}
