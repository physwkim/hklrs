//! Lattice readers and the HKL-12 gate, shared by the tests that compare
//! hkl's B or UB.

use hklrs::Lattice;
use serde_json::Value;

use crate::common::f;

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
