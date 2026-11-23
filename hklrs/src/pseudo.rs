//! Read-only pseudo axes of `hkl-pseudoaxis-common-q.c` (reals to pseudo):
//! `q`, `q2` (q, alpha), `tth2` (tth, alpha, `hkl-pseudoaxis-common-tth.c`)
//! `qper_qpar`, and `incidence`/`emergence` (`hkl-pseudoaxis-common-readonly.c`). Q is `k_f − k_i` with 2π, Å⁻¹.
//!
//! Each engine is a kernel on a [`Scattering`] (so the inverse core can call
//! it inside a residual) and an `_at` function that builds the scattering
//! from a geometry position.

use nalgebra::{Point3, UnitQuaternion, Vector3};

use std::f64::consts::FRAC_PI_2;

use crate::{Error, Geometry, Rad, Scattering, Wavelength, angle_between, scattering};

/// Smallest length of the unit surface normal's projection on the plane
/// normal to a beam for its azimuth to be defined; the azimuth error is
/// about eps over this, 1e-7 rad.
pub const AZIMUTH_MIN: f64 = 1e-9;

/// Engine `q`: |Q| = 4π sin θ / λ, negative when k_f points to −y or −z in
/// the lab frame (`common-q.c:96`). hkl's sign convention for the
/// four-circle presets, where k_f stays in one plane.
pub fn q_signed(s: &Scattering) -> f64 {
    q_sign(&s.kf) * s.q
}

/// The sign of [`q_signed`] for a k_f along `kf`.
pub(crate) fn q_sign(kf: &Vector3<f64>) -> f64 {
    if kf.y < 0.0 || kf.z < 0.0 { -1.0 } else { 1.0 }
}

/// −1 below zero, else 1: hkl's qper and qpar signs.
fn sign(dot: f64) -> f64 {
    if dot < 0.0 { -1.0 } else { 1.0 }
}

/// The sign of qpar ([`qper_qpar`]) for Q along `q`, k_i along `ki` and the
/// lab surface normal `n`.
pub(crate) fn qpar_sign(q: &Vector3<f64>, ki: &Vector3<f64>, n: &Vector3<f64>) -> f64 {
    sign(q.dot(&ki.cross(n)))
}

/// Engine `q2`: (|Q| ≥ 0, alpha), alpha = atan2(k_f,z, k_f,y), the angle from
/// y of k_f projected on the plane normal to x (`common-q.c:134-140`).
pub fn q2(s: &Scattering) -> (f64, Rad) {
    let kf = s.kf.into_inner();
    (s.q, Rad(kf.z.atan2(kf.y)))
}

/// Engine `tth2` (`common-tth.c:51-60`): (2θ, alpha), 2θ the angle between
/// k_i and k_f in [0, π], alpha as in [`q2`]. hkl takes 2θ from an `acos`
/// (HKL-11); [`Scattering::tth`] is `atan2`-based.
pub fn tth2(s: &Scattering) -> (Rad, Rad) {
    (s.tth, q2(s).1)
}

/// Engine `qper_qpar`: (qper, qpar). `surface` is the normal in the sample
/// frame (hkl parameters x, y, z; default (0, 1, 0)), turned by
/// `sample_rotation`, the rotation of the sample stack at these axes.
///
/// qper = ±|(Q·n̂)n̂| signed by Q·n̂; qpar = ±|Q − (Q·n̂)n̂| signed by
/// Q·(k̂i × n̂) (`common-q.c:344-362`). The qpar sign is a convention where
/// k_f lies in the k_i–n̂ plane (the product is 0 there); zero counts as
/// positive, which matches hkl at exact delta = 0 and gamma = 0 (the sign
/// bit of the rounded product, as hkl takes it, differs on −0.0).
pub fn qper_qpar(
    s: &Scattering,
    sample_rotation: &UnitQuaternion<f64>,
    surface: &Vector3<f64>,
) -> Result<(f64, f64), Error> {
    let n = surface_in_lab(sample_rotation, surface)?;
    let q = s.q_lab;
    let along = q.dot(&n);
    Ok((
        sign(along) * along.abs(),
        qpar_sign(&q, &s.ki, &n) * (q - n * along).norm(),
    ))
}

fn surface_in_lab(
    sample_rotation: &UnitQuaternion<f64>,
    surface: &Vector3<f64>,
) -> Result<Vector3<f64>, Error> {
    (sample_rotation * surface)
        .try_normalize(0.0)
        .ok_or_else(|| Error::ZeroVector {
            what: "surface normal".to_owned(),
        })
}

/// Angle from y of `n` projected on the plane normal to `axis` (unit);
/// `None` when `n` is within [`AZIMUTH_MIN`] of `axis`, where hkl returns 0.
fn azimuth(n: &Vector3<f64>, axis: &Vector3<f64>) -> Option<Rad> {
    let projected = n - axis * n.dot(axis);
    (projected.norm() > AZIMUTH_MIN).then(|| angle_between(&projected, &Vector3::y()))
}

/// Engine `incidence` (`common-readonly.c:93-96`, `common-readonly-private.h:42`):
/// (angle(n̂, k_i) − π/2, azimuth), n̂ the surface normal in the lab frame
/// (`surface` turned by `sample_rotation`; hkl parameters x, y, z). The
/// azimuth is the angle from y of n̂ projected on the plane normal to k_i.
///
/// hkl reports the azimuth of the last of `incidence`/`emergence` it
/// computes, because both register the pseudo axis name "azimuth" in one
/// shared list; the value here is each engine's own.
pub fn incidence(
    s: &Scattering,
    sample_rotation: &UnitQuaternion<f64>,
    surface: &Vector3<f64>,
) -> Result<(Rad, Option<Rad>), Error> {
    let n = surface_in_lab(sample_rotation, surface)?;
    let ki = s.ki.into_inner();
    Ok((Rad(angle_between(&n, &ki).0 - FRAC_PI_2), azimuth(&n, &ki)))
}

/// Engine `emergence` (`common-readonly.c:250-253`): (π/2 − angle(n̂, k_f),
/// azimuth). Unlike `incidence`, the azimuth is taken on the plane normal to
/// the lab x axis, not to k_f.
pub fn emergence(
    s: &Scattering,
    sample_rotation: &UnitQuaternion<f64>,
    surface: &Vector3<f64>,
) -> Result<(Rad, Option<Rad>), Error> {
    let n = surface_in_lab(sample_rotation, surface)?;
    let kf = s.kf.into_inner();
    Ok((
        Rad(FRAC_PI_2 - angle_between(&n, &kf).0),
        azimuth(&n, &Vector3::x()),
    ))
}

/// [`q_signed`] at `values`; `point` as for [`scattering`].
pub fn q_at(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
) -> Result<f64, Error> {
    Ok(q_signed(&scattering(geometry, values, wavelength, point)?))
}

/// [`q2`] at `values`; `point` as for [`scattering`].
pub fn q2_at(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
) -> Result<(f64, Rad), Error> {
    Ok(q2(&scattering(geometry, values, wavelength, point)?))
}

/// [`tth2`] at `values`; `point` as for [`scattering`].
pub fn tth2_at(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
) -> Result<(Rad, Rad), Error> {
    Ok(tth2(&scattering(geometry, values, wavelength, point)?))
}

/// [`qper_qpar`] at `values`; `point` as for [`scattering`].
pub fn qper_qpar_at(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
    surface: &Vector3<f64>,
) -> Result<(f64, f64), Error> {
    let s = scattering(geometry, values, wavelength, point)?;
    let sample = geometry.poses(values)?.sample.rotation;
    qper_qpar(&s, &sample, surface)
}

/// [`incidence`] at `values`; `point` as for [`scattering`].
pub fn incidence_at(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
    surface: &Vector3<f64>,
) -> Result<(Rad, Option<Rad>), Error> {
    let s = scattering(geometry, values, wavelength, point)?;
    incidence(&s, &geometry.poses(values)?.sample.rotation, surface)
}

/// [`emergence`] at `values`; `point` as for [`scattering`].
pub fn emergence_at(
    geometry: &Geometry,
    values: &[f64],
    wavelength: Wavelength,
    point: &Point3<f64>,
    surface: &Vector3<f64>,
) -> Result<(Rad, Option<Rad>), Error> {
    let s = scattering(geometry, values, wavelength, point)?;
    emergence(&s, &geometry.poses(values)?.sample.rotation, surface)
}
