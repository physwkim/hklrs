//! Read-only pseudo axes of `hkl-pseudoaxis-common-q.c` (reals to pseudo):
//! `q`, `q2` (q, alpha), `tth2` (tth, alpha, `hkl-pseudoaxis-common-tth.c`)
//! and `qper_qpar`. Q is `k_f − k_i` with 2π, Å⁻¹.
//!
//! Each engine is a kernel on a [`Scattering`] (so the inverse core can call
//! it inside a residual) and an `_at` function that builds the scattering
//! from a geometry position.

use nalgebra::{Point3, UnitQuaternion, Vector3};

use crate::{Error, Geometry, Rad, Scattering, Wavelength, scattering};

/// Engine `q`: |Q| = 4π sin θ / λ, negative when k_f points to −y or −z in
/// the lab frame (`common-q.c:96`). hkl's sign convention for the
/// four-circle presets, where k_f stays in one plane.
pub fn q_signed(s: &Scattering) -> f64 {
    let kf = s.kf.into_inner();
    if kf.y < 0.0 || kf.z < 0.0 { -s.q } else { s.q }
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
/// Q·(k̂i × n̂) (`common-q.c:233-247`). The qpar sign is 0 when k_f lies in
/// the k_i–n̂ plane; zero then counts as positive, where hkl takes the sign
/// bit of the rounded product.
pub fn qper_qpar(
    s: &Scattering,
    sample_rotation: &UnitQuaternion<f64>,
    surface: &Vector3<f64>,
) -> Result<(f64, f64), Error> {
    let n = (sample_rotation * surface)
        .try_normalize(0.0)
        .ok_or_else(|| Error::ZeroVector {
            what: "surface normal".to_owned(),
        })?;
    let q = s.q_lab;
    let along = q.dot(&n);
    let sign = |dot: f64| if dot < 0.0 { -1.0 } else { 1.0 };
    Ok((
        sign(along) * along.abs(),
        sign(q.dot(&s.ki.cross(&n))) * (q - n * along).norm(),
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
