//! Independent checks of an inverse solution, shared by the inverse tests.

use std::f64::consts::{PI, TAU};

use hklrs::nalgebra::{Matrix3, Point3, Vector3};
use hklrs::{Equation, Geometry, Mode, Wavelength, scattering};

/// Largest hkl miss and mode-equation miss (rad, or relative to k²) allowed.
pub const ROUND_TRIP: f64 = 1e-10;

pub fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r > PI { r - TAU } else { r }
}

/// Largest of `d`; NaN when any is NaN.
pub fn largest(d: impl Iterator<Item = f64>) -> f64 {
    d.fold(0.0, |m, d| if d > m || d.is_nan() { d } else { m })
}

/// ψ of `reference` read off an orthonormal frame about Q̂ (e₁ along
/// (k̂f × k̂i) × Q̂), not the oriented-angle formula the solver uses.
pub fn psi_frame(
    geo: &Geometry,
    ub: &Matrix3<f64>,
    reference: &Vector3<f64>,
    values: &[f64],
) -> f64 {
    let poses = geo.poses(values).unwrap();
    let beam = geo.beam_direction().into_inner();
    let (kf, ki) = (poses.detector.rotation * beam, poses.source * beam);
    let e3 = (kf - ki).normalize();
    let e1 = kf.cross(&ki).cross(&e3).normalize();
    let e2 = e3.cross(&e1);
    let p = poses.sample.rotation * (ub * reference);
    p.dot(&e2).atan2(p.dot(&e1))
}

/// Whether `values` brings `hkl` into reflection through `forward`'s
/// scattering, satisfies every equation of `mode`, keeps the held axes
/// exactly at `current` and has its written axes in (−π, π].
pub fn check(
    mode: &Mode,
    ub: &Matrix3<f64>,
    wavelength: Wavelength,
    hkl: &Vector3<f64>,
    current: &[f64],
    values: &[f64],
) -> Result<(), String> {
    let geo = mode.geometry();
    let point = Point3::from(geo.beam_direction().into_inner());
    let s = scattering(geo, values, wavelength, &point).map_err(|e| e.to_string())?;
    let back = ub.lu().solve(&s.q_sample).ok_or("singular UB")?;
    let miss = largest((back - hkl).iter().map(|d| d.abs()));
    within(miss, || {
        format!("{values:?} misses hkl {hkl:?} by {miss:e}")
    })?;
    for (i, (v, c)) in values.iter().zip(current).enumerate() {
        let ok = if mode.written().contains(&i) {
            *v > -PI && *v <= PI
        } else {
            v.to_bits() == c.to_bits()
        };
        if !ok {
            return Err(format!("{values:?}: axis {i} unwrapped or moved from {c}"));
        }
    }
    let k = wavelength.wavenumber();
    for e in mode.extra() {
        let miss = match e {
            Equation::Bisector { sample, detector } => {
                wrap(values[*detector] - 2.0 * values[*sample]).abs()
            }
            Equation::DoubleDiffraction { hkl2 } => {
                let poses = geo.poses(values).unwrap();
                let ki = poses.source * geo.beam_direction().into_inner() * k;
                ((poses.sample.rotation * (ub * hkl2) + ki).norm_squared() - k * k).abs() / (k * k)
            }
            Equation::PsiConstant { reference, psi } => {
                wrap(psi_frame(geo, ub, reference, values) - psi).abs()
            }
        };
        within(miss, || format!("{values:?} misses {e:?} by {miss:e}"))?;
    }
    Ok(())
}

/// NaN is a miss.
fn within(miss: f64, error: impl FnOnce() -> String) -> Result<(), String> {
    if miss <= ROUND_TRIP {
        Ok(())
    } else {
        Err(error())
    }
}
