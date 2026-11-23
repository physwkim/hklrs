//! hkl's SOLEIL SIXS MED "hkl" engine modes (`hkl-engine-soleil-sixs-med.c`)
//! of MED2+2 (mu_fixed, reflectivity, emergence_fixed), MED1+2
//! (pitch_fixed, delta_fixed) and MED2+3 and its v2 (mu_fixed, gamma_fixed,
//! emergence_fixed), for any geometry with the axes they name. Axes a mode
//! does not write hold at `current`; [`slits`] adds MED2+3's slit fit, which
//! v2 does only with eta_a_rotation = 1.

use nalgebra::Vector3;

use super::{Equation, InverseError, Mode};
use crate::{AxisKind, Geometry};

pub fn mu_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "mu_fixed",
        &["omega", "gamma", "delta"],
        |_| vec![],
    )
}

/// gamma ≡ 2·mu (mod 2π).
pub fn reflectivity(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "reflectivity",
        &["mu", "omega", "gamma", "delta"],
        |w| {
            vec![Equation::Bisector {
                sample: w[0],
                detector: w[2],
            }]
        },
    )
}

/// k̂f leaves the surface of sample-holder normal `surface` at `emergence`
/// (rad); hkl's default surface is (0, 1, 0).
pub fn emergence_fixed(
    geometry: &Geometry,
    surface: Vector3<f64>,
    emergence: f64,
) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "emergence_fixed",
        &["mu", "omega", "gamma", "delta"],
        |_| vec![Equation::Emergence { surface, emergence }],
    )
}

pub fn pitch_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "pitch_fixed",
        &["mu", "gamma", "delta"],
        |_| vec![],
    )
}

pub fn delta_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "delta_fixed",
        &["pitch", "mu", "gamma"],
        |_| vec![],
    )
}

pub fn gamma_fixed(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "gamma_fixed",
        &["mu", "omega", "delta"],
        |_| vec![],
    )
}

/// `mode` with eta_a turning the slits normal ẑ (detector frame) into the
/// surface normal to omega's axis.
pub fn slits(mode: Mode<'_>) -> Result<Mode<'_>, InverseError> {
    let geometry = mode.geometry();
    let eta_a = geometry.axis_index("eta_a")?;
    let omega = &geometry.axes()[geometry.axis_index("omega")?];
    let surface = match omega.kind() {
        AxisKind::Rotation { axis } | AxisKind::RotationAbout { axis, .. } => axis.into_inner(),
        _ => {
            return Err(InverseError::NotRotation {
                name: omega.name().to_owned(),
            });
        }
    };
    mode.with_slits(eta_a, surface, Vector3::z())
}
