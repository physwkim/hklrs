//! Geometry presets ported from hkl (`hkl/hkl-engine-*.c`), lab frame x
//! along the beam and z up, axis order as hkl lists it.

use crate::{Error, Geometry, Rad};

/// Kappa arm tilt of the hkl kappa presets.
pub const KAPPA_ALPHA_DEG: f64 = 50.0;

/// Names accepted by [`preset`].
pub const PRESETS: &[&str] = &["E4CV", "E4CH", "E6C", "K4CV"];

const X: [f64; 3] = [1.0, 0.0, 0.0];
const MY: [f64; 3] = [0.0, -1.0, 0.0];
const Z: [f64; 3] = [0.0, 0.0, 1.0];

/// Kappa axis at tilt `alpha` from -y towards -z (hkl `hkl-engine-k4c.c`).
pub fn kappa_axis(alpha: Rad) -> [f64; 3] {
    [0.0, -alpha.0.cos(), -alpha.0.sin()]
}

pub fn preset(name: &str) -> Result<Geometry, Error> {
    let kappa = || kappa_axis(Rad(KAPPA_ALPHA_DEG.to_radians()));
    match name {
        // hkl-engine-e4c.c
        "E4CV" => Geometry::builder(name)
            .sample_rotation("omega", MY)
            .sample_rotation("chi", X)
            .sample_rotation("phi", MY)
            .detector_rotation("tth", MY)
            .build(),
        "E4CH" => Geometry::builder(name)
            .sample_rotation("omega", Z)
            .sample_rotation("chi", X)
            .sample_rotation("phi", Z)
            .detector_rotation("tth", Z)
            .build(),
        // hkl-engine-e6c.c
        "E6C" => Geometry::builder(name)
            .sample_rotation("mu", Z)
            .sample_rotation("omega", MY)
            .sample_rotation("chi", X)
            .sample_rotation("phi", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-k4c.c
        "K4CV" => Geometry::builder(name)
            .sample_rotation("komega", MY)
            .sample_rotation("kappa", kappa())
            .sample_rotation("kphi", MY)
            .detector_rotation("tth", MY)
            .build(),
        _ => Err(Error::UnknownPreset {
            name: name.to_owned(),
        }),
    }
}
