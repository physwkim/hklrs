//! Geometry presets ported from hkl (`hkl/hkl-engine-*.c`), lab frame x
//! along the beam and z up, axis order as hkl lists it.

use crate::{Error, Geometry, Rad, Wavelength};

/// P08 LISA sample-detector distance, metres (`D_SD`).
pub const LISA_SD: f64 = 0.5;
/// P08 LISA monochromator spacings, Å (`D_Si111`, `D_Si220`).
const LISA_D_SI111: f64 = 3.136;
const LISA_D_SI220: f64 = 1.920;

/// Incident-beam deflection of P08 LISA at `wavelength`, for the `tilt` axis:
/// 2(asin(λ/2d₂₂₀) − asin(λ/2d₁₁₁)), hkl `_alpha_max`. k̂ᵢ = R_y(α)·x̂ =
/// (cos α, 0, −sin α), hkl's ki before the source holder.
pub fn lisa_tilt(wavelength: Wavelength) -> Result<Rad, Error> {
    let bragg = |d: f64| {
        let ratio = wavelength.angstrom() / (2.0 * d);
        if ratio > 1.0 {
            Err(Error::BraggImpossible { ratio })
        } else {
            Ok(ratio.asin())
        }
    };
    Ok(Rad(2.0 * (bragg(LISA_D_SI220)? - bragg(LISA_D_SI111)?)))
}

/// Kappa arm tilt of the hkl kappa presets.
pub const KAPPA_ALPHA_DEG: f64 = 50.0;

/// Names accepted by [`preset`], as hkl names them.
pub const PRESETS: &[&str] = &[
    "TwoC",
    "E4CV",
    "E4CH",
    "E6C",
    "K4CV",
    "K6C",
    "ZAXIS",
    "SOLEIL MARS",
    "SOLEIL SIXS MED1+2",
    "SOLEIL SIXS MED2+2",
    "SOLEIL SIXS MED2+3",
    "SOLEIL SIXS MED2+3 v2",
    "SOLEIL SIRIUS KAPPA",
    "SOLEIL SIRIUS TURRET",
    "SOLEIL CRISTAL K4C",
    "SOLEIL NANOSCOPIUM ROBOT",
    "ESRF ID01 PSIC",
    "ESRF BM28 PSIC",
    "APS POLAR",
    "PETRA3 P09 EH2",
    "PETRA3 P23 4C",
    "PETRA3 P23 6C",
    "PYFAI",
    "PETRA3 P08 LISA",
];

const X: [f64; 3] = [1.0, 0.0, 0.0];
const MX: [f64; 3] = [-1.0, 0.0, 0.0];
const Y: [f64; 3] = [0.0, 1.0, 0.0];
const MY: [f64; 3] = [0.0, -1.0, 0.0];
const Z: [f64; 3] = [0.0, 0.0, 1.0];
const MZ: [f64; 3] = [0.0, 0.0, -1.0];

/// Kappa axis at tilt `alpha` from -y towards -z (hkl `hkl-engine-k4c.c`).
pub fn kappa_axis(alpha: Rad) -> [f64; 3] {
    [0.0, -alpha.0.cos(), -alpha.0.sin()]
}

/// Geometry `name` as hkl builds it (`hkl-engine-*.c`): axes in the order hkl
/// first adds them, an axis hkl lists in both holders shared. Lengths are
/// metres; hkl's unit labels (SIRIUS TURRET `basepitch` mrad, CRISTAL K4C
/// `dtx` mm) only change how the value is written, not what it is here.
pub fn preset(name: &str) -> Result<Geometry, Error> {
    let kappa = || kappa_axis(Rad(KAPPA_ALPHA_DEG.to_radians()));
    let b = Geometry::builder(name);
    match name {
        // hkl-engine-2c.c
        "TwoC" => b
            .sample_rotation("omega", MY)
            .detector_rotation("tth", MY)
            .build(),
        // hkl-engine-e4c.c
        "E4CV" => b
            .sample_rotation("omega", MY)
            .sample_rotation("chi", X)
            .sample_rotation("phi", MY)
            .detector_rotation("tth", MY)
            .build(),
        "E4CH" => b
            .sample_rotation("omega", Z)
            .sample_rotation("chi", X)
            .sample_rotation("phi", Z)
            .detector_rotation("tth", Z)
            .build(),
        "SOLEIL MARS" => b
            .sample_rotation("omega", MY)
            .sample_rotation("chi", MX)
            .sample_rotation("phi", Z)
            .detector_rotation("tth", MY)
            .build(),
        // hkl-engine-e6c.c
        "E6C" => b
            .sample_rotation("mu", Z)
            .sample_rotation("omega", MY)
            .sample_rotation("chi", X)
            .sample_rotation("phi", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-k4c.c
        "K4CV" => b
            .sample_rotation("komega", MY)
            .sample_rotation("kappa", kappa())
            .sample_rotation("kphi", MY)
            .detector_rotation("tth", MY)
            .build(),
        // hkl-engine-k6c.c
        "K6C" => b
            .sample_rotation("mu", Z)
            .sample_rotation("komega", MY)
            .sample_rotation("kappa", kappa())
            .sample_rotation("kphi", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-zaxis.c: mu is in both holders
        "ZAXIS" => b
            .sample_rotation("mu", Z)
            .sample_rotation("omega", MY)
            .detector_rotation("mu", Z)
            .detector_rotation("delta", MY)
            .detector_rotation("gamma", Z)
            .build(),
        // hkl-engine-soleil-sixs-med.c: beta / pitch is in both holders
        "SOLEIL SIXS MED1+2" => b
            .sample_rotation("pitch", MY)
            .sample_rotation("mu", Z)
            .detector_rotation("pitch", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .build(),
        "SOLEIL SIXS MED2+2" => b
            .sample_rotation("beta", MY)
            .sample_rotation("mu", Z)
            .sample_rotation("omega", MY)
            .detector_rotation("beta", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .build(),
        "SOLEIL SIXS MED2+3" => b
            .sample_rotation("beta", MY)
            .sample_rotation("mu", Z)
            .sample_rotation("omega", MY)
            .detector_rotation("beta", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .detector_rotation("eta_a", MX)
            .build(),
        "SOLEIL SIXS MED2+3 v2" => b
            .sample_rotation("mu", Z)
            .sample_rotation("omega", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .detector_rotation("eta_a", MX)
            .build(),
        // hkl-engine-soleil-sirius-turret.c: basepitch is in both holders
        "SOLEIL SIRIUS TURRET" => b
            .sample_rotation("basepitch", Y)
            .sample_rotation("thetah", MZ)
            .sample_rotation("alphay", Y)
            .sample_rotation("alphax", X)
            .detector_rotation("basepitch", Y)
            .detector_rotation("delta", MZ)
            .detector_rotation("gamma", MY)
            .build(),
        // hkl-engine-k6c.c
        "SOLEIL SIRIUS KAPPA" => b
            .sample_rotation("mu", MZ)
            .sample_rotation("komega", MY)
            .sample_rotation("kappa", kappa())
            .sample_rotation("kphi", MY)
            .detector_rotation("delta", MZ)
            .detector_rotation("gamma", MY)
            .build(),
        // hkl-engine-soleil-cristal-k4c.c: dtx translates along x after tth
        "SOLEIL CRISTAL K4C" => b
            .sample_rotation("komega", MY)
            .sample_rotation("kappa", kappa())
            .sample_rotation("kphi", MY)
            .detector_rotation("tth", MY)
            .detector_translation("dtx", X)
            .build(),
        // hkl-engine-soleil-nanoscopium-robot.c: r is a length with no axis
        "SOLEIL NANOSCOPIUM ROBOT" => b
            .sample_rotation("rz", MZ)
            .sample_rotation("rs", Y)
            .sample_rotation("rx", MX)
            .detector_parameter("r")
            .detector_rotation("delta", MY)
            .detector_rotation("gamma", MZ)
            .build(),
        // hkl-engine-esrf-id01-psic.c
        "ESRF ID01 PSIC" => b
            .sample_rotation("mu", MZ)
            .sample_rotation("eta", MY)
            .sample_rotation("phi", MZ)
            .detector_rotation("nu", MZ)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-esrf-bm28-psic.c
        "ESRF BM28 PSIC" => b
            .sample_rotation("mu", Z)
            .sample_rotation("eta", MY)
            .sample_rotation("chi", X)
            .sample_rotation("phi", MY)
            .detector_rotation("nu", Z)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-aps-04id-polar.c: tau is in both holders
        "APS POLAR" => b
            .sample_rotation("tau", MY)
            .sample_rotation("mu", Z)
            .sample_rotation("chi", X)
            .sample_rotation("phi", Z)
            .detector_rotation("tau", MY)
            .detector_rotation("gamma", Z)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-petra3-p09-eh2.c: mu is in both holders
        "PETRA3 P09 EH2" => b
            .sample_rotation("mu", MY)
            .sample_rotation("omega", Z)
            .sample_rotation("chi", X)
            .sample_rotation("phi", Z)
            .detector_rotation("mu", MY)
            .detector_rotation("delta", Z)
            .detector_rotation("gamma", MY)
            .build(),
        // hkl-engine-petra3-p23-4c.c
        "PETRA3 P23 4C" => b
            .sample_rotation("omega_t", MY)
            .sample_rotation("mu", MZ)
            .detector_rotation("gamma", MZ)
            .detector_rotation("delta", MY)
            .build(),
        // hkl-engine-petra3-p23-6c.c
        "PETRA3 P23 6C" => b
            .sample_rotation("omega_t", MY)
            .sample_rotation("mu", MZ)
            .sample_rotation("omega", MY)
            .sample_rotation("chi", MX)
            .sample_rotation("phi", Y)
            .detector_rotation("gamma", MZ)
            .detector_rotation("delta", MY)
            .build(),
        // hkl 7be4f79 hkl-engine-pyfai.c: no sample axes; the three
        // translations are in metres and, unlike hkl's kf, move the pixel
        "PYFAI" => b
            .detector_rotation("rot3", X)
            .detector_rotation("rot2", MY)
            .detector_rotation("rot1", MZ)
            .detector_translation("poni1", Z)
            .detector_translation("poni2", Y)
            .detector_translation("sdd", X)
            .build(),
        // hkl 7be4f79 hkl-engine-petra3-p08-lisa.c (not registered upstream).
        // The source stack is mchi (not a sample axis) then `tilt`, the
        // monochromator deflection: set it to `lisa_tilt(wavelength)`.
        // drot turns about (0.5, 0, 0), so the detector point to use is
        // (0.5, 0, 0), as hkl's kf does.
        "PETRA3 P08 LISA" => b
            .source_rotation("mchi", X)
            .source_rotation("tilt", Y)
            .sample_rotation("sphi", Z)
            .detector_rotation("dtth", MZ)
            .detector_translation("dh", Z)
            .detector_rotation_about("drot", MY, [LISA_SD, 0.0, 0.0])
            .build(),
        _ => Err(Error::UnknownPreset {
            name: name.to_owned(),
        }),
    }
}
