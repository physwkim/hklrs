//! hkl's "hkl" engine modes by name, one [`ModeTable`] per preset, for
//! callers that hold a mode as data (a Python binding, a saved
//! configuration) and build it against a geometry when they solve. A
//! geometry's table is the one of the preset it is named after; mode and
//! parameter names are hkl's for that preset, whose modes of one name differ
//! between presets.

use std::collections::BTreeMap;

use nalgebra::Vector3;

use super::{
    InverseError, Mode, aps_04id_polar as polar, e4c, e6c, esrf_bm28_psic as psic, k4c, k6c,
    petra3_p09_eh2 as p09, petra3_p23_4c as p23, pyfai, soleil_nanoscopium_robot as robot,
    soleil_sirius_turret as turret, soleil_sixs_med as med, zaxis,
};
use crate::Geometry;

/// A parameter of a mode, as hkl names it. An angle is radians in a
/// [`ModeSpec`] and degrees where hkl shows it to the user.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeParameter {
    pub name: &'static str,
    pub angle: bool,
}

/// hkl's psi engine mode: reference (Miller indices) and ψ (rad).
pub type PsiEngine = for<'g> fn(&'g Geometry, Vector3<f64>, f64) -> Result<Mode<'g>, InverseError>;

/// A mode from its parameter values in [`Entry::parameters`] order.
type Build = for<'g> fn(&'g Geometry, &[f64]) -> Result<Mode<'g>, InverseError>;

struct Entry {
    name: &'static str,
    parameters: &'static [ModeParameter],
    build: Build,
}

/// The "hkl" engine modes of a preset in hkl's order, its default first,
/// its engine list's parameters and its psi engine.
pub struct ModeTable {
    preset: &'static str,
    modes: &'static [Entry],
    /// hkl's engine-list parameters at their values on a new engine list
    /// (angles in rad); every mode takes them, last.
    engine_list: &'static [(ModeParameter, f64)],
    psi: Option<PsiEngine>,
}

const fn plain(name: &'static str) -> ModeParameter {
    ModeParameter { name, angle: false }
}

const fn angle(name: &'static str) -> ModeParameter {
    ModeParameter { name, angle: true }
}

const NONE: &[ModeParameter] = &[];
const OMEGA: &[ModeParameter] = &[angle("omega")];
const CHI: &[ModeParameter] = &[angle("chi")];
const PHI: &[ModeParameter] = &[angle("phi")];
/// `double_diffraction_parameters`, hkl-pseudoaxis-common-hkl-private.h.
const HKL2: &[ModeParameter] = &[plain("h2"), plain("k2"), plain("l2")];
/// `psi_constant_parameters`, hkl-pseudoaxis-common-hkl-private.h.
const PSI: &[ModeParameter] = &[plain("h2"), plain("k2"), plain("l2"), angle("psi")];
/// `constant_incidence_parameters`, hkl-engine-k6c.c.
const INCIDENCE: &[ModeParameter] = &[
    plain("x"),
    plain("y"),
    plain("z"),
    angle("incidence"),
    angle("azimuth"),
];
/// `emergence_fixed`'s, hkl-engine-soleil-sixs-med.c.
const EMERGENCE: &[ModeParameter] = &[plain("x"), plain("y"), plain("z"), angle("emergence")];
/// hkl's one engine-list parameter, hkl-engine-soleil-sixs-med.c:686:
/// MED2+3 v2 fits the slits as MED2+3 does where it is 1.
pub const ETA_A_ROTATION: ModeParameter = plain("eta_a_rotation");
const ROTATION: &[ModeParameter] = &[ETA_A_ROTATION];
const EMERGENCE_ROTATION: &[ModeParameter] = &[
    plain("x"),
    plain("y"),
    plain("z"),
    angle("emergence"),
    ETA_A_ROTATION,
];

/// The vector of the first three values: hkl2, a reference or a surface.
fn v(p: &[f64]) -> Vector3<f64> {
    Vector3::new(p[0], p[1], p[2])
}

macro_rules! modes {
    ($($name:literal: $parameters:expr, $build:expr;)*) => {
        &[$(Entry { name: $name, parameters: $parameters, build: $build }),*]
    };
}

/// hkl-engine-e4c.c: E4CV, E4CH, SOLEIL MARS.
const E4C: &[Entry] = modes! {
    "bissector": NONE, |g, _| e4c::bissector(g);
    "constant_omega": NONE, |g, _| e4c::constant_omega(g);
    "constant_chi": NONE, |g, _| e4c::constant_chi(g);
    "constant_phi": NONE, |g, _| e4c::constant_phi(g);
    "double_diffraction": HKL2, |g, p| e4c::double_diffraction(g, v(p));
    "psi_constant": PSI, |g, p| e4c::psi_constant(g, v(p), p[3]);
};

/// hkl-engine-k4c.c: K4CV; hkl-engine-soleil-cristal-k4c.c repeats it.
const K4C: &[Entry] = modes! {
    "bissector": NONE, |g, _| k4c::bissector(g);
    "constant_omega": OMEGA, |g, p| k4c::constant_omega(g, p[0]);
    "constant_chi": CHI, |g, p| k4c::constant_chi(g, p[0]);
    "constant_phi": PHI, |g, p| k4c::constant_phi(g, p[0]);
    "double_diffraction": HKL2, |g, p| k4c::double_diffraction(g, v(p));
    "psi_constant": PSI, |g, p| k4c::psi_constant(g, v(p), p[3]);
};

/// hkl-engine-e6c.c: E6C; hkl-engine-petra3-p23-6c.c repeats it.
const E6C: &[Entry] = modes! {
    "bissector_vertical": NONE, |g, _| e6c::bissector_vertical(g);
    "constant_omega_vertical": NONE, |g, _| e6c::constant_omega_vertical(g);
    "constant_chi_vertical": NONE, |g, _| e6c::constant_chi_vertical(g);
    "constant_phi_vertical": NONE, |g, _| e6c::constant_phi_vertical(g);
    "lifting_detector_phi": NONE, |g, _| e6c::lifting_detector_phi(g);
    "lifting_detector_omega": NONE, |g, _| e6c::lifting_detector_omega(g);
    "lifting_detector_mu": NONE, |g, _| e6c::lifting_detector_mu(g);
    "double_diffraction_vertical": HKL2, |g, p| e6c::double_diffraction_vertical(g, v(p));
    "bissector_horizontal": NONE, |g, _| e6c::bissector_horizontal(g);
    "double_diffraction_horizontal": HKL2, |g, p| e6c::double_diffraction_horizontal(g, v(p));
    "psi_constant_vertical": PSI, |g, p| e6c::psi_constant_vertical(g, v(p), p[3]);
    "psi_constant_horizontal": PSI, |g, p| e6c::psi_constant_horizontal(g, v(p), p[3]);
    "constant_mu_horizontal": NONE, |g, _| e6c::constant_mu_horizontal(g);
};

/// hkl-engine-k6c.c, on the preset's [`k6c::Detectors`].
macro_rules! k6c_modes {
    ($d:expr) => {
        modes! {
            "bissector_vertical": NONE, |g, _| k6c::bissector_vertical(g, $d);
            "constant_omega_vertical": OMEGA, |g, p| k6c::constant_omega_vertical(g, $d, p[0]);
            "constant_chi_vertical": CHI, |g, p| k6c::constant_chi_vertical(g, $d, p[0]);
            "constant_phi_vertical": PHI, |g, p| k6c::constant_phi_vertical(g, $d, p[0]);
            "lifting_detector_kphi": NONE, |g, _| k6c::lifting_detector_kphi(g, $d);
            "lifting_detector_komega": NONE, |g, _| k6c::lifting_detector_komega(g, $d);
            "lifting_detector_mu": NONE, |g, _| k6c::lifting_detector_mu(g, $d);
            "double_diffraction_vertical": HKL2,
                |g, p| k6c::double_diffraction_vertical(g, $d, v(p));
            "bissector_horizontal": NONE, |g, _| k6c::bissector_horizontal(g, $d);
            "constant_phi_horizontal": PHI, |g, p| k6c::constant_phi_horizontal(g, $d, p[0]);
            "constant_kphi_horizontal": NONE, |g, _| k6c::constant_kphi_horizontal(g, $d);
            "double_diffraction_horizontal": HKL2,
                |g, p| k6c::double_diffraction_horizontal(g, $d, v(p));
            "psi_constant_vertical": PSI,
                |g, p| k6c::psi_constant_vertical(g, $d, v(p), p[3]);
            "constant_incidence": INCIDENCE,
                |g, p| k6c::constant_incidence(g, $d, v(p), p[3], p[4]);
        }
    };
}

/// hkl-engine-esrf-bm28-psic.c; hkl-engine-esrf-id01-psic.c repeats it.
const PSIC: &[Entry] = modes! {
    "constant_nu_coplanar": NONE, |g, _| psic::constant_nu_coplanar(g);
    "constant_delta_coplanar": NONE, |g, _| psic::constant_delta_coplanar(g);
    "constant_eta_noncoplanar": NONE, |g, _| psic::constant_eta_noncoplanar(g);
};

/// `mode` with MED2+3's slit fit where [`ETA_A_ROTATION`] is 1. hkl reads
/// the value through `(int)`, so 0.5 would silently mean 0: here only 0 and
/// 1 are values.
fn eta_a_rotation(mode: Mode<'_>, value: f64) -> Result<Mode<'_>, InverseError> {
    if value == 1.0 {
        med::slits(mode)
    } else if value == 0.0 {
        Ok(mode)
    } else {
        Err(InverseError::NotSwitch {
            mode: mode.name().into(),
            parameter: ETA_A_ROTATION.name,
            value,
        })
    }
}

/// hkl-engine-soleil-sixs-med.c: MED2+3, its modes with the slit fit.
const MED23: &[Entry] = modes! {
    "mu_fixed": NONE, |g, _| med::mu_fixed(g).and_then(med::slits);
    "gamma_fixed": NONE, |g, _| med::gamma_fixed(g).and_then(med::slits);
    "emergence_fixed": EMERGENCE,
        |g, p| med::emergence_fixed(g, v(p), p[3]).and_then(med::slits);
};

/// MED2+3 v2: MED2+3's modes, the slit fit by [`ETA_A_ROTATION`].
const MED23_V2: &[Entry] = modes! {
    "mu_fixed": ROTATION, |g, p| eta_a_rotation(med::mu_fixed(g)?, p[0]);
    "gamma_fixed": ROTATION, |g, p| eta_a_rotation(med::gamma_fixed(g)?, p[0]);
    "emergence_fixed": EMERGENCE_ROTATION,
        |g, p| eta_a_rotation(med::emergence_fixed(g, v(p), p[3])?, p[4]);
};

static TABLES: [ModeTable; 24] = [
    ModeTable {
        preset: "TwoC",
        modes: &[],
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "E4CV",
        modes: E4C,
        engine_list: &[],
        psi: Some(e4c::psi),
    },
    ModeTable {
        preset: "E4CH",
        modes: E4C,
        engine_list: &[],
        psi: Some(e4c::psi),
    },
    ModeTable {
        preset: "E6C",
        modes: E6C,
        engine_list: &[],
        psi: Some(e6c::psi_vertical),
    },
    ModeTable {
        preset: "K4CV",
        modes: K4C,
        engine_list: &[],
        psi: Some(k4c::psi),
    },
    ModeTable {
        preset: "K6C",
        modes: k6c_modes!(k6c::K6C),
        engine_list: &[],
        psi: Some(k6c::psi_vertical),
    },
    ModeTable {
        preset: "ZAXIS",
        modes: modes! {
            "zaxis": NONE, |g, _| zaxis::zaxis(g);
            "reflectivity": NONE, |g, _| zaxis::reflectivity(g);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL MARS",
        modes: E4C,
        engine_list: &[],
        psi: Some(e4c::psi),
    },
    ModeTable {
        preset: "SOLEIL SIXS MED1+2",
        modes: modes! {
            "pitch_fixed": NONE, |g, _| med::pitch_fixed(g);
            "delta_fixed": NONE, |g, _| med::delta_fixed(g);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL SIXS MED2+2",
        modes: modes! {
            "mu_fixed": NONE, |g, _| med::mu_fixed(g);
            "reflectivity": NONE, |g, _| med::reflectivity(g);
            "emergence_fixed": EMERGENCE, |g, p| med::emergence_fixed(g, v(p), p[3]);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL SIXS MED2+3",
        modes: MED23,
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL SIXS MED2+3 v2",
        modes: MED23_V2,
        engine_list: &[(ETA_A_ROTATION, 0.0)],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL SIRIUS KAPPA",
        modes: k6c_modes!(k6c::SOLEIL_SIRIUS_KAPPA),
        engine_list: &[],
        psi: Some(k6c::psi_vertical_soleil_sirius_kappa),
    },
    ModeTable {
        preset: "SOLEIL SIRIUS TURRET",
        modes: modes! {
            "lifting_detector_thetah": NONE, |g, _| turret::lifting_detector_thetah(g);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL CRISTAL K4C",
        modes: K4C,
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "SOLEIL NANOSCOPIUM ROBOT",
        modes: modes! {
            "lifting detector rz": NONE, |g, _| robot::lifting_detector_rz(g);
            "lifting detector rs": NONE, |g, _| robot::lifting_detector_rs(g);
            "lifting detector rx": NONE, |g, _| robot::lifting_detector_rx(g);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "ESRF ID01 PSIC",
        modes: PSIC,
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "ESRF BM28 PSIC",
        modes: PSIC,
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "APS POLAR",
        modes: modes! {
            "4-circles constant phi horizontal": NONE, |g, _| polar::constant_phi_horizontal(g);
            "zaxis + alpha-fixed": NONE, |g, _| polar::zaxis_alpha_fixed(g);
            "zaxis + beta-fixed": NONE, |g, _| polar::zaxis_beta_fixed(g);
            "zaxis + alpha=beta": NONE, |g, _| polar::zaxis_alpha_eq_beta(g);
            "4-circles bissecting horizontal": NONE, |g, _| polar::bissecting_horizontal(g);
            "4-circles constant mu horizontal": NONE, |g, _| polar::constant_mu_horizontal(g);
            "4-circles constant chi horizontal": NONE, |g, _| polar::constant_chi_horizontal(g);
            "lifting detector tau": NONE, |g, _| polar::lifting_detector_tau(g);
            "lifting detector mu": NONE, |g, _| polar::lifting_detector_mu(g);
            "lifting detector chi": NONE, |g, _| polar::lifting_detector_chi(g);
            "lifting detector phi": NONE, |g, _| polar::lifting_detector_phi(g);
            "psi constant horizontal": PSI,
                |g, p| polar::psi_constant_horizontal(g, v(p), p[3]);
            "psi constant vertical": PSI, |g, p| polar::psi_constant_vertical(g, v(p), p[3]);
        },
        engine_list: &[],
        psi: Some(polar::psi_vertical),
    },
    ModeTable {
        preset: "PETRA3 P09 EH2",
        modes: modes! {
            "zaxis + alpha-fixed": NONE, |g, _| p09::zaxis_alpha_fixed(g);
            "zaxis + beta-fixed": NONE, |g, _| p09::zaxis_beta_fixed(g);
            "zaxis + alpha=beta": NONE, |g, _| p09::zaxis_alpha_eq_beta(g);
            "4-circles bissecting horizontal": NONE, |g, _| p09::bissecting_horizontal(g);
            "4-circles constant omega horizontal": NONE, |g, _| p09::constant_omega_horizontal(g);
            "4-circles constant chi horizontal": NONE, |g, _| p09::constant_chi_horizontal(g);
            "4-circles constant phi horizontal": NONE, |g, _| p09::constant_phi_horizontal(g);
            "lifting detector mu": NONE, |g, _| p09::lifting_detector_mu(g);
            "lifting detector omega": NONE, |g, _| p09::lifting_detector_omega(g);
            "lifting detector chi": NONE, |g, _| p09::lifting_detector_chi(g);
            "lifting detector phi": NONE, |g, _| p09::lifting_detector_phi(g);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "PETRA3 P23 4C",
        modes: modes! {
            "bissector_vertical": NONE, |g, _| p23::bissector_vertical(g);
            "lifting_detector_omega_t": NONE, |g, _| p23::lifting_detector_omega_t(g);
            "lifting_detector_mu": NONE, |g, _| p23::lifting_detector_mu(g);
            "bissector_horizontal": NONE, |g, _| p23::bissector_horizontal(g);
            "psi_constant": PSI, |g, p| p23::psi_constant(g, v(p), p[3]);
        },
        engine_list: &[],
        psi: None,
    },
    ModeTable {
        preset: "PETRA3 P23 6C",
        modes: E6C,
        engine_list: &[],
        psi: Some(e6c::psi_vertical),
    },
    ModeTable {
        preset: "PYFAI",
        modes: modes! {
            "pyfai_default": NONE, |g, _| pyfai::pyfai_default(g);
        },
        engine_list: &[],
        psi: None,
    },
    // hkl-engine-petra3-p08-lisa.c: no builder here for its modes.
    ModeTable {
        preset: "PETRA3 P08 LISA",
        modes: &[],
        engine_list: &[],
        psi: None,
    },
];

impl ModeTable {
    /// The table of preset `name` ([`crate::presets::PRESETS`]); the one of
    /// a geometry is `ModeTable::of(geometry.name())`.
    pub fn of(name: &str) -> Result<&'static Self, InverseError> {
        TABLES
            .iter()
            .find(|t| t.preset == name)
            .ok_or_else(|| InverseError::NotPreset {
                geometry: name.into(),
            })
    }

    pub fn preset(&self) -> &'static str {
        self.preset
    }

    /// hkl's mode names, its default first.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + use<> {
        self.modes.iter().map(|e| e.name)
    }

    /// hkl's default mode as a new engine list has it: the engine-list
    /// parameters at their values there; None on a preset without modes.
    pub fn default_mode(&self) -> Option<ModeSpec> {
        let name = self.modes.first()?.name;
        let values = self.engine_list.iter();
        Some(values.fold(ModeSpec::new(name), |s, (p, v)| s.with(p.name, *v)))
    }

    /// hkl's engine-list parameters at their values on a new engine list,
    /// angles in rad.
    pub fn engine_list(&self) -> &'static [(ModeParameter, f64)] {
        self.engine_list
    }

    /// The parameters of mode `name`, in hkl's order.
    pub fn parameters(&self, name: &str) -> Result<&'static [ModeParameter], InverseError> {
        Ok(self.entry(name)?.parameters)
    }

    /// The modes holding ψ about a reference: those taking `psi`.
    pub fn psi_modes(&self) -> impl Iterator<Item = &'static str> + use<> {
        let modes = self.modes.iter();
        modes
            .filter(|e| e.parameters.iter().any(|p| p.name == "psi"))
            .map(|e| e.name)
    }

    /// hkl's psi engine; None on a preset without one.
    pub fn psi(&self) -> Option<PsiEngine> {
        self.psi
    }

    fn entry(&self, name: &str) -> Result<&'static Entry, InverseError> {
        self.modes
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| InverseError::UnknownMode {
                geometry: self.preset.into(),
                name: name.into(),
            })
    }

    /// Mode `name`'s parameter `parameter`.
    fn parameter(&self, name: &str, parameter: &str) -> Result<ModeParameter, InverseError> {
        self.parameters(name)?
            .iter()
            .find(|p| p.name == parameter)
            .copied()
            .ok_or_else(|| InverseError::UnknownParameter {
                mode: name.into(),
                parameter: parameter.into(),
            })
    }
}

/// A mode by name and its parameters, as the table of the geometry it is
/// built on names them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModeSpec {
    pub name: String,
    /// By name; angles in radians.
    pub parameters: BTreeMap<String, f64>,
}

impl ModeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: BTreeMap::new(),
        }
    }

    /// With parameter `name` at `value` (an angle in radians).
    pub fn with(mut self, name: impl Into<String>, value: f64) -> Self {
        self.parameters.insert(name.into(), value);
        self
    }

    /// With a psi mode's reference (h2, k2, l2) and ψ (rad).
    pub fn with_psi(self, reference: Vector3<f64>, psi: f64) -> Self {
        let (h, k, l) = (reference.x, reference.y, reference.z);
        self.with("h2", h)
            .with("k2", k)
            .with("l2", l)
            .with("psi", psi)
    }

    /// A psi mode's reference and ψ (rad); None on another mode.
    pub fn psi(&self) -> Option<(Vector3<f64>, f64)> {
        let p = |name| self.parameters.get(name).copied();
        Some((Vector3::new(p("h2")?, p("k2")?, p("l2")?), p("psi")?))
    }

    /// Mode `name` of `geometry`'s table from `parameters` in hkl's user
    /// units, angles in degrees; a parameter the mode does not take fails.
    pub fn from_degrees(
        geometry: &Geometry,
        name: impl Into<String>,
        parameters: impl IntoIterator<Item = (String, f64)>,
    ) -> Result<Self, InverseError> {
        let name = name.into();
        let table = ModeTable::of(geometry.name())?;
        let parameters = parameters
            .into_iter()
            .map(|(p, value)| {
                let angle = table.parameter(&name, &p)?.angle;
                Ok((p, if angle { value.to_radians() } else { value }))
            })
            .collect::<Result<_, InverseError>>()?;
        Ok(Self { name, parameters })
    }

    /// Its parameters in hkl's user units, angles in degrees.
    pub fn degrees(&self, geometry: &Geometry) -> Result<BTreeMap<String, f64>, InverseError> {
        let table = ModeTable::of(geometry.name())?;
        self.parameters
            .iter()
            .map(|(p, &value)| {
                let angle = table.parameter(&self.name, p)?.angle;
                Ok((p.clone(), if angle { value.to_degrees() } else { value }))
            })
            .collect()
    }

    /// The mode on `geometry`, from its table: every parameter the mode
    /// takes must be given, and no other.
    pub fn build<'g>(&self, geometry: &'g Geometry) -> Result<Mode<'g>, InverseError> {
        let table = ModeTable::of(geometry.name())?;
        let entry = table.entry(&self.name)?;
        for p in self.parameters.keys() {
            table.parameter(&self.name, p)?;
        }
        let values = entry
            .parameters
            .iter()
            .map(|p| {
                self.parameters
                    .get(p.name)
                    .copied()
                    .ok_or_else(|| InverseError::MissingParameter {
                        mode: self.name.clone(),
                        parameter: p.name,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        (entry.build)(geometry, &values)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Matrix3;

    use super::*;
    use crate::presets::{PRESETS, preset};
    use crate::{InverseRequest, Target, Wavelength};

    #[test]
    fn every_preset_has_one_table() {
        let tables: Vec<_> = TABLES.iter().map(|t| t.preset).collect();
        assert_eq!(tables, PRESETS);
    }

    /// golden/hkl_modes.py: per gi factory, hkl's mode names in order, its
    /// default, each mode's parameters with their angle flag, the psi
    /// engine's mode and the engine list's parameters at their values on a
    /// new list. PYFAI and PETRA3 P08 LISA have no factory there: PYFAI is
    /// `hkl-engine-pyfai.c`'s one parameterless mode, LISA has none here.
    /// The one deliberate difference: every MED2+3 v2 mode takes its engine
    /// list's eta_a_rotation, last, where hkl keeps it outside the modes.
    #[test]
    fn tables_are_hkls() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../golden/hkl_modes.json");
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect(path)).unwrap();
        let golden = golden["presets"].as_object().unwrap();
        let listed: Vec<_> = TABLES
            .iter()
            .filter(|t| !t.engine_list.is_empty())
            .map(|t| (t.preset, t.engine_list))
            .collect();
        let v2: &[_] = &[(ETA_A_ROTATION, 0.0)];
        assert_eq!(listed, [("SOLEIL SIXS MED2+3 v2", v2)]);
        for t in &TABLES {
            let g = preset(t.preset).unwrap();
            let Some(hkl) = golden.get(t.preset) else {
                let pyfai = ["pyfai_default"];
                let expected: &[&str] = if t.preset == "PYFAI" { &pyfai } else { &[] };
                assert_eq!(t.names().collect::<Vec<_>>(), expected, "{}", t.preset);
                assert!(t.modes.iter().all(|e| e.parameters.is_empty()));
                assert!(t.engine_list.is_empty() && t.psi.is_none());
                continue;
            };
            let default = t.default_mode().map(|m| m.name);
            assert_eq!(default.as_deref(), hkl["default"].as_str(), "{}", t.preset);
            let list: Vec<_> = hkl["engine_list"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let angle = p["angle"].as_bool().unwrap();
                    let value = p["value"].as_f64().unwrap();
                    let value = if angle { value.to_radians() } else { value };
                    (p["name"].as_str().unwrap(), angle, value)
                })
                .collect();
            let ours = t.engine_list.iter().map(|(p, v)| (p.name, p.angle, *v));
            assert_eq!(ours.collect::<Vec<_>>(), list, "{}", t.preset);
            let list: Vec<_> = t.engine_list.iter().map(|(p, _)| *p).collect();
            let modes: Vec<_> = hkl["modes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    let parameters: Vec<_> = m["parameters"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| (p["name"].as_str().unwrap(), p["angle"].as_bool().unwrap()))
                        .collect();
                    (m["name"].as_str().unwrap(), parameters)
                })
                .collect();
            let ours: Vec<_> = t
                .modes
                .iter()
                .map(|e| {
                    let (own, tail) = e.parameters.split_at(e.parameters.len() - list.len());
                    assert_eq!(tail, list, "{} {}", t.preset, e.name);
                    let parameters = own.iter().map(|p| (p.name, p.angle)).collect();
                    (e.name, parameters)
                })
                .collect::<Vec<(_, Vec<_>)>>();
            assert_eq!(ours, modes, "{}", t.preset);
            let psi = hkl["psi"].as_array().map(|m| {
                assert_eq!(m.len(), 1);
                let names: Vec<_> = m[0]["parameters"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["name"].as_str().unwrap())
                    .collect();
                assert_eq!(names, ["h2", "k2", "l2"]);
                m[0]["name"].as_str().unwrap()
            });
            let built = t.psi.map(|psi| {
                let mode = psi(&g, Vector3::new(0.0, 0.0, 1.0), 0.3).unwrap();
                mode.name().to_owned()
            });
            assert_eq!(built.as_deref(), psi, "{}", t.preset);
        }
    }

    /// Every value of each mode's parameters, angles in rad; eta_a_rotation
    /// at 1, the slit fit.
    fn values(table: &ModeTable, name: &str) -> ModeSpec {
        let values = [0.2, 0.3, 1.0, 0.4, -0.7];
        let parameters = table.parameters(name).unwrap();
        parameters
            .iter()
            .zip(values)
            .fold(ModeSpec::new(name), |s, (p, v)| {
                s.with(p.name, if *p == ETA_A_ROTATION { 1.0 } else { v })
            })
    }

    #[test]
    fn every_mode_builds_on_its_preset_under_its_own_name() {
        for t in &TABLES {
            let g = preset(t.preset).unwrap();
            for name in t.names() {
                let mode = values(t, name).build(&g).unwrap();
                assert_eq!(mode.name(), name, "{}", t.preset);
            }
            if let Some(default) = t.default_mode() {
                default.build(&g).unwrap();
            }
            if let Some(psi) = t.psi {
                psi(&g, Vector3::new(0.0, 0.0, 1.0), 0.3).unwrap();
            }
        }
    }

    #[test]
    fn parameters_are_checked_by_name() {
        let g = preset("K4CV").unwrap();
        let omega = ModeSpec::new("constant_omega");
        assert_eq!(
            omega.build(&g),
            Err(InverseError::MissingParameter {
                mode: "constant_omega".into(),
                parameter: "omega",
            })
        );
        assert_eq!(
            omega.clone().with("omega", 0.3).build(&g),
            k4c::constant_omega(&g, 0.3)
        );
        assert_eq!(
            omega.clone().with("omega", 0.3).with("psi", 0.1).build(&g),
            Err(InverseError::UnknownParameter {
                mode: "constant_omega".into(),
                parameter: "psi".into(),
            })
        );
        // one name, two meanings: E4CV's constant_omega takes no parameter
        let e4cv = preset("E4CV").unwrap();
        assert_eq!(
            ModeSpec::new("constant_omega").build(&e4cv),
            e4c::constant_omega(&e4cv)
        );
    }

    /// MED2+3 v2's eta_a_rotation: at 1 v2 solves as MED2+3 at beta = 0,
    /// eta_a fitted to the slits; at 0 eta_a holds at `current`. Another
    /// value fails, as does none; a new engine list has it at 0.
    #[test]
    fn hkl_42_eta_a_rotation_is_a_switch() {
        let v2 = preset("SOLEIL SIXS MED2+3 v2").unwrap();
        let med23 = preset("SOLEIL SIXS MED2+3").unwrap();
        let spec = |value| ModeSpec::new("mu_fixed").with("eta_a_rotation", value);
        let current = v2
            .values_from_degrees(&[10.0, 20.0, 30.0, 15.0, 40.0])
            .unwrap();
        let solve = |mode: &Mode, current: &[f64]| {
            InverseRequest {
                mode,
                ub: Matrix3::identity() * 1.5,
                wavelength: Wavelength::new(1.54).unwrap(),
                target: Target::Hkl(Vector3::new(0.3, -0.4, 1.1)),
                current,
            }
            .solve()
            .unwrap()
        };
        let fitted = solve(&spec(1.0).build(&v2).unwrap(), &current);
        let beta = [&[0.0][..], &current].concat();
        let med = solve(&ModeSpec::new("mu_fixed").build(&med23).unwrap(), &beta);
        assert!(!fitted.is_empty());
        assert_eq!(fitted.len(), med.len());
        for (s, m) in fitted.iter().zip(&med) {
            assert_eq!(m[0], 0.0);
            let close = s.iter().zip(&m[1..]).all(|(a, b)| (a - b).abs() < 1e-9);
            assert!(close, "{s:?} {m:?}");
        }
        let eta_a = v2.axis_index("eta_a").unwrap();
        let held = solve(&spec(0.0).build(&v2).unwrap(), &current);
        assert!(!held.is_empty());
        assert!(held.iter().all(|s| s[eta_a] == current[eta_a]));
        assert_eq!(
            spec(0.5).build(&v2),
            Err(InverseError::NotSwitch {
                mode: "mu_fixed".into(),
                parameter: "eta_a_rotation",
                value: 0.5,
            })
        );
        assert_eq!(
            ModeSpec::new("mu_fixed").build(&v2),
            Err(InverseError::MissingParameter {
                mode: "mu_fixed".into(),
                parameter: "eta_a_rotation",
            })
        );
        let table = ModeTable::of(v2.name()).unwrap();
        assert_eq!(table.default_mode(), Some(spec(0.0)));
    }

    #[test]
    fn degrees_convert_the_angles_alone() {
        let g = preset("K6C").unwrap();
        let degrees = [("x", 0.0), ("y", 1.0), ("z", 0.0), ("incidence", 30.0)];
        let degrees = degrees.map(|(p, v)| (p.to_owned(), v));
        let spec = ModeSpec::from_degrees(&g, "constant_incidence", degrees.clone()).unwrap();
        assert_eq!(spec.parameters["y"], 1.0);
        assert_eq!(spec.parameters["incidence"], 30f64.to_radians());
        let back = spec.degrees(&g).unwrap();
        assert_eq!(
            back.keys().collect::<Vec<_>>(),
            ["incidence", "x", "y", "z"]
        );
        for (p, v) in degrees {
            assert!((back[&p] - v).abs() < 1e-12, "{p}");
        }
        let unknown = ModeSpec::from_degrees(&g, "bissector_vertical", [("psi".to_owned(), 1.0)]);
        assert_eq!(
            unknown,
            Err(InverseError::UnknownParameter {
                mode: "bissector_vertical".into(),
                parameter: "psi".into(),
            })
        );
    }

    #[test]
    fn errors_name_the_geometry_mode_and_parameter() {
        let custom = Geometry::builder("custom")
            .sample_rotation("omega", [0.0, -1.0, 0.0])
            .detector_rotation("tth", [0.0, -1.0, 0.0])
            .build()
            .unwrap();
        let e = ModeSpec::new("bissector").build(&custom).unwrap_err();
        assert_eq!(
            e.to_string(),
            "geometry \"custom\" is not a preset: it has no named modes"
        );
        let e6c = preset("E6C").unwrap();
        let e = ModeSpec::new("bissector").build(&e6c).unwrap_err();
        assert_eq!(e.to_string(), "E6C has no mode \"bissector\"");
        let e = ModeSpec::new("psi_constant_vertical")
            .with("h2", 1.0)
            .build(&e6c)
            .unwrap_err();
        assert_eq!(
            e.to_string(),
            "mode \"psi_constant_vertical\" needs parameter \"k2\""
        );
        let e = ModeSpec::new("bissector_vertical")
            .with("h2", 1.0)
            .build(&e6c)
            .unwrap_err();
        assert_eq!(
            e.to_string(),
            "mode \"bissector_vertical\" takes no parameter \"h2\""
        );
    }
}
