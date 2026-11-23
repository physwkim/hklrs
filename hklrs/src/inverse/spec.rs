//! Modes by name, for callers that hold a mode as data (a Python binding, a
//! saved configuration) and build it against a geometry when they solve.

use nalgebra::Vector3;

use super::{InverseError, Mode, e4c, e6c};
use crate::Geometry;

/// What a mode takes besides the geometry.
#[derive(Clone, Copy)]
enum Build {
    Plain(fn(&Geometry) -> Result<Mode<'_>, InverseError>),
    Hkl2(fn(&Geometry, Vector3<f64>) -> Result<Mode<'_>, InverseError>),
    Psi(fn(&Geometry, Vector3<f64>, f64) -> Result<Mode<'_>, InverseError>),
}

/// hkl's "hkl" engine modes for E4C and E6C geometries.
const MODES: [(&str, Build); 19] = [
    ("bissector", Build::Plain(e4c::bissector)),
    ("constant_omega", Build::Plain(e4c::constant_omega)),
    ("constant_chi", Build::Plain(e4c::constant_chi)),
    ("constant_phi", Build::Plain(e4c::constant_phi)),
    ("double_diffraction", Build::Hkl2(e4c::double_diffraction)),
    ("psi_constant", Build::Psi(e4c::psi_constant)),
    ("bissector_vertical", Build::Plain(e6c::bissector_vertical)),
    (
        "constant_omega_vertical",
        Build::Plain(e6c::constant_omega_vertical),
    ),
    (
        "constant_chi_vertical",
        Build::Plain(e6c::constant_chi_vertical),
    ),
    (
        "constant_phi_vertical",
        Build::Plain(e6c::constant_phi_vertical),
    ),
    (
        "lifting_detector_phi",
        Build::Plain(e6c::lifting_detector_phi),
    ),
    (
        "lifting_detector_omega",
        Build::Plain(e6c::lifting_detector_omega),
    ),
    (
        "lifting_detector_mu",
        Build::Plain(e6c::lifting_detector_mu),
    ),
    (
        "double_diffraction_vertical",
        Build::Hkl2(e6c::double_diffraction_vertical),
    ),
    (
        "bissector_horizontal",
        Build::Plain(e6c::bissector_horizontal),
    ),
    (
        "double_diffraction_horizontal",
        Build::Hkl2(e6c::double_diffraction_horizontal),
    ),
    (
        "psi_constant_vertical",
        Build::Psi(e6c::psi_constant_vertical),
    ),
    (
        "psi_constant_horizontal",
        Build::Psi(e6c::psi_constant_horizontal),
    ),
    (
        "constant_mu_horizontal",
        Build::Plain(e6c::constant_mu_horizontal),
    ),
];

/// A mode by name and its parameters. A parameter the named mode does not
/// take is ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeSpec {
    pub name: String,
    /// Miller indices also held in diffraction: the double_diffraction modes.
    pub hkl2: Option<Vector3<f64>>,
    /// Reference (Miller indices) and ψ (rad): the psi_constant modes.
    pub psi: Option<(Vector3<f64>, f64)>,
}

impl ModeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hkl2: None,
            psi: None,
        }
    }

    /// Every name [`ModeSpec::build`] knows; a geometry without a mode's
    /// axes still refuses it at build.
    pub fn names() -> impl Iterator<Item = &'static str> {
        MODES.iter().map(|&(name, _)| name)
    }

    pub fn build<'g>(&self, geometry: &'g Geometry) -> Result<Mode<'g>, InverseError> {
        let name = self.name.as_str();
        let (_, build) = MODES
            .iter()
            .find(|&&(n, _)| n == name)
            .ok_or_else(|| InverseError::UnknownMode { name: name.into() })?;
        let missing = |needs| InverseError::ModeParameter {
            mode: name.into(),
            needs,
        };
        match *build {
            Build::Plain(f) => f(geometry),
            Build::Hkl2(f) => f(geometry, self.hkl2.ok_or_else(|| missing("hkl2"))?),
            Build::Psi(f) => {
                let (reference, psi) = self.psi.ok_or_else(|| missing("reference and psi"))?;
                f(geometry, reference, psi)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::preset;

    /// The names `PyMode::build` in hklrs-py takes, as of this table's
    /// introduction.
    const PY_NAMES: [&str; 19] = [
        "bissector",
        "constant_omega",
        "constant_chi",
        "constant_phi",
        "double_diffraction",
        "psi_constant",
        "bissector_vertical",
        "constant_omega_vertical",
        "constant_chi_vertical",
        "constant_phi_vertical",
        "lifting_detector_phi",
        "lifting_detector_omega",
        "lifting_detector_mu",
        "double_diffraction_vertical",
        "bissector_horizontal",
        "double_diffraction_horizontal",
        "psi_constant_vertical",
        "psi_constant_horizontal",
        "constant_mu_horizontal",
    ];

    /// `PyMode::build`'s mapping, written out apart from [`MODES`].
    fn direct<'g>(
        g: &'g Geometry,
        name: &str,
        hkl2: Vector3<f64>,
        reference: Vector3<f64>,
        psi: f64,
    ) -> Result<Mode<'g>, InverseError> {
        match name {
            "bissector" => e4c::bissector(g),
            "constant_omega" => e4c::constant_omega(g),
            "constant_chi" => e4c::constant_chi(g),
            "constant_phi" => e4c::constant_phi(g),
            "double_diffraction" => e4c::double_diffraction(g, hkl2),
            "psi_constant" => e4c::psi_constant(g, reference, psi),
            "bissector_vertical" => e6c::bissector_vertical(g),
            "constant_omega_vertical" => e6c::constant_omega_vertical(g),
            "constant_chi_vertical" => e6c::constant_chi_vertical(g),
            "constant_phi_vertical" => e6c::constant_phi_vertical(g),
            "lifting_detector_phi" => e6c::lifting_detector_phi(g),
            "lifting_detector_omega" => e6c::lifting_detector_omega(g),
            "lifting_detector_mu" => e6c::lifting_detector_mu(g),
            "double_diffraction_vertical" => e6c::double_diffraction_vertical(g, hkl2),
            "bissector_horizontal" => e6c::bissector_horizontal(g),
            "double_diffraction_horizontal" => e6c::double_diffraction_horizontal(g, hkl2),
            "psi_constant_vertical" => e6c::psi_constant_vertical(g, reference, psi),
            "psi_constant_horizontal" => e6c::psi_constant_horizontal(g, reference, psi),
            "constant_mu_horizontal" => e6c::constant_mu_horizontal(g),
            _ => unreachable!("{name}"),
        }
    }

    #[test]
    fn names_are_the_ones_py_mode_takes() {
        assert_eq!(ModeSpec::names().collect::<Vec<_>>(), PY_NAMES);
    }

    #[test]
    fn every_name_builds_what_its_constructor_builds() {
        let hkl2 = Vector3::new(1.0, 1.0, 0.0);
        let reference = Vector3::new(0.0, 0.0, 1.0);
        let psi = 0.3;
        // each geometry builds its own engine's modes and refuses the other's
        let (e4c, e6c) = PY_NAMES.split_at(6);
        for (geometry, own) in [("E4CV", e4c), ("E6C", e6c)] {
            let g = preset(geometry).unwrap();
            for name in PY_NAMES {
                let spec = ModeSpec {
                    name: name.into(),
                    hkl2: Some(hkl2),
                    psi: Some((reference, psi)),
                };
                let built = spec.build(&g);
                assert_eq!(
                    built,
                    direct(&g, name, hkl2, reference, psi),
                    "{geometry} {name}"
                );
                assert_eq!(built.is_ok(), own.contains(&name), "{geometry} {name}");
                if let Ok(mode) = built {
                    assert_eq!(mode.name(), name);
                }
            }
        }
    }

    #[test]
    fn errors_name_the_mode_and_what_it_lacks() {
        let g = preset("E6C").unwrap();
        assert_eq!(
            ModeSpec::new("bisector").build(&g),
            Err(InverseError::UnknownMode {
                name: "bisector".into()
            })
        );
        for (name, needs) in [
            ("double_diffraction_vertical", "hkl2"),
            ("psi_constant_horizontal", "reference and psi"),
        ] {
            let e = ModeSpec::new(name).build(&g).unwrap_err();
            assert_eq!(
                e,
                InverseError::ModeParameter {
                    mode: name.into(),
                    needs
                }
            );
            assert_eq!(e.to_string(), format!("mode {name:?} needs {needs}"));
        }
    }
}
