//! μ from xrt-rs (design §5.1).

use ndarray::Array1;
use xrt_rs::core::XrtError;
use xrt_rs::materials::data::ScatteringTable;
use xrt_rs::materials::material::{Material as XrtMaterial, MaterialKind};

use super::Attenuation;
use crate::units::positive;
use crate::{Energy, Error, PerMetre};

/// xrt-rs takes energies in eV.
const EV_PER_KEV: f64 = 1e3;
/// xrt-rs gives μ in cm⁻¹.
const PER_METRE_PER_PER_CM: f64 = 1e2;

fn material(e: XrtError) -> Error {
    Error::Material {
        reason: e.to_string(),
    }
}

/// A material by its formula and density. Its μ comes from Chantler's
/// total cross-sections, so it holds the scattering losses that
/// photoabsorption alone leaves out of air, Be and Kapton paths at hard
/// X-ray energies. Each element's table ends at 432.9 keV and starts
/// between 4.1 and 10.7 eV; μ outside it is an error.
#[derive(Clone, Debug)]
pub struct Material(XrtMaterial);

impl Material {
    /// `formula` pairs each element symbol with its count per formula unit;
    /// `density` is in g/cm³.
    pub fn new(formula: &[(&str, f64)], density: f64) -> Result<Self, Error> {
        if formula.is_empty() {
            return Err(Error::Material {
                reason: "the formula has no element".to_owned(),
            });
        }
        let elements: Vec<&str> = formula.iter().map(|&(e, _)| e).collect();
        let quantities = formula
            .iter()
            .map(|&(_, q)| positive("element quantity", q))
            .collect::<Result<Vec<_>, _>>()?;
        XrtMaterial::new(
            &elements,
            Some(&quantities),
            positive("density", density)?,
            MaterialKind::Auto,
            None,
            ScatteringTable::ChantlerTotal,
        )
        .map(Self)
        .map_err(material)
    }
}

impl Attenuation for Material {
    fn mu(&self, energy: Energy) -> Result<PerMetre, Error> {
        let ev = Array1::from_elem(1, energy.kev() * EV_PER_KEV);
        let per_cm = self.0.get_absorption_coefficient(&ev).map_err(material)?[0];
        PerMetre::new(per_cm * PER_METRE_PER_PER_CM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEV: [f64; 5] = [5.0, 8.04778, 17.47934, 30.0, 80.0];

    fn si() -> Material {
        Material::new(&[("Si", 1.0)], 2.33).unwrap()
    }

    fn energy(kev: f64) -> Energy {
        Energy::new(kev).unwrap()
    }

    struct Reference {
        formula: &'static [(&'static str, f64)],
        density: f64,
        per_cm: [f64; 5],
    }

    /// μ in cm⁻¹ at `KEV` from Python xrt (~/work/xrt f0a4c3f6),
    /// `Material(elements, quantities, rho=density, table='Chantler
    /// total').get_absorption_coefficient(kev * 1000.0)`. Air is dry air by
    /// atoms (N₂ 78.084 %, O₂ 20.946 %, Ar 0.934 % by volume) at 20 °C. At
    /// 30 keV xrt's photoabsorption-only 'Chantler' gives Be 0.01869 and air
    /// 0.0001697, so the values above 17 keV pin the table.
    const XRT: [Reference; 4] = [
        Reference {
            formula: &[("Si", 1.0)],
            density: 2.33,
            per_cm: [
                552.2217915406823,
                145.41084620727756,
                14.820627382227508,
                3.3111352008558517,
                0.5211363395855511,
            ],
        },
        Reference {
            formula: &[("N", 1.56168), ("O", 0.41892), ("Ar", 0.00934)],
            density: 1.205e-3,
            per_cm: [
                0.04596401818404888,
                0.011355957852069866,
                0.0012726231812035238,
                0.0004315723753970813,
                0.00020163372143107175,
            ],
        },
        Reference {
            formula: &[("Be", 1.0)],
            density: 1.848,
            per_cm: [
                7.382540374899143,
                1.918844211357442,
                0.4584870442996804,
                0.3282985486013282,
                0.259773726206837,
            ],
        },
        Reference {
            formula: &[("C", 22.0), ("H", 10.0), ("N", 2.0), ("O", 5.0)],
            density: 1.43,
            per_cm: [
                34.41078562606681,
                8.287457084015214,
                1.0106902615053601,
                0.4179322317624668,
                0.23928105009942421,
            ],
        },
    ];

    #[test]
    fn mu_matches_python_xrt() {
        for Reference {
            formula,
            density,
            per_cm,
        } in XRT
        {
            let m = Material::new(formula, density).unwrap();
            for (kev, per_cm) in KEV.into_iter().zip(per_cm) {
                let got = m.mu(energy(kev)).unwrap().per_metre();
                let rel = (got / (per_cm * 100.0) - 1.0).abs();
                assert!(
                    rel <= 1e-14,
                    "{formula:?} at {kev} keV: {got} 1/m, rel {rel}"
                );
            }
        }
    }

    /// keV in, m⁻¹ out, against xrt-rs called in its own units.
    #[test]
    fn units_convert_at_the_boundary() {
        let m = si();
        for kev in KEV {
            let per_cm =
                m.0.get_absorption_coefficient(&Array1::from_elem(1, kev * 1000.0))
                    .unwrap()[0];
            assert_eq!(m.mu(energy(kev)).unwrap().per_metre(), per_cm * 100.0);
        }
    }

    #[test]
    fn energy_outside_the_table_has_no_mu() {
        let m = si();
        // Si's table spans 4.98 eV to 432.9 keV
        for kev in [0.004, 500.0] {
            let e = m.mu(energy(kev)).unwrap_err();
            assert!(matches!(e, Error::Material { .. }), "{kev} keV: {e:?}");
        }
    }

    #[test]
    fn rejects_an_invalid_formula() {
        assert!(matches!(
            Material::new(&[("Xx", 1.0)], 1.0),
            Err(Error::Material { .. })
        ));
        assert!(matches!(
            Material::new(&[], 1.0),
            Err(Error::Material { .. })
        ));
        for (formula, density) in [
            (&[("Si", 1.0)][..], 0.0),
            (&[("Si", 1.0)], f64::NAN),
            (&[("Si", 1.0), ("O", -2.0)], 2.2),
            (&[("Si", f64::INFINITY)], 2.2),
        ] {
            assert!(
                matches!(
                    Material::new(formula, density),
                    Err(Error::NotPositive { .. })
                ),
                "{formula:?} {density}"
            );
        }
    }
}
