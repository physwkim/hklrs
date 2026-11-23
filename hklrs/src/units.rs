//! Unit newtypes: a value's unit is its type, never a flag beside it.

use std::f64::consts::TAU;

use crate::Error;

/// h·c in keV·Å.
pub const HC_KEV_ANGSTROM: f64 = 12.398419843320026;

/// Angle in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Rad(pub f64);

/// Angle in degrees; only crosses the API boundary, never used in arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Deg(pub f64);

impl From<Deg> for Rad {
    fn from(d: Deg) -> Rad {
        Rad(d.0.to_radians())
    }
}

impl From<Rad> for Deg {
    fn from(r: Rad) -> Deg {
        Deg(r.0.to_degrees())
    }
}

fn positive(what: &'static str, value: f64) -> Result<f64, Error> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(Error::NotPositive { what, value })
    }
}

/// X-ray wavelength in Å, finite and positive by construction.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Wavelength(f64);

impl Wavelength {
    pub fn new(angstrom: f64) -> Result<Self, Error> {
        positive("wavelength", angstrom).map(Self)
    }

    pub fn angstrom(self) -> f64 {
        self.0
    }

    /// |k| = 2π/λ in Å⁻¹.
    pub fn wavenumber(self) -> f64 {
        TAU / self.0
    }

    pub fn energy(self) -> Energy {
        Energy(HC_KEV_ANGSTROM / self.0)
    }
}

/// Photon energy in keV, finite and positive by construction.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Energy(f64);

impl Energy {
    pub fn new(kev: f64) -> Result<Self, Error> {
        positive("energy", kev).map(Self)
    }

    pub fn kev(self) -> f64 {
        self.0
    }

    pub fn wavelength(self) -> Wavelength {
        Wavelength(HC_KEV_ANGSTROM / self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavelength_rejects_non_positive() {
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Wavelength::new(v).is_err(), "{v}");
        }
        assert!(Energy::new(0.0).is_err());
    }

    #[test]
    fn energy_round_trip() {
        let w = Wavelength::new(1.54).unwrap();
        assert!((w.energy().wavelength().angstrom() - 1.54).abs() < 1e-14);
        assert!(
            (Energy::new(12.398419843320026)
                .unwrap()
                .wavelength()
                .angstrom()
                - 1.0)
                .abs()
                < 1e-15
        );
    }

    #[test]
    fn deg_rad() {
        assert_eq!(Rad::from(Deg(180.0)).0, std::f64::consts::PI);
        assert_eq!(Deg::from(Rad(std::f64::consts::FRAC_PI_2)).0, 90.0);
    }
}
