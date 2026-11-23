//! Absorption weights (design §5.1). The correction divides the intensity
//! by the weight.

use crate::{Energy, PerMetre};

/// Linear attenuation of a material at a photon energy. The core holds no
/// material tables; a constant μ implements it directly.
pub trait Attenuation {
    fn mu(&self, energy: Energy) -> PerMetre;
}

impl Attenuation for PerMetre {
    fn mu(&self, _energy: Energy) -> PerMetre {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_mu_ignores_energy() {
        let mu = PerMetre::new(1234.5).unwrap();
        for kev in [1.0, 8.048, 100.0] {
            assert_eq!(mu.mu(Energy::new(kev).unwrap()), mu);
        }
    }
}
