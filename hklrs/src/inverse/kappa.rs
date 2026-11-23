//! hkl's "eulerians" pseudo axes (`hkl-pseudoaxis-common-eulerians.c`): a
//! kappa position (komega, kappa, kphi) and the eulerian (omega, chi, phi)
//! that gives the same sample rotation, for hkl's kappa arm (komega and kphi
//! about −y, kappa about `presets::kappa_axis(α)`) against hkl's eulerian
//! cradle (omega, chi, phi about −y, x, −y). Each side has two images of a
//! rotation; [`Branch`] picks one.

use std::f64::consts::{FRAC_PI_2, PI};

use super::{InverseError, finite, wrap};
use crate::Rad;

/// hkl's "solutions" parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    /// chi and kappa of the same sign; hkl `solutions = 1`, its default.
    Same,
    /// chi and kappa of opposite signs; hkl `solutions = 0`.
    Opposite,
}

/// A kappa arm tilted by α ∈ (0, π/2] from the komega axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KappaArm {
    alpha: f64,
}

/// The representative in [−π, π], `a` itself when it is one, so a given
/// −π keeps its sign (gsl_sf_angle_restrict_symm).
fn restrict(a: f64) -> f64 {
    if a.abs() <= PI { a } else { wrap(a) }
}

impl KappaArm {
    pub fn new(alpha: Rad) -> Result<Self, InverseError> {
        if alpha.0 > 0.0 && alpha.0 <= FRAC_PI_2 {
            Ok(Self { alpha: alpha.0 })
        } else {
            Err(InverseError::KappaAlpha { alpha: alpha.0 })
        }
    }

    pub fn alpha(&self) -> Rad {
        Rad(self.alpha)
    }

    /// 2α: the largest |chi| the arm reaches.
    pub fn reach(&self) -> f64 {
        2.0 * self.alpha
    }

    /// (omega, chi, phi) of (komega, kappa, kphi); omega and phi wrapped to
    /// (−π, π], |chi| ≤ 2α (hkl `kappa_to_eulerian`).
    pub fn to_eulerian(&self, [komega, kappa, kphi]: [f64; 3], branch: Branch) -> [f64; 3] {
        let half = restrict(kappa) / 2.0;
        let p = (half.tan() * self.alpha.cos()).atan();
        let chi = 2.0 * (half.sin() * self.alpha.sin()).asin();
        match branch {
            Branch::Same => [
                wrap(komega + p - FRAC_PI_2),
                chi,
                wrap(kphi + p + FRAC_PI_2),
            ],
            Branch::Opposite => [
                wrap(komega + p + FRAC_PI_2),
                -chi,
                wrap(kphi + p - FRAC_PI_2),
            ],
        }
    }

    /// (komega, kappa, kphi) of (omega, chi, phi), komega and kphi wrapped
    /// to (−π, π] (hkl `eulerian_to_kappa`). chi counts modulo 2π (HKL-19)
    /// and must be within ±2α.
    pub fn to_kappa(&self, eulerian: [f64; 3], branch: Branch) -> Result<[f64; 3], InverseError> {
        finite(&eulerian, "eulerian angles")?;
        let [omega, chi, phi] = eulerian;
        let chi = restrict(chi);
        let reach = self.reach();
        if chi.abs() > reach {
            return Err(InverseError::ChiOutOfReach { chi, reach });
        }
        // |chi| ≤ 2α ≤ π bounds the ratio by 1; at |chi| = 2α rounding can
        // step past it.
        let kappa = 2.0
            * ((chi / 2.0).sin() / self.alpha.sin())
                .clamp(-1.0, 1.0)
                .asin();
        // p of this kappa, not hkl's asin(tan(chi/2)/tan α): near |chi| = 2α
        // kappa carries ~1e-8 rad of the ill-conditioning either way, and a p
        // from the same kappa keeps the round trip to these eulerians
        // consistent; a second asin would not.
        let p = ((kappa / 2.0).tan() * self.alpha.cos()).atan();
        Ok(match branch {
            Branch::Same => [
                wrap(omega - p + FRAC_PI_2),
                kappa,
                wrap(phi - p - FRAC_PI_2),
            ],
            Branch::Opposite => [
                wrap(omega + p - FRAC_PI_2),
                -kappa,
                wrap(phi + p + FRAC_PI_2),
            ],
        })
    }

    /// The other kappa position with the same sample rotation (hkl
    /// `kappa_2_kappap`): `to_kappa` of either eulerian image under the
    /// other branch. komega and kphi wrapped to (−π, π]; kappa is the
    /// negated [−π, π] representative, so ±π maps to ∓π and the branch
    /// sign rule holds on the twin too.
    pub fn twin(&self, [komega, kappa, kphi]: [f64; 3]) -> [f64; 3] {
        let kappa = restrict(kappa);
        let p = ((kappa / 2.0).tan() * self.alpha.cos()).atan();
        [
            wrap(komega + 2.0 * p - PI),
            -kappa,
            wrap(kphi + 2.0 * p + PI),
        ]
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use crate::Geometry;
    use crate::presets::kappa_axis;

    const MY: [f64; 3] = [0.0, -1.0, 0.0];
    const GRID: [f64; 7] = [-PI, -2.0, -0.3, 0.0, 0.7, 2.5, PI];

    fn sample(g: &Geometry, values: [f64; 3]) -> nalgebra::Matrix3<f64> {
        let pose = g.poses(&values).unwrap().sample;
        pose.rotation.to_rotation_matrix().into_inner()
    }

    fn kappa_geometry(axis: [f64; 3]) -> Geometry {
        Geometry::builder("kappa")
            .sample_rotation("komega", MY)
            .sample_rotation("kappa", axis)
            .sample_rotation("kphi", MY)
            .build()
            .unwrap()
    }

    fn same(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(&b).all(|(x, y)| wrap(x - y).abs() <= 1e-12)
    }

    #[test]
    fn eulerian_and_twin_give_the_kappa_rotation() {
        let eulerian = Geometry::builder("eulerian")
            .sample_rotation("omega", MY)
            .sample_rotation("chi", [1.0, 0.0, 0.0])
            .sample_rotation("phi", MY)
            .build()
            .unwrap();
        for alpha in [0.2, 50f64.to_radians(), 1.4, FRAC_PI_2] {
            let arm = KappaArm::new(Rad(alpha)).unwrap();
            let kappa = kappa_geometry(kappa_axis(Rad(alpha)));
            for (&ko, (&ka, &kp)) in GRID
                .iter()
                .zip(GRID.iter().rev().zip(GRID.iter().cycle().skip(2)))
            {
                let k = [ko, ka, kp];
                let twin = arm.twin(k);
                let r = sample(&kappa, k);
                assert!((sample(&kappa, twin) - r).amax() <= 1e-12, "twin {k:?}");
                for (b, other) in [
                    (Branch::Same, Branch::Opposite),
                    (Branch::Opposite, Branch::Same),
                ] {
                    let e = arm.to_eulerian(k, b);
                    assert!(e[1].abs() <= arm.reach(), "{alpha} {k:?} {b:?}");
                    assert!(
                        (sample(&eulerian, e) - r).amax() <= 1e-12,
                        "{alpha} {k:?} {b:?}"
                    );
                    assert!(same(arm.to_kappa(e, b).unwrap(), k), "{alpha} {k:?} {b:?}");
                    assert!(same(arm.to_eulerian(twin, other), e), "{alpha} {k:?} {b:?}");
                }
            }
        }
    }

    /// A kappa turning left-handed about its axis (xrayutilities `k+`,
    /// XU-3) is the right-handed turn about the reversed axis; the map does
    /// not hold for it.
    #[test]
    fn left_handed_kappa_breaks_the_map() {
        let arm = KappaArm::new(Rad(50f64.to_radians())).unwrap();
        let axis = kappa_axis(arm.alpha()).map(|c| -c);
        let mirrored = kappa_geometry(axis);
        let right = kappa_geometry(kappa_axis(arm.alpha()));
        let k = [0.3, 1.1, -0.4];
        assert!((sample(&mirrored, k) - sample(&right, k)).amax() > 0.1);
    }

    #[test]
    fn chi_reach_is_two_alpha() {
        let arm = KappaArm::new(Rad(0.7)).unwrap();
        let edge = arm.to_kappa([0.1, 1.4, 0.2], Branch::Same).unwrap();
        assert_eq!(edge[1], PI);
        let past = 1.4f64.next_up();
        assert_eq!(
            arm.to_kappa([0.1, -past, 0.2], Branch::Same),
            Err(InverseError::ChiOutOfReach {
                chi: -past,
                reach: 1.4
            })
        );
        let wrapped = arm
            .to_kappa([0.1, 1.0 + TAU, 0.2], Branch::Opposite)
            .unwrap();
        assert!(same(
            wrapped,
            arm.to_kappa([0.1, 1.0, 0.2], Branch::Opposite).unwrap()
        ));
        assert_eq!(
            arm.to_kappa([0.1, f64::NAN, 0.2], Branch::Same),
            Err(InverseError::NonFinite {
                what: "eulerian angles"
            })
        );
    }

    #[test]
    fn alpha_is_in_zero_to_half_pi() {
        for alpha in [0.0, -0.1, FRAC_PI_2.next_up(), f64::NAN] {
            assert!(matches!(
                KappaArm::new(Rad(alpha)),
                Err(InverseError::KappaAlpha { .. })
            ));
        }
        assert!(KappaArm::new(Rad(FRAC_PI_2)).is_ok());
    }
}
