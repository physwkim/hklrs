//! hkl's PYFAI "hkl" engine mode (`hkl-engine-pyfai.c`), for any geometry
//! with detector axes rot3, rot2, rot1 and no sample axis.

use super::{InverseError, Mode};
use crate::Geometry;

/// hkl's only mode: the three detector rotations alone. With no sample axis
/// Q is fixed in the lab and k̂f on the Ewald sphere has two degrees of
/// freedom for three axes, so it never isolates a solution
/// ([`InverseError::Degenerate`]).
pub fn pyfai_default(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(
        geometry,
        "pyfai_default",
        &["rot3", "rot2", "rot1"],
        |_| vec![],
    )
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Rotation3, Vector3};

    use super::*;
    use crate::presets::preset;
    use crate::{InverseRequest, Target, Wavelength};

    #[test]
    fn pyfai_default_never_isolates_a_solution() {
        let g = preset("PYFAI").unwrap();
        let mode = pyfai_default(&g).unwrap();
        let wavelength = Wavelength::new(1.54).unwrap();
        let k = std::f64::consts::TAU / 1.54;
        let ki = Vector3::x();
        let kf = Rotation3::from_axis_angle(&-Vector3::y_axis(), 0.3) * ki;
        let solve = |q: Vector3<f64>| {
            InverseRequest {
                mode: &mode,
                ub: Matrix3::identity(),
                wavelength,
                target: Target::QSample(q * k),
                current: &[0.0; 6],
            }
            .solve()
        };
        assert_eq!(solve(kf - ki), Err(InverseError::Degenerate));
    }
}
