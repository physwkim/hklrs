//! hkl's PYFAI "hkl" engine mode (`hkl-engine-pyfai.c`), for any geometry
//! with detector axes rot3, rot2, rot1 and no sample axis.

use super::{InverseError, Mode};
use crate::Geometry;

/// hkl's only mode, rot3 held: with no sample axis the target fixes k̂f,
/// whose two degrees of freedom rot2 and rot1 take. hkl writes rot3 too
/// (HKL-41), three axes for two equations, so it never isolates a solution.
pub fn pyfai_default(geometry: &Geometry) -> Result<Mode<'_>, InverseError> {
    Mode::named(geometry, "pyfai_default", &["rot2", "rot1"], |_| vec![])
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use nalgebra::{Matrix3, Vector3};

    use super::*;
    use crate::presets::preset;
    use crate::{InverseRequest, Target, Wavelength};

    /// From rot3 = 0 and rot3 ≠ 0 alike, the k̂f of (rot2, rot1) = (0.3,
    /// −0.5) solves to it and its mirror (rot2 + π, π − rot1), every other
    /// axis held; hkl's three written axes fail the count. A target off the
    /// Ewald sphere fails, naming its Q and the miss.
    #[test]
    fn hkl_41_pyfai_default_holds_rot3() {
        let g = preset("PYFAI").unwrap();
        let mode = pyfai_default(&g).unwrap();
        let hkl = Mode::new(&g, "hkl", &[0, 1, 2], vec![]).unwrap();
        let wavelength = Wavelength::new(1.54).unwrap();
        let k = wavelength.wavenumber();
        let beam = g.beam_direction().into_inner();
        let angle = |a: f64, b: f64| (a - b + PI).rem_euclid(2.0 * PI) - PI;
        for rot3 in [0.0, 0.7] {
            let poses = g.poses(&[rot3, 0.3, -0.5, 0.0, 0.0, 0.0]).unwrap();
            let q = (poses.detector.rotation * beam - poses.source * beam) * k;
            let current = [rot3, 0.1, -0.2, 0.01, 0.02, 0.5];
            let solve = |mode, q| {
                InverseRequest {
                    mode,
                    ub: Matrix3::identity(),
                    wavelength,
                    target: Target::QSample(q),
                    current: &current,
                }
                .solve()
            };
            let solutions = solve(&mode, q).unwrap();
            let expected = [[0.3, -0.5], [0.3 + PI, PI + 0.5]];
            assert_eq!(solutions.len(), expected.len(), "{solutions:?}");
            for (s, e) in solutions.iter().zip(expected) {
                assert_eq!([s[0], s[3], s[4], s[5]], [rot3, 0.01, 0.02, 0.5]);
                let near = angle(s[1], e[0]).abs() < 1e-9 && angle(s[2], e[1]).abs() < 1e-9;
                assert!(near, "rot3 {rot3}: {s:?}");
            }
            let three = solve(&hkl, q);
            let count = InverseError::EquationCount {
                written: 3,
                equations: 2,
            };
            assert_eq!(three, Err(count));
            let off = q * 1.01;
            let Err(InverseError::OffEwaldSphere { q: named, miss }) = solve(&mode, off) else {
                panic!("rot3 {rot3}: an off-sphere target solved");
            };
            assert_eq!(Vector3::from(named), off);
            let expected = (poses.source * beam * k + off).norm() - k;
            assert!((miss - expected).abs() < 1e-12, "{miss} {expected}");
        }
    }
}
