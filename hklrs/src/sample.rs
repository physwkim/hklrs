//! Sample: lattice plus orientation U.

use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};

use crate::{Error, Geometry, Lattice, Wavelength, scattering};

/// Smallest sine of the angle between two reflections, in index space and in
/// the measured sample frame, below which they do not define a plane.
pub const PARALLEL_SIN: f64 = 1e-6;

/// Miller indices and the scattering vector measured for them in the sample
/// frame (Å⁻¹; only its direction is used).
#[derive(Clone, Debug, PartialEq)]
pub struct Reflection {
    hkl: Vector3<f64>,
    q_sample: Vector3<f64>,
}

impl Reflection {
    pub fn new(hkl: Vector3<f64>, q_sample: Vector3<f64>) -> Result<Self, Error> {
        for (what, v) in [("hkl", &hkl), ("q_sample", &q_sample)] {
            if !(v.iter().all(|x| x.is_finite()) && v.norm() > 0.0) {
                return Err(Error::ZeroVector { what: what.into() });
            }
        }
        Ok(Self { hkl, q_sample })
    }

    /// Reflection found at `values` with the scattered beam through `point`
    /// (detector-arm frame, metres).
    pub fn measure(
        hkl: Vector3<f64>,
        geometry: &Geometry,
        values: &[f64],
        wavelength: Wavelength,
        point: &Point3<f64>,
    ) -> Result<Self, Error> {
        Self::new(
            hkl,
            scattering(geometry, values, wavelength, point)?.q_sample,
        )
    }

    pub fn hkl(&self) -> &Vector3<f64> {
        &self.hkl
    }

    pub fn q_sample(&self) -> &Vector3<f64> {
        &self.q_sample
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Orientation {
    /// U given directly.
    Manual { u: UnitQuaternion<f64> },
    /// U from two reflections (Busing & Levy 1967): the primary is exact,
    /// the secondary fixes the rotation about it. U is derived from the
    /// current B on every call, so a lattice change cannot leave it stale.
    BusingLevy {
        primary: Reflection,
        secondary: Reflection,
    },
}

fn sin_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.cross(b).norm() / (a.norm() * b.norm())
}

/// Right-handed orthonormal triad (â, (a×b)×â normalised, a×b normalised) as
/// columns.
fn triad(a: &Vector3<f64>, b: &Vector3<f64>) -> Matrix3<f64> {
    let t1 = a.normalize();
    let t3 = a.cross(b).normalize();
    Matrix3::from_columns(&[t1, t3.cross(&t1), t3])
}

impl Orientation {
    /// Manual U from a matrix that must already be a proper rotation
    /// (‖UᵀU − I‖∞ ≤ 1e-9, det > 0); nothing is silently re-orthogonalised.
    pub fn from_u_matrix(u: &Matrix3<f64>) -> Result<Self, Error> {
        let deviation = (u.transpose() * u - Matrix3::identity()).abs().max();
        if !(deviation <= 1e-9 && u.determinant() > 0.0) {
            return Err(Error::NotRotation { deviation });
        }
        let r = Rotation3::from_matrix_unchecked(*u);
        Ok(Self::Manual {
            u: UnitQuaternion::from_rotation_matrix(&r),
        })
    }

    /// Busing-Levy orientation. Rejects pairs that are parallel in index
    /// space or in the measured sample frame; B is invertible for every
    /// valid lattice, so a pair accepted here is non-parallel for any lattice.
    pub fn busing_levy(primary: Reflection, secondary: Reflection) -> Result<Self, Error> {
        for (space, a, b) in [
            ("index space", &primary.hkl, &secondary.hkl),
            ("the sample frame", &primary.q_sample, &secondary.q_sample),
        ] {
            let sin = sin_between(a, b);
            // NaN when a cross product overflows
            if sin.is_nan() || sin < PARALLEL_SIN {
                return Err(Error::ParallelReflections { space, sin });
            }
        }
        Ok(Self::BusingLevy { primary, secondary })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub name: String,
    pub lattice: Lattice,
    pub orientation: Orientation,
}

impl Sample {
    pub fn new(name: &str, lattice: Lattice, orientation: Orientation) -> Self {
        Self {
            name: name.to_owned(),
            lattice,
            orientation,
        }
    }

    pub fn u(&self) -> UnitQuaternion<f64> {
        match &self.orientation {
            Orientation::Manual { u } => *u,
            Orientation::BusingLevy { primary, secondary } => {
                let b = self.lattice.b_matrix();
                let crystal = triad(&(b * primary.hkl), &(b * secondary.hkl));
                let measured = triad(&primary.q_sample, &secondary.q_sample);
                // both triads are right-handed orthonormal, so the product is
                // a proper rotation up to rounding
                let r = Rotation3::from_matrix_unchecked(measured * crystal.transpose());
                UnitQuaternion::from_rotation_matrix(&r)
            }
        }
    }

    /// UB (Å⁻¹, includes 2π), derived on every call.
    pub fn ub(&self) -> Matrix3<f64> {
        self.u().to_rotation_matrix().into_inner() * self.lattice.b_matrix()
    }

    /// hkl of a scattering vector given in the sample frame:
    /// B⁻¹ Uᵀ q, with B upper triangular and its diagonal positive for every
    /// valid lattice.
    pub fn hkl(&self, q_sample: &Vector3<f64>) -> Vector3<f64> {
        let q_crystal = self.u().inverse() * q_sample;
        self.lattice
            .b_matrix()
            .solve_upper_triangular(&q_crystal)
            .expect("B has a positive diagonal for every valid Lattice")
    }

    /// Surface axes as columns in the sample frame, from the current UB: z
    /// along UB·`normal` (Miller indices of the surface plane), x along the
    /// part of UB·`in_plane` (Miller indices) within that plane, y = z × x.
    /// z is the lattice-plane normal; a miscut surface's own normal differs
    /// from it by the miscut, which UB does not hold.
    pub fn surface_frame(
        &self,
        normal: &Vector3<f64>,
        in_plane: &Vector3<f64>,
    ) -> Result<Rotation3<f64>, Error> {
        for (what, v) in [("surface normal hkl", normal), ("in-plane hkl", in_plane)] {
            if !(v.iter().all(|x| x.is_finite()) && v.norm() > 0.0) {
                return Err(Error::ZeroVector { what: what.into() });
            }
        }
        let ub = self.ub();
        let (n, r) = (ub * normal, ub * in_plane);
        let sin = sin_between(&n, &r);
        // NaN when a cross product overflows
        if sin.is_nan() || sin < PARALLEL_SIN {
            return Err(Error::ParallelDirections {
                what: "surface normal and in-plane hkl",
            });
        }
        // triad's columns are (z, x, y); a cyclic shift keeps it proper
        let t = triad(&n, &r);
        Ok(Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[
            t.column(1).into_owned(),
            t.column(2).into_owned(),
            t.column(0).into_owned(),
        ])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triclinic() -> Lattice {
        Lattice::from_degrees(4.1, 5.3, 6.7, 82.0, 97.0, 101.0).unwrap()
    }

    fn u_true() -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(0.4, -1.1, 2.3)
    }

    /// Reflection measured on a sample with orientation `u`, q scaled by
    /// `scale` (only the direction may matter).
    fn measured(l: &Lattice, u: &UnitQuaternion<f64>, h: [f64; 3], scale: f64) -> Reflection {
        let h = Vector3::from(h);
        Reflection::new(h, scale * (u * (l.b_matrix() * h))).unwrap()
    }

    #[test]
    fn busing_levy_recovers_u() {
        let (l, u) = (triclinic(), u_true());
        let o = Orientation::busing_levy(
            measured(&l, &u, [1.0, 2.0, 3.0], 0.7),
            measured(&l, &u, [-2.0, 1.0, 0.0], 1.3),
        )
        .unwrap();
        let s = Sample::new("t", l, o);
        assert!(s.u().angle_to(&u) < 1e-12, "{}", s.u().angle_to(&u));
        let h = Vector3::new(0.3, -1.0, 2.5);
        assert!((s.hkl(&(u * (l.b_matrix() * h))) - h).abs().max() < 1e-12);
    }

    #[test]
    fn primary_is_exact_secondary_sets_the_plane() {
        // the secondary direction is off by a rotation about the primary
        // normal; the primary must still map exactly onto its q
        let (l, u) = (triclinic(), u_true());
        let p = measured(&l, &u, [1.0, 0.0, 0.0], 1.0);
        let mut q2 = u * (l.b_matrix() * Vector3::new(0.0, 1.0, 0.0));
        q2 += 0.05 * q2.norm() * p.q_sample().normalize();
        let s = Sample::new(
            "t",
            l,
            Orientation::busing_levy(p.clone(), Reflection::new(Vector3::y(), q2).unwrap())
                .unwrap(),
        );
        let along = (s.ub() * p.hkl()).normalize();
        assert!((along - p.q_sample().normalize()).abs().max() < 1e-14);
    }

    #[test]
    fn lattice_change_rederives_u() {
        // same reflections, new lattice: UB·h1 still lies along q1
        let (l, u) = (triclinic(), u_true());
        let o = Orientation::busing_levy(
            measured(&l, &u, [1.0, 1.0, 0.0], 1.0),
            measured(&l, &u, [0.0, 0.0, 1.0], 1.0),
        )
        .unwrap();
        let mut s = Sample::new("t", l, o);
        s.lattice = Lattice::from_degrees(4.0, 5.5, 6.5, 85.0, 95.0, 100.0).unwrap();
        let Orientation::BusingLevy { primary, .. } = &s.orientation else {
            unreachable!()
        };
        let along = (s.ub() * primary.hkl()).normalize();
        assert!((along - primary.q_sample().normalize()).abs().max() < 1e-14);
    }

    #[test]
    fn measure_takes_q_from_the_angles_and_rejects_the_direct_beam() {
        let geo = Geometry::builder("t")
            .sample_rotation("om", [0.0, 0.0, 1.0])
            .detector_rotation("tth", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        let wl = Wavelength::new(1.5).unwrap();
        let p = Point3::new(1.0, 0.0, 0.0);
        let r = Reflection::measure(Vector3::x(), &geo, &[0.2, 0.6], wl, &p).unwrap();
        let s = scattering(&geo, &[0.2, 0.6], wl, &p).unwrap();
        assert_eq!(r.q_sample(), &s.q_sample);
        assert!(matches!(
            Reflection::measure(Vector3::x(), &geo, &[0.2, 0.0], wl, &p),
            Err(Error::ZeroVector { .. })
        ));
    }

    #[test]
    fn surface_frame_puts_the_normal_on_z_and_in_plane_on_x() {
        let (l, u) = (triclinic(), u_true());
        let s = Sample::new("t", l, Orientation::Manual { u });
        // in_plane is not perpendicular to normal in the sample frame
        let (normal, in_plane) = (Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 0.0, 0.0));
        let f = s.surface_frame(&normal, &in_plane).unwrap();
        let m = f.matrix();
        assert!((m.transpose() * m - Matrix3::identity()).abs().max() < 1e-14);
        assert!((m.determinant() - 1.0).abs() < 1e-14);
        let n = f.inverse() * (s.ub() * normal);
        assert!(n.x.abs() < 1e-14 && n.y.abs() < 1e-14 && n.z > 0.0, "{n}");
        let r = f.inverse() * (s.ub() * in_plane);
        assert!(r.y.abs() < 1e-14 && r.x > 0.0, "{r}");
        // only the directions count
        let f2 = s.surface_frame(&(2.0 * normal), &(0.5 * in_plane)).unwrap();
        assert!(f2.angle_to(&f) < 1e-14);
    }

    #[test]
    fn surface_frame_of_a_vertical_surface_is_binoculars_rx() {
        // binoculars QxQyQz on a VERTICAL surface (Python binoculars
        // 93956b2, sixs.py:376-377) takes frame [[1,0,0],[0,0,1],[0,-1,0]]
        // = Rx(−π/2): U = I, surface (010), x along (100)
        let cubic = Lattice::from_degrees(3.92, 3.92, 3.92, 90.0, 90.0, 90.0).unwrap();
        let s = Sample::new(
            "t",
            cubic,
            Orientation::Manual {
                u: UnitQuaternion::identity(),
            },
        );
        let f = s.surface_frame(&Vector3::y(), &Vector3::x()).unwrap();
        let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);
        assert!((f.matrix() - rx).abs().max() < 1e-15, "{}", f.matrix());
    }

    #[test]
    fn surface_frame_rejects_parallel_and_zero_hkl() {
        let s = Sample::new("t", triclinic(), Orientation::Manual { u: u_true() });
        let n = Vector3::new(1.0, 1.0, 1.0);
        assert!(matches!(
            s.surface_frame(&n, &(-2.0 * n)),
            Err(Error::ParallelDirections { .. })
        ));
        assert!(matches!(
            s.surface_frame(&n, &Vector3::zeros()),
            Err(Error::ZeroVector { .. })
        ));
        assert!(matches!(
            s.surface_frame(&Vector3::new(f64::NAN, 0.0, 1.0), &Vector3::x()),
            Err(Error::ZeroVector { .. })
        ));
    }

    #[test]
    fn parallel_pairs_rejected() {
        let (l, u) = (triclinic(), u_true());
        let a = measured(&l, &u, [1.0, 0.0, 0.0], 1.0);
        let index = Orientation::busing_levy(a.clone(), measured(&l, &u, [2.0, 0.0, 0.0], 1.0));
        assert!(matches!(
            index,
            Err(Error::ParallelReflections {
                space: "index space",
                ..
            })
        ));
        // distinct hkl but the same measured direction
        let same_q = Reflection::new(Vector3::y(), *a.q_sample()).unwrap();
        assert!(matches!(
            Orientation::busing_levy(a, same_q),
            Err(Error::ParallelReflections {
                space: "the sample frame",
                ..
            })
        ));
        assert!(Reflection::new(Vector3::zeros(), Vector3::x()).is_err());
        assert!(Reflection::new(Vector3::x(), Vector3::new(f64::NAN, 0.0, 0.0)).is_err());
    }
}
