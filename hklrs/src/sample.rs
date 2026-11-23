//! Sample: lattice plus orientation U.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use crate::{Error, Lattice};

#[derive(Clone, Debug, PartialEq)]
pub enum Orientation {
    /// U given directly.
    Manual { u: UnitQuaternion<f64> },
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
        match self.orientation {
            Orientation::Manual { u } => u,
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
}
