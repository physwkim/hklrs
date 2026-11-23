use nalgebra::Vector3;

use super::kappa::{KappaAxes, restrict};
use super::{InverseError, finite};
use crate::{Axis, AxisKind, Geometry};

/// An equation a mode adds to the three hkl equations. Angles in rad,
/// vectors in Miller indices.
#[derive(Clone, Debug, PartialEq)]
pub enum Equation {
    /// detector ≡ 2·sample (mod 2π), axis indices.
    Bisector { sample: usize, detector: usize },
    /// |R_sample·UB·hkl2 + kᵢ| = |kᵢ|: hkl2 on the Ewald sphere too.
    DoubleDiffraction { hkl2: Vector3<f64> },
    /// ψ of `reference`, as [`super::psi`] defines it, equals `psi`.
    PsiConstant { reference: Vector3<f64>, psi: f64 },
    /// axis ≡ 0 (mod π), axis index: parked at 0 or π.
    ZeroModPi { axis: usize },
    /// a ≡ b (mod 2π), axis indices.
    Equal { a: usize, b: usize },
    /// The eulerians of the mode's kappa arm ([`Mode::new_kappa`]).
    Kappa(Eulerian),
}

/// What a kappa mode holds of the eulerian (omega, chi, phi) of its arm's
/// sample rotation ([`super::kappa::KappaArm::to_eulerian`]). Both eulerian
/// images are that rotation, so either one counts. Angles in rad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Eulerian {
    /// omega ≡ value (mod π).
    Omega(f64),
    /// |chi| = |value|, value taken on [−π, π].
    Chi(f64),
    /// phi ≡ value (mod π).
    Phi(f64),
    /// axis `detector` ≡ 2·omega (mod 2π).
    Bisector { detector: usize },
}

/// Which axes the inverse writes and the equations that fix them, checked
/// against the geometry it borrows, so it cannot outlive or be applied to
/// another one.
#[derive(Clone, Debug, PartialEq)]
pub struct Mode<'g> {
    geometry: &'g Geometry,
    name: String,
    written: Vec<usize>,
    extra: Vec<Equation>,
    kappa: Option<KappaAxes>,
}

/// The axis at `index` and its direction; rotations only.
pub(super) fn rotation(
    geometry: &Geometry,
    index: usize,
) -> Result<(&Axis, Vector3<f64>), InverseError> {
    let axes = geometry.axes();
    let axis = axes.get(index).ok_or(InverseError::NoSuchAxis {
        index,
        count: axes.len(),
    })?;
    match axis.kind() {
        AxisKind::Rotation { axis: a } | AxisKind::RotationAbout { axis: a, .. } => {
            Ok((axis, a.into_inner()))
        }
        _ => Err(InverseError::NotRotation {
            name: axis.name().to_owned(),
        }),
    }
}

fn in_stack(
    geometry: &Geometry,
    index: usize,
    stack: &[usize],
    holder: &'static str,
) -> Result<(), InverseError> {
    let (axis, _) = rotation(geometry, index)?;
    if stack.contains(&index) {
        Ok(())
    } else {
        Err(InverseError::NotInStack {
            name: axis.name().to_owned(),
            holder,
        })
    }
}

fn reference(v: &Vector3<f64>, what: &'static str) -> Result<(), InverseError> {
    finite(v.as_slice(), what)?;
    if v.norm() > 0.0 {
        Ok(())
    } else {
        Err(InverseError::ZeroReference { what })
    }
}

impl Equation {
    fn check(&self, geometry: &Geometry, kappa: Option<&KappaAxes>) -> Result<(), InverseError> {
        match self {
            Self::Bisector { sample, detector } => {
                in_stack(geometry, *sample, geometry.sample_stack(), "sample")?;
                in_stack(geometry, *detector, geometry.detector_stack(), "detector")
            }
            Self::DoubleDiffraction { hkl2 } => reference(hkl2, "hkl2"),
            Self::PsiConstant { reference: r, psi } => {
                reference(r, "psi reference")?;
                finite(&[*psi], "psi")
            }
            Self::ZeroModPi { axis } => rotation(geometry, *axis).map(|_| ()),
            Self::Equal { a, b } => {
                rotation(geometry, *a)?;
                rotation(geometry, *b).map(|_| ())
            }
            Self::Kappa(held) => {
                let kappa = kappa.ok_or(InverseError::NoKappaArm)?;
                match *held {
                    Eulerian::Omega(v) | Eulerian::Phi(v) => finite(&[v], "eulerian angle"),
                    Eulerian::Chi(v) => {
                        finite(&[v], "eulerian angle")?;
                        let (chi, reach) = (restrict(v), kappa.arm().reach());
                        if chi.abs() > reach {
                            return Err(InverseError::ChiOutOfReach { chi, reach });
                        }
                        Ok(())
                    }
                    Eulerian::Bisector { detector } => {
                        in_stack(geometry, detector, geometry.detector_stack(), "detector")
                    }
                }
            }
        }
    }
}

impl<'g> Mode<'g> {
    /// `written` are axis indices, rotations only; there must be exactly one
    /// written axis per equation (3 + `extra.len()`).
    pub fn new(
        geometry: &'g Geometry,
        name: &str,
        written: &[usize],
        extra: Vec<Equation>,
    ) -> Result<Self, InverseError> {
        Self::build(geometry, name, written, extra, None)
    }

    /// [`Mode::new`] on a geometry whose sample axes `kappa` (komega, kappa,
    /// kphi) are a kappa arm: adjacent in the sample stack in that order,
    /// komega and kphi about one direction d, kappa about k with
    /// α = acos(d·k) ∈ (0, π/2]. Its equations may hold the arm's eulerians.
    pub fn new_kappa(
        geometry: &'g Geometry,
        name: &str,
        kappa: [usize; 3],
        written: &[usize],
        extra: Vec<Equation>,
    ) -> Result<Self, InverseError> {
        let kappa = KappaAxes::new(geometry, kappa)?;
        Self::build(geometry, name, written, extra, Some(kappa))
    }

    fn build(
        geometry: &'g Geometry,
        name: &str,
        written: &[usize],
        extra: Vec<Equation>,
        kappa: Option<KappaAxes>,
    ) -> Result<Self, InverseError> {
        for (k, &i) in written.iter().enumerate() {
            let (axis, _) = rotation(geometry, i)?;
            if written[..k].contains(&i) {
                return Err(InverseError::WrittenTwice {
                    name: axis.name().to_owned(),
                });
            }
        }
        if written.len() != 3 + extra.len() {
            return Err(InverseError::EquationCount {
                written: written.len(),
                equations: 3 + extra.len(),
            });
        }
        for e in &extra {
            e.check(geometry, kappa.as_ref())?;
        }
        Ok(Self {
            geometry,
            name: name.to_owned(),
            written: written.to_vec(),
            extra,
            kappa,
        })
    }

    /// [`Mode::new`] with the written axes by name; `extra` gets their
    /// indices.
    pub(super) fn named(
        geometry: &'g Geometry,
        name: &str,
        written: &[&str],
        extra: impl FnOnce(&[usize]) -> Vec<Equation>,
    ) -> Result<Self, InverseError> {
        let w = written
            .iter()
            .map(|n| geometry.axis_index(n))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(geometry, name, &w, extra(&w))
    }

    pub fn geometry(&self) -> &'g Geometry {
        self.geometry
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Written axis indices, in Jacobian column order.
    pub fn written(&self) -> &[usize] {
        &self.written
    }

    pub fn extra(&self) -> &[Equation] {
        &self.extra
    }

    pub fn kappa(&self) -> Option<&KappaAxes> {
        self.kappa.as_ref()
    }
}
