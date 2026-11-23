//! The mode's residual and its analytic Jacobian over the written axes.

use nalgebra::{DMatrix, DVector, Matrix3, UnitQuaternion, Vector3};

use super::jet::{Scalar, Vec3};
use super::{Equation, Mode, wrap};
use crate::{AxisKind, Error, Geometry};

/// Length (in units of k, or relative to the reference) below which a
/// direction ψ is built from is taken as null.
const NULL: f64 = 1e-10;

pub(super) struct System<'a> {
    geometry: &'a Geometry,
    written: &'a [usize],
    extra: &'a [Equation],
    current: &'a [f64],
    /// Target Q in the sample frame, units of k.
    q: Vector3<f64>,
    /// UB in units of k.
    ub: Matrix3<f64>,
}

pub(super) struct Eval {
    pub f: DVector<f64>,
    pub j: DMatrix<f64>,
}

/// Lab directions of the written axes of `stack` at `values`: each axis
/// rotated by the outer axes of the stack, zero for axes not in it.
fn lab_axes(
    geometry: &Geometry,
    written: &[usize],
    stack: &[usize],
    values: &[f64],
) -> Vec<Vector3<f64>> {
    let mut prefix = UnitQuaternion::identity();
    let mut out = vec![Vector3::zeros(); written.len()];
    for &i in stack {
        let axis = &geometry.axes()[i];
        if let (Some(c), AxisKind::Rotation { axis: a } | AxisKind::RotationAbout { axis: a, .. }) =
            (written.iter().position(|&w| w == i), axis.kind())
        {
            out[c] = prefix * a.into_inner();
        }
        prefix *= axis.transform(values[i]).rotation;
    }
    out
}

/// k̂f, k̂i and the sample rotation as jets over `written`; the rotations
/// themselves come from [`Geometry::poses`].
struct Frames {
    sample: UnitQuaternion<f64>,
    sample_axes: Vec<Vector3<f64>>,
    kf: Vec3,
    ki: Vec3,
}

impl Frames {
    fn new(geometry: &Geometry, written: &[usize], values: &[f64]) -> Result<Self, Error> {
        let poses = geometry.poses(values)?;
        let beam = geometry.beam_direction().into_inner();
        let axes = |stack| lab_axes(geometry, written, stack, values);
        Ok(Self {
            sample: poses.sample.rotation,
            sample_axes: axes(geometry.sample_stack()),
            kf: Vec3::rotated(
                &poses.detector.rotation,
                &axes(geometry.detector_stack()),
                &beam,
            ),
            ki: Vec3::rotated(&poses.source, &axes(geometry.source_stack()), &beam),
        })
    }

    /// A sample-frame vector in the lab.
    fn sample(&self, u: &Vector3<f64>) -> Vec3 {
        Vec3::rotated(&self.sample, &self.sample_axes, u)
    }
}

/// hkl's ψ (`hkl_psi_write`) of the lab vector `reference`; `None` where it
/// is undefined.
fn psi(kf: &Vec3, ki: &Vec3, reference: &Vec3) -> Option<Scalar> {
    let qn = kf.sub(ki).normalized(NULL)?;
    let n = kf.cross(ki).cross(&qn);
    let p = reference.sub(&qn.scaled(&reference.dot(&qn)));
    if n.v.norm() <= NULL || p.v.norm() <= NULL * reference.v.norm() {
        return None;
    }
    Some(Scalar::atan2(&qn.dot(&n.cross(&p)), &n.dot(&p)))
}

/// ψ of the sample-frame vector `reference` at `values`.
pub(super) fn psi_at(
    geometry: &Geometry,
    reference: &Vector3<f64>,
    values: &[f64],
) -> Result<Option<f64>, Error> {
    let fr = Frames::new(geometry, &[], values)?;
    Ok(psi(&fr.kf, &fr.ki, &fr.sample(reference)).map(|s| s.v))
}

impl<'a> System<'a> {
    /// `ub` and `q` in units of k; `current` already checked against the
    /// geometry.
    pub fn new(mode: &'a Mode<'a>, ub: Matrix3<f64>, q: Vector3<f64>, current: &'a [f64]) -> Self {
        Self {
            geometry: mode.geometry(),
            written: mode.written(),
            extra: mode.extra(),
            current,
            q,
            ub,
        }
    }

    pub fn current(&self) -> &[f64] {
        self.current
    }

    pub fn is_written(&self, axis: usize) -> bool {
        self.written.contains(&axis)
    }

    /// Every axis: `x` on the written ones, `current` elsewhere.
    pub fn values(&self, x: &[f64]) -> Vec<f64> {
        let mut v = self.current.to_vec();
        for (&i, &xi) in self.written.iter().zip(x) {
            v[i] = xi;
        }
        v
    }

    pub fn written_values(&self, values: &[f64]) -> Vec<f64> {
        self.written.iter().map(|&i| values[i]).collect()
    }

    /// Residuals (3 hkl rows, then the extra equations; dimensionless) and
    /// their Jacobian over the written axes at `values`, every axis given.
    /// `None` where the values are invalid or an equation is undefined.
    pub fn eval(&self, values: &[f64]) -> Option<Eval> {
        let fr = Frames::new(self.geometry, self.written, values).ok()?;
        let hkl = fr.sample(&self.q).sub(&fr.kf.sub(&fr.ki));
        let mut rows: Vec<Scalar> = (0..3).map(|i| hkl.component(i)).collect();
        for e in self.extra {
            rows.push(match e {
                Equation::Bisector { sample, detector } => {
                    let mut d = vec![0.0; self.written.len()];
                    for (axis, coefficient) in [(detector, 1.0), (sample, -2.0)] {
                        if let Some(c) = self.written.iter().position(|w| w == axis) {
                            d[c] += coefficient;
                        }
                    }
                    Scalar {
                        v: wrap(values[*detector] - 2.0 * values[*sample]),
                        d,
                    }
                }
                Equation::DoubleDiffraction { hkl2 } => {
                    let kf2 = fr.sample(&(self.ub * hkl2)).add(&fr.ki);
                    kf2.dot(&kf2).sub(&fr.ki.dot(&fr.ki))
                }
                Equation::PsiConstant {
                    reference,
                    psi: target,
                } => {
                    let p = psi(&fr.kf, &fr.ki, &fr.sample(&(self.ub * reference)))?;
                    Scalar {
                        v: wrap(p.v - target),
                        d: p.d,
                    }
                }
            });
        }
        Some(Eval {
            f: DVector::from_iterator(rows.len(), rows.iter().map(|r| r.v)),
            j: DMatrix::from_fn(rows.len(), self.written.len(), |r, c| rows[r].d[c]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::e4c;
    use crate::presets::preset;

    /// Analytic Jacobian against a central difference of the residual, for
    /// every E4C mode, at points away from the wrap cut of the angular rows.
    #[test]
    fn jacobian_matches_central_difference() {
        let ub = Matrix3::new(0.9, 0.1, -0.2, 0.05, 1.1, 0.3, -0.1, 0.2, 0.8);
        let q = Vector3::new(0.3, -0.5, 0.7);
        let h = 1e-6;
        for name in ["E4CV", "E4CH"] {
            let geo = preset(name).unwrap();
            let modes = [
                e4c::bissector(&geo).unwrap(),
                e4c::constant_omega(&geo).unwrap(),
                e4c::constant_chi(&geo).unwrap(),
                e4c::constant_phi(&geo).unwrap(),
                e4c::double_diffraction(&geo, Vector3::new(1.0, -1.0, 0.5)).unwrap(),
                e4c::psi_constant(&geo, Vector3::new(0.2, 0.0, 1.0), 0.4).unwrap(),
            ];
            for values in [
                [0.3, -0.4, 1.2, 0.9],
                [-2.0, 1.1, -0.7, -1.4],
                [1.0, 2.5, 3.0, 2.2],
            ] {
                for mode in &modes {
                    let s = System::new(mode, ub, q, &values);
                    let x = s.written_values(&values);
                    let j = s.eval(&values).unwrap().j;
                    for c in 0..x.len() {
                        let at = |d: f64| {
                            let mut y = x.clone();
                            y[c] += d;
                            s.eval(&s.values(&y)).unwrap().f
                        };
                        let (plus, minus) = (at(h), at(-h));
                        for r in 0..j.nrows() {
                            let numeric = wrap(plus[r] - minus[r]) / (2.0 * h);
                            let miss = (j[(r, c)] - numeric).abs();
                            assert!(
                                miss <= 1e-8 * (1.0 + numeric.abs()),
                                "{name} {} {values:?} J[{r},{c}] = {} vs {numeric}",
                                mode.name(),
                                j[(r, c)]
                            );
                        }
                    }
                }
            }
        }
    }
}
