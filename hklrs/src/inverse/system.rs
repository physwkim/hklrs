//! The mode's residual and its analytic Jacobian over the written axes.

use nalgebra::{DMatrix, DVector, Matrix3, UnitQuaternion, Vector3};

use super::jet::{Scalar, Vec3};
use super::{Equation, Mode, wrap};
use crate::{Axis, AxisKind, Error, Geometry};

/// Sine below which an angle ψ is built from (θ, π − 2θ, Q to the
/// reference) is taken as null; each test divides out its operands' own
/// scale, so none depends on λ or on the units of k̂f and k̂i (HKL-25).
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
    let (f, i) = (kf.v.norm(), ki.v.norm());
    // |kf − ki| = (f + i)·sin θ and |n| = f·i·sin 2θ for f = i.
    let qn = kf.sub(ki).normalized(NULL * (f + i))?;
    let n = kf.cross(ki).cross(&qn);
    let p = reference.sub(&qn.scaled(&reference.dot(&qn)));
    if n.v.norm() <= NULL * f * i || p.v.norm() <= NULL * reference.v.norm() {
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

    pub fn axis(&self, axis: usize) -> &'a Axis {
        &self.geometry.axes()[axis]
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
                Equation::ZeroModPi { axis } => Scalar {
                    v: wrap(2.0 * values[*axis]) / 2.0,
                    d: self.written.iter().map(|w| f64::from(w == axis)).collect(),
                },
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
    use crate::inverse::{e4c, e6c};
    use crate::presets::preset;

    fn modes(geo: &Geometry) -> Vec<Mode<'_>> {
        let hkl2 = Vector3::new(1.0, -1.0, 0.5);
        let reference = Vector3::new(0.2, 0.0, 1.0);
        if geo.axes().len() == 4 {
            vec![
                e4c::bissector(geo).unwrap(),
                e4c::constant_omega(geo).unwrap(),
                e4c::constant_chi(geo).unwrap(),
                e4c::constant_phi(geo).unwrap(),
                e4c::double_diffraction(geo, hkl2).unwrap(),
                e4c::psi_constant(geo, reference, 0.4).unwrap(),
            ]
        } else {
            vec![
                e6c::bissector_vertical(geo).unwrap(),
                e6c::lifting_detector_phi(geo).unwrap(),
                e6c::lifting_detector_mu(geo).unwrap(),
                e6c::double_diffraction_vertical(geo, hkl2).unwrap(),
                e6c::bissector_horizontal(geo).unwrap(),
                e6c::double_diffraction_horizontal(geo, hkl2).unwrap(),
                e6c::psi_constant_horizontal(geo, reference, 0.4).unwrap(),
            ]
        }
    }

    /// ψ of `reference` with k̂f at 2θ from k̂i about z, k̂f and k̂i scaled
    /// by k and `reference` by 1/k, at λ = 0.1 Å and 50 Å.
    fn psi_at_two_wavelengths(kf: Vector3<f64>, reference: Vector3<f64>) -> [Option<f64>; 2] {
        let constant = |v| Vec3 { v, d: vec![] };
        [0.1, 50.0].map(|lambda: f64| {
            let k = std::f64::consts::TAU / lambda;
            let ki = constant(Vector3::x() * k);
            psi(&constant(kf * k), &ki, &constant(reference / k)).map(|s| s.v)
        })
    }

    /// The same answer at both wavelengths, `None` when `defined` is false;
    /// ψ is conditioned as 1/sin of the near-null angle, ~1e-7 rad at
    /// 10·NULL.
    fn same_at_two_wavelengths(kf: Vector3<f64>, reference: Vector3<f64>, defined: bool) {
        let [a, b] = psi_at_two_wavelengths(kf, reference);
        assert_eq!(a.is_some(), defined, "{kf:?} {reference:?}: {a:?}");
        assert_eq!(b.is_some(), defined, "{kf:?} {reference:?}: {b:?}");
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }

    /// HKL-25: each null test of ψ at one side of its boundary, the same
    /// at λ = 0.1 Å and 50 Å.
    #[test]
    fn psi_null_tests_are_scale_free() {
        let reference = Vector3::new(0.3, 0.2, 1.0);
        for (s, defined) in [(0.1 * NULL, false), (10.0 * NULL, true)] {
            // θ → 0: Q has no direction.
            let two_theta = 2.0 * s;
            same_at_two_wavelengths(
                Vector3::new(two_theta.cos(), two_theta.sin(), 0.0),
                reference,
                defined,
            );
            // 2θ → π: no scattering plane.
            same_at_two_wavelengths(Vector3::new(-s.cos(), s.sin(), 0.0), reference, defined);
            // The reference along Q.
            let kf = Vector3::new(1.0_f64.cos(), 1.0_f64.sin(), 0.0);
            let q = (kf - Vector3::x()).normalize();
            same_at_two_wavelengths(kf, q * s.cos() + Vector3::z() * s.sin(), defined);
        }
    }

    /// Analytic Jacobian against a central difference of the residual, for
    /// every E4C mode and every kind of E6C mode, at points away from the
    /// wrap cut of the angular rows.
    #[test]
    fn jacobian_matches_central_difference() {
        let ub = Matrix3::new(0.9, 0.1, -0.2, 0.05, 1.1, 0.3, -0.1, 0.2, 0.8);
        let q = Vector3::new(0.3, -0.5, 0.7);
        let h = 1e-6;
        let points = [
            [0.3, -0.4, 1.2, 0.9, -0.6, 0.2],
            [-2.0, 1.1, -0.7, -1.4, 0.8, -1.3],
            [1.0, 2.5, 3.0, 2.2, 2.9, 0.7],
        ];
        for name in ["E4CV", "E4CH", "E6C"] {
            let geo = preset(name).unwrap();
            let n = geo.axes().len();
            for point in &points {
                let values = &point[..n];
                for mode in &modes(&geo) {
                    let s = System::new(mode, ub, q, values);
                    let x = s.written_values(values);
                    let j = s.eval(values).unwrap().j;
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
