//! Diffractometer geometry: one axis table, a sample stack and a detector
//! stack of indices into it. An axis listed in both stacks (hkl ZAXIS `mu`,
//! PETRA3 P23 `omega_t`) is one definition with one value.

use nalgebra::{Isometry3, Point3, Translation3, Unit, UnitQuaternion, Vector3};

use crate::{Error, units::Deg};

#[derive(Clone, Debug, PartialEq)]
pub enum AxisKind {
    /// Right-handed rotation about `axis` through the origin; value in rad.
    Rotation { axis: Unit<Vector3<f64>> },
    /// Right-handed rotation about `axis` through `origin` (metres); value in rad.
    RotationAbout {
        axis: Unit<Vector3<f64>>,
        origin: Vector3<f64>,
    },
    /// Translation along `direction`; value in metres.
    Translation { direction: Unit<Vector3<f64>> },
    /// A value carried with the geometry that moves nothing.
    Parameter,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    name: String,
    kind: AxisKind,
}

impl Axis {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> &AxisKind {
        &self.kind
    }

    pub fn is_rotation(&self) -> bool {
        matches!(
            self.kind,
            AxisKind::Rotation { .. } | AxisKind::RotationAbout { .. }
        )
    }

    /// The rigid motion this axis applies at `value`.
    pub fn transform(&self, value: f64) -> Isometry3<f64> {
        match &self.kind {
            AxisKind::Rotation { axis } => Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_axis_angle(axis, value),
            ),
            AxisKind::RotationAbout { axis, origin } => {
                let r = UnitQuaternion::from_axis_angle(axis, value);
                Isometry3::from_parts(Translation3::from(origin - r * origin), r)
            }
            AxisKind::Translation { direction } => Isometry3::from_parts(
                Translation3::from(direction.into_inner() * value),
                UnitQuaternion::identity(),
            ),
            AxisKind::Parameter => Isometry3::identity(),
        }
    }
}

/// Holder poses at one set of axis values, in the lab frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poses {
    pub sample: Isometry3<f64>,
    pub detector: Isometry3<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    name: String,
    axes: Vec<Axis>,
    sample: Vec<usize>,
    detector: Vec<usize>,
    beam_direction: Unit<Vector3<f64>>,
}

fn unit(v: [f64; 3], what: &str) -> Result<Unit<Vector3<f64>>, Error> {
    let v = Vector3::from(v);
    if v.iter().all(|x| x.is_finite()) {
        if let Some(u) = Unit::try_new(v, 0.0) {
            return Ok(u);
        }
    }
    Err(Error::ZeroVector {
        what: what.to_owned(),
    })
}

/// Builds a [`Geometry`]. Stacks are listed outermost first. Adding an axis
/// name that already exists reuses it when the kind matches, so a shared
/// axis is written once per holder and defined once.
#[derive(Clone, Debug)]
pub struct GeometryBuilder {
    name: String,
    axes: Vec<Axis>,
    sample: Vec<usize>,
    detector: Vec<usize>,
    beam_direction: [f64; 3],
    error: Option<Error>,
}

#[derive(Clone, Copy)]
enum Holder {
    Sample,
    Detector,
}

impl GeometryBuilder {
    fn push(mut self, holder: Holder, name: &str, kind: Result<AxisKind, Error>) -> Self {
        if self.error.is_some() {
            return self;
        }
        let kind = match kind {
            Ok(k) => k,
            Err(e) => {
                self.error = Some(e);
                return self;
            }
        };
        let idx = match self.axes.iter().position(|a| a.name == name) {
            Some(i) if self.axes[i].kind == kind => i,
            Some(_) => {
                self.error = Some(Error::AxisConflict {
                    name: name.to_owned(),
                });
                return self;
            }
            None => {
                self.axes.push(Axis {
                    name: name.to_owned(),
                    kind,
                });
                self.axes.len() - 1
            }
        };
        let (stack, label) = match holder {
            Holder::Sample => (&mut self.sample, "sample"),
            Holder::Detector => (&mut self.detector, "detector"),
        };
        if stack.contains(&idx) {
            self.error = Some(Error::AxisRepeated {
                name: name.to_owned(),
                holder: label,
            });
        } else {
            stack.push(idx);
        }
        self
    }

    fn rotation(name: &str, axis: [f64; 3]) -> Result<AxisKind, Error> {
        Ok(AxisKind::Rotation {
            axis: unit(axis, name)?,
        })
    }

    fn rotation_about(name: &str, axis: [f64; 3], origin: [f64; 3]) -> Result<AxisKind, Error> {
        let origin = Vector3::from(origin);
        if !origin.iter().all(|x| x.is_finite()) {
            return Err(Error::ZeroVector {
                what: format!("{name} origin"),
            });
        }
        Ok(AxisKind::RotationAbout {
            axis: unit(axis, name)?,
            origin,
        })
    }

    fn translation(name: &str, direction: [f64; 3]) -> Result<AxisKind, Error> {
        Ok(AxisKind::Translation {
            direction: unit(direction, name)?,
        })
    }

    pub fn sample_rotation(self, name: &str, axis: [f64; 3]) -> Self {
        self.push(Holder::Sample, name, Self::rotation(name, axis))
    }

    pub fn sample_rotation_about(self, name: &str, axis: [f64; 3], origin: [f64; 3]) -> Self {
        self.push(
            Holder::Sample,
            name,
            Self::rotation_about(name, axis, origin),
        )
    }

    pub fn sample_translation(self, name: &str, direction: [f64; 3]) -> Self {
        self.push(Holder::Sample, name, Self::translation(name, direction))
    }

    pub fn sample_parameter(self, name: &str) -> Self {
        self.push(Holder::Sample, name, Ok(AxisKind::Parameter))
    }

    pub fn detector_rotation(self, name: &str, axis: [f64; 3]) -> Self {
        self.push(Holder::Detector, name, Self::rotation(name, axis))
    }

    pub fn detector_rotation_about(self, name: &str, axis: [f64; 3], origin: [f64; 3]) -> Self {
        self.push(
            Holder::Detector,
            name,
            Self::rotation_about(name, axis, origin),
        )
    }

    pub fn detector_translation(self, name: &str, direction: [f64; 3]) -> Self {
        self.push(Holder::Detector, name, Self::translation(name, direction))
    }

    pub fn detector_parameter(self, name: &str) -> Self {
        self.push(Holder::Detector, name, Ok(AxisKind::Parameter))
    }

    /// Incident beam direction in the lab frame; default +x.
    pub fn beam_direction(mut self, direction: [f64; 3]) -> Self {
        self.beam_direction = direction;
        self
    }

    pub fn build(self) -> Result<Geometry, Error> {
        if let Some(e) = self.error {
            return Err(e);
        }
        Ok(Geometry {
            beam_direction: unit(self.beam_direction, "beam direction")?,
            name: self.name,
            axes: self.axes,
            sample: self.sample,
            detector: self.detector,
        })
    }
}

impl Geometry {
    pub fn builder(name: &str) -> GeometryBuilder {
        GeometryBuilder {
            name: name.to_owned(),
            axes: Vec::new(),
            sample: Vec::new(),
            detector: Vec::new(),
            beam_direction: [1.0, 0.0, 0.0],
            error: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every axis in value order.
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub fn axis_index(&self, name: &str) -> Result<usize, Error> {
        self.axes
            .iter()
            .position(|a| a.name == name)
            .ok_or_else(|| Error::UnknownAxis {
                name: name.to_owned(),
            })
    }

    /// Sample stack, outermost first, as indices into [`Geometry::axes`].
    pub fn sample_stack(&self) -> &[usize] {
        &self.sample
    }

    /// Detector stack, outermost first, as indices into [`Geometry::axes`].
    pub fn detector_stack(&self) -> &[usize] {
        &self.detector
    }

    pub fn beam_direction(&self) -> Unit<Vector3<f64>> {
        self.beam_direction
    }

    fn check(&self, values: &[f64]) -> Result<(), Error> {
        if values.len() != self.axes.len() {
            return Err(Error::AxisCount {
                expected: self.axes.len(),
                got: values.len(),
            });
        }
        match values.iter().position(|v| !v.is_finite()) {
            Some(index) => Err(Error::NonFinite { index }),
            None => Ok(()),
        }
    }

    fn chain(&self, stack: &[usize], values: &[f64]) -> Isometry3<f64> {
        stack.iter().fold(Isometry3::identity(), |acc, &i| {
            acc * self.axes[i].transform(values[i])
        })
    }

    /// The only place a stack is applied: `values` are in axis order, radians
    /// for rotations, metres for translations.
    pub fn poses(&self, values: &[f64]) -> Result<Poses, Error> {
        self.check(values)?;
        Ok(Poses {
            sample: self.chain(&self.sample, values),
            detector: self.chain(&self.detector, values),
        })
    }

    /// Converts boundary values (degrees for rotations, metres for
    /// translations, parameters unchanged) to the values [`Geometry::poses`] takes.
    pub fn values_from_degrees(&self, user: &[f64]) -> Result<Vec<f64>, Error> {
        self.check(user)?;
        Ok(self
            .axes
            .iter()
            .zip(user)
            .map(|(a, &v)| {
                if a.is_rotation() {
                    crate::Rad::from(Deg(v)).0
                } else {
                    v
                }
            })
            .collect())
    }

    /// Inverse of [`Geometry::values_from_degrees`].
    pub fn values_to_degrees(&self, values: &[f64]) -> Result<Vec<f64>, Error> {
        self.check(values)?;
        Ok(self
            .axes
            .iter()
            .zip(values)
            .map(|(a, &v)| {
                if a.is_rotation() {
                    Deg::from(crate::Rad(v)).0
                } else {
                    v
                }
            })
            .collect())
    }
}

impl Poses {
    /// Sample origin in the lab frame.
    pub fn sample_position(&self) -> Point3<f64> {
        self.sample * Point3::origin()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use nalgebra::{Rotation3, Vector3};

    use super::*;

    fn rot_xyz(axis: usize, a: f64) -> Rotation3<f64> {
        let (s, c) = a.sin_cos();
        match axis {
            0 => Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
                1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c,
            )),
            1 => Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
                c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c,
            )),
            _ => Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
                c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0,
            )),
        }
    }

    #[test]
    fn rotation_matches_rot_x_y_z() {
        for (i, v) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .into_iter()
            .enumerate()
        {
            let g = Geometry::builder("t")
                .sample_rotation("r", v)
                .build()
                .unwrap();
            for a in [-2.5, -0.3, 0.0, 0.7, 3.0] {
                let m = g.poses(&[a]).unwrap().sample.rotation.to_rotation_matrix();
                assert!((m.matrix() - rot_xyz(i, a).matrix()).abs().max() < 1e-15);
            }
        }
    }

    #[test]
    fn minus_circle_is_negated_axis() {
        let p = Geometry::builder("t")
            .sample_rotation("r", [0.0, 1.0, 0.0])
            .build()
            .unwrap();
        let m = Geometry::builder("t")
            .sample_rotation("r", [0.0, -1.0, 0.0])
            .build()
            .unwrap();
        let a = p.poses(&[0.4]).unwrap().sample;
        let b = m.poses(&[-0.4]).unwrap().sample;
        assert!((a.to_homogeneous() - b.to_homogeneous()).abs().max() < 1e-15);
    }

    #[test]
    fn outermost_first() {
        // omega about -y outside chi about +x: R = R_omega · R_chi.
        let g = Geometry::builder("t")
            .sample_rotation("omega", [0.0, -1.0, 0.0])
            .sample_rotation("chi", [1.0, 0.0, 0.0])
            .build()
            .unwrap();
        let (w, c) = (0.3, 1.1);
        let r = g.poses(&[w, c]).unwrap().sample.rotation;
        let expect = UnitQuaternion::from_axis_angle(&-Vector3::y_axis(), w)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), c);
        assert!(r.angle_to(&expect) < 1e-15);
    }

    #[test]
    fn shared_axis_is_one_value() {
        let g = Geometry::builder("t")
            .sample_rotation("mu", [0.0, 0.0, 1.0])
            .sample_rotation("omega", [0.0, -1.0, 0.0])
            .detector_rotation("mu", [0.0, 0.0, 1.0])
            .detector_rotation("delta", [0.0, -1.0, 0.0])
            .build()
            .unwrap();
        assert_eq!(g.axes().len(), 3);
        assert_eq!(g.sample_stack(), &[0, 1]);
        assert_eq!(g.detector_stack(), &[0, 2]);
        let p = g.poses(&[FRAC_PI_2, 0.0, 0.0]).unwrap();
        let x = p.detector * Point3::new(1.0, 0.0, 0.0);
        assert!((x - Point3::new(0.0, 1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn builder_rejects_conflicts() {
        let e = Geometry::builder("t")
            .sample_rotation("mu", [0.0, 0.0, 1.0])
            .detector_rotation("mu", [0.0, 1.0, 0.0])
            .build();
        assert_eq!(e, Err(Error::AxisConflict { name: "mu".into() }));
        let e = Geometry::builder("t")
            .sample_rotation("a", [0.0, 0.0, 1.0])
            .sample_rotation("a", [0.0, 0.0, 1.0])
            .build();
        assert!(matches!(e, Err(Error::AxisRepeated { .. })));
        assert!(
            Geometry::builder("t")
                .sample_rotation("a", [0.0; 3])
                .build()
                .is_err()
        );
        assert!(
            Geometry::builder("t")
                .beam_direction([0.0; 3])
                .build()
                .is_err()
        );
    }

    #[test]
    fn rotation_about_origin_and_translation() {
        // Rotating (2,0,0) by 90° about z through (1,0,0) gives (1,1,0);
        // a translation inside it is rotated with it.
        let g = Geometry::builder("t")
            .detector_rotation_about("r", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
            .detector_translation("t", [1.0, 0.0, 0.0])
            .build()
            .unwrap();
        let p = g.poses(&[FRAC_PI_2, 0.5]).unwrap().detector * Point3::new(2.0, 0.0, 0.0);
        assert!((p - Point3::new(1.0, 1.5, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn values_checked() {
        let g = Geometry::builder("t")
            .sample_rotation("r", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        assert_eq!(
            g.poses(&[]),
            Err(Error::AxisCount {
                expected: 1,
                got: 0
            })
        );
        assert_eq!(g.poses(&[f64::NAN]), Err(Error::NonFinite { index: 0 }));
    }
}
