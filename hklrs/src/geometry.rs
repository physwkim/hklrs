//! Diffractometer geometry: one axis table, a sample stack and a detector
//! stack of indices into it. An axis listed in both stacks (hkl ZAXIS `mu`,
//! PETRA3 P23 `omega_t`) is one definition with one value.

use std::f64::consts::{PI, TAU};

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
    min: f64,
    max: f64,
    /// Some exactly for a rotation.
    cut: Option<f64>,
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

    /// Inclusive range `(min, max)`, an infinite end unbounded; unbounded by
    /// default.
    pub fn limits(&self) -> (f64, f64) {
        (self.min, self.max)
    }

    pub fn within_limits(&self, value: f64) -> bool {
        self.min <= value && value <= self.max
    }

    /// A rotation's values are represented in [cut, cut + 2π); −π by
    /// default. `None` for any other axis.
    pub fn cut_point(&self) -> Option<f64> {
        self.cut
    }

    /// The equivalent of `value` in this rotation's cut window, `value`
    /// itself when already there or when the axis is not a rotation.
    pub fn wrap(&self, value: f64) -> f64 {
        let Some(cut) = self.cut else {
            return value;
        };
        let end = cut + TAU;
        if (cut..end).contains(&value) {
            return value;
        }
        // Monotonic rounding keeps cut + r in [cut, end]; it reaches end
        // only for a value a rounding below the cut.
        let w = cut + (value - cut).rem_euclid(TAU);
        if w < end { w } else { cut }
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
    /// Rotation of the incident beam: `source * beam_direction` is k̂ᵢ.
    pub source: UnitQuaternion<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    name: String,
    axes: Vec<Axis>,
    sample: Vec<usize>,
    detector: Vec<usize>,
    source: Vec<usize>,
    beam_direction: Unit<Vector3<f64>>,
}

fn unit(v: [f64; 3], what: &str) -> Result<Unit<Vector3<f64>>, Error> {
    let v = Vector3::from(v);
    if v.iter().all(|x| x.is_finite())
        && let Some(u) = Unit::try_new(v, 0.0)
    {
        return Ok(u);
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
    source: Vec<usize>,
    beam_direction: [f64; 3],
    error: Option<Error>,
}

#[derive(Clone, Copy)]
enum Holder {
    Sample,
    Detector,
    Source,
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
                let mut axis = Axis {
                    name: name.to_owned(),
                    kind,
                    min: f64::NEG_INFINITY,
                    max: f64::INFINITY,
                    cut: None,
                };
                axis.cut = axis.is_rotation().then_some(-PI);
                self.axes.push(axis);
                self.axes.len() - 1
            }
        };
        let (stack, label) = match holder {
            Holder::Sample => (&mut self.sample, "sample"),
            Holder::Detector => (&mut self.detector, "detector"),
            Holder::Source => (&mut self.source, "source"),
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

    /// Rotation of the incident beam, outermost first. Only rotations: a
    /// translation cannot change a plane wave's direction.
    pub fn source_rotation(self, name: &str, axis: [f64; 3]) -> Self {
        self.push(Holder::Source, name, Self::rotation(name, axis))
    }

    /// Incident beam direction in the lab frame with the source stack at
    /// zero; default +x.
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
            source: self.source,
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
            source: Vec::new(),
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

    /// Limits axis `name` to `[min, max]` (rad or m; an infinite end is
    /// unbounded). The inverse rejects a solution that takes a written axis
    /// outside its limits, after wrapping it to its cut point.
    pub fn set_limits(&mut self, name: &str, min: f64, max: f64) -> Result<(), Error> {
        let i = self.axis_index(name)?;
        if min.is_nan() || max.is_nan() || min > max {
            return Err(Error::AxisLimits {
                name: name.to_owned(),
                min,
                max,
            });
        }
        (self.axes[i].min, self.axes[i].max) = (min, max);
        Ok(())
    }

    /// Represents rotation `name` in [cut, cut + 2π) rad.
    pub fn set_cut_point(&mut self, name: &str, cut: f64) -> Result<(), Error> {
        let i = self.axis_index(name)?;
        if !cut.is_finite() {
            return Err(Error::CutPoint {
                name: name.to_owned(),
                cut,
            });
        }
        match &mut self.axes[i].cut {
            Some(c) => {
                *c = cut;
                Ok(())
            }
            None => Err(Error::NoCutPoint {
                name: name.to_owned(),
            }),
        }
    }

    /// Sample stack, outermost first, as indices into [`Geometry::axes`].
    pub fn sample_stack(&self) -> &[usize] {
        &self.sample
    }

    /// Detector stack, outermost first, as indices into [`Geometry::axes`].
    pub fn detector_stack(&self) -> &[usize] {
        &self.detector
    }

    /// Source stack, outermost first, as indices into [`Geometry::axes`].
    pub fn source_stack(&self) -> &[usize] {
        &self.source
    }

    /// Incident beam direction with the source stack at zero.
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
            source: self.chain(&self.source, values).rotation,
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

    /// XU-5: xu loads one global `kappa_dir` for every kappa circle, and one
    /// of its variants reads it from an uninitialised buffer. Two kappa
    /// circles with different axes each turn about their own.
    #[test]
    fn xu_5_each_kappa_circle_turns_about_its_own_axis() {
        let a1 = crate::presets::kappa_axis(crate::Rad(50f64.to_radians()));
        let a2 = crate::presets::kappa_axis(crate::Rad(-30f64.to_radians()));
        let g = Geometry::builder("kk")
            .sample_rotation("k1", a1)
            .sample_rotation("k2", a2)
            .build()
            .unwrap();
        let about = |axis: [f64; 3], v: f64| {
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vector3::from(axis)), v)
        };
        let (v1, v2) = (0.7, -1.2);
        let got = g.poses(&[v1, v2]).unwrap().sample.rotation;
        assert!(got.angle_to(&(about(a1, v1) * about(a2, v2))) < 1e-15);
        for shared in [about(a1, v1) * about(a1, v2), about(a2, v1) * about(a2, v2)] {
            assert!(got.angle_to(&shared) > 0.1);
        }
        let second = g.poses(&[0.0, v2]).unwrap().sample.rotation;
        assert!(second.angle_to(&about(a2, v2)) < 1e-15);
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

    fn rotation_and_translation() -> Geometry {
        Geometry::builder("t")
            .sample_rotation("r", [0.0, 0.0, 1.0])
            .sample_translation("t", [1.0, 0.0, 0.0])
            .build()
            .unwrap()
    }

    #[test]
    fn wrap_is_the_identity_on_the_window_and_half_open() {
        let mut g = rotation_and_translation();
        let r = &g.axes()[0];
        assert_eq!(r.cut_point(), Some(-PI));
        assert_eq!(r.wrap(-PI), -PI);
        assert_eq!(r.wrap(PI), -PI);
        let inside = 0.1 + 1e-17;
        assert_eq!(r.wrap(inside).to_bits(), inside.to_bits());
        assert!((r.wrap(3.5) - (3.5 - TAU)).abs() <= 1e-15);
        assert!((r.wrap(-7.0) - (-7.0 + TAU)).abs() <= 1e-15);
        // One ulp below the cut must not land on the open end.
        let below = f64::from_bits((-PI).to_bits() + 1);
        assert!(below < -PI);
        let w = r.wrap(below);
        assert!((-PI..PI).contains(&w), "{w}");
        g.set_cut_point("r", 0.0).unwrap();
        let r = &g.axes()[0];
        assert_eq!(r.wrap(0.0), 0.0);
        assert_eq!(r.wrap(TAU), 0.0);
        assert!((r.wrap(-0.5) - (TAU - 0.5)).abs() <= 1e-15);
        let t = &g.axes()[1];
        assert_eq!(t.cut_point(), None);
        assert_eq!(t.wrap(9.0), 9.0);
    }

    #[test]
    fn limits_are_inclusive_and_checked() {
        let mut g = rotation_and_translation();
        assert_eq!(g.axes()[0].limits(), (f64::NEG_INFINITY, f64::INFINITY));
        g.set_limits("t", -0.5, 0.5).unwrap();
        let t = &g.axes()[1];
        assert!(t.within_limits(-0.5) && t.within_limits(0.5));
        assert!(!t.within_limits(f64::from_bits(0.5f64.to_bits() + 1)));
        g.set_limits("r", 1.0, 1.0).unwrap();
        g.set_limits("r", f64::NEG_INFINITY, 2.0).unwrap();
        assert!(g.axes()[0].within_limits(-1e300));
        for (min, max) in [(1.0, 0.0), (f64::NAN, 1.0), (0.0, f64::NAN)] {
            assert!(matches!(
                g.set_limits("r", min, max),
                Err(Error::AxisLimits { .. })
            ));
        }
        assert_eq!(g.axes()[0].limits(), (f64::NEG_INFINITY, 2.0));
        assert_eq!(
            g.set_limits("x", 0.0, 1.0),
            Err(Error::UnknownAxis { name: "x".into() })
        );
        for cut in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                g.set_cut_point("r", cut),
                Err(Error::CutPoint { .. })
            ));
        }
        assert_eq!(
            g.set_cut_point("t", 0.0),
            Err(Error::NoCutPoint { name: "t".into() })
        );
        assert_eq!(g.axes()[0].cut_point(), Some(-PI));
    }

    #[test]
    fn source_stack_rotates_the_beam_and_empty_is_identity() {
        let empty = Geometry::builder("t")
            .sample_rotation("r", [0.0, 0.0, 1.0])
            .build()
            .unwrap();
        assert_eq!(
            empty.poses(&[0.3]).unwrap().source,
            UnitQuaternion::identity()
        );
        // outer x then inner y: R_x(a) R_y(b) x̂ = R_x(a) (cos b, 0, -sin b)
        let g = Geometry::builder("t")
            .source_rotation("a", [1.0, 0.0, 0.0])
            .source_rotation("b", [0.0, 1.0, 0.0])
            .build()
            .unwrap();
        assert_eq!(g.source_stack(), &[0, 1]);
        let (a, b) = (0.7_f64, 0.2_f64);
        let ki = g.poses(&[a, b]).unwrap().source * g.beam_direction();
        let want = Vector3::new(b.cos(), a.sin() * b.sin(), -a.cos() * b.sin());
        assert!((ki.into_inner() - want).norm() < 1e-15, "{ki:?}");
    }
}
