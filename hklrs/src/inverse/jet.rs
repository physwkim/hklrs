//! Forward-mode derivatives along the written axes, so each residual and its
//! Jacobian row come from one expression.

use nalgebra::{UnitQuaternion, Vector3};

#[derive(Clone, Debug)]
pub(super) struct Scalar {
    pub v: f64,
    pub d: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(super) struct Vec3 {
    pub v: Vector3<f64>,
    pub d: Vec<Vector3<f64>>,
}

impl Scalar {
    pub fn atan2(y: &Scalar, x: &Scalar) -> Scalar {
        let r2 = x.v * x.v + y.v * y.v;
        Scalar {
            v: y.v.atan2(x.v),
            d: x.d
                .iter()
                .zip(&y.d)
                .map(|(dx, dy)| (x.v * dy - y.v * dx) / r2)
                .collect(),
        }
    }

    pub fn sub(&self, o: &Scalar) -> Scalar {
        Scalar {
            v: self.v - o.v,
            d: self.d.iter().zip(&o.d).map(|(a, b)| a - b).collect(),
        }
    }

    pub fn div(&self, o: &Scalar) -> Scalar {
        Scalar {
            v: self.v / o.v,
            d: self
                .d
                .iter()
                .zip(&o.d)
                .map(|(a, b)| (a * o.v - self.v * b) / (o.v * o.v))
                .collect(),
        }
    }
}

impl Vec3 {
    /// `rotation · u` for a constant `u`; `axes[j]` is the lab direction of
    /// written axis j in the rotating stack (zero when it is not in it), and
    /// d(R u)/dθⱼ = axes[j] × R u.
    pub fn rotated(
        rotation: &UnitQuaternion<f64>,
        axes: &[Vector3<f64>],
        u: &Vector3<f64>,
    ) -> Vec3 {
        let v = rotation * u;
        Vec3 {
            d: axes.iter().map(|a| a.cross(&v)).collect(),
            v,
        }
    }

    pub fn component(&self, i: usize) -> Scalar {
        Scalar {
            v: self.v[i],
            d: self.d.iter().map(|d| d[i]).collect(),
        }
    }

    pub fn add(&self, o: &Vec3) -> Vec3 {
        Vec3 {
            v: self.v + o.v,
            d: self.d.iter().zip(&o.d).map(|(a, b)| a + b).collect(),
        }
    }

    pub fn sub(&self, o: &Vec3) -> Vec3 {
        Vec3 {
            v: self.v - o.v,
            d: self.d.iter().zip(&o.d).map(|(a, b)| a - b).collect(),
        }
    }

    pub fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3 {
            v: self.v.cross(&o.v),
            d: self
                .d
                .iter()
                .zip(&o.d)
                .map(|(a, b)| a.cross(&o.v) + self.v.cross(b))
                .collect(),
        }
    }

    pub fn dot(&self, o: &Vec3) -> Scalar {
        Scalar {
            v: self.v.dot(&o.v),
            d: self
                .d
                .iter()
                .zip(&o.d)
                .map(|(a, b)| a.dot(&o.v) + self.v.dot(b))
                .collect(),
        }
    }

    pub fn scaled(&self, s: &Scalar) -> Vec3 {
        Vec3 {
            v: self.v * s.v,
            d: self
                .d
                .iter()
                .zip(&s.d)
                .map(|(a, ds)| a * s.v + self.v * *ds)
                .collect(),
        }
    }

    /// Constant along every one of `n` written axes.
    pub fn constant(v: Vector3<f64>, n: usize) -> Vec3 {
        Vec3 {
            v,
            d: vec![Vector3::zeros(); n],
        }
    }

    /// |v|; at v = 0, where it has no derivative, the derivative is zero.
    pub fn norm(&self) -> Scalar {
        let n = self.v.norm();
        Scalar {
            v: n,
            d: self
                .d
                .iter()
                .map(|d| if n > 0.0 { self.v.dot(d) / n } else { 0.0 })
                .collect(),
        }
    }

    /// The unit vector, `None` when |v| ≤ `min`.
    pub fn normalized(&self, min: f64) -> Option<Vec3> {
        let n = self.v.norm();
        if n <= min {
            return None;
        }
        let u = self.v / n;
        Some(Vec3 {
            d: self.d.iter().map(|d| (d - u * u.dot(d)) / n).collect(),
            v: u,
        })
    }
}
