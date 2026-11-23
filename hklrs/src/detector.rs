//! Pixel detector in the detector-arm frame (all axes at zero), metres.
//!
//! A detector is a set of modules that tile its image. A module is a
//! rectangle of the image placed in the arm frame: a corner and two pixel
//! axes, each a direction with the pixel coordinates along it. A regular
//! detector is one module of uniform pixels; a point detector is one pixel
//! without extent and a linear detector one row.

use std::ops::Range;

use nalgebra::{Isometry3, Point3, Unit, UnitQuaternion, Vector3};

use crate::{Error, PARALLEL_SIN};

/// Pixels along one module axis, coordinates in metres from the module
/// corner.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelAxis(Kind);

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    /// One pixel without extent, at the corner.
    Point,
    /// `n` pixels of `pitch`, pixel i spanning `i·pitch..(i + 1)·pitch`.
    Uniform {
        dir: Unit<Vector3<f64>>,
        n: usize,
        pitch: f64,
    },
    /// Pixel i spans `edges[i]..edges[i + 1]`.
    Edges {
        dir: Unit<Vector3<f64>>,
        edges: Vec<f64>,
    },
}

fn finite_dir(dir: &Unit<Vector3<f64>>) -> Result<(), Error> {
    if dir.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::PixelAxis {
            reason: "the direction is not finite",
        })
    }
}

impl PixelAxis {
    pub fn point() -> Self {
        Self(Kind::Point)
    }

    pub fn uniform(dir: Unit<Vector3<f64>>, n: usize, pitch: f64) -> Result<Self, Error> {
        finite_dir(&dir)?;
        if n == 0 || !(pitch.is_finite() && pitch > 0.0) {
            return Err(Error::PixelAxis {
                reason: "a uniform axis needs a pixel and a finite positive pitch",
            });
        }
        Ok(Self(Kind::Uniform { dir, n, pitch }))
    }

    /// Pixel i spans `edges[i]..edges[i + 1]`, so `edges` holds one more
    /// value than there are pixels.
    pub fn edges(dir: Unit<Vector3<f64>>, edges: Vec<f64>) -> Result<Self, Error> {
        finite_dir(&dir)?;
        if edges.len() < 2
            || !edges.iter().all(|e| e.is_finite())
            || !edges.windows(2).all(|w| w[0] < w[1])
        {
            return Err(Error::PixelAxis {
                reason: "edges must be finite and increasing, two or more",
            });
        }
        Ok(Self(Kind::Edges { dir, edges }))
    }

    pub fn pixel_count(&self) -> usize {
        match &self.0 {
            Kind::Point => 1,
            Kind::Uniform { n, .. } => *n,
            Kind::Edges { edges, .. } => edges.len() - 1,
        }
    }

    /// `None` for a point, which has no extent.
    pub fn dir(&self) -> Option<Unit<Vector3<f64>>> {
        match &self.0 {
            Kind::Point => None,
            Kind::Uniform { dir, .. } | Kind::Edges { dir, .. } => Some(*dir),
        }
    }

    /// The pitch of a uniform axis; `None` for a point or edges.
    pub fn pitch(&self) -> Option<f64> {
        match &self.0 {
            Kind::Uniform { pitch, .. } => Some(*pitch),
            _ => None,
        }
    }

    /// Coordinate of the centre of pixel `i`; panics unless `i <
    /// pixel_count()`.
    pub fn centre(&self, i: usize) -> f64 {
        assert!(
            i < self.pixel_count(),
            "pixel {i} of an axis of {}",
            self.pixel_count()
        );
        match &self.0 {
            Kind::Point => 0.0,
            Kind::Uniform { pitch, .. } => (i as f64 + 0.5) * pitch,
            Kind::Edges { edges, .. } => 0.5 * (edges[i] + edges[i + 1]),
        }
    }

    /// Extent of pixel `i`, zero for a point; panics unless `i <
    /// pixel_count()`.
    pub fn width(&self, i: usize) -> f64 {
        assert!(
            i < self.pixel_count(),
            "pixel {i} of an axis of {}",
            self.pixel_count()
        );
        match &self.0 {
            Kind::Point => 0.0,
            Kind::Uniform { pitch, .. } => *pitch,
            Kind::Edges { edges, .. } => edges[i + 1] - edges[i],
        }
    }

    /// From the corner to the centre of pixel `i`.
    fn offset(&self, i: usize) -> Vector3<f64> {
        self.offset_by(self.centre(i))
    }

    /// `distance` along the axis direction; zero for a point.
    fn offset_by(&self, distance: f64) -> Vector3<f64> {
        self.dir()
            .map_or_else(Vector3::zeros, |d| d.into_inner() * distance)
    }

    /// Pixels `range` as an axis of their own, and how far its corner moves
    /// along this axis.
    fn slice(&self, range: Range<usize>) -> (Self, f64) {
        match &self.0 {
            Kind::Point => (self.clone(), 0.0),
            Kind::Uniform { dir, pitch, .. } => (
                Self(Kind::Uniform {
                    dir: *dir,
                    n: range.len(),
                    pitch: *pitch,
                }),
                range.start as f64 * pitch,
            ),
            Kind::Edges { dir, edges } => (
                Self(Kind::Edges {
                    dir: *dir,
                    edges: edges[range.start..=range.end].to_vec(),
                }),
                0.0,
            ),
        }
    }

    /// The first `blocks` runs of `k` pixels, each one pixel spanning its
    /// run.
    fn bin(&self, k: usize, blocks: usize) -> Self {
        match &self.0 {
            Kind::Point => self.clone(),
            Kind::Uniform { dir, pitch, .. } => Self(Kind::Uniform {
                dir: *dir,
                n: blocks,
                pitch: pitch * k as f64,
            }),
            Kind::Edges { dir, edges } => Self(Kind::Edges {
                dir: *dir,
                edges: edges.iter().step_by(k).take(blocks + 1).copied().collect(),
            }),
        }
    }

    fn rotated(&self, rotation: &UnitQuaternion<f64>) -> Self {
        let mut axis = self.clone();
        match &mut axis.0 {
            Kind::Point => {}
            Kind::Uniform { dir, .. } | Kind::Edges { dir, .. } => *dir = rotation * *dir,
        }
        axis
    }
}

/// A rectangle of the image from `at` (slow, fast), its pixels placed at
/// `corner + slow coordinate · slow direction + fast coordinate · fast
/// direction`.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    at: (usize, usize),
    corner: Point3<f64>,
    slow: PixelAxis,
    fast: PixelAxis,
}

impl Module {
    pub fn new(
        at: (usize, usize),
        corner: Point3<f64>,
        slow: PixelAxis,
        fast: PixelAxis,
    ) -> Result<Self, Error> {
        finite(&corner.coords, "module corner")?;
        if let (Some(s), Some(f)) = (slow.dir(), fast.dir()) {
            let sin = s.cross(&f.into_inner()).norm();
            if sin < PARALLEL_SIN {
                return Err(Error::ParallelDirections {
                    what: "module pixel axes",
                });
            }
        }
        Ok(Self {
            at,
            corner,
            slow,
            fast,
        })
    }

    /// Image index of the module's pixel (0, 0).
    pub fn at(&self) -> (usize, usize) {
        self.at
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.slow.pixel_count(), self.fast.pixel_count())
    }

    pub fn corner(&self) -> Point3<f64> {
        self.corner
    }

    pub fn slow(&self) -> &PixelAxis {
        &self.slow
    }

    pub fn fast(&self) -> &PixelAxis {
        &self.fast
    }

    /// Unit normal `slow × fast`; `None` when an axis is a point, so the
    /// pixels span no area.
    pub fn normal(&self) -> Option<Unit<Vector3<f64>>> {
        let (s, f) = (self.slow.dir()?, self.fast.dir()?);
        Some(Unit::new_normalize(s.cross(&f.into_inner())))
    }

    /// Centre of the module's pixel (slow, fast); panics outside its shape.
    pub(crate) fn pixel(&self, slow: usize, fast: usize) -> Point3<f64> {
        self.corner + self.slow.offset(slow) + self.fast.offset(fast)
    }

    /// Area vector of the module's pixel (slow, fast), along
    /// `slow × fast`, m²; zero when [`Module::normal`] is `None`.
    pub(crate) fn area(&self, slow: usize, fast: usize) -> Vector3<f64> {
        match (self.slow.dir(), self.fast.dir()) {
            (Some(s), Some(f)) => {
                s.cross(&f.into_inner()) * (self.slow.width(slow) * self.fast.width(fast))
            }
            _ => Vector3::zeros(),
        }
    }

    fn covers(&self, slow: usize, fast: usize) -> bool {
        let (n1, n2) = self.shape();
        (self.at.0..self.at.0 + n1).contains(&slow) && (self.at.1..self.at.1 + n2).contains(&fast)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PixelDetector {
    shape: (usize, usize),
    /// Tile the image: every pixel is in exactly one.
    modules: Vec<Module>,
    /// Row-major over `shape`, `Some` only when a pixel is masked, so an
    /// all-clear mask and no mask are one value.
    mask: Option<Vec<bool>>,
}

fn finite(v: &Vector3<f64>, what: &str) -> Result<(), Error> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::ZeroVector {
            what: what.to_owned(),
        })
    }
}

fn any_masked(masked: Vec<bool>) -> Option<Vec<bool>> {
    masked.contains(&true).then_some(masked)
}

/// The axis of `n` pixels one `step` apart; a zero step is a point, which
/// only an axis one pixel long can be.
fn step_axis(n: usize, step: Vector3<f64>, what: &str) -> Result<PixelAxis, Error> {
    match Unit::try_new_and_get(step, 0.0) {
        Some((dir, pitch)) => PixelAxis::uniform(dir, n, pitch),
        None if n == 1 => Ok(PixelAxis::point()),
        None => Err(Error::ZeroVector {
            what: what.to_owned(),
        }),
    }
}

impl PixelDetector {
    /// One module of uniform pixels: `origin` is the centre of pixel
    /// (0, 0) and `e_slow`/`e_fast` are one-pixel steps, so their lengths
    /// are the pixel pitches. A zero step makes an axis one pixel long a
    /// point, without extent; along a longer axis it is refused.
    pub fn new(
        shape: (usize, usize),
        origin: Vector3<f64>,
        e_slow: Vector3<f64>,
        e_fast: Vector3<f64>,
    ) -> Result<Self, Error> {
        if shape.0 == 0 || shape.1 == 0 {
            return Err(Error::DetectorShape {
                slow: shape.0,
                fast: shape.1,
            });
        }
        finite(&origin, "detector origin")?;
        finite(&e_slow, "e_slow")?;
        finite(&e_fast, "e_fast")?;
        let module = Module::new(
            (0, 0),
            Point3::from(origin - (e_slow + e_fast) / 2.0),
            step_axis(shape.0, e_slow, "e_slow")?,
            step_axis(shape.1, e_fast, "e_fast")?,
        )?;
        Self::from_modules(shape, vec![module])
    }

    /// One pixel at `distance` metres along `direction`.
    pub fn point(direction: Unit<Vector3<f64>>, distance: f64) -> Result<Self, Error> {
        if !(distance.is_finite() && distance > 0.0) {
            return Err(Error::NotPositive {
                what: "detector distance",
                value: distance,
            });
        }
        Self::new(
            (1, 1),
            direction.into_inner() * distance,
            Vector3::zeros(),
            Vector3::zeros(),
        )
    }

    /// `modules` must tile the image of `shape`: each inside it, none
    /// overlapping, every pixel in one.
    pub fn from_modules(shape: (usize, usize), modules: Vec<Module>) -> Result<Self, Error> {
        if shape.0 == 0 || shape.1 == 0 {
            return Err(Error::DetectorShape {
                slow: shape.0,
                fast: shape.1,
            });
        }
        let span = |m: &Module| {
            let ((a1, a2), (n1, n2)) = (m.at, m.shape());
            (a1..a1 + n1, a2..a2 + n2)
        };
        let overlap = |a: &Range<usize>, b: &Range<usize>| a.start < b.end && b.start < a.end;
        let inside = modules.iter().all(|m| {
            let (s, f) = span(m);
            s.end <= shape.0 && f.end <= shape.1
        });
        let disjoint = modules.iter().enumerate().all(|(i, a)| {
            let (s, f) = span(a);
            modules[i + 1..].iter().all(|b| {
                let (t, g) = span(b);
                !(overlap(&s, &t) && overlap(&f, &g))
            })
        });
        let covered: usize = modules.iter().map(|m| m.shape().0 * m.shape().1).sum();
        if !(inside && disjoint && covered == shape.0 * shape.1) {
            return Err(Error::ModuleTiling { shape });
        }
        Ok(Self {
            shape,
            modules,
            mask: None,
        })
    }

    /// `masked[slow * shape.1 + fast]` excludes pixel (slow, fast); replaces
    /// any earlier mask.
    pub fn with_mask(self, masked: Vec<bool>) -> Result<Self, Error> {
        if self.shape.0.checked_mul(self.shape.1) != Some(masked.len()) {
            return Err(Error::MaskShape {
                len: masked.len(),
                shape: self.shape,
            });
        }
        Ok(Self {
            mask: any_masked(masked),
            ..self
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn modules(&self) -> &[Module] {
        &self.modules
    }

    /// Row-major mask over the shape, `None` when no pixel is masked.
    pub fn mask(&self) -> Option<&[bool]> {
        self.mask.as_deref()
    }

    fn index(&self, slow: usize, fast: usize) -> Result<usize, Error> {
        if slow >= self.shape.0 || fast >= self.shape.1 {
            return Err(Error::PixelOutside {
                slow,
                fast,
                shape: self.shape,
            });
        }
        Ok(slow * self.shape.1 + fast)
    }

    /// Centre of pixel (slow, fast) in the detector-arm frame.
    pub fn pixel(&self, slow: usize, fast: usize) -> Result<Point3<f64>, Error> {
        self.index(slow, fast)?;
        let m = self
            .modules
            .iter()
            .find(|m| m.covers(slow, fast))
            .expect("the modules tile the image");
        Ok(m.pixel(slow - m.at.0, fast - m.at.1))
    }

    pub fn is_masked(&self, slow: usize, fast: usize) -> Result<bool, Error> {
        let i = self.index(slow, fast)?;
        Ok(self.mask.as_ref().is_some_and(|m| m[i]))
    }

    /// Every module moved by `motion`.
    pub(crate) fn transformed(&self, motion: &Isometry3<f64>) -> Self {
        let modules = self
            .modules
            .iter()
            .map(|m| Module {
                at: m.at,
                corner: motion * m.corner,
                slow: m.slow.rotated(&motion.rotation),
                fast: m.fast.rotated(&motion.rotation),
            })
            .collect();
        Self {
            shape: self.shape,
            modules,
            mask: self.mask.clone(),
        }
    }

    /// Pixels `slow` x `fast` as a detector of their own, whose pixel (0, 0)
    /// is this detector's (slow.start, fast.start); a module keeps the part
    /// of it inside the roi.
    pub fn roi(&self, slow: Range<usize>, fast: Range<usize>) -> Result<Self, Error> {
        if slow.is_empty() || fast.is_empty() || slow.end > self.shape.0 || fast.end > self.shape.1
        {
            return Err(Error::Roi {
                slow,
                fast,
                shape: self.shape,
            });
        }
        let clip = |at: usize, n: usize, roi: &Range<usize>| {
            at.max(roi.start) - at..(at + n).min(roi.end).saturating_sub(at)
        };
        let modules = self
            .modules
            .iter()
            .filter_map(|m| {
                let (n1, n2) = m.shape();
                let (r1, r2) = (clip(m.at.0, n1, &slow), clip(m.at.1, n2, &fast));
                if r1.is_empty() || r2.is_empty() {
                    return None;
                }
                let at = (
                    m.at.0 + r1.start - slow.start,
                    m.at.1 + r2.start - fast.start,
                );
                let (s, ds) = m.slow.slice(r1);
                let (f, df) = m.fast.slice(r2);
                Some(Module {
                    at,
                    corner: m.corner + m.slow.offset_by(ds) + m.fast.offset_by(df),
                    slow: s,
                    fast: f,
                })
            })
            .collect();
        let width = self.shape.1;
        let mask = self.mask.as_ref().map(|m| {
            slow.clone()
                .flat_map(|s| fast.clone().map(move |f| m[s * width + f]))
                .collect()
        });
        Ok(Self {
            shape: (slow.len(), fast.len()),
            modules,
            mask: mask.and_then(any_masked),
        })
    }

    /// Blocks of `slow` x `fast` pixels as one pixel each, spanning the
    /// block. Trailing pixels that fill no block are dropped, with any
    /// module left without a block; a block is masked when any of its
    /// pixels is. A block must lie in one module.
    pub fn bin(&self, slow: usize, fast: usize) -> Result<Self, Error> {
        if slow == 0 || fast == 0 || slow > self.shape.0 || fast > self.shape.1 {
            return Err(Error::Bin {
                slow,
                fast,
                shape: self.shape,
            });
        }
        let shape = (self.shape.0 / slow, self.shape.1 / fast);
        // the binned module's pixels, as a range of raw pixels cut at the
        // last whole block
        let cut = |at: usize, n: usize, k: usize, blocks: usize| {
            let end = blocks * k;
            let range = at.min(end)..(at + n).min(end);
            (range.start % k == 0 && range.end % k == 0).then_some(range)
        };
        let mut modules = Vec::with_capacity(self.modules.len());
        for m in &self.modules {
            let (n1, n2) = m.shape();
            let (Some(r1), Some(r2)) = (
                cut(m.at.0, n1, slow, shape.0),
                cut(m.at.1, n2, fast, shape.1),
            ) else {
                return Err(Error::BinModule {
                    slow,
                    fast,
                    at: m.at,
                });
            };
            if r1.is_empty() || r2.is_empty() {
                continue;
            }
            modules.push(Module {
                at: (r1.start / slow, r2.start / fast),
                corner: m.corner,
                slow: m.slow.bin(slow, r1.len() / slow),
                fast: m.fast.bin(fast, r2.len() / fast),
            });
        }
        let width = self.shape.1;
        let mask = self.mask.as_ref().map(|m| {
            (0..shape.0)
                .flat_map(|bs| {
                    (0..shape.1).map(move |bf| {
                        (bs * slow..(bs + 1) * slow)
                            .any(|s| (bf * fast..(bf + 1) * fast).any(|f| m[s * width + f]))
                    })
                })
                .collect()
        });
        Ok(Self {
            shape,
            modules,
            mask: mask.and_then(any_masked),
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Translation3;

    use super::*;

    fn area() -> PixelDetector {
        PixelDetector::new(
            (5, 7),
            Vector3::new(0.3, -0.02, 0.01),
            Vector3::new(1e-3, 2e-4, -3e-4),
            Vector3::new(-1e-4, 7e-4, 5e-4),
        )
        .unwrap()
    }

    fn close(a: Point3<f64>, b: Point3<f64>) -> bool {
        (a - b).norm() <= 1e-15
    }

    const FAST_EDGES: [f64; 5] = [0.0, 1e-3, 3.5e-3, 4.5e-3, 5.5e-3];

    /// Shape (4, 6): a uniform module in columns 0-1 and, in columns 2-5,
    /// one with uneven fast pixels, placed and turned apart from it.
    fn tiled() -> PixelDetector {
        let a = Module::new(
            (0, 0),
            Point3::new(0.5, 0.0, 0.0),
            PixelAxis::uniform(Vector3::z_axis(), 4, 1e-3).unwrap(),
            PixelAxis::uniform(-Vector3::y_axis(), 2, 2e-3).unwrap(),
        )
        .unwrap();
        let b = Module::new(
            (0, 2),
            Point3::new(0.49, -0.005, 0.0),
            PixelAxis::uniform(Vector3::z_axis(), 4, 1e-3).unwrap(),
            PixelAxis::edges(
                Unit::new_normalize(Vector3::new(-0.1, -1.0, 0.0)),
                FAST_EDGES.to_vec(),
            )
            .unwrap(),
        )
        .unwrap();
        PixelDetector::from_modules((4, 6), vec![a, b]).unwrap()
    }

    #[test]
    fn new_is_one_module_of_the_steps_pitch() {
        let det = area();
        let origin = Vector3::new(0.3, -0.02, 0.01);
        let (e_slow, e_fast) = (
            Vector3::new(1e-3, 2e-4, -3e-4),
            Vector3::new(-1e-4, 7e-4, 5e-4),
        );
        let [m] = det.modules() else {
            panic!("one module")
        };
        assert_eq!((m.at(), m.shape()), ((0, 0), (5, 7)));
        assert_eq!(m.slow().pitch(), Some(e_slow.norm()));
        assert_eq!(m.fast().pitch(), Some(e_fast.norm()));
        for (s, f) in [(0, 0), (4, 0), (0, 6), (3, 5)] {
            let p = origin + e_slow * s as f64 + e_fast * f as f64;
            assert!(close(det.pixel(s, f).unwrap(), Point3::from(p)));
        }
    }

    #[test]
    fn a_zero_step_is_a_point_only_along_one_pixel() {
        let point = PixelDetector::point(Vector3::x_axis(), 0.7).unwrap();
        let [m] = point.modules() else {
            panic!("one module")
        };
        assert_eq!(
            (m.slow(), m.fast()),
            (&PixelAxis::point(), &PixelAxis::point())
        );
        assert_eq!(m.normal(), None);
        assert_eq!(point.pixel(0, 0).unwrap(), Point3::new(0.7, 0.0, 0.0));

        let e_fast = Vector3::new(0.0, 0.0, 1e-3);
        let row = PixelDetector::new((1, 9), Vector3::x(), Vector3::zeros(), e_fast).unwrap();
        let m = &row.modules()[0];
        assert_eq!(
            (m.slow(), m.fast().pitch()),
            (&PixelAxis::point(), Some(1e-3))
        );
        assert_eq!(
            row.bin(1, 3).unwrap().modules()[0].fast().pitch(),
            Some(3e-3)
        );
        assert_eq!(
            PixelDetector::new((2, 9), Vector3::x(), Vector3::zeros(), e_fast),
            Err(Error::ZeroVector {
                what: "e_slow".to_owned()
            })
        );
    }

    #[test]
    fn modules_must_tile_the_image() {
        let ms = tiled().modules().to_vec();
        let module = |at: (usize, usize)| Module {
            at,
            ..ms[1].clone()
        };
        for (shape, modules) in [
            // pixels uncovered
            ((4, 7), ms.clone()),
            ((4, 6), vec![ms[0].clone()]),
            ((4, 6), vec![]),
            // outside
            ((4, 6), vec![ms[0].clone(), module((0, 3))]),
            // overlapping, with as many pixels as the image
            ((4, 6), vec![ms[0].clone(), module((0, 1))]),
        ] {
            assert_eq!(
                PixelDetector::from_modules(shape, modules),
                Err(Error::ModuleTiling { shape })
            );
        }
    }

    #[test]
    fn axes_and_modules_refuse_degenerate_input() {
        let x = Vector3::x_axis();
        for axis in [
            PixelAxis::uniform(x, 0, 1e-3),
            PixelAxis::uniform(x, 3, 0.0),
            PixelAxis::uniform(x, 3, f64::NAN),
            PixelAxis::uniform(
                Unit::new_unchecked(Vector3::new(f64::NAN, 0.0, 0.0)),
                3,
                1e-3,
            ),
            PixelAxis::edges(x, vec![0.0]),
            PixelAxis::edges(x, vec![0.0, 1e-3, 1e-3]),
            PixelAxis::edges(x, vec![0.0, f64::INFINITY]),
        ] {
            assert!(matches!(axis, Err(Error::PixelAxis { .. })), "{axis:?}");
        }
        let skew = Unit::new_normalize(Vector3::new(1.0, 1e-7, 0.0));
        assert_eq!(
            Module::new(
                (0, 0),
                Point3::origin(),
                PixelAxis::uniform(x, 2, 1e-3).unwrap(),
                PixelAxis::uniform(skew, 2, 1e-3).unwrap(),
            ),
            Err(Error::ParallelDirections {
                what: "module pixel axes"
            })
        );
    }

    #[test]
    fn a_pixel_is_at_its_module_coordinates() {
        let det = tiled();
        let dir = Unit::new_normalize(Vector3::new(-0.1, -1.0, 0.0)).into_inner();
        for s in 0..4 {
            let z = Vector3::z() * ((s as f64 + 0.5) * 1e-3);
            for f in 0..2 {
                let y = Vector3::y() * ((f as f64 + 0.5) * 2e-3);
                assert_eq!(det.pixel(s, f).unwrap(), Point3::new(0.5, 0.0, 0.0) + z - y);
            }
            for f in 0..4 {
                let c = 0.5 * (FAST_EDGES[f] + FAST_EDGES[f + 1]);
                let p = Point3::new(0.49, -0.005, 0.0) + z + dir * c;
                assert!(close(det.pixel(s, f + 2).unwrap(), p));
            }
        }
        assert!(matches!(det.pixel(4, 0), Err(Error::PixelOutside { .. })));
    }

    #[test]
    fn roi_pixel_is_the_parent_pixel_offset_by_the_roi_start() {
        for det in [area(), tiled()] {
            let (n1, n2) = det.shape();
            for (slow, fast) in [(1..4, 2..n2), (0..n1, 1..3), (2..3, 3..4)] {
                let r = det.roi(slow.clone(), fast.clone()).unwrap();
                assert_eq!(r.shape(), (slow.len(), fast.len()));
                for s in 0..slow.len() {
                    for f in 0..fast.len() {
                        assert!(close(
                            r.pixel(s, f).unwrap(),
                            det.pixel(s + slow.start, f + fast.start).unwrap()
                        ));
                    }
                }
            }
        }
        // the roi keeps the part of each module inside it
        let r = tiled().roi(0..4, 3..5).unwrap();
        let [m] = r.modules() else {
            panic!("one module")
        };
        assert_eq!((m.at(), m.shape()), ((0, 0), (4, 2)));
    }

    #[test]
    fn roi_rejects_empty_and_outside_ranges() {
        let det = area();
        for (slow, fast) in [(2..2, 0..7), (0..5, 3..3), (0..6, 0..7), (0..5, 1..8)] {
            assert!(matches!(det.roi(slow, fast), Err(Error::Roi { .. })));
        }
    }

    #[test]
    fn bin_pixel_is_its_block_centre_and_drops_the_remainder() {
        let det = area();
        let b = det.bin(2, 3).unwrap();
        assert_eq!(b.shape(), (2, 2));
        for bs in 0..2 {
            for bf in 0..2 {
                let sum = (0..2)
                    .flat_map(|s| (0..3).map(move |f| (2 * bs + s, 3 * bf + f)))
                    .map(|(s, f)| det.pixel(s, f).unwrap().coords)
                    .sum::<Vector3<f64>>();
                assert!(close(b.pixel(bs, bf).unwrap(), Point3::from(sum / 6.0)));
            }
        }
        assert_eq!(det.bin(1, 1).unwrap(), det);
        let t = tiled();
        assert_eq!(t.bin(1, 1).unwrap(), t);
    }

    #[test]
    fn a_binned_pixel_spans_its_block_in_one_module() {
        let det = tiled();
        let b = det.bin(2, 2).unwrap();
        assert_eq!(b.shape(), (2, 3));
        let m = &b.modules()[1];
        assert_eq!((m.at(), m.shape()), ((0, 1), (2, 2)));
        assert_eq!(m.fast().centre(1), 0.5 * (3.5e-3 + 5.5e-3));
        assert_eq!(m.fast().width(1), 5.5e-3 - 3.5e-3);
        assert_eq!(m.slow().pitch(), Some(2e-3));
        let dir = m.fast().dir().unwrap().into_inner();
        let p = Point3::new(0.49, -0.005, 0.0) + Vector3::z() * 3e-3 + dir * 4.5e-3;
        assert!(close(b.pixel(1, 2).unwrap(), p));

        // a block of columns 1-3 would hold pixels of both modules
        assert_eq!(
            det.bin(1, 3),
            Err(Error::BinModule {
                slow: 1,
                fast: 3,
                at: (0, 0)
            })
        );
        let shifted = det.roi(0..4, 1..6).unwrap();
        assert!(matches!(shifted.bin(1, 2), Err(Error::BinModule { .. })));
        // the remainder is dropped: rows, columns, and a module wholly in it
        let b = det.bin(3, 2).unwrap();
        assert_eq!((b.shape(), b.modules().len()), ((1, 3), 2));
        let narrow = det.roi(0..4, 0..5).unwrap().bin(1, 2).unwrap();
        assert_eq!((narrow.shape(), narrow.modules().len()), ((4, 2), 2));
        let last = det.roi(0..4, 0..3).unwrap().bin(1, 2).unwrap();
        assert_eq!((last.shape(), last.modules().len()), ((4, 1), 1));
    }

    #[test]
    fn bin_rejects_zero_and_oversized_factors() {
        let det = area();
        for (s, f) in [(0, 1), (1, 0), (6, 1), (1, 8)] {
            assert!(matches!(det.bin(s, f), Err(Error::Bin { .. })));
        }
    }

    #[test]
    fn a_linear_detector_is_one_row() {
        let det = PixelDetector::new(
            (1, 9),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 1e-3),
        )
        .unwrap();
        let b = det.roi(0..1, 1..9).unwrap().bin(1, 3).unwrap();
        assert_eq!(b.shape(), (1, 2));
        assert!(close(b.pixel(0, 1).unwrap(), det.pixel(0, 5).unwrap()));
        assert!(matches!(det.bin(2, 1), Err(Error::Bin { .. })));
    }

    #[test]
    fn transformed_moves_every_pixel() {
        let det = tiled();
        let motion = Isometry3::from_parts(
            Translation3::new(0.1, -0.2, 0.3),
            UnitQuaternion::from_euler_angles(0.3, -0.2, 0.7),
        );
        let moved = det.transformed(&motion);
        for s in 0..4 {
            for f in 0..6 {
                let p = motion * det.pixel(s, f).unwrap();
                assert!(close(moved.pixel(s, f).unwrap(), p));
            }
        }
    }

    #[test]
    fn mask_follows_roi_and_bin() {
        let mut m = vec![false; 35];
        m[2 * 7 + 3] = true;
        let det = area().with_mask(m).unwrap();
        assert!(det.is_masked(2, 3).unwrap());
        assert!(!det.is_masked(2, 4).unwrap());
        assert!(matches!(
            det.is_masked(5, 0),
            Err(Error::PixelOutside { .. })
        ));

        let r = det.roi(1..5, 2..6).unwrap();
        assert!(r.is_masked(1, 1).unwrap());
        assert_eq!(r.mask().unwrap().iter().filter(|&&x| x).count(), 1);
        // the roi leaves the masked pixel out: no mask remains
        assert_eq!(det.roi(3..5, 0..7).unwrap().mask(), None);

        // block (1, 1) of 2x3 holds pixel (2, 3); every other block is clear
        let b = det.bin(2, 3).unwrap();
        assert_eq!(b.mask().unwrap(), &[false, false, false, true]);
    }

    #[test]
    fn an_all_clear_mask_is_no_mask() {
        let det = area();
        assert_eq!(det.clone().with_mask(vec![false; 35]).unwrap(), det);
        assert!(matches!(
            det.with_mask(vec![true; 34]),
            Err(Error::MaskShape { len: 34, .. })
        ));
    }
}
