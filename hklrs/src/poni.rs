//! pyFAI PONI geometry to and from a [`PixelDetector`].
//!
//! pyFAI (2026.9.0, `Geometry.calc_pos_zyx`) puts pixel (d1, d2) at
//! `R · (p1 − poni1, p2 − poni2, dist)`, components along its axes 1, 2, 3,
//! with `p = pixel · (physical index + ½)` and `R = R3(rot3)·R2(rot2)·R1(rot1)`
//! (`Geometry.rotation_matrix`). The orientation maps the image index to
//! the physical index (`Detector._reorder_indexes_from_orientation`).
//! [`PoniFile`] reads pyFAI's `.poni` text.

use std::collections::HashMap;

use nalgebra::{Matrix3, Unit, Vector3};
use serde_json::Value;
use thiserror::Error;

use crate::{Error, PixelDetector, Rad, Wavelength};

/// Smallest sine of the angle between `up` and the beam for them to define
/// pyFAI's axis 1.
const UP_BEAM_MIN_SIN: f64 = 1e-6;

/// Largest |cos| between a detector's pixel axes whose shear a PONI, which
/// has none, may drop.
const MAX_SKEW_COS: f64 = 1e-9;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PoniError {
    #[error("{what} is not finite: {value}")]
    NonFinite { what: &'static str, value: f64 },
    #[error("pyFAI orientation {0} is not one of 1-4")]
    Orientation(u32),
    #[error("the up direction is along the beam (sin of their angle = {sin:e})")]
    UpAlongBeam { sin: f64 },
    #[error("a PONI needs two pixels along each axis to read the pitch, got {slow}x{fast}")]
    TooFewPixels { slow: usize, fast: usize },
    #[error("the pixel axes are not perpendicular (cos of their angle = {cos:e})")]
    Skewed { cos: f64 },
    #[error("{key}: {value:?} is not what pyFAI reads there")]
    Syntax { key: String, value: String },
    #[error("PONI version {0} is neither 1 nor in 2-3")]
    Version(f64),
    #[error("the PONI file gives no {0}")]
    Missing(&'static str),
    #[error("pyFAI detector {0:?} takes its pixel layout from pyFAI, not the file")]
    NamedDetector(String),
    #[error("pyFAI's {0} moves the pixels in a way a PONI importer does not model")]
    Unsupported(&'static str),
    #[error(transparent)]
    Core(#[from] Error),
}

/// Image orientation (pyFAI `detectors.orientation.Orientation`, EXIF
/// numbering); pyFAI supports 1-4 and calls 3 its native one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    TopLeft = 1,
    TopRight = 2,
    BottomRight = 3,
    BottomLeft = 4,
}

impl Orientation {
    /// For the slow and the fast axis: the physical index step per image
    /// index (−1 where the image runs against pyFAI's axis) and the physical
    /// index of image index 0. In f64 so that a zero shape reaches
    /// `PixelDetector::new`'s check.
    fn index_map(self, shape: (usize, usize)) -> [(f64, f64); 2] {
        let (flip1, flip2) = match self {
            Orientation::TopLeft => (true, true),
            Orientation::TopRight => (true, false),
            Orientation::BottomRight => (false, false),
            Orientation::BottomLeft => (false, true),
        };
        let map = |flip: bool, n: usize| {
            if flip {
                (-1.0, n as f64 - 1.0)
            } else {
                (1.0, 0.0)
            }
        };
        [map(flip1, shape.0), map(flip2, shape.1)]
    }
}

impl TryFrom<u32> for Orientation {
    type Error = PoniError;

    fn try_from(value: u32) -> Result<Self, PoniError> {
        match value {
            1 => Ok(Orientation::TopLeft),
            2 => Ok(Orientation::TopRight),
            3 => Ok(Orientation::BottomRight),
            4 => Ok(Orientation::BottomLeft),
            _ => Err(PoniError::Orientation(value)),
        }
    }
}

/// Lab directions of pyFAI's axes: axis 3 along the incident beam, axis 1
/// up (pyFAI `doc/source/geometry.rst`, "Default geometry") and axis 2 =
/// axis 1 × axis 3, because pyFAI's (1, 2, 3) is left-handed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabAxes {
    /// Columns: the lab directions of pyFAI axes 1, 2, 3.
    axes: Matrix3<f64>,
}

impl LabAxes {
    /// Axis 1 is the part of `up` normal to `beam`, so `up` need not be
    /// exactly perpendicular to a tilted beam.
    pub fn new(beam: Unit<Vector3<f64>>, up: Unit<Vector3<f64>>) -> Result<Self, PoniError> {
        let sin = beam.cross(&up).norm();
        // NaN when a component is not finite
        if sin.is_nan() || sin < UP_BEAM_MIN_SIN {
            return Err(PoniError::UpAlongBeam { sin });
        }
        let a1 = (up.into_inner() - beam.into_inner() * beam.dot(&up)).normalize();
        let a3 = beam.into_inner();
        Ok(Self {
            axes: Matrix3::from_columns(&[a1, a1.cross(&a3), a3]),
        })
    }
}

/// pyFAI's rotations as `Geometry.rotation_matrix` writes them, in its axis
/// components.
fn rot1(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c)
}

fn rot2(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c)
}

fn rot3(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

fn rotation(r1: f64, r2: f64, r3: f64) -> Matrix3<f64> {
    rot3(r3) * rot2(r2) * rot1(r1)
}

/// A pyFAI detector geometry. Lengths in metres; `shape` is (slow, fast),
/// pyFAI's (dim1, dim2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poni {
    pub dist: f64,
    pub poni1: f64,
    pub poni2: f64,
    pub rot1: Rad,
    pub rot2: Rad,
    pub rot3: Rad,
    pub pixel1: f64,
    pub pixel2: f64,
    pub shape: (usize, usize),
    pub orientation: Orientation,
}

fn finite(what: &'static str, value: f64) -> Result<(), PoniError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PoniError::NonFinite { what, value })
    }
}

fn positive(what: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::NotPositive { what, value })
    }
}

impl Poni {
    /// The detector in the lab, where `lab` places pyFAI's axes, positions
    /// relative to the sample.
    pub fn detector(&self, lab: &LabAxes) -> Result<PixelDetector, PoniError> {
        for (what, v) in [
            ("dist", self.dist),
            ("poni1", self.poni1),
            ("poni2", self.poni2),
            ("rot1", self.rot1.0),
            ("rot2", self.rot2.0),
            ("rot3", self.rot3.0),
        ] {
            finite(what, v)?;
        }
        positive("pixel1", self.pixel1)?;
        positive("pixel2", self.pixel2)?;
        let r = lab.axes * rotation(self.rot1.0, self.rot2.0, self.rot3.0);
        let [(step1, first1), (step2, first2)] = self.orientation.index_map(self.shape);
        let origin = r * Vector3::new(
            self.pixel1 * (first1 + 0.5) - self.poni1,
            self.pixel2 * (first2 + 0.5) - self.poni2,
            self.dist,
        );
        Ok(PixelDetector::new(
            self.shape,
            origin,
            r.column(0) * (step1 * self.pixel1),
            r.column(1) * (step2 * self.pixel2),
        )?)
    }

    /// The PONI of `detector` read with `orientation`; [`Poni::detector`]
    /// of it gives `detector` back. rot2 comes out in [−π/2, π/2] and rot1,
    /// rot3 in [−π, π]; at rot2 = ±π/2, where rot1 and rot3 turn about one
    /// axis, rot3 takes what rot1 leaves.
    pub fn from_detector(
        detector: &PixelDetector,
        orientation: Orientation,
        lab: &LabAxes,
    ) -> Result<Poni, PoniError> {
        let shape = detector.shape();
        if shape.0 < 2 || shape.1 < 2 {
            return Err(PoniError::TooFewPixels {
                slow: shape.0,
                fast: shape.1,
            });
        }
        let p00 = detector.pixel(0, 0)?.coords;
        // across the whole detector, so the rounding of the positions is
        // divided by n − 1
        let e_slow = (detector.pixel(shape.0 - 1, 0)?.coords - p00) / (shape.0 - 1) as f64;
        let e_fast = (detector.pixel(0, shape.1 - 1)?.coords - p00) / (shape.1 - 1) as f64;
        let (pixel1, pixel2) = (e_slow.norm(), e_fast.norm());
        positive("pixel1", pixel1)?;
        positive("pixel2", pixel2)?;
        let [(step1, first1), (step2, first2)] = orientation.index_map(shape);
        let u1 = e_slow * (step1 / pixel1);
        let u2 = e_fast * (step2 / pixel2);
        let cos = u1.dot(&u2);
        if cos.is_nan() || cos.abs() > MAX_SKEW_COS {
            return Err(PoniError::Skewed { cos });
        }
        // pyFAI's axes are left-handed, so axis 3 is u2 × u1
        let r = lab.axes.transpose() * Matrix3::from_columns(&[u1, u2, u2.cross(&u1)]);
        let rot2 = r[(2, 0)].atan2(r[(2, 1)].hypot(r[(2, 2)]));
        let rot1 = (-r[(2, 1)]).atan2(r[(2, 2)]);
        let r3 = r * self::rot1(rot1).transpose() * self::rot2(rot2).transpose();
        let rot3 = r3[(1, 0)].atan2(r3[(0, 0)]);
        // through the rotation the PONI holds, so the round trip closes on it
        let local = (lab.axes * rotation(rot1, rot2, rot3)).transpose() * p00;
        Ok(Poni {
            dist: local.z,
            poni1: pixel1 * (first1 + 0.5) - local.x,
            poni2: pixel2 * (first2 + 0.5) - local.y,
            rot1: Rad(rot1),
            rot2: Rad(rot2),
            rot3: Rad(rot3),
            pixel1,
            pixel2,
            shape,
            orientation,
        })
    }
}

/// Name of pyFAI's generic flat detector, the one whose layout a PONI file
/// gives in full; `Detector.factory` also finds it by its lower-case name.
const GENERIC: &str = "detector";

/// A `.poni` file as pyFAI 2026.9.0 reads it (`io/ponifile.py`
/// `PoniFile.read_from_file` and `read_from_dict`). Lengths in metres.
#[derive(Clone, Debug, PartialEq)]
pub struct PoniFile {
    /// The `poni_version` written (1 if none), raised as pyFAI does: to 2 by
    /// a `Detector_config`, to 2.1 when that mentions an orientation, to 3
    /// by `Parallax`.
    pub version: f64,
    pub detector: DetectorConfig,
    pub parallax: bool,
    pub dist: f64,
    pub poni1: f64,
    pub poni2: f64,
    pub rot1: Rad,
    pub rot2: Rad,
    pub rot3: Rad,
    /// The file's; hklrs takes the wavelength per request, not from the
    /// detector.
    pub wavelength: Option<Wavelength>,
}

/// The detector as a PONI file names it: version 1's `pixelsize1/2` and
/// `splinefile`, or the `Detector_config` entries pyFAI's generic detector
/// takes (`Detector.__init__`, `Detector.factory`).
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorConfig {
    /// pyFAI detector name; `Detector` is the generic flat one.
    pub name: String,
    pub pixel1: Option<f64>,
    pub pixel2: Option<f64>,
    /// Never in a version 1 file, nor for a named detector of the shape
    /// pyFAI gives its class.
    pub max_shape: Option<(usize, usize)>,
    /// `None` where the file has none or 0: the detector class's default.
    pub orientation: Option<Orientation>,
    pub binning: (u64, u64),
    pub splinefile: Option<String>,
}

fn syntax(key: &str, value: impl ToString) -> PoniError {
    PoniError::Syntax {
        key: key.to_string(),
        value: value.to_string(),
    }
}

fn count(key: &str, v: &Value) -> Result<u64, PoniError> {
    v.as_u64().ok_or_else(|| syntax(key, v))
}

fn pair(key: &str, v: &Value) -> Result<(u64, u64), PoniError> {
    match v.as_array().map(Vec::as_slice) {
        Some([a, b]) => Ok((count(key, a)?, count(key, b)?)),
        _ => Err(syntax(key, v)),
    }
}

impl DetectorConfig {
    fn version_1(keys: &HashMap<String, &str>, name: Option<&str>) -> Result<Self, PoniError> {
        let pixel = |key: &str| {
            keys.get(key)
                .map(|v| v.parse::<f64>().map_err(|_| syntax(key, v)))
                .transpose()
        };
        Ok(Self {
            name: name.unwrap_or("Detector").to_string(),
            pixel1: pixel("pixelsize1")?,
            pixel2: pixel("pixelsize2")?,
            max_shape: None,
            orientation: None,
            binning: (1, 1),
            splinefile: keys
                .get("splinefile")
                .filter(|s| s.to_lowercase() != "none")
                .map(|s| s.to_string()),
        })
    }

    fn from_json(name: &str, json: Option<&str>) -> Result<Self, PoniError> {
        let config: HashMap<String, Value> = match json.map(serde_json::from_str) {
            None => HashMap::new(),
            // Detector.factory lower-cases the keys
            Some(Ok(Value::Object(map))) => map
                .into_iter()
                .map(|(k, v)| (k.to_lowercase(), v))
                .collect(),
            Some(_) => return Err(syntax("detector_config", json.unwrap_or_default())),
        };
        let get = |key: &str| config.get(key).filter(|v| !v.is_null());
        // Detector.__init__: a zero pitch is no pitch
        let pixel = |key: &str| match get(key) {
            None => Ok(None),
            Some(v) => match v.as_f64() {
                Some(x) => Ok((x != 0.0).then_some(x)),
                None => Err(syntax(key, v)),
            },
        };
        let max_shape = match get("max_shape") {
            None => None,
            Some(v) => {
                let n = |x: u64| usize::try_from(x).map_err(|_| syntax("max_shape", v));
                let (a, b) = pair("max_shape", v)?;
                Some((n(a)?, n(b)?))
            }
        };
        let orientation = match get("orientation") {
            None => None,
            Some(v) => match u32::try_from(count("orientation", v)?) {
                Ok(0) => None,
                Ok(n) => Some(Orientation::try_from(n)?),
                Err(_) => return Err(syntax("orientation", v)),
            },
        };
        // Detector.factory applies a binning only when it is truthy
        let binning = match get("binning") {
            None => (1, 1),
            Some(v) if v.is_array() => pair("binning", v)?,
            Some(v) => match count("binning", v)? {
                0 => (1, 1),
                n => (n, n),
            },
        };
        if binning.0 == 0 || binning.1 == 0 {
            return Err(syntax("binning", format!("{binning:?}")));
        }
        let splinefile = match get("splinefile") {
            None => None,
            Some(v) => Some(
                v.as_str()
                    .ok_or_else(|| syntax("splinefile", v))?
                    .to_string(),
            ),
        };
        Ok(Self {
            name: name.to_string(),
            pixel1: pixel("pixel1")?,
            pixel2: pixel("pixel2")?,
            max_shape,
            orientation,
            binning,
            splinefile,
        })
    }
}

impl PoniFile {
    pub fn parse(text: &str) -> Result<Self, PoniError> {
        // read_from_file: lines without ':' or from '#' are skipped and a
        // later key wins; Python's text mode ends lines at \n, \r\n and \r
        let mut keys = HashMap::new();
        for line in text.split(['\n', '\r']) {
            if line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once(':') {
                keys.insert(key.trim().to_lowercase(), value.trim());
            }
        }
        let float = |key: &str| {
            keys.get(key)
                .map(|v| v.parse::<f64>().map_err(|_| syntax(key, v)))
                .transpose()
        };
        let need = |key: &'static str| float(key)?.ok_or(PoniError::Missing(key));

        let mut version = float("poni_version")?.unwrap_or(1.0);
        // Python's max keeps a NaN that f64::max drops
        if version.is_nan() {
            return Err(PoniError::Version(version));
        }
        let json = keys.get("detector_config").copied();
        if let Some(json) = json {
            // pyFAI tests the JSON text, not its keys
            version = version.max(if json.contains("orientation") {
                2.1
            } else {
                2.0
            });
        }
        let parallax = keys.get("parallax");
        if parallax.is_some() {
            version = version.max(3.0);
        }
        let name = keys.get("detector").copied();
        let detector = if version == 1.0 {
            DetectorConfig::version_1(&keys, name)?
        } else if (2.0..=3.0).contains(&version) {
            DetectorConfig::from_json(name.ok_or(PoniError::Missing("detector"))?, json)?
        } else {
            // pyFAI refuses above 3 and leaves the detector unset between
            return Err(PoniError::Version(version));
        };
        let dist = match float("distance")? {
            Some(d) => d,
            None => float("dist")?.ok_or(PoniError::Missing("distance"))?,
        };
        Ok(PoniFile {
            version,
            detector,
            parallax: parallax.is_some_and(|v| v.to_lowercase() == "true"),
            dist,
            poni1: need("poni1")?,
            poni2: need("poni2")?,
            rot1: Rad(need("rot1")?),
            rot2: Rad(need("rot2")?),
            rot3: Rad(need("rot3")?),
            wavelength: float("wavelength")?
                .map(|m| Wavelength::new(m * 1e10))
                .transpose()?,
        })
    }

    /// The PONI of pyFAI's generic flat detector. A named detector's pitch,
    /// shape and default orientation come from pyFAI's class, so the caller
    /// builds its [`Poni`] from these fields and that layout.
    pub fn poni(&self) -> Result<Poni, PoniError> {
        let d = &self.detector;
        if d.name.to_lowercase() != GENERIC {
            return Err(PoniError::NamedDetector(d.name.clone()));
        }
        if d.splinefile.is_some() {
            return Err(PoniError::Unsupported("spline file"));
        }
        if self.parallax {
            return Err(PoniError::Unsupported("parallax correction"));
        }
        if d.binning != (1, 1) {
            return Err(PoniError::Unsupported("binning"));
        }
        Ok(Poni {
            dist: self.dist,
            poni1: self.poni1,
            poni2: self.poni2,
            rot1: self.rot1,
            rot2: self.rot2,
            rot3: self.rot3,
            pixel1: d.pixel1.ok_or(PoniError::Missing("pixel1"))?,
            pixel2: d.pixel2.ok_or(PoniError::Missing("pixel2"))?,
            shape: d.max_shape.ok_or(PoniError::Missing("max_shape"))?,
            orientation: d.orientation.unwrap_or(Orientation::BottomRight),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;

    #[test]
    fn orientation_numbers_are_1_to_4() {
        for n in 1..=4 {
            assert_eq!(Orientation::try_from(n).unwrap() as u32, n);
        }
        for n in [0, 5, 8] {
            assert_eq!(Orientation::try_from(n), Err(PoniError::Orientation(n)));
        }
    }

    #[test]
    fn lab_axes_take_the_part_of_up_normal_to_the_beam_and_are_left_handed() {
        let x = Vector3::x_axis();
        let tilted = Unit::new_normalize(Vector3::new(0.3, 0.0, 1.0));
        let f = LabAxes::new(x, tilted).unwrap();
        assert_eq!(f.axes.column(0), Vector3::z());
        assert_eq!(f.axes.column(1), Vector3::y());
        assert_eq!(f.axes.column(2), Vector3::x());
        assert!(matches!(
            LabAxes::new(x, x),
            Err(PoniError::UpAlongBeam { .. })
        ));
    }

    const ORIENTATIONS: [Orientation; 4] = [
        Orientation::TopLeft,
        Orientation::TopRight,
        Orientation::BottomRight,
        Orientation::BottomLeft,
    ];

    fn oblique() -> LabAxes {
        let beam = Unit::new_normalize(Vector3::new(0.3, -0.5, 0.81));
        let up = Unit::new_normalize(beam.cross(&Vector3::new(0.2, 0.7, -0.4)));
        LabAxes::new(beam, up).unwrap()
    }

    fn poni(rot: (f64, f64, f64), orientation: Orientation) -> Poni {
        Poni {
            dist: 0.31,
            poni1: 0.0071,
            poni2: -0.0253,
            rot1: Rad(rot.0),
            rot2: Rad(rot.1),
            rot3: Rad(rot.2),
            pixel1: 7.5e-5,
            pixel2: 1.72e-4,
            shape: (213, 347),
            orientation,
        }
    }

    /// Largest distance between the same pixel of `a` and `b`; NaN if any is.
    fn gap(a: &PixelDetector, b: &PixelDetector) -> f64 {
        assert_eq!(a.shape(), b.shape());
        let (n1, n2) = a.shape();
        let mut worst = 0.0f64;
        for (i, j) in [
            (0, 0),
            (n1 - 1, 0),
            (0, n2 - 1),
            (n1 - 1, n2 - 1),
            (n1 / 3, n2 / 2),
        ] {
            let d = (a.pixel(i, j).unwrap() - b.pixel(i, j).unwrap()).norm();
            if d > worst || d.is_nan() {
                worst = d;
            }
        }
        worst
    }

    /// Largest field difference, lengths in metres and angles in radians;
    /// NaN if any is.
    fn field_gap(a: &Poni, b: &Poni) -> f64 {
        assert_eq!((a.shape, a.orientation), (b.shape, b.orientation));
        let d = [
            a.dist - b.dist,
            a.poni1 - b.poni1,
            a.poni2 - b.poni2,
            a.rot1.0 - b.rot1.0,
            a.rot2.0 - b.rot2.0,
            a.rot3.0 - b.rot3.0,
            a.pixel1 - b.pixel1,
            a.pixel2 - b.pixel2,
        ];
        d.iter()
            .map(|x| x.abs())
            .fold(0.0, |m, x| if x > m || x.is_nan() { x } else { m })
    }

    #[test]
    fn export_gives_back_the_fields_in_the_principal_ranges() {
        for lab in [
            oblique(),
            LabAxes::new(Vector3::x_axis(), Vector3::z_axis()).unwrap(),
        ] {
            for o in ORIENTATIONS {
                for rot in [
                    (0.0, 0.0, 0.0),
                    (0.21, 0.0, 0.0),
                    (0.0, -0.13, 0.0),
                    (0.0, 0.0, 0.37),
                    (0.21, -0.13, 0.37),
                    (2.5, -1.2, -2.9),
                    (-3.1, 1.5, 3.1),
                ] {
                    let p = poni(rot, o);
                    let back = Poni::from_detector(&p.detector(&lab).unwrap(), o, &lab).unwrap();
                    let d = field_gap(&p, &back);
                    assert!(d <= 1e-12, "{o:?} {rot:?}: {d}");
                }
            }
        }
    }

    /// Angles outside the principal ranges and at rot2 = ±π/2, where rot1
    /// and rot3 share an axis, come back as another PONI of the same detector.
    #[test]
    fn import_export_import_is_the_identity() {
        let lab = oblique();
        for o in ORIENTATIONS {
            for read_as in ORIENTATIONS {
                for rot in [
                    (0.21, -0.13, 0.37),
                    (4.0, 2.0, -3.5),
                    (-7.0, 3.3, 9.0),
                    (0.4, FRAC_PI_2, -0.9),
                    (0.4, -FRAC_PI_2, -0.9),
                    (PI, 0.0, -PI),
                ] {
                    let det = poni(rot, o).detector(&lab).unwrap();
                    let back = Poni::from_detector(&det, read_as, &lab).unwrap();
                    assert_eq!(back.orientation, read_as);
                    let d = gap(&det, &back.detector(&lab).unwrap());
                    assert!(d <= 1e-14, "{o:?} read as {read_as:?}, {rot:?}: {d} m");
                }
            }
        }
    }

    #[test]
    fn export_needs_two_pixels_per_axis_and_perpendicular_axes() {
        let lab = oblique();
        let o = Orientation::BottomRight;
        let x = Vector3::new(1e-4, 0.0, 0.0);
        let y = Vector3::new(0.0, 1e-4, 0.0);
        let at = Vector3::new(0.3, 0.0, 0.0);
        for shape in [(1, 5), (5, 1)] {
            let det = PixelDetector::new(shape, at, x, y).unwrap();
            assert_eq!(
                Poni::from_detector(&det, o, &lab),
                Err(PoniError::TooFewPixels {
                    slow: shape.0,
                    fast: shape.1
                })
            );
        }
        let skewed = PixelDetector::new((5, 5), at, x, y + x * 1e-6).unwrap();
        assert!(matches!(
            Poni::from_detector(&skewed, o, &lab),
            Err(PoniError::Skewed { .. })
        ));
    }

    // Each text below was read by pyFAI 2026.9.0 `PoniFile(path)`; the
    // comment before it gives what pyFAI made of it.
    const GEOMETRY: &str =
        "Distance: 0.31\nPoni1: 0.0071\nPoni2: 0.0253\nRot1: 0.21\nRot2: -0.13\nRot3: 0.37\n";

    fn parse(head: &str) -> Result<PoniFile, PoniError> {
        PoniFile::parse(&format!("{head}{GEOMETRY}"))
    }

    #[test]
    fn parse_reads_lines_as_pyfai_does() {
        // version 2.1, pixels 1e-4 and 2e-4, max_shape (10, 20), orientation 2
        let text = "poni_version: 2.1\r\nDetector: Detector\rDetector_config: {\"pixel1\": 1e-4, \
                    \"pixel2\": 2e-4, \"max_shape\": [10, 20], \"orientation\": 2}\r\n"
            .to_string()
            + &GEOMETRY.replace('\n', "\r");
        let f = PoniFile::parse(&text).unwrap();
        assert_eq!((f.version, f.dist, f.rot3), (2.1, 0.31, Rad(0.37)));
        let p = f.poni().unwrap();
        assert_eq!((p.pixel1, p.pixel2, p.shape), (1e-4, 2e-4, (10, 20)));
        assert_eq!(p.orientation, Orientation::TopRight);

        // pixel1 3e-4, dist 0.5, rot1 0.21, rot3 0.4: keys in any case and
        // spacing, the later line wins, `dist` stands for `Distance`, and a
        // commented line is not read
        let f = PoniFile::parse(
            "  DETECTOR  :  Detector \npixelsize1: 1e-4\nPIXELSIZE1: 3e-4\npixelsize2: 2e-4\n\
             dist: 0.5\nPoni1: 0.0071\nPoni2: 0.0253\nRot1: 0.21\nRot2: -0.13\nRot3: 0.37\n\
             Rot3: 0.4\n# Rot1: 9\n",
        )
        .unwrap();
        assert_eq!(f.detector.pixel1, Some(3e-4));
        assert_eq!((f.dist, f.rot1, f.rot3), (0.5, Rad(0.21), Rad(0.4)));
        // dist 0.31: `Distance` over `dist`
        let f = parse("dist: 0.5\n").unwrap();
        assert_eq!(f.dist, 0.31);
    }

    #[test]
    fn parse_reads_the_detector_as_pyfai_does() {
        let v1 = "pixelsize1: 1e-4\npixelsize2: 2e-4\n";
        // generic Detector without a spline; version 1 has no shape
        for head in [
            format!("Detector: Detector\n{v1}splinefile: NoNe\n"),
            v1.to_string(),
        ] {
            let f = parse(&head).unwrap();
            assert_eq!(f.version, 1.0);
            assert_eq!(
                (f.detector.name.as_str(), f.detector.splinefile.as_deref()),
                ("Detector", None)
            );
            assert_eq!(f.poni(), Err(PoniError::Missing("max_shape")));
        }
        // version 2: a Detector_config raises the written 1
        let f = parse(
            "poni_version: 1\nDetector: Detector\nDetector_config: {\"pixel1\": 1e-4, \
             \"pixel2\": 2e-4, \"max_shape\": [10, 20]}\n",
        )
        .unwrap();
        assert_eq!(f.version, 2.0);
        // generic, orientation 3: the name and the config keys in any case
        let f = parse(
            "poni_version: 2.1\nDetector: detector\nDetector_config: {\"Pixel1\": 1e-4, \
             \"PIXEL2\": 2e-4, \"Max_Shape\": [10, 20], \"orientation\": 0}\n",
        )
        .unwrap();
        assert_eq!(f.detector.orientation, None);
        let p = f.poni().unwrap();
        assert_eq!((p.pixel1, p.pixel2, p.shape), (1e-4, 2e-4, (10, 20)));
        assert_eq!(p.orientation, Orientation::BottomRight);
        // version 3, parallax on
        let f = parse(
            "poni_version: 2.1\nDetector: Detector\nDetector_config: {\"pixel1\": 1e-4, \
             \"pixel2\": 2e-4, \"orientation\": 3}\nParallax:  TRUE \n",
        )
        .unwrap();
        assert_eq!((f.version, f.parallax), (3.0, true));
        assert_eq!(f.poni(), Err(PoniError::Unsupported("parallax correction")));
        // binning (2, 2): pixels 2e-4 and 4e-4, shape (5, 10)
        let f = parse(
            "poni_version: 2\nDetector: Detector\nDetector_config: {\"pixel1\": 1e-4, \
             \"pixel2\": 2e-4, \"max_shape\": [10, 20], \"binning\": [2, 2]}\n",
        )
        .unwrap();
        assert_eq!(f.detector.binning, (2, 2));
        assert_eq!(f.poni(), Err(PoniError::Unsupported("binning")));
    }

    #[test]
    fn parse_refuses_what_pyfai_refuses_or_leaves_without_a_detector() {
        let v1 = "Detector: Detector\npixelsize1: 1e-4\npixelsize2: 2e-4\n";
        let config = "Detector_config: {\"pixel1\": 1e-4, \"pixel2\": 2e-4";
        // no detector
        assert_eq!(
            parse(&format!("poni_version: 1.5\n{v1}")),
            Err(PoniError::Version(1.5))
        );
        assert!(matches!(
            parse(&format!("poni_version: nan\n{v1}")),
            Err(PoniError::Version(v)) if v.is_nan()
        ));
        // RuntimeError: too recent
        assert_eq!(
            parse(&format!(
                "poni_version: 4\nDetector: Detector\n{config}}}\n"
            )),
            Err(PoniError::Version(4.0))
        );
        // KeyError 'detector'
        assert_eq!(
            parse(&format!("poni_version: 2\n{config}}}\n")),
            Err(PoniError::Missing("detector"))
        );
        // RuntimeError: unsupported orientation
        assert_eq!(
            parse(&format!(
                "poni_version: 2.1\nDetector: Detector\n{config}, \"orientation\": 5}}\n"
            )),
            Err(PoniError::Orientation(5))
        );
        // JSONDecodeError
        assert!(matches!(
            parse(&format!("poni_version: 2\nDetector: Detector\n{config}\n")),
            Err(PoniError::Syntax { key, .. }) if key == "detector_config"
        ));
        // ValueError: could not convert string to float
        assert!(matches!(
            PoniFile::parse(&format!("{v1}{}", GEOMETRY.replace("Rot1: 0.21", "Rot1: abc"))),
            Err(PoniError::Syntax { key, value }) if key == "rot1" && value == "abc"
        ));
        // rot2 None, which hklrs refuses
        assert_eq!(
            PoniFile::parse(&format!("{v1}{}", GEOMETRY.replace("Rot2: -0.13\n", ""))),
            Err(PoniError::Missing("rot2"))
        );
    }
}
