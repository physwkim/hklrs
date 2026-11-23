//! `hklrs.frame_map`: detector frames into a cube, the step a live map and
//! a batch reduction share. The "angles" projection's coordinates, and so
//! its binnings and cube axes, are degrees here.

use hklrs::frame_map::FrameMap;
use hklrs::grid::{Binning, Cube};
use hklrs::normalise::{FrameScalars, PixelFields, Region};
use hklrs::projection::Projection;
use hklrs::{Sample, Wavelength};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::normalise::{PyFrameScalars, PyPixelFields, PyRegion, raw_frame};
use super::{PyGeometry, PyPixelDetector, PySample, err, unit, xu_options};

/// Whether a cube axis is one of the angles projection's, radians in the
/// core.
fn angular(name: &str) -> bool {
    Projection::Angles.names().contains(&name)
}

fn to_core(v: f64, angular: bool) -> f64 {
    if angular { v.to_radians() } else { v }
}

fn from_core(v: f64, angular: bool) -> f64 {
    if angular { v.to_degrees() } else { v }
}

fn projection(
    name: &str,
    sample: Option<Sample>,
    normal: Option<[f64; 3]>,
) -> PyResult<Projection> {
    let p = match (name, sample, normal) {
        ("qlab", None, None) => Projection::QLab,
        ("qsample", None, None) => Projection::QSample,
        ("hkl", Some(s), None) => Projection::Hkl(s),
        ("qpar_qper", None, Some(n)) => Projection::QparQper {
            normal: unit("normal", n)?,
        },
        ("angles", None, None) => Projection::Angles,
        ("qlab" | "qsample" | "hkl" | "qpar_qper" | "angles", ..) => {
            return Err(PyValueError::new_err(format!(
                "projection {name:?}: sample= goes with \"hkl\" alone and normal= \
                 with \"qpar_qper\" alone"
            )));
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown projection {name:?}"
            )));
        }
    };
    Ok(p)
}

/// Σcounts, Σweight and contributions per bin over axes that grow to hold
/// every pixel added; `Cube()` is the empty cube.
#[pyclass(name = "Cube", module = "hklrs.frame_map")]
struct PyCube(Cube);

impl PyCube {
    fn shape(&self) -> Vec<usize> {
        match self.0.axes() {
            [] => vec![0],
            axes => axes.iter().map(|a| a.size()).collect(),
        }
    }

    fn array<'py, T: numpy::Element>(
        &self,
        py: Python<'py>,
        v: Vec<T>,
    ) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
        PyArray1::from_vec(py, v).reshape(self.shape())
    }
}

#[pymethods]
impl PyCube {
    #[new]
    fn new() -> Self {
        Self(Cube::default())
    }

    /// Per axis (name, step, anchor, first, size): bin k of the axis spans
    /// [anchor + k·step, anchor + (k+1)·step) and the cube holds bins
    /// first..first+size; degrees for the angles projection.
    #[getter]
    fn axes(&self) -> Vec<(String, f64, f64, i64, usize)> {
        self.0
            .axes()
            .iter()
            .map(|a| {
                let (b, ang) = (a.binning(), angular(a.name()));
                (
                    a.name().to_owned(),
                    from_core(b.step(), ang),
                    from_core(b.anchor(), ang),
                    a.first(),
                    a.size(),
                )
            })
            .collect()
    }

    /// Σcounts per bin, an array of the axes' sizes.
    #[getter]
    fn photons<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.array(py, self.0.photons().to_vec())
    }

    /// Σweight per bin.
    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.array(py, self.0.weight().to_vec())
    }

    #[getter]
    fn contributions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<u64>>> {
        self.array(py, self.0.contributions().to_vec())
    }

    /// Σcounts / Σweight per bin; NaN where Σweight is 0.
    #[getter]
    fn intensity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let v = self.0.intensity().map(|v| v.unwrap_or(f64::NAN)).collect();
        self.array(py, v)
    }
}

/// What every frame of one scan shares: the geometry, the raw detector,
/// one (step, anchor) binning per dimension of `projection`, and keyword
/// options:
///
/// - `sample` for the "hkl" projection, `normal` (sample frame) for
///   "qpar_qper"; the others are "qlab", "qsample" and "angles";
/// - `region`: an `hklrs.normalise.Region` on the raw frame, the whole
///   frame unbinned when omitted;
/// - `fields`: `hklrs.normalise.PixelFields` of the raw shape, none when
///   omitted; another shape is a ValueError here, not at `add`.
#[pyclass(name = "FrameMap", module = "hklrs.frame_map", frozen)]
struct PyFrameMap(FrameMap);

#[pymethods]
impl PyFrameMap {
    #[new]
    #[pyo3(signature = (geometry, detector, binnings, projection="qsample", **options))]
    fn new(
        geometry: &PyGeometry,
        detector: &PyPixelDetector,
        binnings: Vec<(f64, f64)>,
        projection: &str,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let raw_shape = detector.0.shape();
        let (mut sample, mut normal) = (None, None);
        let mut region = Region::new(0..raw_shape.0, 0..raw_shape.1, (1, 1));
        let mut fields = PixelFields::new(raw_shape);
        xu_options(options, |key, value| {
            match key {
                "sample" => sample = Some(value.extract::<PyRef<'_, PySample>>()?.0.clone()),
                "normal" => normal = Some(value.extract()?),
                "region" => region = value.extract::<PyRef<'_, PyRegion>>()?.0.clone(),
                "fields" => fields = value.extract::<PyRef<'_, PyPixelFields>>()?.0.clone(),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        let projection = self::projection(projection, sample, normal)?;
        let ang = matches!(projection, Projection::Angles);
        let binnings = binnings
            .into_iter()
            .map(|(step, anchor)| Binning::new(to_core(step, ang), to_core(anchor, ang)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        FrameMap::new(
            geometry.0.clone(),
            projection,
            &detector.0,
            &region,
            fields,
            binnings,
        )
        .map(Self)
        .map_err(err)
    }

    /// The binned detector, whose pixels the frames are mapped on.
    #[getter]
    fn detector(&self) -> PyPixelDetector {
        PyPixelDetector(self.0.detector().clone())
    }

    /// Adds one frame to `cube`: `axes` in the geometry's axis order
    /// (degrees / metres), `wavelength` in Å, `raw` the raw image and
    /// `scalars` the frame's `hklrs.normalise.FrameScalars`, none when
    /// omitted. On error the cube is unchanged.
    #[pyo3(signature = (cube, axes, wavelength, raw, scalars=None))]
    fn add(
        &self,
        py: Python<'_>,
        mut cube: PyRefMut<'_, PyCube>,
        axes: Vec<f64>,
        wavelength: f64,
        raw: PyReadonlyArray2<'_, f64>,
        scalars: Option<PyRef<'_, PyFrameScalars>>,
    ) -> PyResult<()> {
        let raw = raw_frame(&raw)?;
        let wavelength = Wavelength::new(wavelength).map_err(err)?;
        let s = scalars.map_or_else(FrameScalars::new, |s| s.0);
        let cube = &mut cube.0;
        py.detach(|| {
            let frame = self.0.map(&axes, wavelength, &raw, &s)?;
            self.0.add(cube, &frame)
        })
        .map_err(err)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCube>()?;
    m.add_class::<PyFrameMap>()
}
