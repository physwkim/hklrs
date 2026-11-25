//! `hklrs.frame_map`: detector frames into a cube, the step a live map and
//! a batch reduction share. Angle coordinates, as the "angles"
//! projection's, and so their binnings and cube axes, are degrees here.

use hklrs::frame_map::FrameMap;
use hklrs::grid::{Binning, Coord, CoordKind, Cube};
use hklrs::nalgebra::{Matrix3, Rotation3};
use hklrs::normalise::{FrameScalars, PixelFields, Region};
use hklrs::projection::Projection;
use hklrs::{Sample, Wavelength};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::normalise::{PyFrameScalars, PyPixelFields, PyRegion};
use super::{PyGeometry, PyPixelDetector, PySample, err, image, unit, xu_options};

/// A value of a coordinate of `kind` from Python's units to the core's:
/// degrees to radians for an angle.
fn to_core(v: f64, kind: CoordKind) -> f64 {
    match kind {
        CoordKind::Angle => v.to_radians(),
        CoordKind::Linear => v,
    }
}

fn from_core(v: f64, kind: CoordKind) -> f64 {
    match kind {
        CoordKind::Angle => v.to_degrees(),
        CoordKind::Linear => v,
    }
}

/// `rows` as a rotation, held to `hklrs::proper_rotation`.
fn rotation(rows: [[f64; 3]; 3]) -> PyResult<Rotation3<f64>> {
    hklrs::proper_rotation("frame", &Matrix3::from_fn(|i, j| rows[i][j])).map_err(err)
}

fn projection(
    name: &str,
    sample: Option<Sample>,
    normal: Option<[f64; 3]>,
    frame: Option<[[f64; 3]; 3]>,
) -> PyResult<Projection> {
    let p = match (name, sample, normal, frame) {
        ("qlab", None, None, None) => Projection::QLab,
        ("qsample", None, None, None) => Projection::QSample,
        ("qsurface", None, None, Some(f)) => Projection::QSurface {
            frame: rotation(f)?,
        },
        ("hkl", Some(s), None, None) => Projection::Hkl(s),
        ("qpar_qper", None, Some(n), None) => Projection::QparQper {
            normal: unit("normal", n)?,
        },
        ("angles", None, None, None) => Projection::Angles,
        ("qlab" | "qsample" | "qsurface" | "hkl" | "qpar_qper" | "angles", ..) => {
            return Err(PyValueError::new_err(format!(
                "projection {name:?}: frame= goes with \"qsurface\" alone, sample= \
                 with \"hkl\" alone and normal= with \"qpar_qper\" alone"
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
    /// first..first+size; degrees for an angle axis.
    #[getter]
    fn axes(&self) -> Vec<(String, f64, f64, i64, usize)> {
        self.0
            .axes()
            .iter()
            .map(|a| {
                let (b, kind) = (a.binning(), a.kind());
                (
                    a.name().to_owned(),
                    from_core(b.step(), kind),
                    from_core(b.anchor(), kind),
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
/// - `frame` for the "qsurface" projection, a proper rotation (rows) whose
///   columns are the surface axes in the sample frame, as
///   `Sample.surface_frame` returns it; `sample` for "hkl"; `normal`
///   (sample frame) for "qpar_qper"; the others are "qlab", "qsample" and
///   "angles";
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
        let (mut sample, mut normal, mut frame) = (None, None, None);
        let mut region = Region::new(0..raw_shape.0, 0..raw_shape.1, (1, 1));
        let mut fields = PixelFields::new();
        xu_options(options, |key, value| {
            match key {
                "sample" => sample = Some(value.extract::<PyRef<'_, PySample>>()?.0.clone()),
                "normal" => normal = Some(value.extract()?),
                "frame" => frame = Some(value.extract()?),
                "region" => region = value.extract::<PyRef<'_, PyRegion>>()?.0.clone(),
                "fields" => fields = value.extract::<PyRef<'_, PyPixelFields>>()?.0.clone(),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        let projection = self::projection(projection, sample, normal, frame)?;
        let coords = projection.coords();
        let binnings = binnings
            .into_iter()
            .enumerate()
            .map(|(i, (step, anchor))| {
                // a binning past the projection's dimensions, which
                // FrameMap::new refuses, has no kind to convert by
                let kind = coords.get(i).map_or(CoordKind::Linear, Coord::kind);
                Binning::new(to_core(step, kind), to_core(anchor, kind))
            })
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
        let raw = image(&raw)?;
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
