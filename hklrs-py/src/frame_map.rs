//! `hklrs.frame_map`: detector frames into a cube, the step a live map and
//! a batch reduction share. Angle coordinates, as the "angles"
//! projection's or a custom one's declared so, and so their binnings and
//! cube axes, are degrees here.

use hklrs::frame_map::FrameMap;
use hklrs::grid::{Binning, Coord, CoordKind, Cube, Projected};
use hklrs::nalgebra::{Matrix3, Rotation3};
use hklrs::normalise::{FrameScalars, PixelFields, Region};
use hklrs::projection::{Custom, FrameContext, Projection, ProjectionError};
use hklrs::{Error, Image, Sample, Wavelength};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

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

/// A Python exception raised in a custom function, carried through the core.
fn wrapped(e: PyErr) -> Error {
    Error::Projection(ProjectionError::new(e))
}

/// `e` as a Python exception: one a custom function raised as it was
/// raised, any other a ValueError.
fn unwrapped(py: Python<'_>, e: Error) -> PyErr {
    if let Error::Projection(p) = &e
        && let Some(raised) = p.get_ref().downcast_ref::<PyErr>()
    {
        return raised.clone_ref(py);
    }
    err(e)
}

/// The frame a custom function is called for: `axes` as given to
/// `FrameMap.add` (degrees / metres, to rounding), `wavelength` in Å.
#[pyclass(name = "FrameContext", module = "hklrs.frame_map", frozen, get_all)]
struct PyFrameContext {
    axes: Vec<f64>,
    wavelength: f64,
}

/// One frame of a custom projection: `function(planes, context)` in
/// Python's units, its returned planes in the core's, one per `kinds`.
fn call(
    py: Python<'_>,
    function: &Py<PyAny>,
    base: &Projected,
    frame: &FrameContext<'_>,
    kinds: &[CoordKind],
) -> Result<Vec<Image<f64>>, Error> {
    let planes = base
        .coords()
        .iter()
        .zip(base.planes())
        .map(|(c, p)| {
            let v = p.data().iter().map(|&v| from_core(v, c.kind())).collect();
            PyArray1::from_vec(py, v).reshape([p.shape().0, p.shape().1])
        })
        .collect::<PyResult<Vec<_>>>()
        .map_err(wrapped)?;
    let context = PyFrameContext {
        axes: frame.geometry.values_to_degrees(frame.values)?,
        wavelength: frame.wavelength.angstrom(),
    };
    let args = (PyTuple::new(py, planes).map_err(wrapped)?, context);
    let out = function.call1(py, args).map_err(wrapped)?;
    let out = out
        .bind(py)
        .extract::<Vec<Bound<'_, PyAny>>>()
        .map_err(wrapped)?
        .iter()
        .enumerate()
        .map(|(i, p)| {
            p.extract::<PyReadonlyArray2<'_, f64>>().map_err(|_| {
                let got = match (p.getattr("ndim"), p.getattr("dtype")) {
                    (Ok(n), Ok(d)) => format!("a {n}-D {d} array"),
                    _ => p.get_type().to_string(),
                };
                PyTypeError::new_err(format!(
                    "custom projection plane {i} is {got}, not a 2-D float64 array"
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()
        .map_err(wrapped)?;
    if out.len() != kinds.len() {
        return Err(Error::Dimensions {
            expected: kinds.len(),
            got: out.len(),
        });
    }
    out.iter()
        .zip(kinds)
        .map(|(a, &kind)| {
            let a = a.as_array();
            Image::new(a.dim(), a.iter().map(|&v| to_core(v, kind)).collect())
        })
        .collect()
}

/// A projection whose coordinates `function(planes, context)` computes,
/// once per frame, from the built-in projection `base`'s: `planes` a tuple
/// of 2-D float64 arrays, one per base axis (degrees for an angle), and
/// `context` an `hklrs.frame_map.FrameContext`. It returns a sequence of
/// one 2-D float64 array per entry of `axes`, each of the planes' shape;
/// `axes` lists (name, "angle" | "linear"), an angle in degrees. `sample`,
/// `normal` and `frame` are `base`'s options, as `FrameMap` takes them. An
/// exception the function raises comes out of `FrameMap.add` as it was
/// raised. `context.axes` arrive converted back from radians, so they can
/// differ from `add`'s inputs in the last digits.
#[pyclass(name = "Custom", module = "hklrs.frame_map", frozen)]
struct PyCustom(Custom);

#[pymethods]
impl PyCustom {
    #[new]
    #[pyo3(signature = (function, axes, base="qsample", *, sample=None, normal=None, frame=None))]
    fn new(
        function: Bound<'_, PyAny>,
        axes: Vec<(String, String)>,
        base: &str,
        sample: Option<PyRef<'_, PySample>>,
        normal: Option<[f64; 3]>,
        frame: Option<[[f64; 3]; 3]>,
    ) -> PyResult<Self> {
        if !function.is_callable() {
            return Err(PyTypeError::new_err("function is not callable"));
        }
        let base = projection(base, sample.map(|s| s.0.clone()), normal, frame)?;
        let coords = axes
            .into_iter()
            .map(|(name, kind)| match kind.as_str() {
                "angle" => Ok(Coord::new(name, CoordKind::Angle)),
                "linear" => Ok(Coord::new(name, CoordKind::Linear)),
                _ => Err(PyValueError::new_err(format!(
                    "axis {name:?}: kind {kind:?} is not \"angle\" or \"linear\""
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let kinds: Vec<CoordKind> = coords.iter().map(Coord::kind).collect();
        let function = function.unbind();
        let f = move |base: &Projected, frame: &FrameContext<'_>| {
            Python::attach(|py| call(py, &function, base, frame, &kinds))
        };
        Custom::new(base, coords, f).map(Self).map_err(err)
    }
}

/// What `FrameMap` takes as its projection: a built-in's name, or a custom
/// one.
#[derive(FromPyObject)]
enum ProjectionArg<'py> {
    Name(String),
    Custom(Bound<'py, PyCustom>),
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
/// one (step, anchor) binning per dimension of `projection`, a built-in's
/// name or a `Custom`, and keyword options:
///
/// - `frame` for the "qsurface" projection, a proper rotation (rows) whose
///   columns are the surface axes in the sample frame, as
///   `Sample.surface_frame` returns it; `sample` for "hkl"; `normal`
///   (sample frame) for "qpar_qper"; the others are "qlab", "qsample" and
///   "angles"; a `Custom` takes its base's options itself;
/// - `region`: an `hklrs.normalise.Region` on the raw frame, the whole
///   frame unbinned when omitted;
/// - `fields`: `hklrs.normalise.PixelFields` of the raw shape, none when
///   omitted; another shape is a ValueError here, not at `add`.
#[pyclass(name = "FrameMap", module = "hklrs.frame_map", frozen)]
struct PyFrameMap(FrameMap);

#[pymethods]
impl PyFrameMap {
    #[new]
    #[pyo3(
        signature = (geometry, detector, binnings, projection=ProjectionArg::Name("qsample".to_owned()), **options),
        text_signature = "(geometry, detector, binnings, projection=\"qsample\", **options)"
    )]
    fn new(
        geometry: &PyGeometry,
        detector: &PyPixelDetector,
        binnings: Vec<(f64, f64)>,
        projection: ProjectionArg<'_>,
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
        let projection = match projection {
            ProjectionArg::Name(name) => self::projection(&name, sample, normal, frame)?,
            ProjectionArg::Custom(_) if sample.is_some() || normal.is_some() || frame.is_some() => {
                return Err(PyValueError::new_err(
                    "a Custom projection takes its base's sample=, normal= or frame= itself",
                ));
            }
            ProjectionArg::Custom(c) => Projection::Custom(c.get().0.clone()),
        };
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
    /// omitted. On error the cube is unchanged; an exception a custom
    /// function raised is raised again.
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
        .map_err(|e| unwrapped(py, e))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCube>()?;
    m.add_class::<PyCustom>()?;
    m.add_class::<PyFrameContext>()?;
    m.add_class::<PyFrameMap>()
}
