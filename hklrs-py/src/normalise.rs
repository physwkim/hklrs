//! `hklrs.normalise`: the intensity normaliser. Per pixel it returns counts,
//! weight and a reason code (0 = has a value), as `solid_angle` does;
//! counts and weight are NaN where masked.

use hklrs::absorption::Masked;
use hklrs::normalise as core;
use hklrs::{Image, Weights};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{MASKED, PyPixelDetector, err, image, masked_code};

type Pixels<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u8>>,
);

/// What one frame contributes to every pixel's weight. A monitor needs its
/// reference; a value that is not positive and finite is a ValueError, never
/// replaced by 1.
#[pyclass(name = "FrameScalars", module = "hklrs.normalise", frozen)]
pub(crate) struct PyFrameScalars(pub(crate) core::FrameScalars);

#[pymethods]
impl PyFrameScalars {
    /// `monitor` and `reference` (monitor counts of the frame and the value
    /// they are scaled to, e.g. `mean_monitor` of the scan), `count_time` in
    /// s, and the absorber's `transmission` (fraction > 0).
    #[new]
    #[pyo3(signature = (monitor=None, reference=None, count_time=None, transmission=None))]
    fn new(
        monitor: Option<f64>,
        reference: Option<f64>,
        count_time: Option<f64>,
        transmission: Option<f64>,
    ) -> PyResult<Self> {
        let mut s = core::FrameScalars::new();
        match (monitor, reference) {
            (Some(m), Some(r)) => s = s.monitor(m, r).map_err(err)?,
            (None, None) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "monitor and reference are given together",
                ));
            }
        }
        if let Some(t) = count_time {
            s = s.count_time(t).map_err(err)?;
        }
        if let Some(t) = transmission {
            s = s.transmission(t).map_err(err)?;
        }
        Ok(Self(s))
    }

    /// These scalars with the transmitted fraction `absorber(data)`, any
    /// Python callable of one argument (a function, lambda, bound method or
    /// partial). xrayutilities's `absfun` is its inverse.
    fn absorber(&self, absorber: &Bound<'_, PyAny>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let t: f64 = absorber.call1((data,))?.extract()?;
        self.0.transmission(t).map(Self).map_err(err)
    }
}

/// The mean of a scan's monitor (xrayutilities's default `av_mon`); every
/// value must be positive and finite.
#[pyfunction]
fn mean_monitor(monitor: Vec<f64>) -> PyResult<f64> {
    core::mean_monitor(&monitor).map_err(err)
}

/// The detector's dark rate (counts per second) and flat field (relative
/// gain), arrays of the whole raw frame's `shape`. A flat pixel that is not
/// finite or is <= 1e-5 is masked (`BadFlat`).
#[pyclass(name = "PixelFields", module = "hklrs.normalise", frozen)]
pub(crate) struct PyPixelFields(pub(crate) core::PixelFields);

#[pymethods]
impl PyPixelFields {
    #[new]
    #[pyo3(signature = (shape, dark_rate=None, flat=None))]
    fn new(
        shape: (usize, usize),
        dark_rate: Option<PyReadonlyArray2<'_, f64>>,
        flat: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let mut f = core::PixelFields::new(shape);
        if let Some(d) = dark_rate {
            f = f.dark_rate(image(&d)?).map_err(err)?;
        }
        if let Some(g) = flat {
            f = f.flat(image(&g)?).map_err(err)?;
        }
        Ok(Self(f))
    }
}

/// The blocks of one `Region`, made by `Region.apply`.
#[pyclass(name = "Reducer", module = "hklrs.normalise", frozen)]
struct PyReducer(core::Reducer);

#[pymethods]
impl PyReducer {
    /// The binned detector's (slow, fast) shape.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }
}

/// A roi (`slow`, `fast`: half-open (start, end) on the raw frame) cut into
/// blocks of `bin` = (slow, fast) pixels.
#[pyclass(name = "Region", module = "hklrs.normalise", frozen)]
pub(crate) struct PyRegion(pub(crate) core::Region);

#[pymethods]
impl PyRegion {
    #[new]
    fn new(slow: (usize, usize), fast: (usize, usize), bin: (usize, usize)) -> Self {
        Self(core::Region::new(slow.0..slow.1, fast.0..fast.1, bin))
    }

    /// (binned detector, reducer): the detector's `roi` and `bin` and the
    /// reducer of the same blocks, for `Normaliser`.
    fn apply(&self, detector: &PyPixelDetector) -> PyResult<(PyPixelDetector, PyReducer)> {
        let (d, r) = self.0.apply(&detector.0).map_err(err)?;
        Ok((PyPixelDetector(d), PyReducer(r)))
    }
}

/// The weights of `solid_angle` and `polarisation`: (values, reasons), each
/// of its own shape.
fn geometric(
    (values, reasons): (PyReadonlyArray2<'_, f64>, PyReadonlyArray2<'_, u8>),
) -> PyResult<Weights> {
    let codes = reasons
        .as_array()
        .iter()
        .map(|&c| match c {
            0 => Ok(None),
            c => MASKED
                .iter()
                .find(|&&m| masked_code(m) == c)
                .map(|&m| Some(m))
                .ok_or_else(|| PyValueError::new_err(format!("{c} is not a reason code"))),
        })
        .collect::<PyResult<Vec<Option<Masked>>>>()?;
    let reasons = Image::new((reasons.shape()[0], reasons.shape()[1]), codes).map_err(err)?;
    Weights::from_values(&image(&values)?, &reasons).map_err(err)
}

/// The per-pixel stages of one region of one detector: `Normaliser(fields,
/// reducer)` with the `PixelFields` of the raw frame and the `Reducer` of
/// `Region.apply`. Fields of another shape than the region's detector are a
/// ValueError here, before any frame.
#[pyclass(name = "Normaliser", module = "hklrs.normalise", frozen)]
struct PyNormaliser(core::Normaliser);

#[pymethods]
impl PyNormaliser {
    #[new]
    fn new(fields: &PyPixelFields, reducer: &PyReducer) -> PyResult<Self> {
        core::Normaliser::new(fields.0.clone(), reducer.0.clone())
            .map(Self)
            .map_err(err)
    }

    /// One frame: `raw` is the whole raw frame (the fields' shape);
    /// `weights` are (values, reasons) pairs from `solid_angle` and
    /// `polarisation` (or `absorption`) of the binned detector. Returns
    /// (counts, weight, reasons) of the binned shape; the corrected value is
    /// counts / weight, a bin's by summing both. A masked pixel has NaN and
    /// its `Masked` code. Precedence: Detector, a weight's own reason,
    /// BadFlat, BadCount, Opaque.
    #[pyo3(signature = (raw, scalars, weights=Vec::new()))]
    fn normalise<'py>(
        &self,
        py: Python<'py>,
        raw: PyReadonlyArray2<'_, f64>,
        scalars: &PyFrameScalars,
        weights: Vec<(PyReadonlyArray2<'_, f64>, PyReadonlyArray2<'_, u8>)>,
    ) -> PyResult<Pixels<'py>> {
        let raw = image(&raw)?;
        let weights = weights
            .into_iter()
            .map(geometric)
            .collect::<PyResult<Vec<_>>>()?;
        let refs: Vec<&Weights> = weights.iter().collect();
        let frame = self.0.normalise(&raw, &scalars.0, &refs).map_err(err)?;
        let (counts, weight, reasons): (Vec<f64>, Vec<f64>, Vec<u8>) = frame
            .data
            .iter()
            .map(|p| match p {
                Ok(p) => (p.counts(), p.weight().get(), 0),
                Err(m) => (f64::NAN, f64::NAN, masked_code(*m)),
            })
            .collect();
        let shape = frame.shape();
        Ok((
            PyArray1::from_vec(py, counts).reshape(shape)?,
            PyArray1::from_vec(py, weight).reshape(shape)?,
            PyArray1::from_vec(py, reasons).reshape(shape)?,
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrameScalars>()?;
    m.add_class::<PyPixelFields>()?;
    m.add_class::<PyRegion>()?;
    m.add_class::<PyReducer>()?;
    m.add_class::<PyNormaliser>()?;
    m.add_function(wrap_pyfunction!(mean_monitor, m)?)
}
