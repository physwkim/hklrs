//! `hklrs.cut`: line cuts over pixel Q and normalised counts. Lengths are
//! Å⁻¹; every angle (the coordinate along an angular cut, the window of an
//! angular integration direction, the bin step and anchor, the footprint
//! width) is in degrees here and radians in the core. A result holds the
//! sums and their ratio per bin.

use hklrs::absorption::{Masked, Weight};
use hklrs::cut::{Cut, CutKind, CutResult, Footprint, IntDir};
use hklrs::grid::Binning;
use hklrs::nalgebra::Vector3;
use hklrs::normalise::Normalised;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{MASKED, err, masked_code};

fn cut_err(e: hklrs::cut::CutError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCut>()?;
    m.add_class::<PyFootprint>()?;
    m.add_class::<PyCutResult>()
}

/// The reason for a code of a reasons array; `None` for 0.
fn reason(code: u8) -> PyResult<Option<Masked>> {
    if code == 0 {
        return Ok(None);
    }
    MASKED
        .iter()
        .copied()
        .find(|&m| masked_code(m) == code)
        .map(Some)
        .ok_or_else(|| PyValueError::new_err(format!("unknown reason code {code}")))
}

fn angular_along(k: &CutKind) -> bool {
    matches!(k, CutKind::Omega | CutKind::Radial | CutKind::TwoTheta)
}

/// How a pixel's weight is laid on the cut axis: at its coordinate, or over
/// `width`, in the units of the coordinate along the cut (degrees for
/// omega, radial and twotheta cuts, Å⁻¹ for the others).
#[pyclass(name = "Footprint", frozen)]
struct PyFootprint(Option<f64>);

#[pymethods]
impl PyFootprint {
    #[staticmethod]
    fn point() -> Self {
        Self(None)
    }

    #[staticmethod]
    fn width(width: f64) -> Self {
        Self(Some(width))
    }

    fn __repr__(&self) -> String {
        match self.0 {
            None => "Footprint.point()".into(),
            Some(w) => format!("Footprint.width({w})"),
        }
    }
}

/// A line cut: the coordinate along it and a window of `half_width`
/// (strict `<`) in the integration direction `intdir`, binned by
/// `floor((x - anchor) / step)`.
///
/// kind: "qz", "qy", "qx", "omega", "radial", "twotheta" or "line" (which
/// takes `line=(point, direction)` and ignores `at`). intdir: "q", "omega",
/// "2theta" or "radial". `at` is the Q position of the cut (Å⁻¹).
#[pyclass(name = "Cut", frozen)]
struct PyCut {
    cut: Cut,
    angular_along: bool,
    angular_perp: bool,
}

fn parse_kind(kind: &str, line: Option<([f64; 3], [f64; 3])>) -> PyResult<CutKind> {
    Ok(match kind {
        "qz" => CutKind::Qz,
        "qy" => CutKind::Qy,
        "qx" => CutKind::Qx,
        "omega" => CutKind::Omega,
        "radial" => CutKind::Radial,
        "twotheta" | "2theta" => CutKind::TwoTheta,
        "line" => match line {
            Some((point, dir)) => CutKind::Line { point, dir },
            None => {
                return Err(PyValueError::new_err(
                    "a line cut needs line=(point, direction)",
                ));
            }
        },
        k => return Err(PyValueError::new_err(format!("unknown cut kind {k:?}"))),
    })
}

fn kind_name(k: &CutKind) -> &'static str {
    match k {
        CutKind::Qz => "qz",
        CutKind::Qy => "qy",
        CutKind::Qx => "qx",
        CutKind::Omega => "omega",
        CutKind::Radial => "radial",
        CutKind::TwoTheta => "twotheta",
        CutKind::Line { .. } => "line",
    }
}

fn parse_intdir(s: &str) -> PyResult<IntDir> {
    Ok(match s {
        "q" => IntDir::Q,
        "omega" => IntDir::Omega,
        "2theta" | "twotheta" => IntDir::TwoTheta,
        "radial" => IntDir::Radial,
        s => {
            return Err(PyValueError::new_err(format!(
                "unknown integration direction {s:?}"
            )));
        }
    })
}

fn intdir_name(d: IntDir) -> &'static str {
    match d {
        IntDir::Q => "q",
        IntDir::Omega => "omega",
        IntDir::TwoTheta => "2theta",
        IntDir::Radial => "radial",
    }
}

fn to_core(v: f64, angular: bool) -> f64 {
    if angular { v.to_radians() } else { v }
}

fn from_core(v: f64, angular: bool) -> f64 {
    if angular { v.to_degrees() } else { v }
}

#[pymethods]
impl PyCut {
    #[new]
    #[pyo3(signature = (kind, intdir, half_width, binning, at=[0.0; 3], line=None))]
    fn new(
        kind: &str,
        intdir: &str,
        half_width: f64,
        binning: (f64, f64),
        at: [f64; 3],
        line: Option<([f64; 3], [f64; 3])>,
    ) -> PyResult<Self> {
        let (step, anchor) = binning;
        let kind = parse_kind(kind, line)?;
        let intdir = parse_intdir(intdir)?;
        let (aa, ap) = (angular_along(&kind), intdir != IntDir::Q);
        let binning = Binning::new(to_core(step, aa), to_core(anchor, aa)).map_err(err)?;
        let cut = Cut::new(kind, intdir, at, to_core(half_width, ap), binning).map_err(cut_err)?;
        Ok(Self {
            cut,
            angular_along: aa,
            angular_perp: ap,
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        kind_name(&self.cut.kind())
    }

    #[getter]
    fn intdir(&self) -> &'static str {
        intdir_name(self.cut.intdir())
    }

    /// (step, anchor) of the bins along the cut.
    #[getter]
    fn binning(&self) -> (f64, f64) {
        let b = self.cut.binning();
        (
            from_core(b.step(), self.angular_along),
            from_core(b.anchor(), self.angular_along),
        )
    }

    #[getter]
    fn half_width(&self) -> f64 {
        from_core(self.cut.half_width(), self.angular_perp)
    }

    /// Sums the pixels over the cut. `q` is (N, 3) in Å⁻¹, `counts` and
    /// `weight` are length N, `reasons` is a length-N uint8 array of
    /// `Masked` codes, 0 for a pixel with a value. A reason wins over the
    /// counts, counts that are not finite are `BadCount`, and then a weight
    /// that is not positive and finite is `Opaque`.
    fn accumulate(
        &self,
        wavelength: f64,
        footprint: &PyFootprint,
        q: PyReadonlyArray2<f64>,
        counts: PyReadonlyArray1<f64>,
        weight: PyReadonlyArray1<f64>,
        reasons: PyReadonlyArray1<u8>,
    ) -> PyResult<PyCutResult> {
        let py = q.py();
        let n = q.shape()[0];
        if q.shape()[1] != 3
            || [counts.len(), weight.len(), reasons.len()]
                .iter()
                .any(|&l| l != n)
        {
            return Err(PyValueError::new_err(format!(
                "q must be (N, 3) and counts, weight and reasons of length N; got {:?}, {}, {}, {}",
                q.shape(),
                counts.len(),
                weight.len(),
                reasons.len()
            )));
        }
        let wl = hklrs::Wavelength::new(wavelength).map_err(err)?;
        let fp = match footprint.0 {
            None => Footprint::Point,
            Some(w) => Footprint::Width(to_core(w, self.angular_along)),
        };
        let (q, counts, weight, reasons) = (
            q.as_array(),
            counts.as_slice()?,
            weight.as_slice()?,
            reasons.as_slice()?,
        );
        let pixels = (0..n)
            .map(|i| {
                let value = match reason(reasons[i])? {
                    Some(m) => Err(m),
                    None => Normalised::new(counts[i], Weight::new(weight[i], Masked::Opaque)),
                };
                Ok((Vector3::new(q[[i, 0]], q[[i, 1]], q[[i, 2]]), value))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result = py
            .detach(|| self.cut.accumulate(wl, fp, pixels))
            .map_err(err)?;
        Ok(PyCutResult {
            result,
            angular: self.angular_along,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Cut({:?}, {:?}, half_width={})",
            self.kind(),
            self.intdir(),
            self.half_width()
        )
    }
}

/// The sums of a cut per bin that received anything, ascending, and
/// `intensity`, their ratio.
#[pyclass(name = "CutResult", frozen)]
struct PyCutResult {
    result: CutResult,
    angular: bool,
}

impl PyCutResult {
    fn column<'py, T: numpy::Element>(
        &self,
        py: Python<'py>,
        f: impl Fn(i64, &hklrs::cut::Bin) -> T,
    ) -> Bound<'py, PyArray1<T>> {
        PyArray1::from_vec(py, self.result.bins().map(|(k, b)| f(k, b)).collect())
    }
}

#[pymethods]
impl PyCutResult {
    /// Global bin indices, `floor((x - anchor) / step)`.
    #[getter]
    fn bins<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.column(py, |k, _| k)
    }

    /// Bin centres along the cut, degrees for an angular cut.
    #[getter]
    fn centres<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let b = self.result.binning();
        self.column(py, |k, _| {
            from_core(b.edge(k) + b.step() / 2.0, self.angular)
        })
    }

    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.column(py, |_, b| b.counts)
    }

    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.column(py, |_, b| b.weight)
    }

    #[getter]
    fn contributions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.column(py, |_, b| b.contributions)
    }

    /// Σcounts / Σweight per bin; NaN where Σweight is 0.
    #[getter]
    fn intensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(
            py,
            self.result
                .intensity()
                .map(|(_, v)| v.unwrap_or(f64::NAN))
                .collect(),
        )
    }

    /// Pixels not accumulated, by cause: every `Masked` name, then
    /// "no_coordinate" and "outside".
    #[getter]
    fn skipped<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.result.skipped();
        let d = PyDict::new(py);
        for m in MASKED {
            d.set_item(format!("{m:?}"), s.masked(m))?;
        }
        d.set_item("no_coordinate", s.no_coordinate)?;
        d.set_item("outside", s.outside)?;
        Ok(d)
    }

    /// The sum of two results of the same binning and footprint, for frames
    /// accumulated apart.
    fn merge(&self, other: &PyCutResult) -> PyResult<PyCutResult> {
        if self.angular != other.angular {
            return Err(PyValueError::new_err("results of different cut kinds"));
        }
        Ok(PyCutResult {
            result: self
                .result
                .clone()
                .merge(other.result.clone())
                .map_err(cut_err)?,
            angular: self.angular,
        })
    }

    fn __len__(&self) -> usize {
        self.result.bins().count()
    }
}
