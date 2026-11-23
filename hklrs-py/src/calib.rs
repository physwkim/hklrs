//! `hklrs.calib`: detector calibration. Angles are degrees (frame angles and
//! the angle parameters alike), lengths metres, wavelength Å, as everywhere
//! in hklrs. A `Fit` exists only for a converged run; every other outcome is
//! an exception of the `CalibError` family, one class per failure.

use hklrs::calib::peak::PeakError;
use hklrs::calib::{self as core, CalibError as Failure, ParamId, Reject};
use hklrs::xu;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{PyGeometry, PySample, PyXuArea, PyXuLinear, err};

create_exception!(
    hklrs.calib,
    CalibError,
    PyValueError,
    "A calibration failed."
);
create_exception!(
    hklrs.calib,
    NotConverged,
    CalibError,
    "No stopping rule was met; carries reason, best (name -> value), cost, iters."
);
create_exception!(
    hklrs.calib,
    Unidentifiable,
    CalibError,
    "The frames do not determine some free parameters; carries params."
);
create_exception!(
    hklrs.calib,
    TooFewFrames,
    CalibError,
    "Fewer residuals than free parameters; carries have and need."
);
create_exception!(
    hklrs.calib,
    AmbiguousDirections,
    CalibError,
    "Two detector direction sets fit within 10 %; carries best, runner_up, best_cost, runner_up_cost."
);
create_exception!(
    hklrs.calib,
    BadStart,
    CalibError,
    "A start value is not finite; carries param and value."
);
create_exception!(
    hklrs.calib,
    BadFrame,
    CalibError,
    "A frame is invalid; carries index and what."
);
create_exception!(
    hklrs.calib,
    BadProblem,
    CalibError,
    "No free parameters, fewer residuals than free parameters, or a hkl fit without both kinds of frame."
);
create_exception!(
    hklrs.calib,
    BadModel,
    CalibError,
    "The geometry or detector parameters do not build a model."
);
create_exception!(
    hklrs.calib,
    OffsetAxisNotRotation,
    CalibError,
    "The axis the angle offset shifts is not a rotation; carries axis."
);
create_exception!(
    hklrs.calib,
    PeakFailure,
    PyValueError,
    "Peak finding failed on its input (shape, mask, non-finite or too small)."
);
create_exception!(
    hklrs.calib,
    PeakRejected,
    PeakFailure,
    "The frame holds no usable peak; carries reason (and slow, fast for a saturated pixel)."
);

fn snake(camel: &str) -> String {
    let mut out = String::new();
    for (i, c) in camel.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// The variant name of a payload-carrying enum's `Debug` text.
fn variant(d: &dyn std::fmt::Debug) -> String {
    let text = format!("{d:?}");
    snake(
        text.split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap_or(""),
    )
}

fn id_name(id: ParamId) -> &'static str {
    match id {
        ParamId::Cch1 => "cch1",
        ParamId::Cch2 => "cch2",
        ParamId::Pwidth1 => "pwidth1",
        ParamId::Pwidth2 => "pwidth2",
        ParamId::Distance => "distance",
        ParamId::TiltAzimuth => "tiltazimuth",
        ParamId::Tilt => "tilt",
        ParamId::Detrot => "detrot",
        ParamId::AngleOffset => "angle_offset",
        ParamId::Wavelength => "wavelength",
        ParamId::SampleTilt => "sample_tilt",
        ParamId::SampleAzimuth => "sample_azimuth",
    }
}

fn dir_name(d: xu::Direction) -> String {
    let v = d.vector();
    let axis = (0..3).find(|&i| v[i] != 0.0).unwrap_or(0);
    format!(
        "{}{}",
        ["x", "y", "z"][axis],
        if v[axis] < 0.0 { '-' } else { '+' }
    )
}

fn dirs(d: &[xu::Direction]) -> Vec<String> {
    d.iter().map(|&d| dir_name(d)).collect()
}

fn calib_err(py: Python<'_>, e: Failure) -> PyErr {
    let msg = e.to_string();
    let set = |e: PyErr, attrs: Vec<(&str, Py<PyAny>)>| {
        let v = e.value(py);
        for (k, a) in attrs {
            // plain values cannot fail to set on a fresh exception
            let _ = v.setattr(k, a);
        }
        e
    };
    macro_rules! any {
        ($v:expr) => {
            $v.into_pyobject(py).map(|o| o.into_any().unbind()).unwrap()
        };
    }
    match e {
        Failure::NotConverged {
            reason,
            best,
            cost,
            iters,
        } => {
            let best: Vec<(&str, f64)> = best.iter().map(|&(id, v)| (id_name(id), v)).collect();
            set(
                NotConverged::new_err(msg),
                vec![
                    ("reason", any!(variant(&reason))),
                    (
                        "best",
                        any!(best.into_iter().collect::<Vec<_>>().into_py_dict_pairs(py)),
                    ),
                    ("cost", any!(cost)),
                    ("iters", any!(iters)),
                ],
            )
        }
        Failure::Unidentifiable { params } => set(
            Unidentifiable::new_err(msg),
            vec![(
                "params",
                any!(params.iter().map(|&p| id_name(p)).collect::<Vec<_>>()),
            )],
        ),
        Failure::TooFewFrames { have, need } => set(
            TooFewFrames::new_err(msg),
            vec![("have", any!(have)), ("need", any!(need))],
        ),
        Failure::AmbiguousDirections {
            best,
            runner_up,
            best_cost,
            runner_up_cost,
        } => set(
            AmbiguousDirections::new_err(msg),
            vec![
                ("best", any!(dirs(&best))),
                ("runner_up", any!(dirs(&runner_up))),
                ("best_cost", any!(best_cost)),
                ("runner_up_cost", any!(runner_up_cost)),
            ],
        ),
        Failure::BadStart { id, value } => set(
            BadStart::new_err(msg),
            vec![("param", any!(id_name(id))), ("value", any!(value))],
        ),
        Failure::BadFrame { index, what } => set(
            BadFrame::new_err(msg),
            vec![("index", any!(index)), ("what", any!(what))],
        ),
        Failure::BadProblem => BadProblem::new_err(msg),
        Failure::Model(m) => BadModel::new_err(m.to_string()),
        Failure::OffsetAxisNotRotation(axis) => set(
            OffsetAxisNotRotation::new_err(msg),
            vec![("axis", any!(axis))],
        ),
    }
}

/// `{name: value}` in the given order.
trait IntoPyDictPairs {
    fn into_py_dict_pairs(self, py: Python<'_>) -> Py<PyDict>;
}

impl IntoPyDictPairs for Vec<(&str, f64)> {
    fn into_py_dict_pairs(self, py: Python<'_>) -> Py<PyDict> {
        let d = PyDict::new(py);
        for (k, v) in self {
            // keys and floats always convert
            let _ = d.set_item(k, v);
        }
        d.unbind()
    }
}

fn peak_err(py: Python<'_>, e: PeakError) -> PyErr {
    match e {
        PeakError::Rejected(r) => {
            let out = PeakRejected::new_err(e_text(&r));
            let v = out.value(py);
            let _ = v.setattr("reason", variant(&r));
            if let Reject::Saturated { slow, fast } = r {
                let _ = v.setattr("slow", slow);
                let _ = v.setattr("fast", fast);
            }
            out
        }
        other => PeakFailure::new_err(other.to_string()),
    }
}

fn e_text(r: &Reject) -> String {
    format!("frame rejected: {r:?}")
}

/// A start value: `Param`, or a plain float meaning `Param.fixed(x)`.
fn param(o: &Bound<'_, PyAny>) -> PyResult<core::Param> {
    if let Ok(p) = o.extract::<PyRef<'_, PyParam>>() {
        return Ok(p.0);
    }
    o.extract::<f64>()
        .map(core::Param::fixed)
        .map_err(|_| PyTypeError::new_err("a start value is a Param or a float (held fixed)"))
}

fn param_or(o: Option<&Bound<'_, PyAny>>, default: f64) -> PyResult<core::Param> {
    o.map_or(Ok(core::Param::fixed(default)), param)
}

/// A start value: `Param(start)` is fitted, `Param(value, free=False)` or
/// `Param.fixed(value)` is held. Angles in degrees.
#[pyclass(name = "Param", module = "hklrs.calib", frozen)]
struct PyParam(core::Param);

#[pymethods]
impl PyParam {
    #[new]
    #[pyo3(signature = (start, free=true))]
    fn new(start: f64, free: bool) -> Self {
        Self(core::Param { start, free })
    }

    #[staticmethod]
    fn fixed(value: f64) -> Self {
        Self(core::Param::fixed(value))
    }

    #[getter]
    fn start(&self) -> f64 {
        self.0.start
    }

    #[getter]
    fn free(&self) -> bool {
        self.0.free
    }

    fn __repr__(&self) -> String {
        format!(
            "Param({}, free={})",
            self.0.start,
            if self.0.free { "True" } else { "False" }
        )
    }
}

/// A peak: fractional pixel `pos` = (slow, fast), centres at integers, and
/// its standard error `sigma`. An infinite sigma component says the frame
/// carries no information on that axis.
#[pyclass(name = "Peak", module = "hklrs.calib", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyPeak(core::Peak);

#[pymethods]
impl PyPeak {
    #[new]
    fn new(pos: [f64; 2], sigma: [f64; 2]) -> Self {
        Self(core::Peak { pos, sigma })
    }

    /// A row detector's peak: slow index 0, no information along it.
    #[staticmethod]
    fn from_row(pos: f64, sigma: f64) -> Self {
        Self(core::Peak::from(core::Peak1 { pos, sigma }))
    }

    #[getter]
    fn pos(&self) -> [f64; 2] {
        self.0.pos
    }

    #[getter]
    fn sigma(&self) -> [f64; 2] {
        self.0.sigma
    }

    fn __repr__(&self) -> String {
        format!("Peak(pos={:?}, sigma={:?})", self.0.pos, self.0.sigma)
    }
}

/// What a frame shows: the direct beam, or the Bragg peak of `hkl`. A frame
/// is a reflection by this kind alone, never by its indices being non-zero.
#[pyclass(name = "FrameKind", module = "hklrs.calib", frozen)]
struct PyFrameKind(core::FrameKind);

#[pymethods]
impl PyFrameKind {
    #[staticmethod]
    fn primary_beam() -> Self {
        Self(core::FrameKind::PrimaryBeam)
    }

    #[staticmethod]
    fn reflection(hkl: [f64; 3]) -> Self {
        Self(core::FrameKind::Reflection(hkl.into()))
    }

    #[getter]
    fn is_reflection(&self) -> bool {
        matches!(self.0, core::FrameKind::Reflection(_))
    }

    /// The indices of a reflection, `None` for the primary beam.
    #[getter]
    fn hkl(&self) -> Option<[f64; 3]> {
        match self.0 {
            core::FrameKind::Reflection(h) => Some([h.x, h.y, h.z]),
            core::FrameKind::PrimaryBeam => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.hkl() {
            Some(h) => format!("FrameKind.reflection({h:?})"),
            None => "FrameKind.primary_beam()".into(),
        }
    }
}

/// One frame: the axis values (degrees for rotations) it was taken at, its
/// peak and kind. `index` is its place in your list, named in `BadFrame`.
#[pyclass(name = "CalFrame", module = "hklrs.calib", frozen)]
struct PyCalFrame(core::CalFrame);

#[pymethods]
impl PyCalFrame {
    #[new]
    #[pyo3(signature = (angles, peak, kind=None, index=0))]
    fn new(
        py: Python<'_>,
        angles: Vec<f64>,
        peak: &PyPeak,
        kind: Option<&PyFrameKind>,
        index: usize,
    ) -> PyResult<Self> {
        let kind = kind.map_or(core::FrameKind::PrimaryBeam, |k| k.0);
        core::CalFrame::new(index, angles, peak.0, kind)
            .map(Self)
            .map_err(|e| calib_err(py, e))
    }

    #[getter]
    fn angles(&self) -> Vec<f64> {
        self.0.angles().to_vec()
    }

    #[getter]
    fn peak(&self) -> PyPeak {
        PyPeak(self.0.peak())
    }

    #[getter]
    fn kind(&self) -> PyFrameKind {
        PyFrameKind(self.0.kind())
    }
}

/// A goniometer written as xrayutilities' `QConversion` is: circles outermost
/// first, each `(name, "z+")`, and the beam direction `r_i`. The sample
/// circles and the detector circles turn about the named lab axis; a
/// negative direction turns the other way. Frame angles are given in the
/// order sample circles then detector circles.
#[pyfunction]
#[pyo3(signature = (detector, sample=Vec::new(), beam=[1.0, 0.0, 0.0], name="xu"))]
fn xu_geometry(
    detector: Vec<(String, String)>,
    sample: Vec<(String, String)>,
    beam: [f64; 3],
    name: &str,
) -> PyResult<PyGeometry> {
    let mut b = hklrs::Geometry::builder(name);
    for (n, d) in &sample {
        b = b.sample_rotation(n, direction(d)?.vector().into_inner().into());
    }
    for (n, d) in &detector {
        b = b.detector_rotation(n, direction(d)?.vector().into_inner().into());
    }
    b.beam_direction(beam).build().map(PyGeometry).map_err(err)
}

fn direction(text: &str) -> PyResult<xu::Direction> {
    text.parse().map_err(err)
}

/// What an area detector fit is told about the rig. `directions` are the
/// two detector directions (`"z+"`, `"y+"`); `None` searches the beam
/// perpendicular candidates. `offset_axis` names the axis `angle_offset`
/// shifts. `sample` is needed for reflection frames.
#[pyclass(name = "AreaModel", module = "hklrs.calib", frozen)]
struct PyAreaModel(core::AreaModel);

#[pymethods]
impl PyAreaModel {
    #[new]
    #[pyo3(signature = (geometry, nch, sample=None, offset_axis=None, directions=None))]
    fn new(
        geometry: &PyGeometry,
        nch: [usize; 2],
        sample: Option<&PySample>,
        offset_axis: Option<String>,
        directions: Option<[String; 2]>,
    ) -> PyResult<Self> {
        let directions = directions
            .map(|[a, b]| Ok::<_, PyErr>([direction(&a)?, direction(&b)?]))
            .transpose()?;
        Ok(Self(core::AreaModel {
            geometry: geometry.0.clone(),
            sample: sample.map(|s| s.0.clone()),
            offset_axis,
            nch,
            directions,
        }))
    }
}

/// [`AreaModel`] for a row detector: one `direction` and `nch` channels.
#[pyclass(name = "LinearModel", module = "hklrs.calib", frozen)]
struct PyLinearModel(core::LinearModel);

#[pymethods]
impl PyLinearModel {
    #[new]
    #[pyo3(signature = (geometry, nch, sample=None, offset_axis=None, direction=None))]
    fn new(
        geometry: &PyGeometry,
        nch: usize,
        sample: Option<&PySample>,
        offset_axis: Option<String>,
        direction: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self(core::LinearModel {
            geometry: geometry.0.clone(),
            sample: sample.map(|s| s.0.clone()),
            offset_axis,
            nch,
            direction: direction.as_deref().map(direction_of).transpose()?,
        }))
    }
}

fn direction_of(text: &str) -> PyResult<xu::Direction> {
    direction(text)
}

/// Start values of an area fit, each a `Param` or a float held fixed. The
/// wavelength has no default: a fit that frees it needs reflection frames.
/// `tiltazimuth`, `tilt`, `detrot` and `angle_offset` default to fixed 0.
#[pyclass(name = "AreaStart", module = "hklrs.calib", frozen)]
struct PyAreaStart(core::AreaStart);

#[pymethods]
impl PyAreaStart {
    #[new]
    #[pyo3(signature = (cch1, cch2, pwidth1, pwidth2, distance, wavelength,
                        tiltazimuth=None, tilt=None, detrot=None, angle_offset=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cch1: &Bound<'_, PyAny>,
        cch2: &Bound<'_, PyAny>,
        pwidth1: &Bound<'_, PyAny>,
        pwidth2: &Bound<'_, PyAny>,
        distance: &Bound<'_, PyAny>,
        wavelength: &Bound<'_, PyAny>,
        tiltazimuth: Option<&Bound<'_, PyAny>>,
        tilt: Option<&Bound<'_, PyAny>>,
        detrot: Option<&Bound<'_, PyAny>>,
        angle_offset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(core::AreaStart {
            cch1: param(cch1)?,
            cch2: param(cch2)?,
            pwidth1: param(pwidth1)?,
            pwidth2: param(pwidth2)?,
            distance: param(distance)?,
            tiltazimuth: param_or(tiltazimuth, 0.0)?,
            tilt: param_or(tilt, 0.0)?,
            detrot: param_or(detrot, 0.0)?,
            angle_offset: param_or(angle_offset, 0.0)?,
            wavelength: param(wavelength)?,
        }))
    }
}

/// Start values of a linear fit: `cch`, `pwidth`, `distance`, `tilt`,
/// `angle_offset`, `wavelength`.
#[pyclass(name = "LinearStart", module = "hklrs.calib", frozen)]
struct PyLinearStart(core::LinearStart);

#[pymethods]
impl PyLinearStart {
    #[new]
    #[pyo3(signature = (cch, pwidth, distance, wavelength, tilt=None, angle_offset=None))]
    fn new(
        cch: &Bound<'_, PyAny>,
        pwidth: &Bound<'_, PyAny>,
        distance: &Bound<'_, PyAny>,
        wavelength: &Bound<'_, PyAny>,
        tilt: Option<&Bound<'_, PyAny>>,
        angle_offset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(core::LinearStart {
            cch: param(cch)?,
            pwidth: param(pwidth)?,
            distance: param(distance)?,
            tilt: param_or(tilt, 0.0)?,
            angle_offset: param_or(angle_offset, 0.0)?,
            wavelength: param(wavelength)?,
        }))
    }
}

/// An area start plus the sample misorientation `Rz(azimuth)·Ry(tilt)` of
/// `calibrate_area_hkl`, degrees, default fixed 0.
#[pyclass(name = "AreaHklStart", module = "hklrs.calib", frozen)]
struct PyAreaHklStart(core::AreaHklStart);

#[pymethods]
impl PyAreaHklStart {
    #[new]
    #[pyo3(signature = (area, sample_tilt=None, sample_azimuth=None))]
    fn new(
        area: &PyAreaStart,
        sample_tilt: Option<&Bound<'_, PyAny>>,
        sample_azimuth: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(core::AreaHklStart {
            area: area.0,
            sample_tilt: param_or(sample_tilt, 0.0)?,
            sample_azimuth: param_or(sample_azimuth, 0.0)?,
        }))
    }
}

#[pyclass(name = "LinearHklStart", module = "hklrs.calib", frozen)]
struct PyLinearHklStart(core::LinearHklStart);

#[pymethods]
impl PyLinearHklStart {
    #[new]
    #[pyo3(signature = (line, sample_tilt=None, sample_azimuth=None))]
    fn new(
        line: &PyLinearStart,
        sample_tilt: Option<&Bound<'_, PyAny>>,
        sample_azimuth: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(core::LinearHklStart {
            line: line.0,
            sample_tilt: param_or(sample_tilt, 0.0)?,
            sample_azimuth: param_or(sample_azimuth, 0.0)?,
        }))
    }
}

enum Value {
    Area(xu::Area),
    Linear(xu::Linear),
}

/// A converged calibration. `value` is the fitted `hklrs.xu.Area` or
/// `Linear`. `params` holds every parameter by name, fitted or held (angles
/// in degrees; `cch1`, `pwidth1` ... for an area detector, `cch`, `pwidth`
/// for a row; an hkl fit adds `wavelength`, `angle_offset`, `sample_tilt`,
/// `sample_azimuth`); `directions` are the detector directions used.
/// `free_ids`, `values` and `sigma` cover the free parameters only (a held
/// parameter has no sigma), in the order of `cov`. `rms` is in units of
/// sigma: about 1 for a good model, large for a local minimum (several
/// starts are the remedy). `rejected` lists `(frame index, reason)`; `stop`
/// is the rule that ended the run: gradient, cost or param.
#[pyclass(name = "Fit", module = "hklrs.calib", frozen)]
struct PyFit {
    value: Value,
    params: Vec<(&'static str, f64)>,
    directions: Vec<String>,
    free: Vec<core::Fitted>,
    cov: Vec<Vec<f64>>,
    rms: f64,
    dof: usize,
    rejected: Vec<(usize, Reject)>,
    iters: usize,
    stop: core::Stop,
}

impl PyFit {
    fn pack<T>(fit: &core::Fit<T>, value: Value, extra: Option<[f64; 4]>) -> Self {
        let cov = fit.cov();
        let mut params: Vec<(&'static str, f64)> = match &value {
            Value::Area(a) => vec![
                ("cch1", a.cch1),
                ("cch2", a.cch2),
                ("pwidth1", a.pwidth1),
                ("pwidth2", a.pwidth2),
                ("distance", a.distance),
                ("tiltazimuth", a.tiltazimuth.0),
                ("tilt", a.tilt.0),
                ("detrot", a.detrot.0),
            ],
            Value::Linear(l) => vec![
                ("cch", l.cch),
                ("pwidth", l.pwidth),
                ("distance", l.distance),
                ("tilt", l.tilt.0),
            ],
        };
        if let Some([wl, off, tilt, az]) = extra {
            params.extend([
                ("wavelength", wl),
                ("angle_offset", off),
                ("sample_tilt", tilt),
                ("sample_azimuth", az),
            ]);
        }
        let directions = match &value {
            Value::Area(a) => dirs(&[a.dir1, a.dir2]),
            Value::Linear(l) => dirs(&[l.dir]),
        };
        Self {
            value,
            params,
            directions,
            free: fit.free().to_vec(),
            cov: (0..cov.nrows())
                .map(|i| (0..cov.ncols()).map(|j| cov[(i, j)]).collect())
                .collect(),
            rms: fit.rms(),
            dof: fit.dof(),
            rejected: fit.rejected().to_vec(),
            iters: fit.iters(),
            stop: fit.stop(),
        }
    }
}

#[pymethods]
impl PyFit {
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.value {
            Value::Area(a) => Py::new(py, PyXuArea(a.clone()))?.into_any(),
            Value::Linear(l) => Py::new(py, PyXuLinear(l.clone()))?.into_any(),
        })
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.params {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    #[getter]
    fn directions(&self) -> Vec<String> {
        self.directions.clone()
    }

    #[getter]
    fn free_ids(&self) -> Vec<&'static str> {
        self.free.iter().map(|f| id_name(f.id)).collect()
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for f in &self.free {
            d.set_item(id_name(f.id), f.value)?;
        }
        Ok(d)
    }

    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for f in &self.free {
            d.set_item(id_name(f.id), f.sigma)?;
        }
        Ok(d)
    }

    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        PyArray2::from_vec2(py, &self.cov).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn rms(&self) -> f64 {
        self.rms
    }

    #[getter]
    fn dof(&self) -> usize {
        self.dof
    }

    #[getter]
    fn iters(&self) -> usize {
        self.iters
    }

    #[getter]
    fn stop(&self) -> String {
        variant(&self.stop)
    }

    #[getter]
    fn rejected(&self) -> Vec<(usize, String)> {
        self.rejected
            .iter()
            .map(|(i, r)| (*i, variant(r)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Fit(rms={}, dof={}, stop={}, free={:?})",
            self.rms,
            self.dof,
            variant(&self.stop),
            self.free_ids()
        )
    }
}

fn frames_of(frames: Vec<PyRef<'_, PyCalFrame>>) -> Vec<core::CalFrame> {
    frames.iter().map(|f| f.0.clone()).collect()
}

/// Fit an area detector to `frames` (see `calibrate_linear` for rows). Every
/// start runs; the lowest-cost converged one wins. Raises a `CalibError`
/// subclass when no run converges or the problem is ill-posed.
#[pyfunction]
fn calibrate_area(
    py: Python<'_>,
    model: &PyAreaModel,
    frames: Vec<PyRef<'_, PyCalFrame>>,
    starts: Vec<PyRef<'_, PyAreaStart>>,
) -> PyResult<PyFit> {
    let (frames, starts) = (
        frames_of(frames),
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
    );
    let fit = py
        .detach(|| core::calibrate_area(&model.0, &frames, &starts))
        .map_err(|e| calib_err(py, e))?;
    Ok(PyFit::pack(&fit, Value::Area(fit.value().clone()), None))
}

/// Fit a row detector; the frames' peaks carry no slow-axis information
/// (`Peak.from_row`).
#[pyfunction]
fn calibrate_linear(
    py: Python<'_>,
    model: &PyLinearModel,
    frames: Vec<PyRef<'_, PyCalFrame>>,
    starts: Vec<PyRef<'_, PyLinearStart>>,
) -> PyResult<PyFit> {
    let (frames, starts) = (
        frames_of(frames),
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
    );
    let fit = py
        .detach(|| core::calibrate_linear(&model.0, &frames, &starts))
        .map_err(|e| calib_err(py, e))?;
    Ok(PyFit::pack(&fit, Value::Linear(fit.value().clone()), None))
}

/// xu's `area_detector_calib_hkl`: the detector, the sample misorientation
/// and (when free) the wavelength from primary-beam and reflection frames,
/// at least one of each. Frames are told apart by `FrameKind`.
#[pyfunction]
fn calibrate_area_hkl(
    py: Python<'_>,
    model: &PyAreaModel,
    frames: Vec<PyRef<'_, PyCalFrame>>,
    starts: Vec<PyRef<'_, PyAreaHklStart>>,
) -> PyResult<PyFit> {
    let (frames, starts) = (
        frames_of(frames),
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
    );
    let fit = py
        .detach(|| core::calibrate_area_hkl(&model.0, &frames, &starts))
        .map_err(|e| calib_err(py, e))?;
    let h = fit.value();
    Ok(PyFit::pack(
        &fit,
        Value::Area(h.area.clone()),
        Some([
            h.wavelength,
            h.angle_offset.0,
            h.sample_tilt.0,
            h.sample_azimuth.0,
        ]),
    ))
}

/// [`calibrate_area_hkl`] for a row detector. One detector circle sees only
/// tilt·sin(azimuth) of a (0 0 L) reflection's misorientation: fit both and
/// it raises `Unidentifiable`.
#[pyfunction]
fn calibrate_linear_hkl(
    py: Python<'_>,
    model: &PyLinearModel,
    frames: Vec<PyRef<'_, PyCalFrame>>,
    starts: Vec<PyRef<'_, PyLinearHklStart>>,
) -> PyResult<PyFit> {
    let (frames, starts) = (
        frames_of(frames),
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
    );
    let fit = py
        .detach(|| core::calibrate_linear_hkl(&model.0, &frames, &starts))
        .map_err(|e| calib_err(py, e))?;
    let h = fit.value();
    Ok(PyFit::pack(
        &fit,
        Value::Linear(h.line.clone()),
        Some([
            h.wavelength,
            h.angle_offset.0,
            h.sample_tilt.0,
            h.sample_azimuth.0,
        ]),
    ))
}

fn options(
    window: Option<usize>,
    tol: Option<f64>,
    max_passes: Option<usize>,
    border: Option<usize>,
    edge_fraction: Option<f64>,
    min_snr: Option<f64>,
) -> core::peak::PeakOptions {
    let d = core::peak::PeakOptions::default();
    core::peak::PeakOptions {
        window: window.unwrap_or(d.window),
        tol: tol.unwrap_or(d.tol),
        max_passes: max_passes.unwrap_or(d.max_passes),
        border: border.unwrap_or(d.border),
        edge_fraction: edge_fraction.unwrap_or(d.edge_fraction),
        min_snr: min_snr.unwrap_or(d.min_snr),
    }
}

/// The peak of a frame (float64 counts, shape (slow, fast)): centre of mass
/// of a re-centred window above the local background, with its Poisson
/// error. `mask` is True at bad pixels. A frame without a usable peak raises
/// `PeakRejected` with its `reason` (no_signal, saturated, touches_edge,
/// not_converged); unlike xu's `_peak_position`, a background level does not
/// pull the peak away (XU-20).
#[pyfunction]
#[pyo3(signature = (image, mask=None, saturation=None, window=None, tol=None,
                    max_passes=None, border=None, edge_fraction=None, min_snr=None))]
#[allow(clippy::too_many_arguments)]
fn find_peak(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    saturation: Option<f64>,
    window: Option<usize>,
    tol: Option<f64>,
    max_passes: Option<usize>,
    border: Option<usize>,
    edge_fraction: Option<f64>,
    min_snr: Option<f64>,
) -> PyResult<PyPeak> {
    let image = super::image(&image)?;
    let mask = mask.as_ref().map(super::image).transpose()?;
    let opts = options(window, tol, max_passes, border, edge_fraction, min_snr);
    py.detach(|| core::peak::find_peak(&image, mask.as_ref(), saturation, &opts))
        .map(PyPeak)
        .map_err(|e| peak_err(py, e))
}

/// The peak of one row of channels, refined by a pseudo-Voigt fit; the
/// returned `Peak` is `Peak.from_row` (slow 0, infinite slow sigma).
#[pyfunction]
#[pyo3(signature = (row, mask=None, saturation=None, window=None, tol=None,
                    max_passes=None, border=None, edge_fraction=None, min_snr=None))]
#[allow(clippy::too_many_arguments)]
fn find_peak_1d(
    py: Python<'_>,
    row: PyReadonlyArray1<'_, f64>,
    mask: Option<PyReadonlyArray1<'_, bool>>,
    saturation: Option<f64>,
    window: Option<usize>,
    tol: Option<f64>,
    max_passes: Option<usize>,
    border: Option<usize>,
    edge_fraction: Option<f64>,
    min_snr: Option<f64>,
) -> PyResult<PyPeak> {
    let data: Vec<f64> = row.as_array().iter().copied().collect();
    let mask: Option<Vec<bool>> = mask.map(|m| m.as_array().iter().copied().collect());
    let opts = options(window, tol, max_passes, border, edge_fraction, min_snr);
    py.detach(|| core::peak::find_peak_1d(&data, mask.as_deref(), saturation, &opts))
        .map(|p| PyPeak(p.into()))
        .map_err(|e| peak_err(py, e))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyParam>()?;
    m.add_class::<PyPeak>()?;
    m.add_class::<PyFrameKind>()?;
    m.add_class::<PyCalFrame>()?;
    m.add_class::<PyAreaModel>()?;
    m.add_class::<PyLinearModel>()?;
    m.add_class::<PyAreaStart>()?;
    m.add_class::<PyLinearStart>()?;
    m.add_class::<PyAreaHklStart>()?;
    m.add_class::<PyLinearHklStart>()?;
    m.add_class::<PyFit>()?;
    m.add("CalibError", py.get_type::<CalibError>())?;
    m.add("NotConverged", py.get_type::<NotConverged>())?;
    m.add("Unidentifiable", py.get_type::<Unidentifiable>())?;
    m.add("TooFewFrames", py.get_type::<TooFewFrames>())?;
    m.add("AmbiguousDirections", py.get_type::<AmbiguousDirections>())?;
    m.add("BadStart", py.get_type::<BadStart>())?;
    m.add("BadFrame", py.get_type::<BadFrame>())?;
    m.add("BadProblem", py.get_type::<BadProblem>())?;
    m.add("BadModel", py.get_type::<BadModel>())?;
    m.add(
        "OffsetAxisNotRotation",
        py.get_type::<OffsetAxisNotRotation>(),
    )?;
    m.add("PeakFailure", py.get_type::<PeakFailure>())?;
    m.add("PeakRejected", py.get_type::<PeakRejected>())?;
    m.add_function(wrap_pyfunction!(xu_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_area, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_linear, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_area_hkl, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_linear_hkl, m)?)?;
    m.add_function(wrap_pyfunction!(find_peak, m)?)?;
    m.add_function(wrap_pyfunction!(find_peak_1d, m)?)
}
