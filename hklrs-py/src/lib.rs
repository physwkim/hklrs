//! Python bindings: conversions only. Angles are degrees at this boundary,
//! translations metres, wavelength Å.

use hklrs::nalgebra::{Matrix3, Point3, Unit, Vector3};
use hklrs::xu;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: hklrs::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

create_exception!(hklrs, InverseError, PyValueError, "The inverse failed.");
create_exception!(
    hklrs,
    Unreachable,
    InverseError,
    "|Q| exceeds 4 pi / wavelength; carries q and q_max (Å⁻¹)."
);
create_exception!(
    hklrs,
    NoSolution,
    InverseError,
    "No axis values satisfy the mode."
);

fn inverse_err(py: Python<'_>, e: hklrs::InverseError) -> PyErr {
    use hklrs::InverseError as E;
    let msg = e.to_string();
    match e {
        E::Unreachable { q, q_max } => {
            let err = Unreachable::new_err(msg);
            let v = err.value(py);
            // Plain floats cannot fail to set on a fresh exception.
            let _ = v.setattr("q", q);
            let _ = v.setattr("q_max", q_max);
            err
        }
        E::NoSolution => NoSolution::new_err(msg),
        E::Geometry(g) => err(g),
        _ => InverseError::new_err(msg),
    }
}

fn rows(m: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[(i, j)]))
}

fn v3(v: &Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

#[pyclass(name = "Lattice", frozen)]
struct PyLattice(hklrs::Lattice);

#[pymethods]
impl PyLattice {
    /// Lengths in Å, angles in degrees.
    #[new]
    fn new(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> PyResult<Self> {
        hklrs::Lattice::from_degrees(a, b, c, alpha, beta, gamma)
            .map(Self)
            .map_err(err)
    }

    /// (a, b, c, alpha, beta, gamma), degrees.
    #[getter]
    fn parameters(&self) -> (f64, f64, f64, f64, f64, f64) {
        let l = &self.0;
        let d = |r: hklrs::Rad| hklrs::Deg::from(r).0;
        (l.a(), l.b(), l.c(), d(l.alpha()), d(l.beta()), d(l.gamma()))
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.0.volume()
    }

    /// Busing-Levy B with 2π.
    fn b_matrix(&self) -> [[f64; 3]; 3] {
        rows(&self.0.b_matrix())
    }

    fn metric(&self) -> [[f64; 3]; 3] {
        rows(&self.0.metric())
    }

    /// Crystallographic reciprocal lattice, without 2π.
    fn reciprocal(&self) -> Self {
        Self(self.0.reciprocal())
    }

    fn d_spacing(&self, hkl: [f64; 3]) -> f64 {
        self.0.d_spacing(&Vector3::from(hkl))
    }
}

#[pyclass(name = "Geometry", frozen)]
struct PyGeometry(hklrs::Geometry);

#[pymethods]
impl PyGeometry {
    #[staticmethod]
    fn preset(name: &str) -> PyResult<Self> {
        hklrs::presets::preset(name).map(Self).map_err(err)
    }

    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    #[getter]
    fn axis_names(&self) -> Vec<String> {
        self.0.axes().iter().map(|a| a.name().to_owned()).collect()
    }

    #[getter]
    fn sample_axes(&self) -> Vec<String> {
        self.0
            .sample_stack()
            .iter()
            .map(|&i| self.0.axes()[i].name().to_owned())
            .collect()
    }

    #[getter]
    fn detector_axes(&self) -> Vec<String> {
        self.0
            .detector_stack()
            .iter()
            .map(|&i| self.0.axes()[i].name().to_owned())
            .collect()
    }

    #[getter]
    fn source_axes(&self) -> Vec<String> {
        self.0
            .source_stack()
            .iter()
            .map(|&i| self.0.axes()[i].name().to_owned())
            .collect()
    }

    #[getter]
    fn beam_direction(&self) -> [f64; 3] {
        v3(&self.0.beam_direction())
    }
}

/// Axis values (degrees / metres, geometry axis order), wavelength (Å) and
/// detector point (detector-arm frame, metres; default 1 m along the beam).
fn kernel_inputs(
    geo: &hklrs::Geometry,
    axes: &[f64],
    wavelength: f64,
    point: Option<[f64; 3]>,
) -> PyResult<(Vec<f64>, hklrs::Wavelength, Point3<f64>)> {
    let values = geo.values_from_degrees(axes).map_err(err)?;
    let wl = hklrs::Wavelength::new(wavelength).map_err(err)?;
    let p = point.map_or_else(
        || Point3::from(geo.beam_direction().into_inner()),
        Point3::from,
    );
    Ok((values, wl, p))
}

#[pyclass(name = "Reflection", frozen)]
struct PyReflection(hklrs::Reflection);

#[pymethods]
impl PyReflection {
    /// Miller indices found at `axes`; arguments as for `forward`.
    #[new]
    #[pyo3(signature = (geometry, hkl, axes, wavelength, point=None))]
    fn new(
        geometry: &PyGeometry,
        hkl: [f64; 3],
        axes: Vec<f64>,
        wavelength: f64,
        point: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let geo = &geometry.0;
        let (values, wl, p) = kernel_inputs(geo, &axes, wavelength, point)?;
        hklrs::Reflection::measure(Vector3::from(hkl), geo, &values, wl, &p)
            .map(Self)
            .map_err(err)
    }

    #[getter]
    fn hkl(&self) -> [f64; 3] {
        v3(self.0.hkl())
    }

    /// Measured Q in the sample frame, Å⁻¹ with 2π.
    #[getter]
    fn q_sample(&self) -> [f64; 3] {
        v3(self.0.q_sample())
    }
}

#[pyclass(name = "Sample", frozen)]
struct PySample(hklrs::Sample);

#[pymethods]
impl PySample {
    /// `u` is a proper rotation matrix (rows); identity when omitted.
    #[new]
    #[pyo3(signature = (name, lattice, u=None))]
    fn new(name: &str, lattice: &PyLattice, u: Option<[[f64; 3]; 3]>) -> PyResult<Self> {
        let u = u.map_or_else(Matrix3::identity, |m| Matrix3::from_fn(|i, j| m[i][j]));
        let o = hklrs::Orientation::from_u_matrix(&u).map_err(err)?;
        Ok(Self(hklrs::Sample::new(name, lattice.0, o)))
    }

    /// U from two reflections: the primary exactly, the secondary fixes the
    /// rotation about it.
    #[staticmethod]
    fn busing_levy(
        name: &str,
        lattice: &PyLattice,
        primary: &PyReflection,
        secondary: &PyReflection,
    ) -> PyResult<Self> {
        let o =
            hklrs::Orientation::busing_levy(primary.0.clone(), secondary.0.clone()).map_err(err)?;
        Ok(Self(hklrs::Sample::new(name, lattice.0, o)))
    }

    fn u(&self) -> [[f64; 3]; 3] {
        rows(&self.0.u().to_rotation_matrix().into_inner())
    }

    fn ub(&self) -> [[f64; 3]; 3] {
        rows(&self.0.ub())
    }
}

/// An E4C inverse mode, validated against the geometry it was built for.
#[pyclass(name = "Mode", frozen)]
struct PyMode {
    geometry: hklrs::Geometry,
    name: String,
    hkl2: Option<Vector3<f64>>,
    reference: Option<Vector3<f64>>,
    psi: f64,
}

impl PyMode {
    fn build(&self) -> Result<hklrs::Mode<'_>, hklrs::InverseError> {
        use hklrs::inverse::e4c;
        let g = &self.geometry;
        match self.name.as_str() {
            "bissector" => e4c::bissector(g),
            "constant_omega" => e4c::constant_omega(g),
            "constant_chi" => e4c::constant_chi(g),
            "constant_phi" => e4c::constant_phi(g),
            "double_diffraction" => e4c::double_diffraction(g, self.hkl2.unwrap_or_default()),
            _ => e4c::psi_constant(g, self.reference.unwrap_or_default(), self.psi),
        }
    }
}

const MODES: [&str; 6] = [
    "bissector",
    "constant_omega",
    "constant_chi",
    "constant_phi",
    "double_diffraction",
    "psi_constant",
];

#[pymethods]
impl PyMode {
    /// E4C mode by name: bissector, constant_omega, constant_chi,
    /// constant_phi, double_diffraction (needs `hkl2`) or psi_constant
    /// (needs `reference` and `psi`, degrees). The constant_* modes hold
    /// their axis at the `current` passed to `solve`.
    #[new]
    #[pyo3(signature = (geometry, name, hkl2=None, reference=None, psi=None))]
    fn new(
        py: Python<'_>,
        geometry: &PyGeometry,
        name: &str,
        hkl2: Option<[f64; 3]>,
        reference: Option<[f64; 3]>,
        psi: Option<f64>,
    ) -> PyResult<Self> {
        if !MODES.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "unknown mode {name:?}; one of {MODES:?}"
            )));
        }
        let need = |ok: bool, what: &str| {
            if ok {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("mode {name:?} needs {what}")))
            }
        };
        match name {
            "double_diffraction" => need(hkl2.is_some(), "hkl2")?,
            "psi_constant" => need(reference.is_some() && psi.is_some(), "reference and psi")?,
            _ => {}
        }
        let mode = Self {
            geometry: geometry.0.clone(),
            name: name.to_owned(),
            hkl2: hkl2.map(Vector3::from),
            reference: reference.map(Vector3::from),
            psi: psi.map_or(0.0, f64::to_radians),
        };
        mode.build().map_err(|e| inverse_err(py, e))?;
        Ok(mode)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
}

fn request<'a>(
    mode: &'a hklrs::Mode<'a>,
    sample: &PySample,
    hkl: [f64; 3],
    wavelength: f64,
    current: &'a [f64],
) -> PyResult<hklrs::InverseRequest<'a>> {
    Ok(hklrs::InverseRequest {
        mode,
        ub: sample.0.ub(),
        wavelength: hklrs::Wavelength::new(wavelength).map_err(err)?,
        target: hklrs::Target::Hkl(Vector3::from(hkl)),
        current,
    })
}

/// Every solution for `hkl`, nearest to `current` first. `current` and the
/// results are all axes in geometry order, degrees / metres; held axes are
/// returned as given. Raises `Unreachable` when |Q| > 4π/λ·(1 + 1e-12), `NoSolution`
/// when the mode has none.
#[pyfunction]
fn solve(
    py: Python<'_>,
    mode: &PyMode,
    sample: &PySample,
    hkl: [f64; 3],
    wavelength: f64,
    current: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let ie = |e| inverse_err(py, e);
    let m = mode.build().map_err(ie)?;
    let geo = m.geometry();
    let cur = geo.values_from_degrees(&current).map_err(err)?;
    let all = request(&m, sample, hkl, wavelength, &cur)?
        .solve()
        .map_err(ie)?;
    all.iter()
        .map(|s| geo.values_to_degrees(s).map_err(err))
        .collect()
}

/// One solution: `picker` "closest" (nearest to `current`) or "first" (the
/// first the search reached). Arguments and errors as for `solve`.
#[pyfunction]
#[pyo3(signature = (mode, sample, hkl, wavelength, current, picker="closest"))]
fn pick(
    py: Python<'_>,
    mode: &PyMode,
    sample: &PySample,
    hkl: [f64; 3],
    wavelength: f64,
    current: Vec<f64>,
    picker: &str,
) -> PyResult<Vec<f64>> {
    let picker = match picker {
        "closest" => hklrs::Picker::Closest,
        "first" => hklrs::Picker::First,
        other => {
            return Err(PyValueError::new_err(format!(
                "picker must be \"closest\" or \"first\", got {other:?}"
            )));
        }
    };
    let ie = |e| inverse_err(py, e);
    let m = mode.build().map_err(ie)?;
    let geo = m.geometry();
    let cur = geo.values_from_degrees(&current).map_err(err)?;
    let s = request(&m, sample, hkl, wavelength, &cur)?
        .pick(picker)
        .map_err(ie)?;
    geo.values_to_degrees(&s).map_err(err)
}

/// ψ of `reference` (Miller indices) at `axes`, degrees.
#[pyfunction]
fn psi(
    py: Python<'_>,
    geometry: &PyGeometry,
    sample: &PySample,
    reference: [f64; 3],
    axes: Vec<f64>,
) -> PyResult<f64> {
    let values = geometry.0.values_from_degrees(&axes).map_err(err)?;
    hklrs::inverse::psi(
        &geometry.0,
        &sample.0.ub(),
        &Vector3::from(reference),
        &values,
    )
    .map(|p| hklrs::Deg::from(hklrs::Rad(p)).0)
    .map_err(|e| inverse_err(py, e))
}

/// Forward for one detector point. `axes` in geometry axis order (degrees /
/// metres); `point` in the detector-arm frame, metres, default 1 m along the
/// beam. Returns a dict of hkl, q_lab, q_sample, kf, ki, tth (degrees) and q (Å⁻¹).
#[pyfunction]
#[pyo3(signature = (geometry, sample, axes, wavelength, point=None))]
fn forward<'py>(
    py: Python<'py>,
    geometry: &PyGeometry,
    sample: &PySample,
    axes: Vec<f64>,
    wavelength: f64,
    point: Option<[f64; 3]>,
) -> PyResult<Bound<'py, PyDict>> {
    let geo = &geometry.0;
    let (values, wl, p) = kernel_inputs(geo, &axes, wavelength, point)?;
    let f = hklrs::forward(geo, &sample.0, &values, wl, &p).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("hkl", v3(&f.hkl))?;
    let s = &f.scattering;
    d.set_item("q_lab", v3(&s.q_lab))?;
    d.set_item("q_sample", v3(&s.q_sample))?;
    d.set_item("kf", v3(&s.kf))?;
    d.set_item("ki", v3(&s.ki))?;
    d.set_item("tth", hklrs::Deg::from(s.tth).0)?;
    d.set_item("q", s.q)?;
    Ok(d)
}

/// Read-only pseudo axes of libhkl's q, q2, tth2, qper_qpar, incidence and
/// emergence engines at `axes` (geometry axis order, degrees / metres):
/// `q` -> (q,), `q2` -> (q, alpha), `tth2` -> (tth, alpha), `qper_qpar` ->
/// (qper, qpar), `incidence`/`emergence` -> (angle, azimuth or None). Angles in
/// degrees, q in Å⁻¹. `surface` is the sample-frame normal (default y) for the
/// engines that take one; `point` as for `forward`.
#[pyfunction]
#[pyo3(signature = (geometry, engine, axes, wavelength, surface=None, point=None))]
fn pseudo(
    geometry: &PyGeometry,
    engine: &str,
    axes: Vec<f64>,
    wavelength: f64,
    surface: Option<[f64; 3]>,
    point: Option<[f64; 3]>,
) -> PyResult<Vec<Option<f64>>> {
    use hklrs::pseudo as ps;
    let geo = &geometry.0;
    let (values, wl, p) = kernel_inputs(geo, &axes, wavelength, point)?;
    let n = Vector3::from(surface.unwrap_or([0.0, 1.0, 0.0]));
    let d = |r: hklrs::Rad| Some(hklrs::Deg::from(r).0);
    let out = match engine {
        "q" => vec![Some(ps::q_at(geo, &values, wl, &p).map_err(err)?)],
        "q2" => ps::q2_at(geo, &values, wl, &p)
            .map(|(q, a)| vec![Some(q), d(a)])
            .map_err(err)?,
        "tth2" => ps::tth2_at(geo, &values, wl, &p)
            .map(|(t, a)| vec![d(t), d(a)])
            .map_err(err)?,
        "qper_qpar" => ps::qper_qpar_at(geo, &values, wl, &p, &n)
            .map(|(a, b)| vec![Some(a), Some(b)])
            .map_err(err)?,
        "incidence" => ps::incidence_at(geo, &values, wl, &p, &n)
            .map(|(a, z)| vec![d(a), z.and_then(d)])
            .map_err(err)?,
        "emergence" => ps::emergence_at(geo, &values, wl, &p, &n)
            .map(|(a, z)| vec![d(a), z.and_then(d)])
            .map_err(err)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pseudo engine {other:?}"
            )));
        }
    };
    Ok(out)
}

/// Pixels in the detector-arm frame (all axes zero), metres. Built by the
/// xu importers; opaque apart from its shape, mask and pixel positions.
#[pyclass(name = "PixelDetector", frozen)]
struct PyPixelDetector(hklrs::PixelDetector);

#[pymethods]
impl PyPixelDetector {
    /// (slow, fast) pixel counts.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }

    /// Bool array of the detector shape, True where a pixel is masked; None
    /// when no pixel is.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        self.0
            .mask()
            .map(|m| PyArray1::from_slice(py, m).reshape(self.0.shape()))
            .transpose()
    }

    /// The detector with `mask` (bool array of its shape, True excludes the
    /// pixel) in place of any earlier mask.
    fn with_mask(&self, mask: PyReadonlyArray2<'_, bool>) -> PyResult<Self> {
        let (slow, fast) = self.0.shape();
        if mask.shape() != [slow, fast] {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} does not match the detector shape ({slow}, {fast})",
                mask.shape()
            )));
        }
        let masked = mask.as_array().iter().copied().collect();
        self.0.clone().with_mask(masked).map(Self).map_err(err)
    }

    /// Centre of pixel (slow, fast) in the detector-arm frame.
    fn pixel(&self, slow: usize, fast: usize) -> PyResult<[f64; 3]> {
        self.0.pixel(slow, fast).map(|p| v3(&p.coords)).map_err(err)
    }

    /// Pixels `slow` x `fast`, each a half-open (start, end), as a detector of
    /// their own whose pixel (0, 0) is this one's (slow start, fast start).
    fn roi(&self, slow: (usize, usize), fast: (usize, usize)) -> PyResult<Self> {
        self.0
            .roi(slow.0..slow.1, fast.0..fast.1)
            .map(Self)
            .map_err(err)
    }

    /// Blocks of `slow` x `fast` pixels as one pixel each, at the block
    /// centre. Trailing pixels that fill no block are dropped; a block is
    /// masked when any of its pixels is.
    fn bin(&self, slow: usize, fast: usize) -> PyResult<Self> {
        self.0.bin(slow, fast).map(Self).map_err(err)
    }
}

fn xu_direction(text: &str) -> PyResult<xu::Direction> {
    text.parse().map_err(err)
}

/// `r_i` of the frame the direction strings are given in.
fn xu_beam(beam: [f64; 3]) -> PyResult<Unit<Vector3<f64>>> {
    let v = Vector3::from(beam);
    v.iter()
        .all(|x| x.is_finite())
        .then(|| Unit::try_new(v, 0.0))
        .flatten()
        .ok_or_else(|| {
            err(hklrs::Error::ZeroVector {
                what: "xu beam".to_owned(),
            })
        })
}

/// Each keyword of `options` as (name, value); a name `apply` does not take
/// is a TypeError, as for any unexpected keyword.
fn xu_options(
    options: Option<&Bound<'_, PyDict>>,
    mut apply: impl FnMut(&str, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<()> {
    for (key, value) in options.iter().flat_map(|d| d.iter()) {
        let key: String = key.extract()?;
        if !apply(&key, &value)? {
            return Err(PyTypeError::new_err(format!(
                "unexpected keyword argument {key:?}"
            )));
        }
    }
    Ok(())
}

/// xrayutilities `init_area`, lengths in metres. Keyword options: `tilt`,
/// `tiltazimuth`, `detrot` (degrees), `nav` = (nav1, nav2) and `roi` =
/// (start1, end1, start2, end2), half-open. Each nav must divide its roi
/// length, and the roi must lie on the chip.
#[pyclass(name = "Area", module = "hklrs.xu", frozen)]
struct PyXuArea(xu::Area);

impl PyXuArea {
    fn with(mut area: xu::Area, options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        xu_options(options, |key, value| {
            match key {
                "tilt" => area.tilt = hklrs::Deg(value.extract()?),
                "tiltazimuth" => area.tiltazimuth = hklrs::Deg(value.extract()?),
                "detrot" => area.detrot = hklrs::Deg(value.extract()?),
                "nav" => area.nav = value.extract()?,
                "roi" => area.roi = value.extract()?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self(area))
    }
}

#[pymethods]
impl PyXuArea {
    #[new]
    #[pyo3(signature = (dir1, dir2, cch, nch, distance, pwidth, **options))]
    fn new(
        dir1: &str,
        dir2: &str,
        cch: [f64; 2],
        nch: [usize; 2],
        distance: f64,
        pwidth: [f64; 2],
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (d1, d2) = (xu_direction(dir1)?, xu_direction(dir2)?);
        Self::with(xu::Area::new(d1, d2, cch, nch, distance, pwidth), options)
    }

    /// `init_area` with `chpdeg` = (chpdeg1, chpdeg2) channels per degree
    /// instead of a distance and pixel widths.
    #[staticmethod]
    #[pyo3(signature = (dir1, dir2, cch, nch, chpdeg, **options))]
    fn from_chpdeg(
        dir1: &str,
        dir2: &str,
        cch: [f64; 2],
        nch: [usize; 2],
        chpdeg: [f64; 2],
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (d1, d2) = (xu_direction(dir1)?, xu_direction(dir2)?);
        let area = xu::Area::from_chpdeg(d1, d2, cch, nch, chpdeg).map_err(err)?;
        Self::with(area, options)
    }

    /// The detector for xu's `r_i` = `beam`, in the frame the direction
    /// strings are given in.
    fn detector(&self, beam: [f64; 3]) -> PyResult<PyPixelDetector> {
        self.0
            .detector(xu_beam(beam)?)
            .map(PyPixelDetector)
            .map_err(err)
    }
}

/// xrayutilities `init_linear` as a detector of one row, lengths in metres.
/// Keyword options: `tilt` (degrees), `nav` and `roi` = (start, end),
/// half-open. nav must divide the roi length, and the roi must lie on the
/// row.
#[pyclass(name = "Linear", module = "hklrs.xu", frozen)]
struct PyXuLinear(xu::Linear);

impl PyXuLinear {
    fn with(mut linear: xu::Linear, options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        xu_options(options, |key, value| {
            match key {
                "tilt" => linear.tilt = hklrs::Deg(value.extract()?),
                "nav" => linear.nav = value.extract()?,
                "roi" => linear.roi = value.extract()?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(Self(linear))
    }
}

#[pymethods]
impl PyXuLinear {
    #[new]
    #[pyo3(signature = (dir, cch, nch, distance, pwidth, **options))]
    fn new(
        dir: &str,
        cch: f64,
        nch: usize,
        distance: f64,
        pwidth: f64,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let linear = xu::Linear::new(xu_direction(dir)?, cch, nch, distance, pwidth);
        Self::with(linear, options)
    }

    /// `init_linear` with `chpdeg` channels per degree instead of a distance
    /// and pixel width.
    #[staticmethod]
    #[pyo3(signature = (dir, cch, nch, chpdeg, **options))]
    fn from_chpdeg(
        dir: &str,
        cch: f64,
        nch: usize,
        chpdeg: f64,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let linear = xu::Linear::from_chpdeg(xu_direction(dir)?, cch, nch, chpdeg).map_err(err)?;
        Self::with(linear, options)
    }

    /// The detector for xu's `r_i` = `beam`, in the frame the direction
    /// string is given in.
    fn detector(&self, beam: [f64; 3]) -> PyResult<PyPixelDetector> {
        self.0
            .detector(xu_beam(beam)?)
            .map(PyPixelDetector)
            .map_err(err)
    }
}

#[pyfunction]
fn presets() -> Vec<&'static str> {
    hklrs::presets::PRESETS.to_vec()
}

#[pymodule]
#[pyo3(name = "hklrs")]
fn hklrs_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLattice>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyReflection>()?;
    m.add_class::<PySample>()?;
    m.add_class::<PyMode>()?;
    m.add_class::<PyPixelDetector>()?;
    let xu = PyModule::new(m.py(), "xu")?;
    xu.add_class::<PyXuArea>()?;
    xu.add_class::<PyXuLinear>()?;
    m.add_submodule(&xu)?;
    // lets `import hklrs.xu` find the submodule of an extension module
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("hklrs.xu", &xu)?;
    m.add("InverseError", m.py().get_type::<InverseError>())?;
    m.add("Unreachable", m.py().get_type::<Unreachable>())?;
    m.add("NoSolution", m.py().get_type::<NoSolution>())?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(pick, m)?)?;
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    Ok(())
}
