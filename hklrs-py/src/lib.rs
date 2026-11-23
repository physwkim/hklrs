//! Python bindings: conversions only. Angles are degrees at this boundary,
//! translations metres, wavelength Å.

mod normalise;

use hklrs::absorption::{Absorber, Beam, DetectorPath, Face, Layer, Masked, Slab};
use hklrs::nalgebra::{Matrix3, Point3, Unit, Vector3};
use hklrs::{poni, xu};
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
create_exception!(
    hklrs,
    NoSolutionInLimits,
    NoSolution,
    "Every solution takes an axis outside its limits; carries them as \
     solutions (degrees / metres, nearest to current first)."
);

/// `geometry` is the one the failing call solved on, for the units of
/// `NoSolutionInLimits.solutions`.
fn inverse_err(py: Python<'_>, geometry: &hklrs::Geometry, e: hklrs::InverseError) -> PyErr {
    use hklrs::InverseError as E;
    let msg = e.to_string();
    match e {
        E::NoSolutionInLimits(outside) => {
            let solutions = match outside
                .iter()
                .map(|s| geometry.values_to_degrees(s))
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(s) => s,
                Err(e) => return err(e),
            };
            let err = NoSolutionInLimits::new_err(msg);
            let _ = err.value(py).setattr("solutions", solutions);
            err
        }
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

#[pyclass(name = "Geometry")]
struct PyGeometry(hklrs::Geometry);

impl PyGeometry {
    fn axis(&self, name: &str) -> PyResult<&hklrs::Axis> {
        let i = self.0.axis_index(name).map_err(err)?;
        Ok(&self.0.axes()[i])
    }
}

/// `value` of `axis` in degrees for a rotation, as given otherwise.
fn user_value(axis: &hklrs::Axis, value: f64) -> f64 {
    if axis.is_rotation() {
        degrees(hklrs::Rad(value))
    } else {
        value
    }
}

/// Inverse of `user_value`.
fn internal_value(axis: &hklrs::Axis, value: f64) -> f64 {
    if axis.is_rotation() {
        radians(value).0
    } else {
        value
    }
}

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

    /// Limits axis `name` to [min, max], inclusive, degrees for a rotation
    /// and metres otherwise; an infinite end is unbounded. The inverse keeps
    /// the solutions whose axes are all within their limits: a written
    /// rotation after wrapping to its cut point, a held axis at its value in
    /// `current`, never wrapped. A mode solves on its geometry as it is when
    /// solving.
    fn set_limits(&mut self, name: &str, min: f64, max: f64) -> PyResult<()> {
        let axis = self.axis(name)?;
        let (lo, hi) = (internal_value(axis, min), internal_value(axis, max));
        self.0.set_limits(name, lo, hi).map_err(|e| match e {
            // in the caller's units, not radians
            hklrs::Error::AxisLimits { name, .. } => {
                err(hklrs::Error::AxisLimits { name, min, max })
            }
            e => err(e),
        })
    }

    /// (min, max) of axis `name`, units as for `set_limits`; unbounded,
    /// (-inf, inf), unless set. A rotation's limits are kept in radians, so
    /// the degrees read back can differ from those set in the last bit.
    fn limits(&self, name: &str) -> PyResult<(f64, f64)> {
        let axis = self.axis(name)?;
        let (lo, hi) = axis.limits();
        Ok((user_value(axis, lo), user_value(axis, hi)))
    }

    /// Represents rotation `name` in [cut, cut + 360) degrees: the inverse
    /// returns each written rotation there, -180 unless set.
    fn set_cut_point(&mut self, name: &str, cut: f64) -> PyResult<()> {
        let internal = internal_value(self.axis(name)?, cut);
        self.0.set_cut_point(name, internal).map_err(|e| match e {
            // in the caller's degrees, not radians
            hklrs::Error::CutPoint { name, .. } => err(hklrs::Error::CutPoint { name, cut }),
            e => err(e),
        })
    }

    /// The cut point of rotation `name`, degrees; None for any other axis.
    /// Kept in radians, so it can differ from the one set in the last bit.
    fn cut_point(&self, name: &str) -> PyResult<Option<f64>> {
        let axis = self.axis(name)?;
        Ok(axis.cut_point().map(|c| user_value(axis, c)))
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
    let p = point.map_or_else(
        || Point3::from(geo.beam_direction().into_inner()),
        Point3::from,
    );
    Ok((
        geo.values_from_degrees(axes).map_err(err)?,
        hklrs::Wavelength::new(wavelength).map_err(err)?,
        p,
    ))
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
    /// Shared, not copied: limits set on it after the mode is made apply.
    geometry: Py<PyGeometry>,
    name: String,
    hkl2: Option<Vector3<f64>>,
    reference: Option<Vector3<f64>>,
    /// rad
    psi: Option<f64>,
}

impl PyMode {
    /// The one place a name meets its mode: unknown names and missing
    /// parameters fail here, at construction and at every solve alike.
    fn build<'g>(&self, py: Python<'_>, g: &'g hklrs::Geometry) -> PyResult<hklrs::Mode<'g>> {
        use hklrs::inverse::{e4c, e6c};
        let name = self.name.as_str();
        let missing = |what| PyValueError::new_err(format!("mode {name:?} needs {what}"));
        let hkl2 = || self.hkl2.ok_or_else(|| missing("hkl2"));
        let psi = || {
            self.reference
                .zip(self.psi)
                .ok_or_else(|| missing("reference and psi"))
        };
        let mode = match name {
            "bissector" => e4c::bissector(g),
            "constant_omega" => e4c::constant_omega(g),
            "constant_chi" => e4c::constant_chi(g),
            "constant_phi" => e4c::constant_phi(g),
            "double_diffraction" => e4c::double_diffraction(g, hkl2()?),
            "psi_constant" => {
                let (reference, psi) = psi()?;
                e4c::psi_constant(g, reference, psi)
            }
            "bissector_vertical" => e6c::bissector_vertical(g),
            "constant_omega_vertical" => e6c::constant_omega_vertical(g),
            "constant_chi_vertical" => e6c::constant_chi_vertical(g),
            "constant_phi_vertical" => e6c::constant_phi_vertical(g),
            "lifting_detector_phi" => e6c::lifting_detector_phi(g),
            "lifting_detector_omega" => e6c::lifting_detector_omega(g),
            "lifting_detector_mu" => e6c::lifting_detector_mu(g),
            "double_diffraction_vertical" => e6c::double_diffraction_vertical(g, hkl2()?),
            "bissector_horizontal" => e6c::bissector_horizontal(g),
            "double_diffraction_horizontal" => e6c::double_diffraction_horizontal(g, hkl2()?),
            "psi_constant_vertical" => {
                let (reference, psi) = psi()?;
                e6c::psi_constant_vertical(g, reference, psi)
            }
            "psi_constant_horizontal" => {
                let (reference, psi) = psi()?;
                e6c::psi_constant_horizontal(g, reference, psi)
            }
            "constant_mu_horizontal" => e6c::constant_mu_horizontal(g),
            _ => return Err(PyValueError::new_err(format!("unknown mode {name:?}"))),
        };
        mode.map_err(|e| inverse_err(py, g, e))
    }
}

#[pymethods]
impl PyMode {
    /// hkl's "hkl" engine mode by name, for a geometry with that engine's
    /// axes. E4C: bissector, constant_omega, constant_chi, constant_phi,
    /// double_diffraction, psi_constant. E6C: bissector_vertical,
    /// constant_omega_vertical, constant_chi_vertical, constant_phi_vertical,
    /// lifting_detector_phi, lifting_detector_omega, lifting_detector_mu,
    /// double_diffraction_vertical, bissector_horizontal,
    /// double_diffraction_horizontal, psi_constant_vertical,
    /// psi_constant_horizontal, constant_mu_horizontal. double_diffraction*
    /// needs `hkl2`, psi_constant* `reference` and `psi` (degrees). Axes a
    /// mode does not write hold at the `current` passed to `solve`.
    #[new]
    #[pyo3(signature = (geometry, name, hkl2=None, reference=None, psi=None))]
    fn new(
        py: Python<'_>,
        geometry: Bound<'_, PyGeometry>,
        name: &str,
        hkl2: Option<[f64; 3]>,
        reference: Option<[f64; 3]>,
        psi: Option<f64>,
    ) -> PyResult<Self> {
        let mode = Self {
            geometry: geometry.clone().unbind(),
            name: name.to_owned(),
            hkl2: hkl2.map(Vector3::from),
            reference: reference.map(Vector3::from),
            psi: psi.map(f64::to_radians),
        };
        mode.build(py, &geometry.try_borrow()?.0)?;
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

/// Every solution for `hkl` within the axis limits, nearest to `current`
/// first. `current` and the results are all axes in geometry order, degrees
/// / metres; written rotations are wrapped to their cut points and held
/// axes are returned as given. Raises `Unreachable` when
/// |Q| > 4π/λ·(1 + 1e-12), `NoSolution` when the mode has none and its
/// subclass `NoSolutionInLimits` when every one is outside the limits.
#[pyfunction]
fn solve(
    py: Python<'_>,
    mode: &PyMode,
    sample: &PySample,
    hkl: [f64; 3],
    wavelength: f64,
    current: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let geometry = mode.geometry.try_borrow(py)?;
    let geo = &geometry.0;
    let ie = |e| inverse_err(py, geo, e);
    let m = mode.build(py, geo)?;
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
    let geometry = mode.geometry.try_borrow(py)?;
    let geo = &geometry.0;
    let ie = |e| inverse_err(py, geo, e);
    let m = mode.build(py, geo)?;
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
    .map_err(|e| inverse_err(py, &geometry.0, e))
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
/// detector table (`detector`), `sixs` and the xu and PONI importers;
/// opaque apart from its shape, mask and pixel positions.
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

fn unit(what: &str, v: [f64; 3]) -> PyResult<Unit<Vector3<f64>>> {
    let v = Vector3::from(v);
    v.iter()
        .all(|x| x.is_finite())
        .then(|| Unit::try_new(v, 0.0))
        .flatten()
        .ok_or_else(|| {
            err(hklrs::Error::ZeroVector {
                what: what.to_owned(),
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
            .detector(unit("xu beam", beam)?)
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
            .detector(unit("xu beam", beam)?)
            .map(PyPixelDetector)
            .map_err(err)
    }
}

fn poni_err(e: poni::PoniError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn orientation(n: u32) -> PyResult<poni::Orientation> {
    poni::Orientation::try_from(n).map_err(poni_err)
}

/// pyFAI's axes in the lab: axis 3 along `beam`, axis 1 along the part of
/// `up` normal to it.
fn lab_axes(beam: [f64; 3], up: [f64; 3]) -> PyResult<poni::LabAxes> {
    poni::LabAxes::new(unit("beam", beam)?, unit("up", up)?).map_err(poni_err)
}

fn degrees(r: hklrs::Rad) -> f64 {
    hklrs::Deg::from(r).0
}

fn radians(d: f64) -> hklrs::Rad {
    hklrs::Deg(d).into()
}

/// pyFAI's geometry of a flat detector, lengths in metres and rotations in
/// degrees (a `.poni` file's are radians): `Poni(dist, (poni1, poni2),
/// (rot1, rot2, rot3), (pixel1, pixel2), shape, orientation)`, `shape` being
/// (slow, fast) = pyFAI's (dim1, dim2) and `orientation` pyFAI's 1-4.
#[pyclass(name = "Poni", module = "hklrs.poni", frozen)]
struct PyPoni(poni::Poni);

#[pymethods]
impl PyPoni {
    #[new]
    fn new(
        dist: f64,
        poni: [f64; 2],
        rot: [f64; 3],
        pixel: [f64; 2],
        shape: (usize, usize),
        orientation: u32,
    ) -> PyResult<Self> {
        Ok(Self(poni::Poni {
            dist,
            poni1: poni[0],
            poni2: poni[1],
            rot1: radians(rot[0]),
            rot2: radians(rot[1]),
            rot3: radians(rot[2]),
            pixel1: pixel[0],
            pixel2: pixel[1],
            shape,
            orientation: self::orientation(orientation)?,
        }))
    }

    /// The PONI of `detector` read in `orientation`, lab axes as for
    /// `detector`, whose `detector` gives `detector` back. rot2 comes out in
    /// [-90, 90] and rot1, rot3 in [-180, 180].
    #[staticmethod]
    fn from_detector(
        detector: &PyPixelDetector,
        orientation: u32,
        beam: [f64; 3],
        up: [f64; 3],
    ) -> PyResult<Self> {
        let lab = lab_axes(beam, up)?;
        poni::Poni::from_detector(&detector.0, self::orientation(orientation)?, &lab)
            .map(Self)
            .map_err(poni_err)
    }

    /// The detector with pyFAI's axis 3 along `beam` and axis 1 along the
    /// part of `up` normal to it, positions relative to the sample.
    fn detector(&self, beam: [f64; 3], up: [f64; 3]) -> PyResult<PyPixelDetector> {
        self.0
            .detector(&lab_axes(beam, up)?)
            .map(PyPixelDetector)
            .map_err(poni_err)
    }

    #[getter]
    fn dist(&self) -> f64 {
        self.0.dist
    }

    #[getter]
    fn poni1(&self) -> f64 {
        self.0.poni1
    }

    #[getter]
    fn poni2(&self) -> f64 {
        self.0.poni2
    }

    #[getter]
    fn rot1(&self) -> f64 {
        degrees(self.0.rot1)
    }

    #[getter]
    fn rot2(&self) -> f64 {
        degrees(self.0.rot2)
    }

    #[getter]
    fn rot3(&self) -> f64 {
        degrees(self.0.rot3)
    }

    #[getter]
    fn pixel1(&self) -> f64 {
        self.0.pixel1
    }

    #[getter]
    fn pixel2(&self) -> f64 {
        self.0.pixel2
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.0.shape
    }

    #[getter]
    fn orientation(&self) -> u32 {
        self.0.orientation as u32
    }
}

/// A `.poni` file as pyFAI reads it, lengths in metres and rotations in
/// degrees. `detector` is pyFAI's detector name; `orientation` is None where
/// the file gives none, which is 3 for pyFAI's generic `Detector`.
#[pyclass(name = "PoniFile", module = "hklrs.poni", frozen)]
struct PyPoniFile(poni::PoniFile);

#[pymethods]
impl PyPoniFile {
    /// The PONI of pyFAI's generic flat detector. A named detector takes its
    /// pixels, shape and default orientation from pyFAI, so it is a
    /// ValueError here, as are a spline, parallax and binning.
    fn poni(&self) -> PyResult<PyPoni> {
        self.0.poni().map(PyPoni).map_err(poni_err)
    }

    /// `poni_version` as pyFAI raises it from the file's contents.
    #[getter]
    fn version(&self) -> f64 {
        self.0.version
    }

    #[getter]
    fn detector(&self) -> &str {
        &self.0.detector.name
    }

    #[getter]
    fn pixel1(&self) -> Option<f64> {
        self.0.detector.pixel1
    }

    #[getter]
    fn pixel2(&self) -> Option<f64> {
        self.0.detector.pixel2
    }

    #[getter]
    fn max_shape(&self) -> Option<(usize, usize)> {
        self.0.detector.max_shape
    }

    #[getter]
    fn orientation(&self) -> Option<u32> {
        self.0.detector.orientation.map(|o| o as u32)
    }

    #[getter]
    fn binning(&self) -> (u64, u64) {
        self.0.detector.binning
    }

    #[getter]
    fn splinefile(&self) -> Option<&str> {
        self.0.detector.splinefile.as_deref()
    }

    #[getter]
    fn parallax(&self) -> bool {
        self.0.parallax
    }

    #[getter]
    fn dist(&self) -> f64 {
        self.0.dist
    }

    #[getter]
    fn poni1(&self) -> f64 {
        self.0.poni1
    }

    #[getter]
    fn poni2(&self) -> f64 {
        self.0.poni2
    }

    #[getter]
    fn rot1(&self) -> f64 {
        degrees(self.0.rot1)
    }

    #[getter]
    fn rot2(&self) -> f64 {
        degrees(self.0.rot2)
    }

    #[getter]
    fn rot3(&self) -> f64 {
        degrees(self.0.rot3)
    }
}

/// The `.poni` text as pyFAI reads it, and beside it the file's wavelength
/// (Å), None if it gives none.
#[pyfunction]
fn parse(text: &str) -> PyResult<(PyPoniFile, Option<f64>)> {
    let file = poni::PoniFile::parse(text).map_err(poni_err)?;
    let wavelength = file.wavelength.map(hklrs::Wavelength::angstrom);
    Ok((PyPoniFile(file), wavelength))
}

type Planes<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

fn planes(py: Python<'_>, p: hklrs::Planes) -> PyResult<Planes<'_>> {
    let shape = p.shape();
    let plane = |v: Vec<f64>| PyArray1::from_vec(py, v).reshape(shape);
    Ok((plane(p.x)?, plane(p.y)?, plane(p.z)?))
}

/// Q in the sample frame (Å⁻¹, with 2π) of every pixel of `detector` at
/// `axes` (geometry axis order, degrees / metres): (qx, qy, qz), each a
/// (slow, fast) array.
#[pyfunction]
fn q_map<'py>(
    py: Python<'py>,
    geometry: &PyGeometry,
    axes: Vec<f64>,
    wavelength: f64,
    detector: &PyPixelDetector,
) -> PyResult<Planes<'py>> {
    let (geo, det) = (&geometry.0, &detector.0);
    let values = geo.values_from_degrees(&axes).map_err(err)?;
    let wl = hklrs::Wavelength::new(wavelength).map_err(err)?;
    let q = py
        .detach(|| hklrs::q_map(geo, &values, wl, det))
        .map_err(err)?;
    planes(py, q)
}

/// Pixel positions relative to the sample in the lab frame (metres) of
/// every pixel of `detector` at `axes` (geometry axis order, degrees /
/// metres): (x, y, z), each a (slow, fast) array.
#[pyfunction]
fn detector_positions<'py>(
    py: Python<'py>,
    geometry: &PyGeometry,
    axes: Vec<f64>,
    detector: &PyPixelDetector,
) -> PyResult<Planes<'py>> {
    let (geo, det) = (&geometry.0, &detector.0);
    let values = geo.values_from_degrees(&axes).map_err(err)?;
    let p = py
        .detach(|| hklrs::detector_positions(geo, &values, det))
        .map_err(err)?;
    planes(py, p)
}

/// The code a reasons array holds for `m`; 0 is a pixel with a value.
fn masked_code(m: Masked) -> u8 {
    match m {
        Masked::Unlit => 1,
        Masked::GrazingExit => 2,
        Masked::ThroughBulk => 3,
        Masked::Opaque => 4,
        Masked::EdgeOn => 5,
        Masked::Extinct => 6,
        Masked::Detector => 7,
        Masked::BadFlat => 8,
        Masked::BadCount => 9,
    }
}

/// Every reason, for the Python enum; a new variant fails to compile in
/// `masked_code` first.
const MASKED: [Masked; 9] = [
    Masked::Unlit,
    Masked::GrazingExit,
    Masked::ThroughBulk,
    Masked::Opaque,
    Masked::EdgeOn,
    Masked::Extinct,
    Masked::Detector,
    Masked::BadFlat,
    Masked::BadCount,
];

type WeightArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u8>>);

/// Row-major values and reasons: the weight and 0, or NaN and the reason's
/// code.
fn split(w: hklrs::Weights) -> ((usize, usize), Vec<f64>, Vec<u8>) {
    let (values, reasons) = w
        .data
        .iter()
        .map(|w| match w {
            Ok(v) => (v.get(), 0),
            Err(m) => (f64::NAN, masked_code(*m)),
        })
        .unzip();
    (w.shape(), values, reasons)
}

fn weight_arrays(
    py: Python<'_>,
    (shape, values, reasons): ((usize, usize), Vec<f64>, Vec<u8>),
) -> PyResult<WeightArrays<'_>> {
    Ok((
        PyArray1::from_vec(py, values).reshape(shape)?,
        PyArray1::from_vec(py, reasons).reshape(shape)?,
    ))
}

/// Solid angle of every pixel (sr) at `axes`, arguments as for
/// `detector_positions`: (values, reasons), each a (slow, fast) array. A
/// pixel with no value has NaN and its `Masked` reason, `EdgeOn` here; the
/// others have reason 0.
#[pyfunction]
fn solid_angle<'py>(
    py: Python<'py>,
    geometry: &PyGeometry,
    axes: Vec<f64>,
    detector: &PyPixelDetector,
) -> PyResult<WeightArrays<'py>> {
    let (geo, det) = (&geometry.0, &detector.0);
    let values = geo.values_from_degrees(&axes).map_err(err)?;
    let w = py
        .detach(|| hklrs::solid_angle(geo, &values, det).map(split))
        .map_err(err)?;
    weight_arrays(py, w)
}

/// Polarisation factor of every pixel at `axes`, arguments as for
/// `detector_positions`, for a beam whose electric field lies along `e`
/// (lab frame, projected normal to the beam) with degree `factor` in
/// [-1, 1]: 1 fully along `e`, 0 unpolarised. Values and reasons as for
/// `solid_angle`, `Extinct` where the factor vanishes.
#[pyfunction]
fn polarisation<'py>(
    py: Python<'py>,
    geometry: &PyGeometry,
    axes: Vec<f64>,
    detector: &PyPixelDetector,
    e: [f64; 3],
    factor: f64,
) -> PyResult<WeightArrays<'py>> {
    let (geo, det) = (&geometry.0, &detector.0);
    let values = geo.values_from_degrees(&axes).map_err(err)?;
    let pol = hklrs::Polarisation {
        e: Vector3::from(e),
        factor,
    };
    let w = py
        .detach(|| hklrs::polarisation(geo, &values, det, pol).map(split))
        .map_err(err)?;
    weight_arrays(py, w)
}

fn face(normal: [f64; 3], axis: [f64; 3], length: f64, width: f64) -> PyResult<Face> {
    Face::new(normal.into(), axis.into(), length, width).map_err(err)
}

/// A flat sample: its entrance face, `normal` pointing out of the sample,
/// `length` along the part of `axis` in the face plane and `width` across it
/// (sample frame, centred on the sample position), and `thickness` along the
/// normal; metres.
#[pyclass(name = "Slab", frozen)]
struct PySlab(Slab);

#[pymethods]
impl PySlab {
    #[new]
    fn new(
        normal: [f64; 3],
        axis: [f64; 3],
        length: f64,
        width: f64,
        thickness: f64,
    ) -> PyResult<Self> {
        Slab::new(face(normal, axis, length, width)?, thickness)
            .map(Self)
            .map_err(err)
    }

    /// A crystal too thick for the beam to leave through the back.
    #[staticmethod]
    fn bulk(normal: [f64; 3], axis: [f64; 3], length: f64, width: f64) -> PyResult<Self> {
        Ok(Self(Slab::bulk(face(normal, axis, length, width)?)))
    }
}

/// Beam profile across the incident beam with unit total flux, centred on
/// the sample position. Its first size lies along `axis` (lab frame)
/// projected across the beam, the second across both; metres.
#[pyclass(name = "Beam", frozen)]
struct PyBeam(Beam);

#[pymethods]
impl PyBeam {
    /// Uniform over a `width[0]` x `width[1]` rectangle.
    #[staticmethod]
    fn top_hat(axis: [f64; 3], width: [f64; 2]) -> PyResult<Self> {
        Beam::top_hat(axis.into(), width).map(Self).map_err(err)
    }

    /// Gaussian with full widths at half maximum `fwhm`.
    #[staticmethod]
    fn gaussian_fwhm(axis: [f64; 3], fwhm: [f64; 2]) -> PyResult<Self> {
        Beam::gaussian_fwhm(axis.into(), fwhm)
            .map(Self)
            .map_err(err)
    }
}

fn per_metre(mu: f64) -> PyResult<hklrs::PerMetre> {
    hklrs::PerMetre::new(mu).map_err(err)
}

fn layer(layer: Option<(f64, f64)>) -> PyResult<Option<Layer>> {
    layer
        .map(|(mu, thickness)| Layer::new(per_metre(mu)?, thickness).map_err(err))
        .transpose()
}

/// What absorbs: the sample `slab` of linear attenuation `mu` (1/m) lit by
/// `beam`, and on the way to each pixel `air` (1/m), a `window` transmitted
/// and a `sensor` absorbed, each (mu 1/m, thickness m along the detector
/// normal). An absent term weighs 1. Every mu is the same at all energies.
#[pyclass(name = "Absorber", frozen)]
struct PyAbsorber(Absorber<hklrs::PerMetre>);

#[pymethods]
impl PyAbsorber {
    #[new]
    #[pyo3(signature = (slab, beam, mu, air=None, window=None, sensor=None))]
    fn new(
        slab: &PySlab,
        beam: &PyBeam,
        mu: f64,
        air: Option<f64>,
        window: Option<(f64, f64)>,
        sensor: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        Ok(Self(Absorber {
            slab: slab.0,
            beam: beam.0,
            sample: per_metre(mu)?,
            detector: DetectorPath {
                air: air.map(per_metre).transpose()?,
                window: layer(window)?,
                sensor: layer(sensor)?,
            },
        }))
    }
}

/// Absorption weight of every pixel at `axes`, arguments as for `q_map`:
/// the sample's effective path (m) times the detector path's transmission,
/// by which the correction divides. Values and reasons as for
/// `solid_angle`; where the sample and the detector path both mask, the
/// pixel carries the sample's reason. A window or sensor needs a detector
/// with a pixel area.
#[pyfunction]
fn absorption<'py>(
    py: Python<'py>,
    geometry: &PyGeometry,
    axes: Vec<f64>,
    wavelength: f64,
    detector: &PyPixelDetector,
    absorber: &PyAbsorber,
) -> PyResult<WeightArrays<'py>> {
    let (geo, det) = (&geometry.0, &detector.0);
    let values = geo.values_from_degrees(&axes).map_err(err)?;
    let wl = hklrs::Wavelength::new(wavelength).map_err(err)?;
    let w = py
        .detach(|| hklrs::absorption(geo, &values, wl, det, &absorber.0).map(split))
        .map_err(err)?;
    weight_arrays(py, w)
}

#[pyfunction]
fn presets() -> Vec<&'static str> {
    hklrs::presets::PRESETS.to_vec()
}

/// Names of the binoculars detector table rows, in binoculars id order.
#[pyfunction]
fn detectors() -> Vec<&'static str> {
    hklrs::Detector::ALL.map(hklrs::Detector::name).to_vec()
}

/// The binoculars detector table row `name`, with its mask, in binoculars'
/// frame: pixel (slow, fast) is image (row, column), columns run along -y
/// and rows along +z or -z from a pixel corner at the origin, and x is 0
/// except on the Cirpad.
#[pyfunction]
fn detector(name: &str) -> PyResult<PyPixelDetector> {
    hklrs::Detector::ALL
        .into_iter()
        .find(|d| d.name() == name)
        .map(|d| PyPixelDetector(d.pixel_detector()))
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown detector {name:?}; one of {:?}",
                detectors()
            ))
        })
}

/// binoculars' SIXS calibration: `detector` moved so that its pixel
/// `centre` (slow, fast) lies at (`sdd`, 0, 0) metres, then turned `detrot`
/// degrees about x. The mask is kept. The central pixel lands on x = `sdd`
/// even on the Cirpad, whose pixels have an x.
#[pyfunction]
fn sixs(
    detector: &PyPixelDetector,
    centre: (usize, usize),
    sdd: f64,
    detrot: f64,
) -> PyResult<PyPixelDetector> {
    hklrs::sixs(&detector.0, centre, sdd, radians(detrot).0)
        .map(PyPixelDetector)
        .map_err(err)
}

fn submodule(
    m: &Bound<'_, PyModule>,
    name: &str,
    fill: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let sub = PyModule::new(m.py(), name)?;
    fill(&sub)?;
    m.add_submodule(&sub)?;
    // lets `import hklrs.<name>` find the submodule of an extension module
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(format!("hklrs.{name}"), &sub)
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
    m.add_class::<PySlab>()?;
    m.add_class::<PyBeam>()?;
    m.add_class::<PyAbsorber>()?;
    submodule(m, "xu", |xu| {
        xu.add_class::<PyXuArea>()?;
        xu.add_class::<PyXuLinear>()
    })?;
    submodule(m, "normalise", normalise::register)?;
    submodule(m, "poni", |p| {
        p.add_class::<PyPoni>()?;
        p.add_class::<PyPoniFile>()?;
        p.add_function(wrap_pyfunction!(parse, p)?)
    })?;
    m.add("InverseError", m.py().get_type::<InverseError>())?;
    m.add("Unreachable", m.py().get_type::<Unreachable>())?;
    m.add("NoSolution", m.py().get_type::<NoSolution>())?;
    m.add(
        "NoSolutionInLimits",
        m.py().get_type::<NoSolutionInLimits>(),
    )?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(pick, m)?)?;
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add_function(wrap_pyfunction!(detectors, m)?)?;
    m.add_function(wrap_pyfunction!(detector, m)?)?;
    m.add_function(wrap_pyfunction!(sixs, m)?)?;
    m.add_function(wrap_pyfunction!(q_map, m)?)?;
    m.add_function(wrap_pyfunction!(detector_positions, m)?)?;
    m.add_function(wrap_pyfunction!(solid_angle, m)?)?;
    m.add_function(wrap_pyfunction!(polarisation, m)?)?;
    m.add_function(wrap_pyfunction!(absorption, m)?)?;
    let members: Vec<(String, u8)> = MASKED
        .iter()
        .map(|&r| (format!("{r:?}"), masked_code(r)))
        .collect();
    let kwargs = PyDict::new(m.py());
    kwargs.set_item("module", "hklrs")?;
    let masked = m
        .py()
        .import("enum")?
        .getattr("IntEnum")?
        .call(("Masked", members), Some(&kwargs))?;
    masked.setattr(
        "__doc__",
        "Why a pixel has no weight: the codes of a reasons array, 0 being none.",
    )?;
    m.add("Masked", masked)?;
    Ok(())
}
