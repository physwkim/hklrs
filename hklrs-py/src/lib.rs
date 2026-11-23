//! Python bindings: conversions only. Angles are degrees at this boundary,
//! translations metres, wavelength Å.

use hklrs::nalgebra::{Matrix3, Point3, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: hklrs::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
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
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    Ok(())
}
