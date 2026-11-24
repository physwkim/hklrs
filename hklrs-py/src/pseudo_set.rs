//! Inverses of hkl's pseudo-axis engines: `hklrs.PseudoMode` and its
//! `pseudo_solve` / `pseudo_pick`, the writing side of `hklrs.pseudo`.
//! Angles are degrees here, radians in the core; q is Å⁻¹.

use hklrs::inverse::{pseudo, soleil_sixs_med};
use hklrs::nalgebra::{Matrix3, Vector3};
use hklrs::{ModeTable, PsiEngine};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{PyGeometry, PySample, err, inverse_error, parse_picker};

/// An engine and its pseudo values, rad and Å⁻¹.
#[derive(Clone, Copy)]
enum Engine {
    Q {
        q: f64,
    },
    Q2 {
        q: f64,
        alpha: f64,
    },
    Tth2 {
        tth: f64,
        alpha: f64,
    },
    QperQpar {
        surface: Vector3<f64>,
        qper: f64,
        qpar: f64,
    },
    Eulerians([f64; 3]),
    Psi {
        reference: Vector3<f64>,
        psi: f64,
    },
}

impl Engine {
    fn name(&self) -> &'static str {
        match self {
            Engine::Q { .. } => "q",
            Engine::Q2 { .. } => "q2",
            Engine::Tth2 { .. } => "tth2",
            Engine::QperQpar { .. } => "qper_qpar",
            Engine::Eulerians(_) => "eulerians",
            Engine::Psi { .. } => "psi",
        }
    }
}

/// hkl's psi engine mode on the preset `geometry` is named after.
fn psi_mode(geometry: &str) -> PyResult<PsiEngine> {
    let table = ModeTable::of(geometry).ok();
    table
        .and_then(ModeTable::psi)
        .ok_or_else(|| PyValueError::new_err(format!("hkl has no psi engine on {geometry:?}")))
}

/// One of hkl's pseudo-axis engines set to its pseudo values, as an inverse
/// mode validated against the geometry it was built for.
#[pyclass(name = "PseudoMode", frozen)]
struct PyPseudoMode {
    /// Shared, not copied: limits set on it after the mode is made apply.
    geometry: Py<PyGeometry>,
    engine: Engine,
    slits: bool,
}

impl PyPseudoMode {
    /// The one place an engine meets its mode, at construction and at every
    /// solve alike.
    fn build<'g>(&self, py: Python<'_>, g: &'g hklrs::Geometry) -> PyResult<hklrs::Mode<'g>> {
        let mode = match self.engine {
            Engine::Q { q } => pseudo::q(g, q),
            Engine::Q2 { q, alpha } => pseudo::q2(g, q, alpha),
            Engine::Tth2 { tth, alpha } => pseudo::tth2(g, tth, alpha),
            Engine::QperQpar {
                surface,
                qper,
                qpar,
            } => pseudo::qper_qpar(g, surface, qper, qpar),
            Engine::Eulerians(_) => pseudo::eulerians(g),
            Engine::Psi { reference, psi } => psi_mode(g.name())?(g, reference, psi),
        };
        let mode = if self.slits {
            mode.and_then(soleil_sixs_med::slits)
        } else {
            mode
        };
        mode.map_err(|e| inverse_error::to_py(py, g, e))
    }

    /// UB and target of a request; for psi, hkl0 and the sample's UB.
    fn target(
        &self,
        sample: Option<&PySample>,
        hkl0: Option<[f64; 3]>,
    ) -> PyResult<(Matrix3<f64>, hklrs::Target)> {
        match self.engine {
            Engine::Psi { .. } => {
                let (sample, hkl0) = sample
                    .zip(hkl0)
                    .ok_or_else(|| PyValueError::new_err("engine \"psi\" needs sample and hkl0"))?;
                Ok((sample.0.ub(), hklrs::Target::Hkl(Vector3::from(hkl0))))
            }
            // No equation of the other engines reads UB.
            Engine::Eulerians(e) => Ok((Matrix3::identity(), hklrs::Target::Eulerians(e))),
            _ => Ok((Matrix3::identity(), hklrs::Target::Detector)),
        }
    }

    /// `f` on the request these arguments make, with the geometry it solves
    /// on; `current` in degrees / metres.
    fn with_request<T>(
        &self,
        py: Python<'_>,
        wavelength: f64,
        current: &[f64],
        sample: Option<&PySample>,
        hkl0: Option<[f64; 3]>,
        f: impl FnOnce(&hklrs::Geometry, hklrs::InverseRequest<'_>) -> PyResult<T>,
    ) -> PyResult<T> {
        let geometry = self.geometry.try_borrow(py)?;
        let geo = &geometry.0;
        let mode = self.build(py, geo)?;
        let (ub, target) = self.target(sample, hkl0)?;
        let current = geo
            .values_from_degrees(current)
            .map_err(|e| inverse_error::to_py(py, geo, e.into()))?;
        let request = hklrs::InverseRequest {
            mode: &mode,
            ub,
            wavelength: hklrs::Wavelength::new(wavelength).map_err(err)?,
            target,
            current: &current,
        };
        f(geo, request)
    }
}

#[pymethods]
impl PyPseudoMode {
    /// `engine` at `values`, as `hklrs.pseudo` reads them: "q" (q,), "q2"
    /// (q, alpha), "tth2" (tth, alpha), "qper_qpar" (qper, qpar) about the
    /// sample-holder normal `surface` (default y), "eulerians" (omega, chi,
    /// phi) and "psi" (psi,) of the Miller `reference`, on the presets with
    /// hkl's psi engine. Angles in degrees, q in Å⁻¹. `slits` turns
    /// eta_a after the solve as SOLEIL SIXS MED2+3 does. Axes the mode does
    /// not write hold at the `current` of a solve.
    #[new]
    #[pyo3(signature = (geometry, engine, values, surface=None, reference=None, slits=false))]
    fn new(
        py: Python<'_>,
        geometry: Bound<'_, PyGeometry>,
        engine: &str,
        values: Vec<f64>,
        surface: Option<[f64; 3]>,
        reference: Option<[f64; 3]>,
        slits: bool,
    ) -> PyResult<Self> {
        let takes = |names: &[&str]| {
            if values.len() == names.len() {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "engine {engine:?} takes {names:?}, got {} values",
                    values.len()
                )))
            }
        };
        let rad = |i: usize| values[i].to_radians();
        let engine = match engine {
            "q" => {
                takes(&["q"])?;
                Engine::Q { q: values[0] }
            }
            "q2" => {
                takes(&["q", "alpha"])?;
                Engine::Q2 {
                    q: values[0],
                    alpha: rad(1),
                }
            }
            "tth2" => {
                takes(&["tth", "alpha"])?;
                Engine::Tth2 {
                    tth: rad(0),
                    alpha: rad(1),
                }
            }
            "qper_qpar" => {
                takes(&["qper", "qpar"])?;
                Engine::QperQpar {
                    surface: Vector3::from(surface.unwrap_or([0.0, 1.0, 0.0])),
                    qper: values[0],
                    qpar: values[1],
                }
            }
            "eulerians" => {
                takes(&["omega", "chi", "phi"])?;
                Engine::Eulerians([rad(0), rad(1), rad(2)])
            }
            "psi" => {
                takes(&["psi"])?;
                let reference = reference
                    .ok_or_else(|| PyValueError::new_err("engine \"psi\" needs reference"))?;
                Engine::Psi {
                    reference: Vector3::from(reference),
                    psi: rad(0),
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown pseudo engine {other:?}"
                )));
            }
        };
        let mode = Self {
            geometry: geometry.clone().unbind(),
            engine,
            slits,
        };
        mode.build(py, &geometry.try_borrow()?.0)?;
        Ok(mode)
    }

    #[getter]
    fn engine(&self) -> &'static str {
        self.engine.name()
    }

    /// The axes the engine writes, in the mode's order; the slits axis is
    /// not one.
    fn written(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let geometry = self.geometry.try_borrow(py)?;
        let g = &geometry.0;
        let mode = self.build(py, g)?;
        Ok(mode
            .written()
            .iter()
            .map(|&i| g.axes()[i].name().to_owned())
            .collect())
    }
}

/// Every solution of `mode` within the axis limits, nearest to `current`
/// first; psi solves with Q at `hkl0` of `sample`. `current` and the
/// results are all axes in geometry order, degrees / metres; written
/// rotations are wrapped to their cut points and held axes are returned as
/// given, to the last bit of the degree to radian round trip. Raises `NoSolution` when the mode has none, `NoSolutionInLimits`
/// when every one is outside the limits, and the `InverseError` subclass
/// named after any other failure.
#[pyfunction]
#[pyo3(signature = (mode, wavelength, current, sample=None, hkl0=None))]
fn pseudo_solve(
    py: Python<'_>,
    mode: &PyPseudoMode,
    wavelength: f64,
    current: Vec<f64>,
    sample: Option<PyRef<'_, PySample>>,
    hkl0: Option<[f64; 3]>,
) -> PyResult<Vec<Vec<f64>>> {
    mode.with_request(
        py,
        wavelength,
        &current,
        sample.as_deref(),
        hkl0,
        |geo, r| {
            r.solve()
                .map_err(|e| inverse_error::to_py(py, geo, e))?
                .iter()
                .map(|s| geo.values_to_degrees(s).map_err(err))
                .collect()
        },
    )
}

/// One solution: `picker` "closest" (nearest to `current`) or "first" (the
/// first the search reached). Arguments and errors as for `pseudo_solve`.
#[pyfunction]
#[pyo3(signature = (mode, wavelength, current, sample=None, hkl0=None, picker="closest"))]
fn pseudo_pick(
    py: Python<'_>,
    mode: &PyPseudoMode,
    wavelength: f64,
    current: Vec<f64>,
    sample: Option<PyRef<'_, PySample>>,
    hkl0: Option<[f64; 3]>,
    picker: &str,
) -> PyResult<Vec<f64>> {
    let picker = parse_picker(picker)?;
    mode.with_request(
        py,
        wavelength,
        &current,
        sample.as_deref(),
        hkl0,
        |geo, r| {
            let s = r
                .pick(picker)
                .map_err(|e| inverse_error::to_py(py, geo, e))?;
            geo.values_to_degrees(&s).map_err(err)
        },
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPseudoMode>()?;
    m.add_function(wrap_pyfunction!(pseudo_solve, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_pick, m)?)
}
