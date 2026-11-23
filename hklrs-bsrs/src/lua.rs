//! The Diffractometer's methods for Lua, on a type of their own: Lua names
//! a method by its Rust name, and the device's typed methods keep theirs.

use std::path::Path;
use std::sync::Arc;

use bsrs::lua_methods;
use futures::try_join;
use hklrs::nalgebra::Matrix3;
use hklrs::{Lattice, ModeSpec};
use serde::{Deserialize, Serialize};

use crate::config::{Config, ModeConfig, Positions, PsiConfig, named, ordered};
use crate::diffractometer::rows;
use crate::error::Fault;
use crate::{DiffError, Diffractometer, Measured};

/// The Lua methods of a [`Diffractometer`], published on its Lua global
/// when registered under the device's own name.
///
/// Each argument has one accepted shape (PY2-5): `hkl` is `{h, k, l}`,
/// `positions` is a table by geometry axis name in degrees and metres
/// (every axis, no other key), a matrix is a table of three rows.
/// Wavelength, sample and mode are the device's current ones.
///
/// - `add_reflection(name, hkl, positions)`: records a reflection of the
///   current sample.
/// - `compute_ub(primary, secondary)`: orients the sample from two
///   reflections, returns UB.
/// - `set_lattice(a, b, c, alpha, beta, gamma)`: Å and degrees.
/// - `set_ub(ub)`
/// - `set_mode(name, params)`: `params` is `{}`, `{hkl2 = hkl}` or
///   `{psi = {reference = hkl, angle = degrees}}`.
/// - `forward(positions)`: the hkl at `positions`.
/// - `inverse(hkl, positions)`: the positions reaching `hkl`, closest to
///   `positions`.
/// - `ca(hkl)`: the positions a move to `hkl` would go to, without moving.
/// - `wh()`: `{hkl, psi, positions, wavelength}` at the readbacks, `psi`
///   in degrees on a geometry with a psi mode and nil where the reading's
///   ψ is NaN.
/// - `save_config(path)`: writes [`Diffractometer::config`] as hklrs JSON
///   (`.json`) or YAML (`.yml`, `.yaml`).
/// - `load_config(path)`: [`Diffractometer::load_config`] from hklrs JSON or
///   YAML, or hklpy2 schema-3 YAML (a `.yml`/`.yaml` with its `_header`).
///
/// ```
/// use std::sync::Arc;
///
/// use bsrs::backends::soft::SoftMotor;
/// use bsrs::engine::RunEngine;
/// use bsrs::host::manager_lua::ManagerLuaState;
/// use bsrs::qs::{LuaEvaluator, Registry};
/// use hklrs::nalgebra::UnitQuaternion;
/// use hklrs::presets::preset;
/// use hklrs::{Lattice, Orientation, Sample, Wavelength};
/// use hklrs_bsrs::{Diffractometer, LuaDiffractometer, RealAxis, WavelengthSource};
/// use tokio::sync::Mutex;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let reals = ["omega", "chi", "phi", "tth"]
///     .map(|a| (a, Arc::new(SoftMotor::new(a, Some(0.0))) as Arc<dyn RealAxis>));
/// let u = UnitQuaternion::identity();
/// let sample = Sample::new("cubic", Lattice::cubic(1.54).unwrap(), Orientation::Manual { u });
/// let wavelength = WavelengthSource::Fixed(Wavelength::new(1.54).unwrap());
/// let dx = Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals.into(), wavelength, sample);
/// let dx = Arc::new(dx.unwrap());
///
/// let mut registry = Registry::new();
/// registry.register_readable("e4cv", dx.clone());
/// registry.register_lua_methods("e4cv", Arc::new(LuaDiffractometer::new(dx)));
/// let engine = Some(Arc::new(RunEngine::new(vec![])));
/// let lua = ManagerLuaState::new(Arc::new(Mutex::new(engine)), Arc::new(registry));
///
/// let r = lua.eval("return e4cv:ca({0, 0, 1}).tth").await;
/// assert_eq!(r.error, None);
/// let tth: f64 = r.return_value.unwrap().parse().unwrap();
/// assert!((tth - 60.0).abs() < 1e-9);
/// # }
/// ```
pub struct LuaDiffractometer {
    dx: Arc<Diffractometer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModeParams {
    hkl2: Option<[f64; 3]>,
    psi: Option<PsiConfig>,
}

#[derive(Serialize)]
struct Where {
    hkl: [f64; 3],
    /// The reading's ψ: NaN goes to nil as the reading's goes to null.
    psi: Option<f64>,
    positions: Positions,
    /// Å.
    wavelength: f64,
}

impl LuaDiffractometer {
    pub fn new(dx: Arc<Diffractometer>) -> Self {
        Self { dx }
    }

    fn ordered(&self, positions: &Positions) -> Result<Vec<f64>, DiffError> {
        ordered(&self.dx.geometry(), positions)
    }

    fn named(&self, values: &[f64]) -> Positions {
        named(&self.dx.geometry(), values)
    }
}

#[lua_methods]
impl LuaDiffractometer {
    #[lua_method]
    async fn add_reflection(
        &self,
        name: String,
        hkl: [f64; 3],
        positions: Positions,
    ) -> Result<(), Fault> {
        let measured = Measured {
            hkl,
            positions: self.ordered(&positions)?,
            wavelength: self.dx.wavelength().await?,
        };
        Ok(self.dx.add_reflection(&name, measured)?)
    }

    #[lua_method]
    fn compute_ub(&self, primary: String, secondary: String) -> Result<[[f64; 3]; 3], DiffError> {
        Ok(rows(&self.dx.compute_ub(&primary, &secondary)?))
    }

    #[lua_method]
    fn set_lattice(
        &self,
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<(), DiffError> {
        let lattice = Lattice::from_degrees(a, b, c, alpha, beta, gamma)?;
        self.dx.set_lattice(lattice);
        Ok(())
    }

    #[lua_method]
    fn set_ub(&self, ub: [[f64; 3]; 3]) -> Result<(), DiffError> {
        self.dx.set_ub(&Matrix3::from_fn(|i, j| ub[i][j]))
    }

    #[lua_method]
    fn set_mode(&self, name: String, params: ModeParams) -> Result<(), DiffError> {
        let mode = ModeConfig {
            name,
            hkl2: params.hkl2,
            psi: params.psi,
        };
        self.dx.set_mode(ModeSpec::from(&mode))
    }

    #[lua_method]
    async fn forward(&self, positions: Positions) -> Result<[f64; 3], Fault> {
        let positions = self.ordered(&positions)?;
        let wavelength = self.dx.wavelength().await?;
        Ok(self.dx.hkl(&positions, wavelength)?.into())
    }

    #[lua_method]
    async fn inverse(&self, hkl: [f64; 3], positions: Positions) -> Result<Positions, Fault> {
        let positions = self.ordered(&positions)?;
        let wavelength = self.dx.wavelength().await?;
        Ok(self.named(&self.dx.inverse(hkl, &positions, wavelength)?))
    }

    #[lua_method]
    async fn ca(&self, hkl: [f64; 3]) -> Result<Positions, Fault> {
        let (current, wavelength) = try_join!(self.dx.positions(), self.dx.wavelength())?;
        Ok(self.named(&self.dx.inverse(hkl, &current, wavelength)?))
    }

    #[lua_method]
    async fn wh(&self) -> Result<Where, Fault> {
        let (current, wavelength) = try_join!(self.dx.positions(), self.dx.wavelength())?;
        let (hkl, psi) = self.dx.pseudos(&current, wavelength)?;
        Ok(Where {
            hkl: hkl.into(),
            psi,
            positions: self.named(&current),
            wavelength: wavelength.angstrom(),
        })
    }

    #[lua_method]
    fn save_config(&self, path: String) -> Result<(), DiffError> {
        self.dx.config().write(Path::new(&path))
    }

    #[lua_method]
    fn load_config(&self, path: String) -> Result<(), DiffError> {
        self.dx.load_config(&Config::read(Path::new(&path))?)
    }
}
