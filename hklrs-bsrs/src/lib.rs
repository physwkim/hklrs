//! bsrs device over an hklrs diffractometer.
//!
//! Values cross this boundary in degrees for rotations, metres for
//! translations and Å for lattice constants and the wavelength.

mod config;
mod diffractometer;
mod error;
mod hklpy2;
mod lua;
mod plans;
mod pseudo;
mod rsm;
#[cfg(feature = "hdf5")]
mod rsm_hdf5;
mod samples;

pub use config::{
    AxisConfig, Config, GeometryConfig, LatticeConfig, ModeConfig, ReflectionConfig,
    SCHEMA_VERSION, SampleConfig,
};
pub use diffractometer::{Diffractometer, RealAxis, WavelengthSource};
pub use error::DiffError;
pub use lua::LuaDiffractometer;
pub use plans::{hkl_list_scan, hkl_scan, mv_hkl, psi_scan, zone_scan};
pub use pseudo::PseudoAxis;
pub use rsm::{
    FailedFrame, FrameFailure, FrameReader, ReadError, RsmCallback, RsmConfig, RsmError, RsmMap,
    RsmMonitor, RsmValue,
};
#[cfg(feature = "hdf5")]
pub use rsm_hdf5::{Hdf5FrameError, Hdf5FrameReader};
pub use samples::Measured;
