//! bsrs device over an hklrs diffractometer.
//!
//! Values cross this boundary in degrees for rotations, metres for
//! translations and Å for lattice constants and the wavelength.

mod diffractometer;
mod error;
mod plans;
mod pseudo;
mod rsm;
#[cfg(feature = "hdf5")]
mod rsm_hdf5;
mod samples;

pub use diffractometer::{Diffractometer, RealAxis, WavelengthSource};
pub use error::DiffError;
pub use plans::{hkl_list_scan, hkl_scan, mv_hkl, psi_scan, zone_scan};
pub use pseudo::PseudoAxis;
pub use rsm::{
    FailedFrame, FrameFailure, FrameReader, RsmCallback, RsmConfig, RsmError, RsmMap, RsmMonitor,
    RsmValue,
};
#[cfg(feature = "hdf5")]
pub use rsm_hdf5::{Hdf5FrameError, Hdf5FrameReader};
pub use samples::Measured;
