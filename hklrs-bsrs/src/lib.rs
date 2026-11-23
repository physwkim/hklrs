//! bsrs device over an hklrs diffractometer.
//!
//! Values cross this boundary in degrees for rotations, metres for
//! translations and Å for lattice constants and the wavelength.

mod diffractometer;
mod error;
mod pseudo;
mod samples;

pub use diffractometer::{Diffractometer, RealAxis, WavelengthSource};
pub use error::DiffError;
pub use pseudo::PseudoAxis;
pub use samples::Measured;
