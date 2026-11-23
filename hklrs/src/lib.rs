//! Diffractometer geometry and reciprocal-space conversion.
//!
//! Units are fixed: radians, Å for lattice and wavelength, metres for
//! detector geometry and translations, keV for energy. Q includes 2π.

mod detector;
mod error;
mod forward;
mod geometry;
mod lattice;
pub mod presets;
mod sample;
mod units;

pub use detector::PixelDetector;
pub use error::Error;
pub use forward::{Forward, angle_between, forward};
pub use geometry::{Axis, AxisKind, Geometry, GeometryBuilder, Poses};
pub use lattice::Lattice;
pub use sample::{Orientation, Sample};
pub use units::{Deg, Energy, HC_KEV_ANGSTROM, Rad, Wavelength};

pub use nalgebra;
