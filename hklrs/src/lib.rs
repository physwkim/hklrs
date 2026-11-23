//! Diffractometer geometry and reciprocal-space conversion.
//!
//! Units are fixed: radians, Å for lattice and wavelength, metres for
//! detector geometry and translations, keV for energy. Q includes 2π.

mod detector;
mod error;
mod forward;
mod geometry;
pub mod inverse;
mod lattice;
pub mod presets;
pub mod pseudo;
mod qmap;
mod sample;
mod units;
pub mod xu;

pub use detector::PixelDetector;
pub use error::Error;
pub use forward::{Forward, Scattering, angle_between, forward, scattering, scattering_at};
pub use geometry::{Axis, AxisKind, Geometry, GeometryBuilder, Poses};
pub use inverse::{Equation, Eulerian, InverseError, InverseRequest, Mode, Picker, Slits, Target};
pub use lattice::Lattice;
pub use qmap::{
    Frame, Planes, Polarisation, PolarisationPlane, QMap, Weights, detector_positions,
    polarisation, q_map, solid_angle,
};
pub use sample::{Orientation, PARALLEL_SIN, Reflection, Sample};
pub use units::{Deg, Energy, HC_KEV_ANGSTROM, Rad, Wavelength};

pub use nalgebra;
