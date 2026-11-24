//! Diffractometer geometry and reciprocal-space conversion.
//!
//! Units are fixed: radians, Å for lattice and wavelength, metres for
//! detector geometry and translations, keV for energy. Q includes 2π.

pub mod absorption;
pub mod calib;
pub mod cut;
mod detector;
mod detector_table;
mod error;
mod forward;
pub mod frame_map;
mod geometry;
pub mod grid;
mod image;
pub mod inverse;
mod lattice;
pub mod normalise;
pub mod poni;
pub mod presets;
pub mod projection;
pub mod pseudo;
mod qmap;
mod sample;
mod sixs;
mod units;
pub mod xu;

pub use detector::{Module, PixelAxis, PixelDetector};
pub use detector_table::{Detector, Layout, Row, Strip, Tiling};
pub use error::Error;
pub use forward::{
    Forward, Scattering, angle_between, detector_position, forward, scattered_direction,
    scattering, scattering_at,
};
pub use geometry::{Axis, AxisKind, Geometry, GeometryBuilder, Poses};
pub use image::Image;
pub use inverse::{
    Equation, Eulerian, InverseError, InverseRequest, Mode, ModeParameters, ModeSpec, Picker,
    Slits, Target,
};
pub use lattice::Lattice;
pub use qmap::{
    Planes, Polarisation, QMap, Weights, absorption, detector_positions, polarisation, q_map,
    solid_angle,
};
pub use sample::{Orientation, PARALLEL_SIN, Reflection, Sample, proper_rotation};
pub use sixs::sixs;
pub use units::{Deg, Energy, HC_KEV_ANGSTROM, PerMetre, Rad, Wavelength};

pub use nalgebra;
