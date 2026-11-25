use thiserror::Error;

use crate::grid::{Binning, Coord, CoordKind};
use crate::projection::ProjectionError;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
    #[error("{what} must be finite and positive, got {value}")]
    NotPositive { what: &'static str, value: f64 },
    /// A width along a coordinate of `kind`: a bin step, a cut half-width
    /// or a footprint width.
    #[error("{what} must be finite and positive, got {value}")]
    Width {
        what: &'static str,
        value: f64,
        kind: CoordKind,
    },
    #[error("lattice angle {name} = {value} rad is outside (0, pi)")]
    LatticeAngle { name: &'static str, value: f64 },
    #[error("lattice angles give a cell volume squared of {v2} <= 0")]
    LatticeVolume { v2: f64 },
    #[error("vector {what} must be finite and non-zero")]
    ZeroVector { what: String },
    #[error("axis {name:?} is defined twice with different kinds")]
    AxisConflict { name: String },
    #[error("axis {name:?} appears twice in the {holder} stack")]
    AxisRepeated { name: String, holder: &'static str },
    #[error("unknown axis {name:?}")]
    UnknownAxis { name: String },
    #[error("expected {expected} axis values, got {got}")]
    AxisCount { expected: usize, got: usize },
    #[error("axis value {index} is not finite")]
    NonFinite { index: usize },
    #[error("detector shape must be non-zero, got {slow}x{fast}")]
    DetectorShape { slow: usize, fast: usize },
    #[error("pixel ({slow}, {fast}) is outside the detector shape {shape:?}")]
    PixelOutside {
        slow: usize,
        fast: usize,
        shape: (usize, usize),
    },
    #[error("the detector point coincides with the sample position")]
    DetectorAtSample,
    #[error("roi {slow:?} x {fast:?} is empty or outside the detector shape {shape:?}")]
    Roi {
        slow: std::ops::Range<usize>,
        fast: std::ops::Range<usize>,
        shape: (usize, usize),
    },
    #[error("bin {slow}x{fast} is zero or larger than the detector shape {shape:?}")]
    Bin {
        slow: usize,
        fast: usize,
        shape: (usize, usize),
    },
    #[error("{what} must be finite, got {value}")]
    NotFinite { what: &'static str, value: f64 },
    #[error("xrayutilities direction {text:?} is not one of x+ x- y+ y- z+ z-")]
    XuDirection { text: String },
    #[error("xrayutilities roi {roi:?} spans {} pixels, which Nav {nav} does not divide", .roi.len())]
    NavRoi {
        roi: std::ops::Range<usize>,
        nav: usize,
    },
    #[error("{what} are parallel")]
    ParallelDirections { what: &'static str },
    #[error("the two reflections are parallel in {space} (sin of their angle = {sin:e})")]
    ParallelReflections { space: &'static str, sin: f64 },
    #[error("{what} is not a proper rotation (max |{what}ᵀ·{what} − I| = {deviation}, or det ≤ 0)")]
    NotRotation { what: &'static str, deviation: f64 },
    #[error("no Bragg reflection: lambda/2d = {ratio} > 1")]
    BraggImpossible { ratio: f64 },
    #[error("unknown geometry preset {name:?}")]
    UnknownPreset { name: String },
    #[error("axis {name:?} limits [{min}, {max}] are not ordered numbers")]
    AxisLimits {
        name: String,
        min: f64,
        max: f64,
        kind: CoordKind,
    },
    #[error("axis {name:?} cut point {cut} is not finite")]
    CutPoint { name: String, cut: f64 },
    #[error("axis {name:?} is not a rotation and has no cut point")]
    NoCutPoint { name: String },
    #[error(
        "a detector module has a point axis, so its pixels span no area: no solid angle, and no normal for a window or sensor"
    )]
    NoPixelArea,
    #[error("polarisation factor must be in [-1, 1], got {value}")]
    PolarisationFactor { value: f64 },
    #[error("pixel axis: {reason}")]
    PixelAxis { reason: &'static str },
    #[error(
        "the detector modules do not tile the image {shape:?}: one is outside, two overlap, or a pixel is in none"
    )]
    ModuleTiling { shape: (usize, usize) },
    #[error("bin {slow}x{fast} puts pixels of the module at {at:?} and of another in one block")]
    BinModule {
        slow: usize,
        fast: usize,
        at: (usize, usize),
    },
    #[error("detector id {id} is not in the table of {count}")]
    UnknownDetector { id: u32, count: usize },
    #[error("coordinate {value} has no bin of width {step}")]
    NoBin {
        value: f64,
        step: f64,
        kind: CoordKind,
    },
    #[error("expected {expected} dimensions, one per projection axis, got {got}")]
    Dimensions { expected: usize, got: usize },
    #[error("{what} has {len} pixels, expected {expected}")]
    PixelCount {
        what: &'static str,
        len: usize,
        expected: usize,
    },
    /// Two (slow, fast) shapes that must agree; a pixel count would not
    /// tell a transposed shape from the right one.
    #[error("{what} shape {shape:?} does not match {expected:?}")]
    ShapeMismatch {
        what: &'static str,
        shape: (usize, usize),
        expected: (usize, usize),
    },
    #[error("a custom projection needs at least one coordinate")]
    NoCoords,
    #[error("coordinate {name:?} appears twice in the custom projection")]
    CoordRepeated { name: String },
    #[error("custom projection: {0}")]
    Projection(ProjectionError),
    /// Per axis, the coordinate (name and kind) and binning compared.
    #[error("grids differ: {left:?} vs {right:?}")]
    GridMismatch {
        left: Vec<(Coord, Binning)>,
        right: Vec<(Coord, Binning)>,
    },
    #[error("a dark field is a rate per second: the frame needs a count time")]
    DarkNeedsTime,
    #[error("the cube's bin count does not fit in memory")]
    CubeSize,
    #[error("the ray meets no detector module within its pixels")]
    RayOutside,
    #[error("the ray meets the detector only behind its origin")]
    RayBehind,
    #[error("the ray is parallel to the detector")]
    RayParallel,
    #[error("a point detector has no extent for a ray to meet")]
    PointDetector,
    #[cfg(feature = "xrt")]
    #[error("material: {reason}")]
    Material { reason: String },
    #[cfg(feature = "hdf5")]
    #[error("HDF5: {reason}")]
    Hdf5 { reason: String },
    #[cfg(feature = "hdf5")]
    #[error("binoculars cube file: {reason}")]
    CubeFile { reason: String },
    #[cfg(feature = "hdf5")]
    #[error(
        "binoculars centres bin k on k·step: axis {axis:?} of step {step} is \
         anchored at {anchor}, not at −step/2"
    )]
    CubeFileAnchor {
        axis: String,
        step: f64,
        anchor: f64,
    },
}
