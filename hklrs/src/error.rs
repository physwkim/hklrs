use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
    #[error("{what} must be finite and positive, got {value}")]
    NotPositive { what: &'static str, value: f64 },
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
    #[error("the two reflections are parallel in {space} (sin of their angle = {sin:e})")]
    ParallelReflections { space: &'static str, sin: f64 },
    #[error("U is not a proper rotation (max |UᵀU − I| = {deviation}, or det ≤ 0)")]
    NotRotation { deviation: f64 },
    #[error("no Bragg reflection: lambda/2d = {ratio} > 1")]
    BraggImpossible { ratio: f64 },
    #[error("unknown geometry preset {name:?}")]
    UnknownPreset { name: String },
    #[error("axis {name:?} limits [{min}, {max}] are not ordered numbers")]
    AxisLimits { name: String, min: f64, max: f64 },
    #[error("axis {name:?} cut point {cut} is not finite")]
    CutPoint { name: String, cut: f64 },
    #[error("axis {name:?} is not a rotation and has no cut point")]
    NoCutPoint { name: String },
}
