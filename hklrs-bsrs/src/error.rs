use bsrs::core::BsrsError;
use serde_json::Value;

/// Every variant names the value it rejected (PY2-6).
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum DiffError {
    #[error("unknown sample {name:?}; known: {known:?}")]
    UnknownSample { name: String, known: Vec<String> },
    #[error("sample {name:?} already exists")]
    DuplicateSample { name: String },
    #[error("sample {name:?} is the only sample and cannot be removed")]
    LastSample { name: String },
    #[error("sample {sample:?} has no reflection {name:?}; known: {known:?}")]
    UnknownReflection {
        sample: String,
        name: String,
        known: Vec<String>,
    },
    #[error("sample {sample:?} already has a reflection {name:?}")]
    DuplicateReflection { sample: String, name: String },
    #[error("axis {name:?} is given more than one real")]
    DuplicateAxis { name: String },
    #[error("axis {name:?} has no real")]
    MissingAxis { name: String },
    #[error("positions give no value for axis {axis:?}")]
    MissingPosition { axis: String },
    #[error("wavelength signal {signal:?} reads fields {fields:?}, not exactly one")]
    WavelengthFields { signal: String, fields: Vec<String> },
    #[error("wavelength signal {signal:?} reads {value}, not a number")]
    WavelengthValue { signal: String, value: Value },
    #[error("the wavelength is read from signal {signal:?} and cannot be set")]
    WavelengthFromSignal { signal: String },
    #[error("configuration key {key:?} is not settable; settable: {settable:?}")]
    ConfigKey { key: String, settable: Vec<String> },
    #[error("configuration {key:?} = {value} is not a {expected}")]
    ConfigValue {
        key: String,
        value: Value,
        expected: &'static str,
    },
    #[error("mode {mode:?}: {source}")]
    Mode {
        mode: String,
        source: hklrs::InverseError,
    },
    #[error("geometry {geometry:?} has no hkl mode")]
    NoMode { geometry: String },
    #[error("geometry {geometry:?} has psi modes {candidates:?}, not exactly one: name the mode")]
    PsiMode {
        geometry: String,
        candidates: Vec<String>,
    },
    #[error("mode {mode:?} has no psi: set a psi mode")]
    NoPsi { mode: String },
    #[error("psi about reference {reference:?}: {source}")]
    Psi {
        reference: [f64; 3],
        source: hklrs::InverseError,
    },
    #[error("hkl {hkl:?}: {source}")]
    Solve {
        hkl: [f64; 3],
        source: hklrs::InverseError,
    },
    #[error("point {index}: {source}")]
    Point {
        index: usize,
        source: Box<DiffError>,
    },
    #[error("data key {key:?} is read by both {first:?} and {second:?}")]
    DataKeyCollision {
        key: String,
        first: String,
        second: String,
    },
    #[error("hkl {start:?} and {stop:?} span no zone: both must be non-zero and not parallel")]
    Zone { start: [f64; 3], stop: [f64; 3] },
    #[error("axis {name:?} limits [{min}, {max}] are not an interval")]
    Limits { name: String, min: f64, max: f64 },
    #[error(
        "configuration geometry {found:?} with axes {found_axes:?} is not the device's \
         {expected:?} with axes {expected_axes:?}"
    )]
    ConfigGeometry {
        found: String,
        found_axes: Vec<String>,
        expected: String,
        expected_axes: Vec<String>,
    },
    #[error("configuration has no {field}")]
    ConfigMissing { field: String },
    #[error("{format} configuration schema version {found} is not {expected}")]
    ConfigSchema {
        format: &'static str,
        found: u64,
        expected: u64,
    },
    #[error("{format} configuration: {message}")]
    ConfigParse {
        format: &'static str,
        message: String,
    },
    #[error("configuration file {path:?}: {message}")]
    ConfigFile { path: String, message: String },
    #[error("sample {sample:?}: {source}")]
    InSample {
        sample: String,
        source: Box<DiffError>,
    },
    #[error("reflection {name:?}: {source}")]
    InReflection {
        name: String,
        source: Box<DiffError>,
    },
    #[error("hklpy2 {field} = {value} is not supported")]
    Hklpy2Unsupported { field: String, value: String },
    #[error(
        "hklpy2 file is self-inconsistent: its UB {ub:?} is not U·B {u_b:?} of its U and \
         lattice, and the sample has no two orienting reflections to decide which is meant"
    )]
    Hklpy2UbUndecided {
        ub: Box<[[f64; 3]; 3]>,
        u_b: Box<[[f64; 3]; 3]>,
    },
    #[error(
        "hklpy2 file is self-inconsistent: its UB {ub:?} is not U·B {u_b:?} of its U and \
         lattice, and UB {measured:?} of orienting reflections {:?} and {:?} does not match \
         exactly one of them",
        reflections[0],
        reflections[1]
    )]
    Hklpy2UbUnmatched {
        ub: Box<[[f64; 3]; 3]>,
        u_b: Box<[[f64; 3]; 3]>,
        reflections: [String; 2],
        measured: Box<[[f64; 3]; 3]>,
    },
    #[error(transparent)]
    Core(#[from] hklrs::Error),
}

/// A device read that failed, or a diffractometer error; either's own
/// message.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Fault {
    #[error(transparent)]
    Device(#[from] BsrsError),
    #[error(transparent)]
    Diff(#[from] DiffError),
}

impl From<DiffError> for BsrsError {
    fn from(e: DiffError) -> Self {
        BsrsError::InvalidValue(e.to_string())
    }
}
