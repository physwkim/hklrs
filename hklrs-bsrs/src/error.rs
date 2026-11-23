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
    #[error("hkl {hkl:?}: {source}")]
    Solve {
        hkl: [f64; 3],
        source: hklrs::InverseError,
    },
    #[error(transparent)]
    Core(#[from] hklrs::Error),
}

impl From<DiffError> for BsrsError {
    fn from(e: DiffError) -> Self {
        BsrsError::InvalidValue(e.to_string())
    }
}
