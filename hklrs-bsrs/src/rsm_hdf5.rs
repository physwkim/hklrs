//! [`FrameReader`] for the image files areaDetector's NDFileHDF5 plugin
//! writes under bsrs 0.6.2's `AdHdfWriter`.
//!
//! bsrs's `StreamResource` has no `spec`; the writer marks its resources
//! with mimetype `application/x-hdf5`, the uri `{directory_uri}{filename}.h5`
//! and the parameters `dataset` (`/entry/data/data`) and `chunk_shape`
//! (`host/areadetector.rs:1298`, 1646-1675, 1828). The default
//! `directory_uri` is `file://localhost` followed by the absolute directory,
//! not percent-encoded (`devices/path_provider.rs:295-298`). The dataset is
//! `[frames, slow, fast]` (NumExtraDims=0, `host/areadetector.rs:972`), and
//! a datum's indices count its frames along the first axis.

use std::path::PathBuf;

use bsrs::event_model::StreamResource;
use hklrs::normalise::RawFrame;
use rust_hdf5::{DatatypeMessage, H5File, Hdf5Error};

use crate::rsm::FrameReader;

const MIMETYPE: &str = "application/x-hdf5";
/// The only file uri bsrs emits: `file://localhost` + an absolute path.
const LOCALHOST: &str = "file://localhost/";
/// Integers up to 2^53 are exact in f64.
const EXACT: u128 = 1 << f64::MANTISSA_DIGITS;

/// Reads frame `index` of the resource's dataset from its file, opened on
/// each read: the IOC keeps appending to it in SWMR mode
/// (`host/areadetector.rs:976`), and a file kept open would not see the
/// frames written after it was opened.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hdf5FrameReader;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Hdf5FrameError {
    /// Not an image resource of an HDF5 file: another mimetype, such as
    /// the soft detector's, or no string `dataset`, such as bsrs's
    /// `Hdf5FrameSink` byte stream, whose parameter is `path`.
    #[error("resource {mimetype:?} with parameters {parameters:?} is not an HDF5 image dataset")]
    UnknownResource {
        mimetype: String,
        /// The parameter names, sorted.
        parameters: Vec<String>,
    },
    #[error("uri {uri:?} is not a file on this host, {LOCALHOST}...")]
    NotLocal { uri: String },
    #[error("file {} does not exist", path.display())]
    MissingFile { path: PathBuf },
    #[error("frame {index} is out of range: the dataset holds {frames} frames")]
    IndexOutOfRange { index: u64, frames: usize },
    #[error("dataset of shape {shape:?} is not [frames, slow, fast]")]
    NotFrames { shape: Vec<usize> },
    #[error("pixel {pixel}, {value}, has no exact f64")]
    Inexact { pixel: usize, value: i128 },
    #[error("HDF5: {0}")]
    Hdf5(String),
}

impl From<Hdf5Error> for Hdf5FrameError {
    fn from(e: Hdf5Error) -> Self {
        Self::Hdf5(e.to_string())
    }
}

impl Hdf5FrameReader {
    /// Frame `index` in the dataset's own (slow, fast) shape. Integer
    /// pixels are converted exactly or refused; float pixels are widened.
    pub fn frame(&self, resource: &StreamResource, index: u64) -> Result<RawFrame, Hdf5FrameError> {
        let dataset = match resource.parameters.get("dataset") {
            Some(serde_json::Value::String(d)) if resource.mimetype == MIMETYPE => d,
            _ => {
                let mut parameters: Vec<String> = resource.parameters.keys().cloned().collect();
                parameters.sort();
                return Err(Hdf5FrameError::UnknownResource {
                    mimetype: resource.mimetype.clone(),
                    parameters,
                });
            }
        };
        let path = match resource.uri.strip_prefix(LOCALHOST) {
            Some(rest) => PathBuf::from(format!("/{rest}")),
            None => {
                return Err(Hdf5FrameError::NotLocal {
                    uri: resource.uri.clone(),
                });
            }
        };
        let file = H5File::open(&path).map_err(|e| match path.try_exists() {
            Ok(false) => Hdf5FrameError::MissingFile { path: path.clone() },
            _ => e.into(),
        })?;
        let ds = file.dataset(dataset)?;
        let shape = ds.shape();
        let &[frames, slow, fast] = shape.as_slice() else {
            return Err(Hdf5FrameError::NotFrames { shape });
        };
        let i = usize::try_from(index)
            .ok()
            .filter(|&i| i < frames)
            .ok_or(Hdf5FrameError::IndexOutOfRange { index, frames })?;
        let (starts, counts) = ([i, 0, 0], [1, slow, fast]);
        let data = match ds.datatype()? {
            DatatypeMessage::FloatingPoint { .. } => {
                ds.read_numeric_slice_as::<f64>(&starts, &counts)?
            }
            _ => ds
                .read_numeric_slice_as::<i128>(&starts, &counts)?
                .into_iter()
                .enumerate()
                .map(|(pixel, value)| {
                    if value.unsigned_abs() <= EXACT {
                        Ok(value as f64)
                    } else {
                        Err(Hdf5FrameError::Inexact { pixel, value })
                    }
                })
                .collect::<Result<_, _>>()?,
        };
        RawFrame::new((slow, fast), data).map_err(|e| Hdf5FrameError::Hdf5(e.to_string()))
    }
}

impl FrameReader for Hdf5FrameReader {
    fn read(
        &self,
        resource: &StreamResource,
        index: u64,
    ) -> Result<RawFrame, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.frame(resource, index)?)
    }
}
