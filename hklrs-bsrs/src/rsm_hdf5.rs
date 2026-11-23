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
use hklrs::Image;
use rust_hdf5::format::FormatError;
use rust_hdf5::format::messages::filter::{
    BSHUF_COMPRESS_LZ4, FILTER_BITGROOM, FILTER_BITROUND, FILTER_BLOSC, FILTER_BLOSC2,
    FILTER_BSHUF, FILTER_BZIP2, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_JPEG, FILTER_LZ4,
    FILTER_LZF, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, FILTER_ZFP,
    FILTER_ZSTD,
};
use rust_hdf5::{DatatypeMessage, Filter, H5File, Hdf5Error};

use crate::rsm::{FrameReader, ReadError};

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
    /// A filter of the dataset's pipeline this reader does not decode, such
    /// as NDFileHDF5's JPEG.
    #[error("HDF5 filter {id} ({name}) is not one this reader decodes")]
    UnsupportedFilter { id: u16, name: String },
    /// A pixel type rust-hdf5 does not read as a number, such as a compound.
    #[error("pixel type not readable: {0}")]
    UnsupportedType(String),
    /// Metadata whose checksum does not match: read while the writer was
    /// rewriting it, or corrupt. libhdf5's SWMR reader reads such metadata
    /// again (`H5Pset_metadata_read_attempts`); rust-hdf5's `H5File` reads
    /// it once.
    #[error("HDF5: {0}")]
    ChecksumMismatch(String),
    #[error("HDF5: {0}")]
    Hdf5(String),
}

impl From<Hdf5Error> for Hdf5FrameError {
    fn from(e: Hdf5Error) -> Self {
        match e {
            Hdf5Error::TypeMismatch(datatype) => Self::UnsupportedType(datatype),
            Hdf5Error::Format(FormatError::ChecksumMismatch { .. }) => {
                Self::ChecksumMismatch(e.to_string())
            }
            // rust-hdf5's I/O-layer error type is private (`pub(crate) mod
            // io`), so a mismatch it raises is known by its text.
            Hdf5Error::IoLayer(ref io)
                if io
                    .to_string()
                    .starts_with("format error: checksum mismatch") =>
            {
                Self::ChecksumMismatch(e.to_string())
            }
            e => Self::Hdf5(e.to_string()),
        }
    }
}

/// The first filter of `pipeline` this reader does not decode. The others
/// are those rust-hdf5 0.7.14 decodes with the workspace's features
/// (deflate, lz4, blosc, zstd), each read from a file in
/// `golden/ndfilehdf5_filters.h5`.
fn undecodable(pipeline: &[Filter]) -> Option<Hdf5FrameError> {
    pipeline.iter().find_map(|filter| {
        let name = match filter.id {
            FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 | FILTER_SZIP | FILTER_NBIT
            | FILTER_SCALEOFFSET | FILTER_LZF | FILTER_BLOSC | FILTER_LZ4 | FILTER_ZSTD
            | FILTER_BITGROOM | FILTER_BITROUND => return None,
            // cd_values[4] is the compression after the bit transpose, none
            // when absent, as rust-hdf5 reads it.
            FILTER_BSHUF => match filter.cd_values.get(4).copied().unwrap_or(0) {
                0 | BSHUF_COMPRESS_LZ4 => return None,
                3 => "bitshuffle with zstd".into(),
                other => format!("bitshuffle with compression {other}"),
            },
            FILTER_BZIP2 => "bzip2".into(),
            FILTER_ZFP => "zfp".into(),
            FILTER_JPEG => "jpeg".into(),
            FILTER_BLOSC2 => "blosc2".into(),
            _ => "unknown".into(),
        };
        Some(Hdf5FrameError::UnsupportedFilter {
            id: filter.id,
            name,
        })
    })
}

impl Hdf5FrameReader {
    /// Frame `index` in the dataset's own (slow, fast) shape. Integer
    /// pixels are converted exactly or refused; float pixels are widened.
    pub fn frame(
        &self,
        resource: &StreamResource,
        index: u64,
    ) -> Result<Image<f64>, Hdf5FrameError> {
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
        if let Some(e) = undecodable(&ds.filters()?) {
            return Err(e);
        }
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
        Image::new((slow, fast), data).map_err(|e| Hdf5FrameError::Hdf5(e.to_string()))
    }
}

/// Not readable yet: a file the IOC has not created, which it creates at
/// its first frame (LazyOpen, `host/areadetector.rs:975`); a frame past
/// those flushed, as when the datum is sent after a failed flush
/// (`host/areadetector.rs:1933`); and metadata read mid-rewrite.
impl FrameReader for Hdf5FrameReader {
    fn read(&self, resource: &StreamResource, index: u64) -> Result<Image<f64>, ReadError> {
        self.frame(resource, index).map_err(|e| match e {
            Hdf5FrameError::MissingFile { .. }
            | Hdf5FrameError::IndexOutOfRange { .. }
            | Hdf5FrameError::ChecksumMismatch(_) => ReadError::NotYet(e.into()),
            Hdf5FrameError::UnknownResource { .. }
            | Hdf5FrameError::NotLocal { .. }
            | Hdf5FrameError::NotFrames { .. }
            | Hdf5FrameError::Inexact { .. }
            | Hdf5FrameError::UnsupportedFilter { .. }
            | Hdf5FrameError::UnsupportedType(_)
            | Hdf5FrameError::Hdf5(_) => ReadError::Permanent(e.into()),
        })
    }
}
