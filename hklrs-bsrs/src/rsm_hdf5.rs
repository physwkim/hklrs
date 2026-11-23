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

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use bsrs::event_model::StreamResource;
use hklrs::Image;
use rust_hdf5::format::FormatError;
use rust_hdf5::format::messages::filter::{
    BSHUF_COMPRESS_LZ4, FILTER_BITGROOM, FILTER_BITROUND, FILTER_BLOSC, FILTER_BLOSC2,
    FILTER_BSHUF, FILTER_BZIP2, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_JPEG, FILTER_LZ4,
    FILTER_LZF, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, FILTER_ZFP,
    FILTER_ZSTD,
};
use rust_hdf5::format::superblock::{FLAG_SWMR_WRITE, FLAG_WRITE_ACCESS, HDF5_SIGNATURE};
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
    /// A file libhdf5's SWMR reader does not open yet (`H5Fint.c:1478-1488`
    /// of ADSupport's HDF5 1.10.1): its superblock not flushed yet, or open
    /// for writing outside SWMR, as NDFileHDF5 has it from `H5Fcreate` until
    /// it has laid the file out (`NDFileHDF5.cpp:279`, 326).
    #[error("file {} is being created: its writer has not started SWMR", path.display())]
    Creating { path: PathBuf },
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

/// Who writes a file as a read of it begins.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Writer {
    /// None: what the read finds is final.
    None,
    /// A SWMR writer, which may yet write what the read does not find.
    Swmr,
}

/// The file's writer, from the consistency flags of a version 2 or 3
/// superblock, which libhdf5 sets as it opens a file for writing, adds SWMR
/// to in `H5Fstart_swmr_write` and clears as it closes the file, after the
/// rest of its metadata (`H5Fint.c:1459-1463`, `H5F.c:1718`,
/// `H5Fint.c:933-945`); or why a SWMR reader does not open it yet. A file
/// whose first 8 bytes are not written yet has no superblock yet; a version
/// 0 or 1 superblock carries no flags libhdf5 reads (`H5Fint.c:1451`), and
/// it and a file that is not HDF5 have no writer.
fn writer(path: &Path) -> Result<Writer, Hdf5FrameError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(Hdf5FrameError::MissingFile {
                path: path.to_owned(),
            });
        }
        // As unreadable to H5File::open, which says why.
        Err(_) => return Ok(Writer::None),
    };
    let creating = || Hdf5FrameError::Creating {
        path: path.to_owned(),
    };
    let mut blank = true;
    // H5FD_locate_signature's offsets: 0, then each power of two from 512.
    for at in std::iter::once(0).chain((9..63).map(|bit| 1u64 << bit)) {
        let mut head = Vec::with_capacity(12);
        let read = file
            .seek(SeekFrom::Start(at))
            .and_then(|_| (&mut file).take(12).read_to_end(&mut head));
        match read {
            Ok(n) if n >= 8 => {}
            Ok(_) => break,
            Err(_) => return Ok(Writer::None),
        }
        if head[..8] == HDF5_SIGNATURE {
            return match head[8..] {
                [version, _, _, flags] if version >= 2 => {
                    match flags & (FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE) {
                        0 => Ok(Writer::None),
                        f if f == FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE => Ok(Writer::Swmr),
                        _ => Err(creating()),
                    }
                }
                [_, _, _, _] => Ok(Writer::None),
                _ => Err(creating()),
            };
        }
        if at == 0 {
            blank = head[..8] == [0; 8];
        }
    }
    if blank {
        Err(creating())
    } else {
        Ok(Writer::None)
    }
}

/// The file and dataset of an image resource bsrs's AdHdfWriter emits.
fn locate(resource: &StreamResource) -> Result<(PathBuf, &str), Hdf5FrameError> {
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
    match resource.uri.strip_prefix(LOCALHOST) {
        Some(rest) => Ok((PathBuf::from(format!("/{rest}")), dataset)),
        None => Err(Hdf5FrameError::NotLocal {
            uri: resource.uri.clone(),
        }),
    }
}

impl Hdf5FrameReader {
    /// Frame `index` in the dataset's own (slow, fast) shape. Integer
    /// pixels are converted exactly or refused; float pixels are widened.
    pub fn frame(
        &self,
        resource: &StreamResource,
        index: u64,
    ) -> Result<Image<f64>, Hdf5FrameError> {
        let (path, dataset) = locate(resource)?;
        writer(&path)?;
        read_frame(&path, dataset, index)
    }
}

fn read_frame(path: &Path, dataset: &str, index: u64) -> Result<Image<f64>, Hdf5FrameError> {
    let file = H5File::open(path)?;
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

/// Not readable yet: a file the IOC has not created, which it creates at
/// its first frame (LazyOpen, `host/areadetector.rs:975`), or has not
/// started SWMR on; and, while a SWMR writer has the file, what it may yet
/// write: a frame past those flushed, as when the datum is sent after a
/// failed flush (`host/areadetector.rs:1933`), and metadata read
/// mid-rewrite. What a file no writer has lacks, it lacks for good.
impl FrameReader for Hdf5FrameReader {
    fn read(&self, resource: &StreamResource, index: u64) -> Result<Image<f64>, ReadError> {
        let (path, dataset) = locate(resource).map_err(|e| ReadError::Permanent(e.into()))?;
        let writer = writer(&path).map_err(|e| ReadError::NotYet(e.into()))?;
        read_frame(&path, dataset, index).map_err(|e| match e {
            Hdf5FrameError::IndexOutOfRange { .. }
            | Hdf5FrameError::ChecksumMismatch(_)
            | Hdf5FrameError::Hdf5(_)
                if writer == Writer::Swmr =>
            {
                ReadError::NotYet(e.into())
            }
            _ => ReadError::Permanent(e.into()),
        })
    }
}
