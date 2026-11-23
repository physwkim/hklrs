//! Hdf5FrameReader on files in the layout areaDetector's NDFileHDF5 plugin
//! writes under bsrs 0.6.2's AdHdfWriter: `/entry/data/data` of
//! `[frames, slow, fast]`, chunked a frame at a time, streamed in SWMR mode,
//! and named by the resource that writer emits.
#![cfg(feature = "hdf5")]

use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use bsrs::devices::path_provider::PathInfo;
use bsrs::event_model::{
    DataKey, Document, Dtype, Event, EventDescriptor, RunStart, RunStop, StreamDatum, StreamRange,
    StreamResource,
};
use hklrs::frame_map::FrameMap;
use hklrs::grid::{Binning, Cube};
use hklrs::nalgebra::Vector3;
use hklrs::normalise::{FrameScalars, PixelFields, Region};
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{Image, PixelDetector, Wavelength};
use hklrs_bsrs::{
    FailedFrame, FrameFailure, FrameReader, Hdf5FrameError, Hdf5FrameReader, ReadError,
    RsmCallback, RsmConfig, RsmValue,
};
use rust_hdf5::format::superblock::{FLAG_WRITE_ACCESS, SuperblockV2V3};
use rust_hdf5::swmr::SwmrFileWriter;
use rust_hdf5::{Complex32, H5Type};
use serde_json::{Value, json};

const LAMBDA: f64 = 1.54;
const SHAPE: (usize, usize) = (4, 6);
const IMAGE: &str = "det";
const DELTA: f64 = 25.0;
const KOMEGA: [f64; 5] = [18.0, 19.0, 20.0, 21.0, 22.0];
/// `AD_HDF_DATASET`, `host/areadetector.rs:1298`.
const DATASET: &str = "/entry/data/data";

/// K6C mu, komega, kappa, kphi, gamma, delta.
fn readbacks(komega: f64) -> [f64; 6] {
    [3.0, komega, -30.0, 40.0, 12.0, DELTA]
}

/// Frame `index`'s pixels, every pixel of every frame different.
fn pixels(index: usize) -> Vec<u16> {
    let n = SHAPE.0 * SHAPE.1;
    (0..n).map(|p| (1 + index * n + p) as u16).collect()
}

fn frame_map() -> FrameMap {
    let detector = PixelDetector::new(
        SHAPE,
        Vector3::new(0.5, -0.003, -0.002),
        Vector3::new(0.0, 0.0, 0.001),
        Vector3::new(0.0, 0.001, 0.0),
    )
    .unwrap();
    FrameMap::new(
        preset("K6C").unwrap(),
        Projection::QSample,
        &detector,
        &Region::new(0..4, 0..6, (2, 2)),
        PixelFields::new(),
        vec![Binning::new(0.01, 0.0).unwrap(); 3],
    )
    .unwrap()
}

/// Frames `0..n` through FrameMap alone, from the pixels written.
fn direct(n: usize) -> Cube {
    let map = frame_map();
    let wavelength = Wavelength::new(LAMBDA).unwrap();
    let mut cube = Cube::default();
    for (i, &komega) in KOMEGA[..n].iter().enumerate() {
        let data = pixels(i).into_iter().map(f64::from).collect();
        let raw = Image::new(SHAPE, data).unwrap();
        let frame = map
            .map(&readbacks(komega), wavelength, &raw, &FrameScalars::new())
            .unwrap();
        map.add(&mut cube, &frame).unwrap();
    }
    cube
}

/// An empty directory of its own for each test.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("rsm_hdf5")
        .join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The `.h5` uri AdHdfWriter puts in its resource for file `scan` in
/// `dir`, through bsrs's own PathInfo (`host/areadetector.rs:1828`).
fn uri(dir: &Path) -> String {
    let info = PathInfo::new(dir.to_path_buf(), "scan", 0, None).unwrap();
    format!("{}{}.h5", info.directory_uri(), info.filename())
}

/// The resource AdHdfWriter emits for its image (`ad_stream_resource`,
/// `host/areadetector.rs:1646-1675`), chunked a frame at a time.
fn ad_resource(uri: String, dataset: &str, chunk_shape: &[u64]) -> StreamResource {
    StreamResource {
        uid: "res".into(),
        data_key: IMAGE.into(),
        mimetype: "application/x-hdf5".into(),
        uri,
        parameters: HashMap::from([
            ("dataset".to_owned(), json!(dataset)),
            ("chunk_shape".to_owned(), json!(chunk_shape)),
        ]),
        run_start: None,
    }
}

fn image_resource(dir: &Path) -> StreamResource {
    ad_resource(uri(dir), DATASET, &[1, SHAPE.0 as u64, SHAPE.1 as u64])
}

/// A pixel type, as the bytes `append_frame` takes.
trait Pixel: H5Type + Copy {
    fn bytes(self) -> Vec<u8>;
}

impl Pixel for u16 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

impl Pixel for u64 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

impl Pixel for f32 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

impl Pixel for f64 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

impl Pixel for Complex32 {
    fn bytes(self) -> Vec<u8> {
        [self.re, self.im]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect()
    }
}

/// One dataset of `scan.h5` a SWMR writer still has open.
struct Stream {
    writer: SwmrFileWriter,
    ds: usize,
}

impl Stream {
    /// One frame, flushed as AdHdfWriter flushes before each datum.
    fn append<T: Pixel>(&mut self, frame: &[T]) {
        let bytes: Vec<u8> = frame.iter().flat_map(|&v| v.bytes()).collect();
        self.writer.append_frame(self.ds, &bytes).unwrap();
        self.writer.flush().unwrap();
    }

    fn close(self) {
        self.writer.close().unwrap();
    }
}

/// `scan.h5` in `dir` with `frames` appended to `name` under `group`, one
/// chunk per frame as NDFileHDF5 chunks them.
fn stream<T: Pixel>(
    dir: &Path,
    group: &str,
    name: &str,
    frame: &[u64],
    frames: &[Vec<T>],
) -> Stream {
    let mut writer = SwmrFileWriter::create(dir.join("scan.h5")).unwrap();
    let mut parent = String::new();
    for g in group.split('/').filter(|g| !g.is_empty()) {
        writer
            .create_group(if parent.is_empty() { "/" } else { &parent }, g)
            .unwrap();
        parent = format!("{parent}/{g}");
    }
    let chunk: Vec<u64> = std::iter::once(1).chain(frame.iter().copied()).collect();
    let ds = writer
        .create_streaming_dataset_chunked::<T>(name, frame, &chunk)
        .unwrap();
    writer.assign_dataset_to_group(group, ds).unwrap();
    writer.start_swmr().unwrap();
    let mut stream = Stream { writer, ds };
    for f in frames {
        stream.append(f);
    }
    stream
}

/// The image dataset of `n` frames of [`pixels`], streamed as NDFileHDF5
/// writes it.
fn images(dir: &Path, n: usize) -> Stream {
    let frames: Vec<Vec<u16>> = (0..n).map(pixels).collect();
    stream(
        dir,
        "/entry/data",
        "data",
        &[SHAPE.0 as u64, SHAPE.1 as u64],
        &frames,
    )
}

// -- documents of run "run", stream "primary" ---------------------------------

fn data_key(external: Option<&str>) -> DataKey {
    DataKey {
        source: "test".into(),
        dtype: Dtype::Number,
        shape: vec![],
        dtype_numpy: None,
        external: external.map(str::to_owned),
        units: None,
        precision: None,
        object_name: None,
        dims: None,
        limits: None,
        choices: None,
    }
}

/// Start, descriptor and `resource`, then for each frame `0..events` its
/// datum and its event `i + 1`, then stop: a step scan's order.
fn run(resource: StreamResource, events: usize) -> Vec<Document> {
    let mut docs = vec![
        Document::Start(RunStart {
            uid: "run".into(),
            ..Default::default()
        }),
        Document::Descriptor(EventDescriptor {
            uid: "desc".into(),
            run_start: "run".into(),
            time: 0.0,
            data_keys: HashMap::from([
                (IMAGE.to_owned(), data_key(Some("STREAM:"))),
                ("komega".to_owned(), data_key(None)),
                ("delta".to_owned(), data_key(None)),
            ]),
            configuration: HashMap::new(),
            name: Some("primary".into()),
            hints: None,
            object_keys: HashMap::new(),
        }),
        Document::StreamResource(resource),
    ];
    for i in 0..events as u64 {
        docs.push(Document::StreamDatum(StreamDatum {
            uid: format!("datum-{i}"),
            stream_resource: "res".into(),
            descriptor: "desc".into(),
            indices: StreamRange {
                start: i,
                stop: i + 1,
            },
            seq_nums: StreamRange {
                start: i + 1,
                stop: i + 2,
            },
        }));
        let data: HashMap<String, Value> = HashMap::from([
            ("komega".to_owned(), json!(KOMEGA[i as usize])),
            ("delta".to_owned(), json!(DELTA)),
        ]);
        docs.push(Document::Event(Event {
            uid: format!("event-{}", i + 1),
            descriptor: "desc".into(),
            time: 0.0,
            seq_num: i + 1,
            data,
            timestamps: HashMap::new(),
            filled: HashMap::new(),
        }));
    }
    docs.push(Document::Stop(RunStop {
        uid: "stop".into(),
        run_start: "run".into(),
        ..Default::default()
    }));
    docs
}

/// Whether FrameReader calls the failed read not readable yet.
fn not_yet(resource: &StreamResource, index: u64) -> bool {
    match FrameReader::read(&Hdf5FrameReader, resource, index) {
        Err(ReadError::NotYet(_)) => true,
        Err(ReadError::Permanent(_)) => false,
        Ok(_) => panic!("frame {index} was read"),
    }
}

/// One read: the frame index, when, and whether it was not readable yet.
type Read = (u64, Instant, bool);

/// Hdf5FrameReader that reports each read that fails.
struct Watched(Sender<Read>);

impl FrameReader for Watched {
    fn read(&self, resource: &StreamResource, index: u64) -> Result<Image<f64>, ReadError> {
        let at = Instant::now();
        let read = Hdf5FrameReader.read(resource, index);
        if let Err(e) = &read {
            let _ = self.0.send((index, at, matches!(e, ReadError::NotYet(_))));
        }
        read
    }
}

/// The next failed read, waiting up to 5 s for it.
fn next_failure(reads: &Receiver<Read>) -> Read {
    reads.recv_timeout(Duration::from_secs(5)).unwrap()
}

fn rsm(reader: impl FrameReader + 'static, read_deadline: Duration) -> RsmCallback {
    let field = |name: &str| RsmValue::Field(name.to_owned());
    RsmCallback::new(RsmConfig {
        map: frame_map(),
        stream: "primary".into(),
        image_key: IMAGE.into(),
        axes: vec![
            RsmValue::Fixed(3.0),
            field("komega"),
            RsmValue::Fixed(-30.0),
            RsmValue::Fixed(40.0),
            RsmValue::Fixed(12.0),
            field("delta"),
        ],
        wavelength: RsmValue::Fixed(Wavelength::new(LAMBDA).unwrap()),
        monitor: None,
        count_time: None,
        seed: None,
        reader: Arc::new(reader),
        read_deadline,
    })
    .unwrap()
}

// -- mapping ------------------------------------------------------------------

/// The frames of a file the IOC is still streaming, mapped through
/// RsmCallback, make the cube FrameMap makes of the pixels written.
#[test]
fn a_streaming_file_maps_as_frame_map_does() {
    let dir = scratch("a_streaming_file_maps_as_frame_map_does");
    let writer = images(&dir, 5);
    let rsm = rsm(Hdf5FrameReader, RsmConfig::DEFAULT_READ_DEADLINE);
    for doc in run(image_resource(&dir), 5) {
        rsm.document(&doc);
    }
    let map = rsm.finish().unwrap();
    writer.close();
    assert_eq!(map.failed, []);
    assert_eq!(map.mapped, [1, 2, 3, 4, 5]);
    assert_eq!(map.cube, direct(5));
}

/// A float image is read as stored, in the dataset's (slow, fast) shape:
/// a (3, 5) frame of another frame shape than the map's.
#[test]
fn a_float_frame_is_read_in_its_own_shape() {
    let dir = scratch("a_float_frame_is_read_in_its_own_shape");
    let frames: Vec<Vec<f32>> = (0..2)
        .map(|i| (0..15).map(|p| 0.25 + (i * 15 + p) as f32).collect())
        .collect();
    let writer = stream(&dir, "/entry/data", "data", &[3, 5], &frames);
    let resource = ad_resource(uri(&dir), DATASET, &[1, 3, 5]);
    let frame = Hdf5FrameReader.frame(&resource, 1).unwrap();
    writer.close();
    assert_eq!(frame.shape(), (3, 5));
    let expected: Vec<f64> = frames[1].iter().map(|&v| f64::from(v)).collect();
    assert_eq!(frame.data(), expected);
}

/// The file is opened on each read, so a frame the writer appends after a
/// read found it missing is there on the next.
#[test]
fn a_frame_appended_after_a_read_is_read_next_time() {
    let dir = scratch("a_frame_appended_after_a_read_is_read_next_time");
    let mut writer = images(&dir, 2);
    let resource = image_resource(&dir);
    assert_eq!(
        Hdf5FrameReader.frame(&resource, 2).unwrap_err(),
        Hdf5FrameError::IndexOutOfRange {
            index: 2,
            frames: 2
        }
    );
    assert!(not_yet(&resource, 2));
    writer.append(&pixels(2));
    let frame = Hdf5FrameReader.frame(&resource, 2).unwrap();
    writer.close();
    let expected: Vec<f64> = pixels(2).into_iter().map(f64::from).collect();
    assert_eq!(frame.data(), expected);
}

/// Event 3's datum comes before the IOC has written frame 2, as when
/// bsrs's flush before the datum failed: the worker's first read finds it
/// past the file, the IOC then writes it, and the worker's next read maps
/// it.
#[test]
fn a_frame_appended_after_its_first_read_maps() {
    let dir = scratch("a_frame_appended_after_its_first_read_maps");
    let mut writer = images(&dir, 2);
    let (sender, reads) = mpsc::channel();
    let rsm = rsm(Watched(sender), RsmConfig::DEFAULT_READ_DEADLINE);
    for doc in run(image_resource(&dir), 3) {
        rsm.document(&doc);
    }
    let (index, _, waiting) = next_failure(&reads);
    assert_eq!((index, waiting), (2, true));
    writer.append(&pixels(2));
    let map = rsm.finish().unwrap();
    writer.close();
    assert_eq!(map.failed, []);
    assert_eq!(map.mapped, [1, 2, 3]);
    assert_eq!(map.cube, direct(3));
}

/// A datum past the frames in the file whose frame is never written:
/// through RsmCallback, frame 3 is read again until the deadline after it
/// was paired and fails as out of range of the 3 written; the others map.
#[test]
fn a_frame_never_written_fails_at_the_deadline() {
    let dir = scratch("a_frame_never_written_fails_at_the_deadline");
    let writer = images(&dir, 3);
    let deadline = Duration::from_millis(150);
    let (sender, reads) = mpsc::channel();
    let rsm = rsm(Watched(sender), deadline);
    let fed = Instant::now();
    for doc in run(image_resource(&dir), 4) {
        rsm.document(&doc);
    }
    let map = rsm.finish().unwrap();
    writer.close();
    assert_eq!(
        map.failed,
        [FailedFrame {
            seq_num: 4,
            reason: FrameFailure::Read(
                "frame 3 is out of range: the dataset holds 3 frames".into()
            )
        }]
    );
    assert_eq!(map.mapped, [1, 2, 3]);
    assert_eq!(map.cube, direct(3));
    let failed: Vec<Read> = reads.try_iter().collect();
    assert!(failed.len() >= 2, "{} reads", failed.len());
    assert!(
        failed
            .iter()
            .all(|&(index, _, waiting)| index == 3 && waiting)
    );
    assert!(failed[failed.len() - 1].1 - fed >= deadline);
}

/// The superblock read while its writer rewrites it, made here by
/// flipping a byte of its end-of-file address (bytes 28..36 of a version
/// 2 or 3 superblock): its checksum does not match, which is not readable
/// yet, and the frame reads once the superblock is whole again.
#[test]
fn a_metadata_checksum_mismatch_is_not_readable_yet() {
    let dir = scratch("a_metadata_checksum_mismatch_is_not_readable_yet");
    let writer = images(&dir, 1);
    let resource = image_resource(&dir);
    let path = dir.join("scan.h5");
    let flip = |byte: u8| {
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(28)).unwrap();
        file.write_all(&[byte]).unwrap();
    };
    let whole = std::fs::read(&path).unwrap()[28];
    flip(!whole);
    let e = Hdf5FrameReader.frame(&resource, 0).unwrap_err();
    assert!(matches!(e, Hdf5FrameError::ChecksumMismatch(_)), "{e:?}");
    assert!(not_yet(&resource, 0));
    flip(whole);
    let frame = Hdf5FrameReader.frame(&resource, 0).unwrap();
    writer.close();
    let expected: Vec<f64> = pixels(0).into_iter().map(f64::from).collect();
    assert_eq!(frame.data(), expected);
}

/// NDFileHDF5's file as `H5Fcreate` leaves it until it flushes the
/// superblock (`H5Fint.c:1290-1306` of ADSupport's HDF5 1.10.1).
#[test]
fn an_empty_file_is_being_created() {
    let dir = scratch("an_empty_file_is_being_created");
    std::fs::write(dir.join("scan.h5"), b"").unwrap();
    let resource = image_resource(&dir);
    assert_eq!(
        Hdf5FrameReader.frame(&resource, 0).unwrap_err(),
        Hdf5FrameError::Creating {
            path: dir.join("scan.h5")
        }
    );
    assert!(not_yet(&resource, 0));
}

/// NDFileHDF5's file from `H5Fcreate` until it starts SWMR: its
/// superblock's write-access flag set, the SWMR flag not
/// (`H5Fint.c:1459-1461`), which libhdf5's SWMR reader refuses
/// (`H5Fint.c:1478-1484`). Made here from a closed file whose frame reads.
#[test]
fn a_file_written_outside_swmr_is_being_created() {
    let dir = scratch("a_file_written_outside_swmr_is_being_created");
    images(&dir, 1).close();
    let path = dir.join("scan.h5");
    let mut superblock = SuperblockV2V3::decode(&std::fs::read(&path).unwrap()).unwrap();
    superblock.file_consistency_flags = FLAG_WRITE_ACCESS;
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.write_all(&superblock.encode()).unwrap();
    let resource = image_resource(&dir);
    assert_eq!(
        Hdf5FrameReader.frame(&resource, 0).unwrap_err(),
        Hdf5FrameError::Creating { path }
    );
    assert!(not_yet(&resource, 0));
}

/// A dataset or a frame a SWMR writer has not written is not readable yet;
/// once the writer has closed the file, what it lacks fails at once.
#[test]
fn what_a_closed_file_lacks_fails_at_once() {
    let dir = scratch("what_a_closed_file_lacks_fails_at_once");
    let writer = images(&dir, 1);
    let frame = image_resource(&dir);
    let dataset = ad_resource(uri(&dir), "/entry/data/other", &[1, 4, 6]);
    assert!(not_yet(&frame, 1));
    assert!(not_yet(&dataset, 0));
    writer.close();
    assert_eq!(
        Hdf5FrameReader.frame(&frame, 1).unwrap_err(),
        Hdf5FrameError::IndexOutOfRange {
            index: 1,
            frames: 1
        }
    );
    assert!(!not_yet(&frame, 1));
    assert!(!not_yet(&dataset, 0));
}

// -- errors, one each ---------------------------------------------------------

/// The soft detector's resource and bsrs's Hdf5FrameSink byte stream,
/// whose dataset parameter is `path` (`stream/sinks/hdf5.rs:246-257`).
#[test]
fn a_resource_that_is_no_hdf5_image_is_unknown() {
    let soft = StreamResource {
        uid: "res".into(),
        data_key: IMAGE.into(),
        mimetype: "application/x-bsrs-soft-frames".into(),
        uri: "memory://soft-frames".into(),
        parameters: HashMap::new(),
        run_start: None,
    };
    let sink = StreamResource {
        mimetype: "application/x-hdf5".into(),
        uri: "file://localhost/data/sink.h5".into(),
        parameters: HashMap::from([("path".to_owned(), json!("/entry/instrument/det/data"))]),
        ..soft.clone()
    };
    assert_eq!(
        Hdf5FrameReader.frame(&soft, 0).unwrap_err(),
        Hdf5FrameError::UnknownResource {
            mimetype: "application/x-bsrs-soft-frames".into(),
            parameters: vec![],
        }
    );
    assert_eq!(
        Hdf5FrameReader.frame(&sink, 0).unwrap_err(),
        Hdf5FrameError::UnknownResource {
            mimetype: "application/x-hdf5".into(),
            parameters: vec!["path".into()],
        }
    );
    assert!(!not_yet(&soft, 0));
    assert!(!not_yet(&sink, 0));
}

/// A directory uri of another host, as bsrs's own tests give one
/// (`host/areadetector.rs:3306`), and one of another scheme.
#[test]
fn a_uri_off_this_host_is_not_local() {
    for uri in [
        "file://nfs-share/something/scan.h5",
        "http://files.example/scans/scan.h5",
    ] {
        let resource = ad_resource(uri.into(), DATASET, &[1, 4, 6]);
        assert_eq!(
            Hdf5FrameReader.frame(&resource, 0).unwrap_err(),
            Hdf5FrameError::NotLocal { uri: uri.into() }
        );
        assert!(!not_yet(&resource, 0));
    }
}

#[test]
fn a_file_not_written_is_missing() {
    let dir = scratch("a_file_not_written_is_missing");
    assert_eq!(
        Hdf5FrameReader.frame(&image_resource(&dir), 0).unwrap_err(),
        Hdf5FrameError::MissingFile {
            path: dir.join("scan.h5")
        }
    );
    assert!(not_yet(&image_resource(&dir), 0));
}

/// The resource AdHdfWriter emits for an NDAttribute: a 1-D dataset of one
/// scalar per frame (`host/areadetector.rs:1959-1971`).
#[test]
fn an_attribute_dataset_is_not_frames() {
    let dir = scratch("an_attribute_dataset_is_not_frames");
    let writer = stream(
        &dir,
        "/entry/instrument/NDAttributes",
        "I0",
        &[],
        &[vec![1.5f64], vec![2.5]],
    );
    let resource = ad_resource(uri(&dir), "/entry/instrument/NDAttributes/I0", &[16384]);
    let e = Hdf5FrameReader.frame(&resource, 0).unwrap_err();
    assert!(!not_yet(&resource, 0));
    writer.close();
    assert_eq!(e, Hdf5FrameError::NotFrames { shape: vec![2] });
}

/// A 64-bit count past 2^53, which f64 would round to its neighbour.
#[test]
fn an_integer_pixel_past_2_pow_53_is_inexact() {
    let dir = scratch("an_integer_pixel_past_2_pow_53_is_inexact");
    let value = (1u64 << 53) + 1;
    let writer = stream(&dir, "/entry/data", "data", &[1, 2], &[vec![7u64, value]]);
    let resource = ad_resource(uri(&dir), DATASET, &[1, 1, 2]);
    let e = Hdf5FrameReader.frame(&resource, 0).unwrap_err();
    assert!(!not_yet(&resource, 0));
    writer.close();
    assert_eq!(
        e,
        Hdf5FrameError::Inexact {
            pixel: 1,
            value: i128::from(value)
        }
    );
}

/// A compound pixel, rust-hdf5's complex number.
#[test]
fn a_compound_pixel_is_an_unsupported_type() {
    let dir = scratch("a_compound_pixel_is_an_unsupported_type");
    let frame = vec![Complex32 { re: 1.0, im: 2.0 }; SHAPE.0 * SHAPE.1];
    let writer = stream(
        &dir,
        "/entry/data",
        "data",
        &[SHAPE.0 as u64, SHAPE.1 as u64],
        &[frame],
    );
    let e = Hdf5FrameReader.frame(&image_resource(&dir), 0).unwrap_err();
    assert!(!not_yet(&image_resource(&dir), 0));
    writer.close();
    assert!(matches!(e, Hdf5FrameError::UnsupportedType(_)), "{e:?}");
}

/// A file at the uri that is not HDF5.
#[test]
fn a_file_that_is_not_hdf5_fails_in_hdf5() {
    let dir = scratch("a_file_that_is_not_hdf5_fails_in_hdf5");
    std::fs::write(dir.join("scan.h5"), b"not an HDF5 file").unwrap();
    let e = Hdf5FrameReader.frame(&image_resource(&dir), 0).unwrap_err();
    assert!(matches!(e, Hdf5FrameError::Hdf5(_)), "{e:?}");
    assert!(!not_yet(&image_resource(&dir), 0));
}

// -- the compressions NDFileHDF5 writes ---------------------------------------

/// Dataset `/{name}/data` of golden/ndfilehdf5_filters.h5, which
/// golden/ndfilehdf5_filters.py writes: two frames of 64 x 48, a chunk each.
fn compressed(name: &str) -> StreamResource {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../golden/ndfilehdf5_filters.h5")
        .canonicalize()
        .unwrap();
    ad_resource(
        format!("file://localhost{}", path.display()),
        &format!("/{name}/data"),
        &[1, 64, 48],
    )
}

/// Frame `f` of every dataset there: pixel (r, c) is
/// (101 f + 37 (r / 4) + 11 (c / 3)) % 4096.
fn compressed_pixels(f: usize) -> Vec<f64> {
    (0..64)
        .flat_map(|r| (0..48).map(move |c| ((101 * f + 37 * (r / 4) + 11 * (c / 3)) % 4096) as f64))
        .collect()
}

/// Both frames of each dataset read as written.
fn reads_as_written(names: &[&str]) {
    for name in names {
        for f in 0..2 {
            let frame = Hdf5FrameReader
                .frame(&compressed(name), f as u64)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(frame.shape(), (64, 48), "{name}");
            assert_eq!(frame.data(), compressed_pixels(f), "{name} frame {f}");
        }
    }
}

/// NDFileHDF5's N-bit, on a 12-bit integer type at bit offset 0 and at 4,
/// and at the type's full 16 bits, which the filter passes through.
#[test]
fn nbit_is_read() {
    reads_as_written(&["nbit", "nbit_offset4", "nbit_full"]);
}

/// NDFileHDF5's szip (nearest neighbour), and szip with entropy coding.
#[test]
fn szip_is_read() {
    reads_as_written(&["szip", "szip_ec"]);
}

#[test]
fn deflate_is_read() {
    reads_as_written(&["deflate"]);
}

/// Each of blosc's six compressors with byte shuffle, and lz4 without
/// shuffle and with bit shuffle.
#[test]
fn blosc_is_read_with_each_compressor() {
    reads_as_written(&[
        "blosc_blosclz",
        "blosc_lz4",
        "blosc_lz4hc",
        "blosc_snappy",
        "blosc_zlib",
        "blosc_zstd",
        "blosc_lz4_noshuffle",
        "blosc_lz4_bitshuffle",
    ]);
}

#[test]
fn bitshuffle_lz4_is_read() {
    reads_as_written(&["bshuf_lz4"]);
}

#[test]
fn lz4_is_read() {
    reads_as_written(&["lz4"]);
}

/// The filters rust-hdf5 decodes that NDFileHDF5 does not write, read
/// as before.
#[test]
fn the_other_filters_rust_hdf5_decodes_are_read() {
    reads_as_written(&[
        "shuffle_deflate",
        "fletcher32",
        "scaleoffset",
        "lzf",
        "zstd",
        "bshuf_none",
        "bitgroom",
        "bitround",
    ]);
}

/// NDFileHDF5's JPEG, and bitshuffle with zstd and bzip2, which it does
/// not write: refused by name, for good.
#[test]
fn a_filter_not_decoded_is_named() {
    for (name, id, filter) in [
        ("jpeg", 32019, "jpeg"),
        ("bshuf_zstd", 32008, "bitshuffle with zstd"),
        ("bzip2", 307, "bzip2"),
    ] {
        let e = Hdf5FrameReader.frame(&compressed(name), 0).unwrap_err();
        assert_eq!(
            e,
            Hdf5FrameError::UnsupportedFilter {
                id,
                name: filter.into()
            }
        );
        assert_eq!(
            e.to_string(),
            format!("HDF5 filter {id} ({filter}) is not one this reader decodes")
        );
        assert!(!not_yet(&compressed(name), 0));
    }
}
