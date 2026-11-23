//! RsmCallback against areaDetector's C simDetector IOC, whose NDFileHDF5
//! plugin streams the frames in SWMR mode while the scan runs: every frame
//! is mapped live from the growing file before the IOC closes it, and that
//! map is the one the closed file gives.
//!
//! `tests/ioc/simdetector-ioc.sh` builds and runs the IOC; its `test`
//! subcommand runs these with the Channel Access environment that finds it.
//! The IOC's PV prefix is `HKLRS_IOC_PREFIX`. The tests share the one IOC,
//! so each holds a lock file for its whole run.
//!
//! The cam runs simDetector's LinearRamp with Gain 1, GainX 1, GainY 16 and
//! no background: pixel (i, j) is `j + 16 i` plus the frames since the
//! reset (`simDetector.cpp:132-133`, 199-204), which every frame of the
//! closed file is checked against, apart from any HDF5 reader.
#![cfg(feature = "ioc")]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use bsrs::backends::epics_ca::EpicsCaBackend;
use bsrs::backends::soft::SoftMotor;
use bsrs::core::msg::{MovableObj, ReadableObj, StageableObj, TriggerableObj};
use bsrs::core::status::Status;
use bsrs::core::{Msg, plan_box};
use bsrs::devices::{BackendFromPv, StaticFilenameProvider, StaticPathProvider, TriggerInfo};
use bsrs::engine::{DocumentCallback, RunEngine};
use bsrs::event_model::{
    DataKey, Document, Dtype, Event, EventDescriptor, RunStart, RunStop, StreamDatum, StreamRange,
    StreamResource,
};
use bsrs::host::areadetector::{
    AD_COLOR_MODE_MONO, AD_IMAGE_MODE_SINGLE, AreaDetectorHdf, NdFileHdf5, area_detector_hdf,
};
use bsrs::protocols_async::{
    DetectorControl, Flyable, Preparable, SignalBackend, Stageable, StreamAsset, WritesStreamAssets,
};
use futures::StreamExt;
use hklrs::frame_map::FrameMap;
use hklrs::grid::Binning;
use hklrs::nalgebra::Vector3;
use hklrs::normalise::{PixelFields, Region};
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{Image, PixelDetector, Wavelength};
use hklrs_bsrs::{
    FrameReader, Hdf5FrameReader, ReadError, RsmCallback, RsmConfig, RsmMap, RsmValue,
};
use serde_json::{Value, json};
use tokio::time::timeout;

const TIMEOUT: Duration = Duration::from_secs(10);
/// The detector's name, which is its image data key.
const IMAGE: &str = "det";
/// simDetectorConfig's port in `tests/ioc/st.cmd`.
const PORT: &str = "SIM1";
const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const EXPOSURE: f64 = 0.01;
const LAMBDA: f64 = 1.54;
/// delta, read from a motor that stays put.
const DELTA: f64 = 25.0;
/// `AD_HDF_DATASET`, `host/areadetector.rs:1298`.
const DATASET: &str = "/entry/data/data";

/// komega at frame `index`.
fn komega(index: u64) -> f64 {
    18.0 + 0.5 * index as f64
}

// -- the IOC ------------------------------------------------------------------

/// The IOC of `HKLRS_IOC_PREFIX`, held by this test until dropped.
struct Ioc {
    cam: String,
    hdf: String,
    _lock: File,
}

impl Ioc {
    fn lock() -> Self {
        let prefix = std::env::var("HKLRS_IOC_PREFIX").expect(
            "HKLRS_IOC_PREFIX, the simDetector IOC's PV prefix: see tests/ioc/simdetector-ioc.sh",
        );
        let name: String = prefix
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(std::env::temp_dir().join(format!("hklrs-rsm-ioc-{name}.lock")))
            .unwrap();
        lock.lock().unwrap();
        Self {
            cam: format!("{prefix}cam1:"),
            hdf: format!("{prefix}HDF1:"),
            _lock: lock,
        }
    }
}

async fn channel<V>(pv: &str) -> EpicsCaBackend<V>
where
    V: Clone + Send + Sync + 'static,
    EpicsCaBackend<V>: SignalBackend<V> + BackendFromPv,
{
    let ch = EpicsCaBackend::<V>::from_pv(pv);
    ch.connect(TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("connect {pv}: {e}"));
    ch
}

async fn put<V>(pv: &str, v: V)
where
    V: Clone + Send + Sync + 'static,
    EpicsCaBackend<V>: SignalBackend<V> + BackendFromPv,
{
    let ch = channel::<V>(pv).await;
    timeout(TIMEOUT, SignalBackend::put(&ch, Some(v)))
        .await
        .unwrap_or_else(|_| panic!("put {pv} timed out"))
        .unwrap_or_else(|e| panic!("put {pv}: {e}"));
}

async fn get<V>(pv: &str) -> V
where
    V: Clone + Send + Sync + 'static,
    EpicsCaBackend<V>: SignalBackend<V> + BackendFromPv,
{
    let ch = channel::<V>(pv).await;
    SignalBackend::get_value(&ch)
        .await
        .unwrap_or_else(|e| panic!("get {pv}: {e}"))
}

async fn plugin(ioc: &Ioc) -> NdFileHdf5 {
    let hdf = NdFileHdf5::new(&ioc.hdf);
    hdf.connect(TIMEOUT).await.expect("connect HDF1");
    hdf
}

/// NDFileHDF5's compressions and their settings.
#[derive(Clone, Copy, Debug)]
enum Compression {
    None,
    /// NumDataBits, at DataBitsOffset 0.
    NBit(i64),
    Szip,
    /// ZLevel.
    Zlib(i64),
    /// BloscCompressor and BloscShuffle.
    Blosc(&'static str, &'static str),
    Bslz4,
    Lz4,
}

impl Compression {
    /// Its `Compression` choice, `NDFileHDF5.template:585-607`.
    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::NBit(_) => "N-bit",
            Self::Szip => "szip",
            Self::Zlib(_) => "zlib",
            Self::Blosc(..) => "Blosc",
            Self::Bslz4 => "BSLZ4",
            Self::Lz4 => "LZ4",
        }
    }
}

/// `det` writing `scan.h5` in `dir` with `chunk` frames to a chunk, cam and
/// plugin brought to a known state whatever an earlier run left, as bsrs's
/// own `sim_detector_ioc.rs` `ready_cam` does, the ramp reset and one
/// warmup frame taken.
async fn detector(
    ioc: &Ioc,
    dir: &Path,
    chunk: i64,
    compression: Compression,
) -> Arc<AreaDetectorHdf> {
    let provider = StaticPathProvider::new(Arc::new(StaticFilenameProvider::new("scan")), dir);
    let det = area_detector_hdf(IMAGE, &ioc.cam, &ioc.hdf, Arc::new(provider));
    det.connect(TIMEOUT).await.expect("connect");
    let cam = det.control();
    DetectorControl::disarm(cam).await.expect("disarm");
    let hdf = plugin(ioc).await;
    hdf.file.stop_capture().await.expect("stop capture");
    hdf.file
        .plugin
        .set_source_port(PORT)
        .await
        .expect("NDArrayPort");
    hdf.file
        .plugin
        .set_blocking(true)
        .await
        .expect("BlockingCallbacks");
    hdf.file
        .plugin
        .set_enabled(true)
        .await
        .expect("EnableCallbacks");

    let at = |prefix: &str, suffix: &str| format!("{prefix}{suffix}");
    for (suffix, v) in [
        ("NumImages", 1),
        ("NumExposures", 1),
        ("BinX", 1),
        ("BinY", 1),
        ("MinX", 0),
        ("MinY", 0),
        ("SizeX", WIDTH as i64),
        ("SizeY", HEIGHT as i64),
    ] {
        put::<i64>(&at(&ioc.cam, suffix), v).await;
    }
    for (suffix, label) in [
        ("ImageMode", AD_IMAGE_MODE_SINGLE),
        ("ColorMode", AD_COLOR_MODE_MONO),
        ("DataType", "UInt16"),
        ("SimMode", "LinearRamp"),
    ] {
        put::<String>(&at(&ioc.cam, suffix), label.into()).await;
    }
    put::<bool>(&at(&ioc.cam, "ArrayCallbacks"), true).await;
    // Gain, like every sim parameter, resets the ramp (`simDetector.cpp:974`).
    for (suffix, v) in [
        ("AcquireTime", EXPOSURE),
        ("AcquirePeriod", EXPOSURE),
        ("GainX", 1.0),
        ("GainY", 16.0),
        ("Noise", 0.0),
        ("Offset", 0.0),
        ("Gain", 1.0),
    ] {
        put::<f64>(&at(&ioc.cam, suffix), v).await;
    }
    timeout(TIMEOUT, async {
        cam.nd_attributes_file.put(String::new()).await.await
    })
    .await
    .expect("NDAttributesFile put timed out")
    .expect("clear NDAttributesFile");
    cam.warmup().await.expect("warmup");

    // The plugin flushes every `chunk` frames by itself; NumFramesFlush
    // below NumFramesChunks is refused (`NDFileHDF5.cpp:1860`).
    put::<i64>(&at(&ioc.hdf, "NumFramesChunks"), chunk).await;
    put::<i64>(&at(&ioc.hdf, "NumFramesFlush"), chunk).await;
    put::<String>(&at(&ioc.hdf, "Compression"), compression.label().into()).await;
    match compression {
        Compression::NBit(bits) => {
            put::<i64>(&at(&ioc.hdf, "NumDataBits"), bits).await;
            put::<i64>(&at(&ioc.hdf, "DataBitsOffset"), 0).await;
        }
        Compression::Zlib(level) => put::<i64>(&at(&ioc.hdf, "ZLevel"), level).await,
        Compression::Blosc(compressor, shuffle) => {
            put::<String>(&at(&ioc.hdf, "BloscCompressor"), compressor.into()).await;
            put::<String>(&at(&ioc.hdf, "BloscShuffle"), shuffle.into()).await;
            put::<i64>(&at(&ioc.hdf, "BloscLevel"), 5).await;
        }
        Compression::None | Compression::Szip | Compression::Bslz4 | Compression::Lz4 => {}
    }
    Arc::new(det)
}

async fn finish(status: Status, what: &str) {
    match timeout(TIMEOUT, status).await {
        Ok(r) => r.unwrap_or_else(|e| panic!("{what}: {e:?}")),
        Err(_) => panic!("{what}: not done within {TIMEOUT:?}"),
    }
}

async fn drain(det: &AreaDetectorHdf) -> (Vec<StreamResource>, Vec<StreamDatum>) {
    let up_to = det.get_index().await.expect("index");
    let mut resources = Vec::new();
    let mut datums = Vec::new();
    let mut docs = det.collect_asset_docs(up_to, "desc").expect("collect");
    while let Some(doc) = docs.next().await {
        match doc {
            StreamAsset::Resource(r) => resources.push(r),
            StreamAsset::Datum(d) => datums.push(d),
        }
    }
    (resources, datums)
}

/// An empty directory of its own for each test, which the IOC writes in.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("rsm_ioc")
        .join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// -- mapping ------------------------------------------------------------------

fn frame_map() -> FrameMap {
    let detector = PixelDetector::new(
        (HEIGHT, WIDTH),
        Vector3::new(0.5, -0.024, -0.032),
        Vector3::new(0.0, 0.0, 0.001),
        Vector3::new(0.0, 0.001, 0.0),
    )
    .unwrap();
    FrameMap::new(
        preset("K6C").unwrap(),
        Projection::QSample,
        &detector,
        &Region::new(0..HEIGHT, 0..WIDTH, (2, 2)),
        PixelFields::new(),
        vec![Binning::new(0.01, 0.0).unwrap(); 3],
    )
    .unwrap()
}

fn rsm(reader: impl FrameReader + 'static, read_deadline: Duration) -> Arc<RsmCallback> {
    let field = |name: &str| RsmValue::Field(name.to_owned());
    let config = RsmConfig {
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
    };
    Arc::new(RsmCallback::new(config).unwrap())
}

/// The map of `docs` read from the closed file, each frame read once.
fn closed_map(docs: &[Document]) -> RsmMap {
    let rsm = rsm(Hdf5FrameReader, Duration::ZERO);
    for doc in docs {
        rsm.document(doc);
    }
    rsm.finish().unwrap()
}

/// The constant `image` adds to the ramp `j + 16 i`, or why it is no ramp.
fn ramp(image: &Image<f64>) -> Result<i64, String> {
    if image.shape() != (HEIGHT, WIDTH) {
        return Err(format!("shape {:?}", image.shape()));
    }
    let data = image.data();
    let c = data[0];
    for (p, &v) in data.iter().enumerate() {
        let (i, j) = (p / WIDTH, p % WIDTH);
        if v - (j + 16 * i) as f64 != c {
            return Err(format!("pixel ({i}, {j}) is {v}, pixel (0, 0) {c}"));
        }
    }
    Ok(c as i64)
}

/// The closed file's frames `0..frames`: each a ramp, each one frame on from
/// the one before.
fn closed_ramps(resource: &StreamResource, frames: u64) -> Vec<i64> {
    let ramps: Vec<i64> = (0..frames)
        .map(|k| {
            let image = Hdf5FrameReader.frame(resource, k).unwrap();
            ramp(&image).unwrap_or_else(|e| panic!("closed frame {k}: {e}"))
        })
        .collect();
    for (k, &c) in ramps.iter().enumerate() {
        assert_eq!(c, ramps[0] + k as i64, "closed frame {k}");
    }
    ramps
}

/// The resource AdHdfWriter emits for `h5`, for reads before it is emitted.
fn file_resource(h5: &Path) -> StreamResource {
    StreamResource {
        uid: "file".into(),
        data_key: IMAGE.into(),
        mimetype: "application/x-hdf5".into(),
        uri: format!("file://localhost{}", h5.display()),
        parameters: HashMap::from([("dataset".to_owned(), json!(DATASET))]),
        run_start: None,
    }
}

/// `text` with each run of digits as `#`, so reasons that differ only in an
/// address or a count are counted together.
fn squash(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        if !c.is_ascii_digit() {
            out.push(c);
        } else if !out.ends_with('#') {
            out.push('#');
        }
    }
    out
}

// -- readers ------------------------------------------------------------------

/// Hdf5FrameReader that reports each read that fails: the frame, whether
/// not readable yet, and why.
struct Watched(mpsc::Sender<(u64, bool, String)>);

impl FrameReader for Watched {
    fn read(&self, resource: &StreamResource, index: u64) -> Result<Image<f64>, ReadError> {
        let read = Hdf5FrameReader.read(resource, index);
        if let Err(e) = &read {
            let not_yet = matches!(e, ReadError::NotYet(_));
            let _ = self.0.send((index, not_yet, e.to_string()));
        }
        read
    }
}

/// Every frame of the file read over and over on a thread of its own while
/// the IOC creates, writes and flushes it: an RsmCallback worker's reads
/// without its pauses.
struct Hammer {
    stop: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<Hammered>,
}

#[derive(Debug, Default)]
struct Hammered {
    reads: usize,
    /// Per frame, the ramp constants its reads found.
    found: BTreeMap<u64, BTreeSet<i64>>,
    /// Frames read that are no ramp.
    torn: Vec<String>,
    /// The reasons reads failed with, digits as `#`, and how often.
    not_yet: BTreeMap<String, usize>,
    permanent: BTreeMap<String, usize>,
}

impl Hammer {
    fn start(h5: &Path, frames: u64) -> Self {
        let resource = file_resource(h5);
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let thread = std::thread::spawn(move || {
            let mut out = Hammered::default();
            while !running.load(Ordering::Relaxed) {
                for index in 0..frames {
                    out.reads += 1;
                    match Hdf5FrameReader.read(&resource, index) {
                        Ok(image) => match ramp(&image) {
                            Ok(c) => {
                                out.found.entry(index).or_default().insert(c);
                            }
                            Err(e) => out.torn.push(format!("frame {index}: {e}")),
                        },
                        Err(ReadError::NotYet(e)) => {
                            *out.not_yet.entry(squash(&e.to_string())).or_default() += 1;
                        }
                        Err(ReadError::Permanent(e)) => {
                            *out.permanent.entry(squash(&e.to_string())).or_default() += 1;
                        }
                    }
                }
            }
            out
        });
        Self { stop, thread }
    }

    fn stop(self) -> Hammered {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.join().unwrap()
    }
}

/// FlushNow put over and over, each waiting for its flush, while frames
/// are written.
struct Flusher {
    stop: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<(usize, usize)>,
}

impl Flusher {
    async fn start(ioc: &Ioc) -> Self {
        let hdf = plugin(ioc).await;
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let task = tokio::spawn(async move {
            let (mut done, mut refused) = (0, 0);
            while !running.load(Ordering::Relaxed) {
                match timeout(TIMEOUT, hdf.flush()).await {
                    Ok(Ok(())) => done += 1,
                    Ok(Err(_)) => refused += 1,
                    Err(_) => panic!("FlushNow not done within {TIMEOUT:?}"),
                }
            }
            (done, refused)
        });
        Self { stop, task }
    }

    /// Flushes done and refused.
    async fn stop(self) -> (usize, usize) {
        self.stop.store(true, Ordering::Relaxed);
        self.task.await.unwrap()
    }
}

// -- outcome ------------------------------------------------------------------

/// What held while the IOC still had the file open: `live`'s map once
/// every frame was mapped or failed, and whether the plugin wrote in SWMR
/// mode.
#[derive(Debug)]
struct WhileOpen {
    map: RsmMap,
    swmr_active: bool,
}

async fn while_open(live: &RsmCallback, frames: u64, hdf: &str) -> WhileOpen {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let map = live.snapshot().unwrap();
        if map.mapped.len() + map.failed.len() == frames as usize || Instant::now() > deadline {
            let swmr_active = get::<bool>(&format!("{hdf}SWMRActive_RBV")).await;
            return WhileOpen { map, swmr_active };
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

struct Outcome {
    frames: u64,
    live: RsmMap,
    while_open: WhileOpen,
    closed: RsmMap,
    ramps: Vec<i64>,
    hammered: Hammered,
}

impl Outcome {
    /// Every frame mapped live, all before the file was closed, into the
    /// map the closed file gives; every frame the hammer read whole was the
    /// closed file's frame, and none failed for good.
    fn check(&self) {
        let h = &self.hammered;
        eprintln!(
            "ramps {:?}; hammer: {} reads, frames found {:?}, not yet {:?}, failed {:?}, torn {:?}",
            self.ramps,
            h.reads,
            h.found.keys().collect::<Vec<_>>(),
            h.not_yet,
            h.permanent,
            h.torn,
        );
        let all: Vec<u64> = (1..=self.frames).collect();
        assert_eq!(self.closed.failed, []);
        assert_eq!(self.closed.mapped, all);
        assert_eq!(self.live.failed, []);
        assert_eq!(self.live.mapped, all);
        assert!(self.while_open.swmr_active, "SWMRActive_RBV off");
        assert_eq!(self.while_open.map.mapped, all, "mapped while open");
        assert!(
            self.while_open.map == self.live,
            "the map grew after the file closed"
        );
        assert!(
            self.live == self.closed,
            "the live map is not the closed file's"
        );
        assert_eq!(h.torn, Vec::<String>::new());
        assert_eq!(h.permanent, BTreeMap::new());
        for (index, found) in &h.found {
            let closed = BTreeSet::from([self.ramps[*index as usize]]);
            assert_eq!(found, &closed, "frame {index} read live");
        }
    }
}

// -- documents ------------------------------------------------------------------

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

/// A fly scan's documents for `frames` frames of `resource`: start,
/// descriptor, the resource, one datum for them all and their events.
fn fly_docs(resource: StreamResource, frames: u64) -> Vec<Document> {
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
        Document::StreamDatum(StreamDatum {
            uid: "datum".into(),
            stream_resource: resource.uid.clone(),
            descriptor: "desc".into(),
            indices: StreamRange {
                start: 0,
                stop: frames,
            },
            seq_nums: StreamRange {
                start: 1,
                stop: frames + 1,
            },
        }),
    ];
    docs.insert(2, Document::StreamResource(resource));
    for k in 0..frames {
        let data: HashMap<String, Value> = HashMap::from([
            ("komega".to_owned(), json!(komega(k))),
            ("delta".to_owned(), json!(DELTA)),
        ]);
        docs.push(Document::Event(Event {
            uid: format!("event-{}", k + 1),
            descriptor: "desc".into(),
            time: 0.0,
            seq_num: k + 1,
            data,
            timestamps: HashMap::new(),
            filled: HashMap::new(),
        }));
    }
    docs
}

fn stop() -> Document {
    Document::Stop(RunStop {
        uid: "stop".into(),
        run_start: "run".into(),
        ..Default::default()
    })
}

// -- runs ---------------------------------------------------------------------

/// A RunEngine step scan of komega over `frames` points that triggers the
/// detector at each, the live map subscribed; it unstages, which closes the
/// file, only once the live map has every frame.
async fn step_scan(test: &str, frames: u64, chunk: i64, compression: Compression) -> Outcome {
    let ioc = Ioc::lock();
    let dir = scratch(test);
    let det = detector(&ioc, &dir, chunk, compression).await;
    let hammer = Hammer::start(&dir.join("scan.h5"), frames);
    let live = rsm(Hdf5FrameReader, RsmConfig::DEFAULT_READ_DEADLINE);
    let docs = Arc::new(Mutex::new(Vec::new()));
    let capture: DocumentCallback = {
        let docs = docs.clone();
        Arc::new(move |doc: &Document| docs.lock().unwrap().push(doc.clone()))
    };

    let motor = Arc::new(SoftMotor::new("komega", Some(0.0)));
    let delta = Arc::new(SoftMotor::new("delta", Some(DELTA)));
    let mut msgs = vec![
        Msg::OpenRun(Default::default()),
        Msg::Stage(det.clone() as Arc<dyn StageableObj>),
    ];
    let wait = |group: &str| Msg::Wait {
        group: group.into(),
        error_on_timeout: true,
        timeout: None,
    };
    for k in 0..frames {
        msgs.extend([
            Msg::Set {
                obj: motor.clone() as Arc<dyn MovableObj>,
                value: komega(k),
                group: Some("move".into()),
            },
            wait("move"),
            Msg::Trigger {
                obj: det.clone() as Arc<dyn TriggerableObj>,
                group: Some("trigger".into()),
            },
            wait("trigger"),
            Msg::Create {
                stream_name: "primary".into(),
            },
            Msg::Read(motor.clone() as Arc<dyn ReadableObj>),
            Msg::Read(delta.clone() as Arc<dyn ReadableObj>),
            Msg::Read(det.clone() as Arc<dyn ReadableObj>),
            Msg::Save,
        ]);
    }
    let open = Arc::new(Mutex::new(None));
    let unstage = {
        let (live, det, open, hdf) = (live.clone(), det.clone(), open.clone(), ioc.hdf.clone());
        async move {
            let at_close = while_open(&live, frames, &hdf).await;
            *open.lock().unwrap() = Some(at_close);
            Msg::Unstage(det as Arc<dyn StageableObj>)
        }
    };
    let plan = futures::stream::iter(msgs)
        .chain(futures::stream::once(unstage))
        .chain(futures::stream::iter([Msg::CloseRun {
            exit_status: "success".into(),
            reason: None,
        }]));
    let re = RunEngine::new(vec![]);
    re.subscribe(live.callback());
    re.subscribe(capture);
    let result = re.run_async(plan_box(plan)).await.unwrap();
    assert_eq!(result.exit_status, "success", "{result:?}");

    let live = live.finish().unwrap();
    let hammered = hammer.stop();
    let docs = docs.lock().unwrap().clone();
    let resource = docs
        .iter()
        .find_map(|d| match d {
            Document::StreamResource(r) => Some(r.clone()),
            _ => None,
        })
        .expect("a resource");
    Outcome {
        frames,
        live,
        while_open: open.lock().unwrap().take().expect("unstaged"),
        closed: closed_map(&docs),
        ramps: closed_ramps(&resource, frames),
        hammered,
    }
}

/// A fly scan of `frames` frames, `livetime` each, whose datum and events
/// the live map has before the first frame, so before the file exists;
/// FlushNow put over and over while the frames are written when `flush_now`.
async fn fly_scan(
    test: &str,
    frames: u64,
    livetime: Duration,
    chunk: i64,
    compression: Compression,
    flush_now: bool,
) -> Outcome {
    let ioc = Ioc::lock();
    let dir = scratch(test);
    let h5 = dir.join("scan.h5");
    let det = detector(&ioc, &dir, chunk, compression).await;
    let hammer = Hammer::start(&h5, frames);
    Stageable::stage(&*det).await.expect("stage");
    let info = TriggerInfo {
        number_of_events: u32::try_from(frames).unwrap(),
        livetime: Some(livetime),
        deadtime: Some(Duration::from_millis(10)),
        ..TriggerInfo::default()
    };
    finish(Preparable::prepare(&*det, info).await, "prepare").await;
    // LazyOpen: the plugin creates the file at its first frame.
    assert!(!h5.exists(), "{} before the first frame", h5.display());
    let (resources, datums) = drain(&det).await;
    assert_eq!((resources.len(), datums.len()), (1, 0));
    let resource = resources.into_iter().next().unwrap();

    let (reads, failed) = mpsc::channel();
    let live = rsm(Watched(reads), Duration::from_secs(30));
    let docs = fly_docs(resource, frames);
    for doc in &docs {
        live.document(doc);
    }
    for _ in 0..3 {
        let (index, not_yet, why) = failed
            .recv_timeout(TIMEOUT)
            .expect("a read of frame 0 before the file exists");
        assert_eq!(index, 0);
        assert!(not_yet && why.contains("does not exist"), "{why}");
    }
    assert!(!h5.exists(), "{} before kickoff", h5.display());

    let flusher = match flush_now {
        true => Some(Flusher::start(&ioc).await),
        false => None,
    };
    finish(Flyable::kickoff(&*det).await, "kickoff").await;
    finish(Flyable::complete(&*det).await, "complete").await;
    if let Some(flusher) = flusher {
        let (done, refused) = flusher.stop().await;
        eprintln!("FlushNow: {done} done, {refused} refused");
        assert!(done > 0, "no FlushNow done while the frames were written");
    }
    let at_close = while_open(&live, frames, &ioc.hdf).await;
    let (resources, datums) = drain(&det).await;
    assert_eq!(resources.len(), 0);
    let indices: Vec<(u64, u64)> = datums
        .iter()
        .map(|d| (d.indices.start, d.indices.stop))
        .collect();
    assert_eq!(indices, [(0, frames)]);
    Stageable::unstage(&*det).await.expect("unstage");

    live.document(&stop());
    let live = live.finish().unwrap();
    let hammered = hammer.stop();
    let mut live_failures: BTreeMap<String, usize> = BTreeMap::new();
    for (_, _, why) in failed.try_iter() {
        *live_failures.entry(squash(&why)).or_default() += 1;
    }
    eprintln!("live reads failed after the first 3: {live_failures:?}");
    let resource = match &docs[2] {
        Document::StreamResource(r) => r.clone(),
        other => panic!("{other:?}"),
    };
    let mut docs = docs;
    docs.push(stop());
    Outcome {
        frames,
        live,
        while_open: at_close,
        closed: closed_map(&docs),
        ramps: closed_ramps(&resource, frames),
        hammered,
    }
}

// -- tests --------------------------------------------------------------------

/// A step scan on soft motors through bsrs's RunEngine: each frame is read
/// after bsrs flushes the file and sends its datum.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn a_step_scan_maps_live_as_from_the_closed_file() {
    step_scan(
        "a_step_scan_maps_live_as_from_the_closed_file",
        8,
        1,
        Compression::None,
    )
    .await
    .check();
}

/// The first frame's datum before the plugin has created the file: its
/// reads find no file, then the frame once the IOC writes it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn a_frame_announced_before_the_file_exists_is_read_once_written() {
    fly_scan(
        "a_frame_announced_before_the_file_exists_is_read_once_written",
        5,
        Duration::from_millis(200),
        1,
        Compression::None,
        false,
    )
    .await
    .check();
}

/// Frames read as the plugin writes them four to a chunk while FlushNow
/// flushes partly filled chunks between them.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn frames_read_while_the_ioc_flushes_map_as_from_the_closed_file() {
    fly_scan(
        "frames_read_while_the_ioc_flushes_map_as_from_the_closed_file",
        40,
        Duration::from_millis(40),
        4,
        Compression::None,
        true,
    )
    .await
    .check();
}

/// NDFileHDF5 makes its dataset one frame long and writes frame 0 into its
/// chunk cache (`NDFileHDF5Dataset.cpp:89`, 125): with no flush due, the
/// file holds frame 0 in its extent but not its chunk until FlushNow.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn a_frame_the_ioc_has_not_flushed_is_not_read() {
    let ioc = Ioc::lock();
    let dir = scratch("a_frame_the_ioc_has_not_flushed_is_not_read");
    let resource = file_resource(&dir.join("scan.h5"));
    let det = detector(&ioc, &dir, 1, Compression::None).await;
    put::<i64>(&format!("{}NumFramesFlush", ioc.hdf), 1000).await;
    Stageable::stage(&*det).await.expect("stage");
    let info = TriggerInfo {
        number_of_events: 2,
        livetime: Some(Duration::from_secs(2)),
        deadtime: Some(Duration::from_millis(10)),
        ..TriggerInfo::default()
    };
    finish(Preparable::prepare(&*det, info).await, "prepare").await;
    finish(Flyable::kickoff(&*det).await, "kickoff").await;
    // NDPluginFile counts a streamed frame once writeFile has returned.
    let captured = format!("{}NumCaptured_RBV", ioc.hdf);
    let deadline = Instant::now() + TIMEOUT;
    while get::<i64>(&captured).await < 1 {
        assert!(Instant::now() < deadline, "frame 0 not written");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(get::<bool>(&format!("{}SWMRActive_RBV", ioc.hdf)).await);
    match Hdf5FrameReader.read(&resource, 0) {
        Err(ReadError::NotYet(e)) => {
            assert!(e.to_string().contains("has not been written"), "{e}")
        }
        Err(ReadError::Permanent(e)) => panic!("frame 0 before FlushNow failed: {e}"),
        Ok(image) => panic!("frame 0 before FlushNow read: {:?}", ramp(&image)),
    }
    timeout(TIMEOUT, plugin(&ioc).await.flush())
        .await
        .expect("FlushNow timed out")
        .expect("FlushNow");
    let flushed = Hdf5FrameReader
        .read(&resource, 0)
        .expect("frame 0 after FlushNow");
    finish(Flyable::complete(&*det).await, "complete").await;
    Stageable::unstage(&*det).await.expect("unstage");
    assert_eq!(ramp(&flushed), Ok(closed_ramps(&resource, 2)[0]));
}

/// Compressed frames, two to a chunk, read while FlushNow writes partly
/// filled chunks the next frame compresses again.
async fn compressed(test: &str, compression: Compression) {
    fly_scan(test, 16, Duration::from_millis(40), 2, compression, true)
        .await
        .check();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn nbit_frames_map_live_as_from_the_closed_file() {
    compressed(
        "nbit_frames_map_live_as_from_the_closed_file",
        Compression::NBit(12),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn zlib_frames_map_live_as_from_the_closed_file() {
    compressed(
        "zlib_frames_map_live_as_from_the_closed_file",
        Compression::Zlib(6),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn blosc_frames_map_live_as_from_the_closed_file() {
    compressed(
        "blosc_frames_map_live_as_from_the_closed_file",
        Compression::Blosc("LZ4", "Bit"),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn bslz4_frames_map_live_as_from_the_closed_file() {
    compressed(
        "bslz4_frames_map_live_as_from_the_closed_file",
        Compression::Bslz4,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn lz4_frames_map_live_as_from_the_closed_file() {
    compressed(
        "lz4_frames_map_live_as_from_the_closed_file",
        Compression::Lz4,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the simDetector IOC; see tests/ioc/simdetector-ioc.sh"]
async fn szip_frames_map_live_as_from_the_closed_file() {
    compressed(
        "szip_frames_map_live_as_from_the_closed_file",
        Compression::Szip,
    )
    .await;
}
