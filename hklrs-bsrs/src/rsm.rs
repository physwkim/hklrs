//! Live reciprocal-space map of a bsrs run (design §8.5). An
//! [`RsmCallback`] pairs each image event with its stream datum and maps
//! the frame through [`FrameMap`] into a [`Cube`] that grows to hold it.
//!
//! bsrs runs callbacks inline, so the callback only pairs documents: one
//! worker thread reads, maps and adds the frames in the order they are
//! paired, and is the only writer of the map. A frame its reader cannot
//! read yet is read again until [`RsmConfig::read_deadline`] after it was
//! paired, and the frames behind it wait for it while their own deadlines
//! run.

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bsrs::engine::DocumentCallback;
use bsrs::event_model::{
    Document, EventDescriptor, StreamDatum, StreamResource, unpack_event_page,
};
use hklrs::Wavelength;
use hklrs::frame_map::{FrameMap, MappedFrame};
use hklrs::grid::Cube;
use hklrs::normalise::{FrameScalars, RawFrame};
use serde_json::Value;
use thiserror::Error;

/// Where a frame's number comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum RsmValue<T> {
    /// A field of the event or, when the stream's events have no such
    /// field, of its descriptor's configuration.
    Field(String),
    /// The same value on every frame.
    Fixed(T),
}

/// A monitor field and the reference its counts are scaled to.
#[derive(Clone, Debug, PartialEq)]
pub struct RsmMonitor {
    pub field: String,
    pub reference: f64,
}

/// Turns a stream resource and a frame index into that frame's pixels.
pub trait FrameReader: Send + Sync {
    /// Frame `index` of `resource`: the whole raw image in the shape it
    /// was read in, which [`FrameMap::map`] refuses unless it is the raw
    /// shape.
    fn read(&self, resource: &StreamResource, index: u64) -> Result<RawFrame, ReadError>;
}

/// Why a [`FrameReader`] has no frame. The reader only says which kind of
/// failure it is; whether to read again is the [`RsmCallback`] worker's.
#[derive(Debug, Error)]
pub enum ReadError {
    /// A later read may find it: the detector may still be writing it.
    #[error("not readable yet: {0}")]
    NotYet(Box<dyn std::error::Error + Send + Sync>),
    /// No later read will.
    #[error("{0}")]
    Permanent(Box<dyn std::error::Error + Send + Sync>),
}

/// What an [`RsmCallback`] maps.
pub struct RsmConfig {
    /// The geometry, projection, detector, region, fields and binnings
    /// every frame shares.
    pub map: FrameMap,
    /// The stream whose events are frames; bluesky numbers events per
    /// stream, so a frame is its seq_num.
    pub stream: String,
    /// The stream's external (`STREAM:`) data key of the images.
    pub image_key: String,
    /// One per geometry axis, in its order: degrees for rotations, metres
    /// for translations.
    pub axes: Vec<RsmValue<f64>>,
    /// Å when read from a field.
    pub wavelength: RsmValue<Wavelength>,
    pub monitor: Option<RsmMonitor>,
    /// A count time field, s.
    pub count_time: Option<String>,
    /// The cube the frames are added to; `None` starts empty.
    pub seed: Option<Cube>,
    pub reader: Arc<dyn FrameReader>,
    /// How long after a frame is paired, its event and its datum both in,
    /// it is read again while [`ReadError::NotYet`]; then it fails with the
    /// last reason. Every frame is read once at least: a frame the worker
    /// reaches after its deadline, or every frame when this is zero, is
    /// read once.
    pub read_deadline: Duration,
}

impl RsmConfig {
    /// For a frame its datum announces before the file holds it: bsrs
    /// flushes the file before each datum but sends the datum even when
    /// that flush fails.
    pub const DEFAULT_READ_DEADLINE: Duration = Duration::from_secs(10);
}

/// The pause between reads of a frame not readable yet.
const POLL: Duration = Duration::from_millis(20);

/// Why the map does not hold the whole run.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RsmError {
    #[error("{got} axis sources for a geometry of {expected} axes")]
    AxisCount { expected: usize, got: usize },
    #[error("stream {stream:?} has no external STREAM: data key {key:?}")]
    NoImage { stream: String, key: String },
    #[error("stream {stream:?} has no number {field:?} in its events or configuration")]
    NoField { stream: String, field: String },
    #[error("the run has not stopped")]
    NotStopped,
    #[error("mapping thread: {0}")]
    Worker(String),
}

/// Why one frame is not in the cube.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FrameFailure {
    #[error("no stream datum by the stop document")]
    NoDatum,
    #[error("the event has no number {field:?}")]
    NoField { field: String },
    #[error("reading the frame: {0}")]
    Read(String),
    #[error(transparent)]
    Map(#[from] hklrs::Error),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FailedFrame {
    pub seq_num: u64,
    pub reason: FrameFailure,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RsmMap {
    pub cube: Cube,
    /// The seq_nums of the frames in `cube`, in the order they were added.
    pub mapped: Vec<u64>,
    /// In the order they were recorded.
    pub failed: Vec<FailedFrame>,
}

/// A bsrs document callback that maps the first run it sees: its Start
/// opens the map and its Stop closes it; documents of other runs are
/// ignored.
pub struct RsmCallback {
    fields: Fields,
    phase: Mutex<Phase>,
    worker: Mutex<Option<JoinHandle<()>>>,
    out: Arc<Mutex<Out>>,
}

impl RsmCallback {
    pub fn new(config: RsmConfig) -> Result<Self, RsmError> {
        let expected = config.map.geometry().axes().len();
        if config.axes.len() != expected {
            return Err(RsmError::AxisCount {
                expected,
                got: config.axes.len(),
            });
        }
        let out = Arc::new(Mutex::new(Out {
            map: RsmMap {
                cube: config.seed.unwrap_or_default(),
                ..RsmMap::default()
            },
            error: None,
        }));
        let (jobs, receiver) = mpsc::channel();
        let worker = {
            let (map, reader, out) = (config.map, config.reader, Arc::clone(&out));
            let deadline = config.read_deadline;
            std::thread::Builder::new()
                .name("rsm".into())
                .spawn(move || work(&map, &*reader, deadline, receiver, &out))
                .map_err(|e| RsmError::Worker(e.to_string()))?
        };
        Ok(Self {
            fields: Fields {
                stream: config.stream,
                image_key: config.image_key,
                axes: config.axes,
                wavelength: config.wavelength,
                monitor: config.monitor,
                count_time: config.count_time,
            },
            phase: Mutex::new(Phase::Idle(jobs)),
            worker: Mutex::new(Some(worker)),
            out,
        })
    }

    /// For [`bsrs::engine::RunEngine::subscribe`]: it has no filter for
    /// stream resources and datums, so the callback takes every document.
    pub fn callback(self: &Arc<Self>) -> DocumentCallback {
        let me = Arc::clone(self);
        Arc::new(move |doc| me.document(doc))
    }

    pub fn document(&self, doc: &Document) {
        let mut phase = lock(&self.phase);
        match doc {
            Document::Start(start) => {
                *phase = match std::mem::replace(&mut *phase, Phase::Stopped) {
                    Phase::Idle(jobs) => Phase::Open(Run::new(start.uid.clone(), jobs)),
                    other => other,
                };
            }
            Document::Stop(stop) => {
                if let Phase::Open(run) = &mut *phase
                    && run.uid == stop.run_start
                {
                    run.fail_waiting();
                    // drops the sender: the worker drains its queue and ends
                    *phase = Phase::Stopped;
                }
            }
            _ => {
                let Phase::Open(run) = &mut *phase else {
                    return;
                };
                match doc {
                    Document::Descriptor(d) => run.descriptor(&self.fields, d),
                    Document::StreamResource(r) => run.resource(&self.fields.image_key, r),
                    Document::StreamDatum(d) => run.datum(d),
                    Document::Event(e) => {
                        if let Some(bindings) = run.bindings(&e.descriptor) {
                            let data = e.data.clone();
                            run.event(e.seq_num, Pending { bindings, data });
                        }
                    }
                    Document::EventPage(page) => {
                        if let Some(bindings) = run.bindings(&page.descriptor) {
                            for e in unpack_event_page(page) {
                                let bindings = Arc::clone(&bindings);
                                run.event(
                                    e.seq_num,
                                    Pending {
                                        bindings,
                                        data: e.data,
                                    },
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// The map so far: frames still queued for the worker are not in it.
    pub fn snapshot(&self) -> Result<RsmMap, RsmError> {
        lock(&self.out).result()
    }

    /// The whole run's map, once its Stop document has been seen and the
    /// worker has added every frame. Frames still not readable hold it up
    /// to [`RsmConfig::read_deadline`] after the last of them was paired.
    pub fn finish(&self) -> Result<RsmMap, RsmError> {
        if !matches!(*lock(&self.phase), Phase::Stopped) {
            return Err(RsmError::NotStopped);
        }
        let worker = lock(&self.worker).take();
        if let Some(worker) = worker
            && worker.join().is_err()
        {
            let mut out = lock(&self.out);
            out.error.get_or_insert(RsmError::Worker("panicked".into()));
        }
        lock(&self.out).result()
    }
}

/// A worker panic is reported by [`RsmCallback::finish`]; its poisoned lock
/// must not also panic the run engine that calls the callback.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The configured sources, resolved per descriptor by [`Fields::bind`].
struct Fields {
    stream: String,
    image_key: String,
    axes: Vec<RsmValue<f64>>,
    wavelength: RsmValue<Wavelength>,
    monitor: Option<RsmMonitor>,
    count_time: Option<String>,
}

impl Fields {
    fn bind(&self, d: &EventDescriptor) -> Result<Bindings, RsmError> {
        let image = d.data_keys.get(&self.image_key);
        if !image.is_some_and(|k| {
            k.external
                .as_deref()
                .is_some_and(|e| e.starts_with("STREAM:"))
        }) {
            return Err(RsmError::NoImage {
                stream: self.stream.clone(),
                key: self.image_key.clone(),
            });
        }
        let field = |name: &String| {
            if d.data_keys.contains_key(name) {
                return Ok(Scalar::Event(name.clone()));
            }
            d.configuration
                .values()
                .find_map(|c| c.data.get(name))
                .and_then(Value::as_f64)
                .map(Scalar::Value)
                .ok_or_else(|| RsmError::NoField {
                    stream: self.stream.clone(),
                    field: name.clone(),
                })
        };
        let axes = self
            .axes
            .iter()
            .map(|a| match a {
                RsmValue::Field(name) => field(name),
                RsmValue::Fixed(v) => Ok(Scalar::Value(*v)),
            })
            .collect::<Result<_, _>>()?;
        let wavelength = match &self.wavelength {
            RsmValue::Field(name) => field(name)?,
            RsmValue::Fixed(w) => Scalar::Value(w.angstrom()),
        };
        let monitor = match &self.monitor {
            Some(m) => Some((field(&m.field)?, m.reference)),
            None => None,
        };
        let count_time = self.count_time.as_ref().map(field).transpose()?;
        Ok(Bindings {
            axes,
            wavelength,
            monitor,
            count_time,
        })
    }
}

/// One descriptor's sources: a field of each event, or a number fixed
/// for the descriptor.
struct Bindings {
    axes: Vec<Scalar>,
    wavelength: Scalar,
    monitor: Option<(Scalar, f64)>,
    count_time: Option<Scalar>,
}

enum Scalar {
    Event(String),
    Value(f64),
}

enum Phase {
    /// Before the run's Start document.
    Idle(Sender<Job>),
    Open(Run),
    /// After the run's Stop document; nothing more reaches the worker.
    Stopped,
}

/// The open run's pairing state. Seq_nums count the mapped stream's
/// events across its descriptors, so datums and waiting events are kept by
/// seq_num, not per descriptor.
struct Run {
    uid: String,
    jobs: Sender<Job>,
    /// The mapped stream's descriptors that bound, by uid.
    descriptors: HashMap<String, Arc<Bindings>>,
    /// Stream resources of the image key, by uid.
    resources: HashMap<String, Arc<StreamResource>>,
    /// The image datums, by their first seq_num.
    datums: BTreeMap<u64, Datum>,
    /// Events whose datum has not arrived, by seq_num.
    waiting: BTreeMap<u64, Pending>,
}

struct Datum {
    seq_stop: u64,
    index_start: u64,
    resource: Arc<StreamResource>,
}

struct Pending {
    bindings: Arc<Bindings>,
    data: HashMap<String, Value>,
}

impl Run {
    fn new(uid: String, jobs: Sender<Job>) -> Self {
        Self {
            uid,
            jobs,
            descriptors: HashMap::new(),
            resources: HashMap::new(),
            datums: BTreeMap::new(),
            waiting: BTreeMap::new(),
        }
    }

    /// The worker ends only by a panic, which [`RsmCallback::finish`]
    /// reports, so a job it can no longer take needs no record here.
    fn send(&self, job: Job) {
        let _ = self.jobs.send(job);
    }

    fn descriptor(&mut self, fields: &Fields, d: &EventDescriptor) {
        if d.run_start != self.uid || d.name.as_deref() != Some(fields.stream.as_str()) {
            return;
        }
        match fields.bind(d) {
            Ok(bindings) => {
                self.descriptors.insert(d.uid.clone(), Arc::new(bindings));
            }
            Err(e) => self.send(Job::Refused(e)),
        }
    }

    fn resource(&mut self, image_key: &str, r: &StreamResource) {
        let this_run = r.run_start.as_deref().is_none_or(|u| u == self.uid);
        if this_run && r.data_key == image_key {
            self.resources.insert(r.uid.clone(), Arc::new(r.clone()));
        }
    }

    fn datum(&mut self, d: &StreamDatum) {
        if !self.descriptors.contains_key(&d.descriptor) {
            return;
        }
        let Some(resource) = self.resources.get(&d.stream_resource) else {
            return;
        };
        let seq = d.seq_nums.start..d.seq_nums.stop;
        self.datums.insert(
            seq.start,
            Datum {
                seq_stop: seq.end,
                index_start: d.indices.start,
                resource: Arc::clone(resource),
            },
        );
        let released: Vec<u64> = self.waiting.range(seq).map(|(&s, _)| s).collect();
        for seq_num in released {
            let pending = self
                .waiting
                .remove(&seq_num)
                .expect("in the range just read");
            self.dispatch(seq_num, pending);
        }
    }

    /// The bindings of a descriptor of the mapped stream; none for any
    /// other descriptor, whose events are not frames.
    fn bindings(&self, descriptor: &str) -> Option<Arc<Bindings>> {
        self.descriptors.get(descriptor).cloned()
    }

    fn event(&mut self, seq_num: u64, pending: Pending) {
        if self.frame_of(seq_num).is_some() {
            self.dispatch(seq_num, pending);
        } else {
            self.waiting.insert(seq_num, pending);
        }
    }

    /// The resource and frame index of `seq_num`:
    /// `indices.start + (seq_num − seq_nums.start)` of its datum.
    fn frame_of(&self, seq_num: u64) -> Option<(Arc<StreamResource>, u64)> {
        let (&start, d) = self.datums.range(..=seq_num).next_back()?;
        (seq_num < d.seq_stop).then(|| (Arc::clone(&d.resource), d.index_start + (seq_num - start)))
    }

    fn dispatch(&self, seq_num: u64, pending: Pending) {
        let (resource, index) = self.frame_of(seq_num).expect("dispatched once paired");
        self.send(Job::Frame(Frame {
            seq_num,
            bindings: pending.bindings,
            data: pending.data,
            resource,
            index,
            ready: Instant::now(),
        }));
    }

    fn fail_waiting(&mut self) {
        for seq_num in std::mem::take(&mut self.waiting).into_keys() {
            self.send(Job::Failed(FailedFrame {
                seq_num,
                reason: FrameFailure::NoDatum,
            }));
        }
    }
}

enum Job {
    Frame(Frame),
    Failed(FailedFrame),
    Refused(RsmError),
}

struct Frame {
    seq_num: u64,
    bindings: Arc<Bindings>,
    data: HashMap<String, Value>,
    resource: Arc<StreamResource>,
    index: u64,
    /// When its event and its datum were both in: its read deadline counts
    /// from here, not from the worker's first read, which the frames ahead
    /// of it delay.
    ready: Instant,
}

impl Frame {
    fn value(&self, s: &Scalar) -> Result<f64, FrameFailure> {
        match s {
            Scalar::Value(v) => Ok(*v),
            Scalar::Event(field) => {
                self.data
                    .get(field)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| FrameFailure::NoField {
                        field: field.clone(),
                    })
            }
        }
    }

    fn map(
        &self,
        map: &FrameMap,
        reader: &dyn FrameReader,
        deadline: Duration,
    ) -> Result<MappedFrame, FrameFailure> {
        let b = &self.bindings;
        let readbacks = b
            .axes
            .iter()
            .map(|a| self.value(a))
            .collect::<Result<Vec<_>, _>>()?;
        let wavelength = Wavelength::new(self.value(&b.wavelength)?)?;
        let mut scalars = FrameScalars::new();
        if let Some((monitor, reference)) = &b.monitor {
            scalars = scalars.monitor(self.value(monitor)?, *reference)?;
        }
        if let Some(t) = &b.count_time {
            scalars = scalars.count_time(self.value(t)?)?;
        }
        let raw = self.read(reader, deadline)?;
        Ok(map.map(&readbacks, wavelength, &raw, &scalars)?)
    }

    /// Reads the frame again every [`POLL`] while it is not readable yet,
    /// until `deadline` after it was ready. The worker reads nothing else
    /// meanwhile, so the frames behind it keep their order.
    fn read(&self, reader: &dyn FrameReader, deadline: Duration) -> Result<RawFrame, FrameFailure> {
        loop {
            let reason = match reader.read(&self.resource, self.index) {
                Ok(raw) => return Ok(raw),
                Err(ReadError::Permanent(e)) => return Err(FrameFailure::Read(e.to_string())),
                Err(ReadError::NotYet(e)) => e,
            };
            let left = deadline.saturating_sub(self.ready.elapsed());
            if left.is_zero() {
                return Err(FrameFailure::Read(reason.to_string()));
            }
            std::thread::sleep(left.min(POLL));
        }
    }
}

/// What the worker has made of the jobs so far.
struct Out {
    map: RsmMap,
    /// The first reason the map is not the whole run's.
    error: Option<RsmError>,
}

impl Out {
    fn result(&self) -> Result<RsmMap, RsmError> {
        match &self.error {
            Some(e) => Err(e.clone()),
            None => Ok(self.map.clone()),
        }
    }
}

fn work(
    map: &FrameMap,
    reader: &dyn FrameReader,
    deadline: Duration,
    jobs: Receiver<Job>,
    out: &Mutex<Out>,
) {
    for job in jobs {
        match job {
            Job::Frame(frame) => {
                let mapped = frame.map(map, reader, deadline);
                let mut out = lock(out);
                let added = mapped.and_then(|m| Ok(map.add(&mut out.map.cube, &m)?));
                match added {
                    Ok(()) => out.map.mapped.push(frame.seq_num),
                    Err(reason) => out.map.failed.push(FailedFrame {
                        seq_num: frame.seq_num,
                        reason,
                    }),
                }
            }
            Job::Failed(failed) => lock(out).map.failed.push(failed),
            Job::Refused(e) => {
                lock(out).error.get_or_insert(e);
            }
        }
    }
}
