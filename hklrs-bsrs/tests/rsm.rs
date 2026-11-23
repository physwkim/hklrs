//! RsmCallback on K6C over komega: step scans through a RunEngine with
//! bsrs's soft detector, whose frames a synthetic reader supplies, and
//! hand-built documents for the orders a step scan does not produce.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bsrs::backends::soft::SoftMotor;
use bsrs::backends::soft::detector::soft_detector;
use bsrs::core::msg::{MovableObj, ReadableObj, StageableObj, TriggerableObj};
use bsrs::core::{Msg, plan_box};
use bsrs::engine::RunEngine;
use bsrs::event_model::{
    Configuration, DataKey, Document, Dtype, Event, EventDescriptor, RunStart, RunStop,
    StreamDatum, StreamRange, StreamResource,
};
use hklrs::frame_map::FrameMap;
use hklrs::grid::{Binning, Cube};
use hklrs::nalgebra::Vector3;
use hklrs::normalise::{FrameScalars, PixelFields, RawFrame, Region};
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{PixelDetector, Wavelength};
use hklrs_bsrs::{
    FailedFrame, FrameFailure, FrameReader, RsmCallback, RsmConfig, RsmError, RsmValue,
};
use serde_json::{Value, json};

const LAMBDA: f64 = 1.54;
const SHAPE: (usize, usize) = (4, 6);
/// The soft detector's name, which is its image data key.
const IMAGE: &str = "det";
/// delta, read from a motor that stays put.
const DELTA: f64 = 25.0;
const KOMEGA: [f64; 7] = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0];

/// K6C mu, komega, kappa, kphi, gamma, delta.
fn readbacks(komega: f64) -> [f64; 6] {
    [3.0, komega, -30.0, 40.0, 12.0, DELTA]
}

/// Frame `index`'s pixels: every pixel of every frame different.
fn pixels(index: u64) -> Vec<f64> {
    let pixels = (SHAPE.0 * SHAPE.1) as u64;
    (0..pixels)
        .map(|p| (1 + index * pixels + p) as f64)
        .collect()
}

/// Frame `index`'s raw image.
fn raw(index: u64) -> RawFrame {
    RawFrame::new(SHAPE, pixels(index)).unwrap()
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
        PixelFields::new(SHAPE),
        vec![Binning::new(0.01, 0.0).unwrap(); 3],
    )
    .unwrap()
}

/// The frames' cube through FrameMap alone, added in the order given.
fn direct(frames: &[(f64, u64)]) -> Cube {
    let map = frame_map();
    let wavelength = Wavelength::new(LAMBDA).unwrap();
    let mut cube = Cube::default();
    for &(komega, index) in frames {
        let frame = map
            .map(
                &readbacks(komega),
                wavelength,
                &raw(index),
                &FrameScalars::new(),
            )
            .unwrap();
        map.add(&mut cube, &frame).unwrap();
    }
    cube
}

/// Synthetic frames for any resource; `fail` is unreadable, `slow` takes
/// a while and `transposed` is read in the (fast, slow) shape.
#[derive(Default)]
struct Frames {
    fail: Option<u64>,
    slow: Option<u64>,
    transposed: Option<u64>,
}

impl FrameReader for Frames {
    fn read(
        &self,
        _resource: &StreamResource,
        index: u64,
    ) -> Result<RawFrame, Box<dyn std::error::Error + Send + Sync>> {
        if self.slow == Some(index) {
            std::thread::sleep(Duration::from_millis(50));
        }
        if self.fail == Some(index) {
            return Err(format!("frame {index} is unreadable").into());
        }
        if self.transposed == Some(index) {
            return Ok(RawFrame::new((SHAPE.1, SHAPE.0), pixels(index))?);
        }
        Ok(raw(index))
    }
}

fn rsm(frames: Frames) -> Arc<RsmCallback> {
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
        reader: Arc::new(frames),
    };
    Arc::new(RsmCallback::new(config).unwrap())
}

/// A step scan of komega over `points` that triggers the soft detector
/// at each, with `rsm` subscribed.
async fn scan(rsm: &Arc<RsmCallback>, points: &[f64]) {
    let det = Arc::new(soft_detector(IMAGE));
    let komega = Arc::new(SoftMotor::new("komega", Some(0.0)));
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
    for &value in points {
        msgs.extend([
            Msg::Set {
                obj: komega.clone() as Arc<dyn MovableObj>,
                value,
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
            Msg::Read(komega.clone() as Arc<dyn ReadableObj>),
            Msg::Read(delta.clone() as Arc<dyn ReadableObj>),
            Msg::Read(det.clone() as Arc<dyn ReadableObj>),
            Msg::Save,
        ]);
    }
    msgs.extend([
        Msg::Unstage(det.clone() as Arc<dyn StageableObj>),
        Msg::CloseRun {
            exit_status: "success".into(),
            reason: None,
        },
    ]);
    let re = RunEngine::new(vec![]);
    re.subscribe(rsm.callback());
    let result = re
        .run_async(plan_box(futures::stream::iter(msgs)))
        .await
        .unwrap();
    assert_eq!(result.exit_status, "success", "{result:?}");
}

// -- hand-built documents of run "run", stream "primary" ---------------------

fn start() -> Document {
    Document::Start(RunStart {
        uid: "run".into(),
        ..Default::default()
    })
}

fn stop() -> Document {
    Document::Stop(RunStop {
        uid: "stop".into(),
        run_start: "run".into(),
        ..Default::default()
    })
}

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

/// The stream's descriptor: the image, komega and the `fields` given, and
/// `configuration` as the configuration of one object.
fn descriptor(fields: &[&str], configuration: &[(&str, f64)]) -> Document {
    let mut data_keys = HashMap::from([
        (IMAGE.to_owned(), data_key(Some("STREAM:"))),
        ("komega".to_owned(), data_key(None)),
    ]);
    for &f in fields {
        data_keys.insert(f.to_owned(), data_key(None));
    }
    let data = configuration
        .iter()
        .map(|&(k, v)| (k.to_owned(), json!(v)))
        .collect();
    Document::Descriptor(EventDescriptor {
        uid: "desc".into(),
        run_start: "run".into(),
        time: 0.0,
        data_keys,
        configuration: HashMap::from([(
            "mono".to_owned(),
            Configuration {
                data,
                ..Default::default()
            },
        )]),
        name: Some("primary".into()),
        hints: None,
        object_keys: HashMap::new(),
    })
}

fn resource() -> Document {
    Document::StreamResource(StreamResource {
        uid: "res".into(),
        data_key: IMAGE.into(),
        mimetype: "application/x-test".into(),
        uri: "memory://test".into(),
        parameters: HashMap::new(),
        run_start: Some("run".into()),
    })
}

/// Frames `index..` of the resource are events `seq`.
fn datum(seq: std::ops::Range<u64>, index: u64) -> Document {
    Document::StreamDatum(StreamDatum {
        uid: format!("datum-{}", seq.start),
        stream_resource: "res".into(),
        descriptor: "desc".into(),
        indices: StreamRange {
            start: index,
            stop: index + (seq.end - seq.start),
        },
        seq_nums: StreamRange {
            start: seq.start,
            stop: seq.end,
        },
    })
}

/// Event `seq`, at komega of frame `seq − 1`, with delta unless `delta`
/// is false.
fn event(seq: u64, delta: bool) -> Document {
    let mut data: HashMap<String, Value> =
        HashMap::from([("komega".to_owned(), json!(KOMEGA[seq as usize - 1]))]);
    if delta {
        data.insert("delta".to_owned(), json!(DELTA));
    }
    Document::Event(Event {
        uid: format!("event-{seq}"),
        descriptor: "desc".into(),
        time: 0.0,
        seq_num: seq,
        data,
        timestamps: HashMap::new(),
        filled: HashMap::new(),
    })
}

fn feed(rsm: &RsmCallback, docs: impl IntoIterator<Item = Document>) {
    for doc in docs {
        rsm.document(&doc);
    }
}

/// Frames `0..n` at their komega, as events `1..=n`.
fn frames(n: u64) -> Vec<(f64, u64)> {
    (0..n).map(|i| (KOMEGA[i as usize], i)).collect()
}

// -- boundaries ---------------------------------------------------------------

/// Event 1's datum comes before it, as a step scan sends it; event 2's
/// after it, which waits in the queue until the datum arrives.
#[test]
fn a_datum_before_or_after_its_event_maps_the_frame() {
    let rsm = rsm(Frames::default());
    feed(
        &rsm,
        [
            start(),
            descriptor(&["delta"], &[]),
            resource(),
            datum(1..2, 0),
            event(1, true),
            event(2, true),
            datum(2..3, 1),
            stop(),
        ],
    );
    let map = rsm.finish().unwrap();
    assert_eq!(map.failed, []);
    assert_eq!(map.mapped, [1, 2]);
    assert_eq!(map.cube, direct(&frames(2)));
}

/// Event 2's datum never comes: at the stop document it is a failed
/// frame, and the run is not finished before that document.
#[test]
fn an_event_waiting_at_the_stop_document_is_a_failed_frame() {
    let rsm = rsm(Frames::default());
    feed(
        &rsm,
        [
            start(),
            descriptor(&["delta"], &[]),
            resource(),
            datum(1..2, 0),
            event(1, true),
            event(2, true),
        ],
    );
    assert_eq!(rsm.finish(), Err(RsmError::NotStopped));
    rsm.document(&stop());
    let map = rsm.finish().unwrap();
    assert_eq!(
        map.failed,
        [FailedFrame {
            seq_num: 2,
            reason: FrameFailure::NoDatum
        }]
    );
    assert_eq!(map.mapped, [1]);
    assert_eq!(map.cube, direct(&frames(1)));
}

/// The reader fails on frame 1, event 2 of the scan; the other four map.
#[tokio::test]
async fn a_reader_failure_fails_only_its_frame() {
    let rsm = rsm(Frames {
        fail: Some(1),
        ..Frames::default()
    });
    scan(&rsm, &KOMEGA[..5]).await;
    let map = rsm.finish().unwrap();
    assert_eq!(
        map.failed,
        [FailedFrame {
            seq_num: 2,
            reason: FrameFailure::Read("frame 1 is unreadable".into())
        }]
    );
    assert_eq!(map.mapped, [1, 3, 4, 5]);
    let mut others = frames(5);
    others.remove(1);
    assert_eq!(map.cube, direct(&others));
}

/// The reader states the shape it read: frame 2 comes back transposed,
/// (6, 4) for the raw (4, 6), and fails as a shape mismatch, not as a
/// frame read row-major; the other four map.
#[tokio::test]
async fn a_frame_read_in_another_shape_fails_only_its_frame() {
    let rsm = rsm(Frames {
        transposed: Some(2),
        ..Frames::default()
    });
    scan(&rsm, &KOMEGA[..5]).await;
    let map = rsm.finish().unwrap();
    let mismatch = hklrs::Error::ShapeMismatch {
        what: "raw frame",
        shape: (6, 4),
        expected: (4, 6),
    };
    assert_eq!(
        map.failed,
        [FailedFrame {
            seq_num: 3,
            reason: FrameFailure::Map(mismatch)
        }]
    );
    assert_eq!(map.mapped, [1, 2, 4, 5]);
    let mut others = frames(5);
    others.remove(2);
    assert_eq!(map.cube, direct(&others));
}

/// Six events wait, one datum releases them, and a seventh pairs at once;
/// the first frame is the slowest to read. They reach the cube in seq_num
/// order, adding exactly as FrameMap does in that order.
#[test]
fn frames_keep_their_order_across_the_worker() {
    let rsm = rsm(Frames {
        slow: Some(0),
        ..Frames::default()
    });
    feed(
        &rsm,
        [start(), descriptor(&["delta"], &[]), resource()]
            .into_iter()
            .chain((1..=6).map(|seq| event(seq, true)))
            .chain([datum(1..8, 0), event(7, true), stop()]),
    );
    let map = rsm.finish().unwrap();
    assert_eq!(map.failed, []);
    assert_eq!(map.mapped, [1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(map.cube, direct(&frames(7)));
}

/// The scan's whole map is a FrameMap over its frames.
#[tokio::test]
async fn the_cube_is_a_direct_frame_map_over_the_same_frames() {
    let rsm = rsm(Frames::default());
    scan(&rsm, &KOMEGA[..5]).await;
    let map = rsm.finish().unwrap();
    assert_eq!(map.failed, []);
    assert_eq!(map.mapped, [1, 2, 3, 4, 5]);
    assert_eq!(map.cube, direct(&frames(5)));
}

/// A bound field the events lack is read from the descriptor's
/// configuration; in neither, the stream is refused at its descriptor.
#[test]
fn a_bound_field_comes_from_the_events_or_the_configuration() {
    let from_configuration = rsm(Frames::default());
    feed(
        &from_configuration,
        [
            start(),
            descriptor(&[], &[("delta", DELTA)]),
            resource(),
            datum(1..3, 0),
            event(1, false),
            event(2, false),
            stop(),
        ],
    );
    let map = from_configuration.finish().unwrap();
    assert_eq!(map.mapped, [1, 2]);
    assert_eq!(map.cube, direct(&frames(2)));

    let in_neither = rsm(Frames::default());
    feed(
        &in_neither,
        [
            start(),
            descriptor(&[], &[]),
            resource(),
            datum(1..2, 0),
            event(1, false),
            stop(),
        ],
    );
    assert_eq!(
        in_neither.finish(),
        Err(RsmError::NoField {
            stream: "primary".into(),
            field: "delta".into()
        })
    );
}
