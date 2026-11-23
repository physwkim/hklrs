//! hkl plans on E4CV soft motors: cubic a = λ = 1.54, U = I, so (0, 0, 1)
//! is at omega 30, chi 0, phi 0, tth 60.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bsrs::backends::soft::SoftMotor;
use bsrs::callbacks::CapturingSink;
use bsrs::core::msg::{MovableObj, NamedObj, ReadableObj};
use bsrs::core::{BsrsError, DataKey, Document, Plan, ReadingValue};
use bsrs::engine::{DocumentSink, RunEngine, RunResult};
use bsrs::event_model::{Event, RunStart};
use hklrs::nalgebra::{UnitQuaternion, Vector3};
use hklrs::presets::preset;
use hklrs::{Lattice, ModeSpec, Orientation, Sample, Wavelength};
use hklrs_bsrs::{
    Diffractometer, RealAxis, WavelengthSource, hkl_list_scan, hkl_scan, mv_hkl, psi_scan,
    zone_scan,
};
use serde_json::json;

const AXES: [&str; 4] = ["omega", "chi", "phi", "tth"];
const LAMBDA: f64 = 1.54;
const AT_001: [f64; 4] = [30.0, 0.0, 0.0, 60.0];

fn device(lattice: Lattice, u: UnitQuaternion<f64>) -> Arc<Diffractometer> {
    let sample = Sample::new("sample", lattice, Orientation::Manual { u });
    let reals = AXES
        .into_iter()
        .zip(AT_001)
        .map(|(a, v)| (a, Arc::new(SoftMotor::new(a, Some(v))) as Arc<dyn RealAxis>))
        .collect();
    let source = WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap());
    let dx = Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals, source, sample);
    Arc::new(dx.unwrap())
}

fn e4cv() -> Arc<Diffractometer> {
    device(Lattice::cubic(LAMBDA).unwrap(), UnitQuaternion::identity())
}

fn detector(name: &str) -> Arc<dyn ReadableObj> {
    Arc::new(SoftMotor::new(name, Some(7.0)))
}

struct Run {
    result: Result<RunResult, BsrsError>,
    documents: Vec<Document>,
}

impl Run {
    fn start(&self) -> &RunStart {
        self.documents
            .iter()
            .find_map(|d| match d {
                Document::Start(s) => Some(s),
                _ => None,
            })
            .expect("a start document")
    }

    fn events(&self) -> Vec<&Event> {
        self.documents
            .iter()
            .filter_map(|d| match d {
                Document::Event(e) => Some(e),
                _ => None,
            })
            .collect()
    }

    fn succeeded(&self) {
        let result = self.result.as_ref().unwrap();
        assert_eq!(result.exit_status, "success", "{result:?}");
    }

    /// Failed before any document, with this message.
    fn failed_before_open(&self) -> String {
        assert!(self.documents.is_empty(), "{:?}", self.documents);
        let result = self.result.as_ref().unwrap();
        assert_eq!(result.exit_status, "fail", "{result:?}");
        assert!(result.run_uids.is_empty(), "{result:?}");
        result.reason.clone()
    }
}

async fn run(plan: Plan) -> Run {
    let sink = Arc::new(CapturingSink::new());
    let re = RunEngine::new(vec![sink.clone() as Arc<dyn DocumentSink>]);
    let result = re.run_async(plan).await;
    Run {
        result,
        documents: sink.snapshot().await,
    }
}

fn hkl_of(event: &Event) -> [f64; 3] {
    ["h", "k", "l"].map(|c| event.data[&format!("e4cv_{c}")].as_f64().unwrap())
}

fn assert_close(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} != {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < tolerance, "{a:?} != {b:?}");
    }
}

#[tokio::test]
async fn hkl_scan_reads_back_the_commanded_hkl() {
    let dx = e4cv();
    let [_, k, l] = dx.hkl_axes();
    k.set_dyn(0.0).await.await.unwrap();
    assert!(dx.target().is_some());

    let run = run(hkl_scan(
        dx.clone(),
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        3,
        [],
    ))
    .await;
    run.succeeded();
    let commanded = [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0]];
    let events = run.events();
    assert_eq!(events.len(), commanded.len());
    for (event, hkl) in events.iter().zip(&commanded) {
        assert_close(&hkl_of(event), hkl, 1e-9);
    }

    let start = &run.start().extra;
    assert_eq!(start["plan_name"], json!("hkl_scan"));
    assert_eq!(start["hkl"], json!(commanded));
    assert_eq!(
        start["plan_args"],
        json!({"start": [0.0, 0.0, 1.0], "stop": [1.0, 0.0, 1.0], "num": 3, "detectors": ["e4cv"]})
    );
    let config = &start["diffractometers"]["e4cv"];
    assert_eq!(config["geometry"], json!("E4CV"));
    assert_eq!(config["mode"], json!("bissector"));
    assert_eq!(config["solver"], json!("hklrs"));
    assert_eq!(config["version"], json!(env!("CARGO_PKG_VERSION")));
    assert_eq!(config["wavelength"], json!(LAMBDA));
    assert_eq!(
        config["lattice"],
        json!([LAMBDA, LAMBDA, LAMBDA, 90.0, 90.0, 90.0])
    );
    assert_eq!(
        config["U"],
        json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    );
    assert!(config["UB"].is_array());

    // the scan moved the reals, so a pseudo set starts from the readbacks,
    // not from the setpoint k = 0 left before it
    assert_eq!(dx.target(), None);
    l.set_dyn(0.5).await.await.unwrap();
    let (p, w) = (
        dx.positions().await.unwrap(),
        dx.wavelength().await.unwrap(),
    );
    let hkl: [f64; 3] = dx.hkl(&p, w).unwrap().into();
    assert_close(&hkl, &[1.0, 0.0, 0.5], 1e-9);
}

/// PY2-15: hklpy2 left the event bundle open at the failed point and lost
/// every later point.
#[tokio::test]
async fn an_unreachable_point_fails_before_the_run_opens() {
    let dx = e4cv();
    let points = vec![[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [1.0, 0.0, 1.0]];
    let run = run(hkl_list_scan(dx.clone(), points, [])).await;
    let message = run.failed_before_open();
    assert!(
        message.contains("hkl_list_scan: point 1: hkl [3.0, 0.0, 1.0]: "),
        "{message}"
    );
    assert!(message.contains("exceeds"), "{message}");
    assert_eq!(dx.positions().await.unwrap(), AT_001);
    // and released the reals
    let [_, _, l] = dx.hkl_axes();
    l.set_dyn(1.0).await.await.unwrap();
}

/// PY2-16: hklpy2's scan_zone crashed on a tuple of detectors.
#[tokio::test]
async fn detectors_are_any_iterable() {
    async fn read(detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>) -> f64 {
        let dx = e4cv();
        let run = run(hkl_scan(dx, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1, detectors)).await;
        run.succeeded();
        run.events()[0].data["det"].as_f64().unwrap()
    }
    assert_eq!(read([detector("det")]).await, 7.0);
    assert_eq!(read(std::iter::once(detector("det"))).await, 7.0);
    assert_eq!(read(Some(detector("det"))).await, 7.0);
    assert_eq!(read(vec![detector("det")]).await, 7.0);
}

#[tokio::test]
async fn a_data_key_read_twice_fails_naming_both_readers() {
    let dx = e4cv();
    let clash = detector("e4cv_omega");
    let run = run(hkl_scan(
        dx.clone(),
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        2,
        [clash],
    ))
    .await;
    let message = run.failed_before_open();
    assert!(
        message
            .contains(r#"hkl_scan: data key "e4cv_omega" is read by both "e4cv_omega" and "e4cv""#),
        "{message}"
    );
    assert_eq!(dx.positions().await.unwrap(), AT_001);
}

#[tokio::test]
async fn a_reader_given_twice_is_read_once() {
    let dx = e4cv();
    let det = detector("det");
    let omega = dx.reals()[0].clone() as Arc<dyn ReadableObj>;
    let detectors = [dx.clone() as Arc<dyn ReadableObj>, omega, det.clone(), det];
    let run = run(hkl_scan(dx, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], 2, detectors)).await;
    run.succeeded();
    assert_eq!(run.start().extra["detectors"], json!(["e4cv", "det"]));
    assert_eq!(run.events().len(), 2);
}

/// Picked closest to the readbacks, the solution changes branch part way
/// along this line (tth changes sign); the scan, each point seeded by the
/// one before, stays on its first branch (design §8.3 step 1).
#[tokio::test]
async fn hkl_scan_stays_on_one_branch_across_a_branch_boundary() {
    let dx = e4cv();
    let (start, stop, num) = ([-1.0, 0.3, -0.3], [-0.2, 0.3, -1.0], 7);
    let point = |i: usize| {
        let f = i as f64 / (num - 1) as f64;
        [0, 1, 2].map(|j| start[j] + (stop[j] - start[j]) * f)
    };
    let wavelength = dx.wavelength().await.unwrap();
    let fresh: Vec<Vec<f64>> = (0..num)
        .map(|i| dx.inverse(point(i), &AT_001, wavelength).unwrap())
        .collect();
    assert!(
        fresh
            .windows(2)
            .any(|w| w[0][3].signum() != w[1][3].signum()),
        "{fresh:?}"
    );

    let run = run(hkl_scan(dx.clone(), start, stop, num, [])).await;
    run.succeeded();
    let events = run.events();
    let path: Vec<[f64; 4]> = events
        .iter()
        .map(|e| AXES.map(|a| e.data[a].as_f64().unwrap()))
        .collect();
    assert_eq!(path.len(), num);
    assert_close(&path[0], &fresh[0], 1e-9);
    for (i, step) in path.windows(2).enumerate() {
        let largest = (0..4)
            .map(|j| (step[1][j] - step[0][j]).abs())
            .fold(0.0, f64::max);
        assert!(largest < 15.0, "step {i}: {path:?}");
    }
    for (i, event) in events.iter().enumerate() {
        assert_close(&hkl_of(event), &point(i), 1e-9);
    }
}

/// A detector whose every read sets h, as a pseudo set issued mid-scan
/// would, and records the set's outcome; `broken` then fails the read.
struct Probe {
    det: SoftMotor,
    h: Arc<dyn MovableObj>,
    broken: bool,
    sets: Mutex<Vec<Result<(), String>>>,
}

impl Probe {
    fn new(dx: &Arc<Diffractometer>, broken: bool) -> Arc<Self> {
        let [h, _, _] = dx.hkl_axes();
        Arc::new(Self {
            det: SoftMotor::new("probe", Some(7.0)),
            h: Arc::new(h),
            broken,
            sets: Mutex::new(Vec::new()),
        })
    }

    fn sets(&self) -> Vec<Result<(), String>> {
        self.sets.lock().unwrap().clone()
    }
}

impl NamedObj for Probe {
    fn name(&self) -> &str {
        self.det.name()
    }
}

#[async_trait]
impl ReadableObj for Probe {
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        let set = self.h.set_dyn(0.5).await.await;
        self.sets
            .lock()
            .unwrap()
            .push(set.map_err(|e| e.to_string()));
        if self.broken {
            return Err(BsrsError::Backend("probe broke".into()));
        }
        self.det.read_dyn().await
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        self.det.describe_dyn().await
    }
}

fn refused(sets: &[Result<(), String>]) {
    for set in sets {
        let message = set.as_ref().unwrap_err();
        let expected = "e4cv_h: invalid value: hkl_scan holds the reals until its run ends";
        assert!(message.contains(expected), "{message}");
    }
}

/// Refused rather than queued: a queued set would move the reals after the
/// scan, and one issued from the scan's own messages, as here, would wait
/// on its own run.
#[tokio::test]
async fn a_pseudo_set_during_a_scan_is_refused_and_leaves_its_points() {
    let (start, stop) = ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0]);
    let undisturbed = run(hkl_scan(e4cv(), start, stop, 3, [detector("probe")])).await;
    undisturbed.succeeded();

    let dx = e4cv();
    let probe = Probe::new(&dx, false);
    let scanned = run(hkl_scan(dx.clone(), start, stop, 3, [probe.clone() as _])).await;
    scanned.succeeded();
    let sets = probe.sets();
    assert_eq!(sets.len(), 3);
    refused(&sets);
    let data = |run: &Run| -> Vec<_> { run.events().iter().map(|e| e.data.clone()).collect() };
    assert_eq!(data(&scanned), data(&undisturbed));
    let last = AXES.map(|a| scanned.events()[2].data[a].as_f64().unwrap());
    assert_eq!(dx.positions().await.unwrap(), last);

    // released when the run ends
    let [h, _, _] = dx.hkl_axes();
    h.set_dyn(0.5).await.await.unwrap();
    assert_close(&dx.target().unwrap(), &[0.5, 0.0, 1.0], 1e-9);
}

#[tokio::test]
async fn a_scan_failing_mid_run_releases_the_reals() {
    let dx = e4cv();
    let probe = Probe::new(&dx, true);
    let failed = run(hkl_scan(
        dx.clone(),
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        3,
        [probe.clone() as _],
    ))
    .await;
    let result = failed.result.as_ref().unwrap();
    assert_eq!(result.exit_status, "fail", "{result:?}");
    assert!(result.reason.contains("probe broke"), "{result:?}");
    failed.start();
    let sets = probe.sets();
    assert_eq!(sets.len(), 1);
    refused(&sets);

    let [h, _, _] = dx.hkl_axes();
    h.set_dyn(0.5).await.await.unwrap();
    assert_close(&dx.target().unwrap(), &[0.5, 0.0, 1.0], 1e-9);
}

#[tokio::test]
async fn mv_hkl_moves_the_reals_and_sets_the_setpoint() {
    let dx = e4cv();
    let moved = run(mv_hkl(dx.clone(), [1.0, 0.0, 1.0])).await;
    assert!(moved.documents.is_empty());
    moved.result.unwrap();
    let at = dx.positions().await.unwrap();
    assert_close(&[at[0], at[3]], &[45.0, 90.0], 1e-9);
    assert_eq!(dx.target(), Some([1.0, 0.0, 1.0]));

    // fails as an unreachable h set does, in its status, with nothing moved
    let at = dx.positions().await.unwrap();
    let message = run_fail(mv_hkl(dx.clone(), [3.0, 0.0, 1.0])).await;
    assert!(message.contains("mv_hkl: "), "{message}");
    assert!(message.contains("hkl [3.0, 0.0, 1.0]: "), "{message}");
    assert_eq!(dx.positions().await.unwrap(), at);
    assert_eq!(dx.target(), Some([1.0, 0.0, 1.0]));
}

async fn run_fail(plan: Plan) -> String {
    run(plan).await.failed_before_open()
}

#[tokio::test]
async fn psi_scan_holds_hkl_in_its_own_mode() {
    let dx = e4cv();
    let reference = [0.0, 0.0, 1.0];
    let plan = psi_scan(
        dx.clone(),
        Some("psi_constant"),
        [1.0, 0.0, 1.0],
        reference,
        [20.0, 40.0],
        3,
        [],
    );
    let run = run(plan).await;
    run.succeeded();
    let events = run.events();
    assert_eq!(events.len(), 3);
    let ub = dx.current_sample().ub();
    for (event, psi) in events.iter().zip([20.0_f64, 30.0, 40.0]) {
        assert_close(&hkl_of(event), &[1.0, 0.0, 1.0], 1e-9);
        let degrees = AXES.map(|a| event.data[a].as_f64().unwrap());
        let geometry = dx.geometry();
        let values = geometry.values_from_degrees(&degrees).unwrap();
        let read = hklrs::inverse::psi(&geometry, &ub, &Vector3::from(reference), &values);
        assert!((read.unwrap() - psi.to_radians()).abs() < 1e-9);
    }
    let start = &run.start().extra;
    assert_eq!(
        start["diffractometers"]["e4cv"]["mode"],
        json!("psi_constant")
    );
    assert_eq!(dx.mode(), Some(ModeSpec::new("bissector")));
}

#[tokio::test]
async fn psi_scan_without_a_mode_takes_the_geometrys_only_psi_mode() {
    let given = run(psi_scan(
        e4cv(),
        Some("psi_constant"),
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [20.0, 40.0],
        3,
        [],
    ))
    .await;
    let omitted = run(psi_scan(
        e4cv(),
        None,
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [20.0, 40.0],
        3,
        [],
    ))
    .await;
    omitted.succeeded();
    let extra = &omitted.start().extra;
    assert_eq!(
        extra["diffractometers"]["e4cv"]["mode"],
        json!("psi_constant")
    );
    assert_eq!(extra["plan_args"]["mode"], json!(null));
    let at = |run: &Run| -> Vec<[f64; 4]> {
        let events = run.events();
        events
            .iter()
            .map(|e| AXES.map(|a| e.data[a].as_f64().unwrap()))
            .collect()
    };
    assert_eq!(at(&omitted), at(&given));
}

#[tokio::test]
async fn psi_scan_without_a_mode_refuses_naming_the_candidates() {
    let geometry = preset("E6C").unwrap();
    let names: Vec<String> = geometry
        .axes()
        .iter()
        .map(|a| a.name().to_owned())
        .collect();
    let reals = names
        .iter()
        .map(|a| {
            (
                a.as_str(),
                Arc::new(SoftMotor::new(a, Some(0.0))) as Arc<dyn RealAxis>,
            )
        })
        .collect();
    let sample = Sample::new(
        "sample",
        Lattice::cubic(LAMBDA).unwrap(),
        Orientation::Manual {
            u: UnitQuaternion::identity(),
        },
    );
    let source = WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap());
    let dx = Diffractometer::new("e6c", geometry, reals, source, sample).unwrap();
    let plan = psi_scan(
        Arc::new(dx),
        None,
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [20.0, 40.0],
        3,
        [],
    );
    let message = run_fail(plan).await;
    assert!(message.contains("psi_scan: geometry \"E6C\" "), "{message}");
    assert!(
        message.contains(r#"["psi_constant_vertical", "psi_constant_horizontal"]"#),
        "{message}"
    );
}

/// PY2-11: hklpy2's scan_zone turns and scales with the direct-lattice
/// matrix, which puts this midpoint at (0.9428, 0.5657, 0).
#[tokio::test]
async fn zone_scan_interpolates_in_the_reciprocal_metric() {
    let lattice = Lattice::orthorhombic(3.0, 5.0, 7.0).unwrap();
    // a general U keeps (0, 1, 0) off bissector's chi = 90° continuum
    let dx = device(lattice, UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3));
    let run = run(zone_scan(dx, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3, [])).await;
    run.succeeded();
    // 45° between a* and b*, length (1/3 + 1/5)/2 in units of 2π/Å
    let middle = [0.8 * 0.5_f64.sqrt(), 4.0 / 3.0 * 0.5_f64.sqrt(), 0.0];
    let events = run.events();
    assert_close(&hkl_of(events[0]), &[1.0, 0.0, 0.0], 1e-9);
    assert_close(&hkl_of(events[1]), &middle, 1e-9);
    assert_close(&hkl_of(events[2]), &[0.0, 1.0, 0.0], 1e-9);
    let commanded: Vec<f64> =
        serde_json::from_value::<Vec<[f64; 3]>>(run.start().extra["hkl"].clone())
            .unwrap()
            .concat();
    assert_close(&commanded[3..6], &middle, 1e-12);
}

#[tokio::test]
async fn a_zone_needs_two_directions() {
    let dx = e4cv();
    let message = run_fail(zone_scan(dx, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 3, [])).await;
    assert!(
        message.contains("zone_scan: hkl [1.0, 0.0, 0.0] and [2.0, 0.0, 0.0] span no zone"),
        "{message}"
    );
}
