//! h, k, l pseudo axes on E4CV soft motors: cubic a = λ = 1.54, U = I, so
//! (0, 0, 1) is at omega 30, chi 0, phi 0, tth 60.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use bsrs::backends::soft::SoftMotor;
use bsrs::callbacks::CapturingSink;
use bsrs::core::msg::{
    ConfigurableObj, DynLocation, LocatableObj, MovableObj, NamedObj, ReadableObj,
};
use bsrs::core::status::{Status, StatusError};
use bsrs::core::{BsrsError, ConfigureArgs, DataKey, Document, ReadingValue};
use bsrs::engine::{DocumentSink, RunEngine};
use hklrs::nalgebra::{UnitQuaternion, Vector3};
use hklrs::presets::preset;
use hklrs::{InverseError, Lattice, ModeSpec, Orientation, Sample, Wavelength};
use hklrs_bsrs::{DiffError, Diffractometer, RealAxis, WavelengthSource, mv_hkl};
use serde_json::json;
use tokio::sync::Notify;

const AXES: [&str; 4] = ["omega", "chi", "phi", "tth"];
const LAMBDA: f64 = 1.54;
const AT_001: [f64; 4] = [30.0, 0.0, 0.0, 60.0];

fn e4cv(reals: Vec<Arc<dyn RealAxis>>) -> Arc<Diffractometer> {
    let sample = Sample::new(
        "sample",
        Lattice::cubic(LAMBDA).unwrap(),
        Orientation::Manual {
            u: UnitQuaternion::identity(),
        },
    );
    let reals = AXES.into_iter().zip(reals).collect();
    let source = WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap());
    let dx = Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals, source, sample);
    Arc::new(dx.unwrap())
}

fn soft(at: [f64; 4]) -> Vec<Arc<dyn RealAxis>> {
    AXES.iter()
        .zip(at)
        .map(|(&a, v)| Arc::new(SoftMotor::new(a, Some(v))) as Arc<dyn RealAxis>)
        .collect()
}

async fn hkl(dx: &Diffractometer) -> [f64; 3] {
    let (p, w) = (
        dx.positions().await.unwrap(),
        dx.wavelength().await.unwrap(),
    );
    dx.hkl(&p, w).unwrap().into()
}

fn assert_close(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} != {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < tolerance, "{a:?} != {b:?}");
    }
}

#[tokio::test]
async fn a_set_replaces_one_component_and_moves_every_real() {
    let dx = e4cv(soft(AT_001));
    assert_eq!(dx.mode(), Some(ModeSpec::new("bissector")));
    let [h, k, l] = dx.hkl_axes();
    assert_eq!(h.name(), "e4cv_h");

    h.set_dyn(1.0).await.await.unwrap();
    assert_close(&hkl(&dx).await, &[1.0, 0.0, 1.0], 1e-9);
    let at = dx.positions().await.unwrap();
    // |(1, 0, 1)|·λ/2a = sin 45°, and bissector puts omega at tth/2
    assert_close(&[at[0], at[3]], &[45.0, 90.0], 1e-9);
    let DynLocation { setpoint, readback } = h.locate_dyn().await.unwrap();
    assert_eq!(setpoint, 1.0);
    assert!((readback - 1.0).abs() < 1e-9);

    // the setpoint, not the readback, carries the other components
    l.set_dyn(0.0).await.await.unwrap();
    assert_close(&hkl(&dx).await, &[1.0, 0.0, 0.0], 1e-9);
    let target = dx.target().unwrap();
    assert_eq!([target[0], target[2]], [1.0, 0.0]);
    assert!(target[1].abs() < 1e-12);
    assert!(k.locate_dyn().await.unwrap().setpoint.abs() < 1e-12);
}

#[tokio::test]
async fn an_unreachable_set_fails_naming_its_target_and_moves_nothing() {
    let dx = e4cv(soft(AT_001));
    let [h, ..] = dx.hkl_axes();
    // |(3, 0, 1)|·λ/2a > 1
    let e = h.set_dyn(3.0).await.await.unwrap_err();
    let StatusError::Failed(message) = e else {
        panic!("{e:?}")
    };
    assert!(message.starts_with("e4cv_h: "), "{message}");
    assert!(message.contains("hkl [3.0, "), "{message}");
    assert!(message.contains("exceeds"), "{message}");
    assert_eq!(dx.target(), None);
    assert_eq!(dx.positions().await.unwrap(), AT_001);
}

#[tokio::test]
async fn concurrent_sets_reach_the_reals_in_setpoint_order() {
    let dx = e4cv(soft(AT_001));
    let [h, k, _] = dx.hkl_axes();
    let (a, b) = futures::join!(h.set_dyn(1.0), k.set_dyn(1.0));
    a.await.unwrap();
    b.await.unwrap();
    assert_close(&hkl(&dx).await, &[1.0, 1.0, 1.0], 1e-9);
    assert_close(&dx.target().unwrap(), &[1.0, 1.0, 1.0], 1e-12);
}

/// A soft motor whose first set waits for `release`, after signalling
/// `entered`.
struct Held {
    motor: SoftMotor,
    held: AtomicBool,
    entered: Notify,
    release: Notify,
}

impl NamedObj for Held {
    fn name(&self) -> &str {
        self.motor.name()
    }
}

#[async_trait]
impl MovableObj for Held {
    async fn set_dyn(&self, value: f64) -> Status {
        if self.held.swap(false, Ordering::SeqCst) {
            self.entered.notify_one();
            self.release.notified().await;
        }
        self.motor.set_dyn(value).await
    }
}

#[async_trait]
impl LocatableObj for Held {
    async fn locate_dyn(&self) -> Result<DynLocation, BsrsError> {
        self.motor.locate_dyn().await
    }
}

#[async_trait]
impl ReadableObj for Held {
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        self.motor.read_dyn().await
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        self.motor.describe_dyn().await
    }
}

#[tokio::test]
async fn a_pseudo_set_during_mv_hkl_leaves_the_setpoint_where_the_reals_went() {
    let omega = Arc::new(Held {
        motor: SoftMotor::new("omega", Some(AT_001[0])),
        held: AtomicBool::new(true),
        entered: Notify::new(),
        release: Notify::new(),
    });
    let mut reals = soft(AT_001);
    reals[0] = omega.clone();
    let dx = e4cv(reals);
    let [h, ..] = dx.hkl_axes();
    let re = RunEngine::new(vec![]);
    let h_set = async {
        omega.entered.notified().await;
        // mv_hkl is now part way through moving the reals
        let set = async { h.set_dyn(0.5).await.await.unwrap() };
        let release = async {
            for _ in 0..100 {
                tokio::task::yield_now().await;
            }
            omega.release.notify_one();
        };
        futures::join!(set, release);
    };
    let (result, ()) = futures::join!(re.run_async(mv_hkl(dx.clone(), [1.0, 0.0, 1.0])), h_set);
    result.unwrap();
    // the h set came second, from the setpoint mv_hkl left
    assert_close(&dx.target().unwrap(), &[0.5, 0.0, 1.0], 1e-12);
    assert_close(&hkl(&dx).await, &[0.5, 0.0, 1.0], 1e-9);
}

/// A device on a preset starts in hkl's default mode for that preset and
/// moves to an hkl in it.
#[tokio::test]
async fn a_preset_device_moves_in_hkls_default_mode() {
    for (name, default, target) in [
        ("K4CV", "bissector", [0.0, 0.0, 1.0]),
        ("K6C", "bissector_vertical", [0.0, 0.0, 1.0]),
        ("ZAXIS", "zaxis", [1.0, 0.0, 0.5]),
        ("SOLEIL SIXS MED2+3", "mu_fixed", [1.0, 0.0, 0.5]),
    ] {
        let geometry = preset(name).unwrap();
        let reals = geometry.axes().iter().map(|a| {
            let real = Arc::new(SoftMotor::new(a.name(), Some(0.0))) as Arc<dyn RealAxis>;
            (a.name(), real)
        });
        let sample = Sample::new(
            "sample",
            Lattice::cubic(LAMBDA).unwrap(),
            Orientation::Manual {
                u: UnitQuaternion::identity(),
            },
        );
        let source = WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap());
        let dx = Diffractometer::new("dx", geometry.clone(), reals.collect(), source, sample);
        let dx = Arc::new(dx.unwrap());
        assert_eq!(dx.mode(), Some(ModeSpec::new(default)), "{name}");
        let re = RunEngine::new(vec![]);
        re.run_async(mv_hkl(dx.clone(), target)).await.unwrap();
        assert_close(&hkl(&dx).await, &target, 1e-9);
    }
}

#[tokio::test]
async fn the_mode_decides_which_axes_move() {
    let dx = e4cv(soft(AT_001));
    dx.set_mode(ModeSpec::new("constant_omega")).unwrap();
    let [h, ..] = dx.hkl_axes();
    h.set_dyn(1.0).await.await.unwrap();
    assert_close(&hkl(&dx).await, &[1.0, 0.0, 1.0], 1e-9);
    // held, up to the degree-radian round trip
    assert!((dx.positions().await.unwrap()[0] - 30.0).abs() < 1e-12);

    let e = dx
        .set_mode(ModeSpec::new("bissector_vertical"))
        .unwrap_err();
    assert!(
        matches!(&e, DiffError::Mode { mode, .. } if mode == "bissector_vertical"),
        "{e:?}"
    );
    assert_eq!(
        dx.set_mode(ModeSpec::new("psi_constant")),
        Err(DiffError::Mode {
            mode: "psi_constant".into(),
            source: InverseError::MissingParameter {
                mode: "psi_constant".into(),
                parameter: "h2"
            }
        })
    );
    assert_eq!(dx.mode(), Some(ModeSpec::new("constant_omega")));

    let configure = |mode: &str| ConfigureArgs {
        values: HashMap::from([("e4cv_mode".to_owned(), json!(mode))]),
    };
    dx.configure_dyn(configure("constant_phi")).await.unwrap();
    let config = dx.read_configuration_dyn().await.unwrap();
    assert_eq!(config["e4cv_mode"].value, json!("constant_phi"));
    let e = dx.configure_dyn(configure("nope")).await.unwrap_err();
    assert!(e.to_string().contains("\"nope\""), "{e}");
}

fn psi_constant(psi: f64) -> ModeSpec {
    ModeSpec::new("psi_constant").with_psi(Vector3::z(), psi.to_radians())
}

#[tokio::test]
async fn a_psi_set_holds_hkl_and_later_hkl_sets_keep_psi() {
    let dx = e4cv(soft(AT_001));
    let [h, _, l] = dx.hkl_axes();
    h.set_dyn(1.0).await.await.unwrap();
    let held = dx.target().unwrap();
    dx.set_mode(psi_constant(60.0)).unwrap();
    let psi = dx.psi_axis();
    assert_eq!(psi.name(), "e4cv_psi");
    // before a ψ move, the mode's ψ is the setpoint
    let before = psi.locate_dyn().await.unwrap();
    assert!((before.setpoint - 60.0).abs() < 1e-12);
    assert!((before.readback - 90.0).abs() > 1.0, "{}", before.readback);

    psi.set_dyn(90.0).await.await.unwrap();
    assert_close(&hkl(&dx).await, &[1.0, 0.0, 1.0], 1e-9);
    assert_eq!(dx.target(), Some(held));
    assert_eq!(dx.mode(), Some(psi_constant(90.0)));
    let DynLocation { setpoint, readback } = psi.locate_dyn().await.unwrap();
    assert!((setpoint - 90.0).abs() < 1e-12);
    assert!((readback - 90.0).abs() < 1e-9, "{readback}");

    l.set_dyn(0.5).await.await.unwrap();
    assert_close(&hkl(&dx).await, &[1.0, 0.0, 0.5], 1e-9);
    let readback = psi.locate_dyn().await.unwrap().readback;
    assert!((readback - 90.0).abs() < 1e-9, "{readback}");
}

#[tokio::test]
async fn psi_reads_back_on_the_branch_its_setpoint_names() {
    async fn set(psi: f64) -> (Arc<Diffractometer>, DynLocation) {
        let dx = e4cv(soft(AT_001));
        let [h, ..] = dx.hkl_axes();
        h.set_dyn(1.0).await.await.unwrap();
        dx.set_mode(psi_constant(60.0)).unwrap();
        let axis = dx.psi_axis();
        axis.set_dyn(psi).await.await.unwrap();
        let location = axis.locate_dyn().await.unwrap();
        let read = axis.read_dyn().await.unwrap()["e4cv_psi"].value.clone();
        assert_eq!(read, json!(location.readback));
        (dx, location)
    }
    for (psi, principal) in [(200.0, -160.0), (-200.0, 160.0)] {
        let (dx, DynLocation { setpoint, readback }) = set(psi).await;
        assert!((setpoint - psi).abs() < 1e-9, "{setpoint}");
        assert!((readback - psi).abs() < 1e-9, "{readback}");
        // one ψ, two names: the same reals as its principal value
        let (same, location) = set(principal).await;
        assert!((location.readback - principal).abs() < 1e-9, "{location:?}");
        let at = dx.positions().await.unwrap();
        assert_close(&at, &same.positions().await.unwrap(), 1e-9);
    }
}

/// The device's ψ and the ψ axis's read, which must agree.
async fn psi_read(dx: &Arc<Diffractometer>) -> serde_json::Value {
    let device = dx.read_dyn().await.unwrap()["e4cv_psi"].value.clone();
    let axis = dx.psi_axis().read_dyn().await.unwrap()["e4cv_psi"]
        .value
        .clone();
    assert_eq!(device, axis);
    device
}

#[tokio::test]
async fn the_reading_carries_psi_as_the_psi_axis_reads_it() {
    let dx = e4cv(soft(AT_001));
    let psi = dx.psi_axis();
    // NaN outside a psi mode
    assert_eq!(psi_read(&dx).await, json!(null));
    // and where ψ is undefined: Q at (0, 0, 1) lies along the reference
    dx.set_mode(psi_constant(30.0)).unwrap();
    assert_eq!(psi_read(&dx).await, json!(null));
    assert!(psi.locate_dyn().await.is_err());

    let [h, ..] = dx.hkl_axes();
    h.set_dyn(1.0).await.await.unwrap();
    let read = psi_read(&dx).await.as_f64().unwrap();
    assert!((read - 30.0).abs() < 1e-9, "{read}");
    // on the setpoint's branch, as the axis locates it
    psi.set_dyn(200.0).await.await.unwrap();
    let read = psi_read(&dx).await.as_f64().unwrap();
    assert_eq!(read, psi.locate_dyn().await.unwrap().readback);
    assert!((read - 200.0).abs() < 1e-9, "{read}");
}

/// A detector whose reads switch the device between a psi mode and
/// bissector, as mode changes made elsewhere during a run would.
struct Switch {
    dx: Arc<Diffractometer>,
    det: SoftMotor,
    reads: AtomicUsize,
}

impl NamedObj for Switch {
    fn name(&self) -> &str {
        self.det.name()
    }
}

#[async_trait]
impl ReadableObj for Switch {
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        let mode = match self.reads.fetch_add(1, Ordering::SeqCst) % 2 {
            0 => psi_constant(30.0),
            _ => ModeSpec::new("bissector"),
        };
        self.dx.set_mode(mode).unwrap();
        self.det.read_dyn().await
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        self.det.describe_dyn().await
    }
}

#[tokio::test]
async fn a_mode_change_mid_run_keeps_every_event_on_its_descriptor() {
    let dx = e4cv(soft(AT_001));
    let [h, ..] = dx.hkl_axes();
    h.set_dyn(1.0).await.await.unwrap();
    let switch = Arc::new(Switch {
        dx: dx.clone(),
        det: SoftMotor::new("det", Some(7.0)),
        reads: AtomicUsize::new(0),
    });
    let sink = Arc::new(CapturingSink::new());
    let re = RunEngine::new(vec![sink.clone() as Arc<dyn DocumentSink>]);
    let detectors = vec![switch as Arc<dyn ReadableObj>, dx as Arc<dyn ReadableObj>];
    re.run_async(bsrs::plans::count(detectors, 4))
        .await
        .unwrap();

    let documents = sink.snapshot().await;
    let mut keys = HashMap::new();
    let mut psi = Vec::new();
    for document in &documents {
        match document {
            Document::Descriptor(d) => {
                let mut k: Vec<&String> = d.data_keys.keys().collect();
                k.sort();
                keys.insert(d.uid.clone(), k);
            }
            Document::Event(e) => {
                let mut k: Vec<&String> = e.data.keys().collect();
                k.sort();
                assert_eq!(k, keys[&e.descriptor]);
                psi.push(e.data["e4cv_psi"].clone());
            }
            _ => {}
        }
    }
    assert_eq!(psi.len(), 4);
    assert!(psi.contains(&json!(null)), "{psi:?}");
    assert!(psi.iter().any(|v| v.is_f64()), "{psi:?}");
}

#[tokio::test]
async fn an_unreachable_psi_keeps_the_mode_the_setpoint_and_the_reals() {
    let dx = e4cv(soft(AT_001));
    let [h, ..] = dx.hkl_axes();
    h.set_dyn(1.0).await.await.unwrap();
    dx.set_mode(psi_constant(60.0)).unwrap();
    let (target, at) = (dx.target(), dx.positions().await.unwrap());
    // ψ = 0 puts chi at 180°, where omega and phi turn about one line
    let e = dx.psi_axis().set_dyn(0.0).await.await.unwrap_err();
    let StatusError::Failed(message) = e else {
        panic!("{e:?}")
    };
    assert!(message.starts_with("e4cv_psi: "), "{message}");
    assert!(message.contains("hkl [1.0, "), "{message}");
    assert_eq!(dx.mode(), Some(psi_constant(60.0)));
    assert_eq!(dx.target(), target);
    assert_eq!(dx.positions().await.unwrap(), at);
}

#[tokio::test]
async fn psi_refuses_outside_a_psi_mode_and_where_it_is_undefined() {
    let dx = e4cv(soft(AT_001));
    let psi = dx.psi_axis();
    let no_psi = "mode \"bissector\" has no psi: set a psi mode";
    let e = psi.set_dyn(30.0).await.await.unwrap_err();
    let StatusError::Failed(message) = e else {
        panic!("{e:?}")
    };
    assert!(message.starts_with("e4cv_psi: "), "{message}");
    assert!(message.ends_with(no_psi), "{message}");
    let e = psi.locate_dyn().await.unwrap_err().to_string();
    assert!(e.ends_with(no_psi), "{e}");
    assert_eq!(dx.target(), None);
    assert_eq!(dx.positions().await.unwrap(), AT_001);

    // Q at (0, 0, 1) lies along the reference
    dx.set_mode(psi_constant(30.0)).unwrap();
    let e = psi.locate_dyn().await.unwrap_err().to_string();
    assert!(
        e.contains("psi about reference [0.0, 0.0, 1.0]: psi is undefined"),
        "{e}"
    );
}

#[tokio::test]
async fn generic_plans_move_and_read_the_pseudo_axes() {
    let dx = e4cv(soft(AT_001));
    let [h, _, l] = dx.hkl_axes().map(Arc::new);
    let sink = Arc::new(CapturingSink::new());
    let re = RunEngine::new(vec![sink.clone() as Arc<dyn DocumentSink>]);
    let points = vec![0.5, 1.0, -0.5];
    let plan = bsrs::plans::list_scan(
        vec![l.clone() as Arc<dyn ReadableObj>],
        h.clone(),
        h.clone(),
        points.clone(),
    );
    let result = re.run_async(plan).await.unwrap();
    assert_eq!(result.exit_status, "success");
    let events: Vec<_> = sink
        .snapshot()
        .await
        .into_iter()
        .filter_map(|d| match d {
            Document::Event(e) => Some(e),
            _ => None,
        })
        .collect();
    assert_eq!(events.len(), points.len());
    for (event, point) in events.iter().zip(&points) {
        let read = |key: &str| event.data[key].as_f64().unwrap();
        assert_close(&[read("e4cv_h"), read("e4cv_l")], &[*point, 1.0], 1e-9);
    }

    re.run_async(bsrs::plans::stubs::mv(l.clone(), 0.5))
        .await
        .unwrap();
    assert_close(&hkl(&dx).await, &[-0.5, 0.0, 0.5], 1e-9);
}

/// A soft motor that counts the pause stops it receives.
struct Counted {
    motor: SoftMotor,
    stops: AtomicUsize,
}

impl NamedObj for Counted {
    fn name(&self) -> &str {
        self.motor.name()
    }
}

#[async_trait]
impl MovableObj for Counted {
    async fn set_dyn(&self, value: f64) -> Status {
        self.motor.set_dyn(value).await
    }

    async fn stop_on_pause(&self, _success: bool) -> Result<(), BsrsError> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl LocatableObj for Counted {
    async fn locate_dyn(&self) -> Result<DynLocation, BsrsError> {
        self.motor.locate_dyn().await
    }
}

#[async_trait]
impl ReadableObj for Counted {
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        self.motor.read_dyn().await
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        self.motor.describe_dyn().await
    }
}

#[tokio::test]
async fn a_pause_stop_reaches_every_real() {
    let counted: Vec<Arc<Counted>> = AXES
        .iter()
        .zip(AT_001)
        .map(|(&a, v)| {
            Arc::new(Counted {
                motor: SoftMotor::new(a, Some(v)),
                stops: AtomicUsize::new(0),
            })
        })
        .collect();
    let dx = e4cv(
        counted
            .iter()
            .map(|c| c.clone() as Arc<dyn RealAxis>)
            .collect(),
    );
    let stops = || -> Vec<usize> {
        counted
            .iter()
            .map(|c| c.stops.load(Ordering::SeqCst))
            .collect()
    };
    let [_, k, _] = dx.hkl_axes();
    k.stop_on_pause(true).await.unwrap();
    assert_eq!(stops(), [1; 4]);
    // the engine stops what a run set when the run ends
    let re = RunEngine::new(vec![]);
    re.run_async(mv_hkl(dx.clone(), [1.0, 0.0, 1.0]))
        .await
        .unwrap();
    assert_eq!(stops(), [2; 4]);
}
