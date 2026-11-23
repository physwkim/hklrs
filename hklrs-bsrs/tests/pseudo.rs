//! h, k, l pseudo axes on E4CV soft motors: cubic a = λ = 1.54, U = I, so
//! (0, 0, 1) is at omega 30, chi 0, phi 0, tth 60.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use bsrs::backends::soft::SoftMotor;
use bsrs::callbacks::CapturingSink;
use bsrs::core::msg::{
    ConfigurableObj, DynLocation, LocatableObj, MovableObj, NamedObj, ReadableObj,
};
use bsrs::core::status::{Status, StatusError};
use bsrs::core::{BsrsError, ConfigureArgs, DataKey, Document, ReadingValue};
use bsrs::engine::{DocumentSink, RunEngine};
use hklrs::nalgebra::UnitQuaternion;
use hklrs::presets::preset;
use hklrs::{InverseError, Lattice, ModeSpec, Orientation, Sample, Wavelength};
use hklrs_bsrs::{DiffError, Diffractometer, RealAxis, WavelengthSource};
use serde_json::json;

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
            source: InverseError::ModeParameter {
                mode: "psi_constant".into(),
                needs: "reference and psi"
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
    let [_, k, _] = dx.hkl_axes();
    k.stop_on_pause(true).await.unwrap();
    let stops: Vec<usize> = counted
        .iter()
        .map(|c| c.stops.load(Ordering::SeqCst))
        .collect();
    assert_eq!(stops, [1; 4]);
}
