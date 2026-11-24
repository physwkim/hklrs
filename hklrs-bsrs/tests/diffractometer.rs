//! The Diffractometer device on E4CV soft motors. A cubic cell with a = λ
//! puts (1, 0, 0) at omega 30, chi 0, phi 90, tth 60 for U = I.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use bsrs::backends::soft::SoftMotor;
use bsrs::callbacks::CapturingSink;
use bsrs::core::msg::{ConfigurableObj, MovableObj, ReadableObj};
use bsrs::core::{ConfigureArgs, Document};
use bsrs::engine::{DocumentSink, RunEngine};
use hklrs::nalgebra::{Matrix3, UnitQuaternion};
use hklrs::presets::preset;
use hklrs::{InverseError, Lattice, ModeSpec, Orientation, Sample, Wavelength};
use hklrs_bsrs::{DiffError, Diffractometer, Measured, RealAxis, WavelengthSource};
use serde_json::{Value, json};

const AXES: [&str; 4] = ["omega", "chi", "phi", "tth"];
const LAMBDA: f64 = 1.54;

fn wavelength() -> Wavelength {
    Wavelength::new(LAMBDA).unwrap()
}

fn motors(at: [f64; 4]) -> Vec<Arc<SoftMotor>> {
    AXES.iter()
        .zip(at)
        .map(|(&a, v)| Arc::new(SoftMotor::new(a, Some(v))))
        .collect()
}

fn cubic(name: &str, u: UnitQuaternion<f64>) -> Sample {
    Sample::new(
        name,
        Lattice::cubic(LAMBDA).unwrap(),
        Orientation::Manual { u },
    )
}

fn e4cv_with(
    motors: &[Arc<SoftMotor>],
    source: WavelengthSource,
    sample: Sample,
) -> Diffractometer {
    let reals = AXES
        .iter()
        .zip(motors)
        .map(|(&a, m)| (a, m.clone() as Arc<dyn RealAxis>))
        .collect();
    Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals, source, sample).unwrap()
}

fn e4cv(motors: &[Arc<SoftMotor>]) -> Diffractometer {
    let sample = cubic("sample", UnitQuaternion::identity());
    e4cv_with(motors, WavelengthSource::Fixed(wavelength()), sample)
}

fn number(reading: &HashMap<String, bsrs::core::ReadingValue>, key: &str) -> f64 {
    reading[key].value.as_f64().unwrap()
}

fn assert_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{a:?} != {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-12, "{a:?} != {b:?}");
    }
}

#[tokio::test]
async fn read_gives_readbacks_and_the_hkl_they_reach() {
    let m = motors([30.0, 0.0, 90.0, 60.0]);
    let dx = e4cv(&m);
    let r = dx.read_dyn().await.unwrap();
    assert_eq!(r.len(), 8);
    // E4CV has a psi mode, but bissector has no ψ
    assert_eq!(r["e4cv_psi"].value, Value::Null);
    let hkl = dx.hkl_keys().map(|k| number(&r, &k));
    assert_close(&hkl, &[1.0, 0.0, 0.0]);
    let axes: Vec<f64> = dx.axis_keys().iter().map(|k| number(&r, k)).collect();
    assert_close(&axes, &[30.0, 0.0, 90.0, 60.0]);

    m[2].set_dyn(0.0).await.await.unwrap();
    let r = dx.read_dyn().await.unwrap();
    assert_close(&dx.hkl_keys().map(|k| number(&r, &k)), &[0.0, 0.0, 1.0]);
}

#[tokio::test]
async fn describe_has_every_read_key_with_its_units() {
    let dx = e4cv(&motors([30.0, 0.0, 90.0, 60.0]));
    let d = dx.describe_dyn().await.unwrap();
    let mut read: Vec<String> = dx.read_dyn().await.unwrap().into_keys().collect();
    let mut described: Vec<String> = d.keys().cloned().collect();
    read.sort();
    described.sort();
    assert_eq!(read, described);
    assert_eq!(d["e4cv_tth"].units.as_deref(), Some("deg"));
    assert_eq!(d["e4cv_h"].units, None);
    assert_eq!(d["e4cv_psi"].units.as_deref(), Some("deg"));
    assert_eq!(dx.hint_fields().unwrap(), dx.hkl_keys().to_vec());
}

/// ψ is read on a geometry with any psi mode, one (E4CV) or two (E6C), and
/// not on one with none (ZAXIS).
#[tokio::test]
async fn psi_is_read_on_a_geometry_with_a_psi_mode() {
    for (geometry, psi) in [("E6C", true), ("ZAXIS", false)] {
        let geometry = preset(geometry).unwrap();
        let reals = geometry.axes().iter().map(|a| {
            let real = Arc::new(SoftMotor::new(a.name(), Some(0.0))) as Arc<dyn RealAxis>;
            (a.name(), real)
        });
        let source = WavelengthSource::Fixed(wavelength());
        let sample = cubic("sample", UnitQuaternion::identity());
        let reals = reals.collect();
        let dx = Diffractometer::new("dx", geometry.clone(), reals, source, sample).unwrap();
        let read = dx.read_dyn().await.unwrap();
        let described = dx.describe_dyn().await.unwrap();
        assert_eq!(read.contains_key("dx_psi"), psi, "{}", geometry.name());
        let mut read: Vec<String> = read.into_keys().collect();
        let mut described: Vec<String> = described.into_keys().collect();
        read.sort();
        described.sort();
        assert_eq!(read, described);
    }
}

#[test]
fn reals_must_cover_each_axis_once() {
    let m = motors([0.0; 4]);
    let real = |i: usize| m[i].clone() as Arc<dyn RealAxis>;
    let sample = || cubic("sample", UnitQuaternion::identity());
    let source = || WavelengthSource::Fixed(wavelength());
    let build =
        |reals| Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals, source(), sample());

    let missing = build(vec![("omega", real(0)), ("chi", real(1)), ("tth", real(3))]);
    assert_eq!(
        missing.err(),
        Some(DiffError::MissingAxis { name: "phi".into() })
    );
    let twice = build(vec![
        ("omega", real(0)),
        ("chi", real(1)),
        ("chi", real(2)),
        ("tth", real(3)),
    ]);
    assert_eq!(
        twice.err(),
        Some(DiffError::DuplicateAxis { name: "chi".into() })
    );
    let unknown = build(vec![("kappa", real(0))]);
    assert_eq!(
        unknown.err(),
        Some(DiffError::Core(hklrs::Error::UnknownAxis {
            name: "kappa".into()
        }))
    );
    // any order is taken
    let shuffled = build(vec![
        ("tth", real(3)),
        ("phi", real(2)),
        ("chi", real(1)),
        ("omega", real(0)),
    ]);
    assert_eq!(shuffled.unwrap().reals()[3].name(), "tth");
}

fn measured(hkl: [f64; 3], positions: [f64; 4]) -> Measured {
    Measured {
        hkl,
        positions: positions.to_vec(),
        wavelength: wavelength(),
    }
}

#[tokio::test]
async fn compute_ub_orients_the_sample_from_two_reflections() {
    let m = motors([30.0, 0.0, 90.0, 60.0]);
    let sample = cubic("sample", UnitQuaternion::from_euler_angles(0.3, -0.7, 1.1));
    let dx = e4cv_with(&m, WavelengthSource::Fixed(wavelength()), sample);
    dx.add_reflection("r1", measured([0.0, 0.0, 1.0], [30.0, 0.0, 0.0, 60.0]))
        .unwrap();
    dx.add_reflection("r2", measured([0.0, 1.0, 0.0], [30.0, 90.0, 0.0, 60.0]))
        .unwrap();
    let ub = dx.compute_ub("r1", "r2").unwrap();
    let expected = Matrix3::identity() * (TAU / LAMBDA);
    assert!((ub - expected).abs().max() < 1e-12, "{ub}");
    let r = dx.read_dyn().await.unwrap();
    assert_close(&dx.hkl_keys().map(|k| number(&r, &k)), &[1.0, 0.0, 0.0]);

    assert_eq!(
        dx.add_reflection("r1", measured([1.0, 0.0, 0.0], [30.0, 0.0, 90.0, 60.0])),
        Err(DiffError::DuplicateReflection {
            sample: "sample".into(),
            name: "r1".into()
        })
    );
}

/// PY2-6: hklpy2 `calc_UB` reports "None unknown" for a bad reflection name.
#[test]
fn py2_6_errors_name_what_they_reject() {
    let dx = e4cv(&motors([0.0; 4]));
    dx.add_reflection("r1", measured([0.0, 0.0, 1.0], [30.0, 0.0, 0.0, 60.0]))
        .unwrap();
    for (primary, secondary) in [("r1", "r9"), ("r9", "r1")] {
        let e = dx.compute_ub(primary, secondary).unwrap_err();
        assert_eq!(
            e,
            DiffError::UnknownReflection {
                sample: "sample".into(),
                name: "r9".into(),
                known: vec!["r1".into()],
            }
        );
        assert!(e.to_string().contains("\"r9\""), "{e}");
    }
    let e = dx.select_sample("ghost").unwrap_err();
    assert!(e.to_string().contains("\"ghost\""), "{e}");
}

/// PY2-3: hklpy2 `remove_sample` of a non-current sample switches the
/// current sample to the first one.
#[test]
fn py2_3_removing_another_sample_keeps_the_current_one() {
    let dx = e4cv(&motors([0.0; 4]));
    for name in ["A", "B"] {
        dx.add_sample(cubic(name, UnitQuaternion::identity()))
            .unwrap();
    }
    dx.select_sample("A").unwrap();
    dx.remove_sample("B").unwrap();
    assert_eq!(dx.current_sample().name, "A");
    assert_eq!(dx.sample_names(), ["A", "sample"]);

    // the current one removed: the first remaining becomes current
    dx.remove_sample("A").unwrap();
    assert_eq!(dx.current_sample().name, "sample");
    assert_eq!(
        dx.remove_sample("sample"),
        Err(DiffError::LastSample {
            name: "sample".into()
        })
    );
    assert_eq!(
        dx.add_sample(cubic("sample", UnitQuaternion::identity())),
        Err(DiffError::DuplicateSample {
            name: "sample".into()
        })
    );
}

/// PY2-4: hklpy2's `remove_sample` documents no error for an unknown name
/// and raises `KeyError`.
#[test]
fn py2_4_removing_an_unknown_sample_is_an_error() {
    let dx = e4cv(&motors([0.0; 4]));
    dx.add_sample(cubic("A", UnitQuaternion::identity()))
        .unwrap();
    dx.select_sample("A").unwrap();
    assert_eq!(
        dx.remove_sample("B"),
        Err(DiffError::UnknownSample {
            name: "B".into(),
            known: vec!["A".into(), "sample".into()],
        })
    );
    assert_eq!(dx.sample_names(), ["A", "sample"]);
    assert_eq!(dx.current_sample().name, "A");
}

/// PY2-2: hklpy2's `core.mode = "nonsense"` returns normally and keeps the
/// old mode.
#[test]
fn py2_2_an_unknown_mode_is_an_error_and_keeps_the_mode() {
    let dx = e4cv(&motors([0.0; 4]));
    dx.set_mode(ModeSpec::new("constant_phi")).unwrap();
    let unknown = Err(DiffError::Mode {
        mode: "nonsense".into(),
        source: InverseError::UnknownMode {
            geometry: "E4CV".into(),
            name: "nonsense".into(),
        },
    });
    assert_eq!(dx.set_mode(ModeSpec::new("nonsense")), unknown);
    assert_eq!(dx.mode(), Some(ModeSpec::new("constant_phi")));
    let values = HashMap::from([("e4cv_mode".to_owned(), json!("nonsense"))]);
    assert_eq!(dx.configure(&values), unknown);
    assert_eq!(dx.mode(), Some(ModeSpec::new("constant_phi")));
}

/// PY2-14: hklpy2's `set_solver` keeps the previous geometry's mode, and
/// every solve then fails on it. A device's mode is one its geometry
/// builds: E6C's default is its own, another geometry's mode or
/// configuration is refused, and the device still solves after.
#[test]
fn py2_14_the_mode_is_always_one_the_geometry_builds() {
    let geometry = preset("E6C").unwrap();
    let reals = geometry
        .axes()
        .iter()
        .map(|a| {
            let real = Arc::new(SoftMotor::new(a.name(), Some(0.0))) as Arc<dyn RealAxis>;
            (a.name(), real)
        })
        .collect();
    let source = WavelengthSource::Fixed(wavelength());
    let sample = cubic("sample", UnitQuaternion::identity());
    let dx = Diffractometer::new("e6c", geometry.clone(), reals, source, sample).unwrap();
    let default = Some(ModeSpec::new("bissector_vertical"));
    assert_eq!(dx.mode(), default);
    let solves = || dx.inverse([1.0, 0.0, 0.0], &[0.0; 6], wavelength()).is_ok();
    assert!(solves());
    assert!(matches!(
        dx.set_mode(ModeSpec::new("bissector")),
        Err(DiffError::Mode { .. })
    ));
    let e4cv_config = e4cv(&motors([0.0; 4])).config();
    assert!(matches!(
        dx.load_config(&e4cv_config),
        Err(DiffError::ConfigGeometry { .. })
    ));
    assert_eq!(dx.mode(), default);
    assert!(solves());
}

/// HKL-4: hkl's `UB_set` documents B as kept and fits the lattice: a = 5
/// and UB = (2π/6)·R read back a = 6. `set_ub` keeps the lattice and
/// refuses a UB that is no rotation of it.
#[test]
fn hkl_4_set_ub_keeps_the_lattice() {
    let lattice = Lattice::cubic(5.0).unwrap();
    let sample = Sample::new(
        "sample",
        lattice,
        Orientation::Manual {
            u: UnitQuaternion::identity(),
        },
    );
    let dx = e4cv_with(
        &motors([0.0; 4]),
        WavelengthSource::Fixed(wavelength()),
        sample,
    );
    let r = UnitQuaternion::from_euler_angles(0.3, -0.4, 1.2).to_rotation_matrix();
    let before = dx.current_sample();
    let e = dx.set_ub(&(TAU / 6.0 * r.matrix())).unwrap_err();
    assert!(
        matches!(e, DiffError::Core(hklrs::Error::NotRotation { .. })),
        "{e}"
    );
    assert_eq!(dx.current_sample(), before);
    let ub = r.matrix() * lattice.b_matrix();
    dx.set_ub(&ub).unwrap();
    let after = dx.current_sample();
    assert_eq!(after.lattice, lattice);
    assert!((after.ub() - ub).abs().max() < 1e-12);
}

#[tokio::test]
async fn configuration_reaches_the_descriptor() {
    let dx: Arc<dyn ReadableObj> = Arc::new(e4cv(&motors([30.0, 0.0, 90.0, 60.0])));
    let sink = Arc::new(CapturingSink::new());
    let re = RunEngine::new(vec![sink.clone() as Arc<dyn DocumentSink>]);
    let result = re.run_async(bsrs::plans::count(vec![dx], 1)).await.unwrap();
    assert_eq!(result.exit_status, "success");
    let docs = sink.snapshot().await;
    let descriptor = docs
        .iter()
        .find_map(|d| match d {
            Document::Descriptor(d) => Some(d),
            _ => None,
        })
        .unwrap();
    let config = &descriptor.configuration["e4cv"];
    let flat = |key: &str| -> Vec<f64> {
        let v = &config.data[key];
        let rows = v.as_array().unwrap().iter();
        rows.flat_map(|x| x.as_array().cloned().unwrap_or_else(|| vec![x.clone()]))
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let k = TAU / LAMBDA;
    assert_close(&flat("e4cv_UB"), &[k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, k]);
    assert_close(
        &flat("e4cv_U"),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    assert_close(
        &flat("e4cv_lattice"),
        &[LAMBDA, LAMBDA, LAMBDA, 90.0, 90.0, 90.0],
    );
    assert_eq!(config.data["e4cv_geometry"], json!("E4CV"));
    assert_eq!(config.data["e4cv_sample"], json!("sample"));
    assert_eq!(config.data["e4cv_wavelength"], json!(LAMBDA));
    assert_eq!(config.data["e4cv_mode"], json!("bissector"));
    let mut keys: Vec<&String> = config.data_keys.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "e4cv_U",
            "e4cv_UB",
            "e4cv_geometry",
            "e4cv_lattice",
            "e4cv_mode",
            "e4cv_sample",
            "e4cv_wavelength"
        ]
    );
    assert!(descriptor.data_keys.contains_key("e4cv_h"));
}

fn args(pairs: &[(&str, Value)]) -> ConfigureArgs {
    ConfigureArgs {
        values: pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

#[tokio::test]
async fn configure_applies_all_or_nothing() {
    let dx = e4cv(&motors([30.0, 0.0, 90.0, 60.0]));
    dx.add_sample(cubic("A", UnitQuaternion::identity()))
        .unwrap();

    let e = dx
        .configure_dyn(args(&[
            ("e4cv_sample", json!("A")),
            ("e4cv_wavelength", json!(-1.0)),
        ]))
        .await
        .unwrap_err();
    assert!(e.to_string().contains("-1"), "{e}");
    assert_eq!(dx.current_sample().name, "sample");

    let e = dx
        .configure_dyn(args(&[("e4cv_UB", json!(1.0))]))
        .await
        .unwrap_err();
    assert!(e.to_string().contains("\"e4cv_UB\""), "{e}");
    assert_eq!(
        dx.configure(&args(&[("e4cv_wavelength", json!("1.0"))]).values),
        Err(DiffError::ConfigValue {
            key: "e4cv_wavelength".into(),
            value: json!("1.0"),
            expected: "number"
        })
    );

    dx.configure_dyn(args(&[
        ("e4cv_sample", json!("A")),
        ("e4cv_wavelength", json!(1.0)),
    ]))
    .await
    .unwrap();
    assert_eq!(dx.current_sample().name, "A");
    assert_eq!(dx.wavelength().await.unwrap().angstrom(), 1.0);
}

#[tokio::test]
async fn wavelength_signal_is_read_and_not_set() {
    let m = motors([30.0, 0.0, 90.0, 60.0]);
    let signal = Arc::new(SoftMotor::new("mono", Some(LAMBDA)));
    let sample = cubic("sample", UnitQuaternion::identity());
    let dx = e4cv_with(&m, WavelengthSource::Signal(signal.clone()), sample);
    let r = dx.read_dyn().await.unwrap();
    assert_close(&dx.hkl_keys().map(|k| number(&r, &k)), &[1.0, 0.0, 0.0]);
    assert_eq!(
        dx.set_wavelength(wavelength()),
        Err(DiffError::WavelengthFromSignal {
            signal: "mono".into()
        })
    );

    // the signal moves and the hkl follows: |hkl| = a/λ at the same angles
    signal.set_dyn(2.0 * LAMBDA).await.await.unwrap();
    let r = dx.read_dyn().await.unwrap();
    assert_close(&dx.hkl_keys().map(|k| number(&r, &k)), &[0.5, 0.0, 0.0]);

    signal.set_dyn(-1.0).await.await.unwrap();
    let e = dx.read_dyn().await.unwrap_err();
    assert!(e.to_string().contains("-1"), "{e}");

    // a readable with more than one field is not a wavelength
    let several: Arc<dyn ReadableObj> = Arc::new(e4cv(&motors([0.0; 4])));
    let dx = e4cv_with(
        &m,
        WavelengthSource::Signal(several),
        cubic("sample", UnitQuaternion::identity()),
    );
    let e = dx.wavelength().await.unwrap_err();
    assert!(e.to_string().contains("e4cv_h"), "{e}");
}

#[test]
fn limits_and_cut_points_reach_the_inverse() {
    let dx = e4cv(&motors([0.0; 4]));
    let seed = [-30.0, 0.0, -90.0, -60.0];
    let solve = || dx.inverse([1.0, 0.0, 0.0], &seed, wavelength()).unwrap();
    assert_close(&solve(), &seed);
    dx.set_cut_point("phi", 0.0).unwrap();
    assert_close(&solve(), &[-30.0, 0.0, 270.0, -60.0]);
    dx.set_limits("tth", 0.0, 180.0).unwrap();
    assert_close(&solve(), &[-150.0, 0.0, 270.0, 60.0]);

    for (min, max) in [(10.0, -10.0), (f64::NAN, 10.0)] {
        let e = dx.set_limits("tth", min, max).unwrap_err();
        let DiffError::Limits { name, .. } = &e else {
            panic!("{e}")
        };
        assert_eq!(name, "tth");
        assert!(e.to_string().contains("10]"), "{e}");
    }
    let geometry = dx.geometry();
    assert_eq!(geometry.axes()[3].limits(), (0.0, 180.0_f64.to_radians()));
    for e in [dx.set_limits("nu", 0.0, 1.0), dx.set_cut_point("nu", 0.0)] {
        let e = e.unwrap_err();
        assert!(e.to_string().contains("\"nu\""), "{e}");
    }
}
