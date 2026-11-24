//! Configuration save and load, and hklpy2 import against
//! golden/hklpy2_config.py: hklpy2 (352700b6) exports an E4CV with a
//! hexagonal sample oriented from two reflections, limits and cut points;
//! imported here it gives hklpy2's UB and forward solutions.

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bsrs::backends::soft::SoftMotor;
use bsrs::core::msg::ReadableObj;
use bsrs::engine::RunEngine;
use bsrs::host::manager_lua::ManagerLuaState;
use bsrs::qs::{LuaEvaluator, Registry};
use hklrs::nalgebra::{UnitQuaternion, Vector3};
use hklrs::presets::preset;
use hklrs::{InverseRequest, Lattice, ModeSpec, Orientation, Sample, Target, Wavelength};
use hklrs_bsrs::{
    Config, DiffError, Diffractometer, LuaDiffractometer, Measured, RealAxis, WavelengthSource,
};
use serde_json::{Map, Value};
use tokio::sync::Mutex;

const AXES: [&str; 4] = ["omega", "chi", "phi", "tth"];
const LAMBDA: f64 = 1.54;
/// hkl solves numerically, to about 2e-7 deg.
const DEG: f64 = 1e-5;

fn golden(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../golden")
        .join(file)
}

fn read(file: &str) -> String {
    std::fs::read_to_string(golden(file)).unwrap()
}

fn e4cv_with(wavelength: WavelengthSource) -> Arc<Diffractometer> {
    let reals = AXES
        .map(|a| {
            (
                a,
                Arc::new(SoftMotor::new(a, Some(0.0))) as Arc<dyn RealAxis>,
            )
        })
        .into();
    let u = UnitQuaternion::identity();
    let sample = Sample::new(
        "cubic",
        Lattice::cubic(LAMBDA).unwrap(),
        Orientation::Manual { u },
    );
    let dx = Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals, wavelength, sample);
    Arc::new(dx.unwrap())
}

fn e4cv() -> Arc<Diffractometer> {
    e4cv_with(WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap()))
}

fn imported(file: &str) -> Arc<Diffractometer> {
    let dx = e4cv();
    dx.load_config(&Config::read(&golden(file)).unwrap())
        .unwrap();
    dx
}

fn matrix(v: &Value) -> [[f64; 3]; 3] {
    serde_json::from_value(v.clone()).unwrap()
}

fn assert_matrix(found: [[f64; 3]; 3], expected: [[f64; 3]; 3], tolerance: f64) {
    for (f, e) in found.iter().flatten().zip(expected.iter().flatten()) {
        assert!((f - e).abs() < tolerance, "{found:?} vs {expected:?}");
    }
}

fn ub(dx: &Diffractometer) -> [[f64; 3]; 3] {
    let ub = dx.current_sample().ub();
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| ub[(i, j)]))
}

fn u(dx: &Diffractometer) -> [[f64; 3]; 3] {
    let u = dx.current_sample().u().to_rotation_matrix().into_inner();
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| u[(i, j)]))
}

/// Every solution in degrees, wrapped to the cut points and within the
/// limits.
fn solutions(dx: &Diffractometer, hkl: [f64; 3]) -> Vec<Vec<f64>> {
    let geometry = dx.geometry();
    let mode = dx.mode().unwrap().build(&geometry).unwrap();
    let current = vec![0.0; AXES.len()];
    let request = InverseRequest {
        mode: &mode,
        ub: dx.current_sample().ub(),
        wavelength: Wavelength::new(LAMBDA).unwrap(),
        target: Target::Hkl(Vector3::from(hkl)),
        current: &current,
    };
    let solved = request.solve().unwrap();
    let degrees = solved
        .iter()
        .map(|s| geometry.values_to_degrees(s).unwrap());
    degrees.collect()
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < DEG)
}

/// hklpy2's UB, U·B of the imported sample, and its forward solutions: the
/// set hklrs finds after limits and cut points is the set hklpy2 keeps.
fn check_case(index: usize, mode: ModeSpec) {
    let doc: Value = serde_json::from_str(&read("hklpy2_config.json")).unwrap();
    let case = &doc["cases"][index];
    let dx = imported(case["file"].as_str().unwrap());
    assert_eq!(dx.current_sample().name, "hexagonal");
    assert_eq!(dx.mode(), Some(mode));
    assert_matrix(ub(&dx), matrix(&case["UB"]), 1e-9);
    assert_matrix(u(&dx), matrix(&case["U"]), 1e-9);
    for target in case["forward"].as_array().unwrap() {
        let hkl: [f64; 3] = serde_json::from_value(target["hkl"].clone()).unwrap();
        let expected: Vec<Vec<f64>> = serde_json::from_value(target["solutions"].clone()).unwrap();
        let found = solutions(&dx, hkl);
        assert_eq!(
            found.len(),
            expected.len(),
            "{hkl:?}: {found:?} vs {expected:?}"
        );
        for e in &expected {
            assert!(
                found.iter().any(|f| same(f, e)),
                "{hkl:?}: {e:?} not in {found:?}"
            );
        }
        // the device move: one of them
        let current = [0.0; 4];
        let wavelength = Wavelength::new(LAMBDA).unwrap();
        let moved = dx.inverse(hkl, &current, wavelength).unwrap();
        assert!(
            expected.iter().any(|e| same(&moved, e)),
            "{hkl:?}: {moved:?}"
        );
    }
}

#[test]
fn hklpy2_bissector_gives_hklpy2_ub_and_solutions() {
    check_case(0, ModeSpec::new("bissector"));
}

#[test]
fn hklpy2_psi_constant_takes_its_extras() {
    let mode = ModeSpec {
        psi: Some((Vector3::new(0.0, 0.0, 1.0), 30f64.to_radians())),
        ..ModeSpec::new("psi_constant")
    };
    check_case(1, mode);
}

#[test]
fn hklpy2_reflections_orient_to_hklpy2_ub() {
    let doc: Value = serde_json::from_str(&read("hklpy2_config.json")).unwrap();
    let dx = imported("hklpy2_e4cv.yml");
    // hklpy2 records the angles as angstrom (PY2-13); they are degrees
    let ub = dx.compute_ub("r100", "r001").unwrap();
    let ub = [0, 1, 2].map(|i| [0, 1, 2].map(|j| ub[(i, j)]));
    assert_matrix(ub, matrix(&doc["cases"][0]["UB"]), 1e-9);
}

#[test]
fn hklpy2_limits_and_cut_points_carry_over() {
    let geometry = imported("hklpy2_e4cv.yml").geometry();
    let expected = [
        ("omega", -120.0, 120.0, -180.0),
        ("chi", -100.0, 100.0, -180.0),
        ("phi", 0.0, 360.0, 0.0),
        ("tth", -5.0, 170.0, -180.0),
    ];
    for (axis, (name, min, max, cut)) in geometry.axes().iter().zip(expected) {
        assert_eq!(axis.name(), name);
        let (lo, hi) = axis.limits();
        assert_eq!([lo, hi], [min, max].map(f64::to_radians), "{name}");
        assert_eq!(axis.cut_point(), Some(f64::to_radians(cut)), "{name}");
    }
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// hklpy2_e4cv.yml with its hexagonal sample, whose UB is U·B and the UB of
/// its orienting reflections r100 and r001, edited.
fn hexagonal(edit: impl FnOnce(&mut Map<String, Value>)) -> String {
    let mut doc: Value = serde_saphyr::from_str(&read("hklpy2_e4cv.yml")).unwrap();
    edit(doc["samples"]["hexagonal"].as_object_mut().unwrap());
    serde_saphyr::to_string(&doc).unwrap()
}

fn set(sample: &mut Map<String, Value>, field: &str, value: impl serde::Serialize) {
    sample.insert(field.into(), serde_json::to_value(value).unwrap());
}

/// The hexagonal sample of the edited file, imported.
fn hexagonal_imported(edit: impl FnOnce(&mut Map<String, Value>)) -> (Config, Arc<Diffractometer>) {
    let config = Config::from_hklpy2(&hexagonal(edit)).unwrap();
    let dx = e4cv();
    dx.load_config(&config).unwrap();
    assert_eq!(dx.current_sample().name, "hexagonal");
    (config, dx)
}

/// hklpy2's UB and U from r100 and r001.
fn oriented() -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let doc: Value = serde_json::from_str(&read("hklpy2_config.json")).unwrap();
    let case = &doc["cases"][0];
    (matrix(&case["UB"]), matrix(&case["U"]))
}

#[test]
fn hklpy2_ub_matching_the_reflections_replaces_a_stale_u() {
    // hklpy2's UB setter leaves U as it was (PY2-21)
    let (_, dx) = hexagonal_imported(|s| set(s, "U", IDENTITY));
    let (expected_ub, expected_u) = oriented();
    assert_matrix(ub(&dx), expected_ub, 1e-9);
    assert_matrix(u(&dx), expected_u, 1e-9);
}

#[test]
fn hklpy2_u_matching_the_reflections_is_kept_over_a_stale_ub() {
    // hklpy2's U setter leaves UB as it was (PY2-21)
    let (config, dx) = hexagonal_imported(|s| set(s, "UB", IDENTITY));
    let file = hexagonal(|_| ());
    let file: Value = serde_saphyr::from_str(&file).unwrap();
    let file_u: [[f64; 3]; 3] =
        serde_json::from_value(file["samples"]["hexagonal"]["U"].clone()).unwrap();
    let sample = config.samples.iter().find(|s| s.name == "hexagonal");
    assert_eq!(sample.unwrap().u, file_u);
    assert_matrix(ub(&dx), oriented().0, 1e-9);
}

fn ub_rejection(text: &str) -> (String, DiffError) {
    let e = Config::from_hklpy2(text).unwrap_err();
    let message = e.to_string();
    let DiffError::InSample { sample, source } = e else {
        panic!("{message}")
    };
    assert_eq!(sample, "hexagonal");
    (message, *source)
}

#[test]
fn hklpy2_stale_ub_without_two_orienting_reflections_is_rejected() {
    let empty: [&str; 0] = [];
    let none = hexagonal(|s| {
        set(s, "UB", IDENTITY);
        set(s, "reflections", Map::new());
        set(s, "reflections_order", empty);
    });
    let one = hexagonal(|s| {
        set(s, "UB", IDENTITY);
        s["reflections"].as_object_mut().unwrap().remove("r001");
        set(s, "reflections_order", ["r100"]);
    });
    // hklpy2 orients from the first two of reflections_order only
    let unordered = hexagonal(|s| {
        set(s, "UB", IDENTITY);
        set(s, "reflections_order", empty);
    });
    for text in [none, one, unordered] {
        let (message, e) = ub_rejection(&text);
        let DiffError::Hklpy2UbUndecided { ub, u_b } = e else {
            panic!("{message}")
        };
        assert_eq!(*ub, IDENTITY);
        assert_matrix(*u_b, oriented().0, 1e-9);
        let start = "sample \"hexagonal\": hklpy2 file is self-inconsistent: \
                     its UB [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] is not U·B [[";
        assert!(message.starts_with(start), "{message}");
    }
}

#[test]
fn hklpy2_ub_and_u_both_unlike_the_reflections_are_rejected() {
    let quarter_turn = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let text = hexagonal(|s| {
        set(s, "UB", IDENTITY);
        set(s, "U", quarter_turn);
    });
    let (message, e) = ub_rejection(&text);
    let DiffError::Hklpy2UbUnmatched {
        ub,
        reflections,
        measured,
        ..
    } = e
    else {
        panic!("{message}")
    };
    assert_eq!(*ub, IDENTITY);
    assert_eq!(reflections, ["r100", "r001"]);
    assert_matrix(*measured, oriented().0, 1e-9);
}

#[test]
fn hklpy2_test_file_is_rejected_for_a_sample_without_reflections() {
    // hklpy2's own test file (written by 0.6.2.dev4) stores UB = U = I for
    // its a = 1 default sample, not U·B = 2π·I, and no reflections
    let e = Config::read(&golden("hklpy2_e4cv_orient.yml")).unwrap_err();
    let message = e.to_string();
    let DiffError::InSample { sample, source } = e else {
        panic!("{message}")
    };
    assert_eq!(sample, "sample");
    let DiffError::Hklpy2UbUndecided { ub, u_b } = *source else {
        panic!("{message}")
    };
    assert_eq!(*ub, IDENTITY);
    assert_matrix(*u_b, IDENTITY.map(|row| row.map(|x| x * TAU)), 1e-12);
    assert!(
        message.starts_with(
            "sample \"sample\": hklpy2 file is self-inconsistent: its UB \
             [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] is not U·B [[6.28318530717958"
        ),
        "{message}"
    );
}

fn edited(file: &str, from: &str, to: &str) -> String {
    let text = read(file);
    assert_eq!(text.matches(from).count(), 1, "{from}");
    text.replace(from, to)
}

#[test]
fn hklpy2_errors_name_what_they_reject() {
    let cases = [
        (
            edited(
                "hklpy2_e4cv.yml",
                "config_schema_version: 3",
                "config_schema_version: 2",
            ),
            "hklpy2 YAML configuration schema version 2 is not 3",
        ),
        (
            edited(
                "hklpy2_e4cv_psi.yml",
                "  bissector: {}",
                "  bissector: {chi: 90.0}",
            ),
            "hklpy2 presets.bissector = {\"chi\": 90.0} is not supported",
        ),
        (
            edited("hklpy2_e4cv.yml", "  engine: hkl\n", "  engine: psi\n"),
            "hklpy2 solver.engine = \"psi\" is not supported",
        ),
        (
            edited(
                "hklpy2_e4cv.yml",
                "    cut_point: 0.0\n    class: LimitsConstraint",
                "    cut_point: 0.0\n    class: OtherConstraint",
            ),
            "hklpy2 constraints.phi.class = \"OtherConstraint\" is not supported",
        ),
        (
            edited("hklpy2_e4cv.yml", "    tth: tth\n", "    tth: ttth\n"),
            "configuration geometry \"E4CV\" with axes [\"omega\", \"chi\", \"phi\", \"ttth\"] \
             is not the device's \"E4CV\" with axes [\"omega\", \"chi\", \"phi\", \"tth\"]",
        ),
    ];
    for (text, message) in cases {
        let e = Config::from_hklpy2(&text).and_then(|c| e4cv().load_config(&c));
        assert_eq!(e.unwrap_err().to_string(), message);
    }
    let text = edited(
        "hklpy2_e4cv.yml",
        "          tth: 17.717608294742977\n        reals_units: angstrom",
        "          tth: 17.717608294742977\n        reals_units: mm",
    );
    assert_eq!(
        Config::from_hklpy2(&text).unwrap_err().to_string(),
        "sample \"hexagonal\": reflection \"r001\": hklpy2 reals_units = \"mm\" is not supported"
    );
}

/// Equal up to rounding: angles cross degrees and radians, U a quaternion.
fn assert_close(a: &Value, b: &Value, at: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (x - y).abs() <= 1e-12 * x.abs().max(1.0),
                "{at}: {x} vs {y}"
            );
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{at}");
            for (i, (x, y)) in x.iter().zip(y).enumerate() {
                assert_close(x, y, &format!("{at}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(
                x.keys().collect::<Vec<_>>(),
                y.keys().collect::<Vec<_>>(),
                "{at}"
            );
            for (k, x) in x {
                assert_close(x, &y[k], &format!("{at}.{k}"));
            }
        }
        _ => assert_eq!(a, b, "{at}"),
    }
}

fn assert_same_config(a: &Config, b: &Config) {
    let json = |c: &Config| serde_json::to_value(c).unwrap();
    assert_close(&json(a), &json(b), "config");
}

/// A device with a second sample, a reflection, limits and a psi mode.
fn configured() -> Arc<Diffractometer> {
    let dx = e4cv();
    dx.set_limits("chi", -90.0, f64::INFINITY).unwrap();
    dx.set_cut_point("phi", 0.0).unwrap();
    let u = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let hexagonal = Lattice::from_degrees(3.0, 3.0, 5.0, 90.0, 90.0, 120.0).unwrap();
    dx.add_sample(Sample::new(
        "hexagonal",
        hexagonal,
        Orientation::Manual { u },
    ))
    .unwrap();
    dx.select_sample("hexagonal").unwrap();
    let measured = Measured {
        hkl: [0.0, 0.0, 1.0],
        positions: vec![30.0, 0.0, 0.0, 60.0],
        wavelength: Wavelength::new(LAMBDA).unwrap(),
    };
    dx.add_reflection("r001", measured).unwrap();
    dx.set_mode(ModeSpec {
        psi: Some((Vector3::new(1.0, 0.0, 0.0), 0.5)),
        ..ModeSpec::new("psi_constant")
    })
    .unwrap();
    dx
}

#[test]
fn config_round_trips_through_json_yaml_and_files() {
    let config = configured().config();
    assert_eq!(config.sample, "hexagonal");
    assert_eq!(config.geometry.axes[1].limits, [Some(-90.0), None]);
    assert_eq!(Config::from_json(&config.to_json()).unwrap(), config);
    assert_eq!(Config::from_yaml(&config.to_yaml()).unwrap(), config);
    let dir = tempfile::tempdir().unwrap();
    for name in ["dx.json", "dx.yml", "dx.yaml"] {
        let path = dir.path().join(name);
        config.write(&path).unwrap();
        assert_eq!(Config::read(&path).unwrap(), config, "{name}");
        let dx = e4cv();
        dx.load_config(&Config::read(&path).unwrap()).unwrap();
        assert_same_config(&dx.config(), &config);
        assert_eq!(dx.geometry().axes()[1].limits().1, f64::INFINITY);
    }
}

#[test]
fn config_with_another_device_signal_wavelength_loads_nothing() {
    let config = configured().config();
    let signal = Arc::new(SoftMotor::new("energy", Some(LAMBDA))) as Arc<dyn ReadableObj>;
    let dx = e4cv_with(WavelengthSource::Signal(signal));
    let before = dx.config();
    assert_eq!(before.wavelength, None);
    let e = dx.load_config(&config).unwrap_err();
    assert_eq!(
        e,
        DiffError::WavelengthFromSignal {
            signal: "energy".into()
        }
    );
    assert_eq!(dx.config(), before);
    let config = Config {
        wavelength: None,
        ..config
    };
    dx.load_config(&config).unwrap();
    assert_same_config(&dx.config(), &config);
}

#[test]
fn config_errors_name_what_they_reject_and_load_nothing() {
    let good = configured().config();
    let mut no_cut = good.clone();
    no_cut.geometry.axes[2].cut_point = None;
    let mut nan = good.clone();
    nan.geometry.axes[0].limits = [Some(f64::NAN), None];
    let mut position = good.clone();
    let r = &mut position.samples[1].reflections[0];
    r.positions.remove("tth");
    let mut unknown = good.clone();
    unknown.sample = "missing".into();
    let mut mode = good.clone();
    mode.mode.as_mut().unwrap().psi = None;
    let mut geometry = good.clone();
    geometry.geometry.name = "E6C".into();
    let cases = [
        (
            no_cut,
            "configuration has no geometry.axes.phi.cut_point".to_owned(),
        ),
        (
            nan,
            "axis \"omega\" limits [NaN, inf] are not an interval".to_owned(),
        ),
        (
            position,
            "sample \"hexagonal\": reflection \"r001\": positions give no value for axis \"tth\""
                .to_owned(),
        ),
        (
            unknown,
            DiffError::UnknownSample {
                name: "missing".into(),
                known: vec!["cubic".into(), "hexagonal".into()],
            }
            .to_string(),
        ),
        (
            mode,
            e4cv()
                .set_mode(ModeSpec::new("psi_constant"))
                .unwrap_err()
                .to_string(),
        ),
        (
            geometry,
            "configuration geometry \"E6C\" with axes [\"omega\", \"chi\", \"phi\", \"tth\"] \
             is not the device's \"E4CV\" with axes [\"omega\", \"chi\", \"phi\", \"tth\"]"
                .to_owned(),
        ),
    ];
    for (config, message) in cases {
        let dx = e4cv();
        let before = dx.config();
        assert_eq!(dx.load_config(&config).unwrap_err().to_string(), message);
        assert_eq!(dx.config(), before);
    }
}

#[test]
fn config_text_errors_name_what_they_reject() {
    let mut json: Value = serde_json::from_str(&configured().config().to_json()).unwrap();
    json["schema_version"] = 2.into();
    let e = Config::from_json(&json.to_string()).unwrap_err();
    assert_eq!(
        e.to_string(),
        "hklrs JSON configuration schema version 2 is not 1"
    );
    json["schema_version"] = 1.into();
    json["limits"] = 1.into();
    let e = Config::from_json(&json.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        e.starts_with("hklrs JSON configuration: unknown field `limits`"),
        "{e}"
    );
    let e = Config::read(Path::new("dx.txt")).unwrap_err();
    assert_eq!(
        e.to_string(),
        "configuration file \"dx.txt\": the extension is not .json, .yml or .yaml"
    );
    let e = Config::read(Path::new("/nonexistent/dx.json"))
        .unwrap_err()
        .to_string();
    assert!(
        e.starts_with("configuration file \"/nonexistent/dx.json\": "),
        "{e}"
    );
}

fn lua(dx: &Arc<Diffractometer>) -> ManagerLuaState {
    let mut registry = Registry::new();
    registry.register_readable("e4cv", dx.clone());
    registry.register_lua_methods("e4cv", Arc::new(LuaDiffractometer::new(dx.clone())));
    let engine = Some(Arc::new(RunEngine::new(vec![])));
    ManagerLuaState::new(Arc::new(Mutex::new(engine)), Arc::new(registry))
}

#[tokio::test(flavor = "multi_thread")]
async fn lua_loads_hklpy2_and_saves_hklrs_config() {
    let dx = e4cv();
    let lua = lua(&dx);
    let source = golden("hklpy2_e4cv.yml");
    let r = lua
        .eval(&format!("e4cv:load_config({:?})", source.display()))
        .await;
    assert_eq!(r.error, None);
    assert_eq!(dx.current_sample().name, "hexagonal");
    let dir = tempfile::tempdir().unwrap();
    let saved = dir.path().join("e4cv.json");
    let r = lua
        .eval(&format!("e4cv:save_config({:?})", saved.display()))
        .await;
    assert_eq!(r.error, None);
    assert_eq!(Config::read(&saved).unwrap(), dx.config());
    let r = lua.eval("e4cv:load_config(\"e4cv.txt\")").await;
    let e = r.error.unwrap();
    assert!(
        e.contains("configuration file \"e4cv.txt\": the extension"),
        "{e}"
    );
}
