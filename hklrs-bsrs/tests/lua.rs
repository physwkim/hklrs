//! The Diffractometer's Lua methods, called through bsrs's Lua state as the
//! queue server calls them. A cubic cell with a = λ puts (0, 0, 1) at
//! omega 30, chi 0, phi 0, tth 60 and (1, 0, 0) at phi 90 for U = I.

use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

use bsrs::backends::soft::SoftMotor;
use bsrs::core::msg::{MovableObj, NamedObj, ReadableObj};
use bsrs::engine::RunEngine;
use bsrs::host::manager_lua::ManagerLuaState;
use bsrs::qs::{LuaEvaluator, Registry};
use hklrs::nalgebra::{Matrix3, UnitQuaternion, Vector3};
use hklrs::presets::preset;
use hklrs::{Lattice, ModeSpec, Orientation, Sample, Wavelength};
use hklrs_bsrs::{Diffractometer, LuaDiffractometer, RealAxis, WavelengthSource};
use serde_json::Value;
use tokio::sync::Mutex;

const AXES: [&str; 4] = ["omega", "chi", "phi", "tth"];
const LAMBDA: f64 = 1.54;

fn e4cv(at: [f64; 4], u: UnitQuaternion<f64>) -> Arc<Diffractometer> {
    let reals = AXES
        .iter()
        .zip(at)
        .map(|(&a, v)| (a, Arc::new(SoftMotor::new(a, Some(v))) as Arc<dyn RealAxis>))
        .collect();
    let sample = Sample::new(
        "cubic",
        Lattice::cubic(LAMBDA).unwrap(),
        Orientation::Manual { u },
    );
    let wavelength = WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap());
    let dx = Diffractometer::new("e4cv", preset("E4CV").unwrap(), reals, wavelength, sample);
    Arc::new(dx.unwrap())
}

/// A Lua state with `dx` registered under its name.
fn lua(dx: &Arc<Diffractometer>) -> ManagerLuaState {
    let mut registry = Registry::new();
    registry.register_readable(dx.name(), dx.clone());
    let methods = Arc::new(LuaDiffractometer::new(dx.clone()));
    registry.register_lua_methods(dx.name(), methods);
    let engine = Some(Arc::new(RunEngine::new(vec![])));
    ManagerLuaState::new(Arc::new(Mutex::new(engine)), Arc::new(registry))
}

/// The numbers of `expr` in order, nested tables flattened; `axes(p)` lists
/// a positions table in geometry order.
async fn numbers(lua: &ManagerLuaState, expr: &str) -> Vec<f64> {
    let script = format!(
        r#"
local function axes(p) return {{p.omega, p.chi, p.phi, p.tth}} end
local out = {{}}
local function walk(v)
  if type(v) == "table" then
    for _, x in ipairs(v) do walk(x) end
  else
    out[#out + 1] = string.format("%.17g", v)
  end
end
walk({expr})
return table.concat(out, " ")"#
    );
    let r = lua.eval(&script).await;
    assert_eq!(r.error, None, "{expr}");
    let text = r.return_value.unwrap();
    text.split(' ').map(|s| s.parse().unwrap()).collect()
}

/// The error `statement` raises.
async fn error(lua: &ManagerLuaState, statement: &str) -> String {
    let r = lua.eval(statement).await;
    r.error
        .unwrap_or_else(|| panic!("{statement} raised no error"))
}

async fn ok(lua: &ManagerLuaState, statement: &str) {
    let r = lua.eval(statement).await;
    assert_eq!(r.error, None, "{statement}");
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn compute_ub_orients_the_sample_from_reflections_added_in_lua() {
    let dx = e4cv([0.0; 4], UnitQuaternion::from_euler_angles(0.3, -0.7, 1.1));
    let lua = lua(&dx);
    ok(
        &lua,
        r#"e4cv:add_reflection("r1", {0, 0, 1}, {omega = 30, chi = 0, phi = 0, tth = 60})
           e4cv:add_reflection("r2", {0, 1, 0}, {omega = 30, chi = 90, phi = 0, tth = 60})"#,
    )
    .await;
    let ub = numbers(&lua, r#"e4cv:compute_ub("r1", "r2")"#).await;
    let s = TAU / LAMBDA;
    assert_close(&ub, &[s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s]);
    let rust = dx.current_sample().ub();
    assert_close(&ub, rust.transpose().as_slice());

    let e = error(&lua, r#"e4cv:compute_ub("r1", "r9")"#).await;
    assert!(e.contains("\"r9\""), "{e}");
}

/// PY2-5: hklpy2 `add_reflection` is annotated to take an ndarray and
/// rejects it. Here positions take one shape, a table by axis name, and
/// every other shape is rejected by name before anything is recorded.
#[tokio::test(flavor = "multi_thread")]
async fn py2_5_positions_take_one_shape() {
    let dx = e4cv([0.0; 4], UnitQuaternion::identity());
    let lua = lua(&dx);
    let add = |positions: &str| format!(r#"e4cv:add_reflection("r1", {{0, 0, 1}}, {positions})"#);
    for (positions, says) in [
        ("{30, 0, 0, 60}", "arg #2"),
        ("{omega = 30, chi = 0, phi = 0}", "\"tth\""),
        ("{omega = 30, chi = 0, phi = 0, tth = 60, mu = 0}", "\"mu\""),
    ] {
        let e = error(&lua, &add(positions)).await;
        assert!(e.contains(says), "{positions}: {e}");
    }
    let e = error(&lua, r#"e4cv:add_reflection("r1", {0, 0}, {})"#).await;
    assert!(e.contains("arg #1"), "{e}");
    // none of the rejected calls recorded r1
    ok(&lua, &add("{omega = 30, chi = 0, phi = 0, tth = 60}")).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn forward_inverse_ca_and_wh_agree_with_the_device() {
    let start = [10.0, 5.0, -20.0, 40.0];
    let dx = e4cv(start, UnitQuaternion::identity());
    let lua = lua(&dx);
    let w = Wavelength::new(LAMBDA).unwrap();

    let hkl = numbers(
        &lua,
        "e4cv:forward({omega = 30, chi = 0, phi = 90, tth = 60})",
    )
    .await;
    assert_close(&hkl, &[1.0, 0.0, 0.0]);

    let seed = [30.0, 1.0, 89.0, 60.0];
    let inverse = numbers(
        &lua,
        "axes(e4cv:inverse({1, 0, 0}, {omega = 30, chi = 1, phi = 89, tth = 60}))",
    )
    .await;
    assert_close(&inverse, &dx.inverse([1.0, 0.0, 0.0], &seed, w).unwrap());

    let ca = numbers(&lua, "axes(e4cv:ca({0, 0, 1}))").await;
    assert_close(&ca, &dx.inverse([0.0, 0.0, 1.0], &start, w).unwrap());
    assert_close(&ca[3..], &[60.0]);
    // ca computes, it does not move
    assert_close(&dx.positions().await.unwrap(), &start);

    let wh = numbers(
        &lua,
        "(function() local w = e4cv:wh() return {w.hkl, axes(w.positions), w.wavelength} end)()",
    )
    .await;
    let at_start: [f64; 3] = dx.hkl(&start, w).unwrap().into();
    assert_close(&wh[..3], &at_start);
    assert_close(&wh[3..7], &start);
    assert_close(&wh[7..], &[LAMBDA]);
}

/// The reading's ψ.
async fn psi_reading(dx: &Diffractometer) -> Value {
    dx.read_dyn().await.unwrap()["e4cv_psi"].value.clone()
}

/// wh's ψ is the reading's: a number where the reading has one, nil where
/// it has NaN.
#[tokio::test(flavor = "multi_thread")]
async fn wh_carries_psi_as_the_reading_does() {
    let dx = e4cv([30.0, 0.0, 0.0, 60.0], UnitQuaternion::identity());
    let lua = lua(&dx);
    // bissector has no ψ
    assert_eq!(psi_reading(&dx).await, Value::Null);
    ok(&lua, "assert(e4cv:wh().psi == nil)").await;
    // Q at (0, 0, 1) lies along the reference: ψ is undefined
    ok(
        &lua,
        r#"e4cv:set_mode("psi_constant", {psi = {reference = {0, 0, 1}, angle = 30}})"#,
    )
    .await;
    assert_eq!(psi_reading(&dx).await, Value::Null);
    ok(&lua, "assert(e4cv:wh().psi == nil)").await;

    let [h, ..] = dx.hkl_axes();
    h.set_dyn(1.0).await.await.unwrap();
    dx.psi_axis().set_dyn(200.0).await.await.unwrap();
    let psi = numbers(&lua, "e4cv:wh().psi").await;
    assert_eq!(psi, [psi_reading(&dx).await.as_f64().unwrap()]);
    // on the setpoint's branch, as the reading has it
    assert_close(&psi, &[200.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn wh_has_no_psi_on_a_geometry_without_a_psi_mode() {
    let geometry = preset("ZAXIS").unwrap();
    let reals = geometry.axes().iter().map(|a| {
        let real = Arc::new(SoftMotor::new(a.name(), Some(0.0))) as Arc<dyn RealAxis>;
        (a.name(), real)
    });
    let reals = reals.collect();
    let sample = Sample::new(
        "cubic",
        Lattice::cubic(LAMBDA).unwrap(),
        Orientation::Manual {
            u: UnitQuaternion::identity(),
        },
    );
    let wavelength = WavelengthSource::Fixed(Wavelength::new(LAMBDA).unwrap());
    let dx = Diffractometer::new("zaxis", geometry.clone(), reals, wavelength, sample);
    let dx = Arc::new(dx.unwrap());
    let lua = lua(&dx);
    ok(
        &lua,
        "local w = zaxis:wh() assert(w.psi == nil and #w.hkl == 3)",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn set_mode_takes_its_parameters_as_one_table() {
    let dx = e4cv([0.0; 4], UnitQuaternion::identity());
    let lua = lua(&dx);
    ok(
        &lua,
        r#"e4cv:set_mode("psi_constant", {psi = {reference = {1, 0, 0}, angle = 90}})"#,
    )
    .await;
    assert_eq!(
        dx.mode(),
        Some(ModeSpec {
            psi: Some((Vector3::x(), FRAC_PI_2)),
            ..ModeSpec::new("psi_constant")
        })
    );
    ok(
        &lua,
        r#"e4cv:set_mode("double_diffraction", {hkl2 = {0, 1, 0}})"#,
    )
    .await;
    assert_eq!(
        dx.mode(),
        Some(ModeSpec {
            hkl2: Some(Vector3::y()),
            ..ModeSpec::new("double_diffraction")
        })
    );
    ok(&lua, r#"e4cv:set_mode("bissector", {})"#).await;
    assert_eq!(dx.mode(), Some(ModeSpec::new("bissector")));

    for (statement, says) in [
        (r#"e4cv:set_mode("nope", {})"#, "\"nope\""),
        (r#"e4cv:set_mode("psi_constant", {})"#, "reference and psi"),
        (
            r#"e4cv:set_mode("psi_constant", {psi = {reference = {1, 0, 0}}})"#,
            "angle",
        ),
        (r#"e4cv:set_mode("bissector", {hkl3 = {0, 1, 0}})"#, "hkl3"),
    ] {
        let e = error(&lua, statement).await;
        assert!(e.contains(says), "{statement}: {e}");
    }
    // a rejected mode leaves the last one set
    assert_eq!(dx.mode(), Some(ModeSpec::new("bissector")));
}

#[tokio::test(flavor = "multi_thread")]
async fn set_lattice_and_set_ub_set_the_current_sample() {
    let dx = e4cv([0.0; 4], UnitQuaternion::identity());
    let lua = lua(&dx);
    ok(&lua, "e4cv:set_lattice(2, 3, 4, 90, 100, 120)").await;
    let expected = Lattice::from_degrees(2.0, 3.0, 4.0, 90.0, 100.0, 120.0).unwrap();
    assert_eq!(dx.current_sample().lattice, expected);

    let e = error(&lua, "e4cv:set_lattice(2, 3, 4, 200, 100, 120)").await;
    assert!(e.contains("alpha"), "{e}");
    assert_eq!(dx.current_sample().lattice, expected);

    let u = UnitQuaternion::from_euler_angles(0.0, 0.0, FRAC_PI_2);
    let ub: Matrix3<f64> = u.to_rotation_matrix().into_inner() * expected.b_matrix();
    let rows: Vec<String> = (0..3)
        .map(|i| format!("{{{}, {}, {}}}", ub[(i, 0)], ub[(i, 1)], ub[(i, 2)]))
        .collect();
    ok(&lua, &format!("e4cv:set_ub({{{}}})", rows.join(", "))).await;
    let set = dx.current_sample().ub();
    assert!((set - ub).abs().max() < 1e-12, "{set} != {ub}");
}
