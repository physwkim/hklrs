//! hkl plans. Every point is solved before the run opens, so an unreachable
//! point fails the plan with nothing moved and no run left open (PY2-15).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bsrs::core::msg::{MovableObj, NamedObj, ReadableObj};
use bsrs::core::{Msg, Plan, plan_box};
use bsrs::plans::list_scan_nd;
use bsrs::plans::preprocessors::msg_mutator;
use bsrs::plans::stubs::mv;
use futures::{StreamExt, stream};
use hklrs::nalgebra::{Unit, UnitQuaternion, Vector3};
use hklrs::{Lattice, ModeSpec, PARALLEL_SIN};
use serde_json::{Map, Value, json};

use crate::diffractometer::{Snapshot, configuration, psi_mode};
use crate::error::Fault;
use crate::{DiffError, Diffractometer};

/// `num` points from `start` to `stop`, both included.
pub fn hkl_scan(
    dx: Arc<Diffractometer>,
    start: [f64; 3],
    stop: [f64; 3],
    num: usize,
    detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>,
) -> Plan {
    let points = (0..num)
        .map(|i| lerp3(start, stop, fraction(i, num)))
        .collect();
    let args = json!({"start": start, "stop": stop, "num": num});
    scan(dx, detectors, "hkl_scan", args, Trajectory::Hkl(points))
}

pub fn hkl_list_scan(
    dx: Arc<Diffractometer>,
    points: Vec<[f64; 3]>,
    detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>,
) -> Plan {
    let args = json!({ "points": points });
    scan(
        dx,
        detectors,
        "hkl_list_scan",
        args,
        Trajectory::Hkl(points),
    )
}

/// `num` points on the zone of `start` and `stop`: the direction turns
/// from one to the other in the plane they span and the length changes
/// linearly, both measured in the reciprocal metric of the sample's
/// lattice. hklpy2's `scan_zone` measures with the direct-lattice matrix
/// instead (PY2-11).
pub fn zone_scan(
    dx: Arc<Diffractometer>,
    start: [f64; 3],
    stop: [f64; 3],
    num: usize,
    detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>,
) -> Plan {
    let args = json!({"start": start, "stop": stop, "num": num});
    let trajectory = Trajectory::Zone { start, stop, num };
    scan(dx, detectors, "zone_scan", args, trajectory)
}

/// Holds `hkl` while ψ, the azimuth about it of `reference`, steps through
/// `num` values over `psi` (degrees), in psi mode `mode`, or without one in
/// the geometry's psi mode if it has exactly one. The device's own mode is
/// left unchanged.
pub fn psi_scan(
    dx: Arc<Diffractometer>,
    mode: Option<&str>,
    hkl: [f64; 3],
    reference: [f64; 3],
    psi: [f64; 2],
    num: usize,
    detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>,
) -> Plan {
    let args = json!({
        "mode": mode, "hkl": hkl, "reference": reference,
        "psi_start": psi[0], "psi_stop": psi[1], "num": num,
    });
    let trajectory = Trajectory::Psi {
        mode: mode.map(str::to_owned),
        hkl,
        reference,
        psi: (0..num)
            .map(|i| lerp(psi[0], psi[1], fraction(i, num)))
            .collect(),
    };
    scan(dx, detectors, "psi_scan", args, trajectory)
}

/// Moves the reals to `hkl` in the device's mode, closest to their
/// readbacks, and makes `hkl` the pseudo axes' setpoint. Opens no run.
pub fn mv_hkl(dx: Arc<Diffractometer>, hkl: [f64; 3]) -> Plan {
    mv(Arc::new(dx.hkl_move(hkl)), 0.0)
}

enum Trajectory {
    /// Solved in the device's mode.
    Hkl(Vec<[f64; 3]>),
    Zone {
        start: [f64; 3],
        stop: [f64; 3],
        num: usize,
    },
    /// ψ in degrees.
    Psi {
        /// None: the geometry's one psi mode.
        mode: Option<String>,
        hkl: [f64; 3],
        reference: [f64; 3],
        psi: Vec<f64>,
    },
}

/// Every point with the mode it is solved in, and the mode name the run
/// records.
type Path = (Vec<([f64; 3], ModeSpec)>, String);

impl Trajectory {
    fn path(self, snapshot: &Snapshot) -> Result<Path, DiffError> {
        let in_mode = |points: Vec<[f64; 3]>| -> Result<Path, DiffError> {
            let mode = snapshot.mode()?;
            let name = mode.name.clone();
            Ok((
                points.into_iter().map(|p| (p, mode.clone())).collect(),
                name,
            ))
        };
        match self {
            Self::Hkl(points) => in_mode(points),
            Self::Zone { start, stop, num } => {
                in_mode(zone(&snapshot.sample.lattice, start, stop, num)?)
            }
            Self::Psi {
                mode,
                hkl,
                reference,
                psi,
            } => {
                let mode = match mode {
                    Some(mode) => mode,
                    None => psi_mode(&snapshot.geometry)?,
                };
                let at = |psi: f64| ModeSpec {
                    psi: Some((Vector3::from(reference), psi.to_radians())),
                    ..ModeSpec::new(mode.as_str())
                };
                Ok((psi.into_iter().map(|p| (hkl, at(p))).collect(), mode))
            }
        }
    }
}

fn zone(
    lattice: &Lattice,
    start: [f64; 3],
    stop: [f64; 3],
    num: usize,
) -> Result<Vec<[f64; 3]>, DiffError> {
    if num < 2 {
        return Ok(vec![start; num]);
    }
    let b = lattice.b_matrix();
    let (q1, q2) = (b * Vector3::from(start), b * Vector3::from(stop));
    let (n1, n2) = (q1.norm(), q2.norm());
    let normal = q1.cross(&q2);
    // NaN for a zero vector
    let sin = normal.norm() / (n1 * n2);
    if sin.is_nan() || sin < PARALLEL_SIN {
        return Err(DiffError::Zone { start, stop });
    }
    let normal = Unit::new_normalize(normal);
    let angle = q1.angle(&q2);
    let b_inverse = b
        .try_inverse()
        .expect("B has a positive diagonal for every valid Lattice");
    Ok((0..num)
        .map(|i| {
            let t = fraction(i, num);
            let turn = UnitQuaternion::from_axis_angle(&normal, angle * t);
            let q = turn * q1 * (lerp(n1, n2, t) / n1);
            (b_inverse * q).into()
        })
        .collect())
}

/// Position `i` of `num` on [0, 1].
fn fraction(i: usize, num: usize) -> f64 {
    if num < 2 {
        0.0
    } else {
        i as f64 / (num - 1) as f64
    }
}

/// Exact at both ends.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [0, 1, 2].map(|i| lerp(a[i], b[i], t))
}

/// The plan `prepare` gives, or a `Fail` naming plan `name` and the error,
/// before any other message.
fn once<F>(name: &'static str, prepare: F) -> Plan
where
    F: Future<Output = Result<Plan, Fault>> + Send + 'static,
{
    stream::once(async move {
        prepare.await.unwrap_or_else(|e| {
            let fail = Msg::Fail(format!("{name}: {e}"));
            plan_box(stream::iter([fail]))
        })
    })
    .flatten()
    .boxed()
}

fn scan(
    dx: Arc<Diffractometer>,
    detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>,
    name: &'static str,
    mut args: Value,
    trajectory: Trajectory,
) -> Plan {
    let detectors = readers(&dx, detectors);
    args["detectors"] = json!(detectors.iter().map(|d| d.name()).collect::<Vec<_>>());
    once(name, async move {
        let reals = dx.reals().iter().map(|r| r.clone() as Arc<dyn ReadableObj>);
        check_keys(reals.chain(detectors.iter().cloned()).collect()).await?;
        let snapshot = dx.snapshot().await?;
        let (path, mode) = trajectory.path(&snapshot)?;
        let solved = snapshot.solve_path(&path)?;
        let config = configuration(
            &snapshot.geometry,
            &snapshot.sample,
            Some(&mode),
            snapshot.wavelength,
        );
        let mut config: Map<String, Value> = config
            .into_iter()
            .map(|(field, v)| (field.to_owned(), v))
            .collect();
        config.insert("solver".into(), json!("hklrs"));
        config.insert("version".into(), json!(env!("CARGO_PKG_VERSION")));
        let md = HashMap::from([
            ("diffractometers".to_owned(), json!({ dx.name(): config })),
            (
                "hkl".to_owned(),
                json!(path.iter().map(|(hkl, _)| hkl).collect::<Vec<_>>()),
            ),
            ("plan_args".to_owned(), args),
        ]);
        // the reals move outside the pseudo axes from here
        dx.clear_target().await;
        let axes = dx.reals().iter().enumerate().map(|(i, r)| {
            let column = solved.iter().map(|values| values[i]).collect();
            (
                r.clone() as Arc<dyn MovableObj>,
                r.clone() as Arc<dyn ReadableObj>,
                column,
            )
        });
        let plan = list_scan_nd(detectors, axes.collect(), None);
        Ok(msg_mutator(plan, move |msg| match msg {
            Msg::OpenRun(mut meta) => {
                meta.plan_name = Some(name.to_owned());
                meta.extra.extend(md.clone());
                Msg::OpenRun(meta)
            }
            other => other,
        }))
    })
}

/// `detectors` and `dx`, without a real, `dx` or an earlier detector
/// repeated: list_scan_nd reads each reader it is given.
fn readers(
    dx: &Arc<Diffractometer>,
    detectors: impl IntoIterator<Item = Arc<dyn ReadableObj>>,
) -> Vec<Arc<dyn ReadableObj>> {
    let mut seen: HashSet<*const ()> = dx.reals().iter().map(|r| Arc::as_ptr(r).cast()).collect();
    let dx = dx.clone() as Arc<dyn ReadableObj>;
    detectors
        .into_iter()
        .chain([dx])
        .filter(|d| seen.insert(Arc::as_ptr(d).cast()))
        .collect()
}

/// A data key read by two readers aborts the event bundle mid-run; this
/// finds it before the run opens.
async fn check_keys(readers: Vec<Arc<dyn ReadableObj>>) -> Result<(), Fault> {
    let mut owners: HashMap<String, String> = HashMap::new();
    for reader in readers {
        let mut keys: Vec<String> = reader.describe_dyn().await?.into_keys().collect();
        keys.sort();
        for key in keys {
            let second = reader.name().to_owned();
            if let Some(first) = owners.insert(key.clone(), second.clone()) {
                return Err(DiffError::DataKeyCollision { key, first, second }.into());
            }
        }
    }
    Ok(())
}
