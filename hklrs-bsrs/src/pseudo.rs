//! h, k, l and ψ as movable axes, with ophyd's pseudo-positioner
//! semantics: a set replaces one component of the shared setpoint and moves
//! every real. Every move goes through [`Diffractometer::move_to`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bsrs::core::msg::{DynLocation, LocatableObj, MovableObj, NamedObj, ReadableObj};
use bsrs::core::status::{Status, StatusError};
use bsrs::core::{BsrsError, DataKey, Dtype, ReadingValue};
use bsrs::event_model::compose::now;
use serde_json::json;

use crate::Diffractometer;
use crate::diffractometer::{Component, Setpoint, datakey};

pub struct PseudoAxis {
    dx: Arc<Diffractometer>,
    component: Component,
    name: String,
}

impl Diffractometer {
    /// h, k and l, named as this device reads them. Moving them one at a
    /// time solves and moves the reals once per axis.
    pub fn hkl_axes(self: &Arc<Self>) -> [PseudoAxis; 3] {
        let mut names = self.hkl_keys().into_iter();
        [0, 1, 2].map(|index| PseudoAxis {
            dx: self.clone(),
            component: Component::Hkl(index),
            name: names.next().expect("three keys"),
        })
    }

    /// ψ of the device's psi mode, degrees, as `{name}_psi`. A set holds
    /// the hkl setpoint and replaces the mode's ψ, so later h, k, l moves
    /// keep it; the readback is ψ about the mode's reference. The setpoint
    /// is kept as given and the readback is taken on its branch, within
    /// 180° of it: a set to 200° reads back 200°, not −160°. Outside a psi
    /// mode the axis refuses to move or be located.
    pub fn psi_axis(self: &Arc<Self>) -> PseudoAxis {
        PseudoAxis {
            dx: self.clone(),
            component: Component::Psi,
            name: self.key("psi"),
        }
    }

    /// A one-shot move of the whole setpoint to `hkl`.
    pub(crate) fn hkl_move(self: &Arc<Self>, hkl: [f64; 3]) -> HklMove {
        HklMove {
            dx: self.clone(),
            hkl,
            name: self.key("hkl"),
        }
    }
}

/// `mv_hkl`'s move, set through the engine so a pause or the run's end
/// stops the reals as it does for h, k and l. The engine's set carries one
/// number; the destination is `hkl`, so the value is unused.
pub(crate) struct HklMove {
    dx: Arc<Diffractometer>,
    hkl: [f64; 3],
    name: String,
}

impl NamedObj for HklMove {
    fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl MovableObj for HklMove {
    async fn set_dyn(&self, _: f64) -> Status {
        self.dx
            .move_to(Setpoint::Hkl(self.hkl))
            .await
            .unwrap_or_else(|e| Status::fail(StatusError::Failed(format!("mv_hkl: {e}"))))
    }

    async fn stop_on_pause(&self, success: bool) -> Result<(), BsrsError> {
        self.dx.stop_reals(success).await
    }
}

impl NamedObj for PseudoAxis {
    fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl MovableObj for PseudoAxis {
    async fn set_dyn(&self, value: f64) -> Status {
        self.dx
            .move_to(Setpoint::Component {
                component: self.component,
                value,
            })
            .await
            .unwrap_or_else(|e| Status::fail(StatusError::Failed(format!("{}: {e}", self.name))))
    }

    async fn stop_on_pause(&self, success: bool) -> Result<(), BsrsError> {
        self.dx.stop_reals(success).await
    }
}

#[async_trait]
impl LocatableObj for PseudoAxis {
    async fn locate_dyn(&self) -> Result<DynLocation, BsrsError> {
        self.dx.locate(self.component).await
    }
}

#[async_trait]
impl ReadableObj for PseudoAxis {
    async fn read_dyn(&self) -> Result<HashMap<String, ReadingValue>, BsrsError> {
        let readback = self.locate_dyn().await?.readback;
        let value = ReadingValue::new(json!(readback), now());
        Ok(HashMap::from([(self.name.clone(), value)]))
    }

    async fn describe_dyn(&self) -> Result<HashMap<String, DataKey>, BsrsError> {
        let key = datakey(&self.name, Dtype::Number, &[], None);
        Ok(HashMap::from([(self.name.clone(), key)]))
    }
}
