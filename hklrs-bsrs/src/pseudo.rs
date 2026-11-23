//! h, k and l as movable axes, with ophyd's pseudo-positioner semantics:
//! a set replaces one component of the shared setpoint and moves every real.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bsrs::core::msg::{DynLocation, LocatableObj, MovableObj, NamedObj, ReadableObj};
use bsrs::core::status::{Status, StatusError};
use bsrs::core::{BsrsError, DataKey, Dtype, ReadingValue};
use bsrs::event_model::compose::now;
use serde_json::json;

use crate::Diffractometer;
use crate::diffractometer::datakey;

pub struct PseudoAxis {
    dx: Arc<Diffractometer>,
    index: usize,
    name: String,
}

impl Diffractometer {
    /// h, k and l, named as this device reads them. Moving them one at a
    /// time solves and moves the reals once per axis.
    pub fn hkl_axes(self: &Arc<Self>) -> [PseudoAxis; 3] {
        let mut names = self.hkl_keys().into_iter();
        [0, 1, 2].map(|index| PseudoAxis {
            dx: self.clone(),
            index,
            name: names.next().expect("three keys"),
        })
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
            .set_component(self.index, value)
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
        self.dx.locate_component(self.index).await
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
