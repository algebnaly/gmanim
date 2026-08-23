use crate::{
    Scene,
    mobjects::{MobjectId, SpawnPlan},
};

use super::{
    error::TimelineError,
    property::{ErasedProperty, Property, PropertyValue},
};

#[derive(Clone, Debug)]
pub enum SceneOperation {
    Spawn {
        plan: SpawnPlan,
    },
    SetParent {
        child: MobjectId,
        parent: Option<MobjectId>,
    },
    WriteProperty {
        property: ErasedProperty,
        value: PropertyValue,
    },
    Remove {
        target: MobjectId,
    },
}

impl SceneOperation {
    pub fn write<P: Property>(property: P, value: P::Value) -> Self {
        let address = property.address();
        Self::WriteProperty {
            property: ErasedProperty::new(property),
            value: PropertyValue::with_type(address.key.value_type, value),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ClipEvent {
    pub(super) frame: u32,
    pub(super) operation: SceneOperation,
}

impl ClipEvent {
    pub(super) fn new(frame: u32, operation: SceneOperation) -> Self {
        Self { frame, operation }
    }
}

#[derive(Clone, Debug)]
pub(super) struct TimedEvent {
    pub(super) frame: u32,
    pub(super) order: u64,
    pub(super) operation: SceneOperation,
}

pub(super) fn apply_operation(
    scene: &mut Scene,
    operation: &SceneOperation,
) -> Result<(), TimelineError> {
    match operation {
        SceneOperation::Spawn { plan } => {
            scene.world.materialize(plan)?;
        }
        SceneOperation::SetParent { child, parent } => {
            scene.world.set_parent(*child, *parent)?;
        }
        SceneOperation::WriteProperty { property, value } => {
            property.write(scene, value)?;
        }
        SceneOperation::Remove { target } => {
            scene.world.unspawn(*target)?;
        }
    }
    Ok(())
}
