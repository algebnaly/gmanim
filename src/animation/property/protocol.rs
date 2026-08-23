use std::{any::Any, fmt, sync::Arc};

use crate::{
    GMFloat, Scene,
    mobjects::{MobjectId, SceneWorldError},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PropertyTarget {
    Scene,
    Mobject(MobjectId),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PropertyKey {
    pub namespace: &'static str,
    pub name: &'static str,
    pub value_type: &'static str,
}

impl PropertyKey {
    pub const fn new(
        namespace: &'static str,
        name: &'static str,
        value_type: &'static str,
    ) -> Self {
        Self {
            namespace,
            name,
            value_type,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PropertyAddress {
    pub target: PropertyTarget,
    pub key: PropertyKey,
}

impl fmt::Display for PropertyAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target {
            PropertyTarget::Scene => {
                write!(formatter, "scene.{}::{}", self.key.namespace, self.key.name)
            }
            PropertyTarget::Mobject(target) => write!(
                formatter,
                "{target:?}.{}::{}",
                self.key.namespace, self.key.name
            ),
        }
    }
}

pub trait TrackValue: Clone + Send + Sync + 'static {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self;
}

pub trait Property: Clone + Send + Sync + 'static {
    type Value: TrackValue;

    fn address(&self) -> PropertyAddress;
    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError>;
    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError>;

    fn is_present(&self, _scene: &Scene) -> Result<bool, PropertyError> {
        Ok(true)
    }

    fn finalize(&self, _scene: &mut Scene) -> Result<(), PropertyError> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct PropertyValue {
    value_type: &'static str,
    inner: Arc<dyn Any + Send + Sync>,
}

impl PropertyValue {
    pub fn new<T: Clone + Send + Sync + 'static>(value: T) -> Self {
        Self {
            value_type: std::any::type_name::<T>(),
            inner: Arc::new(value),
        }
    }

    pub(crate) fn with_type<T: Clone + Send + Sync + 'static>(
        value_type: &'static str,
        value: T,
    ) -> Self {
        Self {
            value_type,
            inner: Arc::new(value),
        }
    }

    pub fn value_type(&self) -> &'static str {
        self.value_type
    }

    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }
}

impl fmt::Debug for PropertyValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PropertyValue")
            .field("value_type", &self.value_type)
            .finish_non_exhaustive()
    }
}

trait ErasedPropertyBehavior: Send + Sync {
    fn address(&self) -> PropertyAddress;
    fn read(&self, scene: &Scene) -> Result<PropertyValue, PropertyError>;
    fn write(&self, scene: &mut Scene, value: &PropertyValue) -> Result<(), PropertyError>;
}

struct TypedPropertyBehavior<P: Property>(P);

impl<P: Property> ErasedPropertyBehavior for TypedPropertyBehavior<P> {
    fn address(&self) -> PropertyAddress {
        self.0.address()
    }

    fn read(&self, scene: &Scene) -> Result<PropertyValue, PropertyError> {
        Ok(PropertyValue::with_type(
            self.0.address().key.value_type,
            self.0.read(scene)?,
        ))
    }

    fn write(&self, scene: &mut Scene, value: &PropertyValue) -> Result<(), PropertyError> {
        let Some(value) = value.downcast_ref::<P::Value>() else {
            return Err(PropertyError::TypeMismatch {
                address: self.0.address(),
                expected: self.0.address().key.value_type,
                actual: value.value_type(),
            });
        };
        self.0.write(scene, value.clone())
    }
}

#[derive(Clone)]
pub struct ErasedProperty {
    behavior: Arc<dyn ErasedPropertyBehavior>,
}

impl ErasedProperty {
    pub fn new<P: Property>(property: P) -> Self {
        Self {
            behavior: Arc::new(TypedPropertyBehavior(property)),
        }
    }

    pub fn address(&self) -> PropertyAddress {
        self.behavior.address()
    }

    pub fn read(&self, scene: &Scene) -> Result<PropertyValue, PropertyError> {
        self.behavior.read(scene)
    }

    pub fn write(&self, scene: &mut Scene, value: &PropertyValue) -> Result<(), PropertyError> {
        self.behavior.write(scene, value)
    }
}

impl fmt::Debug for ErasedProperty {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ErasedProperty")
            .field(&self.address())
            .finish()
    }
}

#[derive(Debug)]
pub enum PropertyError {
    TypeMismatch {
        address: PropertyAddress,
        expected: &'static str,
        actual: &'static str,
    },
    Scene(SceneWorldError),
}

impl fmt::Display for PropertyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch {
                address,
                expected,
                actual,
            } => write!(
                formatter,
                "cannot write {actual} to {address}; expected {expected}"
            ),
            Self::Scene(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PropertyError {}

impl From<SceneWorldError> for PropertyError {
    fn from(error: SceneWorldError) -> Self {
        Self::Scene(error)
    }
}
