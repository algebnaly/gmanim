use crate::mobjects::SceneWorldError;

use super::{
    property::{PropertyAddress, PropertyError},
    time::FrameRange,
};

#[derive(Debug)]
pub enum TimelineError {
    EmptyClip,
    EventOutsideClip {
        frame: u32,
        duration: u32,
    },
    SampleCount {
        expected: usize,
        actual: usize,
    },
    ConflictingWrites {
        address: PropertyAddress,
        first: FrameRange,
        second: FrameRange,
    },
    Property(PropertyError),
    Scene(SceneWorldError),
}

impl std::fmt::Display for TimelineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyClip => {
                formatter.write_str("animation clips must contain at least one frame")
            }
            Self::EventOutsideClip { frame, duration } => {
                write!(
                    formatter,
                    "event frame {frame} exceeds clip duration {duration}"
                )
            }
            Self::SampleCount { expected, actual } => {
                write!(
                    formatter,
                    "sampled curve contains {actual} values; expected {expected}"
                )
            }
            Self::ConflictingWrites {
                address,
                first,
                second,
            } => write!(
                formatter,
                "overlapping Replace tracks write {address}: {first:?} and {second:?}"
            ),
            Self::Property(error) => error.fmt(formatter),
            Self::Scene(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TimelineError {}

impl From<SceneWorldError> for TimelineError {
    fn from(error: SceneWorldError) -> Self {
        Self::Scene(error)
    }
}

impl From<PropertyError> for TimelineError {
    fn from(error: PropertyError) -> Self {
        Self::Property(error)
    }
}
