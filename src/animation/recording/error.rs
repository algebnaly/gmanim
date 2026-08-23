use crate::mobjects::SceneWorldError;

use super::super::property::PropertyError;

#[derive(Debug)]
pub enum RecordingError {
    EmptyRecording,
    RecordingComplete,
    ForeignFrame,
    UnexpectedFrame { expected: u32, actual: u32 },
    IncompleteRecording { expected: u32, actual: u32 },
    Property(PropertyError),
    Scene(SceneWorldError),
}

impl std::fmt::Display for RecordingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRecording => {
                formatter.write_str("property recordings must contain at least one frame")
            }
            Self::RecordingComplete => formatter.write_str("all recording frames are complete"),
            Self::ForeignFrame => {
                formatter.write_str("property frame belongs to a different recorder")
            }
            Self::UnexpectedFrame { expected, actual } => {
                write!(
                    formatter,
                    "expected recording frame {expected}, got {actual}"
                )
            }
            Self::IncompleteRecording { expected, actual } => write!(
                formatter,
                "property recording contains {actual} samples; expected {expected}"
            ),
            Self::Property(error) => error.fmt(formatter),
            Self::Scene(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RecordingError {}

impl From<PropertyError> for RecordingError {
    fn from(error: PropertyError) -> Self {
        Self::Property(error)
    }
}

impl From<SceneWorldError> for RecordingError {
    fn from(error: SceneWorldError) -> Self {
        Self::Scene(error)
    }
}
