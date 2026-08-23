use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::Scene;

use super::{
    super::{authoring::AnimationClip, property::PropertyAddress},
    error::RecordingError,
    frame::PropertyWriteFrame,
    recorded_track::RecordedTrack,
};

static NEXT_RECORDING_ID: AtomicU64 = AtomicU64::new(1);

pub struct PropertyWriteRecorder {
    id: u64,
    duration: u32,
    next_frame: u32,
    baseline_scene: Scene,
    working_scene: Scene,
    tracks: Vec<Box<dyn RecordedTrack>>,
    track_indices: HashMap<PropertyAddress, usize>,
}

impl PropertyWriteRecorder {
    pub fn new(scene: &Scene, duration: u32) -> Result<Self, RecordingError> {
        if duration == 0 {
            return Err(RecordingError::EmptyRecording);
        }
        Ok(Self {
            id: NEXT_RECORDING_ID.fetch_add(1, Ordering::Relaxed),
            duration,
            next_frame: 1,
            baseline_scene: scene.clone(),
            working_scene: scene.clone(),
            tracks: Vec::new(),
            track_indices: HashMap::new(),
        })
    }

    pub fn duration(&self) -> u32 {
        self.duration
    }

    pub fn next_frame(&self) -> Option<u32> {
        (self.next_frame <= self.duration).then_some(self.next_frame)
    }

    pub fn begin_frame(&self) -> Result<PropertyWriteFrame, RecordingError> {
        if self.next_frame > self.duration {
            return Err(RecordingError::RecordingComplete);
        }
        Ok(PropertyWriteFrame::new(
            self.id,
            self.next_frame,
            self.duration,
            self.working_scene.clone(),
        ))
    }

    pub fn commit_frame(&mut self, frame: PropertyWriteFrame) -> Result<(), RecordingError> {
        let frame = frame.into_recorded();
        if frame.recording_id != self.id {
            return Err(RecordingError::ForeignFrame);
        }
        if frame.frame != self.next_frame {
            return Err(RecordingError::UnexpectedFrame {
                expected: self.next_frame,
                actual: frame.frame,
            });
        }

        for track in &mut self.tracks {
            track.push_from(&frame.scene)?;
        }
        for property in frame.touched {
            let address = property.address();
            if self.track_indices.contains_key(&address) {
                continue;
            }
            let track =
                property.start_track(&self.baseline_scene, self.next_frame, &frame.scene)?;
            self.track_indices.insert(address, self.tracks.len());
            self.tracks.push(track);
        }

        self.working_scene = frame.scene;
        self.next_frame += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<AnimationClip, RecordingError> {
        let expected = self.duration + 1;
        if self.next_frame != expected {
            return Err(RecordingError::IncompleteRecording {
                expected,
                actual: self.next_frame,
            });
        }
        let mut clip = AnimationClip::new(self.duration);
        for track in self.tracks {
            clip = track.append_to(clip);
        }
        Ok(clip)
    }
}
