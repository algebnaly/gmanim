use crate::{Context, Scene, SceneSnapshot};

use super::{
    error::TimelineError,
    operation::{TimedEvent, apply_operation},
    track::CompiledTrack,
};

pub struct CompiledTimeline {
    pub scene: Scene,
    pub ctx: Context,
    initial_scene: SceneSnapshot,
    tracks: Vec<CompiledTrack>,
    events: Vec<TimedEvent>,
    total_frames: u32,
    current_frame: u32,
}

impl CompiledTimeline {
    pub(super) fn new(
        initial_scene: SceneSnapshot,
        ctx: Context,
        tracks: Vec<CompiledTrack>,
        events: Vec<TimedEvent>,
        total_frames: u32,
    ) -> Self {
        Self {
            scene: Scene::from_snapshot(&initial_scene),
            initial_scene,
            ctx,
            tracks,
            events,
            total_frames,
            current_frame: 0,
        }
    }

    pub fn total_frames(&self) -> u32 {
        self.total_frames
    }

    pub fn current_frame(&self) -> u32 {
        self.current_frame
    }

    pub fn advance_frame(&mut self) -> Result<bool, TimelineError> {
        if self.current_frame >= self.total_frames {
            return Ok(false);
        }
        self.seek(self.current_frame + 1)?;
        Ok(true)
    }

    pub fn seek(&mut self, frame: u32) -> Result<(), TimelineError> {
        let target = frame.min(self.total_frames);
        if target == self.current_frame {
            return Ok(());
        }

        if target == self.current_frame + 1 {
            self.apply_events_at(target)?;
            self.apply_tracks_at(target)?;
        } else {
            self.scene.restore(&self.initial_scene);
            for event in self.events.iter().filter(|event| event.frame <= target) {
                apply_operation(&mut self.scene, &event.operation)?;
            }
            self.apply_tracks_at(target)?;
        }
        self.current_frame = target;
        Ok(())
    }

    fn apply_events_at(&mut self, frame: u32) -> Result<(), TimelineError> {
        for event in self.events.iter().filter(|event| event.frame == frame) {
            apply_operation(&mut self.scene, &event.operation)?;
        }
        Ok(())
    }

    fn apply_tracks_at(&mut self, frame: u32) -> Result<(), TimelineError> {
        for track in &self.tracks {
            track.apply(frame, &mut self.scene)?;
        }
        Ok(())
    }
}
