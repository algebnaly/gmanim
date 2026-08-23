use crate::{
    Color, Context, Scene, SceneConfig, SceneSnapshot,
    mobjects::{MobjectId, NodeBundle, SpawnPlan},
};

use super::{
    authoring::{AnimationBuilder, AnimationClip},
    error::TimelineError,
    operation::{SceneOperation, TimedEvent, apply_operation},
    property::Property,
    recording::{PropertyWriteRecorder, RecordingError, SceneView},
    runtime::CompiledTimeline,
    time::FrameRange,
    track::{CompiledTrack, Composition},
};

pub struct TimelineBuilder {
    initial_scene: SceneSnapshot,
    build_scene: Scene,
    ctx: Context,
    cursor: u32,
    tracks: Vec<CompiledTrack>,
    events: Vec<TimedEvent>,
    next_event_order: u64,
}

impl TimelineBuilder {
    pub fn new(scene: Scene, ctx: Context) -> Self {
        Self {
            initial_scene: scene.snapshot(),
            build_scene: scene,
            ctx,
            cursor: 0,
            tracks: Vec::new(),
            events: Vec::new(),
            next_event_order: 0,
        }
    }

    pub fn play(&mut self, animation: impl AnimationBuilder) -> Result<(), TimelineError> {
        let clip = animation.compile(&self.build_scene)?;
        self.append_clip(clip)
    }

    pub fn cursor(&self) -> u32 {
        self.cursor
    }

    pub fn scene_config(&self) -> &SceneConfig {
        &self.ctx.scene_config
    }

    pub fn ctx_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }

    pub fn set_background_color(&mut self, color: Color) {
        self.initial_scene.background_color = color;
        self.build_scene.background_color = color;
    }

    pub fn scene_config_mut(&mut self) -> &mut SceneConfig {
        &mut self.ctx.scene_config
    }

    pub fn scene_view(&self) -> SceneView<'_> {
        SceneView::new(&self.build_scene)
    }

    pub fn record_properties(
        &self,
        duration: u32,
    ) -> Result<PropertyWriteRecorder, RecordingError> {
        PropertyWriteRecorder::new(&self.build_scene, duration)
    }

    pub fn reserve_spawn(
        &mut self,
        bundle: NodeBundle,
        parent: Option<MobjectId>,
    ) -> Result<SpawnPlan, TimelineError> {
        if let Some(parent) = parent {
            self.build_scene.world.get(parent)?;
        }
        let plan = self.build_scene.world.reserve_tree(bundle, parent);
        self.initial_scene
            .synchronize_identities_from(&self.build_scene);
        Ok(plan)
    }

    pub fn apply(&mut self, operation: SceneOperation) -> Result<(), TimelineError> {
        apply_operation(&mut self.build_scene, &operation)?;
        if self.cursor == 0 {
            self.initial_scene = self.build_scene.snapshot();
            return Ok(());
        }

        self.events.push(TimedEvent {
            frame: self.cursor,
            order: self.next_event_order,
            operation,
        });
        self.next_event_order += 1;
        self.events.sort_by_key(|event| (event.frame, event.order));
        Ok(())
    }

    pub fn spawn_now(&mut self, plan: SpawnPlan) -> Result<MobjectId, TimelineError> {
        let root = plan.root();
        self.apply(SceneOperation::Spawn { plan })?;
        Ok(root)
    }

    pub fn add(
        &mut self,
        bundle: NodeBundle,
        parent: Option<MobjectId>,
    ) -> Result<MobjectId, TimelineError> {
        let plan = self.reserve_spawn(bundle, parent)?;
        self.spawn_now(plan)
    }

    pub fn remove(&mut self, target: MobjectId) -> Result<(), TimelineError> {
        self.apply(SceneOperation::Remove { target })
    }

    pub fn set<P: Property>(&mut self, property: P, value: P::Value) -> Result<(), TimelineError> {
        self.apply(SceneOperation::write(property, value))
    }

    pub fn append_clip(&mut self, clip: AnimationClip) -> Result<(), TimelineError> {
        let (duration, tracks, events) = clip.into_parts();
        if duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        for track in &tracks {
            track.validate(&self.build_scene, duration)?;
        }
        for event in &events {
            if event.frame == 0 || event.frame > duration {
                return Err(TimelineError::EventOutsideClip {
                    frame: event.frame,
                    duration,
                });
            }
        }

        let range = FrameRange {
            start: self.cursor,
            end: self.cursor + duration,
        };
        let mut new_tracks = Vec::with_capacity(tracks.len());
        for track in tracks {
            let compiled = track.compile(range);
            self.validate_conflicts(&compiled, &new_tracks)?;
            new_tracks.push(compiled);
        }

        let mut new_events = Vec::with_capacity(events.len());
        for event in events {
            let event = TimedEvent {
                frame: self.cursor + event.frame,
                order: self.next_event_order,
                operation: event.operation,
            };
            self.next_event_order += 1;
            new_events.push(event);
        }
        new_events.sort_by_key(|event| (event.frame, event.order));

        for event in &new_events {
            apply_operation(&mut self.build_scene, &event.operation)?;
        }
        for track in &new_tracks {
            track.apply(range.end, &mut self.build_scene)?;
        }

        self.tracks.extend(new_tracks);
        self.events.extend(new_events);
        self.events.sort_by_key(|event| (event.frame, event.order));
        self.cursor = range.end;
        Ok(())
    }

    fn validate_conflicts(
        &self,
        candidate: &CompiledTrack,
        pending: &[CompiledTrack],
    ) -> Result<(), TimelineError> {
        for existing in self.tracks.iter().chain(pending) {
            if existing.address() == candidate.address()
                && existing.range().overlaps(candidate.range())
                && existing.composition() == Composition::Replace
                && candidate.composition() == Composition::Replace
            {
                return Err(TimelineError::ConflictingWrites {
                    address: candidate.address(),
                    first: existing.range(),
                    second: candidate.range(),
                });
            }
        }
        Ok(())
    }

    pub fn build(self) -> CompiledTimeline {
        CompiledTimeline::new(
            self.initial_scene,
            self.ctx,
            self.tracks,
            self.events,
            self.cursor,
        )
    }
}
