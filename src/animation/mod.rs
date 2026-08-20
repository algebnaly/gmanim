use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    Context, GMFloat, Scene, SceneSnapshot,
    mobjects::{MobjectId, NodeBundle, SceneWorldError, SpawnPlan},
};

mod property;
mod recording;
pub use property::{
    AaLevelProperty, CameraPoseProperty, CameraProjectionProperty, EnvironmentLightProperty,
    ErasedProperty, LayerProperty, PointLightProperty, Property, PropertyAddress, PropertyError,
    PropertyKey, PropertyTarget, PropertyValue, RectangleCornersProperty, TrackValue,
    TransformProperty, ViewportProperty, VisibilityProperty,
};
pub use recording::{PropertyWriteFrame, PropertyWriteRecorder, RecordingError, SceneView};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameRange {
    pub start: u32,
    pub end: u32,
}

impl FrameRange {
    pub fn duration(self) -> u32 {
        self.end - self.start
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Composition {
    Replace,
}

#[derive(Clone)]
pub enum Curve<T> {
    Linear { from: T, to: T },
    Sampled(Arc<[T]>),
}

impl<T> Curve<T> {
    pub fn linear(from: T, to: T) -> Self {
        Self::Linear { from, to }
    }

    pub fn sampled(values: impl Into<Arc<[T]>>) -> Self {
        Self::Sampled(values.into())
    }

    fn validate(&self, duration: u32) -> Result<(), TimelineError> {
        if let Self::Sampled(values) = self
            && values.len() != duration as usize + 1
        {
            return Err(TimelineError::SampleCount {
                expected: duration as usize + 1,
                actual: values.len(),
            });
        }
        Ok(())
    }
}

fn curve_value_at<T: TrackValue>(curve: &Curve<T>, frame: u32, duration: u32) -> T {
    match curve {
        Curve::Linear { from, to } => {
            let alpha = frame as GMFloat / duration as GMFloat;
            TrackValue::interpolate(from, to, alpha)
        }
        Curve::Sampled(values) => values[frame as usize].clone(),
    }
}

trait TrackEvaluator: Send + Sync {
    fn address(&self) -> PropertyAddress;
    fn validate(&self, scene: &Scene, duration: u32) -> Result<(), TimelineError>;
    fn apply(&self, frame: u32, duration: u32, scene: &mut Scene) -> Result<(), TimelineError>;
}

struct TypedTrack<P: Property> {
    property: P,
    curve: Curve<P::Value>,
}

impl<P: Property> TrackEvaluator for TypedTrack<P> {
    fn address(&self) -> PropertyAddress {
        self.property.address()
    }

    fn validate(&self, scene: &Scene, duration: u32) -> Result<(), TimelineError> {
        self.property.read(scene)?;
        self.curve.validate(duration)
    }

    fn apply(&self, frame: u32, duration: u32, scene: &mut Scene) -> Result<(), TimelineError> {
        if !self.property.is_present(scene)? {
            return Ok(());
        }
        self.property
            .write(scene, curve_value_at(&self.curve, frame, duration))?;
        if frame == duration {
            self.property.finalize(scene)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct ClipTrack {
    address: PropertyAddress,
    composition: Composition,
    evaluator: Arc<dyn TrackEvaluator>,
}

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
struct ClipEvent {
    frame: u32,
    operation: SceneOperation,
}

#[derive(Clone)]
pub struct AnimationClip {
    duration: u32,
    tracks: Vec<ClipTrack>,
    events: Vec<ClipEvent>,
}

impl AnimationClip {
    pub fn new(duration: u32) -> Self {
        Self {
            duration,
            tracks: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn duration(&self) -> u32 {
        self.duration
    }

    pub fn track<P: Property>(mut self, property: P, curve: Curve<P::Value>) -> Self {
        let address = property.address();
        self.tracks.push(ClipTrack {
            address,
            composition: Composition::Replace,
            evaluator: Arc::new(TypedTrack { property, curve }),
        });
        self
    }

    pub fn transform(self, target: MobjectId, curve: Curve<Matrix4<GMFloat>>) -> Self {
        self.track(TransformProperty::new(target), curve)
    }

    pub fn rectangle_corners(self, target: MobjectId, curve: Curve<[Point3<GMFloat>; 4]>) -> Self {
        self.track(RectangleCornersProperty::new(target), curve)
    }

    pub fn visibility(self, target: MobjectId, curve: Curve<bool>) -> Self {
        self.track(VisibilityProperty::new(target), curve)
    }

    pub fn layer(self, target: MobjectId, curve: Curve<i32>) -> Self {
        self.track(LayerProperty::new(target), curve)
    }

    pub fn event(mut self, frame: u32, operation: SceneOperation) -> Self {
        self.events.push(ClipEvent { frame, operation });
        self
    }

    pub fn spawn(self, frame: u32, plan: SpawnPlan) -> Self {
        self.event(frame, SceneOperation::Spawn { plan })
    }

    pub fn write<P: Property>(self, frame: u32, property: P, value: P::Value) -> Self {
        self.event(frame, SceneOperation::write(property, value))
    }
}

pub trait AnimationBuilder {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError>;
}

impl AnimationBuilder for AnimationClip {
    fn compile(self, _scene: &Scene) -> Result<AnimationClip, TimelineError> {
        Ok(self)
    }
}

pub struct Move {
    target: MobjectId,
    displacement: Vector3<GMFloat>,
    duration: u32,
}

impl Move {
    pub fn new(target: MobjectId, displacement: Vector3<GMFloat>, duration: u32) -> Self {
        Self {
            target,
            displacement,
            duration,
        }
    }
}

impl AnimationBuilder for Move {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError> {
        if self.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        let start = scene.world.get(self.target)?.transform();
        let values: Arc<[Matrix4<GMFloat>]> = (0..=self.duration)
            .map(|frame| {
                let alpha = frame as GMFloat / self.duration as GMFloat;
                Matrix4::new_translation(&(self.displacement * alpha)) * start
            })
            .collect::<Vec<_>>()
            .into();
        Ok(AnimationClip::new(self.duration).transform(self.target, Curve::Sampled(values)))
    }
}

pub struct Rotate {
    target: MobjectId,
    axis_angle: Vector3<GMFloat>,
    center: Point3<GMFloat>,
    duration: u32,
}

impl Rotate {
    pub fn new(
        target: MobjectId,
        axis_angle: Vector3<GMFloat>,
        center: Point3<GMFloat>,
        duration: u32,
    ) -> Self {
        Self {
            target,
            axis_angle,
            center,
            duration,
        }
    }
}

impl AnimationBuilder for Rotate {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError> {
        if self.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        let start = scene.world.get(self.target)?.transform();
        let values: Arc<[Matrix4<GMFloat>]> = (0..=self.duration)
            .map(|frame| {
                let alpha = frame as GMFloat / self.duration as GMFloat;
                Matrix4::new_rotation_wrt_point(self.axis_angle * alpha, self.center) * start
            })
            .collect::<Vec<_>>()
            .into();
        Ok(AnimationClip::new(self.duration).transform(self.target, Curve::Sampled(values)))
    }
}

pub struct MorphRectangle {
    target: MobjectId,
    target_corners: [Point3<GMFloat>; 4],
    duration: u32,
}

impl MorphRectangle {
    pub fn new(target: MobjectId, target_corners: [Point3<GMFloat>; 4], duration: u32) -> Self {
        Self {
            target,
            target_corners,
            duration,
        }
    }
}

impl AnimationBuilder for MorphRectangle {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError> {
        if self.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        let start = scene.world.rectangle(self.target)?.corners();
        Ok(AnimationClip::new(self.duration).rectangle_corners(
            self.target,
            Curve::Linear {
                from: start,
                to: self.target_corners,
            },
        ))
    }
}

pub struct Wait {
    duration: u32,
}

impl Wait {
    pub fn new(duration: u32) -> Self {
        Self { duration }
    }
}

impl AnimationBuilder for Wait {
    fn compile(self, _scene: &Scene) -> Result<AnimationClip, TimelineError> {
        Ok(AnimationClip::new(self.duration))
    }
}

#[derive(Clone)]
struct CompiledTrack {
    range: FrameRange,
    address: PropertyAddress,
    composition: Composition,
    evaluator: Arc<dyn TrackEvaluator>,
}

impl CompiledTrack {
    fn apply(&self, frame: u32, scene: &mut Scene) -> Result<(), TimelineError> {
        if frame <= self.range.start {
            return Ok(());
        }
        let local_frame = (frame - self.range.start).min(self.range.duration());
        self.evaluator
            .apply(local_frame, self.range.duration(), scene)
    }
}

#[derive(Clone, Debug)]
struct TimedEvent {
    frame: u32,
    order: u64,
    operation: SceneOperation,
}

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

    pub fn scene_config(&self) -> &crate::SceneConfig {
        &self.ctx.scene_config
    }

    pub fn ctx_mut(&mut self) -> &mut crate::Context {
        &mut self.ctx
    }

    pub fn set_background_color(&mut self, color: crate::Color) {
        self.initial_scene.background_color = color;
        self.build_scene.background_color = color;
    }

    pub fn scene_config_mut(&mut self) -> &mut crate::SceneConfig {
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
        if clip.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        for track in &clip.tracks {
            track.evaluator.validate(&self.build_scene, clip.duration)?;
        }
        for event in &clip.events {
            if event.frame == 0 || event.frame > clip.duration {
                return Err(TimelineError::EventOutsideClip {
                    frame: event.frame,
                    duration: clip.duration,
                });
            }
        }

        let range = FrameRange {
            start: self.cursor,
            end: self.cursor + clip.duration,
        };
        let mut new_tracks = Vec::with_capacity(clip.tracks.len());
        for track in clip.tracks {
            let compiled = CompiledTrack {
                range,
                address: track.address,
                composition: track.composition,
                evaluator: track.evaluator,
            };
            self.validate_conflicts(&compiled, &new_tracks)?;
            new_tracks.push(compiled);
        }

        let mut new_events = Vec::with_capacity(clip.events.len());
        for event in clip.events {
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
            if existing.address == candidate.address
                && existing.range.overlaps(candidate.range)
                && existing.composition == Composition::Replace
                && candidate.composition == Composition::Replace
            {
                return Err(TimelineError::ConflictingWrites {
                    address: candidate.address,
                    first: existing.range,
                    second: candidate.range,
                });
            }
        }
        Ok(())
    }

    pub fn build(self) -> CompiledTimeline {
        CompiledTimeline {
            scene: Scene::from_snapshot(&self.initial_scene),
            initial_scene: self.initial_scene,
            ctx: self.ctx,
            tracks: self.tracks,
            events: self.events,
            total_frames: self.cursor,
            current_frame: 0,
        }
    }
}

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

fn apply_operation(scene: &mut Scene, operation: &SceneOperation) -> Result<(), TimelineError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobjects::Rectangle;

    #[test]
    fn seek_is_absolute_and_bidirectional() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder
            .play(Move::new(rectangle, Vector3::new(10.0, 0.0, 0.0), 10))
            .unwrap();
        let mut timeline = builder.build();

        timeline.seek(7).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            7.0
        );
        timeline.seek(2).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            2.0
        );
        timeline.seek(10).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            10.0
        );
    }

    #[test]
    fn sequential_builders_resolve_start_values_at_compile_time() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder
            .play(Move::new(rectangle, Vector3::new(2.0, 0.0, 0.0), 2))
            .unwrap();
        builder
            .play(Move::new(rectangle, Vector3::new(3.0, 0.0, 0.0), 3))
            .unwrap();
        let mut timeline = builder.build();

        timeline.seek(5).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            5.0
        );
        timeline.seek(3).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            3.0
        );
    }

    #[test]
    fn morph_rectangle_compiles_to_corner_track() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let target = [
            Point3::new(-1.0, 2.0, 0.0),
            Point3::new(-2.0, -1.0, 0.0),
            Point3::new(3.0, -2.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder
            .play(MorphRectangle::new(rectangle, target, 10))
            .unwrap();
        let mut timeline = builder.build();

        timeline.seek(5).unwrap();
        let corners = timeline.scene.world.rectangle(rectangle).unwrap().corners();
        let start = Rectangle::default().corners();
        for index in 0..4 {
            assert_eq!(
                corners[index],
                start[index] + (target[index] - start[index]) * 0.5
            );
        }
        timeline.seek(10).unwrap();
        assert_eq!(
            timeline.scene.world.rectangle(rectangle).unwrap().corners(),
            target
        );
    }

    #[test]
    fn sampled_tracks_validate_frame_count() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let curve = Curve::sampled(vec![Matrix4::identity(); 2]);
        let clip = AnimationClip::new(3).transform(rectangle, curve);
        let mut builder = TimelineBuilder::new(scene, Context::default());

        assert!(matches!(
            builder.play(clip),
            Err(TimelineError::SampleCount {
                expected: 4,
                actual: 2
            })
        ));
    }

    #[test]
    fn duplicate_property_writes_in_one_clip_are_rejected() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let curve = Curve::linear(Matrix4::identity(), Matrix4::new_scaling(2.0));
        let clip = AnimationClip::new(10)
            .transform(rectangle, curve.clone())
            .transform(rectangle, curve);
        let mut builder = TimelineBuilder::new(scene, Context::default());

        assert!(matches!(
            builder.play(clip),
            Err(TimelineError::ConflictingWrites { .. })
        ));
    }

    #[test]
    fn structural_events_replay_after_backward_seek() {
        let mut scene = Scene::default();
        let parent = scene.world.spawn_group("parent");
        let rectangle = scene.add_rectangle(Rectangle::default());
        let clip = AnimationClip::new(10).event(
            5,
            SceneOperation::SetParent {
                child: rectangle,
                parent: Some(parent),
            },
        );
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.play(clip).unwrap();
        let mut timeline = builder.build();

        timeline.seek(8).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().parent(),
            Some(parent)
        );
        timeline.seek(2).unwrap();
        assert_eq!(timeline.scene.world.get(rectangle).unwrap().parent(), None);
    }

    #[test]
    fn spawn_events_materialize_reserved_ids_at_the_target_frame() {
        let scene = Scene::default();
        let mut builder = TimelineBuilder::new(scene, Context::default());
        let plan = builder
            .reserve_spawn(NodeBundle::rectangle(Rectangle::default()), None)
            .unwrap();
        let rectangle = plan.root();
        builder
            .append_clip(AnimationClip::new(5).spawn(3, plan))
            .unwrap();
        let mut timeline = builder.build();

        assert!(timeline.scene.world.is_reserved(rectangle));
        timeline.seek(2).unwrap();
        assert!(timeline.scene.world.is_reserved(rectangle));
        timeline.seek(3).unwrap();
        assert!(timeline.scene.world.contains(rectangle));
        assert!(timeline.scene.world.rectangle(rectangle).is_ok());
        timeline.seek(1).unwrap();
        assert!(timeline.scene.world.is_reserved(rectangle));
        timeline.seek(5).unwrap();
        assert!(timeline.scene.world.contains(rectangle));
    }

    #[test]
    fn spawned_trees_keep_reserved_child_ids_and_external_parent() {
        let mut scene = Scene::default();
        let parent = scene.world.spawn_group("parent");
        let mut builder = TimelineBuilder::new(scene, Context::default());
        let plan = builder
            .reserve_spawn(
                NodeBundle::group("spawned-root")
                    .with_child(NodeBundle::rectangle(Rectangle::default())),
                Some(parent),
            )
            .unwrap();
        let ids = plan.ids();
        builder
            .append_clip(AnimationClip::new(2).spawn(1, plan))
            .unwrap();
        let mut timeline = builder.build();

        timeline.seek(1).unwrap();
        assert_eq!(timeline.scene.world.children(parent).unwrap(), [ids[0]]);
        assert_eq!(timeline.scene.world.children(ids[0]).unwrap(), [ids[1]]);
        assert_eq!(
            timeline.scene.world.get(ids[1]).unwrap().parent(),
            Some(ids[0])
        );
    }

    #[test]
    fn remove_events_preserve_identity_for_backward_seek() {
        let scene = Scene::default();
        let mut builder = TimelineBuilder::new(scene, Context::default());
        let plan = builder
            .reserve_spawn(NodeBundle::rectangle(Rectangle::default()), None)
            .unwrap();
        let rectangle = plan.root();
        builder
            .append_clip(AnimationClip::new(2).spawn(1, plan))
            .unwrap();
        builder
            .append_clip(
                AnimationClip::new(2).event(1, SceneOperation::Remove { target: rectangle }),
            )
            .unwrap();
        let mut timeline = builder.build();

        timeline.seek(4).unwrap();
        assert!(timeline.scene.world.is_reserved(rectangle));
        timeline.seek(2).unwrap();
        assert!(timeline.scene.world.contains(rectangle));
        timeline.seek(0).unwrap();
        assert!(timeline.scene.world.is_reserved(rectangle));
        timeline.seek(2).unwrap();
        assert!(timeline.scene.world.contains(rectangle));
    }

    #[test]
    fn frame_zero_boundary_operations_fold_into_the_initial_snapshot() {
        let scene = Scene::default();
        let mut builder = TimelineBuilder::new(scene, Context::default());
        let rectangle = builder
            .add(NodeBundle::rectangle(Rectangle::default()), None)
            .unwrap();
        builder.play(Wait::new(2)).unwrap();
        let mut timeline = builder.build();

        assert_eq!(timeline.total_frames(), 2);
        assert!(timeline.scene.world.contains(rectangle));
        timeline.seek(2).unwrap();
        timeline.seek(0).unwrap();
        assert!(timeline.scene.world.contains(rectangle));
    }

    #[test]
    fn later_boundary_operations_do_not_consume_frames() {
        let mut scene = Scene::default();
        let original = scene.add_rectangle(Rectangle::default());
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.play(Wait::new(2)).unwrap();
        let added = builder
            .add(NodeBundle::rectangle(Rectangle::default()), None)
            .unwrap();
        builder.remove(original).unwrap();
        builder.play(Wait::new(2)).unwrap();
        let mut timeline = builder.build();

        assert_eq!(timeline.total_frames(), 4);
        timeline.seek(1).unwrap();
        assert!(timeline.scene.world.contains(original));
        assert!(timeline.scene.world.is_reserved(added));
        timeline.seek(2).unwrap();
        assert!(timeline.scene.world.is_reserved(original));
        assert!(timeline.scene.world.contains(added));
        timeline.seek(4).unwrap();
        timeline.seek(1).unwrap();
        assert!(timeline.scene.world.contains(original));
        assert!(timeline.scene.world.is_reserved(added));
    }

    #[test]
    fn scene_properties_can_change_at_timeline_boundaries() {
        let scene = Scene::default();
        let initial_pose = scene.camera.pose();
        let target_pose = crate::camera::CameraPose {
            position: Point3::new(3.0, 4.0, 5.0),
            look_at: -Vector3::z(),
            up_direction: Vector3::y(),
        };
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.play(Wait::new(2)).unwrap();
        builder.set(CameraPoseProperty, target_pose).unwrap();
        builder.set(AaLevelProperty, 6).unwrap();
        builder.play(Wait::new(2)).unwrap();
        let mut timeline = builder.build();

        timeline.seek(1).unwrap();
        assert_eq!(timeline.scene.camera.pose(), initial_pose);
        assert_eq!(timeline.scene.aa_level, 1);
        timeline.seek(2).unwrap();
        assert_eq!(timeline.scene.camera.pose(), target_pose);
        assert_eq!(timeline.scene.aa_level, 6);
        timeline.seek(4).unwrap();
        timeline.seek(1).unwrap();
        assert_eq!(timeline.scene.camera.pose(), initial_pose);
        assert_eq!(timeline.scene.aa_level, 1);
    }
}
