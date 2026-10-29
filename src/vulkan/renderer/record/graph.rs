use ash::vk;

use super::super::frame::{FrameExecutionPlan, TrackedImageState};
use super::super::output::RenderOutputs;
use super::plan::RecordingPlan;
use super::{ColorLoad, DepthLoad, Mesh2DPass, Mesh3DPass, RasterRegion};

pub(in crate::vulkan::renderer) const SLOT_IMAGE_COUNT: usize = 16;
pub(in crate::vulkan::renderer) const SHARED_IMAGE_COUNT: usize = 5;
pub(in crate::vulkan::renderer) const TOTAL_GRAPH_IMAGE_COUNT: usize =
    SLOT_IMAGE_COUNT + SHARED_IMAGE_COUNT + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub(in crate::vulkan::renderer) enum GraphImageId {
    // Per-frame-slot images (0..16)
    OutputRgba = 0,
    SdfNormalCoverage = 1,
    SdfMaterialId = 2,
    SdfDepth = 3,
    ResolvedPrimaryNormalDepth = 4,
    ResolvedPrimaryAlbedoCoverage = 5,
    ResolvedSecondaryNormalDepth = 6,
    ResolvedSecondaryAlbedoCoverage = 7,
    ResolvedMaterialIds = 8,
    SurfaceHdr = 9,
    OverlayHdr = 10,
    ResolvedColor = 11,
    SceneColor = 12,
    TransparentBackDepth = 13,
    BloomPing = 14,
    BloomPong = 15,
    // Shared cache images (16..21)
    RasterNormalDepth = 16,
    RasterAlbedo = 17,
    RasterMaterialId = 18,
    MsaaColor = 19,
    MsaaDepth = 20,
    // Active Vulkan Video slot image (21)
    VideoNv12 = 21,
}

impl GraphImageId {
    pub(in crate::vulkan::renderer) const fn index(self) -> usize {
        self as usize
    }

    pub(in crate::vulkan::renderer) const fn slot_index(self) -> Option<usize> {
        let idx = self as usize;
        if idx < SLOT_IMAGE_COUNT {
            Some(idx)
        } else {
            None
        }
    }

    pub(in crate::vulkan::renderer) const fn shared_index(self) -> Option<usize> {
        let idx = self as usize;
        if idx >= SLOT_IMAGE_COUNT && idx < SLOT_IMAGE_COUNT + SHARED_IMAGE_COUNT {
            Some(idx - SLOT_IMAGE_COUNT)
        } else {
            None
        }
    }

    pub(in crate::vulkan::renderer) fn aspect_mask(self) -> vk::ImageAspectFlags {
        match self {
            Self::MsaaDepth => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::vulkan::renderer) enum ImageUsage {
    ClearWrite,
    ComputeStorageWrite,
    ComputeSampledRead,
    ColorAttachmentWrite,
    DepthAttachmentReadWrite,
    TransferRead,
    TransferWrite,
    FragmentSampledRead,
    FragmentGeneralRead,
}

impl ImageUsage {
    pub(in crate::vulkan::renderer) fn tracked_state(self) -> TrackedImageState {
        match self {
            Self::ClearWrite => TrackedImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: vk::PipelineStageFlags2::CLEAR,
                access: vk::AccessFlags2::TRANSFER_WRITE,
            },
            Self::ComputeStorageWrite => TrackedImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            Self::ComputeSampledRead => TrackedImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            Self::ColorAttachmentWrite => TrackedImageState {
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            Self::DepthAttachmentReadWrite => TrackedImageState {
                layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            },
            Self::TransferRead => TrackedImageState {
                layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage: vk::PipelineStageFlags2::COPY,
                access: vk::AccessFlags2::TRANSFER_READ,
            },
            Self::TransferWrite => TrackedImageState {
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                stage: vk::PipelineStageFlags2::COPY,
                access: vk::AccessFlags2::TRANSFER_WRITE,
            },
            Self::FragmentSampledRead => TrackedImageState {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            Self::FragmentGeneralRead => TrackedImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
        }
    }

    pub(in crate::vulkan::renderer) const fn is_write(self) -> bool {
        matches!(
            self,
            Self::ClearWrite
                | Self::ComputeStorageWrite
                | Self::ColorAttachmentWrite
                | Self::DepthAttachmentReadWrite
                | Self::TransferWrite
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::vulkan::renderer) struct ResourceAccess {
    pub(in crate::vulkan::renderer) image: GraphImageId,
    pub(in crate::vulkan::renderer) usage: ImageUsage,
}

impl ResourceAccess {
    pub(in crate::vulkan::renderer) const fn new(image: GraphImageId, usage: ImageUsage) -> Self {
        Self { image, usage }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(in crate::vulkan::renderer) enum GraphPhase {
    Sdf = 0,
    SurfaceAndRaster = 1,
    Postprocess = 2,
    Output = 3,
    Present = 4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vulkan::renderer) struct ColorRasterGraphPass {
    pub(in crate::vulkan::renderer) target: GraphImageId,
    pub(in crate::vulkan::renderer) use_msaa: bool,
    pub(in crate::vulkan::renderer) preserve_multisample: bool,
    pub(in crate::vulkan::renderer) color_load: ColorLoad,
    pub(in crate::vulkan::renderer) depth: Option<(DepthLoad, bool)>,
    pub(in crate::vulkan::renderer) region: RasterRegion,
    pub(in crate::vulkan::renderer) meshes_3d: Option<Mesh3DPass>,
    pub(in crate::vulkan::renderer) grids_3d: bool,
    pub(in crate::vulkan::renderer) meshes_2d: Option<Mesh2DPass>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vulkan::renderer) enum PassKind {
    ClearEmpty {
        background_color: [f32; 4],
    },
    SdfRaymarch {
        extent: vk::Extent2D,
    },
    DeferredOpaque {
        region: RasterRegion,
        preserve_depth: bool,
    },
    SurfaceResolve {
        extent: vk::Extent2D,
    },
    SurfaceLighting {
        extent: vk::Extent2D,
    },
    CopyColorImage {
        src: GraphImageId,
        dst: GraphImageId,
        extent: vk::Extent2D,
    },
    TransparentBackDepth {
        extent: vk::Extent2D,
    },
    ColorRaster(ColorRasterGraphPass),
    SurfaceComposite {
        has_overlay: bool,
        width: u32,
        height: u32,
        raster_scale: u32,
    },
    BloomExtract,
    BloomHorizontal,
    BloomVertical,
    BloomClear,
    ToneMap {
        width: u32,
        height: u32,
    },
    OutputCpuNv12 {
        width: u32,
        height: u32,
    },
    OutputCpuYuv444p {
        width: u32,
        height: u32,
    },
    OutputVulkanVideo {
        width: u32,
        height: u32,
        fused_downsample: bool,
    },
    CopyRgbaToBuffer {
        width: u32,
        height: u32,
        padded_bytes_per_row: u32,
    },
    PresentReady,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::vulkan::renderer) struct RenderGraphPass {
    pub(in crate::vulkan::renderer) phase: GraphPhase,
    pub(in crate::vulkan::renderer) kind: PassKind,
    pub(in crate::vulkan::renderer) accesses: Vec<ResourceAccess>,
}

impl RenderGraphPass {
    fn new(phase: GraphPhase, kind: PassKind, accesses: Vec<ResourceAccess>) -> Self {
        debug_assert!(
            {
                let mut seen = [false; TOTAL_GRAPH_IMAGE_COUNT];
                accesses.iter().all(|access| {
                    let idx = access.image.index();
                    let unique = !seen[idx];
                    seen[idx] = true;
                    unique
                })
            },
            "duplicate GraphImageId in a single RenderGraphPass"
        );
        Self {
            phase,
            kind,
            accesses,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vulkan::renderer) struct BarrierTransition {
    pub(in crate::vulkan::renderer) image: GraphImageId,
    pub(in crate::vulkan::renderer) aspect_mask: vk::ImageAspectFlags,
    pub(in crate::vulkan::renderer) old: TrackedImageState,
    pub(in crate::vulkan::renderer) new: TrackedImageState,
}

#[derive(Clone, Debug)]
pub(in crate::vulkan::renderer) struct ImageStateTable {
    states: [TrackedImageState; TOTAL_GRAPH_IMAGE_COUNT],
}

impl Default for ImageStateTable {
    fn default() -> Self {
        Self {
            states: [TrackedImageState::UNDEFINED; TOTAL_GRAPH_IMAGE_COUNT],
        }
    }
}

impl ImageStateTable {
    pub(in crate::vulkan::renderer) fn from_parts(
        slot_states: &[TrackedImageState; SLOT_IMAGE_COUNT],
        shared_states: &[TrackedImageState; SHARED_IMAGE_COUNT],
        video_layout: Option<vk::ImageLayout>,
    ) -> Self {
        let mut states = [TrackedImageState::UNDEFINED; TOTAL_GRAPH_IMAGE_COUNT];
        states[..SLOT_IMAGE_COUNT].copy_from_slice(slot_states);
        states[SLOT_IMAGE_COUNT..SLOT_IMAGE_COUNT + SHARED_IMAGE_COUNT]
            .copy_from_slice(shared_states);
        if let Some(layout) = video_layout {
            states[GraphImageId::VideoNv12.index()] = TrackedImageState {
                layout,
                stage: vk::PipelineStageFlags2::NONE,
                access: vk::AccessFlags2::NONE,
            };
        }
        Self { states }
    }

    pub(in crate::vulkan::renderer) fn write_back(
        &self,
        slot_states: &mut [TrackedImageState; SLOT_IMAGE_COUNT],
        shared_states: &mut [TrackedImageState; SHARED_IMAGE_COUNT],
        video_layout: Option<&mut vk::ImageLayout>,
    ) {
        slot_states.copy_from_slice(&self.states[..SLOT_IMAGE_COUNT]);
        shared_states.copy_from_slice(
            &self.states[SLOT_IMAGE_COUNT..SLOT_IMAGE_COUNT + SHARED_IMAGE_COUNT],
        );
        if let Some(layout) = video_layout {
            *layout = self.states[GraphImageId::VideoNv12.index()].layout;
        }
    }

    pub(in crate::vulkan::renderer) fn state(&self, image: GraphImageId) -> TrackedImageState {
        self.states[image.index()]
    }

    pub(in crate::vulkan::renderer) fn plan_pass_barriers(
        &mut self,
        accesses: &[ResourceAccess],
        is_present: impl Fn(GraphImageId) -> bool,
    ) -> Vec<BarrierTransition> {
        let mut transitions = Vec::with_capacity(accesses.len());
        for &access in accesses {
            if !is_present(access.image) {
                continue;
            }
            let current = &mut self.states[access.image.index()];
            let required = access.usage.tracked_state();
            if !access.usage.is_write()
                && current.layout == required.layout
                && current.stage == required.stage
                && current.access == required.access
            {
                continue;
            }
            transitions.push(BarrierTransition {
                image: access.image,
                aspect_mask: access.image.aspect_mask(),
                old: *current,
                new: required,
            });
            *current = required;
        }
        transitions
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::vulkan::renderer) struct RenderGraph {
    passes: Vec<RenderGraphPass>,
}

impl RenderGraph {
    pub(in crate::vulkan::renderer) fn compile(
        plan: RecordingPlan,
        outputs: RenderOutputs,
        padded_bytes_per_row: u32,
        bloom_needs_clear: bool,
    ) -> Self {
        let mut passes = Vec::with_capacity(12);

        // Phase 1: SDF / Empty
        if plan.execution == FrameExecutionPlan::Empty {
            passes.push(RenderGraphPass::new(
                GraphPhase::Sdf,
                PassKind::ClearEmpty {
                    background_color: plan.background_color,
                },
                vec![ResourceAccess::new(
                    GraphImageId::OutputRgba,
                    ImageUsage::ClearWrite,
                )],
            ));
        } else if plan.execution.runs_sdf() {
            passes.push(RenderGraphPass::new(
                GraphPhase::Sdf,
                PassKind::SdfRaymarch {
                    extent: plan.ssaa_extent(),
                },
                vec![
                    ResourceAccess::new(
                        GraphImageId::SdfNormalCoverage,
                        ImageUsage::ComputeStorageWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::SdfMaterialId,
                        ImageUsage::ComputeStorageWrite,
                    ),
                    ResourceAccess::new(GraphImageId::SdfDepth, ImageUsage::ComputeStorageWrite),
                ],
            ));
        }

        // Phase 2: Surface & Raster
        if plan.uses_deferred_raster {
            let raster_extent = plan.ssaa_extent();
            passes.push(RenderGraphPass::new(
                GraphPhase::SurfaceAndRaster,
                PassKind::DeferredOpaque {
                    region: RasterRegion::new(
                        raster_extent,
                        plan.camera_clip,
                        plan.camera_raster_scale,
                    ),
                    preserve_depth: plan.has_transparent_meshes || plan.has_grid_3d,
                },
                vec![
                    ResourceAccess::new(
                        GraphImageId::RasterNormalDepth,
                        ImageUsage::ColorAttachmentWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::RasterAlbedo,
                        ImageUsage::ColorAttachmentWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::RasterMaterialId,
                        ImageUsage::ColorAttachmentWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::MsaaDepth,
                        ImageUsage::DepthAttachmentReadWrite,
                    ),
                ],
            ));
        }

        if plan.uses_deferred_raster || plan.execution.runs_sdf() {
            let surface_extent = plan.ssaa_extent();
            passes.push(RenderGraphPass::new(
                GraphPhase::SurfaceAndRaster,
                PassKind::SurfaceResolve {
                    extent: surface_extent,
                },
                vec![
                    ResourceAccess::new(
                        GraphImageId::SdfNormalCoverage,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(GraphImageId::SdfDepth, ImageUsage::ComputeSampledRead),
                    ResourceAccess::new(
                        GraphImageId::SdfMaterialId,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(
                        GraphImageId::RasterNormalDepth,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(GraphImageId::RasterAlbedo, ImageUsage::ComputeSampledRead),
                    ResourceAccess::new(
                        GraphImageId::RasterMaterialId,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedPrimaryNormalDepth,
                        ImageUsage::ComputeStorageWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedPrimaryAlbedoCoverage,
                        ImageUsage::ComputeStorageWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedSecondaryNormalDepth,
                        ImageUsage::ComputeStorageWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedSecondaryAlbedoCoverage,
                        ImageUsage::ComputeStorageWrite,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedMaterialIds,
                        ImageUsage::ComputeStorageWrite,
                    ),
                ],
            ));

            passes.push(RenderGraphPass::new(
                GraphPhase::SurfaceAndRaster,
                PassKind::SurfaceLighting {
                    extent: surface_extent,
                },
                vec![
                    ResourceAccess::new(
                        GraphImageId::ResolvedPrimaryNormalDepth,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedPrimaryAlbedoCoverage,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedSecondaryNormalDepth,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedSecondaryAlbedoCoverage,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(
                        GraphImageId::ResolvedMaterialIds,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(GraphImageId::SurfaceHdr, ImageUsage::ComputeStorageWrite),
                ],
            ));
        }

        if plan.uses_deferred_raster && plan.has_surface_overlay {
            let raster_extent = plan.ssaa_extent();
            if plan.has_transparent_meshes {
                passes.push(RenderGraphPass::new(
                    GraphPhase::SurfaceAndRaster,
                    PassKind::CopyColorImage {
                        src: GraphImageId::SurfaceHdr,
                        dst: GraphImageId::SceneColor,
                        extent: raster_extent,
                    },
                    vec![
                        ResourceAccess::new(GraphImageId::SurfaceHdr, ImageUsage::TransferRead),
                        ResourceAccess::new(GraphImageId::SceneColor, ImageUsage::TransferWrite),
                    ],
                ));
                passes.push(RenderGraphPass::new(
                    GraphPhase::SurfaceAndRaster,
                    PassKind::TransparentBackDepth {
                        extent: raster_extent,
                    },
                    vec![ResourceAccess::new(
                        GraphImageId::TransparentBackDepth,
                        ImageUsage::ColorAttachmentWrite,
                    )],
                ));
            }

            let mut overlay_accesses = vec![
                ResourceAccess::new(GraphImageId::MsaaColor, ImageUsage::ColorAttachmentWrite),
                ResourceAccess::new(GraphImageId::OverlayHdr, ImageUsage::ColorAttachmentWrite),
                ResourceAccess::new(
                    GraphImageId::MsaaDepth,
                    ImageUsage::DepthAttachmentReadWrite,
                ),
            ];
            if plan.has_transparent_meshes {
                overlay_accesses.push(ResourceAccess::new(
                    GraphImageId::SceneColor,
                    ImageUsage::FragmentSampledRead,
                ));
                overlay_accesses.push(ResourceAccess::new(
                    GraphImageId::TransparentBackDepth,
                    ImageUsage::FragmentSampledRead,
                ));
            }
            if plan.has_transparent_meshes || plan.has_grid_3d {
                overlay_accesses.push(ResourceAccess::new(
                    GraphImageId::SdfDepth,
                    ImageUsage::FragmentGeneralRead,
                ));
            }
            passes.push(RenderGraphPass::new(
                GraphPhase::SurfaceAndRaster,
                PassKind::ColorRaster(ColorRasterGraphPass {
                    target: GraphImageId::OverlayHdr,
                    use_msaa: true,
                    preserve_multisample: false,
                    color_load: ColorLoad::Clear([0.0; 4]),
                    depth: Some((
                        if plan.has_transparent_meshes || plan.has_grid_3d {
                            DepthLoad::Load
                        } else {
                            DepthLoad::Discard
                        },
                        false,
                    )),
                    region: RasterRegion::new(raster_extent, None, 1.0),
                    meshes_3d: plan
                        .has_transparent_meshes
                        .then_some(Mesh3DPass::TransparentColor),
                    grids_3d: plan.has_grid_3d,
                    meshes_2d: Some(Mesh2DPass::Depth),
                }),
                overlay_accesses,
            ));
        }

        if plan.execution.runs_raster() && !plan.uses_deferred_raster {
            let raster_extent = plan.raster_extent();
            let overlay_destination = plan.execution == FrameExecutionPlan::SdfRasterComposite;
            let target = if overlay_destination {
                GraphImageId::OverlayHdr
            } else {
                GraphImageId::ResolvedColor
            };

            let mut primary_accesses = vec![
                ResourceAccess::new(GraphImageId::MsaaColor, ImageUsage::ColorAttachmentWrite),
                ResourceAccess::new(target, ImageUsage::ColorAttachmentWrite),
            ];
            if plan.raster_uses_depth {
                primary_accesses.push(ResourceAccess::new(
                    GraphImageId::MsaaDepth,
                    ImageUsage::DepthAttachmentReadWrite,
                ));
            }
            if plan.has_grid_3d {
                primary_accesses.push(ResourceAccess::new(
                    GraphImageId::SdfDepth,
                    ImageUsage::FragmentGeneralRead,
                ));
            }

            passes.push(RenderGraphPass::new(
                GraphPhase::SurfaceAndRaster,
                PassKind::ColorRaster(ColorRasterGraphPass {
                    target,
                    use_msaa: true,
                    preserve_multisample: true,
                    color_load: ColorLoad::Clear(plan.background_color),
                    depth: plan
                        .raster_uses_depth
                        .then_some((DepthLoad::Clear, plan.has_transparent_meshes)),
                    region: RasterRegion::new(
                        raster_extent,
                        plan.camera_clip,
                        plan.camera_raster_scale,
                    ),
                    meshes_3d: None,
                    grids_3d: plan.has_grid_3d,
                    meshes_2d: (!plan.has_transparent_meshes).then_some(if plan.analytic_2d {
                        Mesh2DPass::Analytic
                    } else if plan.raster_uses_depth {
                        Mesh2DPass::Depth
                    } else {
                        Mesh2DPass::Depthless
                    }),
                }),
                primary_accesses,
            ));

            if plan.has_transparent_meshes {
                let copy_src = if plan.execution.runs_sdf() {
                    GraphImageId::SurfaceHdr
                } else {
                    target
                };
                passes.push(RenderGraphPass::new(
                    GraphPhase::SurfaceAndRaster,
                    PassKind::CopyColorImage {
                        src: copy_src,
                        dst: GraphImageId::SceneColor,
                        extent: raster_extent,
                    },
                    vec![
                        ResourceAccess::new(copy_src, ImageUsage::TransferRead),
                        ResourceAccess::new(GraphImageId::SceneColor, ImageUsage::TransferWrite),
                    ],
                ));
                passes.push(RenderGraphPass::new(
                    GraphPhase::SurfaceAndRaster,
                    PassKind::TransparentBackDepth {
                        extent: raster_extent,
                    },
                    vec![ResourceAccess::new(
                        GraphImageId::TransparentBackDepth,
                        ImageUsage::ColorAttachmentWrite,
                    )],
                ));

                let mut transparent_accesses = vec![
                    ResourceAccess::new(GraphImageId::MsaaColor, ImageUsage::ColorAttachmentWrite),
                    ResourceAccess::new(target, ImageUsage::ColorAttachmentWrite),
                    ResourceAccess::new(GraphImageId::SceneColor, ImageUsage::FragmentSampledRead),
                    ResourceAccess::new(
                        GraphImageId::TransparentBackDepth,
                        ImageUsage::FragmentSampledRead,
                    ),
                    ResourceAccess::new(GraphImageId::SdfDepth, ImageUsage::FragmentGeneralRead),
                ];
                if plan.raster_uses_depth {
                    transparent_accesses.push(ResourceAccess::new(
                        GraphImageId::MsaaDepth,
                        ImageUsage::DepthAttachmentReadWrite,
                    ));
                }
                passes.push(RenderGraphPass::new(
                    GraphPhase::SurfaceAndRaster,
                    PassKind::ColorRaster(ColorRasterGraphPass {
                        target,
                        use_msaa: true,
                        preserve_multisample: false,
                        color_load: ColorLoad::Load,
                        depth: plan.raster_uses_depth.then_some((DepthLoad::Load, false)),
                        region: RasterRegion::new(
                            raster_extent,
                            plan.camera_clip,
                            plan.camera_raster_scale,
                        ),
                        meshes_3d: Some(Mesh3DPass::TransparentColor),
                        grids_3d: false,
                        meshes_2d: Some(Mesh2DPass::Depth),
                    }),
                    transparent_accesses,
                ));
            }
        }

        if plan.execution.runs_sdf() || plan.uses_deferred_raster {
            let mut composite_accesses = vec![ResourceAccess::new(
                GraphImageId::SurfaceHdr,
                ImageUsage::ComputeSampledRead,
            )];
            if plan.has_surface_overlay {
                composite_accesses.push(ResourceAccess::new(
                    GraphImageId::OverlayHdr,
                    ImageUsage::ComputeSampledRead,
                ));
            }
            composite_accesses.push(ResourceAccess::new(
                GraphImageId::ResolvedColor,
                ImageUsage::ComputeStorageWrite,
            ));
            passes.push(RenderGraphPass::new(
                GraphPhase::SurfaceAndRaster,
                PassKind::SurfaceComposite {
                    has_overlay: plan.has_surface_overlay,
                    width: plan.width,
                    height: plan.height,
                    raster_scale: plan.ssaa_factor,
                },
                composite_accesses,
            ));
        }

        // Phase 3: Postprocess (Bloom + ToneMap)
        if plan.execution != FrameExecutionPlan::Empty {
            if plan.bloom_enabled {
                passes.push(RenderGraphPass::new(
                    GraphPhase::Postprocess,
                    PassKind::BloomExtract,
                    vec![
                        ResourceAccess::new(
                            GraphImageId::ResolvedColor,
                            ImageUsage::ComputeSampledRead,
                        ),
                        ResourceAccess::new(
                            GraphImageId::BloomPing,
                            ImageUsage::ComputeStorageWrite,
                        ),
                    ],
                ));
                passes.push(RenderGraphPass::new(
                    GraphPhase::Postprocess,
                    PassKind::BloomHorizontal,
                    vec![
                        ResourceAccess::new(
                            GraphImageId::BloomPing,
                            ImageUsage::ComputeSampledRead,
                        ),
                        ResourceAccess::new(
                            GraphImageId::BloomPong,
                            ImageUsage::ComputeStorageWrite,
                        ),
                    ],
                ));
                passes.push(RenderGraphPass::new(
                    GraphPhase::Postprocess,
                    PassKind::BloomVertical,
                    vec![
                        ResourceAccess::new(
                            GraphImageId::BloomPong,
                            ImageUsage::ComputeSampledRead,
                        ),
                        ResourceAccess::new(
                            GraphImageId::BloomPing,
                            ImageUsage::ComputeStorageWrite,
                        ),
                    ],
                ));
            } else if bloom_needs_clear {
                passes.push(RenderGraphPass::new(
                    GraphPhase::Postprocess,
                    PassKind::BloomClear,
                    vec![ResourceAccess::new(
                        GraphImageId::BloomPing,
                        ImageUsage::ClearWrite,
                    )],
                ));
            }
        }

        if plan.runs_postprocess {
            passes.push(RenderGraphPass::new(
                GraphPhase::Postprocess,
                PassKind::ToneMap {
                    width: plan.width,
                    height: plan.height,
                },
                vec![
                    ResourceAccess::new(
                        GraphImageId::ResolvedColor,
                        ImageUsage::ComputeSampledRead,
                    ),
                    ResourceAccess::new(GraphImageId::BloomPing, ImageUsage::ComputeSampledRead),
                    ResourceAccess::new(GraphImageId::OutputRgba, ImageUsage::ComputeStorageWrite),
                ],
            ));
        }

        // Phase 4: Outputs
        if outputs.cpu_nv12 {
            passes.push(RenderGraphPass::new(
                GraphPhase::Output,
                PassKind::OutputCpuNv12 {
                    width: plan.width,
                    height: plan.height,
                },
                vec![ResourceAccess::new(
                    GraphImageId::OutputRgba,
                    ImageUsage::ComputeSampledRead,
                )],
            ));
        }

        if outputs.cpu_yuv444p {
            passes.push(RenderGraphPass::new(
                GraphPhase::Output,
                PassKind::OutputCpuYuv444p {
                    width: plan.width,
                    height: plan.height,
                },
                vec![ResourceAccess::new(
                    GraphImageId::OutputRgba,
                    ImageUsage::ComputeSampledRead,
                )],
            ));
        }

        if outputs.vulkan_video {
            let input_image = if plan.fused_video_downsample {
                GraphImageId::ResolvedColor
            } else {
                GraphImageId::OutputRgba
            };
            passes.push(RenderGraphPass::new(
                GraphPhase::Output,
                PassKind::OutputVulkanVideo {
                    width: plan.width,
                    height: plan.height,
                    fused_downsample: plan.fused_video_downsample,
                },
                vec![
                    ResourceAccess::new(input_image, ImageUsage::ComputeSampledRead),
                    ResourceAccess::new(GraphImageId::VideoNv12, ImageUsage::ComputeStorageWrite),
                ],
            ));
        }

        if outputs.cpu_rgba {
            passes.push(RenderGraphPass::new(
                GraphPhase::Output,
                PassKind::CopyRgbaToBuffer {
                    width: plan.width,
                    height: plan.height,
                    padded_bytes_per_row,
                },
                vec![ResourceAccess::new(
                    GraphImageId::OutputRgba,
                    ImageUsage::TransferRead,
                )],
            ));
        }

        // Phase 5: Present fallback when no CPU/video output consumed OutputRgba
        if !outputs.cpu_nv12 && !outputs.cpu_yuv444p && !outputs.vulkan_video && !outputs.cpu_rgba {
            passes.push(RenderGraphPass::new(
                GraphPhase::Present,
                PassKind::PresentReady,
                vec![ResourceAccess::new(
                    GraphImageId::OutputRgba,
                    ImageUsage::FragmentSampledRead,
                )],
            ));
        }

        Self { passes }
    }

    pub(in crate::vulkan::renderer) fn passes(&self) -> &[RenderGraphPass] {
        &self.passes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::renderer::record::RecordingPlanInput;

    fn base_input() -> RecordingPlanInput {
        RecordingPlanInput {
            width: 320,
            height: 180,
            ssaa_factor: 2,
            analytic_aa_2d: true,
            bloom_enabled: false,
            gpu_profiling: false,
            has_sdf: false,
            has_mesh_indices: false,
            has_prepared_mesh_2d: false,
            all_2d_analytic: true,
            has_opaque_meshes: false,
            has_transparent_meshes: false,
            grid_3d_draw_calls: 0,
            outputs: RenderOutputs::CPU_RGBA_ONLY,
            background_color: [0.0; 4],
            camera_clip: None,
            camera_raster_scale: 2.0,
        }
    }

    #[test]
    fn empty_plan_compiles_clear_and_readback_without_postprocess() {
        let plan = RecordingPlan::new(base_input());
        let graph = RenderGraph::compile(plan, RenderOutputs::CPU_RGBA_ONLY, 1280, true);
        let kinds: Vec<_> = graph.passes().iter().map(|pass| pass.kind).collect();
        assert_eq!(
            kinds,
            vec![
                PassKind::ClearEmpty {
                    background_color: [0.0; 4]
                },
                PassKind::CopyRgbaToBuffer {
                    width: 320,
                    height: 180,
                    padded_bytes_per_row: 1280,
                },
            ]
        );
    }

    #[test]
    fn multiple_compute_outputs_elide_redundant_read_after_read_barriers() {
        let input = RecordingPlanInput {
            has_prepared_mesh_2d: true,
            outputs: RenderOutputs::ALL,
            ..base_input()
        };
        let plan = RecordingPlan::new(input);
        let graph = RenderGraph::compile(plan, RenderOutputs::ALL, 1280, false);

        let mut table = ImageStateTable::default();
        // Pre-populate BloomPing as already cleared and in ComputeSampledRead.
        let _ = table.plan_pass_barriers(
            &[ResourceAccess::new(
                GraphImageId::BloomPing,
                ImageUsage::ComputeSampledRead,
            )],
            |_| true,
        );

        let mut output_barriers = Vec::new();
        for pass in graph.passes() {
            let transitions = table.plan_pass_barriers(&pass.accesses, |_| true);
            if pass.phase == GraphPhase::Output {
                output_barriers.push((pass.kind, transitions));
            }
        }

        // OutputCpuNv12 transitions OutputRgba from ComputeStorageWrite -> ComputeSampledRead.
        assert_eq!(output_barriers[0].1.len(), 1);
        assert_eq!(output_barriers[0].1[0].image, GraphImageId::OutputRgba);

        // OutputCpuYuv444p reads OutputRgba in the exact same ComputeSampledRead state: 0 barriers!
        assert!(output_barriers[1].1.is_empty());

        // OutputVulkanVideo reads OutputRgba (elided) and writes VideoNv12 (1 barrier).
        assert_eq!(output_barriers[2].1.len(), 1);
        assert_eq!(output_barriers[2].1[0].image, GraphImageId::VideoNv12);

        // CopyRgbaToBuffer transitions OutputRgba to TransferRead.
        assert_eq!(output_barriers[3].1.len(), 1);
        assert_eq!(output_barriers[3].1[0].image, GraphImageId::OutputRgba);
    }

    #[test]
    fn surface_resolve_automatically_transitions_inactive_inputs_once() {
        let input = RecordingPlanInput {
            has_sdf: true,
            ..base_input()
        };
        let plan = RecordingPlan::new(input);
        let graph = RenderGraph::compile(plan, RenderOutputs::NONE, 1280, false);

        let mut table = ImageStateTable::default();
        // Frame 1: inactive RasterNormalDepth/Albedo/MaterialId are transitioned from UNDEFINED -> ComputeSampledRead.
        let mut frame1_resolve_transitions = Vec::new();
        for pass in graph.passes() {
            let transitions = table.plan_pass_barriers(&pass.accesses, |_| true);
            if matches!(pass.kind, PassKind::SurfaceResolve { .. }) {
                frame1_resolve_transitions = transitions;
            }
        }
        assert!(
            frame1_resolve_transitions
                .iter()
                .any(|t| t.image == GraphImageId::RasterNormalDepth)
        );

        // Frame 2: inactive Raster* inputs are already in ComputeSampledRead, so SurfaceResolve elides them!
        let mut frame2_resolve_transitions = Vec::new();
        for pass in graph.passes() {
            let transitions = table.plan_pass_barriers(&pass.accesses, |_| true);
            if matches!(pass.kind, PassKind::SurfaceResolve { .. }) {
                frame2_resolve_transitions = transitions;
            }
        }
        assert!(
            !frame2_resolve_transitions
                .iter()
                .any(|t| t.image == GraphImageId::RasterNormalDepth)
        );
    }
}
