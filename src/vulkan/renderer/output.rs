#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderOutputs {
    pub cpu_nv12: bool,
    pub vulkan_video: bool,
    pub cpu_rgba: bool,
    pub cpu_yuv444p: bool,
}

impl RenderOutputs {
    pub const ALL: Self = Self {
        cpu_nv12: true,
        vulkan_video: true,
        cpu_rgba: true,
        cpu_yuv444p: true,
    };

    pub const VULKAN_VIDEO_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: true,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };

    pub const CPU_NV12_ONLY: Self = Self {
        cpu_nv12: true,
        vulkan_video: false,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };

    pub const CPU_RGBA_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: false,
        cpu_rgba: true,
        cpu_yuv444p: false,
    };

    pub const CPU_READBACKS: Self = Self {
        cpu_nv12: true,
        vulkan_video: false,
        cpu_rgba: true,
        cpu_yuv444p: true,
    };

    pub const NONE: Self = Self {
        cpu_nv12: false,
        vulkan_video: false,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };
}
