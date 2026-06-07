use std::sync::{Arc, OnceLock};

#[derive(Clone)]
pub struct WgpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

static GLOBAL_WGPU_CONTEXT: OnceLock<WgpuContext> = OnceLock::new();

impl WgpuContext {
    pub async fn new() -> Option<Self> {
        if let Some(ctx) = GLOBAL_WGPU_CONTEXT.get() {
            return Some(ctx.clone());
        }

        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        let ctx = Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        };

        let _ = GLOBAL_WGPU_CONTEXT.set(ctx.clone());
        Some(ctx)
    }

    pub fn from_existing(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }
}
