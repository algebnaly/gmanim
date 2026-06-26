use ash::{vk, Entry, Instance, Device};
use std::sync::{Arc, OnceLock, Mutex};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

#[derive(Clone)]
pub struct VulkanContext {
    pub instance: Arc<Instance>,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub allocator: Arc<Mutex<Allocator>>,
}

static GLOBAL_VULKAN_CONTEXT: OnceLock<VulkanContext> = OnceLock::new();

impl VulkanContext {
    pub async fn new() -> Option<Self> {
        if let Some(ctx) = GLOBAL_VULKAN_CONTEXT.get() {
            return Some(ctx.clone());
        }

        let entry = unsafe { Entry::load().ok()? };

        let app_name = std::ffi::CString::new("gmanim").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None).ok()? };

        let pdevices = unsafe { instance.enumerate_physical_devices().ok()? };
        let physical_device = pdevices.into_iter().next()?;

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let (queue_family_index, _) = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
            .map(|(i, p)| (i as u32, p))?;

        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: priorities.len() as u32,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None).ok()? };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).ok()?;

        let ctx = Self {
            instance: Arc::new(instance),
            physical_device,
            device: Arc::new(device),
            queue,
            queue_family_index,
            allocator: Arc::new(Mutex::new(allocator)),
        };

        let _ = GLOBAL_VULKAN_CONTEXT.set(ctx.clone());
        Some(ctx)
    }
}
