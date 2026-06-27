use ash::{vk, Device, Entry, Instance};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex, OnceLock};

const VK_QUEUE_VIDEO_ENCODE_BIT_KHR_RAW: u32 = 0x0000_0040;
const OPTIONAL_VIDEO_DEVICE_EXTENSIONS: [&str; 3] = [
    "VK_KHR_video_queue",
    "VK_KHR_video_encode_queue",
    "VK_KHR_video_encode_h264",
];

#[derive(Clone)]
pub struct VulkanContext {
    pub entry: Arc<Entry>,
    pub instance: Arc<Instance>,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub video_encode_queue: Option<vk::Queue>,
    pub video_encode_queue_family_index: Option<u32>,
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

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let (queue_family_index, _) = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| {
                props
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .map(|(i, p)| (i as u32, p))?;
        let video_encode_queue_family_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.as_raw() & VK_QUEUE_VIDEO_ENCODE_BIT_KHR_RAW != 0)
            .map(|(i, _)| i as u32);

        let priorities = [1.0];
        let graphics_queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: priorities.len() as u32,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        };
        let video_queue_info = video_encode_queue_family_index
            .filter(|idx| *idx != queue_family_index)
            .map(|idx| vk::DeviceQueueCreateInfo {
                queue_family_index: idx,
                queue_count: priorities.len() as u32,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            });
        let mut queue_infos = vec![graphics_queue_info];
        if let Some(info) = video_queue_info {
            queue_infos.push(info);
        }

        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .ok()?
        };
        let enabled_extension_names: Vec<CString> = OPTIONAL_VIDEO_DEVICE_EXTENSIONS
            .iter()
            .filter(|extension_name| {
                available_extensions.iter().any(|extension| unsafe {
                    CStr::from_ptr(extension.extension_name.as_ptr()).to_bytes()
                        == extension_name.as_bytes()
                })
            })
            .map(|extension_name| CString::new(*extension_name).unwrap())
            .collect();
        let enabled_extension_ptrs: Vec<*const i8> = enabled_extension_names
            .iter()
            .map(|extension_name| extension_name.as_ptr())
            .collect();

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            enabled_extension_count: enabled_extension_ptrs.len() as u32,
            pp_enabled_extension_names: enabled_extension_ptrs.as_ptr(),
            ..Default::default()
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .ok()?
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let video_encode_queue =
            video_encode_queue_family_index.map(|idx| unsafe { device.get_device_queue(idx, 0) });

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .ok()?;

        let ctx = Self {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            physical_device,
            device: Arc::new(device),
            queue,
            queue_family_index,
            video_encode_queue,
            video_encode_queue_family_index,
            allocator: Arc::new(Mutex::new(allocator)),
        };

        let _ = GLOBAL_VULKAN_CONTEXT.set(ctx.clone());
        Some(ctx)
    }
}
