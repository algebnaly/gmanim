use ash::{Device, Entry, Instance, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::{CStr, CString};
use std::fmt;
use std::sync::{Arc, Mutex};

const VK_QUEUE_VIDEO_ENCODE_BIT_KHR_RAW: u32 = 0x0000_0040;
const OPTIONAL_VIDEO_DEVICE_EXTENSIONS: [&str; 3] = [
    "VK_KHR_video_queue",
    "VK_KHR_video_encode_queue",
    "VK_KHR_video_encode_h264",
];

pub struct VulkanContext {
    pub entry: Arc<Entry>,
    pub instance: Arc<Instance>,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub timestamp_period_ns: f32,
    pub timestamp_valid_bits: u32,
    pub video_encode_queue: Option<vk::Queue>,
    pub video_encode_queue_family_index: Option<u32>,
    pub allocator: Arc<Mutex<Allocator>>,
}

#[derive(Debug)]
pub enum VulkanContextError {
    Loader(ash::LoadingError),
    Vulkan(vk::Result),
    NoPhysicalDevice,
    NoGraphicsComputeQueue,
    TimelineSemaphoreUnsupported,
    Synchronization2Unsupported,
    DynamicRenderingUnsupported,
    TimestampQueriesUnsupported,
    Allocator(String),
}

impl fmt::Display for VulkanContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(formatter, "failed to load Vulkan: {error}"),
            Self::Vulkan(error) => write!(formatter, "Vulkan initialization failed: {error:?}"),
            Self::NoPhysicalDevice => formatter.write_str("no Vulkan physical device is available"),
            Self::NoGraphicsComputeQueue => {
                formatter.write_str("no Vulkan queue supports both graphics and compute")
            }
            Self::TimelineSemaphoreUnsupported => {
                formatter.write_str("timeline semaphores are required but unsupported")
            }
            Self::Synchronization2Unsupported => {
                formatter.write_str("Vulkan synchronization2 is required but unsupported")
            }
            Self::DynamicRenderingUnsupported => {
                formatter.write_str("Vulkan dynamic rendering is required but unsupported")
            }
            Self::TimestampQueriesUnsupported => {
                formatter.write_str("GPU timestamp queries are required but unsupported")
            }
            Self::Allocator(error) => write!(formatter, "failed to create GPU allocator: {error}"),
        }
    }
}

impl std::error::Error for VulkanContextError {}

impl From<vk::Result> for VulkanContextError {
    fn from(error: vk::Result) -> Self {
        Self::Vulkan(error)
    }
}

pub struct TimelineSemaphore {
    device: Arc<Device>,
    semaphore: vk::Semaphore,
}

impl TimelineSemaphore {
    pub fn new(context: &Arc<VulkanContext>, initial_value: u64) -> Result<Arc<Self>, vk::Result> {
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let semaphore = unsafe { context.device.create_semaphore(&create_info, None)? };
        Ok(Arc::new(Self {
            device: context.device.clone(),
            semaphore,
        }))
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

impl VulkanContext {
    pub fn new() -> Result<Arc<Self>, VulkanContextError> {
        let entry = unsafe { Entry::load().map_err(VulkanContextError::Loader)? };

        let app_name = std::ffi::CString::new("gmanim").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let pdevices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = pdevices
            .into_iter()
            .next()
            .ok_or(VulkanContextError::NoPhysicalDevice)?;

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let (queue_family_index, queue_family) = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| {
                props
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .map(|(i, p)| (i as u32, p))
            .ok_or(VulkanContextError::NoGraphicsComputeQueue)?;
        if queue_family.timestamp_valid_bits == 0 {
            return Err(VulkanContextError::TimestampQueriesUnsupported);
        }
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
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

        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let has_all_video_extensions =
            OPTIONAL_VIDEO_DEVICE_EXTENSIONS
                .iter()
                .all(|extension_name| {
                    available_extensions.iter().any(|extension| unsafe {
                        CStr::from_ptr(extension.extension_name.as_ptr()).to_bytes()
                            == extension_name.as_bytes()
                    })
                });
        let enabled_extension_names: Vec<CString> = if has_all_video_extensions {
            OPTIONAL_VIDEO_DEVICE_EXTENSIONS
                .iter()
                .map(|extension_name| CString::new(*extension_name).unwrap())
                .collect()
        } else {
            Vec::new()
        };
        let enabled_extension_ptrs: Vec<*const i8> = enabled_extension_names
            .iter()
            .map(|extension_name| extension_name.as_ptr())
            .collect();

        let mut supported_timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut supported_synchronization2_features =
            vk::PhysicalDeviceSynchronization2Features::default();
        let mut supported_dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut supported_timeline_features)
            .push_next(&mut supported_synchronization2_features)
            .push_next(&mut supported_dynamic_rendering_features);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut physical_device_features);
        }
        if supported_timeline_features.timeline_semaphore != vk::TRUE {
            return Err(VulkanContextError::TimelineSemaphoreUnsupported);
        }
        if supported_synchronization2_features.synchronization2 != vk::TRUE {
            return Err(VulkanContextError::Synchronization2Unsupported);
        }
        if supported_dynamic_rendering_features.dynamic_rendering != vk::TRUE {
            return Err(VulkanContextError::DynamicRenderingUnsupported);
        }

        let mut enabled_timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut enabled_synchronization2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut enabled_dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extension_ptrs)
            .push_next(&mut enabled_timeline_features)
            .push_next(&mut enabled_synchronization2_features)
            .push_next(&mut enabled_dynamic_rendering_features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

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
        .map_err(|error| VulkanContextError::Allocator(error.to_string()))?;

        Ok(Arc::new(Self {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            physical_device,
            device: Arc::new(device),
            queue,
            queue_family_index,
            timestamp_period_ns: physical_device_properties.limits.timestamp_period,
            timestamp_valid_bits: queue_family.timestamp_valid_bits,
            video_encode_queue,
            video_encode_queue_family_index,
            allocator: Arc::new(Mutex::new(allocator)),
        }))
    }
}
