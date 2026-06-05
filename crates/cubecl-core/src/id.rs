use cubecl_common::device::{Device, DeviceId};
use cubecl_ir::HardwareProperties;
use cubecl_runtime::{client::ComputeClient, runtime::Runtime};

/// ID used to identify a Just-in-Time environment.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CubeTuneId {
    device: DeviceId,
    name: &'static str,
    version: &'static str,
    hardware: HardwareProperties,
}

impl CubeTuneId {
    /// Create a new ID.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, device: &R::Device) -> Self {
        Self {
            device: device.to_id(),
            name: R::name(client),
            version: env!("CARGO_PKG_VERSION"),
            hardware: client.properties().hardware.clone(),
        }
    }
}

impl core::fmt::Display for CubeTuneId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "device-{}-{}-{}-v{}-lw{}-p{}-{}-bind{}-smem{}-cc{}x{}x{}-units{}-cd{}x{}x{}-sms{}-cpu{}-tc{}-tcd{}-vec{}-mma{}",
            self.device.type_id,
            self.device.index_id,
            self.name,
            self.version,
            self.hardware.load_width,
            self.hardware.plane_size_min,
            self.hardware.plane_size_max,
            self.hardware.max_bindings,
            self.hardware.max_shared_memory_size,
            self.hardware.max_cube_count.0,
            self.hardware.max_cube_count.1,
            self.hardware.max_cube_count.2,
            self.hardware.max_units_per_cube,
            self.hardware.max_cube_dim.0,
            self.hardware.max_cube_dim.1,
            self.hardware.max_cube_dim.2,
            self.hardware
                .num_streaming_multiprocessors
                .unwrap_or_default(),
            self.hardware.num_cpu_cores.unwrap_or_default(),
            self.hardware.num_tensor_cores.unwrap_or_default(),
            self.hardware.min_tensor_cores_dim.unwrap_or_default(),
            self.hardware.max_vector_size,
            self.hardware.cube_mma_reserved_shared_memory,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::CubeTuneId;
    use alloc::string::ToString;
    use cubecl_common::device::DeviceId;
    use cubecl_ir::HardwareProperties;

    #[test]
    fn test_cube_tune_id_display() {
        let id = CubeTuneId {
            device: DeviceId::new(1, 2),
            name: "test",
            version: "3.4.5",
            hardware: HardwareProperties {
                load_width: 256,
                plane_size_min: 32,
                plane_size_max: 32,
                max_bindings: 8,
                max_shared_memory_size: 128,
                max_cube_count: (1, 2, 3),
                max_units_per_cube: 1024,
                max_cube_dim: (4, 5, 6),
                num_streaming_multiprocessors: Some(148),
                num_cpu_cores: None,
                num_tensor_cores: Some(4),
                min_tensor_cores_dim: Some(8),
                max_vector_size: 16,
                cube_mma_reserved_shared_memory: 64,
            },
        };

        assert_eq!(
            id.to_string(),
            "device-1-2-test-v3.4.5-lw256-p32-32-bind8-smem128-cc1x2x3-units1024-cd4x5x6-sms148-cpu0-tc4-tcd8-vec16-mma64",
        );
    }
}
