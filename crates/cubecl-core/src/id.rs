use cubecl_common::device::{Device, DeviceId};
use cubecl_ir::HardwareProperties;
use cubecl_runtime::client::Client;

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
    pub fn new(client: &Client, device: &impl Device) -> Self {
        Self {
            device: device.to_id(),
            name: client.name(),
            version: env!("CARGO_PKG_VERSION"),
            hardware: client.properties().hardware.clone(),
        }
    }
}

impl core::fmt::Display for CubeTuneId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "device-{}-{}-{}-v{}-lw{}-p{}-{}-bind{}-smem{}-cc{}x{}x{}-units{}-cd{}x{}x{}-sms{}-cpu{}-llc{}-tc{}-tcd{}-vec{}-mma{}",
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
            self.hardware.last_level_cache_size.unwrap_or_default(),
            self.hardware.num_tensor_cores.unwrap_or_default(),
            self.hardware.min_tensor_cores_dim.unwrap_or_default(),
            self.hardware.max_vector_size,
            self.hardware.cube_mma_reserved_shared_memory,
        )
    }
}
