#![cfg(feature = "autotune-checks")]

use super::*;
use crate::client::Client;
use crate::config::{CubeClRuntimeConfig, RuntimeConfig};
use crate::memory_management::{ManagedMemoryHandle, MemoryReport, MemoryUsage};
use crate::server::{
    BufferBinding, CopyDescriptor, CubeCount, Handle, KernelArguments, MemoryLayout,
    MemoryLayoutPolicy, ProfileError, ProfilingToken, Server, ServerError, ServerStorage,
    ServerUtilities,
};
use crate::storage::{ComputeStorage, ManagedResource, StorageHandle, StorageId};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use cubecl_common::bytes::Bytes;
use cubecl_common::device::{DeviceId, DeviceService, ServiceId};
use cubecl_common::profile::{Instant, ProfileDuration, TimingMethod};
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::Mutex;
use cubecl_ir::features::Features;
use cubecl_ir::{DeviceIdentity, DeviceProperties, HardwareProperties, TargetProperties};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ProbeKey(u32);

impl core::fmt::Display for ProbeKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "probe-key-{}", self.0)
    }
}

impl AutotuneKey for ProbeKey {}

#[derive(Clone, Debug)]
struct ProbeInput {
    value: Arc<Mutex<u32>>,
    check: bool,
}

struct ProbeInputs;

impl TuneInputs for ProbeInputs {
    type At<'a> = ProbeInput;

    fn clone_for_check<'a>(inputs: &Self::At<'a>) -> Self::At<'a> {
        ProbeInput {
            value: Arc::new(Mutex::new(*inputs.value.lock())),
            check: true,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ProbeOutput(u32);

impl AutotuneOutput for ProbeOutput {
    fn check_equivalence(&self, other: Self) {
        assert_eq!(self, &other, "candidate output differs from reference");
    }
}

fn probe_set(
    checks: Arc<[AtomicUsize; 2]>,
    regular: Arc<[AtomicUsize; 2]>,
) -> TunableSet<ProbeKey, ProbeInputs, ProbeOutput> {
    TunableSet::new(|_: &ProbeInput| ProbeKey(7), CloneInputGenerator)
        .with(Tunable::new("candidate-a", {
            let checks = checks.clone();
            let regular = regular.clone();
            move |input: ProbeInput| {
                if input.check {
                    checks[0].fetch_add(1, Ordering::Relaxed);
                } else {
                    regular[0].fetch_add(1, Ordering::Relaxed);
                }
                let value = *input.value.lock();
                if input.check {
                    *input.value.lock() += 1;
                }
                Ok::<_, &'static str>(ProbeOutput(value))
            }
        }))
        .with_reference(Tunable::new("reference", {
            let checks = checks.clone();
            let regular = regular.clone();
            move |input: ProbeInput| {
                if input.check {
                    checks[1].fetch_add(1, Ordering::Relaxed);
                } else {
                    regular[1].fetch_add(1, Ordering::Relaxed);
                }
                let value = *input.value.lock();
                if input.check {
                    *input.value.lock() += 1;
                }
                Ok::<_, &'static str>(ProbeOutput(value))
            }
        }))
}

#[test]
#[serial_test::serial]
fn checks_once_per_key_clear_and_environment_generation() {
    let _config = DisabledConfig::install_with_recording(true);
    let device_id = DeviceId {
        type_id: 0x4d30,
        index_id: 0,
    };
    let client = Client::load::<ProbeServer>(device_id);
    let checks = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let regular = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let tuner = LocalTuner::<ProbeKey, DeviceId>::new("m0-checks");
    let operations = tuner.init(&device_id, {
        let checks = checks.clone();
        let regular = regular.clone();
        move || probe_set(checks.clone(), regular.clone())
    });
    let input = ProbeInput {
        value: Arc::new(Mutex::new(9)),
        check: false,
    };

    let first = tuner.execute(&device_id, &client, operations.clone(), input.clone());
    assert_eq!(first, ProbeOutput(9));
    assert_eq!(check_counts(&checks), [1, 1]);
    assert_eq!(*input.value.lock(), 9, "checks must mutate isolated clones");
    let regular_after_first = regular_counts(&regular);
    assert!(regular_after_first.iter().sum::<usize>() > 0);

    let second = tuner.execute(&device_id, &client, operations.clone(), input.clone());
    assert_eq!(second, ProbeOutput(9));
    assert_eq!(check_counts(&checks), [1, 1]);

    tuner.clear();
    let _ = tuner.execute(&device_id, &client, operations.clone(), input.clone());
    assert_eq!(check_counts(&checks), [2, 2]);
    assert_eq!(*input.value.lock(), 9);

    let original_environment = cubecl_environment::environment::active();
    cubecl_environment::environment::activate("m0-checks-generation");
    let _ = tuner.execute(&device_id, &client, operations, input.clone());
    assert_eq!(check_counts(&checks), [3, 3]);
    assert_eq!(*input.value.lock(), 9);
    cubecl_environment::environment::activate(original_environment.as_ref());
}

#[test]
fn clone_for_check_is_candidate_isolated() {
    let input = ProbeInput {
        value: Arc::new(Mutex::new(13)),
        check: false,
    };
    let first = ProbeInputs::clone_for_check(&input);
    let second = ProbeInputs::clone_for_check(&input);
    *second.value.lock() = 99;

    assert!(first.check && second.check);
    assert_eq!(*first.value.lock(), 13);
    assert_eq!(*input.value.lock(), 13);
}

#[test]
#[serial_test::serial]
fn explicit_reference_is_the_comparison_oracle_and_legacy_falls_back() {
    let key = ProbeKey(3);
    {
        let _config = DisabledConfig::install_with_recording(true);
        let explicit = check_autotune_outputs(
            "m0-reference",
            &"device-0",
            &key,
            1,
            vec![
                (0, "candidate-a".into(), Ok(ProbeOutput(4))),
                (1, "reference".into(), Ok(ProbeOutput(7))),
                (2, "candidate-b".into(), Ok(ProbeOutput(7))),
            ],
        );
        assert_eq!(
            explicit
                .iter()
                .map(|result| (result.name.as_str(), result.passed))
                .collect::<Vec<_>>(),
            vec![
                ("candidate-a", false),
                ("candidate-b", true),
                ("reference", true)
            ]
        );
    }

    let _config = DisabledConfig::install();
    let legacy = std::panic::catch_unwind(|| {
        check_autotune_outputs(
            "m0-reference",
            &"device-0",
            &key,
            2,
            vec![
                (0, "candidate-a".into(), Ok(ProbeOutput(7))),
                (1, "candidate-b".into(), Ok(ProbeOutput(5))),
                (2, "legacy-reference".into(), Ok(ProbeOutput(7))),
            ],
        )
    })
    .expect_err("unchecked correctness mismatches must fail fast");
    let legacy_message = panic_message(legacy);
    for detail in [
        "m0-reference",
        "device-0",
        "probe-key-3",
        "candidate-b",
        "legacy-reference",
    ] {
        assert!(
            legacy_message.contains(detail),
            "mismatch diagnostic omitted `{detail}`: {legacy_message}"
        );
    }
}

#[test]
#[serial_test::serial]
fn disabled_logging_keeps_correctness_checks_without_context() {
    let _config = DisabledConfig::install();
    let mut logger = crate::config::Logger::new();
    assert!(AutotuneLogContext::new(&mut logger).is_none());

    let key = ProbeKey(8);
    let checks = check_autotune_outputs(
        "m0-context",
        &"device-0",
        &key,
        0,
        vec![
            (0, "reference".into(), Ok(ProbeOutput(2))),
            (
                1,
                "candidate-with-context-diagnostic".into(),
                Ok(ProbeOutput(2)),
            ),
        ],
    );
    assert_eq!(checks[0].name, "candidate-with-context-diagnostic");
    assert!(checks[0].passed);
}

fn check_counts(counts: &[AtomicUsize; 2]) -> [usize; 2] {
    [
        counts[0].load(Ordering::Relaxed),
        counts[1].load(Ordering::Relaxed),
    ]
}

fn regular_counts(counts: &[AtomicUsize; 2]) -> [usize; 2] {
    check_counts(counts)
}

fn panic_message(payload: Box<dyn core::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).into())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".into())
}

struct DisabledConfig {
    previous: Option<Arc<CubeClRuntimeConfig>>,
    environment: Arc<str>,
}

impl DisabledConfig {
    fn install() -> Self {
        Self::install_with_recording(false)
    }

    fn install_with_recording(recording: bool) -> Self {
        let storage = <CubeClRuntimeConfig as RuntimeConfig>::storage();
        let previous = storage.lock().clone();
        let mut config = CubeClRuntimeConfig::default();
        config.autotune.logger.level = crate::config::autotune::AutotuneLogLevel::Disabled;
        config.autotune.recorder.stdout = recording;
        config.autotune.bench.adaptive = false;
        config.autotune.bench.min_samples = 1;
        config.autotune.bench.max_samples = 1;
        *storage.lock() = Some(Arc::new(config));
        Self {
            previous,
            environment: cubecl_environment::environment::active(),
        }
    }
}

impl Drop for DisabledConfig {
    fn drop(&mut self) {
        cubecl_environment::environment::activate(self.environment.as_ref());
        let storage = <CubeClRuntimeConfig as RuntimeConfig>::storage();
        *storage.lock() = self.previous.take();
    }
}

#[derive(Default)]
struct ProbeStorage;

impl ComputeStorage for ProbeStorage {
    type Resource = ();

    fn alignment(&self) -> usize {
        1
    }
    fn get(&mut self, _handle: &StorageHandle) -> Result<Self::Resource, crate::server::IoError> {
        unimplemented!()
    }
    fn alloc(&mut self, _size: u64) -> Result<StorageHandle, crate::server::IoError> {
        unimplemented!()
    }
    fn dealloc(&mut self, _id: StorageId) {}
    fn flush(&mut self) {}
}

#[derive(Debug)]
struct ProbeServer {
    utilities: Arc<ServerUtilities>,
}

impl ProbeServer {
    fn new(service: ServiceId) -> Self {
        let properties = DeviceProperties::new(
            Features::default(),
            cubecl_ir::MemoryDeviceProperties {
                max_page_size: 1 << 30,
                alignment: 1,
            },
            HardwareProperties {
                load_width: 128,
                plane_size_min: 1,
                plane_size_max: 1,
                max_bindings: 16,
                max_shared_memory_size: 0,
                max_cube_count: (1, 1, 1),
                max_units_per_cube: 1,
                max_cube_dim: (1, 1, 1),
                num_streaming_multiprocessors: None,
                num_cpu_cores: Some(1),
                last_level_cache_size: None,
                num_tensor_cores: None,
                min_tensor_cores_dim: None,
                max_vector_size: 1,
                cube_mma_reserved_shared_memory: 0,
            },
            TimingMethod::System,
            DeviceIdentity {
                name: "m0-probe".into(),
                fingerprint: "m0-probe".into(),
            },
        );
        Self {
            utilities: Arc::new(ServerUtilities::new(
                service,
                "m0-probe",
                properties,
                TargetProperties::default(),
                Arc::new(crate::logging::ServerLogger::default()),
                ProbeLayout,
            )),
        }
    }
}

struct ProbeLayout;

impl MemoryLayoutPolicy for ProbeLayout {
    fn apply(
        &self,
        _service: ServiceId,
        _stream_id: StreamId,
        _descriptors: &[crate::server::MemoryLayoutDescriptor],
    ) -> (Handle, Vec<MemoryLayout>) {
        unimplemented!()
    }
}

impl DeviceService for ProbeServer {
    fn init(device_id: DeviceId) -> Self {
        Self::new(ServiceId::of::<Self>(device_id))
    }

    fn utilities(&self) -> Arc<dyn core::any::Any + Send + Sync> {
        self.utilities.clone()
    }
}

impl Server for ProbeServer {
    fn initialize_memory(
        &mut self,
        _memory: ManagedMemoryHandle,
        _size: u64,
        _stream_id: StreamId,
    ) {
        unimplemented!()
    }
    fn logger(&self) -> Arc<crate::logging::ServerLogger> {
        self.utilities.logger.clone()
    }
    fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }
    fn read(
        &mut self,
        _descriptors: Vec<CopyDescriptor>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        unimplemented!()
    }
    fn write(&mut self, _descriptors: Vec<(CopyDescriptor, Bytes)>, _stream_id: StreamId) {
        unimplemented!()
    }
    fn sync(
        &mut self,
        _handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        unimplemented!()
    }
    fn check(
        &mut self,
        _handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }
    unsafe fn launch(
        &mut self,
        _kernel: Box<dyn crate::kernel::CubeKernel>,
        _count: CubeCount,
        _bindings: KernelArguments,
        _stream_id: StreamId,
        _launch_mode: crate::dry_run::LaunchMode,
    ) {
        unimplemented!()
    }
    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        Ok(())
    }
    fn memory_usage(&mut self, _stream_id: StreamId) -> MemoryUsage {
        MemoryUsage::default()
    }
    fn memory_report(&mut self, _stream_id: StreamId) -> MemoryReport {
        unimplemented!()
    }
    fn memory_cleanup(&mut self, _stream_id: StreamId) {}
    fn start_profile(&mut self, _stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        Ok(ProfilingToken { id: 0 })
    }
    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        _token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        Ok(ProfileDuration::new_system_time(
            Instant::now(),
            Instant::now(),
        ))
    }
    fn allocation_mode(
        &mut self,
        _mode: crate::memory_management::MemoryAllocationMode,
        _stream_id: StreamId,
    ) {
    }
}

impl crate::server::ServerCommunication for ProbeServer {}

impl ServerStorage for ProbeServer {
    type Storage = ProbeStorage;

    fn get_resource(
        &mut self,
        _binding: BufferBinding,
        _stream_id: StreamId,
    ) -> Result<ManagedResource<()>, ServerError> {
        unimplemented!()
    }
}
