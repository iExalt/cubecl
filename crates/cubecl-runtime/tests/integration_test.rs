mod dummy;

use crate::dummy::{DummyDevice, DummyElementwiseAddition, test_client};

use cubecl_common::{
    device::{Device, DeviceId},
    future::block_on,
    stream_id::StreamId,
};
use cubecl_runtime::client::ComputeClient;
#[cfg(feature = "std")]
use cubecl_runtime::lifecycle::RuntimeSession;
use cubecl_runtime::local_tuner;
use cubecl_runtime::server::{CubeCount, Handle, KernelArguments};
use cubecl_runtime::tune::LocalTuner;
#[cfg(feature = "autotune-checks")]
use cubecl_runtime::tune::TuneInputs;
#[cfg(feature = "autotune-checks")]
use cubecl_runtime::tune::{AutotuneOutput, CloneInputGenerator, Tunable, TunableSet};
use dummy::*;
#[cfg(feature = "autotune-checks")]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug, Default)]
struct IndexedDummyDevice(u16);

impl Device for IndexedDummyDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self(device_id.index_id)
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.0,
        }
    }
}

#[test]
fn resource_generation_is_retained_and_validated() {
    let client_a = ComputeClient::<DummyRuntime>::load(&IndexedDummyDevice(101));
    let client_b = ComputeClient::<DummyRuntime>::load(&IndexedDummyDevice(102));
    let generation_a = client_a.generation_id().unwrap();
    let generation_b = client_b.generation_id().unwrap();
    assert_ne!(generation_a, generation_b);

    let handle = client_a.empty(4);
    let binding = handle.clone().binding();
    let layout = client_a.empty_tensor([4].into(), 1);
    let resource = client_a.get_resource(handle.clone()).unwrap();
    assert_eq!(handle.generation_id(), Some(generation_a));
    assert_eq!(binding.generation_id(), Some(generation_a));
    assert_eq!(layout.memory.generation_id(), Some(generation_a));
    assert_eq!(resource.generation_id(), Some(generation_a));

    let same_generation = client_a.clone();
    assert!(same_generation.read_one(handle.clone()).is_ok());

    let read_error = block_on(client_b.read_async(vec![handle.clone()])).unwrap_err();
    assert!(format!("{read_error}").contains("doesn't match client generation"));
    assert!(client_b.get_resource(handle.clone()).is_err());

    let launch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client_b.launch(
            Box::new(KernelTask::new(DummyElementwiseAddition)),
            CubeCount::Static(1, 1, 1),
            KernelArguments::new().with_buffer(handle.binding()),
        );
    }));
    assert!(launch.is_err());

    let unbound = Handle::new(StreamId::current(), 4);
    assert_eq!(unbound.generation_id(), None);
}

#[test]
#[cfg(feature = "std")]
fn runtime_session_deduplicates_and_reports_shared_generation() {
    let mut session = RuntimeSession::new();
    let client = session.client::<DummyRuntime>(&DummyDevice);

    assert_eq!(session.num_pinned_generations(), 1);
    assert!(!session.pin(&client));
    assert_eq!(session.num_pinned_generations(), 1);

    let report = session.shutdown().unwrap();
    assert_eq!(report.closed_generations(), 0);
    assert_eq!(report.closing_generations(), 0);
    assert_eq!(report.shared_generations(), 1);
    assert_eq!(report.stateless_leases(), 0);

    drop(client);
}

#[test]
#[cfg(feature = "std")]
fn runtime_session_final_pin_closes_generation() {
    let mut session = RuntimeSession::new();
    let client = ComputeClient::<DummyRuntime>::load(&IndexedDummyDevice(103));
    assert!(session.pin(&client));
    drop(client);

    let report = session.shutdown().unwrap();
    assert_eq!(report.closed_generations(), 1);
    assert_eq!(report.closing_generations(), 0);
    assert_eq!(report.shared_generations(), 0);
}

#[cfg(feature = "autotune-checks")]
#[derive(Clone)]
struct CheckCountOutput {
    checks: Arc<AtomicUsize>,
}

#[cfg(feature = "autotune-checks")]
impl AutotuneOutput for CheckCountOutput {
    fn check_equivalence(&self, _other: Self) {
        self.checks.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(feature = "autotune-checks")]
#[derive(Clone)]
struct CheckInput {
    handles: Vec<Handle>,
    clones: Arc<AtomicUsize>,
}

#[cfg(feature = "autotune-checks")]
struct CheckTuneInputs;

#[cfg(feature = "autotune-checks")]
impl TuneInputs for CheckTuneInputs {
    type At<'a> = CheckInput;

    fn clone_for_check<'a>(inputs: &Self::At<'a>) -> Self::At<'a> {
        inputs.clones.fetch_add(1, Ordering::SeqCst);
        inputs.clone()
    }
}

#[test_log::test]
fn created_resource_is_the_same_when_read() {
    let client = test_client(&DummyDevice);
    let resource = Vec::from([0, 1, 2]);
    let resource_description = client.create_from_slice(&resource);

    let obtained_resource = client.read_one(resource_description).unwrap().to_vec();

    assert_eq!(resource, obtained_resource)
}

#[test_log::test]
fn empty_allocates_memory() {
    let client = test_client(&DummyDevice);
    let size = 4;
    let resource_description = client.empty(size);
    let empty_resource = client.read_one(resource_description).unwrap();

    assert_eq!(empty_resource.len(), 4);
}

#[test_log::test]
fn execute_elementwise_addition() {
    let client = test_client(&DummyDevice);
    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);

    client.launch(
        Box::new(KernelTask::new(DummyElementwiseAddition)),
        CubeCount::Static(1, 1, 1),
        KernelArguments::new().with_buffers(vec![
            lhs.binding(),
            rhs.binding(),
            out.clone().binding(),
        ]),
    );

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    assert_eq!(obtained_resource, Vec::from([4, 5, 6]))
}

#[test_log::test]
#[cfg(feature = "std")]
fn autotune_basic_addition_execution() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_basic_addition_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([4, 5, 6]));
}

#[test_log::test]
#[cfg(feature = "std")]
fn autotune_basic_multiplication_execution() {
    static TUNER: LocalTuner<String, String> =
        local_tuner!("autotune_basic_multiplication_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::multiplication_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([0, 4, 8]));
}

#[test_log::test]
#[cfg(all(feature = "std", feature = "autotune-checks"))]
fn autotune_checks_once_per_key() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_checks_once_per_key");

    let client = test_client(&DummyDevice);
    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let clones = Arc::new(AtomicUsize::new(0));
    let inputs = CheckInput {
        handles: vec![lhs, rhs, out],
        clones: clones.clone(),
    };
    let checks = Arc::new(AtomicUsize::new(0));

    let test_set: Arc<TunableSet<String, CheckTuneInputs, CheckCountOutput>> = TUNER.init({
        let checks = checks.clone();
        move || {
            let client = test_client(&DummyDevice);
            let add = OneKernelAutotuneOperation::new(
                KernelTask::new(DummyElementwiseAddition),
                client.clone(),
            );
            let add_slow_wrong = OneKernelAutotuneOperation::new(
                KernelTask::new(DummyElementwiseAdditionSlowWrong),
                client,
            );

            TunableSet::new(
                |_inputs: &CheckInput| "autotune-checks-once".to_string(),
                CloneInputGenerator,
            )
            .with(Tunable::new("add", {
                let checks = checks.clone();
                move |inputs: CheckInput| {
                    add.run(inputs.handles)?;
                    Ok::<_, String>(CheckCountOutput {
                        checks: checks.clone(),
                    })
                }
            }))
            .with(Tunable::new("add_slow_wrong", {
                let checks = checks.clone();
                move |inputs: CheckInput| {
                    add_slow_wrong.run(inputs.handles)?;
                    Ok::<_, String>(CheckCountOutput {
                        checks: checks.clone(),
                    })
                }
            }))
        }
    });

    let id = "test".to_string();
    let _ = TUNER.execute(&id, &client, test_set.clone(), inputs.clone());
    assert_eq!(1, checks.load(Ordering::SeqCst));
    assert_eq!(2, clones.load(Ordering::SeqCst));

    let _ = TUNER.execute(&id, &client, test_set.clone(), inputs.clone());
    assert_eq!(1, checks.load(Ordering::SeqCst));
    assert_eq!(2, clones.load(Ordering::SeqCst));

    TUNER.clear();
    let _ = TUNER.execute(&id, &client, test_set, inputs);
    assert_eq!(2, checks.load(Ordering::SeqCst));
    assert_eq!(4, clones.load(Ordering::SeqCst));
}
