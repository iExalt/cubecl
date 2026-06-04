mod dummy;

use crate::dummy::{DummyDevice, DummyElementwiseAddition, test_client};

use cubecl_runtime::local_tuner;
use cubecl_runtime::server::{CubeCount, Handle, KernelArguments};
use cubecl_runtime::tune::{AutotuneOutput, CloneInputGenerator, LocalTuner, Tunable, TunableSet};
use dummy::*;
#[cfg(feature = "autotune-checks")]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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
    let handles = vec![lhs, rhs, out];
    let checks = Arc::new(AtomicUsize::new(0));

    let test_set: Arc<TunableSet<String, Vec<Handle>, CheckCountOutput>> = TUNER.init({
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
                |_inputs: &Vec<Handle>| "autotune-checks-once".to_string(),
                CloneInputGenerator,
            )
            .with(Tunable::new("add", {
                let checks = checks.clone();
                move |inputs| {
                    add.run(inputs)?;
                    Ok::<_, String>(CheckCountOutput {
                        checks: checks.clone(),
                    })
                }
            }))
            .with(Tunable::new("add_slow_wrong", {
                let checks = checks.clone();
                move |inputs| {
                    add_slow_wrong.run(inputs)?;
                    Ok::<_, String>(CheckCountOutput {
                        checks: checks.clone(),
                    })
                }
            }))
        }
    });

    let id = "test".to_string();
    let _ = TUNER.execute(&id, &client, test_set.clone(), handles.clone());
    assert_eq!(1, checks.load(Ordering::SeqCst));

    let _ = TUNER.execute(&id, &client, test_set.clone(), handles.clone());
    assert_eq!(1, checks.load(Ordering::SeqCst));

    TUNER.clear();
    let _ = TUNER.execute(&id, &client, test_set, handles);
    assert_eq!(2, checks.load(Ordering::SeqCst));
}
