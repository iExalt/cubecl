//! Native Metal shutdown tests.
//!
//! The final client release must drain encoded work before the Metal server is dropped. The
//! test keeps only a retained shared buffer, so a framework handle cannot make the assertion pass
//! by retaining the device generation.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;
use objc2_metal::MTLBuffer;
use std::process::Command;

const CHILD_MARKER: &str = "CUBECL_METAL_SHUTDOWN_CHILD";
type R = crate::MetalRuntime;

#[cube(launch_unchecked)]
fn write_answer(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = 42;
    }
}

/// Run the assertion in a child so no other test can retain or shut down the process-wide Metal
/// generation while this test drops its final client.
#[test]
fn final_owner_shutdown_drains_pending_metal_work() {
    if std::env::var_os(CHILD_MARKER).is_none() {
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests_shutdown::final_owner_shutdown_drains_pending_metal_work",
                "--nocapture",
            ])
            .env(CHILD_MARKER, "1")
            .env("RUST_TEST_THREADS", "1")
            .status()
            .expect("failed to start isolated native Metal shutdown test");
        assert!(
            status.success(),
            "isolated native Metal test failed: {status}"
        );
        return;
    }

    let client = R::client(&Default::default());
    // Persistent allocations own a padded buffer with utilization offset zero. Keeping the
    // output initialized makes get_resource available before the launch and gives the test a
    // one-u32 shared buffer whose first four bytes are the kernel's output.
    let output =
        client.memory_persistent_allocation((), |_| client.create_from_slice(u32::as_bytes(&[0])));
    let resource = client
        .get_resource::<crate::compute::MetalServer>(output.clone())
        .expect("initialized output must expose its Metal resource");
    let buffer = resource.resource().clone();
    drop(resource);

    unsafe {
        write_answer::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }

    // Drop every framework owner before releasing the retained MTLBuffer. The final DeviceLease
    // release therefore invokes MetalServer::shutdown, whose drain must flush and wait for the
    // active encoder before the generation closes.
    drop(output);
    drop(client);

    // SAFETY: MetalStorage uses StorageModeShared, the cloned Retained buffer keeps the backing
    // allocation alive after the framework handles are gone, and the one-element allocation is
    // padded with offset zero as established above. The server shutdown has completed before
    // this read, so no GPU/CPU race remains.
    assert!(buffer.inner().length() >= core::mem::size_of::<u32>());
    let value = unsafe { *(buffer.inner().contents().as_ptr() as *const u32) };
    assert_eq!(
        value, 42,
        "final client release must drain pending Metal work"
    );
}
