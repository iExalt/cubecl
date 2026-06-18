#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub use device::*;
pub use runtime::*;

#[cfg(feature = "ptx-wmma")]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::PtxWmmaCompiler;

#[cfg(not(feature = "ptx-wmma"))]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::CudaWmmaCompiler;

pub mod install {
    use std::path::PathBuf;

    pub fn include_path() -> PathBuf {
        let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
        path.push("include");
        path
    }

    pub fn cccl_include_path() -> PathBuf {
        let mut path = include_path();
        path.push("cccl");
        path
    }

    pub fn cuda_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("CUDA_PATH") {
            return Some(PathBuf::from(path));
        }

        #[cfg(target_os = "linux")]
        {
            // If it is installed as part of the distribution
            return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/usr/local/cuda"))
            } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/opt/cuda"))
            } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
                // Maybe the compiler was installed within the user path.
                Some(PathBuf::from("/usr"))
            } else {
                None
            };
        }

        #[cfg(target_os = "windows")]
        {
            return Some(PathBuf::from(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
            ));
        }

        #[allow(unreachable_code)]
        None
    }
}

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    use std::{
        process::Command,
        thread::sleep,
        time::{Duration, Instant},
    };

    use cubecl_common::config::RuntimeConfig;
    use cubecl_core::{
        prelude::{BufferArg, CubeCount, CubeDim},
        runtime_tests::launch::{
            ComptimeTagLaunch, kernel_with_comptime_tag, test_kernel_without_generics,
        },
    };
    use cubecl_runtime::{
        cache_metrics::cache_metrics, config::CubeClRuntimeConfig, lifecycle::RuntimeGuard,
    };

    pub type TestRuntime = crate::CudaRuntime;

    pub use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_core::testgen_launch_dynamic_count!();
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f16);

    #[test]
    fn test_runtime_guard_shutdown() {
        let guard = RuntimeGuard::acquire().unwrap();
        {
            let _client = <TestRuntime as cubecl_core::Runtime>::client(&Default::default());
        }
        guard.shutdown().unwrap();
    }

    #[test]
    fn test_unguarded_cuda_child_process() {
        const CHILD_ENV: &str = "CUBECL_UNGUARDED_CUDA_CHILD";
        const RUNS_ENV: &str = "CUBECL_UNGUARDED_CUDA_RUNS";
        const CHILD_TIMEOUT: Duration = Duration::from_secs(180);

        if std::env::var_os(CHILD_ENV).is_none() {
            let runs = std::env::var(RUNS_ENV)
                .map(|value| {
                    value
                        .parse::<usize>()
                        .expect("invalid CUDA lifecycle run count")
                })
                .unwrap_or(1);

            for run in 0..runs {
                let mut child = Command::new(std::env::current_exe().unwrap())
                    .arg("--exact")
                    .arg("tests::test_unguarded_cuda_child_process")
                    .arg("--test-threads=1")
                    .arg("--nocapture")
                    .env(CHILD_ENV, "1")
                    .spawn()
                    .unwrap();
                let started = Instant::now();

                let status = loop {
                    if let Some(status) = child.try_wait().unwrap() {
                        break status;
                    }
                    if started.elapsed() >= CHILD_TIMEOUT {
                        child.kill().unwrap();
                        let _ = child.wait();
                        panic!("unguarded CUDA child process {run} timed out");
                    }
                    sleep(Duration::from_millis(10));
                };

                assert!(
                    status.success(),
                    "unguarded CUDA child process {run} failed with {status}"
                );
            }
            return;
        }

        let mut config = CubeClRuntimeConfig::default();
        config.compilation.cache = None;
        config.compilation.cache_namespace = None;
        CubeClRuntimeConfig::set(config);

        let before = cache_metrics();
        let client = TestRuntime::client(&Default::default());
        test_kernel_without_generics::<TestRuntime>(client.clone());

        let handle = client.create_from_slice(bytemuck::cast_slice(&[0.0_f32, 1.0]));
        kernel_with_comptime_tag::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ComptimeTagLaunch::new(
                unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
                "lifecycle".to_string(),
            ),
        );
        drop(handle);
        drop(client);

        let compiled = cache_metrics().delta(before);

        assert_eq!(0, compiled.compilation_cache_hits);
        assert!(
            compiled.nvrtc_compilations >= 2,
            "CUDA shutdown did not drain the accepted NVRTC compilation: {compiled:?}"
        );
    }
}
