use cubecl_common::backtrace::BackTrace;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};

use super::storage::gpu::GpuResource;
use crate::{CudaCompiler, compute::stream::Stream};
use crate::{
    CudaComputeKernel,
    install::{cccl_include_path, include_path},
};
use cubecl_core::{
    compilation_cache::CompilationCache,
    hash::StableHash,
    server::ResourceLimitError,
    {ir::DeviceProperties, prelude::*},
};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cudarc::driver::DriverError;
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::str::FromStr;
use std::sync::{Arc, Once};
use std::time::Instant;
use std::{ffi::CStr, os::raw::c_void};

use cubecl_common::cache::CacheOption;

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    ptx_cache: Option<CompilationCache<PtxCacheKey, PtxCacheEntry>>,
    ptx_cache_fingerprint: String,
    pub timestamps: TimestampProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
}

#[derive(Debug)]
pub struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

struct NvrtcProgram(cudarc::nvrtc::sys::nvrtcProgram);

impl Drop for NvrtcProgram {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a live NVRTC program created by `create_program`
        // and this guard owns the only destruction call.
        if let Err(err) = unsafe { cudarc::nvrtc::result::destroy_program(self.0) } {
            log::warn!("Unable to destroy NVRTC program: {err}");
        }
    }
}

fn initialize_nvrtc_shutdown() {
    static INITIALIZE: Once = Once::new();
    INITIALIZE.call_once(|| {
        // Compile once during CUDA server initialization so NVRTC's nested
        // builtins library is loaded before the process-exit shutdown hook.
        let source =
            CString::new("extern \"C\" __global__ void cubecl_shutdown_init() {}").unwrap();
        // SAFETY: Calling NVRTC FFI with a null-terminated source string that
        // outlives the program. The guard destroys the program after registration.
        unsafe {
            let program = NvrtcProgram(
                cudarc::nvrtc::result::create_program(source.as_c_str(), None)
                    .expect("NVRTC shutdown initialization should create a program"),
            );
            cudarc::nvrtc::result::compile_program(program.0, &[] as &[&str])
                .expect("NVRTC shutdown initialization should compile");
            cubecl_common::device_handle::register_device_services_shutdown_hook();
        }
    });
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone, Hash)]
struct PtxCacheKey {
    fingerprint: String,
    kernel: StableHash,
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
        device: cudarc::driver::sys::CUdevice,
    ) -> Self {
        initialize_nvrtc_shutdown();
        let ptx_cache_fingerprint = ptx_cache_fingerprint(&compilation_options, &arch, device);
        Self {
            context,
            module_names: HashMap::new(),
            ptx_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "ptx",
                        CacheOption::default()
                            .name("cuda")
                            .version(format!("{}-strict-ptx-v1", env!("CARGO_PKG_VERSION")))
                            .root(root),
                    ))
                } else {
                    None
                }
            },
            ptx_cache_fingerprint,
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
            properties,
        }
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let cache_key = if let Some(cache) = &self.ptx_cache {
            let cache_key = PtxCacheKey {
                fingerprint: self.ptx_cache_fingerprint.clone(),
                kernel: kernel_id.stable_hash(),
            };

            if let Some(entry) = cache.get(&cache_key) {
                log::trace!("Using PTX cache");
                cubecl_runtime::cache_metrics::record_compilation_cache_hit();

                self.load_ptx(
                    entry.ptx.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    kernel_id.cube_dim,
                    entry.shared_mem_bytes,
                )?;
                return Ok(());
            }
            cubecl_runtime::cache_metrics::record_compilation_cache_miss();
            Some(cache_key)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        self.validate_shared(&kernel_compiled.repr)?;

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let cube_dim = kernel_compiled.cube_dim;
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        logger.log_compilation(&kernel_compiled);

        // SAFETY: Calling NVRTC FFI to create, compile, and extract PTX from a program.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        let nvrtc_start = Instant::now();
        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let program = NvrtcProgram(
                cudarc::nvrtc::result::create_program(source.as_c_str(), None).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err}"),
                        backtrace: BackTrace::capture(),
                    }
                })?,
            );
            let compilation_result = cudarc::nvrtc::result::compile_program(program.0, &options);
            if compilation_result.is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program.0).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;

                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel
                    .compile(
                        &mut Default::default(),
                        &self.compilation_options,
                        mode,
                        kernel.address_type(),
                    )?
                    .source;
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?;
            };
            cudarc::nvrtc::result::get_ptx(program.0).map_err(|err| CompilationError::Generic {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?
        };
        cubecl_runtime::cache_metrics::record_nvrtc_compilation(nvrtc_start.elapsed());

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let result = cache.insert(
                cache_key.unwrap(),
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    shared_mem_bytes: repr.shared_memory_size(),
                    ptx: ptx.clone(),
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save the ptx {err:?}");
            } else {
                cubecl_runtime::cache_metrics::record_compilation_cache_write();
            }
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size(),
        )?;
        Ok(())
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name).unwrap();
        let module_load_start = Instant::now();
        // SAFETY: `ptx` is a valid null-terminated PTX binary from NVRTC. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };
        cubecl_runtime::cache_metrics::record_module_load(module_load_start.elapsed());

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );

        Ok(())
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
    ) -> Result<(), LaunchError> {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        bindings.extend(const_info);

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;
        };

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<CudaComputeKernel>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

fn ptx_cache_fingerprint(
    compilation_options: &CompilationOptions,
    arch: &CudaArchitecture,
    device: cudarc::driver::sys::CUdevice,
) -> String {
    use cubecl_runtime::config::RuntimeConfig;

    let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
    let namespace = config
        .compilation
        .cache_namespace
        .as_deref()
        .unwrap_or(concat!("cubecl-cuda-", env!("CARGO_PKG_VERSION")));
    let device_name =
        cudarc::driver::result::device::get_name(device).unwrap_or_else(|_| "unknown".to_string());
    let device_uuid = cudarc::driver::result::device::get_uuid(device)
        .map(|uuid| {
            uuid.bytes
                .iter()
                .map(|byte| format!("{:02x}", *byte as u8))
                .collect::<String>()
        })
        .unwrap_or_else(|_| "unknown".to_string());
    let driver_version = cuda_driver_version();
    let nvrtc_version = nvrtc_version();

    format!(
        "schema=1;namespace={namespace};cuda_header={};driver={driver_version};nvrtc={nvrtc_version};arch={arch:?};device_name={device_name};device_uuid={device_uuid};options={compilation_options:?};check_mode={:?}",
        cudarc::driver::sys::CUDA_VERSION,
        config.compilation.check_mode,
    )
}

fn cuda_driver_version() -> i32 {
    let mut version = 0;
    // SAFETY: `cuDriverGetVersion` writes one integer to the provided live pointer.
    if unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) }
        .result()
        .is_ok()
    {
        version
    } else {
        0
    }
}

fn nvrtc_version() -> String {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: `nvrtcVersion` writes two integers to the provided live pointers.
    if unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .is_ok()
    {
        format!("{major}.{minor}")
    } else {
        "unknown".to_string()
    }
}
