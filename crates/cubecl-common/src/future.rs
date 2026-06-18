use alloc::boxed::Box;
use core::{future::Future, pin::Pin};

/// A dynamically typed, boxed, future. Useful for futures that need to ensure they
/// are not capturing any of their inputs.
pub type DynFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Spawns a future to run detached. This will use a thread on native, or the browser runtime
/// on WASM.
pub fn spawn_detached_fut(fut: impl Future<Output = ()> + Send + 'static) {
    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            wasm_bindgen_futures::spawn_local(fut);
        } else if #[cfg(feature = "std")] {
            std::thread::spawn(|| block_on(fut));
        } else {
            drop(fut); // Just to prevent unused.
            panic!("spawn_detached_fut is only supported with 'std' or on 'wasm' targets");
        }
    }
}

/// Owns a native future worker and joins it on drop.
#[derive(Debug)]
pub struct JoinOnDrop {
    #[cfg(all(not(target_family = "wasm"), feature = "std"))]
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Error returned when a future worker panics while being joined.
#[derive(Debug)]
pub struct FutureWorkerError;

impl core::fmt::Display for FutureWorkerError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("future worker panicked")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FutureWorkerError {}

impl JoinOnDrop {
    /// Joins the worker, returning an error if it panicked.
    pub fn join(&mut self) -> Result<(), FutureWorkerError> {
        #[cfg(all(not(target_family = "wasm"), feature = "std"))]
        if let Some(handle) = self.handle.take() {
            return handle.join().map_err(|_| FutureWorkerError);
        }

        Ok(())
    }
}

impl Drop for JoinOnDrop {
    fn drop(&mut self) {
        if self.join().is_err() {
            log::warn!("Future worker panicked during shutdown");
        }
    }
}

/// Spawns a future whose native worker can be joined deterministically.
pub fn spawn_joinable_fut(fut: impl Future<Output = ()> + Send + 'static) -> JoinOnDrop {
    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            wasm_bindgen_futures::spawn_local(fut);
            JoinOnDrop {}
        } else if #[cfg(feature = "std")] {
            let handle = std::thread::spawn(|| block_on(fut));
            JoinOnDrop {
                handle: Some(handle),
            }
        } else {
            drop(fut);
            panic!("spawn_joinable_fut is only supported with 'std' or on 'wasm' targets");
        }
    }
}

/// Block until the [future](Future) is completed and returns the result.
#[cfg_attr(feature = "std", allow(clippy::needless_lifetimes))]
#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(fut)))]
pub fn block_on<O>(fut: impl Future<Output = O>) -> O {
    #[cfg(target_family = "wasm")]
    {
        super::reader::read_sync(fut)
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "std")))]
    {
        embassy_futures::block_on(fut)
    }

    #[cfg(all(not(target_family = "wasm"), feature = "std"))]
    {
        futures_lite::future::block_on(fut)
    }
}

#[cfg(all(test, feature = "std", not(target_family = "wasm")))]
mod tests {
    use super::spawn_joinable_fut;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_spawn_joinable_fut_joins_worker() {
        let completed = Arc::new(AtomicBool::new(false));
        let completed_worker = Arc::clone(&completed);
        let mut worker = spawn_joinable_fut(async move {
            completed_worker.store(true, Ordering::Release);
        });

        worker.join().unwrap();

        assert!(completed.load(Ordering::Acquire));
    }

    #[test]
    fn test_spawn_joinable_fut_reports_panic() {
        let mut worker = spawn_joinable_fut(async move {
            panic!("worker panic");
        });

        assert!(worker.join().is_err());
    }
}
