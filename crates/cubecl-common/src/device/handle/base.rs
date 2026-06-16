use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle};

/// An error happened while executing a call.
#[derive(Debug)]
pub struct CallError;

/// Error returned when one or more device runners fail during shutdown.
#[derive(Debug)]
pub struct DeviceServicesShutdownError {
    runner_panics: usize,
}

impl DeviceServicesShutdownError {
    /// Creates a shutdown error for the number of runner threads that panicked.
    #[allow(dead_code)]
    pub(crate) fn new(runner_panics: usize) -> Self {
        Self { runner_panics }
    }

    /// Returns the number of runner threads that panicked.
    pub fn runner_panics(&self) -> usize {
        self.runner_panics
    }
}

impl core::fmt::Display for DeviceServicesShutdownError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "{} device runner thread(s) failed during shutdown",
            self.runner_panics
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DeviceServicesShutdownError {}

#[derive(new, Clone, Debug)]
/// Error when creating a [`DeviceService`].
pub struct ServiceCreationError {
    #[allow(dead_code)] // Debug uses it.
    reason: alloc::string::String,
}

pub(crate) trait DeviceHandleSpec<S: DeviceService>: Sized {
    /// If functions block the current thread even if they are non-blocking.
    const BLOCKING: bool;

    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError>;

    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn new(device_id: DeviceId) -> Self;

    /// Retrieves the device ID for this handle.
    fn device_id(&self) -> DeviceId;

    /// Retrieves the server utilities for this thread.
    fn utilities(&self) -> ServerUtilitiesHandle;

    /// Doesn't flush the service state, but flushes any task enqueued in the communication
    /// channel.
    ///
    /// # Notes
    ///
    /// This is often not necessary, except for distributed operations.
    fn flush_queue(&self);

    /// Executes a task on the dedicated device thread and returns the result of the task.
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::submit`] if you don't need to wait for a returned type.
    fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError>;

    /// Submit a task for execution on the dedicated device thread.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T);

    /// TODO: Docs.
    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError>;
}
