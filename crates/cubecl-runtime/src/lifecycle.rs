use core::sync::atomic::{AtomicBool, Ordering};

use cubecl_common::device_handle::{DeviceServicesShutdownError, shutdown_device_services};

static GUARD_ACQUIRED: AtomicBool = AtomicBool::new(false);

/// Error returned when a process-wide runtime guard already exists.
#[derive(Debug)]
pub struct RuntimeGuardError;

impl core::fmt::Display for RuntimeGuardError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a CubeCL runtime guard is already active")
    }
}

impl std::error::Error for RuntimeGuardError {}

/// Process-wide guard that shuts down `CubeCL` device services when dropped.
///
/// Acquire this guard before creating runtime clients. Values created after the
/// guard are dropped first when the surrounding scope exits, allowing the guard
/// to join device runners before process teardown begins.
#[must_use = "the runtime guard must remain alive while CubeCL clients are in use"]
pub struct RuntimeGuard {
    active: bool,
}

impl RuntimeGuard {
    /// Acquires the process-wide runtime guard.
    ///
    /// # Errors
    ///
    /// Returns an error if another runtime guard is active.
    pub fn acquire() -> Result<Self, RuntimeGuardError> {
        GUARD_ACQUIRED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| RuntimeGuardError)?;

        Ok(Self { active: true })
    }

    /// Shuts down all device services and releases the guard.
    ///
    /// Call this after dropping all `CubeCL` clients and backend-owned values.
    ///
    /// # Errors
    ///
    /// Returns an error if one or more device runner threads panic during shutdown.
    pub fn shutdown(mut self) -> Result<(), DeviceServicesShutdownError> {
        let result = shutdown_device_services();
        self.release();
        result
    }

    fn release(&mut self) {
        if self.active {
            self.active = false;
            GUARD_ACQUIRED.store(false, Ordering::Release);
        }
    }
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        if let Err(err) = shutdown_device_services() {
            log::warn!("CubeCL runtime guard shutdown failed: {err}");
        }
        self.release();
    }
}

/// Shuts down and joins every process-wide `CubeCL` device service.
///
/// Callers must stop creating clients and submitting work before calling this
/// function. A new runtime generation may be initialized after it returns.
///
/// # Errors
///
/// Returns an error if one or more device runner threads panic during shutdown.
pub fn shutdown() -> Result<(), DeviceServicesShutdownError> {
    shutdown_device_services()
}

#[cfg(test)]
mod tests {
    use super::RuntimeGuard;

    #[test]
    fn test_acquire_rejects_overlapping_guards_and_allows_reacquisition() {
        let guard = RuntimeGuard::acquire().unwrap();
        assert!(RuntimeGuard::acquire().is_err());
        guard.shutdown().unwrap();

        RuntimeGuard::acquire().unwrap().shutdown().unwrap();
    }
}
