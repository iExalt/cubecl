use core::sync::atomic::{AtomicBool, Ordering};

use cubecl_common::device_handle::{
    DeviceGenerationId, DeviceLease, DeviceLeaseRelease, DeviceServicesShutdownError,
    shutdown_device_services,
};
pub use cubecl_common::device_handle::{DeviceGenerationMetrics, device_generation_metrics};
use std::{collections::HashSet, vec::Vec};

use crate::{client::ComputeClient, runtime::Runtime};

static GUARD_ACQUIRED: AtomicBool = AtomicBool::new(false);

/// Summary of generation leases released by a [`RuntimeSession`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeSessionShutdown {
    closed_generations: usize,
    closing_generations: usize,
    shared_generations: usize,
    stateless_leases: usize,
}

impl RuntimeSessionShutdown {
    /// Returns the number of generations closed by the session's final lease release.
    pub fn closed_generations(&self) -> usize {
        self.closed_generations
    }

    /// Returns the number of generations whose on-runner join was delegated.
    pub fn closing_generations(&self) -> usize {
        self.closing_generations
    }

    /// Returns the number of generations that remain owned by other external values.
    pub fn shared_generations(&self) -> usize {
        self.shared_generations
    }

    /// Returns the number of released leases without a device-runner generation.
    pub fn stateless_leases(&self) -> usize {
        self.stateless_leases
    }
}

/// Error returned after a runtime session closes one or more panicked runners.
#[derive(Debug)]
pub struct RuntimeSessionShutdownError {
    report: RuntimeSessionShutdown,
    failed_generations: Vec<DeviceGenerationId>,
}

impl RuntimeSessionShutdownError {
    /// Returns the complete release report, including generations that closed with a panic.
    pub fn report(&self) -> RuntimeSessionShutdown {
        self.report
    }

    /// Returns the generations that reached `Closed` after their runner panicked.
    pub fn failed_generations(&self) -> &[DeviceGenerationId] {
        &self.failed_generations
    }
}

impl core::fmt::Display for RuntimeSessionShutdownError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "{} runtime generation(s) closed after a runner panic",
            self.failed_generations.len()
        )
    }
}

impl std::error::Error for RuntimeSessionShutdownError {}

/// Optional scoped owner for selected `CubeCL` runtime generations.
///
/// A session covers only generations explicitly pinned through [`Self::client`] or [`Self::pin`].
/// Later clients for the same runtime and device reuse the pinned generation and its caches. Other
/// runtimes and devices are unaffected. Intermediate lease counts are intentionally opaque.
///
/// Dropping a session or calling [`Self::shutdown`] releases every pin. A final release may block
/// while accepted work, compilation, and device synchronization complete. If other external values
/// remain, the generation stays valid and their final drop performs graceful shutdown.
#[must_use = "the runtime session must remain alive while its generations should stay pinned"]
#[derive(Default)]
pub struct RuntimeSession {
    leases: Vec<DeviceLease>,
    generations: HashSet<DeviceGenerationId>,
}

impl RuntimeSession {
    /// Creates an empty runtime session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a runtime client and pins its generation in this session.
    pub fn client<R: Runtime>(&mut self, device: &R::Device) -> ComputeClient<R> {
        let client = R::client(device);
        self.pin(&client);
        client
    }

    /// Pins an existing client's generation.
    ///
    /// Returns `true` when this call adds a new generation and `false` when that generation was
    /// already covered or the client has no stateful device-runner generation.
    pub fn pin<R: Runtime>(&mut self, client: &ComputeClient<R>) -> bool {
        let Some(generation_id) = client.generation_id() else {
            return false;
        };
        if !self.generations.insert(generation_id) {
            return false;
        }

        self.leases.push(client.clone_lease());
        true
    }

    /// Returns the number of distinct device-runner generations pinned by this session.
    pub fn num_pinned_generations(&self) -> usize {
        self.leases.len()
    }

    /// Releases every pinned generation lease.
    ///
    /// [`RuntimeSessionShutdown::shared_generations`] reports generations still owned by external
    /// values. Those values remain valid and own eventual graceful shutdown.
    ///
    /// # Errors
    ///
    /// Returns an aggregate error after all leases are released if one or more final releases close
    /// a runner that panicked. Such generations are included in the report's closed count.
    pub fn shutdown(mut self) -> Result<RuntimeSessionShutdown, RuntimeSessionShutdownError> {
        self.release_pins()
    }

    fn release_pins(&mut self) -> Result<RuntimeSessionShutdown, RuntimeSessionShutdownError> {
        let leases = core::mem::take(&mut self.leases);
        self.generations.clear();

        let mut report = RuntimeSessionShutdown::default();
        let mut failed_generations = Vec::new();
        for lease in leases {
            match lease.release() {
                Ok(DeviceLeaseRelease::Closed) => report.closed_generations += 1,
                Ok(DeviceLeaseRelease::Closing) => report.closing_generations += 1,
                Ok(DeviceLeaseRelease::Shared) => report.shared_generations += 1,
                Ok(DeviceLeaseRelease::Stateless) => report.stateless_leases += 1,
                Err(error) => {
                    report.closed_generations += 1;
                    failed_generations.push(error.generation_id());
                }
            }
        }

        if failed_generations.is_empty() {
            Ok(report)
        } else {
            Err(RuntimeSessionShutdownError {
                report,
                failed_generations,
            })
        }
    }
}

impl Drop for RuntimeSession {
    fn drop(&mut self) {
        if let Err(error) = self.release_pins() {
            log::warn!("CubeCL runtime session shutdown failed: {error}");
        }
    }
}

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
