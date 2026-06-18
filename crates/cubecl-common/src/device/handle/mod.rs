mod base;

pub use base::*;

use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle};

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

/// Opaque identity for one device-runner generation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DeviceGenerationId(u64);

impl DeviceGenerationId {
    #[cfg(feature = "std")]
    fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Eventually consistent process-wide device-generation lifecycle counters.
///
/// Snapshots are exact at quiescent boundaries. During concurrent creation and closure, the
/// counters may reflect different instants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceGenerationMetrics {
    created_generations: u64,
    closed_generations: u64,
}

impl DeviceGenerationMetrics {
    /// Returns the number of device-runner generations created.
    pub fn created_generations(&self) -> u64 {
        self.created_generations
    }

    /// Returns the number of device-runner generations that reached `Closed`.
    pub fn closed_generations(&self) -> u64 {
        self.closed_generations
    }

    /// Returns the approximate number of generations that have not reached `Closed`.
    pub fn active_generations(&self) -> u64 {
        self.created_generations
            .saturating_sub(self.closed_generations)
    }

    /// Returns the non-negative counter delta from an earlier snapshot.
    pub fn delta(self, earlier: Self) -> Self {
        Self {
            created_generations: self
                .created_generations
                .saturating_sub(earlier.created_generations),
            closed_generations: self
                .closed_generations
                .saturating_sub(earlier.closed_generations),
        }
    }
}

/// Returns process-wide device-generation lifecycle counters.
pub fn device_generation_metrics() -> DeviceGenerationMetrics {
    #[cfg(all(feature = "std", multi_threading))]
    let (created_generations, closed_generations) = channel::generation_metrics();

    #[cfg(not(all(feature = "std", multi_threading)))]
    let (created_generations, closed_generations) = (0, 0);

    DeviceGenerationMetrics {
        created_generations,
        closed_generations,
    }
}

/// Outcome of explicitly releasing a device-runner generation lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLeaseRelease {
    /// The lease was the final external owner and the generation closed cleanly.
    Closed,
    /// The lease was released on its own runner and an off-runner join was delegated.
    Closing,
    /// Other external owners remain and will close the generation after their final release.
    Shared,
    /// The handle implementation has no device-runner generation.
    Stateless,
}

/// Error returned when an explicitly released lease closes a panicked runner.
#[derive(Debug)]
pub struct DeviceLeaseReleaseError {
    generation_id: DeviceGenerationId,
}

impl DeviceLeaseReleaseError {
    #[cfg(feature = "std")]
    fn closed_with_runner_panic(generation_id: DeviceGenerationId) -> Self {
        Self { generation_id }
    }

    /// Returns the generation that reached `Closed` after its runner panicked.
    pub fn generation_id(&self) -> DeviceGenerationId {
        self.generation_id
    }
}

impl core::fmt::Display for DeviceLeaseReleaseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "device runner generation {:?} panicked during shutdown",
            self.generation_id
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DeviceLeaseReleaseError {}

/// Keeps a device-runner generation alive while an escaping value remains usable.
pub struct DeviceLease {
    #[cfg(feature = "std")]
    channel: Option<channel::ExternalLease>,
}

impl DeviceLease {
    fn stateless() -> Self {
        Self {
            #[cfg(feature = "std")]
            channel: None,
        }
    }

    #[cfg(feature = "std")]
    fn channel(lease: channel::ExternalLease) -> Self {
        Self {
            channel: Some(lease),
        }
    }

    /// Returns the runner generation retained by this lease, when applicable.
    pub fn generation_id(&self) -> Option<DeviceGenerationId> {
        #[cfg(feature = "std")]
        return self
            .channel
            .as_ref()
            .map(channel::ExternalLease::generation_id);

        #[cfg(not(feature = "std"))]
        None
    }

    /// Releases this lease and closes its generation when it is the final external owner.
    ///
    /// A final release may block while accepted work, compilation, and backend synchronization
    /// complete. [`DeviceLeaseRelease::Shared`] leaves other external owners valid; their final
    /// release will perform shutdown.
    ///
    /// # Errors
    ///
    /// Returns an error when the final release closes a runner thread that panicked. The
    /// generation is already closed when this error is returned.
    pub fn release(self) -> Result<DeviceLeaseRelease, DeviceLeaseReleaseError> {
        #[cfg(feature = "std")]
        {
            let mut this = self;
            if let Some(lease) = this.channel.take() {
                return lease.release();
            }
        }

        #[cfg(not(feature = "std"))]
        let _ = self;

        Ok(DeviceLeaseRelease::Stateless)
    }
}

impl Clone for DeviceLease {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "std")]
            channel: self.channel.clone(),
        }
    }
}

impl core::fmt::Debug for DeviceLease {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DeviceLease")
            .field("generation_id", &self.generation_id())
            .finish()
    }
}

#[cfg(all(feature = "std", multi_threading))]
type Inner<S> = channel::ChannelDeviceHandle<S>;
// type Inner<S> = mutex::MutexDeviceHandle<S>;
#[cfg(all(feature = "std", not(multi_threading)))]
type Inner<S> = reentrant::ReentrantMutexDeviceHandle<S>;
#[cfg(all(not(feature = "std"), not(multi_threading)))]
type Inner<S> = mutex::MutexDeviceHandle<S>;

/// TODO: Docs
pub struct DeviceHandle<S: DeviceService> {
    handle: Inner<S>,
}

impl<S: DeviceService> Clone for DeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

#[allow(missing_docs)]
impl<S: DeviceService> DeviceHandle<S> {
    pub const fn is_blocking() -> bool {
        Inner::<S>::BLOCKING
    }

    pub fn insert(device_id: super::DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::insert(device_id, service)?,
        })
    }

    pub fn new(device_id: super::DeviceId) -> Self {
        Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::new(device_id),
        }
    }

    pub fn device_id(&self) -> DeviceId {
        self.handle.device_id()
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        self.handle.utilities()
    }

    /// Returns a portable lease for the current device-runner generation.
    pub fn lease(&self) -> DeviceLease {
        self.handle.lease()
    }

    pub fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        self.handle.submit_blocking(task)
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.handle.submit(task)
    }

    pub fn flush_queue(&self) {
        self.handle.flush_queue();
    }

    pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        self.handle.exclusive(task)
    }
}

/// Stops and joins every device runner.
///
/// New submissions must stop before this function is called.
pub fn shutdown_device_services() -> Result<(), DeviceServicesShutdownError> {
    #[cfg(all(feature = "std", multi_threading))]
    return channel::shutdown_device_services();

    #[cfg(not(all(feature = "std", multi_threading)))]
    Ok(())
}

#[cfg(test)]
mod tests_channel {
    type DeviceHandle<S> = channel::ChannelDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod tests_mutex {
    type DeviceHandle<S> = mutex::MutexDeviceHandle<S>;

    include!("./tests.rs");
}

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = reentrant::ReentrantMutexDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod lease_tests {
    use super::{DeviceGenerationMetrics, DeviceLease, DeviceLeaseRelease};

    #[test]
    fn test_device_generation_metrics_delta_saturates() {
        let earlier = DeviceGenerationMetrics {
            created_generations: 5,
            closed_generations: 4,
        };
        let current = DeviceGenerationMetrics {
            created_generations: 3,
            closed_generations: 6,
        };

        let delta = current.delta(earlier);

        assert_eq!(0, delta.created_generations());
        assert_eq!(2, delta.closed_generations());
        assert_eq!(0, delta.active_generations());
    }

    #[test]
    fn test_stateless_lease_release() {
        assert_eq!(
            DeviceLease::stateless().release().unwrap(),
            DeviceLeaseRelease::Stateless
        );
    }
}
