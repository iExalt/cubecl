mod base;

pub use base::*;

use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle, ServiceId};
use core::any::Any;

/// Opaque identity for one device-runner generation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DeviceGenerationId(u64);

impl DeviceGenerationId {
    #[cfg(feature = "std")]
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Process-wide device-generation counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceGenerationMetrics {
    created_generations: u64,
    closed_generations: u64,
}

impl DeviceGenerationMetrics {
    /// Returns the number of generations created.
    pub fn created_generations(&self) -> u64 {
        self.created_generations
    }
    /// Returns the number of generations closed.
    pub fn closed_generations(&self) -> u64 {
        self.closed_generations
    }
    /// Returns the number of active generations.
    pub fn active_generations(&self) -> u64 {
        self.created_generations
            .saturating_sub(self.closed_generations)
    }
    /// Returns the non-negative delta from an earlier snapshot.
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

/// Returns process-wide device-generation counters.
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

/// Outcome of releasing a device-runner generation lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLeaseRelease {
    /// The release closed the generation and joined its runner.
    Closed,
    /// The release initiated closing from the runner thread; a delegated join remains active.
    Closing,
    /// Other external owners still retain the generation.
    Shared,
    /// The lease did not retain a channel-backed generation.
    Stateless,
}

/// Error returned when a released lease closes a panicked runner.
#[derive(Debug)]
pub struct DeviceLeaseReleaseError {
    generation_id: DeviceGenerationId,
}

impl DeviceLeaseReleaseError {
    #[cfg(feature = "std")]
    pub(crate) fn closed_with_runner_panic(generation_id: DeviceGenerationId) -> Self {
        Self { generation_id }
    }
    /// Returns the affected generation.
    pub fn generation_id(&self) -> DeviceGenerationId {
        self.generation_id
    }
}

impl core::fmt::Display for DeviceLeaseReleaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
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
    pub(in crate::device::handle) fn channel(lease: channel::ExternalLease) -> Self {
        Self {
            channel: Some(lease),
        }
    }
    /// Returns the retained generation, if stateful.
    pub fn generation_id(&self) -> Option<DeviceGenerationId> {
        #[cfg(feature = "std")]
        return self
            .channel
            .as_ref()
            .map(channel::ExternalLease::generation_id);
        #[cfg(not(feature = "std"))]
        None
    }
    /// Releases the lease, closing its generation when it is the final owner.
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
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceLease")
            .field("generation_id", &self.generation_id())
            .finish()
    }
}

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

#[cfg(all(feature = "std", multi_threading))]
type Inner = channel::ChannelDeviceHandle;
// type Inner = mutex::MutexDeviceHandle;
#[cfg(all(feature = "std", not(multi_threading)))]
type Inner = reentrant::ReentrantMutexDeviceHandle;
#[cfg(all(not(feature = "std"), not(multi_threading)))]
type Inner = mutex::MutexDeviceHandle;

/// A handle to one service, reached as `S`.
///
/// `S` is the concrete service for a handle built with [`insert`](Self::insert)
/// or [`new`](Self::new), and a trait object for one built with
/// [`seen_as`](Self::seen_as). Either way the service lives where the
/// transport `I` put it; the handle only knows how to see it as `S`.
pub struct DeviceHandle<S: ?Sized, I: DeviceHandleSpec = Inner> {
    handle: I,
    service: ServiceId,
    cast: fn(&mut dyn Any) -> &mut S,
}

impl<S: ?Sized, I: DeviceHandleSpec> Clone for DeviceHandle<S, I> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            service: self.service,
            cast: self.cast,
        }
    }
}

/// The state the transport holds is `S`, or the registry is broken.
fn downcast<S: 'static>(state: &mut dyn Any) -> &mut S {
    state
        .downcast_mut::<S>()
        .expect("State type mismatch in the device registry")
}

#[allow(missing_docs)]
impl<S: DeviceService, I: DeviceHandleSpec> DeviceHandle<S, I> {
    pub fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: I::insert::<S>(device_id, service)?,
            service: ServiceId::of::<S>(device_id),
            cast: downcast::<S>,
        })
    }

    pub fn new(device_id: DeviceId) -> Self {
        Self {
            handle: I::new::<S>(device_id),
            service: ServiceId::of::<S>(device_id),
            cast: downcast::<S>,
        }
    }
}

#[allow(missing_docs)]
impl<S: ?Sized + 'static, I: DeviceHandleSpec> DeviceHandle<S, I> {
    pub const fn is_blocking() -> bool {
        I::BLOCKING
    }

    /// The same service, seen as `T`: `cast` turns the state this handle
    /// already reaches into a `T`, once per task, on the thread that runs it.
    pub fn seen_as<T: ?Sized>(self, cast: fn(&mut dyn Any) -> &mut T) -> DeviceHandle<T, I> {
        DeviceHandle {
            handle: self.handle,
            service: self.service,
            cast,
        }
    }

    pub fn device_id(&self) -> DeviceId {
        self.handle.device_id()
    }

    /// The service this handle reaches: its device and its concrete type,
    /// whatever it is seen as.
    pub fn service_id(&self) -> ServiceId {
        self.service
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        self.handle.utilities()
    }

    /// Returns a lease retaining the current device-runner generation.
    pub fn lease(&self) -> DeviceLease {
        self.handle.lease()
    }

    pub fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let cast = self.cast;
        self.handle.submit_blocking(move |state| task(cast(state)))
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let cast = self.cast;
        self.handle.submit(move |state| task(cast(state)))
    }

    /// Tries to enqueue a task without panicking when the runner is closed.
    pub fn try_submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) -> Result<(), CallError> {
        let cast = self.cast;
        self.handle.try_submit(move |state| task(cast(state)))
    }

    pub fn flush_queue(&self) {
        self.handle.flush_queue();
    }

    pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        self.handle.exclusive(task)
    }

    /// Force-closes all background runner generations for `device_id`. Queued tasks run before the
    /// threads stop. This invalidates live handles and values retained from those generations;
    /// callers must stop using them before shutdown. Channel-backed handles wait up to 30 seconds
    /// for staged draining, then return while a coordinator continues to hold the shutdown gate
    /// and generation pins. A runner thread cannot call this method synchronously.
    ///
    /// Only meaningful for handle implementations with background threads; a
    /// no-op otherwise.
    ///
    /// # Scope
    ///
    /// **This is device-wide, and `S` is ignored.** It shuts down every
    /// [`DeviceService`] registered on `device_id`, across both service stages, not
    /// only `S`. The type parameter selects the handle implementation to dispatch
    /// through, nothing more.
    ///
    /// [`DeviceId`] is not unique across runtimes either: `type_id` is assigned per
    /// runtime, so distinct backends can hand out the same id. Shutting down a
    /// device from one runtime can therefore tear down another runtime's services
    /// on the colliding id, and block while that runtime's handles are still live.
    pub fn shutdown(device_id: DeviceId) {
        I::shutdown(device_id)
    }
}

/// Shuts a device's runner down when dropped. Use [`DeviceFixture`] rather than
/// this directly: the guard alone still requires getting the device id and the
/// declaration order right.
#[cfg(test)]
struct ShutdownGuard {
    device_id: DeviceId,
    shutdown: fn(DeviceId),
}

#[cfg(test)]
impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        (self.shutdown)(self.device_id);
    }
}

/// Hands out a device id no other test is using.
///
/// The channel implementation keys global registries by device id, and the whole
/// crate's tests share one binary, so a hardcoded id collides with whatever test
/// happens to run alongside: one test would shut down another's live runner.
#[cfg(test)]
fn next_test_device_id() -> DeviceId {
    use core::sync::atomic::{AtomicU16, Ordering};

    static NEXT: AtomicU16 = AtomicU16::new(0);

    DeviceId {
        type_id: 0,
        index_id: NEXT.fetch_add(1, Ordering::Relaxed),
    }
}

/// A handle on a device of its own, whose runner is shut down when the fixture drops.
///
/// This is the only correct way to spell the pattern, so it is the only one tests
/// should use. The three hazards are all handled structurally: the device id comes
/// from [`next_test_device_id`] so it cannot collide, the guard is created with it
/// so it cannot be forgotten, and `handle` is declared before `_guard` so it drops
/// first, meaning the shutdown never waits on a handle the fixture itself owns.
///
/// Handles a test creates on top of this one (a second service on the same device,
/// clones) must be locals declared *after* the fixture, which drop in reverse order
/// and so are gone before the shutdown runs.
#[cfg(test)]
pub(crate) struct DeviceFixture<H> {
    handle: H,
    _guard: ShutdownGuard,
    device_id: DeviceId,
}

#[cfg(test)]
impl<H> DeviceFixture<H> {
    pub(crate) fn new(build: fn(DeviceId) -> H, shutdown: fn(DeviceId)) -> Self {
        let device_id = next_test_device_id();

        Self {
            handle: build(device_id),
            _guard: ShutdownGuard {
                device_id,
                shutdown,
            },
            device_id,
        }
    }

    pub(crate) fn device_id(&self) -> DeviceId {
        self.device_id
    }
}

#[cfg(test)]
impl<H> core::ops::Deref for DeviceFixture<H> {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        &self.handle
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

#[cfg(test)]
mod tests_channel {
    type DeviceHandle<S> = super::DeviceHandle<S, channel::ChannelDeviceHandle>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod tests_mutex {
    type DeviceHandle<S> = super::DeviceHandle<S, mutex::MutexDeviceHandle>;

    include!("./tests.rs");
}

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = super::DeviceHandle<S, reentrant::ReentrantMutexDeviceHandle>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}
