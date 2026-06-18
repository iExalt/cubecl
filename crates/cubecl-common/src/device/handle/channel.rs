use crate::{
    device::{
        DeviceId, DeviceService, DeviceServiceStage, ServerUtilitiesHandle,
        handle::{
            CallError, DeviceHandleSpec, DeviceLease, DeviceServicesShutdownError,
            ServiceCreationError,
        },
    },
    stream_id::StreamId,
};
use hashbrown::{HashMap, HashSet};
use std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::RefCell,
    marker::PhantomData,
    ops::Deref,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Condvar, LazyLock, Mutex, Weak},
    vec::Vec,
};

use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

pub(super) use custom_channel::ExternalLease;
use custom_channel::{DeviceClient, GenerationCloser, GenerationUpgrade, WeakDeviceClient};

static CREATED_GENERATIONS: AtomicU64 = AtomicU64::new(0);
static CLOSED_GENERATIONS: AtomicU64 = AtomicU64::new(0);

pub(super) fn generation_metrics() -> (u64, u64) {
    (
        CREATED_GENERATIONS.load(AtomicOrdering::Relaxed),
        CLOSED_GENERATIONS.load(AtomicOrdering::Relaxed),
    )
}

// For debugging and benchmarking.
//
// use normal_channel::DeviceClient;

/// A handle to a specific device context.
///
/// This struct allows sending closures to be executed on a dedicated
/// thread for the specific device, ensuring thread-safe access to
/// the device's state (`S`).
///
/// The `ChannelDeviceHandle` acts as a proxy; it doesn't hold the state `S`
/// itself, but rather a communication channel to the thread where `S` lives.
pub struct ChannelDeviceHandle<S: DeviceService> {
    state: ChannelDeviceState,
    // fn(S) makes this Send+Sync regardless of S, since the handle
    // never actually holds an S — it only sends closures to the runner thread.
    _phantom: PhantomData<fn(S)>,
}

impl<S: DeviceService + 'static> DeviceHandleSpec<S> for ChannelDeviceHandle<S> {
    const BLOCKING: bool = false;

    /// Registers a new service instance for the device and returns a handle.
    ///
    /// If the service type `S` is already initialized on the device's runner thread,
    /// this will return an error.
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        let state = ChannelDeviceState::init(device_id, Some(service))?;

        Ok(Self {
            state,
            _phantom: PhantomData,
        })
    }

    /// Creates a handle for an existing device or starts a new `DeviceRunner` if one
    /// does not exist for the given `device_id`.
    fn new(device_id: DeviceId) -> Self {
        let state = ChannelDeviceState::init::<S>(device_id, None).unwrap();

        Self {
            state,
            _phantom: PhantomData,
        }
    }

    fn device_id(&self) -> DeviceId {
        self.state.client.runner_id().device
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.state.utilities()
    }

    fn lease(&self) -> DeviceLease {
        DeviceLease::channel(self.state.client.external_lease())
    }

    /// Runs `task` on the device thread, blocking until it returns.
    fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let state = self.state.service.clone();
        let current = StreamId::current();
        self.run_scoped(move || {
            state.act_on(|s| {
                let s = s
                    .downcast_mut::<S>()
                    .expect("State type mismatch in Thread Local Storage");
                current.executes(|| task(s))
            })
        })
    }

    /// Asynchronously dispatches a task to the device thread.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.submit_inner::<_, SEND_NO_FLUSH>(task)
            .expect("Can't have an error when submitting a task");
    }

    fn flush_queue(&self) {
        if !is_device_runner_thread(self.state.client.runner_id()) {
            self.state.client.flush();
        }
    }

    /// Runs `task` on the device thread while propagating the caller's
    /// `StreamId`, blocking until it returns.
    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let current = StreamId::current();
        self.run_scoped(move || current.executes(task))
    }
}

const SEND_FLUSH: bool = true;
const SEND_NO_FLUSH: bool = false;

impl<S: DeviceService + 'static> ChannelDeviceHandle<S> {
    /// Asynchronously dispatches a task to the device thread.
    fn submit_inner<T: FnOnce(&mut S) + Send + 'static, const FLUSH: bool>(
        &self,
        task: T,
    ) -> Result<(), CallError> {
        let state = self.state.service.clone();

        let current = StreamId::current();

        let func_init = move || {
            state.act_on(|state| {
                let state = state
                    .downcast_mut::<S>()
                    .expect("State type mismatch in Thread Local Storage");

                current.executes(|| task(state));
            });
        };

        self.send::<_, FLUSH>(func_init)
    }

    /// Dispatches a `FnOnce() -> R` on the device thread and blocks on
    /// its result. `task` may borrow from the caller's stack.
    fn run_scoped<'a, R: Send, T: FnOnce() -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        /// Builds a `'static` shim that consumes `*slot` on the device
        /// thread. The caller has to keep `*slot` alive until the shim has run,
        /// `run_scoped` does this by blocking on `recv.recv()`.
        fn create_shim<W: FnOnce() + Send>(slot: &mut Option<W>) -> impl FnOnce() + Send + 'static {
            // `*mut ()` so the shim is `'static`.
            struct Ptr(*mut ());
            // SAFETY: pointee is `Send` by the bound on `W`; uniqueness of
            // access is upheld by the deref below.
            unsafe impl Send for Ptr {}

            let ptr = Ptr(slot as *mut _ as *mut ());
            move || {
                let _ = &ptr; // capture whole ptr so the closure is Send.
                // SAFETY:
                // - Caller keeps `*slot` alive through the shim's run.
                // - `unwrap_unchecked`: the shim is `FnOnce` run at most
                //   once, so `*slot` is always `Some` on entry.
                // `Option::take` flips `*slot` to `None`, keeping drop
                // correct if the shim ran, panicked, or was never enqueued.
                let f = unsafe { (*(ptr.0 as *mut Option<W>)).take().unwrap_unchecked() };
                f()
            }
        }

        let (sender, recv) = oneshot::channel();
        // Create a slot on the stack that will hold our pointer.
        let mut slot = Some(move || sender.send(task()).unwrap());
        // Send the erased shim to the device thread.
        self.send::<_, SEND_FLUSH>(create_shim(&mut slot))?;
        recv.recv().map_err(|_| CallError)
    }

    /// Dispatches a task to the runner.
    ///
    /// If the current thread is already the runner for this device, it executes
    /// immediately to prevent deadlocks and allow for recursive calls.
    fn send<T: FnOnce() + Send + 'static, const FLUSH: bool>(
        &self,
        task: T,
    ) -> Result<(), CallError> {
        if is_device_runner_thread(self.state.client.runner_id()) {
            if let Err(err) = catch_unwind(AssertUnwindSafe(task)) {
                log::warn!("Task failed: {err:?}");
                return Err(CallError);
            }
        } else {
            self.state.client.enqueue(task)?;

            // We use const boolean to avoid branching in a hot loop.
            if FLUSH {
                self.state.client.flush();
            }
        };

        Ok(())
    }
}

/// Helper to verify if the current execution context is the device's runner thread.
fn is_device_runner_thread(runner_key: &RunnerId) -> bool {
    SERVER_THREAD.with_borrow(|state| state.as_ref() == Some(runner_key))
}

std::thread_local! {
    /// The ID of the device this thread is responsible for.
    static SERVER_THREAD: RefCell<Option<RunnerId>> = const { RefCell::new(None) };

    /// Heterogeneous map of service states owned by this thread.
    #[allow(clippy::type_complexity)]
    static STATES: RefCell<HashMap<TypeId, ServiceState>> = RefCell::new(HashMap::new());
}

/// Internal runner logic to manage background thread spawning.
struct DeviceRunner {}

/// A simple wrapper over a client and a service that is cached with [`CHANNELS`].
#[derive(Clone)]
struct ChannelDeviceState {
    inner: Arc<ChannelDeviceStateInner>,
}

struct ChannelDeviceStateInner {
    client: DeviceClient,
    service: ChannelService,
}

impl Deref for ChannelDeviceState {
    type Target = ChannelDeviceStateInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Cached reference to a device service's state.
///
/// Holds only the `TypeId` for looking up the state in thread-local STATES.
/// This avoids sharing the state across threads.
#[derive(Clone)]
struct ChannelService {
    type_id: TypeId,
    utilities: ServerUtilitiesHandle,
}

struct ServiceState {
    service: RefCell<Box<dyn Any + 'static>>,
    utilities: ServerUtilitiesHandle,
    shutdown: fn(&mut Box<dyn Any + 'static>),
}

impl ServiceState {
    fn new<S: DeviceService>(service: S) -> Self {
        fn shutdown<S: DeviceService>(service: &mut Box<dyn Any + 'static>) {
            service
                .downcast_mut::<S>()
                .expect("State type mismatch in Thread Local Storage")
                .shutdown();
        }

        let utilities = service.utilities();
        Self {
            service: RefCell::new(Box::new(service)),
            utilities,
            shutdown: shutdown::<S>,
        }
    }

    fn shutdown(&mut self) {
        (self.shutdown)(self.service.get_mut());
    }
}

fn shutdown_service_states(states: &mut HashMap<TypeId, ServiceState>) {
    for state in states.values_mut() {
        state.shutdown();
    }
    states.clear();
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct RunnerId {
    device: DeviceId,
    stage: DeviceServiceStage,
}

enum RunnerRegistryEntry {
    Strong(DeviceClient),
    Weak(WeakDeviceClient),
}

impl RunnerRegistryEntry {
    fn new(client: &DeviceClient) -> Self {
        if strong_device_registries() {
            Self::Strong(client.clone())
        } else {
            Self::Weak(client.downgrade())
        }
    }

    fn upgrade(&self) -> GenerationUpgrade {
        match self {
            Self::Strong(client) => client.try_clone_generation(),
            Self::Weak(client) => client.upgrade(),
        }
    }

    fn closer(&self) -> Option<GenerationCloser> {
        match self {
            Self::Strong(client) => Some(client.closer()),
            Self::Weak(client) => client.closer(),
        }
    }
}

enum ChannelRegistryEntry {
    Strong(ChannelDeviceState),
    Weak(Weak<ChannelDeviceStateInner>),
}

impl ChannelRegistryEntry {
    fn new(state: &ChannelDeviceState) -> Self {
        if strong_device_registries() {
            Self::Strong(state.clone())
        } else {
            Self::Weak(Arc::downgrade(&state.inner))
        }
    }

    fn upgrade(&self) -> Option<ChannelDeviceState> {
        match self {
            Self::Strong(state) => Some(state.clone()),
            Self::Weak(state) => state.upgrade().map(|inner| ChannelDeviceState { inner }),
        }
    }
}

static STRONG_DEVICE_REGISTRIES: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("CUBECL_STRONG_DEVICE_REGISTRIES").is_some());

fn strong_device_registries() -> bool {
    *STRONG_DEVICE_REGISTRIES
}

static RUNNERS: spin::Mutex<Option<HashMap<RunnerId, RunnerRegistryEntry>>> =
    spin::Mutex::new(None);
/// Device/service map. The lock is held across the entire `init` sequence so `S::init` runs
/// once per `(DeviceId, TypeId)` pair. This serializes channel creation across all
/// backends.
static CHANNELS: spin::Mutex<Option<HashMap<(RunnerId, TypeId), ChannelRegistryEntry>>> =
    spin::Mutex::new(None);

#[derive(Default)]
struct LifecycleState {
    active_initializers: usize,
    shutting_down: bool,
}

static LIFECYCLE: LazyLock<(Mutex<LifecycleState>, Condvar)> =
    LazyLock::new(|| (Mutex::new(LifecycleState::default()), Condvar::new()));

struct InitializationPermit;

impl InitializationPermit {
    fn acquire(target_stage: DeviceServiceStage) -> Self {
        let caller_stage = SERVER_THREAD.with_borrow(|runner| runner.map(|id| id.stage));
        let (mutex, condition) = &*LIFECYCLE;
        let mut lifecycle = mutex.lock().unwrap();
        while lifecycle.shutting_down {
            match caller_stage {
                Some(caller_stage) if (caller_stage as u8) < (target_stage as u8) => break,
                Some(caller_stage) => {
                    drop(lifecycle);
                    panic!(
                        "device runner at stage {caller_stage:?} cannot initialize stage \
                         {target_stage:?} during shutdown"
                    );
                }
                None => lifecycle = condition.wait(lifecycle).unwrap(),
            }
        }
        lifecycle.active_initializers += 1;
        Self
    }
}

impl Drop for InitializationPermit {
    fn drop(&mut self) {
        let (mutex, condition) = &*LIFECYCLE;
        let mut lifecycle = mutex.lock().unwrap();
        lifecycle.active_initializers -= 1;
        condition.notify_all();
    }
}

struct ShutdownPermit;

impl ShutdownPermit {
    fn acquire() -> Self {
        let (mutex, condition) = &*LIFECYCLE;
        let mut lifecycle = mutex.lock().unwrap();
        while lifecycle.shutting_down {
            lifecycle = condition.wait(lifecycle).unwrap();
        }
        lifecycle.shutting_down = true;
        while lifecycle.active_initializers != 0 {
            lifecycle = condition.wait(lifecycle).unwrap();
        }
        Self
    }

    fn wait_for_initializers(&self) {
        let (mutex, condition) = &*LIFECYCLE;
        let mut lifecycle = mutex.lock().unwrap();
        while lifecycle.active_initializers != 0 {
            lifecycle = condition.wait(lifecycle).unwrap();
        }
    }
}

impl Drop for ShutdownPermit {
    fn drop(&mut self) {
        let (mutex, condition) = &*LIFECYCLE;
        let mut lifecycle = mutex.lock().unwrap();
        lifecycle.shutting_down = false;
        condition.notify_all();
    }
}

/// Stops and joins every device runner.
///
/// Runners are shut down upstream-first so accepted producer work is drained
/// before downstream services are closed.
pub fn shutdown_device_services() -> Result<(), DeviceServicesShutdownError> {
    let permit = ShutdownPermit::acquire();
    let mut runner_panics = 0;
    let downstream_pins = {
        let guard = RUNNERS.lock();
        guard
            .as_ref()
            .into_iter()
            .flat_map(HashMap::iter)
            .filter_map(|(runner_id, entry)| {
                if runner_id.stage != DeviceServiceStage::Downstream {
                    return None;
                }
                match entry.upgrade() {
                    GenerationUpgrade::Ready(client) => Some(client),
                    GenerationUpgrade::Closing(_) | GenerationUpgrade::Gone => None,
                }
            })
            .collect::<Vec<_>>()
    };

    for stage in [DeviceServiceStage::Upstream, DeviceServiceStage::Downstream] {
        permit.wait_for_initializers();

        {
            let mut guard = CHANNELS.lock();
            let channels = guard.get_or_insert_with(HashMap::new);
            let keys = channels
                .keys()
                .filter_map(|(runner_id, type_id)| {
                    (runner_id.stage == stage).then_some((*runner_id, *type_id))
                })
                .collect::<Vec<_>>();

            for key in keys {
                channels.remove(&key);
            }
        }

        let closers = {
            let mut guard = RUNNERS.lock();
            let runners = guard.get_or_insert_with(HashMap::new);
            let keys = runners
                .keys()
                .filter(|runner_id| runner_id.stage == stage)
                .copied()
                .collect::<Vec<_>>();

            keys.into_iter()
                .filter_map(|key| runners.remove(&key))
                .filter_map(|entry| entry.closer())
                .collect::<Vec<_>>()
        };

        let mut closed = HashSet::new();
        for closer in closers {
            if !closed.insert(closer.generation_id()) {
                continue;
            }
            if closer.shutdown().is_err() {
                runner_panics += 1;
            }
        }
    }

    core::mem::drop(downstream_pins);

    if runner_panics == 0 {
        Ok(())
    } else {
        Err(DeviceServicesShutdownError::new(runner_panics))
    }
}

impl ChannelDeviceState {
    pub fn init<S: DeviceService>(
        device_id: DeviceId,
        service: Option<S>,
    ) -> Result<Self, ServiceCreationError> {
        let type_id = TypeId::of::<S>();
        let runner_id = RunnerId {
            device: device_id,
            stage: S::stage(),
        };
        let _permit = InitializationPermit::acquire(runner_id.stage);
        let key = (runner_id, type_id);

        loop {
            // Hold `CHANNELS` across runner lookup and service initialization so concurrent callers
            // can't initialize the same `(RunnerId, TypeId)` twice.
            let mut guard_channel = CHANNELS.lock();
            let channels = guard_channel.get_or_insert_with(HashMap::new);

            if let Some(existing) = channels.get(&key).and_then(ChannelRegistryEntry::upgrade) {
                if existing.client.is_active() {
                    if service.is_some() {
                        return Err(ServiceCreationError::new(
                            "Service already initialized.".into(),
                        ));
                    }
                    return Ok(existing);
                }
                channels.remove(&key);
            }

            // A single device runner can serve multiple [`DeviceService`]. Re-read after every
            // close wait so concurrent waiters observe a replacement installed by another caller.
            let device_client = {
                let mut guard = RUNNERS.lock();
                let runners = guard.get_or_insert_with(HashMap::new);
                let upgrade = runners
                    .get(&runner_id)
                    .map(RunnerRegistryEntry::upgrade)
                    .unwrap_or(GenerationUpgrade::Gone);
                match upgrade {
                    GenerationUpgrade::Ready(client) => Ok(client),
                    GenerationUpgrade::Closing(waiter) => Err(waiter),
                    GenerationUpgrade::Gone => {
                        let client = DeviceRunner::start(runner_id);
                        runners.insert(runner_id, RunnerRegistryEntry::new(&client));
                        Ok(client)
                    }
                }
            };

            let device_client = match device_client {
                Ok(client) => client,
                Err(waiter) => {
                    drop(guard_channel);
                    waiter.wait_closed();
                    continue;
                }
            };

            let (callback, recv) = oneshot::channel();

            // The service initialization function.
            let initialize_service = move || {
                STATES.with(|state| {
                    let mut map = match state.try_borrow_mut() {
                        Ok(map) => map,
                        Err(err) => panic!(
                            "The device service {:?} is already borrowed: {err}",
                            core::any::type_name::<S>()
                        ),
                    };

                    if let Some(existing) = map.get(&type_id) {
                        if service.is_some() {
                            callback.send(Err(())).unwrap();
                        } else {
                            callback
                                .send(Ok(ChannelService {
                                    type_id,
                                    utilities: existing.utilities.clone(),
                                }))
                                .unwrap();
                        }
                        return;
                    }

                    let service = service.unwrap_or_else(|| S::init(device_id));
                    let state = ServiceState::new(service);
                    let utilities = state.utilities.clone();
                    map.insert(type_id, state);
                    callback
                        .send(Ok(ChannelService { type_id, utilities }))
                        .unwrap();
                });
            };

            // Same reason in [`send]` we need to call the function directly if we are on the runner
            // thread.
            if is_device_runner_thread(&runner_id) {
                if let Err(err) = catch_unwind(AssertUnwindSafe(initialize_service)) {
                    return Err(ServiceCreationError::new(std::format!(
                        "Service initialization failed: {err:?}"
                    )));
                };
            } else {
                device_client.enqueue(initialize_service).unwrap();
                device_client.flush();
            };

            let service = recv.recv().unwrap();

            let service = match service {
                Ok(service) => service,
                Err(_) => {
                    return Err(ServiceCreationError::new(
                        "Service already initialized.".into(),
                    ));
                }
            };
            let channel = Self {
                inner: Arc::new(ChannelDeviceStateInner {
                    client: device_client,
                    service,
                }),
            };

            channels.insert(key, ChannelRegistryEntry::new(&channel));

            return Ok(channel);
        }
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.service.utilities.clone()
    }
}

impl ChannelService {
    /// Borrows the service state from thread-local storage and passes it to `f`.
    /// Panics if the state is already borrowed (re-entrant access).
    fn act_on<R>(&self, f: impl FnOnce(&mut Box<dyn Any + 'static>) -> R) -> R {
        STATES.with_borrow(|map| {
            let state = map.get(&self.type_id).expect("Service state not found");
            let mut guard = state
                .service
                .try_borrow_mut()
                .expect("Service state is already borrowed");
            f(&mut guard)
        })
    }
}

impl DeviceRunner {
    /// Spawns a new thread, marks it with the `device_id`, and returns a `DeviceClient`.
    pub fn start(runner_id: RunnerId) -> DeviceClient {
        let (sender_init, recv_init) = oneshot::channel();
        let channel = DeviceClient::new(
            runner_id,
            move || {
                SERVER_THREAD.with_borrow_mut(|cell| *cell = Some(runner_id));
                sender_init.send(()).unwrap();
            },
            || {
                STATES.with_borrow_mut(|states| {
                    shutdown_service_states(states);
                });
            },
        );

        if recv_init.recv().is_err() {
            panic!("Failed to synchronize device runner thread initialization");
        }

        channel
    }
}

impl<S: DeviceService> Clone for ChannelDeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            _phantom: self._phantom,
        }
    }
}

mod task {
    use super::*;
    use core::sync::atomic::{AtomicPtr, Ordering};
    use std::{
        mem::{align_of, size_of},
        panic::{AssertUnwindSafe, catch_unwind},
    };

    /// The maximum size of a closure that can be stored without heap allocation.
    pub const GLOBAL_TASK_MAX_SIZE: usize = 4096;

    /// The maximum size of a closure that can be stored using inlined memory.
    const INLINE_TASK_MAX_SIZE: usize = 40;

    /// One arena slot. `#[repr(C, align(64))]` makes every slot 64-byte
    /// aligned on its own, so the slot alignment does not depend on the layout of any
    /// enclosing type. `GLOBAL_TASK_MAX_SIZE` is a multiple of 64, so there is no
    /// per-slot padding.
    #[repr(C, align(64))]
    pub struct ArenaSlot {
        pub data: [u8; GLOBAL_TASK_MAX_SIZE],
    }

    #[repr(C, align(64))]
    /// A task is how we represent closures in memory without extra allocations.
    ///
    /// It fits in 64 bytes, ensuring multiple threads can initialize tasks at the same time
    /// without causing false sharing.
    pub struct Task {
        // 40 bytes; 64-aligned because it is the first field of a 64-aligned struct.
        data: [u8; INLINE_TASK_MAX_SIZE],
        // 8 bytes (usize/u64 ptr)
        data_large_ptr: AtomicPtr<u8>,
        // 8 bytes (usize/u64 ptr)
        fn_ptr: fn(&mut Task),
        // 8 bytes (usize/u64 ptr)
        drop_ptr: fn(&mut Task),
    }

    const _: () = {
        // ArenaSlot is 4096 bytes and 64-aligned on its own.
        assert!(core::mem::size_of::<ArenaSlot>() == GLOBAL_TASK_MAX_SIZE);
        // `Task::data` lives at offset 0 of a 64-aligned 64-byte struct, which is
        // what lets the router assume the inline slot has `SLOT_ALIGN`-byte alignment.
        assert!(core::mem::size_of::<Task>() == 64);
        assert!(core::mem::align_of::<Task>() == core::mem::align_of::<ArenaSlot>());
        assert!(core::mem::offset_of!(Task, data) == 0);
    };

    impl Task {
        pub fn new(large_data_ptr: *mut u8) -> Self {
            Self {
                data: [0u8; INLINE_TASK_MAX_SIZE],
                data_large_ptr: AtomicPtr::new(large_data_ptr),
                fn_ptr: |_| {},
                drop_ptr: |_| {},
            }
        }

        /// Store `func` in the inline slot, the arena slot, or on the heap depending on
        /// its size and alignment. Both checks are required: writing into a slot whose
        /// alignment is smaller than `align_of::<F>()` would produce a misaligned
        /// `ptr::write` (UB). The boxed fallback uses `Box::new`, whose allocation
        /// satisfies any alignment.
        pub fn init<F: FnOnce() + Send + 'static>(&mut self, func: F) {
            let fits_inline = size_of::<F>() <= INLINE_TASK_MAX_SIZE
                && align_of::<F>() <= align_of::<ArenaSlot>();
            let fits_arena = size_of::<F>() <= GLOBAL_TASK_MAX_SIZE
                && align_of::<F>() <= align_of::<ArenaSlot>();

            if fits_inline {
                // SAFETY: size + align checked above, read back exactly once by fn_ptr.
                unsafe { std::ptr::write(self.data.as_mut_ptr() as *mut F, func) };
                self.fn_ptr = |task| {
                    // SAFETY: Paired with the ptr::write to data above.
                    let f = unsafe { std::ptr::read(task.data.as_mut_ptr() as *mut F) };
                    if let Err(err) = catch_unwind(AssertUnwindSafe(f)) {
                        log::warn!("Task failed: {err:?}");
                    }
                };
                self.drop_ptr = |task| {
                    // SAFETY: Paired with the ptr::write to data above.
                    unsafe { std::ptr::drop_in_place(task.data.as_mut_ptr() as *mut F) };
                };
            } else if fits_arena {
                // SAFETY: size + align checked above, read back exactly once by fn_ptr.
                unsafe {
                    std::ptr::write(self.data_large_ptr.load(Ordering::Relaxed) as *mut F, func)
                };
                self.fn_ptr = |task| {
                    // SAFETY: Paired with the ptr::write to data_large_ptr above.
                    let f = unsafe {
                        std::ptr::read(task.data_large_ptr.load(Ordering::Relaxed) as *mut F)
                    };
                    if let Err(err) = catch_unwind(AssertUnwindSafe(f)) {
                        log::warn!("Task failed: {err:?}");
                    }
                };
                self.drop_ptr = |task| {
                    // SAFETY: Paired with the ptr::write to data_large_ptr above.
                    unsafe {
                        std::ptr::drop_in_place(
                            task.data_large_ptr.load(Ordering::Relaxed) as *mut F
                        )
                    };
                };
            } else {
                // Size or alignment exceeds both slots. Heap-allocate to get a
                // properly-aligned, pointer-sized handle, then recurse as an inline
                // task (the Box is a pointer so it trivially fits inline).
                let boxed: Box<dyn FnOnce() + Send> = Box::new(func);
                self.init(boxed);
            }
        }

        /// Runs the task.
        ///
        /// The task must be initialized and run only once per initialization.
        pub fn run(&mut self) {
            let fn_ptr = core::mem::replace(&mut self.fn_ptr, |_| {});
            self.drop_ptr = |_| {};
            fn_ptr(self)
        }

        /// Drops an initialized task without running it.
        fn discard(&mut self) {
            let drop_ptr = core::mem::replace(&mut self.drop_ptr, |_| {});
            self.fn_ptr = |_| {};
            drop_ptr(self)
        }
    }

    impl Drop for Task {
        fn drop(&mut self) {
            self.discard();
        }
    }
}

/// A normal channel implementation, use for debugging.
#[allow(dead_code)]
mod normal_channel {
    use super::RunnerId;
    use crate::device::handle::CallError;
    use alloc::boxed::Box;
    use std::sync::mpsc::SyncSender;

    /// Buffer size for the command channel.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// The client-side handle used to enqueue tasks.
    pub struct DeviceClient {
        state: SyncSender<Box<dyn FnOnce() + Send + 'static>>,
        runner_id: RunnerId,
    }

    impl Clone for DeviceClient {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                runner_id: self.runner_id,
            }
        }
    }

    impl DeviceClient {
        /// Gets the device id associated to the channel.
        pub fn runner_id(&self) -> &RunnerId {
            &self.runner_id
        }
        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I: FnOnce() + Send + 'static>(runner_id: RunnerId, init: I) -> Self {
            let (sender, recv) = std::sync::mpsc::sync_channel::<Box<dyn FnOnce() + Send + 'static>>(
                CHANNEL_MAX_TASK,
            );

            std::thread::spawn(move || {
                init();
                loop {
                    if let Ok(item) = recv.recv() {
                        item()
                    }
                }
            });

            Self {
                state: sender,
                runner_id,
            }
        }

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() + Send + 'static>(&self, func: F) -> Result<(), CallError> {
            self.state.send(Box::new(func)).map_err(|_| CallError)
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            // Nothing to do.
        }
    }
}

/// We implement a custom channel with automatic batching, no locking and
/// no allocation (most of the time, see [`task`] for more details.
mod custom_channel {
    use crate::device::handle::{
        CallError, DeviceGenerationId, DeviceLeaseRelease,
        channel::{
            CLOSED_GENERATIONS, CREATED_GENERATIONS, RunnerId, is_device_runner_thread,
            task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task},
        },
    };
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering},
        time::Duration,
    };
    use std::{
        panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
        sync::{Arc, Condvar, Mutex, Weak},
        thread::JoinHandle,
        vec::Vec,
    };

    /// Maximum number of [`Task`] that can be queued.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// Number of `spin_loop` iterations before the server starts yielding.
    /// Gives a hot window to absorb back-to-back submits without any syscall.
    const SPIN_BUDGET_SERVER: u32 = 8192;
    /// Number of `thread::yield_now` calls after the spin budget is exhausted,
    /// before the server drops to sleeping.
    const YIELD_BUDGET_SERVER: u32 = 64;
    /// Sleep duration once both the spin and yield budgets are exhausted.
    /// Bounds the wake-up latency from a fully idle state.
    const SLEEP_STEP_SERVER: Duration = Duration::from_micros(150);

    /// The client has the buffer to fill plus we add a factor of two to account for the double
    /// buffering approach.
    const CLIENT_BUDGET_FACTOR: u32 = CHANNEL_MAX_TASK as u32 * 2u32;

    /// Number of `spin_loop` iterations on the client before yielding when the
    /// queue is full. Longer than the server budget because a producer stall is
    /// expected to resolve quickly (the server only needs to swap buffers).
    const SPIN_BUDGET_CLIENT: u32 = SPIN_BUDGET_SERVER * CLIENT_BUDGET_FACTOR;
    /// Number of `thread::yield_now` calls on the client after the spin budget
    /// is exhausted, before dropping to sleeping.
    const YIELD_BUDGET_CLIENT: u32 = YIELD_BUDGET_SERVER * CLIENT_BUDGET_FACTOR;
    /// Sleep duration on the client once both budgets are exhausted. Kept short
    /// to avoid stalling the producer's critical path.
    const SLEEP_STEP_CLIENT: Duration = Duration::from_micros(75);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum GenerationStatus {
        Starting,
        Running,
        Closing,
        Closed,
    }

    enum BeginClose {
        Started,
        Shared,
        AlreadyClosing,
    }

    pub(super) enum GenerationUpgrade {
        Ready(DeviceClient),
        Closing(GenerationWaiter),
        Gone,
    }

    pub(super) struct GenerationWaiter {
        generation: Arc<RuntimeGeneration>,
    }

    impl GenerationWaiter {
        pub(super) fn wait_closed(self) {
            assert!(
                !is_device_runner_thread(&self.generation.runner_id),
                "a device runner cannot wait for its own generation to close"
            );
            if self.generation.join_runner().is_err() {
                log::warn!(
                    "Device runner {:?} panicked before replacement",
                    self.generation.runner_id
                );
            }
        }
    }

    #[derive(Clone)]
    pub(super) struct GenerationCloser {
        generation: Arc<RuntimeGeneration>,
    }

    impl GenerationCloser {
        pub(super) fn generation_id(&self) -> DeviceGenerationId {
            self.generation.id
        }

        pub(super) fn shutdown(&self) -> Result<(), ()> {
            RuntimeGeneration::close_forced(&self.generation).map(|_| ())
        }
    }

    #[derive(Clone)]
    pub(super) struct WeakDeviceClient {
        generation: Weak<RuntimeGeneration>,
    }

    impl WeakDeviceClient {
        pub(super) fn upgrade(&self) -> GenerationUpgrade {
            let Some(generation) = self.generation.upgrade() else {
                return GenerationUpgrade::Gone;
            };
            RuntimeGeneration::upgrade(generation)
        }

        pub(super) fn closer(&self) -> Option<GenerationCloser> {
            self.generation
                .upgrade()
                .map(|generation| GenerationCloser { generation })
        }
    }

    struct GenerationLifecycle {
        status: GenerationStatus,
        runner_panicked: bool,
    }

    struct RuntimeGeneration {
        id: DeviceGenerationId,
        runner_id: RunnerId,
        state: Arc<State>,
        external_count: AtomicUsize,
        internal_count: AtomicUsize,
        lifecycle: Mutex<GenerationLifecycle>,
        closed: Condvar,
        join_handle: Mutex<Option<JoinHandle<()>>>,
        join_delegated: AtomicBool,
    }

    impl RuntimeGeneration {
        fn new(runner_id: RunnerId, state: Arc<State>) -> Arc<Self> {
            static NEXT_GENERATION_ID: core::sync::atomic::AtomicU64 =
                core::sync::atomic::AtomicU64::new(1);

            let generation = Arc::new(Self {
                id: DeviceGenerationId::new(NEXT_GENERATION_ID.fetch_add(1, Ordering::Relaxed)),
                runner_id,
                state,
                external_count: AtomicUsize::new(0),
                internal_count: AtomicUsize::new(0),
                lifecycle: Mutex::new(GenerationLifecycle {
                    status: GenerationStatus::Starting,
                    runner_panicked: false,
                }),
                closed: Condvar::new(),
                join_handle: Mutex::new(None),
                join_delegated: AtomicBool::new(false),
            });
            CREATED_GENERATIONS.fetch_add(1, Ordering::Relaxed);
            generation
        }

        fn external_lease_unchecked(self: &Arc<Self>) -> ExternalLease {
            self.external_count.fetch_add(1, Ordering::Relaxed);
            ExternalLease {
                generation: Some(Arc::clone(self)),
            }
        }

        fn upgrade(generation: Arc<Self>) -> GenerationUpgrade {
            let lifecycle = generation.lifecycle.lock().unwrap();
            match lifecycle.status {
                GenerationStatus::Starting | GenerationStatus::Running => {
                    generation.external_count.fetch_add(1, Ordering::Relaxed);
                    drop(lifecycle);
                    let state = Arc::clone(&generation.state);
                    GenerationUpgrade::Ready(DeviceClient {
                        state,
                        lease: ExternalLease {
                            generation: Some(generation),
                        },
                    })
                }
                GenerationStatus::Closing => {
                    drop(lifecycle);
                    GenerationUpgrade::Closing(GenerationWaiter { generation })
                }
                GenerationStatus::Closed => GenerationUpgrade::Gone,
            }
        }

        fn internal_reference(self: &Arc<Self>) -> InternalReference {
            self.internal_count.fetch_add(1, Ordering::Relaxed);
            InternalReference {
                generation: Arc::clone(self),
            }
        }

        fn set_join_handle(&self, join_handle: JoinHandle<()>) {
            *self.join_handle.lock().unwrap() = Some(join_handle);
        }

        fn mark_running(&self) {
            let mut lifecycle = self.lifecycle.lock().unwrap();
            if lifecycle.status == GenerationStatus::Starting {
                lifecycle.status = GenerationStatus::Running;
            }
        }

        fn mark_closed(&self, panicked: bool) {
            let mut lifecycle = self.lifecycle.lock().unwrap();
            lifecycle.runner_panicked |= panicked;
            if lifecycle.status != GenerationStatus::Closed {
                lifecycle.status = GenerationStatus::Closed;
                CLOSED_GENERATIONS.fetch_add(1, Ordering::Relaxed);
            }
            self.closed.notify_all();
        }

        fn begin_close(&self, only_if_unowned: bool) -> BeginClose {
            let mut lifecycle = self.lifecycle.lock().unwrap();
            match lifecycle.status {
                GenerationStatus::Starting | GenerationStatus::Running => {
                    if only_if_unowned && self.external_count.load(Ordering::Acquire) != 0 {
                        return BeginClose::Shared;
                    }
                    lifecycle.status = GenerationStatus::Closing;
                    drop(lifecycle);
                    self.state.accepting.swap(false, Ordering::SeqCst);
                    self.state.close_requested.store(true, Ordering::Release);
                    BeginClose::Started
                }
                GenerationStatus::Closing | GenerationStatus::Closed => BeginClose::AlreadyClosing,
            }
        }

        fn finish_close(generation: &Arc<Self>) -> Result<DeviceLeaseRelease, ()> {
            if is_device_runner_thread(&generation.runner_id) {
                Self::delegate_join(generation);
                return Ok(DeviceLeaseRelease::Closing);
            }

            generation
                .join_runner()
                .map(|()| DeviceLeaseRelease::Closed)
        }

        fn close_if_unowned(generation: &Arc<Self>) -> Result<DeviceLeaseRelease, ()> {
            match generation.begin_close(true) {
                BeginClose::Shared => Ok(DeviceLeaseRelease::Shared),
                BeginClose::Started | BeginClose::AlreadyClosing => Self::finish_close(generation),
            }
        }

        fn close_forced(generation: &Arc<Self>) -> Result<DeviceLeaseRelease, ()> {
            match generation.begin_close(false) {
                BeginClose::Started | BeginClose::AlreadyClosing => Self::finish_close(generation),
                BeginClose::Shared => unreachable!("forced close ignores external owners"),
            }
        }

        fn close(generation: &Arc<Self>) -> Result<(), ()> {
            Self::close_forced(generation).map(|_| ())
        }

        fn delegate_join(generation: &Arc<Self>) {
            if generation.join_handle.lock().unwrap().is_none() {
                return;
            }
            if generation
                .join_delegated
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return;
            }

            let runner_id = generation.runner_id;
            let join_generation = Arc::clone(generation);
            let result = std::thread::Builder::new()
                .name(std::format!(
                    "DS-J-{}-{}",
                    runner_id.device.type_id,
                    runner_id.device.index_id
                ))
                .spawn(move || {
                    if join_generation.join_runner().is_err() {
                        log::warn!(
                            "Device runner {:?} panicked during delegated shutdown",
                            join_generation.runner_id
                        );
                    }
                });

            if let Err(err) = result {
                generation.join_delegated.store(false, Ordering::Release);
                // The runner still owns an internal reference and will quiesce itself. Keep the
                // join handle available in case another off-runner closer arrives.
                log::warn!(
                    "Failed to spawn joiner for device runner {:?}: {err}",
                    runner_id
                );
            }
        }

        fn join_runner(&self) -> Result<(), ()> {
            let join_handle = self.join_handle.lock().unwrap().take();
            if let Some(join_handle) = join_handle {
                let panicked = join_handle.join().is_err();
                self.mark_closed(panicked);
            } else {
                let mut lifecycle = self.lifecycle.lock().unwrap();
                while lifecycle.status != GenerationStatus::Closed {
                    lifecycle = self.closed.wait(lifecycle).unwrap();
                }
            }

            let lifecycle = self.lifecycle.lock().unwrap();
            if lifecycle.runner_panicked {
                Err(())
            } else {
                Ok(())
            }
        }

        #[cfg(test)]
        fn status(&self) -> GenerationStatus {
            self.lifecycle.lock().unwrap().status
        }
    }

    pub(in crate::device::handle) struct ExternalLease {
        generation: Option<Arc<RuntimeGeneration>>,
    }

    impl ExternalLease {
        fn generation(&self) -> &Arc<RuntimeGeneration> {
            self.generation.as_ref().unwrap()
        }

        pub(in crate::device::handle) fn generation_id(&self) -> DeviceGenerationId {
            self.generation().id
        }

        pub(in crate::device::handle) fn release(
            mut self,
        ) -> Result<
            crate::device::handle::DeviceLeaseRelease,
            crate::device::handle::DeviceLeaseReleaseError,
        > {
            self.release_inner()
        }

        fn release_inner(
            &mut self,
        ) -> Result<
            crate::device::handle::DeviceLeaseRelease,
            crate::device::handle::DeviceLeaseReleaseError,
        > {
            use crate::device::handle::{DeviceLeaseRelease, DeviceLeaseReleaseError};

            let Some(generation) = self.generation.take() else {
                return Ok(DeviceLeaseRelease::Stateless);
            };
            if generation.external_count.fetch_sub(1, Ordering::AcqRel) != 1 {
                return Ok(DeviceLeaseRelease::Shared);
            }

            match RuntimeGeneration::close_if_unowned(&generation) {
                Ok(outcome) => Ok(outcome),
                Err(()) => Err(DeviceLeaseReleaseError::closed_with_runner_panic(
                    generation.id,
                )),
            }
        }
    }

    impl Clone for ExternalLease {
        fn clone(&self) -> Self {
            let generation = self.generation.as_ref().unwrap();
            generation.external_count.fetch_add(1, Ordering::Relaxed);
            Self {
                generation: Some(Arc::clone(generation)),
            }
        }
    }

    impl Drop for ExternalLease {
        fn drop(&mut self) {
            if let Err(err) = self.release_inner() {
                log::warn!("Device lease shutdown failed: {err}");
            }
        }
    }

    struct InternalReference {
        generation: Arc<RuntimeGeneration>,
    }

    impl Drop for InternalReference {
        fn drop(&mut self) {
            self.generation
                .internal_count
                .fetch_sub(1, Ordering::Release);
        }
    }

    /// The client-side handle used to enqueue tasks.
    pub struct DeviceClient {
        state: Arc<State>,
        lease: ExternalLease,
    }

    impl Clone for DeviceClient {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                lease: self.lease.clone(),
            }
        }
    }

    impl DeviceClient {
        pub(super) fn downgrade(&self) -> WeakDeviceClient {
            WeakDeviceClient {
                generation: Arc::downgrade(self.lease.generation()),
            }
        }

        pub(super) fn try_clone_generation(&self) -> GenerationUpgrade {
            RuntimeGeneration::upgrade(Arc::clone(self.lease.generation()))
        }

        pub(super) fn closer(&self) -> GenerationCloser {
            GenerationCloser {
                generation: Arc::clone(self.lease.generation()),
            }
        }

        pub(super) fn is_active(&self) -> bool {
            matches!(
                self.lease.generation().lifecycle.lock().unwrap().status,
                GenerationStatus::Starting | GenerationStatus::Running
            )
        }

        /// Gets the runner id associated to the channel.
        pub fn runner_id(&self) -> &RunnerId {
            &self.lease.generation().runner_id
        }

        pub fn generation_id(&self) -> DeviceGenerationId {
            self.lease.generation().id
        }

        pub fn external_lease(&self) -> ExternalLease {
            self.lease.clone()
        }

        #[cfg(test)]
        pub fn external_count(&self) -> usize {
            self.lease
                .generation()
                .external_count
                .load(Ordering::Acquire)
        }

        #[cfg(test)]
        pub fn internal_count(&self) -> usize {
            self.lease
                .generation()
                .internal_count
                .load(Ordering::Acquire)
        }

        #[cfg(test)]
        pub fn generation_status(&self) -> GenerationStatus {
            self.lease.generation().status()
        }

        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I, S>(runner_id: RunnerId, init: I, shutdown: S) -> Self
        where
            I: FnOnce() + Send + 'static,
            S: FnOnce() + Send + 'static,
        {
            let mut server = Server::new(runner_id);
            let state = server.state.clone();
            let generation = RuntimeGeneration::new(runner_id, Arc::clone(&state));
            let lease = generation.external_lease_unchecked();
            let runner_reference = generation.internal_reference();
            let runner_generation = Arc::clone(&generation);

            let join_handle = std::thread::Builder::new()
                .name(std::format!(
                    "DS{}-{}-{}-g{}",
                    match runner_id.stage {
                        crate::device::DeviceServiceStage::Upstream => "U",
                        crate::device::DeviceServiceStage::Downstream => "D",
                    },
                    runner_id.device.type_id,
                    runner_id.device.index_id,
                    generation.id.0
                ))
                .spawn(move || {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        init();
                        runner_generation.mark_running();
                        server.start();
                        shutdown();
                    }));
                    core::mem::drop(runner_reference);

                    if let Err(err) = result {
                        resume_unwind(err);
                    }
                })
                .unwrap();

            generation.set_join_handle(join_handle);

            Self { state, lease }
        }

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() + Send + 'static>(&self, func: F) -> Result<(), CallError> {
            if !self.state.accepting.load(Ordering::SeqCst) {
                return Err(CallError);
            }
            self.state.active_enqueues.fetch_add(1, Ordering::SeqCst);
            if !self.state.accepting.load(Ordering::SeqCst) {
                self.state.active_enqueues.fetch_sub(1, Ordering::SeqCst);
                return Err(CallError);
            }

            let reference = self.lease.generation().internal_reference();
            self.enqueue_unchecked(move || {
                let _reference = reference;
                func();
            });
            self.state.active_enqueues.fetch_sub(1, Ordering::SeqCst);

            Ok(())
        }

        fn enqueue_unchecked<F: FnOnce() + Send + 'static>(&self, func: F) {
            let mut idle_count: u32 = 0;
            loop {
                let index = self.state.available_index.fetch_add(1, Ordering::Acquire) as usize;
                if index >= CHANNEL_MAX_TASK {
                    // The queue is full; back off until the server flushes/swaps buffers.
                    if idle_count < SPIN_BUDGET_CLIENT {
                        spin_loop();
                    } else if idle_count < SPIN_BUDGET_CLIENT + YIELD_BUDGET_CLIENT {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(SLEEP_STEP_CLIENT);
                    }
                    idle_count = idle_count.saturating_add(1);
                    continue;
                }

                self.state.init_task_at(index, func);
                self.state.enqueued_count.fetch_add(1, Ordering::SeqCst);
                return;
            }
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            if !self.state.accepting.load(Ordering::SeqCst) {
                return;
            }
            self.state.active_enqueues.fetch_add(1, Ordering::SeqCst);
            if !self.state.accepting.load(Ordering::SeqCst) {
                self.state.active_enqueues.fetch_sub(1, Ordering::SeqCst);
                return;
            }

            self.state.publish_partial_buffer();
            self.state.active_enqueues.fetch_sub(1, Ordering::SeqCst);
        }

        /// Stops accepting work, drains queued tasks, and joins the runner.
        pub fn shutdown(&self) -> Result<(), ()> {
            RuntimeGeneration::close(self.lease.generation())
        }

        #[cfg(test)]
        /// Returns whether the runner still accepts task submissions.
        pub fn is_accepting(&self) -> bool {
            self.state.accepting.load(Ordering::Acquire)
        }
    }

    struct State {
        /// Pointer to the current active queue buffer.
        ///
        /// Written by the server thread (Release) after swapping buffers,
        /// read by client threads (Acquire) before writing tasks.
        queue_ptr: AtomicPtr<Task>,
        /// Next available index for writing.
        available_index: AtomicU32,
        /// Number of tasks successfully written and ready for processing.
        enqueued_count: AtomicU32,
        /// Whether clients may enqueue new work.
        accepting: AtomicBool,
        /// Number of clients currently writing queue slots.
        active_enqueues: AtomicU32,
        /// Whether the server should stop after draining every accepted task.
        close_requested: AtomicBool,
    }

    impl State {
        /// Initializes the task at `index` in the current queue with `func`.
        /// Exclusive access per slot is guaranteed by `available_index.fetch_add`.
        fn init_task_at<F: FnOnce() + Send + 'static>(&self, index: usize, func: F) {
            assert!(index < CHANNEL_MAX_TASK, "task index {index} out of bounds");
            // SAFETY: queue_ptr points to a valid buffer of CHANNEL_MAX_TASK tasks,
            // bounds checked above, and the &mut doesn't escape.
            unsafe { &mut *self.queue_ptr.load(Ordering::Acquire).add(index) }.init(func);
        }

        fn publish_partial_buffer(&self) {
            let index_start =
                self.available_index
                    .fetch_add(CHANNEL_MAX_TASK as u32, Ordering::Acquire) as usize;

            if index_start >= CHANNEL_MAX_TASK {
                return;
            }

            for index in index_start..CHANNEL_MAX_TASK {
                self.init_task_at(index, || ());
            }

            self.enqueued_count
                .fetch_add((CHANNEL_MAX_TASK - index_start) as u32, Ordering::SeqCst);
        }
    }

    /// Owns a task buffer and its associated large-closure arena.
    struct TaskBuffer {
        tasks: Vec<Task>,
        _arena: Vec<ArenaSlot>,
    }

    impl TaskBuffer {
        fn new() -> Self {
            let mut arena: Vec<ArenaSlot> =
                Vec::from_iter((0..CHANNEL_MAX_TASK).map(|_| ArenaSlot {
                    data: [0u8; GLOBAL_TASK_MAX_SIZE],
                }));

            let arena_ptr = arena.as_mut_ptr() as *mut u8;
            let tasks = Vec::from_iter((0..CHANNEL_MAX_TASK).map(|index| {
                // SAFETY: Each task owns a non-overlapping `ArenaSlot` region.
                Task::new(unsafe { arena_ptr.add(index * GLOBAL_TASK_MAX_SIZE) })
            }));
            Self {
                tasks,
                _arena: arena,
            }
        }
    }

    /// The server-side runner that processes tasks.
    struct Server {
        state: Arc<State>,
        /// Index into `buffers`: which buffer clients are currently writing to.
        client_buf: usize,
        buffers: [TaskBuffer; 2],
        ready_to_execute: bool,
    }

    impl Server {
        fn new(_runner_id: RunnerId) -> Self {
            let mut buffers = [TaskBuffer::new(), TaskBuffer::new()];

            let state = Arc::new(State {
                queue_ptr: AtomicPtr::new(buffers[0].tasks.as_mut_ptr()),
                available_index: AtomicU32::new(0),
                enqueued_count: AtomicU32::new(0),
                accepting: AtomicBool::new(true),
                active_enqueues: AtomicU32::new(0),
                close_requested: AtomicBool::new(false),
            });

            Self {
                state,
                client_buf: 0,
                buffers,
                ready_to_execute: false,
            }
        }

        /// Main execution loop for the device thread.
        fn start(&mut self) {
            let mut idle_count: u32 = 0;
            loop {
                if self.ready_to_execute {
                    self.execute_tasks();
                    idle_count = 0;
                }

                let queue_size = self.state.enqueued_count.load(Ordering::Acquire) as usize;

                if queue_size >= CHANNEL_MAX_TASK {
                    self.fetch();
                    idle_count = 0;
                    continue;
                }

                if self.state.close_requested.load(Ordering::Acquire)
                    && self.state.active_enqueues.load(Ordering::SeqCst) == 0
                {
                    let queue_size = self.state.enqueued_count.load(Ordering::SeqCst) as usize;
                    if queue_size == 0 {
                        return;
                    }

                    self.state.publish_partial_buffer();
                    continue;
                }

                if idle_count < SPIN_BUDGET_SERVER {
                    spin_loop();
                } else if idle_count < SPIN_BUDGET_SERVER + YIELD_BUDGET_SERVER {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(SLEEP_STEP_SERVER);
                }
                idle_count = idle_count.saturating_add(1);
            }
        }

        fn execute_tasks(&mut self) {
            let server_buf = 1 - self.client_buf;
            for task in &mut self.buffers[server_buf].tasks {
                task.run();
            }
            self.ready_to_execute = false;
        }

        /// Swaps the client and server buffers, allowing the client to start
        /// filling the next buffer while the server processes the current one.
        fn fetch(&mut self) {
            self.client_buf = 1 - self.client_buf;

            self.state.queue_ptr.store(
                self.buffers[self.client_buf].tasks.as_mut_ptr(),
                Ordering::Release,
            );

            self.ready_to_execute = true;

            // Reset indices for the new client buffer
            self.state.enqueued_count.store(0, Ordering::SeqCst);

            // This is what is used for the spin loop on the client size.
            //
            // It is very important to be the last thing to reset.
            self.state.available_index.store(0, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::handle::DeviceLeaseRelease;
    use crate::device::handle::channel::custom_channel::CHANNEL_MAX_TASK;

    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    // A mock service to track state changes and initialization
    struct MockService {
        counter: usize,
        id: DeviceId,
    }

    impl DeviceService for MockService {
        fn init(id: DeviceId) -> Self {
            Self { counter: 0, id }
        }

        fn utilities(&self) -> ServerUtilitiesHandle {
            Arc::new(())
        }
    }

    #[test]
    fn test_shutdown_waits_for_active_task_and_runs_service_shutdown() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 100,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (task_started_tx, task_started_rx) = mpsc::channel();
        let (task_release_tx, task_release_rx) = mpsc::channel();
        let (service_shutdown_tx, service_shutdown_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(
            runner_id,
            || {},
            move || service_shutdown_tx.send(()).unwrap(),
        );

        client
            .enqueue(move || {
                task_started_tx.send(()).unwrap();
                task_release_rx.recv().unwrap();
            })
            .unwrap();
        client.flush();
        task_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("task should start");

        let (runner_shutdown_tx, runner_shutdown_rx) = mpsc::channel();
        let shutdown_client = client.clone();
        let shutdown_thread = std::thread::spawn(move || {
            shutdown_client.shutdown().unwrap();
            runner_shutdown_tx.send(()).unwrap();
        });

        assert!(
            runner_shutdown_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "shutdown must wait for the active task"
        );

        task_release_tx.send(()).unwrap();
        runner_shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("runner should finish after the active task");
        service_shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("service shutdown should run before the runner exits");
        shutdown_thread.join().unwrap();
    }

    #[test]
    fn test_shutdown_drains_buffered_tasks_and_drops_captures() {
        struct DropSpy(Arc<AtomicUsize>);

        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 101,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (task_started_tx, task_started_rx) = mpsc::channel();
        let (task_release_tx, task_release_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(runner_id, || {}, || {});

        client
            .enqueue(move || {
                task_started_tx.send(()).unwrap();
                task_release_rx.recv().unwrap();
            })
            .unwrap();
        client.flush();
        task_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("task should start");

        let drop_count = Arc::new(AtomicUsize::new(0));
        let run_count = Arc::new(AtomicUsize::new(0));
        let spy = DropSpy(Arc::clone(&drop_count));
        let run_count_task = Arc::clone(&run_count);
        client
            .enqueue(move || {
                let _ = &spy;
                run_count_task.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();

        let shutdown_client = client.clone();
        let shutdown_thread = std::thread::spawn(move || shutdown_client.shutdown().unwrap());
        while client.is_accepting() {
            std::thread::yield_now();
        }
        task_release_tx.send(()).unwrap();
        shutdown_thread.join().unwrap();

        assert_eq!(run_count.load(Ordering::SeqCst), 1);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_final_external_drop_off_runner_closes_generation() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 102,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(
            runner_id,
            move || SERVER_THREAD.with_borrow_mut(|state| *state = Some(runner_id)),
            move || shutdown_tx.send(()).unwrap(),
        );

        drop(client);

        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("final external drop should join the runner");
    }

    #[test]
    fn test_explicit_final_lease_release_closes_generation() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 109,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(
            runner_id,
            || {},
            move || shutdown_tx.send(()).unwrap(),
        );
        let (ready_tx, ready_rx) = mpsc::channel();
        client.enqueue(move || ready_tx.send(()).unwrap()).unwrap();
        client.flush();
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let lease = client.external_lease();
        drop(client);

        assert_eq!(lease.release().unwrap(), DeviceLeaseRelease::Closed);
        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("final explicit release should join the runner");
    }

    #[test]
    fn test_explicit_shared_lease_release_leaves_generation_running() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 110,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(
            runner_id,
            || {},
            move || shutdown_tx.send(()).unwrap(),
        );
        let (ready_tx, ready_rx) = mpsc::channel();
        client.enqueue(move || ready_tx.send(()).unwrap()).unwrap();
        client.flush();
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let lease = client.external_lease();

        assert_eq!(lease.release().unwrap(), DeviceLeaseRelease::Shared);
        assert_eq!(
            client.generation_status(),
            custom_channel::GenerationStatus::Running
        );
        assert!(shutdown_rx.recv_timeout(Duration::from_millis(50)).is_err());

        drop(client);
        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("the remaining owner should close the runner");
    }

    #[test]
    fn test_explicit_final_release_reports_closed_runner_panic() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 111,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let client = custom_channel::DeviceClient::new(runner_id, || {}, || panic!("shutdown"));
        let generation_id = client.generation_id();
        let lease = client.external_lease();
        drop(client);

        let error = lease.release().unwrap_err();
        assert_eq!(error.generation_id(), generation_id);
    }

    #[test]
    fn test_explicit_final_release_on_runner_reports_closing() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 112,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (outcome_tx, outcome_rx) = mpsc::channel();
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(
            runner_id,
            move || SERVER_THREAD.with_borrow_mut(|state| *state = Some(runner_id)),
            move || shutdown_tx.send(()).unwrap(),
        );
        let lease = client.external_lease();
        client
            .enqueue(move || outcome_tx.send(lease.release().unwrap()).unwrap())
            .unwrap();
        client.flush();
        drop(client);

        assert_eq!(
            outcome_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            DeviceLeaseRelease::Closing
        );
        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("delegated join should complete after the runner task returns");
    }

    #[test]
    fn test_registered_final_drop_closes_and_recreates_generation() {
        struct LifecycleService {
            shutdown_tx: Option<mpsc::Sender<()>>,
        }

        impl DeviceService for LifecycleService {
            fn init(_id: DeviceId) -> Self {
                unreachable!()
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn shutdown(&mut self) {
                self.shutdown_tx.take().unwrap().send(()).unwrap();
            }
        }

        let device_id = DeviceId {
            type_id: 0,
            index_id: 113,
        };
        let (first_tx, first_rx) = mpsc::channel();
        let first = ChannelDeviceHandle::<LifecycleService>::insert(
            device_id,
            LifecycleService {
                shutdown_tx: Some(first_tx),
            },
        )
        .unwrap();
        let first_generation = first.state.client.generation_id();
        drop(first);
        first_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("weak registries must not pin the first generation");

        let (second_tx, second_rx) = mpsc::channel();
        let second = ChannelDeviceHandle::<LifecycleService>::insert(
            device_id,
            LifecycleService {
                shutdown_tx: Some(second_tx),
            },
        )
        .unwrap();
        assert_ne!(first_generation, second.state.client.generation_id());
        drop(second);
        second_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("replacement generation should also close on final drop");
    }

    #[test]
    fn test_generation_metrics_track_create_and_close() {
        const CHILD_ENV: &str = "CUBECL_GENERATION_METRICS_TEST_CHILD";

        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("test_generation_metrics_track_create_and_close")
                .arg("--test-threads=1")
                .env(CHILD_ENV, "1")
                .status()
                .unwrap();
            assert!(status.success(), "generation-metrics child process failed");
            return;
        }

        let before = crate::device::handle::device_generation_metrics();
        for index_id in [118, 119] {
            let handle = ChannelDeviceHandle::<MockService>::new(DeviceId {
                type_id: 0,
                index_id,
            });
            drop(handle);
        }
        let delta = crate::device::handle::device_generation_metrics().delta(before);

        assert_eq!(2, delta.created_generations());
        assert_eq!(2, delta.closed_generations());
        assert_eq!(0, delta.active_generations());
    }

    #[test]
    fn test_lease_reattaches_service_without_reinitializing() {
        static INIT_CALLS: AtomicUsize = AtomicUsize::new(0);

        struct ReattachService;

        impl DeviceService for ReattachService {
            fn init(_id: DeviceId) -> Self {
                INIT_CALLS.fetch_add(1, Ordering::SeqCst);
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }
        }

        INIT_CALLS.store(0, Ordering::SeqCst);
        let device_id = DeviceId {
            type_id: 0,
            index_id: 114,
        };
        let first = ChannelDeviceHandle::<ReattachService>::new(device_id);
        let generation = first.state.client.generation_id();
        let lease = first.lease();
        drop(first);

        let second = ChannelDeviceHandle::<ReattachService>::new(device_id);
        assert_eq!(generation, second.state.client.generation_id());
        assert_eq!(INIT_CALLS.load(Ordering::SeqCst), 1);

        drop(lease);
        drop(second);
    }

    #[test]
    fn test_closing_generation_waiters_share_one_replacement() {
        use std::sync::{Mutex as StdMutex, OnceLock};

        static INIT_CALLS: AtomicUsize = AtomicUsize::new(0);
        static SHUTDOWN_STARTED: OnceLock<mpsc::Sender<()>> = OnceLock::new();
        static RELEASE_SHUTDOWN: OnceLock<StdMutex<mpsc::Receiver<()>>> = OnceLock::new();

        struct WaitingService {
            block_shutdown: bool,
        }

        impl DeviceService for WaitingService {
            fn init(_id: DeviceId) -> Self {
                INIT_CALLS.fetch_add(1, Ordering::SeqCst);
                Self {
                    block_shutdown: false,
                }
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn shutdown(&mut self) {
                if self.block_shutdown {
                    SHUTDOWN_STARTED.get().unwrap().send(()).unwrap();
                    RELEASE_SHUTDOWN
                        .get()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .recv()
                        .unwrap();
                }
            }
        }

        INIT_CALLS.store(0, Ordering::SeqCst);
        let (started_tx, started_rx) = mpsc::channel();
        SHUTDOWN_STARTED.set(started_tx).unwrap();
        let (release_tx, release_rx) = mpsc::channel();
        RELEASE_SHUTDOWN.set(StdMutex::new(release_rx)).unwrap();

        let device_id = DeviceId {
            type_id: 0,
            index_id: 115,
        };
        let old = ChannelDeviceHandle::<WaitingService>::insert(
            device_id,
            WaitingService {
                block_shutdown: true,
            },
        )
        .unwrap();
        let shutdown_client = old.state.client.clone();
        let shutdown_thread = std::thread::spawn(move || shutdown_client.shutdown().unwrap());
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let first_waiter =
            std::thread::spawn(move || ChannelDeviceHandle::<WaitingService>::new(device_id));
        let second_waiter =
            std::thread::spawn(move || ChannelDeviceHandle::<WaitingService>::new(device_id));
        release_tx.send(()).unwrap();
        shutdown_thread.join().unwrap();

        let first = first_waiter.join().unwrap();
        let second = second_waiter.join().unwrap();
        assert_eq!(
            first.state.client.generation_id(),
            second.state.client.generation_id()
        );
        assert_eq!(INIT_CALLS.load(Ordering::SeqCst), 1);

        drop((old, first, second));
    }

    #[test]
    fn test_concurrent_final_drop_and_reacquire_returns_usable_generation() {
        use std::sync::Barrier;

        const ITERATIONS: u16 = 100;
        for iteration in 0..ITERATIONS {
            let device_id = DeviceId {
                type_id: 0,
                index_id: 400 + iteration,
            };
            let old = ChannelDeviceHandle::<MockService>::new(device_id);
            let barrier = Arc::new(Barrier::new(2));
            let drop_barrier = Arc::clone(&barrier);
            let drop_thread = std::thread::spawn(move || {
                drop_barrier.wait();
                drop(old);
            });
            let acquire_thread = std::thread::spawn(move || {
                barrier.wait();
                ChannelDeviceHandle::<MockService>::new(device_id)
            });

            drop_thread.join().unwrap();
            let current = acquire_thread.join().unwrap();
            assert_eq!(
                current.submit_blocking(|service| service.id).unwrap(),
                device_id
            );
            drop(current);
        }
    }

    #[test]
    fn test_stale_channel_is_not_reused_after_forced_shutdown() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 117,
        };
        let stale = ChannelDeviceHandle::<MockService>::new(device_id);
        let stale_generation = stale.state.client.generation_id();
        stale.state.client.shutdown().unwrap();

        let replacement = ChannelDeviceHandle::<MockService>::new(device_id);
        assert_ne!(stale_generation, replacement.state.client.generation_id());
        assert_eq!(
            replacement.submit_blocking(|service| service.id).unwrap(),
            device_id
        );

        drop((stale, replacement));
    }

    #[test]
    fn test_strong_registry_rollback_switch() {
        const CHILD_ENV: &str = "CUBECL_STRONG_REGISTRY_TEST_CHILD";

        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("test_strong_registry_rollback_switch")
                .arg("--test-threads=1")
                .env(CHILD_ENV, "1")
                .env("CUBECL_STRONG_DEVICE_REGISTRIES", "1")
                .status()
                .unwrap();
            assert!(status.success(), "strong-registry child process failed");
            return;
        }

        struct RollbackService {
            shutdown_tx: Option<mpsc::Sender<()>>,
        }

        impl DeviceService for RollbackService {
            fn init(_id: DeviceId) -> Self {
                unreachable!()
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn shutdown(&mut self) {
                self.shutdown_tx.take().unwrap().send(()).unwrap();
            }
        }

        let device_id = DeviceId {
            type_id: 0,
            index_id: 116,
        };
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let first = ChannelDeviceHandle::<RollbackService>::insert(
            device_id,
            RollbackService {
                shutdown_tx: Some(shutdown_tx),
            },
        )
        .unwrap();
        let generation = first.state.client.generation_id();
        drop(first);

        assert!(
            shutdown_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "strong registry rollback must pin the generation"
        );
        let second = ChannelDeviceHandle::<RollbackService>::new(device_id);
        assert_eq!(generation, second.state.client.generation_id());
        drop(second);

        shutdown_device_services().unwrap();
        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("explicit shutdown should release strong registry pins");
    }

    #[test]
    fn test_final_external_drop_on_runner_delegates_join() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 103,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (task_started_tx, task_started_rx) = mpsc::channel();
        let (task_release_tx, task_release_rx) = mpsc::channel();
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(
            runner_id,
            || {},
            move || shutdown_tx.send(()).unwrap(),
        );
        let task_client = client.clone();

        client
            .enqueue(move || {
                task_started_tx.send(()).unwrap();
                task_release_rx.recv().unwrap();
                drop(task_client);
            })
            .unwrap();
        client.flush();
        task_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("task should start");

        drop(client);
        task_release_tx.send(()).unwrap();

        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("runner should tear down after the triggering task returns");
    }

    #[test]
    fn test_generation_reference_counts_are_independent() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 104,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let (task_started_tx, task_started_rx) = mpsc::channel();
        let (task_release_tx, task_release_rx) = mpsc::channel();
        let client = custom_channel::DeviceClient::new(runner_id, || {}, || {});

        assert_eq!(client.external_count(), 1);
        assert_eq!(client.internal_count(), 1);

        let clone = client.clone();
        assert_eq!(client.external_count(), 2);
        drop(clone);
        assert_eq!(client.external_count(), 1);

        client
            .enqueue(move || {
                task_started_tx.send(()).unwrap();
                task_release_rx.recv().unwrap();
            })
            .unwrap();
        client.flush();
        task_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("task should start");
        assert_eq!(client.internal_count(), 2);

        task_release_tx.send(()).unwrap();
        client.shutdown().unwrap();
        assert_eq!(client.internal_count(), 0);
        assert_eq!(
            client.generation_status(),
            custom_channel::GenerationStatus::Closed
        );
    }

    #[test]
    fn test_concurrent_close_is_idempotent() {
        use std::sync::Barrier;

        const CLOSERS: usize = 4;
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 105,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let shutdown_count = Arc::new(AtomicUsize::new(0));
        let shutdown_count_runner = Arc::clone(&shutdown_count);
        let client = custom_channel::DeviceClient::new(
            runner_id,
            || {},
            move || {
                shutdown_count_runner.fetch_add(1, Ordering::SeqCst);
            },
        );
        let barrier = Arc::new(Barrier::new(CLOSERS));
        let mut closers = Vec::new();

        for _ in 0..CLOSERS {
            let client = client.clone();
            let barrier = Arc::clone(&barrier);
            closers.push(std::thread::spawn(move || {
                barrier.wait();
                client.shutdown().unwrap();
            }));
        }
        for closer in closers {
            closer.join().unwrap();
        }

        assert_eq!(shutdown_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            client.generation_status(),
            custom_channel::GenerationStatus::Closed
        );
    }

    #[test]
    fn test_close_drains_tasks_accepted_concurrently() {
        use std::sync::Barrier;

        const PRODUCERS: usize = 4;
        const ITERATIONS: u16 = 100;

        for iteration in 0..ITERATIONS {
            let runner_id = RunnerId {
                device: DeviceId {
                    type_id: 0,
                    index_id: 200 + iteration,
                },
                stage: DeviceServiceStage::Downstream,
            };
            let client = custom_channel::DeviceClient::new(runner_id, || {}, || {});
            let barrier = Arc::new(Barrier::new(PRODUCERS + 1));
            let accepted = Arc::new(AtomicUsize::new(0));
            let executed = Arc::new(AtomicUsize::new(0));
            let mut producers = Vec::new();

            for _ in 0..PRODUCERS {
                let client = client.clone();
                let barrier = Arc::clone(&barrier);
                let accepted = Arc::clone(&accepted);
                let executed = Arc::clone(&executed);
                producers.push(std::thread::spawn(move || {
                    barrier.wait();
                    if client
                        .enqueue(move || {
                            executed.fetch_add(1, Ordering::SeqCst);
                        })
                        .is_ok()
                    {
                        accepted.fetch_add(1, Ordering::SeqCst);
                    }
                }));
            }

            barrier.wait();
            client.shutdown().unwrap();
            for producer in producers {
                producer.join().unwrap();
            }

            assert_eq!(
                accepted.load(Ordering::SeqCst),
                executed.load(Ordering::SeqCst),
                "Iteration {iteration} lost an accepted task"
            );
        }
    }

    #[test]
    fn test_stale_generation_close_does_not_close_replacement() {
        let runner_id = RunnerId {
            device: DeviceId {
                type_id: 0,
                index_id: 106,
            },
            stage: DeviceServiceStage::Downstream,
        };
        let old_client = custom_channel::DeviceClient::new(runner_id, || {}, || {});
        let stale_client = old_client.clone();
        let old_generation = old_client.generation_id();
        old_client.shutdown().unwrap();

        let new_client = custom_channel::DeviceClient::new(runner_id, || {}, || {});
        let new_generation = new_client.generation_id();
        assert_ne!(old_generation, new_generation);

        stale_client.shutdown().unwrap();
        let (task_tx, task_rx) = mpsc::channel();
        new_client
            .enqueue(move || task_tx.send(()).unwrap())
            .unwrap();
        new_client.flush();
        task_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("stale close must not affect the replacement generation");
        new_client.shutdown().unwrap();
    }

    #[test]
    fn test_basic_execution_and_state_persistence() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 1,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        // Task 1: Increment the counter
        let res = handle
            .submit_blocking(|state| {
                state.counter += 1;
                state.counter
            })
            .unwrap();

        // Task 2: Increment again to ensure it's the same state instance
        let res2 = handle
            .submit_blocking(|state| {
                state.counter += 1;
                state.counter
            })
            .unwrap();

        assert_eq!(res, 1);
        assert_eq!(res2, 2);
    }

    #[test]
    fn test_services_share_generation_by_runner_id() {
        struct OtherService;
        struct UpstreamService;

        impl DeviceService for OtherService {
            fn init(_id: DeviceId) -> Self {
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }
        }

        impl DeviceService for UpstreamService {
            fn init(_id: DeviceId) -> Self {
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn stage() -> DeviceServiceStage {
                DeviceServiceStage::Upstream
            }
        }

        let device_id = DeviceId {
            type_id: 0,
            index_id: 107,
        };
        let first = ChannelDeviceHandle::<MockService>::new(device_id);
        let second = ChannelDeviceHandle::<OtherService>::new(device_id);
        let upstream = ChannelDeviceHandle::<UpstreamService>::new(device_id);

        assert_eq!(
            first.state.client.generation_id(),
            second.state.client.generation_id()
        );
        assert_ne!(
            first.state.client.generation_id(),
            upstream.state.client.generation_id()
        );

        let external_count = first.state.client.external_count();
        let lease = first.lease();
        assert_eq!(
            lease.generation_id(),
            Some(first.state.client.generation_id())
        );
        assert_eq!(first.state.client.external_count(), external_count + 1);
        drop(lease);
        assert_eq!(first.state.client.external_count(), external_count);
    }

    #[test]
    fn test_staged_shutdown_allows_upstream_to_initialize_downstream() {
        const CHILD_ENV: &str = "CUBECL_STAGED_SHUTDOWN_CHILD";

        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("test_staged_shutdown_allows_upstream_to_initialize_downstream")
                .arg("--test-threads=1")
                .env(CHILD_ENV, "1")
                .status()
                .unwrap();
            assert!(status.success(), "staged shutdown child process failed");
            return;
        }

        use std::sync::{Mutex as StdMutex, OnceLock};

        static EVENTS: OnceLock<mpsc::Sender<&'static str>> = OnceLock::new();
        static RELEASE_UPSTREAM: OnceLock<StdMutex<mpsc::Receiver<()>>> = OnceLock::new();

        struct DownstreamService;
        struct ExternalService;
        struct OtherUpstreamService;
        struct UpstreamService;

        impl DeviceService for DownstreamService {
            fn init(_id: DeviceId) -> Self {
                EVENTS.get().unwrap().send("downstream_init").unwrap();
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn shutdown(&mut self) {
                let rejected = catch_unwind(AssertUnwindSafe(|| {
                    ChannelDeviceHandle::<OtherUpstreamService>::new(DeviceId {
                        type_id: 9,
                        index_id: 4,
                    })
                }));
                assert!(rejected.is_err());
                EVENTS
                    .get()
                    .unwrap()
                    .send("downstream_to_upstream_rejected")
                    .unwrap();
                EVENTS.get().unwrap().send("downstream_shutdown").unwrap();
            }
        }

        impl DeviceService for ExternalService {
            fn init(_id: DeviceId) -> Self {
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }
        }

        impl DeviceService for OtherUpstreamService {
            fn init(_id: DeviceId) -> Self {
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn stage() -> DeviceServiceStage {
                DeviceServiceStage::Upstream
            }
        }

        impl DeviceService for UpstreamService {
            fn init(_id: DeviceId) -> Self {
                Self
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn shutdown(&mut self) {
                EVENTS.get().unwrap().send("upstream_start").unwrap();
                RELEASE_UPSTREAM
                    .get()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .recv()
                    .unwrap();

                let downstream = ChannelDeviceHandle::<DownstreamService>::new(DeviceId {
                    type_id: 9,
                    index_id: 2,
                });
                downstream
                    .submit_blocking(|_| {
                        EVENTS.get().unwrap().send("downstream_use").unwrap();
                    })
                    .unwrap();
                assert_eq!(
                    downstream.state.client.generation_status(),
                    custom_channel::GenerationStatus::Running
                );
                drop(downstream);

                let rejected = catch_unwind(AssertUnwindSafe(|| {
                    ChannelDeviceHandle::<OtherUpstreamService>::new(DeviceId {
                        type_id: 9,
                        index_id: 3,
                    })
                }));
                assert!(rejected.is_err());
                EVENTS.get().unwrap().send("same_stage_rejected").unwrap();
                EVENTS.get().unwrap().send("upstream_done").unwrap();
            }

            fn stage() -> DeviceServiceStage {
                DeviceServiceStage::Upstream
            }
        }

        let (events_tx, events_rx) = mpsc::channel();
        EVENTS.set(events_tx).unwrap();
        let (release_tx, release_rx) = mpsc::channel();
        RELEASE_UPSTREAM.set(StdMutex::new(release_rx)).unwrap();

        let upstream = ChannelDeviceHandle::<UpstreamService>::new(DeviceId {
            type_id: 9,
            index_id: 1,
        });
        let (shutdown_done_tx, shutdown_done_rx) = mpsc::channel();
        let shutdown_thread = std::thread::spawn(move || {
            shutdown_device_services().unwrap();
            shutdown_done_tx.send(()).unwrap();
        });

        assert_eq!(
            events_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "upstream_start"
        );

        let (external_done_tx, external_done_rx) = mpsc::channel();
        let external_thread = std::thread::spawn(move || {
            let handle = ChannelDeviceHandle::<ExternalService>::new(DeviceId {
                type_id: 9,
                index_id: 5,
            });
            external_done_tx.send(()).unwrap();
            handle
        });
        assert!(
            external_done_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "external initialization must remain parked during shutdown"
        );

        release_tx.send(()).unwrap();
        let events = (0..6)
            .map(|_| events_rx.recv_timeout(Duration::from_secs(1)).unwrap())
            .collect::<Vec<_>>();
        let position = |event| events.iter().position(|item| *item == event).unwrap();
        assert!(position("downstream_init") < position("downstream_use"));
        assert!(position("same_stage_rejected") < position("upstream_done"));
        assert!(position("downstream_to_upstream_rejected") < position("downstream_shutdown"));

        shutdown_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("shutdown must not count the parked external initializer");
        shutdown_thread.join().unwrap();
        external_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("external initialization should resume after shutdown");
        let external = external_thread.join().unwrap();

        drop((upstream, external));
        shutdown_device_services().unwrap();
    }

    #[test]
    fn test_on_runner_close_releases_service_borrow_before_shutdown() {
        struct ShutdownService {
            shutdown_tx: Option<mpsc::Sender<()>>,
        }

        impl DeviceService for ShutdownService {
            fn init(_id: DeviceId) -> Self {
                unreachable!()
            }

            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }

            fn shutdown(&mut self) {
                self.shutdown_tx.take().unwrap().send(()).unwrap();
            }
        }

        let device_id = DeviceId {
            type_id: 0,
            index_id: 108,
        };
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let handle = ChannelDeviceHandle::<ShutdownService>::insert(
            device_id,
            ShutdownService {
                shutdown_tx: Some(shutdown_tx),
            },
        )
        .unwrap();
        let shutdown_client = handle.state.client.clone();

        handle.submit(move |_| shutdown_client.shutdown().unwrap());
        handle.flush_queue();

        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("service shutdown should run after the task releases its borrow");
    }

    #[test]
    fn test_scoped_tasks_and_lifetimes() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 3,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        let local_val = 42; // This lives on the test stack

        // Test exclusive_scoped
        let result = handle.exclusive(|| local_val + 8).unwrap();

        assert_eq!(result, 50);

        // Test submit_blocking_scoped
        let result_mut = handle
            .submit_blocking(|state| {
                state.counter = local_val;
                state.counter
            })
            .unwrap();

        assert_eq!(result_mut, 42);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_buffer_flushing_at_limit() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 4,
        };

        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let completed_count = Arc::new(AtomicUsize::new(0));

        // We fill exactly CHANNEL_MAX_TASK
        // The last task should trigger a buffer swap/fetch.
        for _ in 0..CHANNEL_MAX_TASK {
            let counter = Arc::clone(&completed_count);
            handle.submit(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Wait for tasks to complete. Miri is very slow with this test, so sleeping fails here.
        let _ = handle.submit_blocking(|_| {});

        assert_eq!(completed_count.load(Ordering::SeqCst), 32);
    }

    #[test]
    fn test_manual_flush_for_partial_buffer() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 5,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let (tx, rx) = oneshot::channel();

        // Send only 1 task (buffer is not full)
        handle.submit(move |_| {
            tx.send(true).unwrap();
        });

        // This would hang forever if flush() didn't fill the buffer with no-ops
        handle.state.client.flush();

        let received = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("Task was not flushed and processed in time");
        assert!(received);
    }

    #[test]
    fn test_closure_captures_are_dropped_after_execution() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 6,
        };

        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        // This atomic counter will track how many times our "Spy" is dropped.
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy(Arc::clone(&drop_count));

        // We capture `spy` in the closure.
        // 1. It is moved into the Task buffer (or a Box if too large).
        // 2. The runner thread's shim uses ptr::read to move it into a local variable.
        // 3. The closure finishes, the local variable goes out of scope, and drop() is called.
        handle
            .submit_blocking(move |_state| {
                // Accessing spy here to ensure it's captured.
                let _ = &spy;
            })
            .expect("Task execution failed");

        // At this point, the blocking call has returned.
        // Because the shim moved the closure and let it go out of scope,
        // the drop count should be exactly 1.
        assert_eq!(
            drop_count.load(Ordering::SeqCst),
            1,
            "Capture was not dropped after execution"
        );
    }

    #[test]
    fn test_large_closure_uses_arena() {
        // Closure captures > 40 bytes (InlineSlot), forcing the arena path.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 7,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        let big_data = [42u8; 128]; // 128 bytes > 40 byte inline limit
        let result = handle
            .submit_blocking(move |_state| {
                // Use big_data to prevent it from being optimized away.
                big_data[0] + big_data[127]
            })
            .unwrap();

        assert_eq!(result, 84);
    }

    #[test]
    fn test_extra_large_closure_uses_box() {
        // Closure captures > 4096 bytes (GLOBAL_TASK_MAX_SIZE), forcing the Box fallback.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 8,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        let huge_data = [7u8; 8192]; // 8KB > 4096 byte arena limit
        let result = handle
            .submit_blocking(move |_state| huge_data[0] + huge_data[8191])
            .unwrap();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_large_closure_drop_is_called() {
        // Verify that Drop runs correctly for closures stored in the arena.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 9,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy {
            counter: Arc<AtomicUsize>,
            _padding: [u8; 128], // Force arena path (> 40 bytes)
        }
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy {
            counter: Arc::clone(&drop_count),
            _padding: [0; 128],
        };

        handle
            .submit_blocking(move |_state| {
                let _ = &spy;
            })
            .unwrap();

        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    /// Concurrent callers racing on the same `(DeviceId, TypeId)` must share a single
    /// `S::init` invocation.
    #[test]
    fn test_init_runs_exactly_once_under_contention() {
        use alloc::vec::Vec;
        use std::sync::Barrier;
        use std::sync::atomic::AtomicUsize;
        use std::thread;

        static INIT_CALLS: AtomicUsize = AtomicUsize::new(0);

        struct CountingService;
        impl DeviceService for CountingService {
            fn init(_: DeviceId) -> Self {
                INIT_CALLS.fetch_add(1, Ordering::SeqCst);
                CountingService
            }
            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }
        }

        INIT_CALLS.store(0, Ordering::SeqCst);

        const THREADS: usize = 4;
        // Unique device_id so the global `CHANNELS` entry is independent of other tests.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 77,
        };

        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                ChannelDeviceHandle::<CountingService>::new(device_id)
            }));
        }
        let generation_ids = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().state.client.generation_id())
            .collect::<Vec<_>>();

        assert_eq!(
            INIT_CALLS.load(Ordering::SeqCst),
            1,
            "CountingService::init must run exactly once across {THREADS} racing callers"
        );
        assert!(
            generation_ids.windows(2).all(|ids| ids[0] == ids[1]),
            "Concurrent initialization must reuse one generation"
        );
    }

    /// If the task panics, `submit_blocking` must return `Err`, drop the
    /// task's captures exactly once, and leave the channel usable.
    #[test]
    fn test_submit_blocking_panic_drops_and_returns_err() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 10,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy(Arc::clone(&drop_count));
        let result = handle.submit_blocking(move |_state| {
            let _ = &spy;
            panic!("boom");
        });

        assert!(result.is_err(), "panicking task must return Err");
        assert_eq!(
            drop_count.load(Ordering::SeqCst),
            1,
            "captures must be dropped exactly once on panic"
        );

        // Channel survives: next task still runs.
        let ok = handle.submit_blocking(|state| state.counter).unwrap();
        assert_eq!(ok, 0);
    }

    /// Same guarantees for `exclusive`.
    #[test]
    fn test_exclusive_panic_drops_and_returns_err() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 11,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy(Arc::clone(&drop_count));
        let result: Result<(), _> = handle.exclusive(move || {
            let _ = &spy;
            panic!("boom");
        });

        assert!(result.is_err());
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
        let ok = handle.exclusive(|| 7).unwrap();
        assert_eq!(ok, 7);
    }

    /// A closure that spills to the arena (size > 48) and carries the maximum arena
    /// alignment (64) must be stored and executed soundly.
    #[test]
    fn test_task_init_arena_aligned_closure() {
        use super::task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task};

        #[repr(align(64))]
        #[derive(Clone, Copy)]
        struct A64 {
            data: [u8; 128],
        }

        // Mirror `TaskBuffer::new`: a 64-aligned 4KB region per slot.
        let mut arena = alloc::boxed::Box::new(ArenaSlot {
            data: [0u8; GLOBAL_TASK_MAX_SIZE],
        });
        let arena_ptr = arena.data.as_mut_ptr();
        let mut task = Task::new(arena_ptr);

        let data = A64 { data: [0xCD; 128] };
        task.init(move || {
            let d = core::hint::black_box(data);
            let _: usize = d.data.iter().map(|&b| b as usize).sum();
        });
        task.run();
    }

    /// A closure whose alignment exceeds the arena slot alignment must take the
    /// boxed fallback.
    #[test]
    fn test_task_init_extremely_over_aligned_closure_uses_box() {
        use super::task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task};

        #[repr(align(256))]
        #[derive(Clone, Copy)]
        struct A256 {
            data: [u8; 256],
        }

        let mut arena = alloc::boxed::Box::new(ArenaSlot {
            data: [0u8; GLOBAL_TASK_MAX_SIZE],
        });
        let arena_ptr = arena.data.as_mut_ptr();
        let mut task = Task::new(arena_ptr);

        let data = A256 { data: [0xAA; 256] };
        task.init(move || {
            let d = core::hint::black_box(data);
            let _: usize = d.data.iter().map(|&b| b as usize).sum();
        });
        task.run();
    }

    #[test]
    fn test_task_drop_discards_closure_captures_without_running() {
        use super::task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task};

        struct DropSpy<const N: usize> {
            counter: Arc<AtomicUsize>,
            _padding: [u8; N],
        }

        impl<const N: usize> Drop for DropSpy<N> {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn assert_discarded<F: FnOnce() + Send + 'static>(
            task_fn: F,
            drop_count: &Arc<AtomicUsize>,
            run_count: &Arc<AtomicUsize>,
        ) {
            let mut arena = alloc::boxed::Box::new(ArenaSlot {
                data: [0u8; GLOBAL_TASK_MAX_SIZE],
            });
            let mut task = Task::new(arena.data.as_mut_ptr());
            task.init(task_fn);
            drop(task);

            assert_eq!(drop_count.load(Ordering::SeqCst), 1);
            assert_eq!(run_count.load(Ordering::SeqCst), 0);
        }

        fn run_case<const N: usize>() {
            let drop_count = Arc::new(AtomicUsize::new(0));
            let run_count = Arc::new(AtomicUsize::new(0));
            let spy = DropSpy {
                counter: Arc::clone(&drop_count),
                _padding: [0; N],
            };
            let run_count_task = Arc::clone(&run_count);

            assert_discarded(
                move || {
                    let _ = &spy;
                    run_count_task.fetch_add(1, Ordering::SeqCst);
                },
                &drop_count,
                &run_count,
            );
        }

        run_case::<0>();
        run_case::<128>();
        run_case::<8192>();
    }
}
