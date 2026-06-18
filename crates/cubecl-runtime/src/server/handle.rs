use cubecl_common::{
    device_handle::{DeviceGenerationId, DeviceLease},
    stream_id::StreamId,
};
use cubecl_zspace::{Shape, Strides};

use crate::{
    memory_management::{ManagedMemoryBinding, ManagedMemoryHandle},
    server::CopyDescriptor,
};

/// Server handle containing the [memory handle](crate::server::Handle).
pub struct Handle {
    /// Memory handle.
    pub memory: ManagedMemoryHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
    lease: Option<DeviceLease>,
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.memory)
            .field("offset_start", &self.offset_start)
            .field("offset_end", &self.offset_end)
            .field("stream", &self.stream)
            .field("size", &self.size)
            .field("generation_id", &self.generation_id())
            .finish()
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
            lease: self.lease.clone(),
        }
    }
}

impl Handle {
    /// Creates a new handle of the given size.
    pub fn from_memory(id: ManagedMemoryHandle, stream: StreamId, size: u64) -> Self {
        Self {
            memory: id,
            offset_start: None,
            offset_end: None,
            stream,
            size,
            lease: None,
        }
    }
    /// Creates a new handle of the given size.
    pub fn new(stream: StreamId, size: u64) -> Self {
        Self {
            memory: ManagedMemoryHandle::new(),
            offset_start: None,
            offset_end: None,
            stream,
            size,
            lease: None,
        }
    }
    /// Checks whether the handle can be mutated in-place without affecting other computation.
    pub fn can_mut(&self) -> bool {
        self.memory.can_mut()
    }

    /// Returns whether both handles alias the same managed memory.
    pub fn is_alias_of(&self, other: &Self) -> bool {
        self.memory.is_alias_of(&other.memory)
    }

    /// Returns the [`Binding`] corresponding to the current handle.
    pub fn binding(self) -> Binding {
        Binding::new(
            self.memory.binding(),
            self.offset_start,
            self.offset_end,
            self.stream,
            self.size,
            self.lease,
        )
    }

    /// Add to the current offset in bytes.
    pub fn offset_start(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_start {
            *val += offset;
        } else {
            self.offset_start = Some(offset);
        }

        self
    }
    /// Add to the current offset in bytes.
    pub fn offset_end(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_end {
            *val += offset;
        } else {
            self.offset_end = Some(offset);
        }

        self
    }

    /// Convert the [handle](Handle) into a [binding](Binding) with shape and stride metadata.
    pub fn copy_descriptor(
        self,
        shape: Shape,
        strides: Strides,
        elem_size: usize,
    ) -> CopyDescriptor {
        CopyDescriptor {
            shape,
            strides,
            elem_size,
            handle: self.binding(),
        }
    }
    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the device-runner generation retained by this handle, when applicable.
    pub fn generation_id(&self) -> Option<DeviceGenerationId> {
        self.lease.as_ref().and_then(DeviceLease::generation_id)
    }

    pub(crate) fn set_lease(&mut self, lease: DeviceLease) {
        self.lease = Some(lease);
    }

    pub(crate) fn clear_lease(&mut self) {
        self.lease = None;
    }
}

/// A binding represents a [Handle] that is bound to managed memory.
///
/// The memory used is known by the compute server.
/// A binding is only valid after being initlized with [`super::ComputeServer::initialize_bindings`]
///
/// # Notes
///
/// A binding is detached from a [`Handle`], meaning that is won't affect [`Handle::can_mut`].
#[derive(Clone, Debug)]
pub struct Binding {
    /// The id of the handle the binding is bound to.
    pub memory: ManagedMemoryBinding,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub size: u64,
    lease: Option<DeviceLease>,
}

impl Binding {
    /// Creates a binding for managed memory.
    pub fn new(
        memory: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
        stream: StreamId,
        size: u64,
        lease: Option<DeviceLease>,
    ) -> Self {
        Self {
            memory,
            offset_start,
            offset_end,
            stream,
            size,
            lease,
        }
    }

    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the device-runner generation retained by this binding, when applicable.
    pub fn generation_id(&self) -> Option<DeviceGenerationId> {
        self.lease.as_ref().and_then(DeviceLease::generation_id)
    }

    pub(crate) fn clear_lease(&mut self) {
        self.lease = None;
    }
}
