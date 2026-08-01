use crate::compute::uninit_vec;
use cubecl_common::backtrace::BackTrace;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::DriverError;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
enum AllocationKind {
    Async,
    Sync,
}

/// Buffer storage for NVIDIA GPUs.
///
/// This struct manages memory resources for CUDA kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    memory: HashMap<StorageId, (cudarc::driver::sys::CUdeviceptr, AllocationKind)>,
    deallocations: Vec<StorageId>,
    ptr_bindings: PtrBindings,
    stream: cudarc::driver::sys::CUstream,
    mem_alignment: usize,
    /// An event on [`stream`](Self::stream), re-recorded after every
    /// stream-ordered allocation.
    ///
    /// `cuMemAllocAsync` serves from a mempool and is *stream-ordered*: the
    /// address comes back to the host immediately, but the pages behind it are
    /// only guaranteed mapped to work ordered after the allocation on the
    /// allocating stream. Recycling makes that concrete — a page freed by an
    /// earlier `cuMemFreeAsync` can be handed straight back out, so a stream
    /// that dereferences it without ordering against the allocation may find no
    /// valid mapping at all. That faults as an illegal address rather than
    /// reading stale bytes.
    ///
    /// Ordering the buffer's *contents* across streams is not enough, because
    /// the hazard is the mapping, not the data. Consumers on another stream
    /// wait on this event before first use; see
    /// [`allocation_event`](Self::allocation_event).
    ///
    /// `None` until the first stream-ordered allocation. Allocations that fell
    /// back to `cuMemAlloc` need no ordering — that memory is context-wide
    /// valid on return — so a storage that only ever took the fallback never
    /// creates an event.
    alloc_event: Option<cudarc::driver::sys::CUevent>,
}

/// A GPU memory resource allocated for CUDA using [`GpuStorage`].
#[derive(Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: u64,
    /// The CUDA binding pointer.
    pub binding: *mut std::ffi::c_void,
    /// The size of the resource.
    pub size: u64,
}

impl GpuResource {
    /// Creates a new [`GpuResource`].
    pub fn new(ptr: u64, binding: *mut std::ffi::c_void, size: u64) -> Self {
        Self { ptr, binding, size }
    }
}

impl GpuStorage {
    /// Creates a new [`GpuStorage`] instance for the specified CUDA stream.
    ///
    /// # Arguments
    ///
    /// * `mem_alignment` - The memory alignment requirement in bytes.
    pub fn new(mem_alignment: usize, stream: cudarc::driver::sys::CUstream) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(),
            stream,
            mem_alignment,
            alloc_event: None,
        }
    }

    /// The event another stream must wait on before it first dereferences
    /// memory this storage allocated. `None` when nothing stream-ordered has
    /// been allocated yet.
    ///
    /// Waiting on the latest recorded state is deliberately conservative: it
    /// orders the consumer after *every* allocation made so far, which includes
    /// whichever one it actually cares about. Re-recording later does not
    /// disturb a wait that was already enqueued.
    pub fn allocation_event(&self) -> Option<cudarc::driver::sys::CUevent> {
        self.alloc_event
    }

    /// Records the allocation event on this storage's stream, creating it on
    /// first use.
    fn record_allocation(&mut self) {
        let event = match self.alloc_event {
            Some(event) => event,
            None => {
                // Timing is disabled: this event exists purely to order streams,
                // and the timing machinery costs more to record.
                let event = match cudarc::driver::result::event::create(
                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                ) {
                    Ok(event) => event,
                    Err(err) => {
                        // Without the event a cross-stream consumer cannot order
                        // against this allocation. Surfacing that as an
                        // allocation failure would be worse than the status quo,
                        // so log and carry on with the pre-existing behaviour.
                        log::error!("Couldn't create the CUDA allocation-ordering event: {err}");
                        return;
                    }
                };
                self.alloc_event = Some(event);
                event
            }
        };

        // SAFETY: `event` was created above (or on an earlier call) and
        // `self.stream` is a valid, initialized CUDA stream.
        if let Err(err) = unsafe { cudarc::driver::result::event::record(event, self.stream) } {
            log::error!("Couldn't record the CUDA allocation-ordering event: {err}");
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    fn perform_deallocations(&mut self) {
        self.deallocations
            .drain(..)
            .filter_map(|id| self.memory.remove(&id))
            // SAFETY: Each `ptr` was obtained from a prior `malloc_async` or `malloc_sync`
            // call and has not been freed yet. The deallocation method matches the allocation kind.
            .for_each(|(ptr, kind)| unsafe {
                match kind {
                    AllocationKind::Async => {
                        let _ = cudarc::driver::result::free_async(ptr, self.stream);
                    }
                    AllocationKind::Sync => {
                        if let Err(e) = cudarc::driver::result::free_sync(ptr) {
                            eprintln!("CUDA free error: {}", e);
                        }
                    }
                }
            });
    }
}

// SAFETY: `GpuResource` contains CUDA device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}
// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw CUDA pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStorage").finish()
    }
}

/// Manages active CUDA buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<cudarc::driver::sys::CUdeviceptr>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [`PtrBindings`] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            // SAFETY: `CUdeviceptr` is a `u64`, valid for any bit pattern. All slots are
            // written via `register()` before being read, so uninitialized values are never observed.
            slots: unsafe { uninit_vec(crate::device::CUDA_MAX_BINDINGS as usize) },
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The CUDA device pointer to register.
    ///
    /// # Returns
    ///
    /// A reference to the registered pointer.
    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr_ref = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor when the ring buffer is full.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr_ref
    }
}

impl ComputeStorage for GpuStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let (ptr, _) = self
            .memory
            .get(&handle.id)
            .expect("Storage handle not found");

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        )
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        // SAFETY: Calling CUDA driver FFI to allocate device memory. First tries async
        // allocation on the stream; falls back to synchronous allocation if that fails.
        // The returned pointer is stored in `self.memory` and freed on deallocation.
        let ptr = unsafe { cudarc::driver::result::malloc_async(self.stream, size as usize) };
        let (ptr, kind) = match ptr {
            Ok(ptr) => {
                // Stream-ordered: the pages are only guaranteed mapped for work
                // ordered after this point on `self.stream`. Publish that point
                // so a consumer on another stream can wait for it.
                self.record_allocation();
                (ptr, AllocationKind::Async)
            }
            Err(_) => unsafe {
                match cudarc::driver::result::malloc_sync(size as usize) {
                    Ok(ptr) => (ptr, AllocationKind::Sync),
                    Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                        return Err(IoError::BufferTooBig {
                            size,
                            backtrace: BackTrace::capture(),
                        });
                    }
                    Err(other) => {
                        return Err(IoError::Unknown {
                            description: format!("CUDA allocation error: {other}"),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
            },
        };

        self.memory.insert(id, (ptr, kind));
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn flush(&mut self) {
        self.perform_deallocations();
    }
}
