use crate::memory_management::MemoryHandle;
use alloc::{sync::Arc, vec::Vec};

/// Managed Memory handle
#[derive(Debug)]
pub struct ManagedMemoryHandle {
    descriptor: Arc<ManagedMemoryDescriptor>,
    // Holds only the reference counts of the handle.
    handle_count: Arc<()>,
}

/// Binding of a memory handle
#[derive(Debug)]
pub struct ManagedMemoryBinding {
    descriptor: Arc<ManagedMemoryDescriptor>,
}

/// A list of bindings that are shared across multiple streams.
#[derive(Debug, Default)]
pub struct SharedMemoryBindings {
    /// The bindings.
    pub bindings: Vec<ManagedMemoryBinding>,
}

impl Clone for ManagedMemoryHandle {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
            handle_count: self.handle_count.clone(),
        }
    }
}

/// Managed memory descriptor.
///
/// The location is behind a `spin::Mutex` for interior mutability: multiple
/// handles share the same descriptor via `Arc`, yet the memory management
/// system needs to update the location after creation (e.g. during
/// `reserve` / `bind`).
///
/// This used to be a `Cell` plus an `unsafe impl Sync`, on the claim that only
/// the device thread ever touches a location. That claim does not hold: a
/// handle travels across streams (a binding resolved by another stream's
/// cursor lookup, a buffer shared between streams), so a read can genuinely
/// race a `reserve`/`bind` on another thread. It was survivable only by
/// accident — `MemoryLocation` used to be 8 bytes with `page`, `pool` and
/// `init` packed into a single word, so a racing reader saw the location move
/// as a unit. Once `page` grew into its own word, a reader could observe the
/// updated `init` alongside a `page` not yet written — a location reading
/// "initialized" while still carrying the null page key.
///
/// A lock costs an uncontended atomic swap per access, on a path that already
/// does refcount traffic per binding. In exchange the location is never
/// observed half-written, whatever its size, and the `unsafe impl Sync` is
/// gone: `spin::Mutex<MemoryLocation>` is `Sync` on its own merits.
pub(crate) struct ManagedMemoryDescriptor {
    pub(crate) id: ManagedMemoryId,
    location: spin::Mutex<MemoryLocation>,
}

impl core::fmt::Debug for ManagedMemoryDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ManagedMemoryDescriptor")
            .field("id", &self.id)
            .field("location", &self.location())
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
/// Managed memory unique identifier.
pub struct ManagedMemoryId {
    pub(crate) value: usize,
}

impl PartialEq for ManagedMemoryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ManagedMemoryDescriptor {}

slotmap::new_key_type! {
    /// Identifies a page in a memory pool: the slot the page occupies, plus the
    /// generation that says *which* occupant of that slot it is.
    ///
    /// Pools keep their pages in a [`SlotMap`](slotmap::SlotMap) rather than a
    /// `Vec`, so a page keeps its key for as long as it lives — freeing another
    /// page never renumbers it. That matters because this key is cached in the
    /// descriptor of every handle pointing into the page, including descriptors
    /// the pool can no longer reach (one orphaned by a `bind`, kept alive by
    /// another stream's binding). Renumbering can only fix the descriptors the
    /// pool still reaches; the rest would be left naming whatever moved into
    /// their old position. The generation closes the remaining gap: once a page
    /// is freed, a key pointing at it fails to resolve rather than aliasing the
    /// next page to land in the slot.
    pub(crate) struct PageKey;
}

#[derive(Clone, Copy, Debug)]
/// Defines where the [`ManagedMemoryId`] is located.
pub(crate) struct MemoryLocation {
    /// The memory pool index in the global memory management.
    pub pool: u8,
    /// The memory page in a memory pool.
    pub page: PageKey,
    /// The memory slice index in a memory page.
    pub slice: u32,
    /// Whether the memory location is known/initialized.
    pub init: u8,
}

impl ManagedMemoryDescriptor {
    /// Update the memory location for the given [`ManagedMemoryId`].
    pub(crate) fn update_location(&self, location: MemoryLocation) {
        *self.location.lock() = location;
    }

    /// Update only the slice position for the given [`ManagedMemoryId`].
    pub(crate) fn update_slice(&self, slice: u32) {
        self.location.lock().slice = slice;
    }

    /// Retrieves the current location.
    pub(crate) fn location(&self) -> MemoryLocation {
        *self.location.lock()
    }

    pub(crate) fn slice(&self) -> usize {
        self.location().slice as usize
    }

    pub(crate) fn page(&self) -> PageKey {
        self.location().page
    }
}

impl MemoryLocation {
    /// Creates the base location every slice of `pool` is stamped from: the
    /// pool position, with the page and slice filled in as they're handed out.
    pub(crate) fn base(pool: u8) -> Self {
        Self {
            pool,
            page: PageKey::default(),
            slice: 0,
            init: 1,
        }
    }

    /// Creates a new uninitialized memory location.
    pub(crate) fn uninit() -> Self {
        Self {
            pool: 0,
            // The null key, which no live page ever equals.
            page: PageKey::default(),
            slice: 0,
            init: 0,
        }
    }
}

impl ManagedMemoryHandle {
    /// Creates a new managed memory handle.
    pub fn new() -> Self {
        let value = Self::gen_id();

        Self {
            descriptor: Arc::new(ManagedMemoryDescriptor {
                id: ManagedMemoryId { value },
                location: spin::Mutex::new(MemoryLocation::uninit()),
            }),
            handle_count: Arc::new(()),
        }
    }

    /// Retrieves the descriptor for the current handle.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// Return whether the current handle can be modified in-place.
    pub fn can_mut(&self) -> bool {
        Arc::strong_count(&self.handle_count) <= 2
    }

    /// Return whether the current handle is free.
    pub fn is_free(&self) -> bool {
        Arc::strong_count(&self.descriptor) <= 1
    }

    /// Returns the binding for the current handle.
    pub fn binding(self) -> ManagedMemoryBinding {
        ManagedMemoryBinding {
            descriptor: self.descriptor.clone(),
        }
    }

    fn gen_id() -> usize {
        static COUNTER: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if value == usize::MAX {
            core::panic!("Memory ID overflowed");
        }
        value
    }
}

impl ManagedMemoryBinding {
    /// Retrieves the descriptor for the current binding.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }
}

impl Default for ManagedMemoryHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ManagedMemoryBinding {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
        }
    }
}

impl MemoryHandle<ManagedMemoryBinding> for ManagedMemoryHandle {
    fn can_mut(&self) -> bool {
        self.can_mut()
    }

    fn binding(self) -> ManagedMemoryBinding {
        self.binding()
    }
}

impl SharedMemoryBindings {
    /// Clears the shared bindings list.
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Returns true if the shared bindings list is empty.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Push a memory binding to the list of shared bindings.
    pub fn push(&mut self, binding: ManagedMemoryBinding) {
        self.bindings.push(binding)
    }
}

impl cubecl_common::pool::Reclaim for SharedMemoryBindings {
    fn reclaim(&mut self) {
        self.clear();
    }
}

/// Calculates a best-effort heuristic for the alignment of row-aligned tensors.
/// Prefers contiguous alignments for unit dimensions, 16-byte minimum alignment for non-unit,
/// scaling with input size up to `buffer_align`.
pub fn optimal_align(shape: usize, elem_size: usize, buffer_align: usize) -> usize {
    if shape == 1 {
        elem_size
    } else {
        (shape * elem_size)
            .next_power_of_two()
            .clamp(16, buffer_align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_id_mutability() {
        let handle1 = ManagedMemoryHandle::new();
        handle1.descriptor().update_slice(4);
        assert_eq!(handle1.descriptor().slice(), 4);

        let handle2 = ManagedMemoryHandle::new();
        handle2
            .clone()
            .descriptor()
            .update_location(handle1.descriptor().location());
        assert_eq!(handle2.descriptor().slice(), 4);
    }

    #[test]
    fn test_location_visible_through_shared_arc() {
        let handle = ManagedMemoryHandle::new();
        let handle2 = handle.clone();

        let mut location = MemoryLocation::base(1);
        location.slice = 3;
        handle.descriptor().update_location(location);

        assert_eq!(handle2.descriptor().location().pool, 1);
        assert_eq!(handle2.descriptor().location().slice, 3);
        assert_eq!(handle2.descriptor().location().init, 1);

        handle.descriptor().update_slice(42);
        assert_eq!(handle2.descriptor().slice(), 42);
    }

    /// A location must never be observed half-written.
    ///
    /// Handles cross stream boundaries, so a reader (another stream resolving a
    /// binding) genuinely races the `reserve`/`bind` that stamps the location.
    /// While the location lived in a `Cell`, that race was only survivable
    /// because the struct happened to fit in one word; a reader could otherwise
    /// see the `init` byte of one write next to the `page` of another — an
    /// "initialized" location carrying the null page key, which resolves to no
    /// page at all.
    #[test_log::test]
    #[cfg(feature = "std")]
    fn location_is_never_observed_half_written() {
        extern crate std;
        use std::{sync::atomic::AtomicBool, sync::atomic::Ordering, thread, vec::Vec};

        let handle = ManagedMemoryHandle::new();
        let stop = std::sync::Arc::new(AtomicBool::new(false));

        // Two real page keys, from a slot map, so `page` differs between the
        // two locations — that field is the one that grew into its own word.
        let mut pages = slotmap::SlotMap::<PageKey, ()>::with_key();
        let key_a = pages.insert(());
        let key_b = pages.insert(());

        // Two complete, valid locations. Any read must return one of them
        // verbatim — never a field from one beside a field from the other.
        let mut a = MemoryLocation::base(1);
        a.page = key_a;
        let mut b = MemoryLocation::base(2);
        b.page = key_b;
        b.slice = 7;

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let reader = handle.clone();
                let stop = stop.clone();
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let seen = reader.descriptor().location();
                        // Every field must belong to the same write. In
                        // particular a location claiming `init` must carry its
                        // pool's real page key, never the null one.
                        let torn = match seen.pool {
                            0 => seen.init != 0 || seen.page != PageKey::default(),
                            1 => seen.init != 1 || seen.page != key_a || seen.slice != 0,
                            2 => seen.init != 1 || seen.page != key_b || seen.slice != 7,
                            _ => true,
                        };
                        assert!(!torn, "observed a half-written location: {seen:?}");
                    }
                })
            })
            .collect();

        for i in 0..2_000_000 {
            handle
                .descriptor()
                .update_location(if i % 2 == 0 { a } else { b });
        }

        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            reader.join().unwrap();
        }
    }
}
