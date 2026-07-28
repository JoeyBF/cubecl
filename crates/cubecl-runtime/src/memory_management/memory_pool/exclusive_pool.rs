use crate::{
    memory_management::{BytesFormat, MemoryLocation, MemoryUsage},
    server::IoError,
    storage::{ComputeStorage, StorageUtilization},
};

use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;
use slotmap::SlotMap;

use super::{
    ManagedMemoryBinding, ManagedMemoryHandle, MemoryPool, PageKey, Slice, calculate_padding,
    page_not_found,
};

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Only one slice is supported per page, due to the limitations in WGPU where each buffer should only bound with
///   either read only or `read_write` slices but not a mix of both.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
pub struct ExclusiveMemoryPool {
    /// Pages are keyed rather than positional: a page keeps its [`PageKey`] for
    /// its whole life, so freeing one never invalidates the keys cached in the
    /// descriptors pointing at the others. See [`PageKey`].
    pages: SlotMap<PageKey, MemoryPage>,
    alignment: u64,
    dealloc_period: u64,
    last_dealloc_check: u64,
    max_alloc_size: u64,
    cur_avg_size: f64,
    location_base: MemoryLocation,
}

impl core::fmt::Display for ExclusiveMemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            " - Exclusive Pool max_alloc_size={}\n",
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for page in self.pages.values() {
            let is_free = page.slice.is_free();
            let size = BytesFormat::new(page.slice.effective_size());

            f.write_fmt(format_args!("   - Page {size} is_free={is_free}\n"))?;
        }

        if !self.pages.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

const SIZE_AVG_DECAY: f64 = 0.01;

// How many times to find the allocation 'free' before deallocating it.
const ALLOC_AFTER_FREE: u32 = 5;

struct MemoryPage {
    slice: Slice,
    alloc_size: u64,
    free_count: u32,
}

impl ExclusiveMemoryPool {
    pub(crate) fn new(
        max_alloc_size: u64,
        alignment: u64,
        dealloc_period: u64,
        pool_pos: u8,
    ) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(max_alloc_size % alignment, 0);

        Self {
            pages: SlotMap::with_key(),
            alignment,
            dealloc_period,
            last_dealloc_check: 0,
            max_alloc_size,
            cur_avg_size: max_alloc_size as f64 / 2.0,
            location_base: MemoryLocation::base(pool_pos),
        }
    }

    /// Finds a free page that can contain the given size
    /// Returns a slice on that page if successful.
    fn get_free_page(&mut self, size: u64) -> Option<&mut MemoryPage> {
        // Return the smallest free page that fits.
        self.pages
            .values_mut()
            .filter(|page| page.alloc_size >= size && page.slice.is_free())
            .min_by_key(|page| page.free_count)
    }

    fn alloc_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<PageKey, IoError> {
        let alloc_size = (self.cur_avg_size as u64)
            .max(size)
            .next_multiple_of(self.alignment);

        let storage = storage.alloc(alloc_size)?;

        let padding = calculate_padding(size, self.alignment);
        let mut slice = Slice::new(storage, padding);

        // Return a smaller part of the slice. By construction, we only ever
        // get a page with a big enough size, so this is ok to do.
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        slice.padding = padding;

        Ok(self.pages.insert(MemoryPage {
            slice,
            alloc_size,
            // Start the allocation at 'almost ready to free'. Every use will decrement this.
            // This means allocations start as "suspected as unused" and over time will be kept for longer.
            free_count: ALLOC_AFTER_FREE - 1,
        }))
    }
}

impl MemoryPool for ExclusiveMemoryPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        self.cur_avg_size =
            self.cur_avg_size * (1.0 - SIZE_AVG_DECAY) + size as f64 * SIZE_AVG_DECAY;

        let padding = calculate_padding(size, self.alignment);

        self.get_free_page(size).map(|page| {
            // Return a smaller part of the slice. By construction, we only ever
            // get a page with a big enough size, so this is ok to do.
            page.slice.storage.utilization = StorageUtilization { offset: 0, size };
            page.slice.padding = padding;
            page.free_count = page.free_count.saturating_sub(1);
            page.slice.handle.clone()
        })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        }

        let key = self.alloc_page(storage, size)?;
        let handle = self.pages[key].slice.handle.clone();
        let mut location = self.location_base;
        location.page = key;
        handle.descriptor().update_location(location);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .pages
            .values()
            .filter(|page| !page.slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices
                .iter()
                .map(|page| page.slice.storage.size())
                .sum(),
            bytes_padding: used_slices.iter().map(|page| page.slice.padding).sum(),
            bytes_reserved: self.pages.values().map(|page| page.alloc_size).sum(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        // Check such that an alloc is free after at most dealloc_period.
        let check_period = self.dealloc_period / (ALLOC_AFTER_FREE as u64);

        if explicit || alloc_nr - self.last_dealloc_check >= check_period {
            self.last_dealloc_check = alloc_nr;

            // Surviving pages keep their key, so a page that was live before the
            // cleanup is still reachable through every descriptor that named it —
            // there is nothing to renumber, and no way for a descriptor to end up
            // naming a page it never pointed at.
            self.pages.retain(|_, page| {
                if !page.slice.is_free() {
                    return true;
                }

                page.free_count += 1;

                // If free found is sufficiently high (ie. we've seen this alloc as free multiple times,
                // without it being used in the meantime), deallocate it.
                if page.free_count >= ALLOC_AFTER_FREE || explicit {
                    storage.dealloc(page.slice.storage.id);
                    return false;
                }

                true
            });
        }
    }

    fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let id_old = old.descriptor();
        let key = id_old.page();
        let page = self.pages.get_mut(key).ok_or_else(|| page_not_found(key))?;
        new.descriptor().update_location(id_old.location());

        page.slice.handle = new;
        page.slice.cursor = cursor;

        Ok(())
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let key = binding.descriptor().page();
        let page = self.pages.get(key).ok_or_else(|| page_not_found(key))?;

        Ok(&page.slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    /// Hands `pool`'s freshly allocated page a new identity and drops it, so the
    /// page reads as free while the returned handle and binding keep naming it.
    ///
    /// This is the shape the bug needs: a descriptor the pool can no longer
    /// reach (in production, a handle orphaned by a `bind` and kept alive by
    /// another stream's binding) that still carries the page's address.
    fn orphan(
        pool: &mut ExclusiveMemoryPool,
        storage: &mut BytesStorage,
        size: u64,
    ) -> (ManagedMemoryHandle, ManagedMemoryBinding) {
        let reserved = pool.alloc(storage, size).unwrap();
        let orphan = reserved.clone();
        let assigned = ManagedMemoryHandle::new();

        pool.bind(reserved, assigned.clone(), 0).unwrap();
        drop(assigned);

        let binding = orphan.clone().binding();
        (orphan, binding)
    }

    #[test_log::test]
    fn stale_binding_does_not_alias_a_recycled_page() {
        let mut storage = BytesStorage::default();
        let mut pool = ExclusiveMemoryPool::new(1024, 32, u64::MAX, 0);

        let (_orphan, stale) = orphan(&mut pool, &mut storage, 1024);
        pool.cleanup(&mut storage, 0, true);
        assert!(pool.find(&stale).is_err(), "the page was deallocated");

        // A later allocation takes the vacated slot. The stale binding named the
        // slot's *previous* occupant, so it must not follow the slot to its new
        // one — the pool has to catch this itself, rather than leaning on the
        // identity check `MemoryManagement::find` applies afterwards.
        let _next = pool.alloc(&mut storage, 1024).unwrap();
        assert!(matches!(pool.find(&stale), Err(IoError::NotFound { .. })));
    }

    #[test_log::test]
    fn surviving_pages_keep_their_key_across_cleanup() {
        let mut storage = BytesStorage::default();
        let mut pool = ExclusiveMemoryPool::new(1024, 32, u64::MAX, 0);

        let first = pool.alloc(&mut storage, 1024).unwrap();
        let kept = pool.alloc(&mut storage, 1024).unwrap();
        let last = pool.alloc(&mut storage, 1024).unwrap();

        let binding = kept.clone().binding();
        let location = kept.descriptor().location();
        drop(first);
        drop(last);

        // Freeing the pages on either side must leave the survivor exactly where
        // it was: no renumbering, so every descriptor naming it stays valid —
        // including any the pool can't reach to update.
        pool.cleanup(&mut storage, 0, true);

        assert_eq!(pool.find(&binding).unwrap().storage.size(), 1024);
        assert_eq!(kept.descriptor().location().page, location.page);
        assert_eq!(pool.get_memory_usage().bytes_reserved, 1024);
    }

    #[test_log::test]
    fn binding_a_stale_handle_errors_instead_of_panicking() {
        let mut storage = BytesStorage::default();
        let mut pool = ExclusiveMemoryPool::new(1024, 32, u64::MAX, 0);

        let (stale, _) = orphan(&mut pool, &mut storage, 1024);
        pool.cleanup(&mut storage, 0, true);

        // `bind` used to index the page vector directly, so a handle naming a
        // page the pool no longer has took the device thread down with it.
        assert!(matches!(
            pool.bind(stale, ManagedMemoryHandle::new(), 0),
            Err(IoError::NotFound { .. })
        ));
    }
}
