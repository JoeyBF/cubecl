use crate::{
    memory_management::{
        BytesFormat, ManagedMemoryHandle, MemoryLocation, MemoryUsage, PageKey,
        memory_pool::{MemoryPage, MemoryPool, Slice, page_not_found},
    },
    server::IoError,
    storage::StorageId,
};
use core::fmt::Display;
use cubecl_common::backtrace::BackTrace;
use slotmap::SlotMap;

pub struct SlicedPool {
    /// Pages are keyed rather than positional: a page keeps its [`PageKey`] for
    /// its whole life, so freeing one never invalidates the keys cached in the
    /// descriptors pointing at the others. See [`PageKey`].
    pages: SlotMap<PageKey, (MemoryPage, StorageId)>,
    page_size: u64,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// Max number of pages (`floor(max_pool_size / page_size)`).
    /// `None` keeps unbounded growth.
    max_pages: Option<u16>,
}

impl SlicedPool {
    pub fn new(
        page_size: u64,
        max_slice_size: u64,
        alignment: u64,
        pool_pos: u8,
        max_pool_size: Option<u64>,
    ) -> Self {
        // A budget smaller than one page shrinks the page to the
        // (alignment-rounded-down) budget, so the cap is honored rather than
        // exceeded by a single page. A budget below the alignment can't fit
        // even the smallest page the device allows, so it yields zero pages:
        // allocations error instead of overshooting the cap.
        let (page_size, max_pages) = match max_pool_size {
            Some(cap) => {
                let page_size = if cap < page_size {
                    (cap / alignment * alignment).max(alignment)
                } else {
                    page_size
                };
                let max_pages = (cap / page_size).min(u16::MAX as u64) as u16;
                (page_size, Some(max_pages))
            }
            None => (page_size, None),
        };

        Self {
            pages: SlotMap::with_key(),
            page_size,
            alignment,
            max_alloc_size: max_slice_size.min(page_size),
            location_base: MemoryLocation::base(pool_pos),
            max_pages,
        }
    }

    /// Allocate a new page and return its key.
    fn alloc_page<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<PageKey, IoError> {
        let storage = storage.alloc(self.page_size)?;
        let storage_id = storage.id;
        let location_base = self.location_base;
        let alignment = self.alignment;

        // The page stamps its key on every slice it hands out, so it has to know
        // the key before it exists — hence `insert_with_key`.
        Ok(self.pages.insert_with_key(|key| {
            let mut location_base = location_base;
            location_base.page = key;

            (
                MemoryPage::new(storage, alignment, location_base),
                storage_id,
            )
        }))
    }
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
            ||
            // If the size is close to the page size so it doesn't create much fragmentation with
            // unused space. Only for unbounded pools: a hard-capped pool is a budget for the
            // allocations `max_slice_size` routes to it, and near-page-size strays would exhaust
            // it (e.g. a small metadata pool whose page size matches an upload staging chunk).
            (self.max_pages.is_none()
                && match self.page_size.checked_sub(size) {
                    Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                    None => false,
                })
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let key = binding.descriptor().page();
        let (page, _) = self.pages.get(key).ok_or_else(|| page_not_found(key))?;
        page.find(binding)
    }

    fn try_reserve(&mut self, size: u64) -> Option<super::ManagedMemoryHandle> {
        for (page, _) in self.pages.values_mut() {
            page.coalesce();
            if let Some(handle) = page.try_reserve(size) {
                return Some(handle);
            }
        }

        None
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<super::ManagedMemoryHandle, crate::server::IoError> {
        // `alloc` is only called after `try_reserve` coalesced every page and
        // found no fit, so hitting the cap here means the working set truly
        // exceeds the budget. `len` counts live pages — a slot vacated by
        // `cleanup` is spare capacity, not a page against the cap.
        if let Some(max_pages) = self.max_pages
            && self.pages.len() >= max_pages as usize
        {
            return Err(IoError::PoolCapacityExceeded {
                size,
                capacity: max_pages as u64 * self.page_size,
                in_use: self.get_memory_usage().bytes_in_use,
                backtrace: BackTrace::capture(),
            });
        }

        let key = self.alloc_page(storage)?;
        let (page, _) = &mut self.pages[key];
        let returned = page.try_reserve(size);

        Ok(returned.expect("effective_size to be smaller than page_size"))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for (page, _) in self.pages.values() {
            let current = page.memory_usage();
            usage = usage.combine(current);
        }

        usage
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if !explicit {
            return;
        }

        // Surviving pages keep their key, so a page that was live before the
        // cleanup is still reachable through every descriptor that named it —
        // there is nothing to renumber, and no way for a descriptor to end up
        // naming a page it never pointed at.
        self.pages.retain(|_, (page, id)| {
            page.coalesce();
            let summary = page.summary(false);

            if summary.amount_free == summary.amount_total {
                storage.dealloc(*id);
                return false;
            }

            true
        });
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let key = reserved.descriptor().page();
        let (page, _) = self.pages.get_mut(key).ok_or_else(|| page_not_found(key))?;

        page.bind(reserved, assigned, cursor)?;

        Ok(())
    }
}

impl Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_alloc_size)
        ))?;
        if let Some(max_pages) = self.max_pages {
            f.write_fmt(format_args!(
                " max_pool_size={}",
                BytesFormat::new(max_pages as u64 * self.page_size)
            ))?;
        }
        f.write_str("\n")?;

        for (page, id) in self.pages.values() {
            let summary = page.summary(false);
            f.write_fmt(format_args!(
                "   - Page {id} num_slices={} =>",
                summary.num_total
            ))?;

            let size_free = BytesFormat::new(summary.amount_free);
            let size_full = BytesFormat::new(summary.amount_full);
            let size_total = BytesFormat::new(summary.amount_total);

            f.write_fmt(format_args!(
                " {size_free} free - {size_full} full - {size_total} total\n"
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryBinding;
    use crate::storage::BytesStorage;

    /// Fills `pool`'s freshly allocated page and then hands the slice a new
    /// identity, dropping it, so the page reads as fully free while the returned
    /// handle and binding keep naming it.
    ///
    /// This is the shape the bug needs: a descriptor the pool can no longer
    /// reach (in production, a handle orphaned by a `bind` and kept alive by
    /// another stream's binding) that still carries the page's address.
    fn orphan(
        pool: &mut SlicedPool,
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
        let mut pool = SlicedPool::new(1024, 1024, 32, 0, None);

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
        let mut pool = SlicedPool::new(1024, 1024, 32, 0, None);

        // Three pages, with only the middle one still holding a live slice.
        let (_first, _) = orphan(&mut pool, &mut storage, 1024);
        let kept = pool.alloc(&mut storage, 1024).unwrap();
        let (_last, _) = orphan(&mut pool, &mut storage, 1024);

        let binding = kept.clone().binding();
        let location = kept.descriptor().location();

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
        let mut pool = SlicedPool::new(1024, 1024, 32, 0, None);

        let (stale, _) = orphan(&mut pool, &mut storage, 1024);
        pool.cleanup(&mut storage, 0, true);

        // `bind` used to index the page vector directly, so a handle naming a
        // page the pool no longer has took the device thread down with it.
        assert!(matches!(
            pool.bind(stale, ManagedMemoryHandle::new(), 0),
            Err(IoError::NotFound { .. })
        ));
    }

    #[test_log::test]
    fn a_vacated_slot_is_reused_rather_than_counted_against_the_cap() {
        let mut storage = BytesStorage::default();
        // Two pages' worth of budget.
        let mut pool = SlicedPool::new(1024, 1024, 32, 0, Some(2048));

        for _ in 0..4 {
            let (_orphan, _) = orphan(&mut pool, &mut storage, 1024);
            pool.cleanup(&mut storage, 0, true);
        }

        // Churn recycles slots instead of accumulating them, so the cap still
        // admits a full two pages afterwards.
        let _a = pool.alloc(&mut storage, 1024).unwrap();
        let _b = pool.alloc(&mut storage, 1024).unwrap();
        assert!(matches!(
            pool.alloc(&mut storage, 1024),
            Err(IoError::PoolCapacityExceeded { .. })
        ));
    }
}
