pub mod cpu;
pub mod gpu;

/// Diagnostic: when set, neither the device nor the pinned-host storage ever
/// returns memory to the driver, leaving pool reuse as the only way a live
/// allocation can be disturbed.
///
/// Covers both storages deliberately. Gating only the device side leaves
/// `cuMemFreeHost` free to pull the host end of an in-flight DMA out from under
/// it, which faults the same way — so a device-only gate cannot tell a
/// premature free from a premature reuse.
///
/// Leaks every allocation. Fine for a timed soak, never for a real run.
pub(crate) static NO_DEALLOC: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    let set = std::env::var("CUBECL_DEBUG_NO_DEALLOC").is_ok();
    if set {
        log::warn!(
            "CUBECL_DEBUG_NO_DEALLOC is set: device and pinned host memory will never be freed"
        );
    }
    set
});
