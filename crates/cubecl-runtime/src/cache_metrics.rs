//! Process-wide cache metrics for startup observability.

use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;

static AUTOTUNE_HITS: AtomicU64 = AtomicU64::new(0);
static AUTOTUNE_MISSES: AtomicU64 = AtomicU64::new(0);
static AUTOTUNE_WRITES: AtomicU64 = AtomicU64::new(0);
static AUTOTUNE_SEED_ENTRIES_LOADED: AtomicU64 = AtomicU64::new(0);
static AUTOTUNE_WRITABLE_ENTRIES_LOADED: AtomicU64 = AtomicU64::new(0);
static COMPILATION_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static COMPILATION_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static COMPILATION_CACHE_WRITES: AtomicU64 = AtomicU64::new(0);
static NVRTC_COMPILATIONS: AtomicU64 = AtomicU64::new(0);
static NVRTC_COMPILATION_NANOS: AtomicU64 = AtomicU64::new(0);
static MODULE_LOADS: AtomicU64 = AtomicU64::new(0);
static MODULE_LOAD_NANOS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of process-wide autotune and compilation-cache activity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheMetrics {
    /// Resolved autotune cache hits.
    pub autotune_hits: u64,
    /// Autotune keys that required tuning.
    pub autotune_misses: u64,
    /// Newly tuned results written to the persistent cache.
    pub autotune_writes: u64,
    /// Seed-cache entries loaded across initialized tuners.
    pub autotune_seed_entries_loaded: u64,
    /// Writable-cache entries loaded across initialized tuners.
    pub autotune_writable_entries_loaded: u64,
    /// Compiled-kernel cache hits.
    pub compilation_cache_hits: u64,
    /// Compiled-kernel cache misses.
    pub compilation_cache_misses: u64,
    /// Compiled kernels written to the persistent cache.
    pub compilation_cache_writes: u64,
    /// Kernels compiled with NVRTC after a compilation-cache miss.
    pub nvrtc_compilations: u64,
    /// Wall time spent compiling kernels with NVRTC.
    pub nvrtc_compilation_time: Duration,
    /// PTX modules loaded into a backend context.
    pub module_loads: u64,
    /// Wall time spent loading PTX modules and resolving entry points.
    pub module_load_time: Duration,
}

impl CacheMetrics {
    /// Returns the non-negative counter delta from an earlier snapshot.
    pub fn delta(self, earlier: Self) -> Self {
        Self {
            autotune_hits: self.autotune_hits.saturating_sub(earlier.autotune_hits),
            autotune_misses: self.autotune_misses.saturating_sub(earlier.autotune_misses),
            autotune_writes: self.autotune_writes.saturating_sub(earlier.autotune_writes),
            autotune_seed_entries_loaded: self
                .autotune_seed_entries_loaded
                .saturating_sub(earlier.autotune_seed_entries_loaded),
            autotune_writable_entries_loaded: self
                .autotune_writable_entries_loaded
                .saturating_sub(earlier.autotune_writable_entries_loaded),
            compilation_cache_hits: self
                .compilation_cache_hits
                .saturating_sub(earlier.compilation_cache_hits),
            compilation_cache_misses: self
                .compilation_cache_misses
                .saturating_sub(earlier.compilation_cache_misses),
            compilation_cache_writes: self
                .compilation_cache_writes
                .saturating_sub(earlier.compilation_cache_writes),
            nvrtc_compilations: self
                .nvrtc_compilations
                .saturating_sub(earlier.nvrtc_compilations),
            nvrtc_compilation_time: self
                .nvrtc_compilation_time
                .saturating_sub(earlier.nvrtc_compilation_time),
            module_loads: self.module_loads.saturating_sub(earlier.module_loads),
            module_load_time: self
                .module_load_time
                .saturating_sub(earlier.module_load_time),
        }
    }

    /// Returns whether the snapshot contains work that a fully warm cache should avoid.
    pub fn has_cache_miss(&self) -> bool {
        self.autotune_misses > 0
            || self.autotune_writes > 0
            || self.compilation_cache_misses > 0
            || self.compilation_cache_writes > 0
            || self.nvrtc_compilations > 0
    }
}

/// Returns the current process-wide cache metrics.
pub fn cache_metrics() -> CacheMetrics {
    CacheMetrics {
        autotune_hits: AUTOTUNE_HITS.load(Ordering::Relaxed),
        autotune_misses: AUTOTUNE_MISSES.load(Ordering::Relaxed),
        autotune_writes: AUTOTUNE_WRITES.load(Ordering::Relaxed),
        autotune_seed_entries_loaded: AUTOTUNE_SEED_ENTRIES_LOADED.load(Ordering::Relaxed),
        autotune_writable_entries_loaded: AUTOTUNE_WRITABLE_ENTRIES_LOADED.load(Ordering::Relaxed),
        compilation_cache_hits: COMPILATION_CACHE_HITS.load(Ordering::Relaxed),
        compilation_cache_misses: COMPILATION_CACHE_MISSES.load(Ordering::Relaxed),
        compilation_cache_writes: COMPILATION_CACHE_WRITES.load(Ordering::Relaxed),
        nvrtc_compilations: NVRTC_COMPILATIONS.load(Ordering::Relaxed),
        nvrtc_compilation_time: Duration::from_nanos(
            NVRTC_COMPILATION_NANOS.load(Ordering::Relaxed),
        ),
        module_loads: MODULE_LOADS.load(Ordering::Relaxed),
        module_load_time: Duration::from_nanos(MODULE_LOAD_NANOS.load(Ordering::Relaxed)),
    }
}

#[doc(hidden)]
/// Records one resolved autotune cache hit.
pub fn record_autotune_hit() {
    AUTOTUNE_HITS.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one autotune key that requires tuning.
pub fn record_autotune_miss() {
    AUTOTUNE_MISSES.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one newly tuned persistent-cache write.
pub fn record_autotune_write() {
    AUTOTUNE_WRITES.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records seed-cache entries loaded by an initialized tuner.
pub fn record_autotune_seed_entries_loaded(entries: u64) {
    AUTOTUNE_SEED_ENTRIES_LOADED.fetch_add(entries, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records writable-cache entries loaded by an initialized tuner.
pub fn record_autotune_writable_entries_loaded(entries: u64) {
    AUTOTUNE_WRITABLE_ENTRIES_LOADED.fetch_add(entries, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one compiled-kernel cache hit.
pub fn record_compilation_cache_hit() {
    COMPILATION_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one compiled-kernel cache miss.
pub fn record_compilation_cache_miss() {
    COMPILATION_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one compiled-kernel persistent-cache write.
pub fn record_compilation_cache_write() {
    COMPILATION_CACHE_WRITES.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one NVRTC kernel compilation and its duration.
pub fn record_nvrtc_compilation(duration: Duration) {
    NVRTC_COMPILATIONS.fetch_add(1, Ordering::Relaxed);
    NVRTC_COMPILATION_NANOS.fetch_add(duration_nanos(duration), Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one PTX module load and its duration.
pub fn record_module_load(duration: Duration) {
    MODULE_LOADS.fetch_add(1, Ordering::Relaxed);
    MODULE_LOAD_NANOS.fetch_add(duration_nanos(duration), Ordering::Relaxed);
}

fn duration_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::CacheMetrics;
    use core::time::Duration;

    #[test]
    fn test_cache_metrics_delta() {
        let earlier = CacheMetrics {
            autotune_hits: 4,
            nvrtc_compilations: 3,
            nvrtc_compilation_time: Duration::from_millis(10),
            ..CacheMetrics::default()
        };
        let current = CacheMetrics {
            autotune_hits: 9,
            nvrtc_compilations: 5,
            nvrtc_compilation_time: Duration::from_millis(25),
            ..CacheMetrics::default()
        };

        let delta = current.delta(earlier);

        assert_eq!(5, delta.autotune_hits);
        assert_eq!(2, delta.nvrtc_compilations);
        assert_eq!(Duration::from_millis(15), delta.nvrtc_compilation_time);
    }

    #[test]
    fn test_cache_metrics_has_cache_miss() {
        let warm = CacheMetrics {
            autotune_hits: 1,
            compilation_cache_hits: 1,
            module_loads: 1,
            ..CacheMetrics::default()
        };
        let cold = CacheMetrics {
            compilation_cache_misses: 1,
            ..CacheMetrics::default()
        };

        assert!(!warm.has_cache_miss());
        assert!(cold.has_cache_miss());
    }
}
