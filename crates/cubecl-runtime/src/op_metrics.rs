//! Process-wide GPU operation counts for deterministic workload observability.
//!
//! The host-paced self-play loop is dominated by the *number* of GPU operations it submits —
//! kernel launches, host/device copies, and synchronizations — not by the cost of any single one.
//! These global counters make that submission volume observable per training generation, so an
//! optimization can be judged by whether it removed launches/copies/syncs rather than by noisy
//! wall-clock. They are the operation-count analogue of [`crate::cache_metrics`].

use core::sync::atomic::{AtomicU64, Ordering};

static KERNEL_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static H2D_COPIES: AtomicU64 = AtomicU64::new(0);
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_COPIES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static SYNCHRONIZATIONS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of process-wide GPU operation counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpMetrics {
    /// Kernels submitted to a device stream.
    pub kernel_launches: u64,
    /// Host-to-device copies submitted.
    pub h2d_copies: u64,
    /// Bytes submitted host-to-device.
    pub h2d_bytes: u64,
    /// Device-to-host copies submitted.
    pub d2h_copies: u64,
    /// Bytes submitted device-to-host.
    pub d2h_bytes: u64,
    /// Stream synchronizations (host waits on device).
    pub synchronizations: u64,
}

impl OpMetrics {
    /// Returns the non-negative counter delta from an earlier snapshot.
    pub fn delta(self, earlier: Self) -> Self {
        Self {
            kernel_launches: self.kernel_launches.saturating_sub(earlier.kernel_launches),
            h2d_copies: self.h2d_copies.saturating_sub(earlier.h2d_copies),
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            d2h_copies: self.d2h_copies.saturating_sub(earlier.d2h_copies),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            synchronizations: self
                .synchronizations
                .saturating_sub(earlier.synchronizations),
        }
    }
}

/// Returns the current process-wide GPU operation counts.
pub fn op_metrics() -> OpMetrics {
    OpMetrics {
        kernel_launches: KERNEL_LAUNCHES.load(Ordering::Relaxed),
        h2d_copies: H2D_COPIES.load(Ordering::Relaxed),
        h2d_bytes: H2D_BYTES.load(Ordering::Relaxed),
        d2h_copies: D2H_COPIES.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
        synchronizations: SYNCHRONIZATIONS.load(Ordering::Relaxed),
    }
}

#[doc(hidden)]
/// Records one kernel launch submitted to a device stream.
pub fn record_kernel_launch() {
    KERNEL_LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one host-to-device copy of `bytes` bytes.
pub fn record_h2d_copy(bytes: u64) {
    H2D_COPIES.fetch_add(1, Ordering::Relaxed);
    H2D_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one device-to-host copy of `bytes` bytes.
pub fn record_d2h_copy(bytes: u64) {
    D2H_COPIES.fetch_add(1, Ordering::Relaxed);
    D2H_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

#[doc(hidden)]
/// Records one stream synchronization (a host wait on device work).
pub fn record_synchronization() {
    SYNCHRONIZATIONS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::OpMetrics;

    #[test]
    fn test_op_metrics_delta() {
        let earlier = OpMetrics {
            kernel_launches: 100,
            h2d_copies: 10,
            h2d_bytes: 1024,
            synchronizations: 4,
            ..OpMetrics::default()
        };
        let current = OpMetrics {
            kernel_launches: 258,
            h2d_copies: 110,
            h2d_bytes: 5120,
            synchronizations: 6,
            ..OpMetrics::default()
        };

        let delta = current.delta(earlier);

        assert_eq!(158, delta.kernel_launches);
        assert_eq!(100, delta.h2d_copies);
        assert_eq!(4096, delta.h2d_bytes);
        assert_eq!(2, delta.synchronizations);
    }
}
