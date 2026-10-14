//! Cached runtime feature checks for kernel dispatch (ADR 0003).
//!
//! Kernel crates choose between a portable path and an intrinsics path once
//! per call, at the slice-level entry point. [`Cache`] stores the answer of a
//! detection function in an atomic so repeated calls skip the detection
//! itself. [`x86_avx2_fma`] is the shared check for the AVX2+FMA kernels and
//! needs the `std` feature (for `is_x86_feature_detected!`).

use core::sync::atomic::{AtomicU8, Ordering};

/// Caches the result of a runtime feature check.
///
/// Intended for a `static` next to the kernels that use it:
///
/// ```ignore
/// static AVX2: Cache = Cache::new();
/// let yes = AVX2.get_or_detect(|| std::is_x86_feature_detected!("avx2"));
/// ```
#[derive(Debug)]
pub struct Cache(AtomicU8);

impl Cache {
    /// An empty cache; the first [`get_or_detect`](Self::get_or_detect) runs the detection.
    pub const fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    /// Returns the cached answer, running `detect` on first use.
    ///
    /// Two threads may both run `detect` on a cold cache; they store the same
    /// answer, so this is harmless.
    #[inline]
    pub fn get_or_detect(&self, detect: fn() -> bool) -> bool {
        // 0 = unknown, 1 = no, 2 = yes
        match self.0.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => {
                let yes = detect();
                self.0.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
                yes
            }
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

/// `true` if the running x86 CPU supports AVX2 and FMA3, detected once.
///
/// Returns `true` without a query when both are enabled at compile time.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub fn x86_avx2_fma() -> bool {
    if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
        return true;
    }
    // Miri cannot execute `cpuid`; report the compile-time answer (false here).
    if cfg!(miri) {
        return false;
    }
    static CACHE: Cache = Cache::new();
    CACHE.get_or_detect(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    // One counter per test so parallel test threads do not interfere.
    static YES_CALLS: AtomicUsize = AtomicUsize::new(0);
    static NO_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn counted_yes() -> bool {
        YES_CALLS.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn counted_no() -> bool {
        NO_CALLS.fetch_add(1, Ordering::Relaxed);
        false
    }

    #[test]
    fn detects_once_and_caches_yes() {
        let c = Cache::new();
        assert!(c.get_or_detect(counted_yes));
        assert!(c.get_or_detect(counted_yes));
        assert!(c.get_or_detect(|| unreachable!("cached answer must be reused")));
        assert_eq!(YES_CALLS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn caches_no() {
        let c = Cache::default();
        assert!(!c.get_or_detect(counted_no));
        assert!(!c.get_or_detect(|| unreachable!("cached answer must be reused")));
        assert_eq!(NO_CALLS.load(Ordering::Relaxed), 1);
    }
}
