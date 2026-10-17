//! AVX2 kernels for `peak` and `fir`. Both match the scalar code bit for bit.
//!
//! `peak` is a maximum of absolute values, so the order does not change the
//! result. The NaN rule matches `f32::max`: a NaN operand is ignored. In
//! `_mm256_max_ps(s, acc)` the second operand is returned when either input is NaN, so
//! `acc` plays that role.
//!
//! `fir` keeps each output's sum in tap order. Eight outputs share one accumulator,
//! one lane per output. Only outputs with every tap in range go through here.

use core::arch::x86_64::*;

/// Maximum absolute value over the first `8 * k` elements of `x`, starting from
/// `0.0`. Returns that maximum and the number of elements covered.
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn peak_body(x: &[f32]) -> (f32, usize) {
    let n = x.len() - x.len() % 8;
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < n {
        // SAFETY: i + 8 <= n <= x.len().
        let s = unsafe { _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i)), abs_mask) };
        acc = _mm256_max_ps(s, acc);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` is 32 bytes.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut m = 0.0f32;
    for l in lanes {
        m = m.max(l);
    }
    (m, n)
}

/// FIR outputs `lo .. hi` (with `lo = taps.len() - 1`, `hi` a multiple of eight
/// past `lo`). Each output sums `taps[k] * x[n - k]` in increasing `k`, starting from
/// `+0.0`, as the scalar code does.
///
/// # Safety
/// The CPU must support AVX2. `taps` is non-empty, `lo <= hi <= x.len()`,
/// `x.len() == y.len()`, and `hi - lo` is a multiple of eight.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn fir_range(taps: &[f32], x: &[f32], y: &mut [f32], lo: usize, hi: usize) {
    let mut n0 = lo;
    while n0 < hi {
        let mut acc = _mm256_setzero_ps();
        for (k, &t) in taps.iter().enumerate() {
            let tv = _mm256_set1_ps(t);
            // SAFETY: n0 >= lo = taps.len() - 1 >= k, so n0 - k is in bounds, and
            // n0 + 8 <= hi <= x.len() keeps the 8-lane load in range.
            let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(n0 - k)) };
            acc = _mm256_add_ps(acc, _mm256_mul_ps(tv, xv));
        }
        // SAFETY: n0 + 8 <= hi <= y.len().
        unsafe { _mm256_storeu_ps(y.as_mut_ptr().add(n0), acc) };
        n0 += 8;
    }
}
