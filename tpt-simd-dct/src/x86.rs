//! AVX2 matrix-vector kernel for the table transforms. Bit-identical to the
//! scalar row-by-row sums in `lib.rs`.
//!
//! The scalar sum for output `k` adds `table[k][i] * x[i]` in increasing `i`.
//! That order cannot be reassociated, so the vector form keeps it: eight
//! outputs share one accumulator, one lane per output, and `i` runs in order.
//! It needs the transposed table `t[i * n_out + k] = table[k][i]`, so each
//! load is contiguous over `k`.

use core::arch::x86_64::*;

/// Computes outputs `0 .. 8 * floor(n_out / 8)` of `out = table * x`.
/// Returns how many outputs were written; the caller finishes the rest.
///
/// # Safety
/// The CPU must support AVX2. `t.len() == m * n_out`, `x.len() == m` and
/// `out.len() == n_out` (checked by the caller).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn matvec_t(
    t: &[f32],
    m: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> usize {
    let body = n_out - n_out % 8;
    let mut k0 = 0;
    while k0 < body {
        // -0.0 is the starting value of `Iterator::sum` for `f32`, so the
        // sign of an all-zero result matches the scalar code.
        let mut acc = _mm256_set1_ps(-0.0);
        for (i, &xi) in x.iter().enumerate().take(m) {
            let xv = _mm256_set1_ps(xi);
            // SAFETY: i < m and k0 + 8 <= body <= n_out, so the 8-lane load at
            // `i * n_out + k0` ends at or before `m * n_out == t.len()`.
            let row = unsafe { _mm256_loadu_ps(t.as_ptr().add(i * n_out + k0)) };
            acc = _mm256_add_ps(acc, _mm256_mul_ps(row, xv));
        }
        // SAFETY: k0 + 8 <= body <= n_out == out.len().
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(k0), acc) };
        k0 += 8;
    }
    body
}
