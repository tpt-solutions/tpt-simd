//! AVX2 fast paths. Bit-identical to [`crate::portable`].

use core::arch::x86_64::*;

/// AVX2 `i16` dot product with wrapping `i32` accumulation.
///
/// # Safety
/// The CPU must support AVX2 (guaranteed when this module is compiled, since
/// it is gated on `target_feature = "avx2"`), and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_i16(a: &[i16], b: &[i16]) -> i32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: i + 32 <= n and both slices have n elements, so the two
        // 32-byte unaligned loads per operand stay in bounds.
        unsafe {
            let x0 = _mm256_loadu_si256(pa.add(i) as *const __m256i);
            let y0 = _mm256_loadu_si256(pb.add(i) as *const __m256i);
            let x1 = _mm256_loadu_si256(pa.add(i + 16) as *const __m256i);
            let y1 = _mm256_loadu_si256(pb.add(i + 16) as *const __m256i);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(x0, y0));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(x1, y1));
        }
        i += 32;
    }
    if i + 16 <= n {
        // SAFETY: i + 16 <= n, one in-bounds 32-byte load per operand.
        unsafe {
            let x0 = _mm256_loadu_si256(pa.add(i) as *const __m256i);
            let y0 = _mm256_loadu_si256(pb.add(i) as *const __m256i);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(x0, y0));
        }
        i += 16;
    }
    let mut lanes = [0i32; 8];
    // SAFETY: `lanes` is 32 bytes, exactly one unaligned __m256i store.
    unsafe {
        _mm256_storeu_si256(
            lanes.as_mut_ptr() as *mut __m256i,
            _mm256_add_epi32(acc0, acc1),
        )
    };
    let mut sum = 0i32;
    for l in lanes {
        sum = sum.wrapping_add(l);
    }
    while i < n {
        sum = sum.wrapping_add(a[i] as i32 * b[i] as i32);
        i += 1;
    }
    sum
}
