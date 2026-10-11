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
    // Tail (< 16 elements): zipped slices so the loop carries no per-element
    // bounds check on `b`.
    for (&x, &y) in a[i..].iter().zip(&b[i..n]) {
        sum = sum.wrapping_add(x as i32 * y as i32);
    }
    sum
}

/// AVX-512BW `i16` dot product: the AVX2 kernel with 512-bit `vpmaddwd`.
///
/// Same result as [`dot_i16`] and [`crate::portable::dot_product_i16`]: each
/// `vpmaddwd` lane is an exact `i32` pair sum and all additions wrap, so the
/// order of accumulation does not change the final value.
///
/// # Safety
/// The CPU must support AVX-512BW (guaranteed when this function is compiled,
/// since it is gated on `target_feature = "avx512bw"`), and `a.len() == b.len()`.
#[cfg(target_feature = "avx512bw")]
#[target_feature(enable = "avx512bw")]
pub(crate) unsafe fn dot_i16_avx512bw(a: &[i16], b: &[i16]) -> i32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut i = 0;
    // Two 512-bit loads per operand per iteration (64 i16 elements).
    while i + 64 <= n {
        // SAFETY: i + 64 <= n and both slices have n elements, so the four
        // 64-byte unaligned loads stay in bounds.
        unsafe {
            let x0 = _mm512_loadu_si512(pa.add(i) as *const __m512i);
            let y0 = _mm512_loadu_si512(pb.add(i) as *const __m512i);
            let x1 = _mm512_loadu_si512(pa.add(i + 32) as *const __m512i);
            let y1 = _mm512_loadu_si512(pb.add(i + 32) as *const __m512i);
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(x0, y0));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(x1, y1));
        }
        i += 64;
    }
    if i + 32 <= n {
        // SAFETY: i + 32 <= n, one in-bounds 64-byte load per operand.
        unsafe {
            let x0 = _mm512_loadu_si512(pa.add(i) as *const __m512i);
            let y0 = _mm512_loadu_si512(pb.add(i) as *const __m512i);
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(x0, y0));
        }
        i += 32;
    }
    let mut lanes = [0i32; 16];
    // SAFETY: `lanes` is 64 bytes, exactly one unaligned __m512i store.
    unsafe {
        _mm512_storeu_si512(
            lanes.as_mut_ptr() as *mut __m512i,
            _mm512_add_epi32(acc0, acc1),
        )
    };
    let mut sum = 0i32;
    for l in lanes {
        sum = sum.wrapping_add(l);
    }
    // Tail (< 32 elements), as in the AVX2 kernel.
    for (&x, &y) in a[i..].iter().zip(&b[i..n]) {
        sum = sum.wrapping_add(x as i32 * y as i32);
    }
    sum
}
