//! AVX2 prefix-sum kernels. Bit-identical to the scalar loops in `lib.rs`:
//! integer addition wraps, and the scan is exact at any association.
//!
//! Each block is scanned in two log-steps inside each 128-bit half. The low
//! half's total is then added to the high half, and the running carry from the
//! previous block is added to everything.

use core::arch::x86_64::*;

/// AVX2 in-place inclusive wrapping prefix sum over `u32`.
///
/// # Safety
/// The CPU must support AVX2 (guaranteed when this module is compiled, since it
/// is gated on `target_feature = "avx2"`).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn prefix_sum_u32(values: &mut [u32]) {
    let n = values.len();
    let p = values.as_mut_ptr();
    let high_mask = _mm256_setr_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    let all3 = _mm256_set1_epi32(3);
    let all7 = _mm256_set1_epi32(7);
    let mut carry = _mm256_setzero_si256();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: i + 8 <= n, so the 8-lane load and store stay in bounds.
        unsafe {
            let x = _mm256_loadu_si256(p.add(i) as *const __m256i);
            // Prefix within each 128-bit half: shift by one, then by two elements.
            let s1 = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
            let s2 = _mm256_add_epi32(s1, _mm256_slli_si256::<8>(s1));
            // Add element 3 (end of the low half) to the high half.
            let low_total = _mm256_and_si256(_mm256_permutevar8x32_epi32(s2, all3), high_mask);
            let s3 = _mm256_add_epi32(s2, low_total);
            let out = _mm256_add_epi32(s3, carry);
            _mm256_storeu_si256(p.add(i) as *mut __m256i, out);
            // Block total from `s3`, not `out`, so the loop-carried chain is one add.
            carry = _mm256_add_epi32(carry, _mm256_permutevar8x32_epi32(s3, all7));
        }
        i += 8;
    }
    // The last block's output is the running total, so the tail starts from it.
    let mut acc = if i > 0 { values[i - 1] } else { 0 };
    for v in &mut values[i..] {
        acc = acc.wrapping_add(*v);
        *v = acc;
    }
}

/// AVX2 in-place inclusive wrapping prefix sum over `u64`.
///
/// # Safety
/// As for [`prefix_sum_u32`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn prefix_sum_u64(values: &mut [u64]) {
    let n = values.len();
    let p = values.as_mut_ptr();
    let high_mask = _mm256_setr_epi64x(0, 0, -1, -1);
    let mut carry = _mm256_setzero_si256();
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: i + 4 <= n, so the 4-lane load and store stay in bounds.
        unsafe {
            let x = _mm256_loadu_si256(p.add(i) as *const __m256i);
            // Prefix within each 128-bit half: shift by one element.
            let s1 = _mm256_add_epi64(x, _mm256_slli_si256::<8>(x));
            // Add element 1 (end of the low half) to the high half.
            let low_total =
                _mm256_and_si256(_mm256_permute4x64_epi64::<0b01_01_01_01>(s1), high_mask);
            let s2 = _mm256_add_epi64(s1, low_total);
            let out = _mm256_add_epi64(s2, carry);
            _mm256_storeu_si256(p.add(i) as *mut __m256i, out);
            // Block total from `s2`, not `out`, so the loop-carried chain is one add.
            carry = _mm256_add_epi64(carry, _mm256_permute4x64_epi64::<0b11_11_11_11>(s2));
        }
        i += 4;
    }
    let mut acc = if i > 0 { values[i - 1] } else { 0 };
    for v in &mut values[i..] {
        acc = acc.wrapping_add(*v);
        *v = acc;
    }
}
