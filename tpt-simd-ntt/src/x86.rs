//! AVX2 butterflies for the NTT. Bit-identical to the scalar [`crate::Montgomery`]
//! methods: each step is the same integer operation, applied to eight lanes.
//!
//! Inputs must be reduced (`< q`), with `q` odd and `< 2^31`, so that every
//! intermediate sum stays below `2^32` and `u + v` or `u - v + q` never
//! wraps in a way the unsigned `min` trick cannot undo.

use core::arch::x86_64::*;

/// Montgomery product of eight lanes: `a * b * 2^-32 mod q`, for `a, b < q`.
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn mul_mont8(a: __m256i, b: __m256i, q: __m256i, q_neg_inv: __m256i) -> __m256i {
    // Even dwords come from `mul_epu32` directly; odd dwords are shifted down first.
    let a_odd = _mm256_srli_epi64::<32>(a);
    let b_odd = _mm256_srli_epi64::<32>(b);
    let t_even = _mm256_mul_epu32(a, b);
    let t_odd = _mm256_mul_epu32(a_odd, b_odd);
    // m = low32(t) * (-q^-1) mod 2^32; `mul_epu32` reads the low dword of each lane.
    let m_even = _mm256_mul_epu32(t_even, q_neg_inv);
    let m_odd = _mm256_mul_epu32(t_odd, q_neg_inv);
    // t + m * q < 2^64, and the result is the high dword of each 64-bit sum.
    let s_even = _mm256_add_epi64(t_even, _mm256_mul_epu32(m_even, q));
    let s_odd = _mm256_add_epi64(t_odd, _mm256_mul_epu32(m_odd, q));
    let u = _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<32>(s_even), s_odd);
    // u < 2q: one conditional subtraction, done as an unsigned minimum.
    _mm256_min_epu32(u, _mm256_sub_epi32(u, q))
}

/// Runs one Cooley-Tukey stage (butterfly half-width `half`) over `a`.
///
/// `tw` holds the stage's `half` twiddles in Montgomery form, contiguous.
/// `half` must be a multiple of eight, and `a.len()` a multiple of `2 * half`.
///
/// # Safety
/// The CPU must support AVX2. Every element of `a` and `tw` must be `< q`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn butterfly_stage(
    a: &mut [u32],
    tw: &[u32],
    half: usize,
    q: u32,
    q_neg_inv: u32,
) {
    debug_assert!(half.is_multiple_of(8) && tw.len() == half && a.len().is_multiple_of(2 * half));
    let qv = _mm256_set1_epi32(q as i32);
    let qn = _mm256_set1_epi32(q_neg_inv as i32);
    let p = a.as_mut_ptr();
    for start in (0..a.len()).step_by(2 * half) {
        let mut k = 0;
        while k < half {
            // SAFETY: start + k + 8 <= start + half and start + half + k + 8 <= start + 2*half
            // <= a.len(); k + 8 <= half == tw.len(). All loads and stores are in bounds.
            unsafe {
                let u = _mm256_loadu_si256(p.add(start + k) as *const __m256i);
                let x = _mm256_loadu_si256(p.add(start + k + half) as *const __m256i);
                let w = _mm256_loadu_si256(tw.as_ptr().add(k) as *const __m256i);
                let v = mul_mont8(x, w, qv, qn);
                // u + v < 2q: subtract q when it is at least q.
                let s = _mm256_add_epi32(u, v);
                let s = _mm256_min_epu32(s, _mm256_sub_epi32(s, qv));
                // u - v, plus q when it wrapped (u < v): the smaller of the two is the result.
                let d = _mm256_sub_epi32(u, v);
                let d = _mm256_min_epu32(d, _mm256_add_epi32(d, qv));
                _mm256_storeu_si256(p.add(start + k) as *mut __m256i, s);
                _mm256_storeu_si256(p.add(start + k + half) as *mut __m256i, d);
            }
            k += 8;
        }
    }
}
