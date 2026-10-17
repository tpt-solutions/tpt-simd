//! AVX2 lane kernels. Bit-identical to [`crate::portable`]: one 8-lane
//! accumulator per sum, separate multiply and add (never FMA), and element `i`
//! in lane `i % 8`.

use crate::LANES;
use core::arch::x86_64::*;

/// AVX2 version of [`crate::portable::inner_product_lanes`].
///
/// # Safety
/// The CPU must support AVX2 (guaranteed when this module is compiled, since
/// it is gated on `target_feature = "avx2"`). `a` and `b` must have the same
/// length, a multiple of eight.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn inner_product_lanes(a: &[f32], b: &[f32]) -> [f32; LANES] {
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        // SAFETY: i + 8 <= a.len() == b.len(), so both 8-lane loads are in bounds.
        unsafe {
            let x = _mm256_loadu_ps(pa.add(i));
            let y = _mm256_loadu_ps(pb.add(i));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(x, y));
        }
        i += LANES;
    }
    store(acc)
}

/// AVX2 version of [`crate::portable::l2_squared_lanes`].
///
/// # Safety
/// As for [`inner_product_lanes`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn l2_squared_lanes(a: &[f32], b: &[f32]) -> [f32; LANES] {
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        // SAFETY: i + 8 <= a.len() == b.len(), so both 8-lane loads are in bounds.
        unsafe {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
        }
        i += LANES;
    }
    store(acc)
}

/// AVX2 version of [`crate::portable::cosine_lanes`].
///
/// # Safety
/// As for [`inner_product_lanes`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn cosine_lanes(
    a: &[f32],
    b: &[f32],
) -> ([f32; LANES], [f32; LANES], [f32; LANES]) {
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut dot = _mm256_setzero_ps();
    let mut na = _mm256_setzero_ps();
    let mut nb = _mm256_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        // SAFETY: i + 8 <= a.len() == b.len(), so both 8-lane loads are in bounds.
        unsafe {
            let x = _mm256_loadu_ps(pa.add(i));
            let y = _mm256_loadu_ps(pb.add(i));
            dot = _mm256_add_ps(dot, _mm256_mul_ps(x, y));
            na = _mm256_add_ps(na, _mm256_mul_ps(x, x));
            nb = _mm256_add_ps(nb, _mm256_mul_ps(y, y));
        }
        i += LANES;
    }
    (store(dot), store(na), store(nb))
}

#[target_feature(enable = "avx2")]
fn store(v: __m256) -> [f32; LANES] {
    let mut out = [0.0f32; LANES];
    // SAFETY: `out` is 32 bytes, exactly one unaligned 8-lane store.
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
    out
}
