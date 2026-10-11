//! AVX2 fast paths. Bit-identical to [`crate::portable`].

use core::arch::x86_64::*;

/// AVX2 `i16` dot product with wrapping `i32` accumulation.
///
/// # Safety
/// The CPU must support AVX2 (guaranteed when this module is compiled, since
/// it is gated on `target_feature = "avx2"`), and `a.len() == b.len()`.
// Not used when AVX-512BW is enabled: `lib.rs` dispatches to the AVX-512 kernel then.
#[cfg(not(target_feature = "avx512bw"))]
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
    if i < n {
        // Remaining 1..=31 elements: one zero-masked load per operand. Masked-off
        // lanes are not accessed and contribute 0, so no scalar tail loop.
        let k = ((1u32 << (n - i)) - 1) as __mmask32;
        // SAFETY: the mask enables only lanes i..n, which are in bounds for both slices.
        unsafe {
            let x = _mm512_maskz_loadu_epi16(k, pa.add(i));
            let y = _mm512_maskz_loadu_epi16(k, pb.add(i));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(x, y));
        }
    }
    let acc = _mm512_add_epi32(acc0, acc1);
    // Fold the 512-bit accumulator to 256 bits in-register, so only 8 lanes
    // are stored and summed in scalar code.
    let mut lanes = [0i32; 8];
    // SAFETY: `lanes` is 32 bytes, exactly one unaligned __m256i store.
    unsafe {
        let folded = _mm256_add_epi32(
            _mm512_castsi512_si256(acc),
            _mm512_extracti64x4_epi64::<1>(acc),
        );
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, folded);
    }
    let mut sum = 0i32;
    for l in lanes {
        sum = sum.wrapping_add(l);
    }
    sum
}

/// AVX-512F `f32` dot product over the full 32-element blocks of `a` and `b`
/// (`a.len() == b.len()`). Returns the four 8-lane accumulators in the same
/// layout as [`crate::portable::dot_f32_tail`] expects: accumulator `k` lane
/// `l` holds the sum over elements `32*i + 8*k + l`. Multiply and add are
/// separate roundings (no FMA), as in the portable path.
///
/// # Safety
/// AVX-512F must be enabled at compile time (the caller's `cfg` guarantees
/// this). Slices must have equal length.
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn dot_f32_blocks_512(a: &[f32], b: &[f32]) -> [tpt_simd_core::F32x8; 4] {
    use tpt_simd_core::F32x8;
    let blocks = a.len() / 32;
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut lo = _mm512_setzero_ps();
    let mut hi = _mm512_setzero_ps();
    for i in 0..blocks {
        // SAFETY: 32*i + 32 <= 32*blocks <= a.len() == b.len().
        unsafe {
            let x0 = _mm512_loadu_ps(pa.add(32 * i));
            let y0 = _mm512_loadu_ps(pb.add(32 * i));
            let x1 = _mm512_loadu_ps(pa.add(32 * i + 16));
            let y1 = _mm512_loadu_ps(pb.add(32 * i + 16));
            lo = _mm512_add_ps(lo, _mm512_mul_ps(x0, y0));
            hi = _mm512_add_ps(hi, _mm512_mul_ps(x1, y1));
        }
    }
    let mut out = [0.0f32; 32];
    // SAFETY: `out` is 128 bytes; both stores write 16 floats each.
    unsafe {
        _mm512_storeu_ps(out.as_mut_ptr(), lo);
        _mm512_storeu_ps(out.as_mut_ptr().add(16), hi);
    }
    core::array::from_fn(|k| F32x8::from_array(core::array::from_fn(|l| out[8 * k + l])))
}
