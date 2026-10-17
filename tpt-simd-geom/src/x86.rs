//! AVX2 ray-box slab test on the SoA layout. Bit-identical to the scalar loop in
//! `lib.rs`. Each `if` in the scalar code becomes a compare and a blend with the same
//! choice, not `min_ps` / `max_ps`, so that ties and signed zeros come out the same.

use core::arch::x86_64::*;

/// Slab test for whole groups of eight boxes, starting at box 0. Writes
/// `out[i]` for `i < 8 * k` and returns that count `8 * k`; the caller does the rest.
///
/// # Safety
/// The CPU must support AVX2. Every slice in `mins` and `maxs` must hold at
/// least `out.len()` values (checked by the caller).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ray_aabb(
    mins: [&[f32]; 3],
    maxs: [&[f32]; 3],
    origin: [f32; 3],
    inv: [f32; 3],
    parallel: [bool; 3],
    t_max: f32,
    out: &mut [f32],
) -> usize {
    let inf = _mm256_set1_ps(f32::INFINITY);
    let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
    let zero = _mm256_setzero_ps();
    let tmax = _mm256_set1_ps(t_max);
    let n = out.len() - out.len() % 8;
    let mut i = 0;
    while i < n {
        // SAFETY: i + 8 <= n <= out.len(), and every box slice holds out.len() values.
        unsafe {
            let mut tnear = ninf;
            let mut tfar = inf;
            // All-ones bits; cleared by a miss on a parallel axis.
            let mut hit = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            for a in 0..3 {
                let o = _mm256_set1_ps(origin[a]);
                let mn = _mm256_loadu_ps(mins[a].as_ptr().add(i));
                let mx = _mm256_loadu_ps(maxs[a].as_ptr().add(i));
                if parallel[a] {
                    // Scalar: miss if origin < min || origin > max. Ordered compares
                    // are false on NaN, the same as the scalar operators.
                    let miss = _mm256_or_ps(
                        _mm256_cmp_ps::<_CMP_LT_OQ>(o, mn),
                        _mm256_cmp_ps::<_CMP_GT_OQ>(o, mx),
                    );
                    hit = _mm256_andnot_ps(miss, hit);
                } else {
                    let iv = _mm256_set1_ps(inv[a]);
                    let t1 = _mm256_mul_ps(_mm256_sub_ps(mn, o), iv);
                    let t2 = _mm256_mul_ps(_mm256_sub_ps(mx, o), iv);
                    // Scalar: (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) }.
                    let swap = _mm256_cmp_ps::<_CMP_LT_OQ>(t1, t2);
                    let lo = _mm256_blendv_ps(t2, t1, swap);
                    let hi = _mm256_blendv_ps(t1, t2, swap);
                    // Scalar: if lo > tnear { tnear = lo }, and likewise for tfar.
                    tnear = _mm256_blendv_ps(tnear, lo, _mm256_cmp_ps::<_CMP_GT_OQ>(lo, tnear));
                    tfar = _mm256_blendv_ps(tfar, hi, _mm256_cmp_ps::<_CMP_LT_OQ>(hi, tfar));
                }
            }
            // Scalar: hit && tnear <= tfar && tfar >= 0 && tnear <= t_max.
            let ok = _mm256_and_ps(
                _mm256_and_ps(hit, _mm256_cmp_ps::<_CMP_LE_OQ>(tnear, tfar)),
                _mm256_and_ps(
                    _mm256_cmp_ps::<_CMP_GE_OQ>(tfar, zero),
                    _mm256_cmp_ps::<_CMP_LE_OQ>(tnear, tmax),
                ),
            );
            // Scalar: if tnear < 0 { 0 } else { tnear }.
            let t = _mm256_blendv_ps(tnear, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(tnear, zero));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_blendv_ps(inf, t, ok));
        }
        i += 8;
    }
    n
}
