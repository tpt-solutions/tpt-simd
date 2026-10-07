//! AVX2 + FMA gemm microkernels.
//!
//! Compiled when AVX2+FMA are enabled at compile time, or when the
//! `runtime-dispatch` feature is on (then [`available`] detects them once and
//! caches the answer).

use core::arch::x86_64::*;

/// `true` if the AVX2+FMA kernels may be called.
#[inline(always)]
pub(crate) fn available() -> bool {
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        true
    }
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    {
        use core::sync::atomic::{AtomicU8, Ordering};
        // 0 = unknown, 1 = no, 2 = yes
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => {
                let yes =
                    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
                STATE.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
                yes
            }
        }
    }
}

/// `out[j*16 + i] = sum_p a[p*16 + i] * b[p*4 + j]` for a 16x4 tile.
///
/// # Safety
/// `a` must be valid for `16*kc` reads, `b` for `4*kc` reads, `out` for 64
/// writes. AVX2 and FMA must be available (see [`available`]).
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn kernel_f32(kc: usize, a: *const f32, b: *const f32, out: *mut f32) {
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 16);
            let bp = b.add(p * 4);
            let a0 = _mm256_loadu_ps(ap);
            let a1 = _mm256_loadu_ps(ap.add(8));
            let b0 = _mm256_broadcast_ss(&*bp);
            let b1 = _mm256_broadcast_ss(&*bp.add(1));
            let b2 = _mm256_broadcast_ss(&*bp.add(2));
            let b3 = _mm256_broadcast_ss(&*bp.add(3));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a1, b0, c01);
            c10 = _mm256_fmadd_ps(a0, b1, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a0, b2, c20);
            c21 = _mm256_fmadd_ps(a1, b2, c21);
            c30 = _mm256_fmadd_ps(a0, b3, c30);
            c31 = _mm256_fmadd_ps(a1, b3, c31);
        }
    }
    // SAFETY: 64 floats are writable at `out`.
    unsafe {
        _mm256_storeu_ps(out, c00);
        _mm256_storeu_ps(out.add(8), c01);
        _mm256_storeu_ps(out.add(16), c10);
        _mm256_storeu_ps(out.add(24), c11);
        _mm256_storeu_ps(out.add(32), c20);
        _mm256_storeu_ps(out.add(40), c21);
        _mm256_storeu_ps(out.add(48), c30);
        _mm256_storeu_ps(out.add(56), c31);
    }
}

/// `out[j*8 + i] = sum_p a[p*8 + i] * b[p*4 + j]` for an 8x4 tile.
///
/// # Safety
/// `a` must be valid for `8*kc` reads, `b` for `4*kc` reads, `out` for 32
/// writes. AVX2 and FMA must be available (see [`available`]).
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn kernel_f64(kc: usize, a: *const f64, b: *const f64, out: *mut f64) {
    let mut c00 = _mm256_setzero_pd();
    let mut c01 = _mm256_setzero_pd();
    let mut c10 = _mm256_setzero_pd();
    let mut c11 = _mm256_setzero_pd();
    let mut c20 = _mm256_setzero_pd();
    let mut c21 = _mm256_setzero_pd();
    let mut c30 = _mm256_setzero_pd();
    let mut c31 = _mm256_setzero_pd();
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 8);
            let bp = b.add(p * 4);
            let a0 = _mm256_loadu_pd(ap);
            let a1 = _mm256_loadu_pd(ap.add(4));
            let b0 = _mm256_broadcast_sd(&*bp);
            let b1 = _mm256_broadcast_sd(&*bp.add(1));
            let b2 = _mm256_broadcast_sd(&*bp.add(2));
            let b3 = _mm256_broadcast_sd(&*bp.add(3));
            c00 = _mm256_fmadd_pd(a0, b0, c00);
            c01 = _mm256_fmadd_pd(a1, b0, c01);
            c10 = _mm256_fmadd_pd(a0, b1, c10);
            c11 = _mm256_fmadd_pd(a1, b1, c11);
            c20 = _mm256_fmadd_pd(a0, b2, c20);
            c21 = _mm256_fmadd_pd(a1, b2, c21);
            c30 = _mm256_fmadd_pd(a0, b3, c30);
            c31 = _mm256_fmadd_pd(a1, b3, c31);
        }
    }
    // SAFETY: 32 doubles are writable at `out`.
    unsafe {
        _mm256_storeu_pd(out, c00);
        _mm256_storeu_pd(out.add(4), c01);
        _mm256_storeu_pd(out.add(8), c10);
        _mm256_storeu_pd(out.add(12), c11);
        _mm256_storeu_pd(out.add(16), c20);
        _mm256_storeu_pd(out.add(20), c21);
        _mm256_storeu_pd(out.add(24), c30);
        _mm256_storeu_pd(out.add(28), c31);
    }
}
