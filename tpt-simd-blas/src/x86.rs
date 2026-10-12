//! AVX2 + FMA gemm microkernels.
//!
//! Compiled when AVX2+FMA are enabled at compile time, or when the
//! `runtime-dispatch` feature is on (then [`available`] detects them once and
//! caches the answer).

use core::arch::x86_64::*;

/// `true` if the AVX2+FMA kernels may be called.
#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
pub(crate) fn available() -> bool {
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        true
    }
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    {
        tpt_simd_core::dispatch::x86_avx2_fma()
    }
}

/// `out[j*16 + i] = sum_p a[p*16 + i] * b[p*6 + j]` for a 16x6 tile.
///
/// Twelve independent accumulators (6 columns x 2 row vectors). An FMA takes
/// 4 cycles and two FMA ports are available, so at least 8 chains are needed
/// to keep them busy; 12 leaves slack for the loads and broadcasts.
///
/// # Safety
/// `a` must be valid for `16*kc` reads, `b` for `6*kc` reads, `out` for 96
/// writes. AVX2 and FMA must be available (see [`available`]).
#[cfg(not(target_feature = "avx512f"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn kernel_f32(kc: usize, a: *const f32, b: *const f32, out: *mut f32) {
    let mut c = [[_mm256_setzero_ps(); 2]; 6];
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 16);
            let bp = b.add(p * 6);
            let a0 = _mm256_loadu_ps(ap);
            let a1 = _mm256_loadu_ps(ap.add(8));
            for (j, cj) in c.iter_mut().enumerate() {
                let bj = _mm256_broadcast_ss(&*bp.add(j));
                cj[0] = _mm256_fmadd_ps(a0, bj, cj[0]);
                cj[1] = _mm256_fmadd_ps(a1, bj, cj[1]);
            }
        }
    }
    // SAFETY: 96 floats are writable at `out`.
    unsafe {
        for (j, cj) in c.iter().enumerate() {
            _mm256_storeu_ps(out.add(j * 16), cj[0]);
            _mm256_storeu_ps(out.add(j * 16 + 8), cj[1]);
        }
    }
}

/// `out[j*8 + i] = sum_p a[p*8 + i] * b[p*6 + j]` for an 8x6 tile.
///
/// Twelve independent accumulators (6 columns x 2 row vectors), for the same
/// latency reason as [`kernel_f32`].
///
/// # Safety
/// `a` must be valid for `8*kc` reads, `b` for `6*kc` reads, `out` for 48
/// writes. AVX2 and FMA must be available (see [`available`]).
#[cfg(not(target_feature = "avx512f"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn kernel_f64(kc: usize, a: *const f64, b: *const f64, out: *mut f64) {
    let mut c = [[_mm256_setzero_pd(); 2]; 6];
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 8);
            let bp = b.add(p * 6);
            let a0 = _mm256_loadu_pd(ap);
            let a1 = _mm256_loadu_pd(ap.add(4));
            for (j, cj) in c.iter_mut().enumerate() {
                let bj = _mm256_broadcast_sd(&*bp.add(j));
                cj[0] = _mm256_fmadd_pd(a0, bj, cj[0]);
                cj[1] = _mm256_fmadd_pd(a1, bj, cj[1]);
            }
        }
    }
    // SAFETY: 48 doubles are writable at `out`.
    unsafe {
        for (j, cj) in c.iter().enumerate() {
            _mm256_storeu_pd(out.add(j * 8), cj[0]);
            _mm256_storeu_pd(out.add(j * 8 + 4), cj[1]);
        }
    }
}

/// Full 16x6 tile that accumulates straight into C: `c[j*ldc + i] += alpha *
/// sum_p a[p*16 + i] * b[p*6 + j]`. Same loop as [`kernel_f32`], but the
/// result is stored with one FMA per vector instead of going through a
/// temporary and a scalar write-back, which is a large share of the work at
/// small `kc`.
///
/// # Safety
/// `a` must be valid for `16*kc` reads, `b` for `6*kc` reads, and `c` for
/// writes to a 16x6 tile with column stride `ldc` (16 contiguous elements per
/// column). AVX2 and FMA must be available (see [`available`]).
#[cfg(not(target_feature = "avx512f"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn kernel_f32_store(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
) {
    let mut acc = [[_mm256_setzero_ps(); 2]; 6];
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 16);
            let bp = b.add(p * 6);
            let a0 = _mm256_loadu_ps(ap);
            let a1 = _mm256_loadu_ps(ap.add(8));
            for (j, cj) in acc.iter_mut().enumerate() {
                let bj = _mm256_broadcast_ss(&*bp.add(j));
                cj[0] = _mm256_fmadd_ps(a0, bj, cj[0]);
                cj[1] = _mm256_fmadd_ps(a1, bj, cj[1]);
            }
        }
    }
    let av = _mm256_set1_ps(alpha);
    // SAFETY: the 16x6 tile at `c` is writable per the safety contract.
    unsafe {
        for (j, cj) in acc.iter().enumerate() {
            let col = c.add(j * ldc);
            let c0 = _mm256_loadu_ps(col);
            let c1 = _mm256_loadu_ps(col.add(8));
            _mm256_storeu_ps(col, _mm256_fmadd_ps(av, cj[0], c0));
            _mm256_storeu_ps(col.add(8), _mm256_fmadd_ps(av, cj[1], c1));
        }
    }
}

/// Full 8x6 `f64` tile that accumulates straight into C; see
/// [`kernel_f32_store`].
///
/// # Safety
/// `a` must be valid for `8*kc` reads, `b` for `6*kc` reads, and `c` for
/// writes to an 8x6 tile with column stride `ldc`. AVX2 and FMA must be
/// available (see [`available`]).
#[cfg(not(target_feature = "avx512f"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn kernel_f64_store(
    kc: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    ldc: usize,
    alpha: f64,
) {
    let mut acc = [[_mm256_setzero_pd(); 2]; 6];
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 8);
            let bp = b.add(p * 6);
            let a0 = _mm256_loadu_pd(ap);
            let a1 = _mm256_loadu_pd(ap.add(4));
            for (j, cj) in acc.iter_mut().enumerate() {
                let bj = _mm256_broadcast_sd(&*bp.add(j));
                cj[0] = _mm256_fmadd_pd(a0, bj, cj[0]);
                cj[1] = _mm256_fmadd_pd(a1, bj, cj[1]);
            }
        }
    }
    let av = _mm256_set1_pd(alpha);
    // SAFETY: the 8x6 tile at `c` is writable per the safety contract.
    unsafe {
        for (j, cj) in acc.iter().enumerate() {
            let col = c.add(j * ldc);
            let c0 = _mm256_loadu_pd(col);
            let c1 = _mm256_loadu_pd(col.add(4));
            _mm256_storeu_pd(col, _mm256_fmadd_pd(av, cj[0], c0));
            _mm256_storeu_pd(col.add(4), _mm256_fmadd_pd(av, cj[1], c1));
        }
    }
}

/// AVX-512F `f32` gemm microkernel: `out[j*32 + i] = sum_p a[p*32 + i] * b[p*4 + j]`
/// for a 32x4 tile. Two 512-bit rows per column give 8 accumulators, the same
/// count as the AVX2 kernel, each doing twice the work per instruction.
///
/// # Safety
/// `a` must be valid for `32*kc` reads, `b` for `4*kc` reads, `out` for 128
/// writes. AVX-512F and FMA must be enabled at compile time (the caller's
/// `cfg` guarantees this).
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f,fma")]
pub(crate) unsafe fn kernel_f32_512(kc: usize, a: *const f32, b: *const f32, out: *mut f32) {
    let mut c00 = _mm512_setzero_ps();
    let mut c01 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();
    let mut c20 = _mm512_setzero_ps();
    let mut c21 = _mm512_setzero_ps();
    let mut c30 = _mm512_setzero_ps();
    let mut c31 = _mm512_setzero_ps();
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 32);
            let bp = b.add(p * 4);
            let a0 = _mm512_loadu_ps(ap);
            let a1 = _mm512_loadu_ps(ap.add(16));
            let b0 = _mm512_set1_ps(*bp);
            let b1 = _mm512_set1_ps(*bp.add(1));
            let b2 = _mm512_set1_ps(*bp.add(2));
            let b3 = _mm512_set1_ps(*bp.add(3));
            c00 = _mm512_fmadd_ps(a0, b0, c00);
            c01 = _mm512_fmadd_ps(a1, b0, c01);
            c10 = _mm512_fmadd_ps(a0, b1, c10);
            c11 = _mm512_fmadd_ps(a1, b1, c11);
            c20 = _mm512_fmadd_ps(a0, b2, c20);
            c21 = _mm512_fmadd_ps(a1, b2, c21);
            c30 = _mm512_fmadd_ps(a0, b3, c30);
            c31 = _mm512_fmadd_ps(a1, b3, c31);
        }
    }
    // SAFETY: 128 floats are writable at `out`.
    unsafe {
        _mm512_storeu_ps(out, c00);
        _mm512_storeu_ps(out.add(16), c01);
        _mm512_storeu_ps(out.add(32), c10);
        _mm512_storeu_ps(out.add(48), c11);
        _mm512_storeu_ps(out.add(64), c20);
        _mm512_storeu_ps(out.add(80), c21);
        _mm512_storeu_ps(out.add(96), c30);
        _mm512_storeu_ps(out.add(112), c31);
    }
}

/// AVX-512F `f64` gemm microkernel: `out[j*16 + i] = sum_p a[p*16 + i] * b[p*4 + j]`
/// for a 16x4 tile (see [`kernel_f32_512`] for the accumulator layout).
///
/// # Safety
/// `a` must be valid for `16*kc` reads, `b` for `4*kc` reads, `out` for 64
/// writes. AVX-512F and FMA must be enabled at compile time.
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f,fma")]
pub(crate) unsafe fn kernel_f64_512(kc: usize, a: *const f64, b: *const f64, out: *mut f64) {
    let mut c00 = _mm512_setzero_pd();
    let mut c01 = _mm512_setzero_pd();
    let mut c10 = _mm512_setzero_pd();
    let mut c11 = _mm512_setzero_pd();
    let mut c20 = _mm512_setzero_pd();
    let mut c21 = _mm512_setzero_pd();
    let mut c30 = _mm512_setzero_pd();
    let mut c31 = _mm512_setzero_pd();
    for p in 0..kc {
        // SAFETY: p < kc, so all reads are within the documented extents.
        unsafe {
            let ap = a.add(p * 16);
            let bp = b.add(p * 4);
            let a0 = _mm512_loadu_pd(ap);
            let a1 = _mm512_loadu_pd(ap.add(8));
            let b0 = _mm512_set1_pd(*bp);
            let b1 = _mm512_set1_pd(*bp.add(1));
            let b2 = _mm512_set1_pd(*bp.add(2));
            let b3 = _mm512_set1_pd(*bp.add(3));
            c00 = _mm512_fmadd_pd(a0, b0, c00);
            c01 = _mm512_fmadd_pd(a1, b0, c01);
            c10 = _mm512_fmadd_pd(a0, b1, c10);
            c11 = _mm512_fmadd_pd(a1, b1, c11);
            c20 = _mm512_fmadd_pd(a0, b2, c20);
            c21 = _mm512_fmadd_pd(a1, b2, c21);
            c30 = _mm512_fmadd_pd(a0, b3, c30);
            c31 = _mm512_fmadd_pd(a1, b3, c31);
        }
    }
    // SAFETY: 64 doubles are writable at `out`.
    unsafe {
        _mm512_storeu_pd(out, c00);
        _mm512_storeu_pd(out.add(8), c01);
        _mm512_storeu_pd(out.add(16), c10);
        _mm512_storeu_pd(out.add(24), c11);
        _mm512_storeu_pd(out.add(32), c20);
        _mm512_storeu_pd(out.add(40), c21);
        _mm512_storeu_pd(out.add(48), c30);
        _mm512_storeu_pd(out.add(56), c31);
    }
}

/// AVX-512F level-1 block kernel for `f32`. `OP` selects the per-element
/// product: 0 = `x*y`, 1 = `x*x` (nrm2), 2 = `|x|` (asum). Returns the 32 lane
/// accumulators (lane `j` sums element `32*i + j` over the blocks `i`) and the
/// number of elements consumed (a multiple of 32). Multiply and add are separate
/// roundings, matching the portable `acc[j] += a[j] * b[j]`.
///
/// # Safety
/// AVX-512F must be enabled at compile time; `y.len() == x.len()` (for `OP != 0`
/// `y` is not read and may be `x`).
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn l1_f32<const OP: u8>(x: &[f32], y: &[f32]) -> ([f32; 32], usize) {
    let blocks = x.len() / 32;
    let (px, py) = (x.as_ptr(), y.as_ptr());
    let mut lo = _mm512_setzero_ps();
    let mut hi = _mm512_setzero_ps();
    for i in 0..blocks {
        // SAFETY: 32*i + 32 <= 32*blocks <= len for both slices.
        unsafe {
            let x0 = _mm512_loadu_ps(px.add(32 * i));
            let x1 = _mm512_loadu_ps(px.add(32 * i + 16));
            let (p0, p1) = match OP {
                0 => {
                    let y0 = _mm512_loadu_ps(py.add(32 * i));
                    let y1 = _mm512_loadu_ps(py.add(32 * i + 16));
                    (_mm512_mul_ps(x0, y0), _mm512_mul_ps(x1, y1))
                }
                1 => (_mm512_mul_ps(x0, x0), _mm512_mul_ps(x1, x1)),
                _ => {
                    let sign = _mm512_set1_epi32(i32::MIN);
                    let abs = |v: __m512| {
                        _mm512_castsi512_ps(_mm512_andnot_si512(sign, _mm512_castps_si512(v)))
                    };
                    (abs(x0), abs(x1))
                }
            };
            lo = _mm512_add_ps(lo, p0);
            hi = _mm512_add_ps(hi, p1);
        }
    }
    let mut out = [0.0f32; 32];
    // SAFETY: `out` is 128 bytes; each store writes 16 floats.
    unsafe {
        _mm512_storeu_ps(out.as_mut_ptr(), lo);
        _mm512_storeu_ps(out.as_mut_ptr().add(16), hi);
    }
    (out, blocks * 32)
}

/// AVX-512F level-1 block kernel for `f64`; see [`l1_f32`]. 16 lane
/// accumulators, two 8-lane registers.
///
/// # Safety
/// As for [`l1_f32`].
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn l1_f64<const OP: u8>(x: &[f64], y: &[f64]) -> ([f64; 16], usize) {
    let blocks = x.len() / 16;
    let (px, py) = (x.as_ptr(), y.as_ptr());
    let mut lo = _mm512_setzero_pd();
    let mut hi = _mm512_setzero_pd();
    for i in 0..blocks {
        // SAFETY: 16*i + 16 <= 16*blocks <= len for both slices.
        unsafe {
            let x0 = _mm512_loadu_pd(px.add(16 * i));
            let x1 = _mm512_loadu_pd(px.add(16 * i + 8));
            let (p0, p1) = match OP {
                0 => {
                    let y0 = _mm512_loadu_pd(py.add(16 * i));
                    let y1 = _mm512_loadu_pd(py.add(16 * i + 8));
                    (_mm512_mul_pd(x0, y0), _mm512_mul_pd(x1, y1))
                }
                1 => (_mm512_mul_pd(x0, x0), _mm512_mul_pd(x1, x1)),
                _ => {
                    let sign = _mm512_set1_epi64(i64::MIN);
                    let abs = |v: __m512d| {
                        _mm512_castsi512_pd(_mm512_andnot_si512(sign, _mm512_castpd_si512(v)))
                    };
                    (abs(x0), abs(x1))
                }
            };
            lo = _mm512_add_pd(lo, p0);
            hi = _mm512_add_pd(hi, p1);
        }
    }
    let mut out = [0.0f64; 16];
    // SAFETY: `out` is 128 bytes; each store writes 8 doubles.
    unsafe {
        _mm512_storeu_pd(out.as_mut_ptr(), lo);
        _mm512_storeu_pd(out.as_mut_ptr().add(8), hi);
    }
    (out, blocks * 16)
}
