//! SIMD transpose, interleave and permutation helpers.
//!
//! Implemented:
//! * [`transpose_8x8_i16`], [`transpose_4x4_f32`] — in-place block
//!   transposes (SSE2 unpack networks on x86_64, scalar elsewhere).
//! * [`interleave_stereo_i16`], [`deinterleave_stereo_i16`] — `LLLL/RRRR`
//!   planar <-> `LRLRLR` packed audio.
//! * [`interleave_yuv420`] — planar I420 to semi-planar NV12.
//! * [`unpack_i8_to_i16`] — sign-extending widen of a byte slice.
//! * [`permute_f32`], [`permute_i32`], [`permute`] — arbitrary lane
//!   permutation (`vpermps` / `vpermd` equivalent).
//! * [`pack_i16_to_i8`], [`pack_i16_to_i8_slice`] — re-exported from
//!   `tpt-simd-saturate`, their home crate.
//!
//! ## Length-mismatch policy
//!
//! Following ADR 0002, slice functions never return errors: every slice
//! length relationship is checked up front with an `assert!` and a mismatch
//! panics with a descriptive message (documented under `# Panics` on each
//! function). Lengths must match **exactly**; there is no silent truncation.
//! Empty slices are valid. Tails that do not fill a whole vector are handled
//! by the same scalar code as the SIMD body, so results are identical for
//! every length.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

pub use tpt_simd_saturate::{pack_i16_to_i8, pack_i16_to_i8_slice};

use tpt_simd_core::{Simd, SimdElement};

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "sse2"
))]
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Transposes
// ---------------------------------------------------------------------------

/// Portable reference for [`transpose_8x8_i16`].
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_permute::transpose_8x8_i16_portable;
/// let mut m = [[0i16; 8]; 8];
/// m[1][2] = 7;
/// transpose_8x8_i16_portable(&mut m);
/// assert_eq!(m[2][1], 7);
/// ```
#[inline]
pub fn transpose_8x8_i16_portable(data: &mut [[i16; 8]; 8]) {
    for i in 0..8 {
        for j in (i + 1)..8 {
            let t = data[i][j];
            data[i][j] = data[j][i];
            data[j][i] = t;
        }
    }
}

/// Transposes an 8x8 `i16` block in place (`data[i][j] <-> data[j][i]`).
///
/// This is the building block of 2-D video DCTs (Appendix B.4).
///
/// Performance: on x86_64 a three-stage `punpck{l,h}{wd,dq,qdq}` network on
/// eight XMM registers (24 shuffles, no memory traffic besides the 8 loads
/// and 8 stores); with AVX2 enabled at compile time, two rows per YMM register
/// (12 shuffles, about 1.5x faster again on independent blocks); scalar swap
/// loop elsewhere.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_permute::transpose_8x8_i16;
/// let mut block = [[0i16; 8]; 8];
/// for r in 0..8 { for c in 0..8 { block[r][c] = (r * 8 + c) as i16; } }
/// transpose_8x8_i16(&mut block);
/// assert_eq!(block[0], [0, 8, 16, 24, 32, 40, 48, 56]);
/// assert_eq!(block[7][7], 63);
/// ```
#[inline]
pub fn transpose_8x8_i16(data: &mut [[i16; 8]; 8]) {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "sse2"
    ))]
    {
        #[cfg(target_feature = "avx2")]
        {
            transpose_8x8_i16_avx2(data)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            transpose_8x8_i16_sse2(data)
        }
    }
    #[cfg(all(not(feature = "scalar-only"), target_arch = "aarch64"))]
    {
        neon::transpose_8x8_i16(data)
    }
    #[cfg(not(any(
        all(
            not(feature = "scalar-only"),
            target_arch = "x86_64",
            target_feature = "sse2"
        ),
        all(not(feature = "scalar-only"), target_arch = "aarch64")
    )))]
    {
        transpose_8x8_i16_portable(data)
    }
}

/// aarch64 NEON transpose. Three `vtrn` stages (16-bit pairs, then 32-bit
/// pairs) leave, for each column pair `{c, c+4}` (or `{2,6}`, `{1,5}`,
/// `{3,7}`), one vector holding that column for rows 0-3 in its low half and
/// rows 4-7 in its high half, two vectors per group. `vcombine` then yields
/// each output row. Values are moved, not computed, so the result is exact.
#[cfg(all(not(feature = "scalar-only"), target_arch = "aarch64"))]
mod neon {
    use core::arch::aarch64::*;

    pub(super) fn transpose_8x8_i16(data: &mut [[i16; 8]; 8]) {
        let p = data.as_mut_ptr().cast::<i16>();
        // SAFETY: NEON is enabled on aarch64. `data` is 128 contiguous bytes;
        // each row is one 16-byte load and one 16-byte store at element
        // offset 8 * k.
        unsafe {
            let r = |k: usize| vld1q_s16(p.add(8 * k));
            let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
            let (r4, r5, r6, r7) = (r(4), r(5), r(6), r(7));
            let s32 = vreinterpretq_s32_s16;
            let s16 = vreinterpretq_s16_s32;

            let t0 = vtrn1q_s16(r0, r1);
            let t1 = vtrn2q_s16(r0, r1);
            let t2 = vtrn1q_s16(r2, r3);
            let t3 = vtrn2q_s16(r2, r3);
            let t4 = vtrn1q_s16(r4, r5);
            let t5 = vtrn2q_s16(r4, r5);
            let t6 = vtrn1q_s16(r6, r7);
            let t7 = vtrn2q_s16(r6, r7);

            // Columns {0,4}, {2,6}, {1,5}, {3,7}; rows 0-3 in `*`, rows 4-7 in `*2`.
            let a = s16(vtrn1q_s32(s32(t0), s32(t2)));
            let b = s16(vtrn2q_s32(s32(t0), s32(t2)));
            let c = s16(vtrn1q_s32(s32(t1), s32(t3)));
            let d = s16(vtrn2q_s32(s32(t1), s32(t3)));
            let a2 = s16(vtrn1q_s32(s32(t4), s32(t6)));
            let b2 = s16(vtrn2q_s32(s32(t4), s32(t6)));
            let c2 = s16(vtrn1q_s32(s32(t5), s32(t7)));
            let d2 = s16(vtrn2q_s32(s32(t5), s32(t7)));

            let out = |k: usize, lo: int16x8_t, hi: int16x8_t| {
                vst1q_s16(
                    p.add(8 * k),
                    vcombine_s16(vget_low_s16(lo), vget_low_s16(hi)),
                );
            };
            let out_hi = |k: usize, lo: int16x8_t, hi: int16x8_t| {
                vst1q_s16(
                    p.add(8 * k),
                    vcombine_s16(vget_high_s16(lo), vget_high_s16(hi)),
                );
            };
            out(0, a, a2); // column 0
            out(1, c, c2); // column 1
            out(2, b, b2); // column 2
            out(3, d, d2); // column 3
            out_hi(4, a, a2); // column 4
            out_hi(5, c, c2); // column 5
            out_hi(6, b, b2); // column 6
            out_hi(7, d, d2); // column 7
        }
    }
}

/// AVX2 8x8 `i16` transpose: two rows per `__m256i`, so the 24 SSE2 shuffles
/// become 12 (4 + 4 `vpunpck` and 4 `vpermq`), halving pressure on the single
/// shuffle port. Bit-identical to the SSE2 and portable paths.
#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline]
fn transpose_8x8_i16_avx2(data: &mut [[i16; 8]; 8]) {
    let p = data.as_mut_ptr().cast::<__m128i>();
    // SAFETY: AVX2 (and therefore SSE2) is enabled at compile time (cfg
    // above). `data` is 128 contiguous bytes, so the eight 16-byte loads at
    // `p.add(0..8)` and the four 32-byte stores at `p.add(0/2/4/6)` are in
    // bounds (unaligned variants are used).
    unsafe {
        // y_k = [row k | row k+4]
        let y0 = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(p)),
            _mm_loadu_si128(p.add(4)),
        );
        let y1 = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(p.add(1))),
            _mm_loadu_si128(p.add(5)),
        );
        let y2 = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(p.add(2))),
            _mm_loadu_si128(p.add(6)),
        );
        let y3 = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(p.add(3))),
            _mm_loadu_si128(p.add(7)),
        );

        let a0 = _mm256_unpacklo_epi16(y0, y1);
        let a1 = _mm256_unpackhi_epi16(y0, y1);
        let a2 = _mm256_unpacklo_epi16(y2, y3);
        let a3 = _mm256_unpackhi_epi16(y2, y3);

        // b_k lane 0 holds rows 0..3 of columns 2k, 2k+1; lane 1 rows 4..7.
        let b0 = _mm256_unpacklo_epi32(a0, a2);
        let b1 = _mm256_unpackhi_epi32(a0, a2);
        let b2 = _mm256_unpacklo_epi32(a1, a3);
        let b3 = _mm256_unpackhi_epi32(a1, a3);

        // Gather qwords [l0.lo, l1.lo, l0.hi, l1.hi] = (row 2k | row 2k+1).
        let q = p.cast::<__m256i>();
        _mm256_storeu_si256(q, _mm256_permute4x64_epi64::<0xD8>(b0));
        _mm256_storeu_si256(q.add(1), _mm256_permute4x64_epi64::<0xD8>(b1));
        _mm256_storeu_si256(q.add(2), _mm256_permute4x64_epi64::<0xD8>(b2));
        _mm256_storeu_si256(q.add(3), _mm256_permute4x64_epi64::<0xD8>(b3));
    }
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
#[inline]
fn transpose_8x8_i16_sse2(data: &mut [[i16; 8]; 8]) {
    let p = data.as_mut_ptr().cast::<__m128i>();
    // SAFETY: SSE2 is enabled at compile time (cfg above). `data` is 128
    // contiguous bytes, so the eight 16-byte unaligned loads at `p.add(0..8)`
    // and the matching stores are in bounds.
    unsafe {
        let r0 = _mm_loadu_si128(p);
        let r1 = _mm_loadu_si128(p.add(1));
        let r2 = _mm_loadu_si128(p.add(2));
        let r3 = _mm_loadu_si128(p.add(3));
        let r4 = _mm_loadu_si128(p.add(4));
        let r5 = _mm_loadu_si128(p.add(5));
        let r6 = _mm_loadu_si128(p.add(6));
        let r7 = _mm_loadu_si128(p.add(7));

        let a0 = _mm_unpacklo_epi16(r0, r1);
        let a1 = _mm_unpackhi_epi16(r0, r1);
        let a2 = _mm_unpacklo_epi16(r2, r3);
        let a3 = _mm_unpackhi_epi16(r2, r3);
        let a4 = _mm_unpacklo_epi16(r4, r5);
        let a5 = _mm_unpackhi_epi16(r4, r5);
        let a6 = _mm_unpacklo_epi16(r6, r7);
        let a7 = _mm_unpackhi_epi16(r6, r7);

        let b0 = _mm_unpacklo_epi32(a0, a2);
        let b1 = _mm_unpackhi_epi32(a0, a2);
        let b2 = _mm_unpacklo_epi32(a1, a3);
        let b3 = _mm_unpackhi_epi32(a1, a3);
        let b4 = _mm_unpacklo_epi32(a4, a6);
        let b5 = _mm_unpackhi_epi32(a4, a6);
        let b6 = _mm_unpacklo_epi32(a5, a7);
        let b7 = _mm_unpackhi_epi32(a5, a7);

        _mm_storeu_si128(p, _mm_unpacklo_epi64(b0, b4));
        _mm_storeu_si128(p.add(1), _mm_unpackhi_epi64(b0, b4));
        _mm_storeu_si128(p.add(2), _mm_unpacklo_epi64(b1, b5));
        _mm_storeu_si128(p.add(3), _mm_unpackhi_epi64(b1, b5));
        _mm_storeu_si128(p.add(4), _mm_unpacklo_epi64(b2, b6));
        _mm_storeu_si128(p.add(5), _mm_unpackhi_epi64(b2, b6));
        _mm_storeu_si128(p.add(6), _mm_unpacklo_epi64(b3, b7));
        _mm_storeu_si128(p.add(7), _mm_unpackhi_epi64(b3, b7));
    }
}

/// Portable reference for [`transpose_4x4_f32`].
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_permute::transpose_4x4_f32_portable;
/// let mut m = [[0.0f32; 4]; 4];
/// m[0][3] = 1.5;
/// transpose_4x4_f32_portable(&mut m);
/// assert_eq!(m[3][0], 1.5);
/// ```
#[inline]
pub fn transpose_4x4_f32_portable(data: &mut [[f32; 4]; 4]) {
    for i in 0..4 {
        for j in (i + 1)..4 {
            let t = data[i][j];
            data[i][j] = data[j][i];
            data[j][i] = t;
        }
    }
}

/// Transposes a 4x4 `f32` block in place. Values (including NaN payloads
/// and signed zeros) are moved bit-for-bit.
///
/// Performance: 4 `unpck{l,h}ps` + 4 `movlhps`/`movhlps` on x86_64; scalar
/// swaps elsewhere.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_permute::transpose_4x4_f32;
/// let mut m = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
///              [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]];
/// transpose_4x4_f32(&mut m);
/// assert_eq!(m[0], [1.0, 5.0, 9.0, 13.0]);
/// ```
#[inline]
pub fn transpose_4x4_f32(data: &mut [[f32; 4]; 4]) {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "sse2"
    ))]
    {
        let p = data.as_mut_ptr().cast::<f32>();
        // SAFETY: SSE (part of SSE2) is enabled at compile time. `data` is 16
        // contiguous f32, so the four 4-lane unaligned loads/stores at
        // offsets 0, 4, 8, 12 are in bounds.
        unsafe {
            let r0 = _mm_loadu_ps(p);
            let r1 = _mm_loadu_ps(p.add(4));
            let r2 = _mm_loadu_ps(p.add(8));
            let r3 = _mm_loadu_ps(p.add(12));
            let t0 = _mm_unpacklo_ps(r0, r1);
            let t1 = _mm_unpackhi_ps(r0, r1);
            let t2 = _mm_unpacklo_ps(r2, r3);
            let t3 = _mm_unpackhi_ps(r2, r3);
            _mm_storeu_ps(p, _mm_movelh_ps(t0, t2));
            _mm_storeu_ps(p.add(4), _mm_movehl_ps(t2, t0));
            _mm_storeu_ps(p.add(8), _mm_movelh_ps(t1, t3));
            _mm_storeu_ps(p.add(12), _mm_movehl_ps(t3, t1));
        }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "sse2"
    )))]
    {
        transpose_4x4_f32_portable(data)
    }
}

// ---------------------------------------------------------------------------
// Interleave / deinterleave
// ---------------------------------------------------------------------------

/// Interleaves two mono channels into packed stereo:
/// `output = [l0, r0, l1, r1, ...]`.
///
/// Performance: `punpcklwd`/`punpckhwd` after auto-vectorisation of the
/// pair loop.
///
/// # Panics
/// If `left.len() != right.len()` or `output.len() != 2 * left.len()`.
///
/// ```
/// use tpt_simd_permute::interleave_stereo_i16;
/// let mut out = [0i16; 6];
/// interleave_stereo_i16(&[1, 2, 3], &[-1, -2, -3], &mut out);
/// assert_eq!(out, [1, -1, 2, -2, 3, -3]);
/// ```
#[inline]
pub fn interleave_stereo_i16(left: &[i16], right: &[i16], output: &mut [i16]) {
    assert_eq!(left.len(), right.len(), "left and right length mismatch");
    assert_eq!(
        output.len(),
        left.len() * 2,
        "output length must be 2 * left.len()"
    );
    for ((o, &l), &r) in output.chunks_exact_mut(2).zip(left).zip(right) {
        o[0] = l;
        o[1] = r;
    }
}

/// Splits packed stereo into two channels:
/// `left = [i0, i2, ...]`, `right = [i1, i3, ...]`. Inverse of
/// [`interleave_stereo_i16`].
///
/// # Panics
/// If `input.len()` is odd, or `left.len() != right.len()`, or
/// `input.len() != 2 * left.len()`.
///
/// ```
/// use tpt_simd_permute::deinterleave_stereo_i16;
/// let (mut l, mut r) = ([0i16; 3], [0i16; 3]);
/// deinterleave_stereo_i16(&[1, -1, 2, -2, 3, -3], &mut l, &mut r);
/// assert_eq!((l, r), ([1, 2, 3], [-1, -2, -3]));
/// ```
#[inline]
pub fn deinterleave_stereo_i16(input: &[i16], left: &mut [i16], right: &mut [i16]) {
    assert!(
        input.len().is_multiple_of(2),
        "stereo input length must be even"
    );
    assert_eq!(left.len(), right.len(), "left and right length mismatch");
    assert_eq!(
        input.len(),
        left.len() * 2,
        "input length must be 2 * left.len()"
    );
    for ((i, l), r) in input.chunks_exact(2).zip(left).zip(right) {
        *l = i[0];
        *r = i[1];
    }
}

/// Converts planar YUV 4:2:0 (I420: Y, U, V planes) to semi-planar NV12:
/// the Y plane is copied unchanged, followed by the chroma samples
/// interleaved as `u0, v0, u1, v1, ...`.
///
/// No image width is needed because the conversion is a plain plane
/// concatenation: `y.len()` must be `4 * u.len()` (one chroma sample per
/// 2x2 luma block).
///
/// # Panics
/// If `u.len() != v.len()`, `y.len() != 4 * u.len()`, or
/// `output.len() != y.len() + 2 * u.len()`.
///
/// ```
/// use tpt_simd_permute::interleave_yuv420;
/// let y = [10u8, 11, 12, 13];
/// let mut out = [0u8; 6];
/// interleave_yuv420(&y, &[20], &[30], &mut out);
/// assert_eq!(out, [10, 11, 12, 13, 20, 30]);
/// ```
#[inline]
pub fn interleave_yuv420(y: &[u8], u: &[u8], v: &[u8], output: &mut [u8]) {
    assert_eq!(u.len(), v.len(), "u and v length mismatch");
    assert_eq!(y.len(), u.len() * 4, "y length must be 4 * u.len()");
    assert_eq!(
        output.len(),
        y.len() + 2 * u.len(),
        "output length must be y.len() + 2 * u.len()"
    );
    let (yo, uvo) = output.split_at_mut(y.len());
    yo.copy_from_slice(y);
    for ((o, &a), &b) in uvo.chunks_exact_mut(2).zip(u).zip(v) {
        o[0] = a;
        o[1] = b;
    }
}

// ---------------------------------------------------------------------------
// Unpack
// ---------------------------------------------------------------------------

/// Sign-extends each `i8` to `i16` (`pmovsxbw`).
///
/// # Panics
/// If `data.len() != output.len()`.
///
/// ```
/// use tpt_simd_permute::unpack_i8_to_i16;
/// let mut out = [0i16; 3];
/// unpack_i8_to_i16(&[-128, 0, 127], &mut out);
/// assert_eq!(out, [-128, 0, 127]);
/// ```
#[inline]
pub fn unpack_i8_to_i16(data: &[i8], output: &mut [i16]) {
    assert_eq!(data.len(), output.len(), "data and output length mismatch");
    // Explicit AVX-512BW path: `vpmovsxbw` converts 32 bytes per instruction.
    // Measured faster than LLVM's autovectorised loop on the native build,
    // which is slower there than on the AVX2 build.
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx512bw"
    ))]
    {
        // SAFETY: AVX-512BW is enabled at compile time; the lengths are equal.
        unsafe { unpack_i8_to_i16_avx512bw(data, output) }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx512bw"
    )))]
    {
        for (o, &x) in output.iter_mut().zip(data) {
            *o = i16::from(x);
        }
    }
}

/// AVX-512BW body of [`unpack_i8_to_i16`]: 32 bytes in, 32 `i16` out per
/// iteration, scalar tail for the remainder.
///
/// # Safety
/// AVX-512BW must be enabled at compile time, and `data.len() == output.len()`.
#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512bw")]
unsafe fn unpack_i8_to_i16_avx512bw(data: &[i8], output: &mut [i16]) {
    use core::arch::x86_64::*;
    let n = data.len();
    let blocks = n / 32;
    for i in 0..blocks {
        // SAFETY: 32*i + 32 <= 32*blocks <= n, so the 32-byte load and the
        // 64-byte store (32 i16) stay in bounds of `data` and `output`.
        unsafe {
            let x = _mm256_loadu_si256(data.as_ptr().add(32 * i).cast());
            let y = _mm512_cvtepi8_epi16(x);
            _mm512_storeu_si512(output.as_mut_ptr().add(32 * i).cast(), y);
        }
    }
    for k in 32 * blocks..n {
        output[k] = i16::from(data[k]);
    }
}

// ---------------------------------------------------------------------------
// General permutation
// ---------------------------------------------------------------------------

/// Generic lane permutation: `result[i] = v[idx[i] % N]`.
///
/// Out-of-range indices wrap modulo `N` (for power-of-two `N` this is the
/// same "use the low bits" rule as `vpermps`), so the function never panics.
///
/// ```
/// use tpt_simd_core::Simd;
/// use tpt_simd_permute::permute;
/// let v = Simd::<i16, 4>::from_array([10, 20, 30, 40]);
/// let r = permute(v, Simd::from_array([3, 2, 1, 5]));
/// assert_eq!(r.to_array(), [40, 30, 20, 20]);
/// ```
#[inline]
pub fn permute<T: SimdElement, const N: usize>(v: Simd<T, N>, idx: Simd<u32, N>) -> Simd<T, N> {
    Simd::from_fn(|i| v[(idx[i] as usize) % N])
}

/// `result[i] = v[idx[i] & 7]` for 8 `f32` lanes (`_mm256_permutevar8x32_ps`).
/// Lanes are moved bit-for-bit.
///
/// Performance: one `vpermps` on AVX2; portable gather loop otherwise.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{F32x8, Simd};
/// use tpt_simd_permute::permute_f32;
/// let v = F32x8::from_fn(|i| i as f32);
/// let r = permute_f32(v, Simd::from_array([7, 6, 5, 4, 3, 2, 1, 0]));
/// assert_eq!(r.to_array(), [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
/// ```
#[inline]
pub fn permute_f32(v: Simd<f32, 8>, idx: Simd<u32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        let (a, i) = (v.to_array(), idx.to_array());
        let mut out = [0.0f32; 8];
        // SAFETY: AVX2 is enabled at compile time (cfg above). All pointers
        // refer to 8-element (32-byte) arrays and access is unaligned.
        unsafe {
            let r = _mm256_permutevar8x32_ps(
                _mm256_loadu_ps(a.as_ptr()),
                _mm256_loadu_si256(i.as_ptr().cast()),
            );
            _mm256_storeu_ps(out.as_mut_ptr(), r);
        }
        Simd::from_array(out)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        permute_f32_portable(v, idx)
    }
}

/// Portable reference for [`permute_f32`].
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{F32x8, Simd};
/// use tpt_simd_permute::permute_f32_portable;
/// let r = permute_f32_portable(F32x8::from_fn(|i| i as f32), Simd::splat(9));
/// assert_eq!(r.to_array(), [1.0; 8]);
/// ```
#[inline]
pub fn permute_f32_portable(v: Simd<f32, 8>, idx: Simd<u32, 8>) -> Simd<f32, 8> {
    Simd::from_fn(|i| v[(idx[i] & 7) as usize])
}

/// `result[i] = v[idx[i] & 7]` for 8 `i32` lanes (`_mm256_permutevar8x32_epi32`).
///
/// Performance: one `vpermd` on AVX2; portable gather loop otherwise.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, Simd};
/// use tpt_simd_permute::permute_i32;
/// let v = I32x8::from_fn(|i| i as i32 * 10);
/// let r = permute_i32(v, Simd::splat(3));
/// assert_eq!(r.to_array(), [30; 8]);
/// ```
#[inline]
pub fn permute_i32(v: Simd<i32, 8>, idx: Simd<u32, 8>) -> Simd<i32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        let (a, i) = (v.to_array(), idx.to_array());
        let mut out = [0i32; 8];
        // SAFETY: AVX2 is enabled at compile time (cfg above). All pointers
        // refer to 8-element (32-byte) arrays and access is unaligned.
        unsafe {
            let r = _mm256_permutevar8x32_epi32(
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(i.as_ptr().cast()),
            );
            _mm256_storeu_si256(out.as_mut_ptr().cast(), r);
        }
        Simd::from_array(out)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        permute_i32_portable(v, idx)
    }
}

/// Portable reference for [`permute_i32`].
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, Simd};
/// use tpt_simd_permute::permute_i32_portable;
/// let r = permute_i32_portable(I32x8::from_fn(|i| i as i32), Simd::splat(10));
/// assert_eq!(r.to_array(), [2; 8]);
/// ```
#[inline]
pub fn permute_i32_portable(v: Simd<i32, 8>, idx: Simd<u32, 8>) -> Simd<i32, 8> {
    Simd::from_fn(|i| v[(idx[i] & 7) as usize])
}

#[cfg(test)]
mod tests;
