//! Storage element types usable as the raw representation of a [`Fixed`](crate::Fixed).
//!
//! The scalar lane algorithms live here so the portable and the AVX2 paths
//! share one definition of the semantics.

use tpt_simd_core::SimdInt;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for i16 {}
}

/// Multiply mode: round to nearest (ties away from zero), saturate to the format range.
pub const MUL_ROUND_SAT: u8 = 0;
/// Multiply mode: round to nearest (ties away from zero), wrap to the storage width.
pub const MUL_ROUND_WRAP: u8 = 1;
/// Multiply mode: truncate (arithmetic shift, floor), wrap to the storage width.
pub const MUL_TRUNC_WRAP: u8 = 2;

/// Smallest raw value of an `I.F` format (`I` counts the sign bit), widened.
#[inline(always)]
pub(crate) const fn min_w(i: u32, f: u32) -> i64 {
    -(1i64 << (i + f - 1))
}

/// Largest raw value of an `I.F` format, widened.
#[inline(always)]
pub(crate) const fn max_w(i: u32, f: u32) -> i64 {
    (1i64 << (i + f - 1)) - 1
}

/// Right shift by `f` with round-to-nearest, ties away from zero.
#[inline(always)]
pub(crate) const fn shr_round_away(w: i64, f: u32) -> i64 {
    if f == 0 {
        w
    } else {
        (w + (1i64 << (f - 1)) - (w < 0) as i64) >> f
    }
}

/// Scalar lane multiply of two widened raw values (see the `MUL_*` modes).
/// The result is already saturated for [`MUL_ROUND_SAT`]; the caller
/// narrows (wraps) it to the storage type.
#[inline(always)]
pub(crate) fn mul_lane<const I: u32, const F: u32, const MODE: u8>(a: i64, b: i64) -> i64 {
    let p = a * b;
    match MODE {
        MUL_ROUND_SAT => shr_round_away(p, F).clamp(min_w(I, F), max_w(I, F)),
        MUL_ROUND_WRAP => shr_round_away(p, F),
        _ => p >> F,
    }
}

/// Portable lane-wise multiply (the behavioural reference).
#[inline(always)]
pub fn mul_lanes_portable<
    T: FixedRepr,
    const I: u32,
    const F: u32,
    const MODE: u8,
    const N: usize,
>(
    a: &[T; N],
    b: &[T; N],
) -> [T; N] {
    core::array::from_fn(|i| T::narrow(mul_lane::<I, F, MODE>(a[i].widen(), b[i].widen())))
}

/// A signed integer type usable as the raw storage of a [`Fixed`](crate::Fixed).
///
/// Sealed: implemented for `i32` and `i16`.
pub trait FixedRepr: SimdInt + sealed::Sealed {
    /// Sign-extend to `i64`.
    fn widen(self) -> i64;
    /// Truncate (wrap) an `i64` to this type.
    fn narrow(w: i64) -> Self;
    /// Saturating, NaN-to-zero `as` cast from `f32`.
    fn from_f32_sat(x: f32) -> Self;
    /// Convert to `f32` (rounded to nearest if the value needs more than 24 bits).
    fn to_f32(self) -> f32;

    /// Lane-wise fixed-point multiply of two raw arrays. Backends override
    /// this with intrinsics; the result is bit-identical to
    /// the portable path.
    #[inline(always)]
    fn mul_lanes<const I: u32, const F: u32, const MODE: u8, const N: usize>(
        a: &[Self; N],
        b: &[Self; N],
    ) -> [Self; N] {
        mul_lanes_portable::<Self, I, F, MODE, N>(a, b)
    }
}

impl FixedRepr for i16 {
    #[inline(always)]
    fn widen(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn narrow(w: i64) -> Self {
        w as i16
    }
    #[inline(always)]
    fn from_f32_sat(x: f32) -> Self {
        x as i16
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl FixedRepr for i32 {
    #[inline(always)]
    fn widen(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn narrow(w: i64) -> Self {
        w as i32
    }
    #[inline(always)]
    fn from_f32_sat(x: f32) -> Self {
        x as i32
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    #[inline(always)]
    fn mul_lanes<const I: u32, const F: u32, const MODE: u8, const N: usize>(
        a: &[Self; N],
        b: &[Self; N],
    ) -> [Self; N] {
        avx2::mul_lanes::<I, F, MODE, N>(a, b)
    }
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2"
))]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// Finish four 64-bit products: round/shift/saturate each 64-bit lane;
    /// the result is in the low 32 bits of every 64-bit lane.
    ///
    /// # Safety
    /// The CPU must support AVX2 (guaranteed by the module's `cfg`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn finish<const I: u32, const F: u32, const MODE: u8>(p: __m256i) -> __m256i {
        let cnt = _mm_cvtsi32_si128(F as i32);
        if MODE == MUL_TRUNC_WRAP {
            return _mm256_srl_epi64(p, cnt);
        }
        let s = if F == 0 {
            p
        } else {
            let neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), p); // -1 if p < 0
            let half = _mm256_set1_epi64x(1i64 << (F - 1));
            _mm256_add_epi64(_mm256_add_epi64(p, half), neg)
        };
        let lo = _mm256_srl_epi64(s, cnt);
        if MODE == MUL_ROUND_WRAP {
            return lo;
        }
        // Saturate: s >> F (arithmetic) outside [MIN, MAX] <=> s outside
        // [MIN << F, (MAX << F) | (2^F - 1)].
        let hi_lim = _mm256_set1_epi64x((max_w(I, F) << F) | ((1i64 << F) - 1));
        let lo_lim = _mm256_set1_epi64x(min_w(I, F) << F);
        let over = _mm256_cmpgt_epi64(s, hi_lim);
        let under = _mm256_cmpgt_epi64(lo_lim, s);
        let r = _mm256_blendv_epi8(lo, _mm256_set1_epi64x(max_w(I, F)), over);
        _mm256_blendv_epi8(r, _mm256_set1_epi64x(min_w(I, F)), under)
    }

    /// Multiply eight i32 lanes using `_mm256_mul_epi32` on even and odd lanes.
    ///
    /// # Safety
    /// The CPU must support AVX2 (guaranteed by the module's `cfg`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul8<const I: u32, const F: u32, const MODE: u8>(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: AVX2 is enabled for this function.
        unsafe {
            let ev = _mm256_mul_epi32(a, b);
            let od = _mm256_mul_epi32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b));
            let e = finish::<I, F, MODE>(ev);
            let o = finish::<I, F, MODE>(od);
            _mm256_blend_epi32::<0b1010_1010>(e, _mm256_slli_epi64::<32>(o))
        }
    }

    #[inline(always)]
    pub(super) fn mul_lanes<const I: u32, const F: u32, const MODE: u8, const N: usize>(
        a: &[i32; N],
        b: &[i32; N],
    ) -> [i32; N] {
        let mut out = [0i32; N];
        let mut i = 0;
        while i + 8 <= N {
            // SAFETY: AVX2 is statically enabled (module cfg); `i + 8 <= N`
            // so the unaligned 256-bit loads/store stay inside the arrays.
            unsafe {
                let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
                let r = mul8::<I, F, MODE>(va, vb);
                _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), r);
            }
            i += 8;
        }
        while i < N {
            out[i] = i32::narrow(mul_lane::<I, F, MODE>(a[i] as i64, b[i] as i64));
            i += 1;
        }
        out
    }
}
