//! SIMD comparisons producing masks.
//!
//! Every comparison takes two full vectors and returns a [`SimdMask`] with
//! one boolean per lane. Shapes match the 256-bit vectors used across the
//! workspace: `i32x8`, `f32x8`, `i16x16` and `i8x32`.
//!
//! Implemented, for each of `i32`, `f32`, `i16`, `i8`: `cmp_gt_*`,
//! `cmp_lt_*`, `cmp_eq_*`, `cmp_ne_*`, `cmp_ge_*`, `cmp_le_*` (24 functions;
//! integer comparisons are signed), plus [`cmp_unord_f32`].
//!
//! Mask helpers: [`mask_any`], [`mask_all`], [`mask_count`],
//! [`mask_to_bitmask`] and [`mask_from_bitmask`].
//!
//! ## AVX2 fast paths
//!
//! When built with AVX2 enabled (e.g. `-C target-cpu=native`) the
//! comparisons use `vpcmpgt*`/`vpcmpeq*`/`vcmpps` and store the result
//! register straight into the mask's raw lanes (no packing), and the mask
//! helpers use `vmovmskps`/`vpmovmskb` + `popcnt` on the raw lanes. Otherwise the
//! portable array implementation is used; results are identical.
//!
//! For "how many elements satisfy a predicate" prefer the fused slice
//! helpers [`count_gt_i32`], [`count_lt_i32`], [`count_eq_i32`],
//! [`count_gt_f32`], [`count_lt_f32`], [`count_eq_f32`]: they never
//! materialise a [`SimdMask`]. Per-vector `cmp_*` + [`mask_count`] in a loop
//! is far slower than a (auto-vectorised) scalar count.
//!
//! ## NaN semantics (`f32`)
//!
//! Comparisons follow IEEE 754 *ordered* semantics, identical to Rust's
//! scalar `<`, `<=`, `>`, `>=`, `==`, `!=` operators:
//!
//! * `gt`, `lt`, `ge`, `le`, `eq` are **false** if either lane is NaN.
//! * `ne` is **true** if either lane is NaN (it is exactly `!eq`).
//! * `-0.0 == +0.0` is true.
//!
//! Consequently `cmp_ge(a, b)` is *not* the negation of `cmp_lt(a, b)` when
//! NaNs are present. Use [`cmp_unord_f32`] to find NaN lanes.
//!
//! ```
//! use tpt_simd_compare::*;
//! use tpt_simd_core::F32x8;
//! let a = F32x8::from_array([1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let b = F32x8::splat(3.0);
//! let m = cmp_gt_f32(a, b);
//! assert_eq!(mask_to_bitmask(m), 0b1111_1000);
//! assert!(mask_any(m) && !mask_all(m));
//! assert_eq!(mask_count(cmp_ne_f32(a, a)), 1); // only the NaN lane
//! ```
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

pub use tpt_simd_core::SimdMask;
use tpt_simd_core::{Simd, SimdElement};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
mod x86 {
    //! AVX2 implementations (selected statically via `target_feature`).
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use tpt_simd_core::{MaskLane, Simd, SimdElement, SimdMask};

    #[inline(always)]
    fn ld_i<T: SimdElement, const N: usize>(a: &Simd<T, N>) -> __m256i {
        assert_eq!(core::mem::size_of::<[T; N]>(), 32);
        // SAFETY: `a.0` is exactly 32 readable bytes (asserted above); the
        // load is unaligned; AVX2 is statically enabled by the cfg.
        unsafe { _mm256_loadu_si256(a.0.as_ptr().cast()) }
    }

    #[inline(always)]
    fn ld_f(a: &Simd<f32, 8>) -> __m256 {
        // SAFETY: `a.0` is 8 f32 = 32 readable bytes; unaligned load.
        unsafe { _mm256_loadu_ps(a.0.as_ptr()) }
    }

    /// A vector of all-ones / zero lanes (same total width as the mask
    /// array) stored straight into the raw lanes of a mask.
    #[inline(always)]
    fn wrap<T: SimdElement, const N: usize>(v: __m256i) -> SimdMask<T, N> {
        assert_eq!(core::mem::size_of::<[T::MaskLane; N]>(), 32);
        let mut out = [T::MaskLane::CLEAR; N];
        // SAFETY: `out` is exactly 32 writable bytes (asserted above); the
        // store is unaligned; AVX2 is statically enabled by the cfg.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) };
        SimdMask::from_raw(out)
    }

    macro_rules! int_family {
        ($t:ty, $n:literal, $gt:ident, $lt:ident, $eq:ident, $ne:ident, $ge:ident, $le:ident,
         $cmpgt:ident, $cmpeq:ident) => {
            #[inline(always)]
            pub fn $gt(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
                // SAFETY: AVX2 is statically enabled.
                wrap::<$t, $n>(unsafe { $cmpgt(ld_i(&a), ld_i(&b)) })
            }
            #[inline(always)]
            pub fn $lt(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
                // SAFETY: AVX2 is statically enabled.
                wrap::<$t, $n>(unsafe { $cmpgt(ld_i(&b), ld_i(&a)) })
            }
            #[inline(always)]
            pub fn $eq(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
                // SAFETY: AVX2 is statically enabled.
                wrap::<$t, $n>(unsafe { $cmpeq(ld_i(&a), ld_i(&b)) })
            }
            #[inline(always)]
            pub fn $ne(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
                // SAFETY: AVX2 is statically enabled.
                wrap::<$t, $n>(unsafe {
                    _mm256_xor_si256($cmpeq(ld_i(&a), ld_i(&b)), _mm256_set1_epi8(-1))
                })
            }
            #[inline(always)]
            pub fn $ge(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
                // SAFETY: AVX2 is statically enabled.
                wrap::<$t, $n>(unsafe {
                    _mm256_xor_si256($cmpgt(ld_i(&b), ld_i(&a)), _mm256_set1_epi8(-1))
                })
            }
            #[inline(always)]
            pub fn $le(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
                // SAFETY: AVX2 is statically enabled.
                wrap::<$t, $n>(unsafe {
                    _mm256_xor_si256($cmpgt(ld_i(&a), ld_i(&b)), _mm256_set1_epi8(-1))
                })
            }
        };
    }

    int_family!(
        i32,
        8,
        cmp_gt_i32,
        cmp_lt_i32,
        cmp_eq_i32,
        cmp_ne_i32,
        cmp_ge_i32,
        cmp_le_i32,
        _mm256_cmpgt_epi32,
        _mm256_cmpeq_epi32
    );
    int_family!(
        i16,
        16,
        cmp_gt_i16,
        cmp_lt_i16,
        cmp_eq_i16,
        cmp_ne_i16,
        cmp_ge_i16,
        cmp_le_i16,
        _mm256_cmpgt_epi16,
        _mm256_cmpeq_epi16
    );
    int_family!(
        i8,
        32,
        cmp_gt_i8,
        cmp_lt_i8,
        cmp_eq_i8,
        cmp_ne_i8,
        cmp_ge_i8,
        cmp_le_i8,
        _mm256_cmpgt_epi8,
        _mm256_cmpeq_epi8
    );

    macro_rules! f32_cmp {
        ($name:ident, $pred:ident) => {
            #[inline(always)]
            pub fn $name(a: Simd<f32, 8>, b: Simd<f32, 8>) -> SimdMask<f32, 8> {
                // SAFETY: AVX is statically enabled.
                wrap::<f32, 8>(unsafe {
                    _mm256_castps_si256(_mm256_cmp_ps::<$pred>(ld_f(&a), ld_f(&b)))
                })
            }
        };
    }
    // Ordered, quiet predicates except `ne` (unordered => true), matching
    // Rust's scalar operators.
    f32_cmp!(cmp_gt_f32, _CMP_GT_OQ);
    f32_cmp!(cmp_lt_f32, _CMP_LT_OQ);
    f32_cmp!(cmp_eq_f32, _CMP_EQ_OQ);
    f32_cmp!(cmp_ne_f32, _CMP_NEQ_UQ);
    f32_cmp!(cmp_ge_f32, _CMP_GE_OQ);
    f32_cmp!(cmp_le_f32, _CMP_LE_OQ);
    f32_cmp!(cmp_unord_f32, _CMP_UNORD_Q);

    /// Pack raw mask lanes (each all-ones or zero) into a bitmask, lane 0 in
    /// bit 0, with `movmskps`/`movmskpd`/`pmovmskb` straight on the lanes.
    #[inline(always)]
    pub fn bitmask<L: MaskLane, const N: usize>(raw: &[L; N]) -> u64 {
        assert!(N <= 64, "to_bitmask supports at most 64 lanes");
        let p = raw.as_ptr().cast::<u8>();
        let w = core::mem::size_of::<L>();
        let mut m = 0u64;
        let mut i = 0;
        match w {
            1 => {
                while i + 32 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe { _mm256_movemask_epi8(_mm256_loadu_si256(p.add(i).cast())) };
                    m |= u64::from(bits as u32) << i;
                    i += 32;
                }
                while i + 16 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe { _mm_movemask_epi8(_mm_loadu_si128(p.add(i).cast())) };
                    m |= u64::from(bits as u32) << i;
                    i += 16;
                }
            }
            2 => {
                while i + 16 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe {
                        let v = _mm256_loadu_si256(p.add(i * 2).cast());
                        let lo = _mm256_castsi256_si128(v);
                        let hi = _mm256_extracti128_si256::<1>(v);
                        _mm_movemask_epi8(_mm_packs_epi16(lo, hi))
                    };
                    m |= u64::from(bits as u32) << i;
                    i += 16;
                }
                while i + 8 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe {
                        let v = _mm_loadu_si128(p.add(i * 2).cast());
                        _mm_movemask_epi8(_mm_packs_epi16(v, v))
                    };
                    m |= u64::from(bits as u32 & 0xFF) << i;
                    i += 8;
                }
            }
            4 => {
                while i + 8 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe { _mm256_movemask_ps(_mm256_loadu_ps(p.add(i * 4).cast())) };
                    m |= u64::from(bits as u32) << i;
                    i += 8;
                }
                while i + 4 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe { _mm_movemask_ps(_mm_loadu_ps(p.add(i * 4).cast())) };
                    m |= u64::from(bits as u32) << i;
                    i += 4;
                }
            }
            8 => {
                while i + 4 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe { _mm256_movemask_pd(_mm256_loadu_pd(p.add(i * 8).cast())) };
                    m |= u64::from(bits as u32) << i;
                    i += 4;
                }
                while i + 2 <= N {
                    // SAFETY: `i + lanes <= N`, so the load stays inside `raw`; AVX2 is enabled.
                    let bits = unsafe { _mm_movemask_pd(_mm_loadu_pd(p.add(i * 8).cast())) };
                    m |= u64::from(bits as u32) << i;
                    i += 2;
                }
            }
            _ => {}
        }
        while i < N {
            m |= raw[i].bit() << i;
            i += 1;
        }
        m
    }

    /// Shared driver for the fused counters: `cmp` maps 8 elements to a
    /// vector of 0 / -1 (32-bit) lanes.
    #[inline(always)]
    fn count_by<T: Copy>(
        data: &[T],
        cmp: impl Fn(&[T; 8]) -> __m256i,
        scalar: impl Fn(T) -> bool,
    ) -> usize {
        let mut total = 0usize;
        // Blocks bound every per-lane u32 counter well below overflow.
        for block in data.chunks(1 << 28) {
            let (body, tail) = block.as_chunks::<8>();
            let (quads, rest) = body.as_chunks::<4>();
            // SAFETY: AVX2 is statically enabled.
            let (mut a0, mut a1, mut a2, mut a3) = unsafe {
                (
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                )
            };
            for q in quads {
                // SAFETY: AVX2 is statically enabled.
                unsafe {
                    a0 = _mm256_sub_epi32(a0, cmp(&q[0]));
                    a1 = _mm256_sub_epi32(a1, cmp(&q[1]));
                    a2 = _mm256_sub_epi32(a2, cmp(&q[2]));
                    a3 = _mm256_sub_epi32(a3, cmp(&q[3]));
                }
            }
            for c in rest {
                // SAFETY: AVX2 is statically enabled.
                a0 = unsafe { _mm256_sub_epi32(a0, cmp(c)) };
            }
            let mut lanes = [0u32; 8];
            // SAFETY: the store writes exactly 32 bytes into `lanes`.
            unsafe {
                let acc = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
                _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
            }
            total += lanes.iter().map(|&x| x as usize).sum::<usize>();
            total += tail.iter().filter(|&&x| scalar(x)).count();
        }
        total
    }

    macro_rules! count_i32 {
        ($name:ident, $cmp:expr, $op:tt) => {
            pub fn $name(data: &[i32], rhs: i32) -> usize {
                // SAFETY: AVX2 is statically enabled.
                let t = unsafe { _mm256_set1_epi32(rhs) };
                count_by(
                    data,
                    // SAFETY: `c` is 8 i32 = 32 readable bytes; AVX2 enabled.
                    |c| unsafe { $cmp(_mm256_loadu_si256(c.as_ptr().cast()), t) },
                    |x| x $op rhs,
                )
            }
        };
    }
    count_i32!(count_gt_i32, |x, t| _mm256_cmpgt_epi32(x, t), >);
    count_i32!(count_lt_i32, |x, t| _mm256_cmpgt_epi32(t, x), <);
    count_i32!(count_eq_i32, |x, t| _mm256_cmpeq_epi32(x, t), ==);

    macro_rules! count_f32 {
        ($name:ident, $pred:ident, $op:tt) => {
            pub fn $name(data: &[f32], rhs: f32) -> usize {
                // SAFETY: AVX is statically enabled.
                let t = unsafe { _mm256_set1_ps(rhs) };
                count_by(
                    data,
                    // SAFETY: `c` is 8 f32 = 32 readable bytes; AVX enabled.
                    |c| unsafe {
                        _mm256_castps_si256(_mm256_cmp_ps::<$pred>(_mm256_loadu_ps(c.as_ptr()), t))
                    },
                    |x| x $op rhs,
                )
            }
        };
    }
    count_f32!(count_gt_f32, _CMP_GT_OQ, >);
    count_f32!(count_lt_f32, _CMP_LT_OQ, <);
    count_f32!(count_eq_f32, _CMP_EQ_OQ, ==);
}

macro_rules! count_fn {
    ($name:ident, $t:ty, $op:tt, $sym:literal) => {
        #[doc = concat!("Number of elements `x` of `data` with `x ", $sym, " rhs`.")]
        #[doc = ""]
        #[doc = "Fused compare + count: no [`SimdMask`] is materialised. On AVX2 builds this"]
        #[doc = "is a vector compare plus per-lane counter accumulation (much faster than"]
        #[doc = "calling `cmp_*` + [`mask_count`] per vector); otherwise a scalar loop."]
        #[doc = "Semantics are those of the scalar operator (a NaN element never matches"]
        #[doc = "`>`, `<` or `==`)."]
        #[doc = ""]
        #[doc = "# Panics"]
        #[doc = "Never."]
        #[inline]
        pub fn $name(data: &[$t], rhs: $t) -> usize {
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
            {
                x86::$name(data, rhs)
            }
            #[cfg(not(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx2"
            )))]
            {
                data.iter().filter(|&&x| x $op rhs).count()
            }
        }
    };
}
count_fn!(count_gt_i32, i32, >, ">");
count_fn!(count_lt_i32, i32, <, "<");
count_fn!(count_eq_i32, i32, ==, "==");
count_fn!(count_gt_f32, f32, >, ">");
count_fn!(count_lt_f32, f32, <, "<");
count_fn!(count_eq_f32, f32, ==, "==");

macro_rules! cmp_fn {
    ($name:ident, $method:ident, $t:ty, $n:literal, $sym:literal, $alias:literal, $same:literal) => {
        #[doc = concat!("Lane-wise `a ", $sym, " b` for [`", $alias, "`](tpt_simd_core::", $alias, ") vectors.")]
        #[doc = ""]
        #[doc = "Performance: one compare instruction (`vpcmp*`/`vcmpps`) on AVX2 builds, stored"]
        #[doc = "as-is into the mask lanes; fuse with a consumer where possible."]
        #[doc = ""]
        #[doc = "# Panics"]
        #[doc = "Never."]
        #[doc = ""]
        #[doc = "```"]
        #[doc = concat!("use tpt_simd_core::", $alias, ";")]
        #[doc = concat!("use tpt_simd_compare::{", stringify!($name), ", mask_all};")]
        #[doc = concat!("let a = ", $alias, "::splat(1 as _);")]
        #[doc = concat!("assert_eq!(mask_all(", stringify!($name), "(a, a)), ", $same, ");")]
        #[doc = "```"]
        #[inline]
        pub fn $name(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
            {
                x86::$name(a, b)
            }
            #[cfg(not(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx2"
            )))]
            {
                a.$method(b)
            }
        }
    };
}

macro_rules! cmp_family {
    ($t:ty, $n:literal, $alias:literal, $gt:ident, $lt:ident, $eq:ident, $ne:ident, $ge:ident, $le:ident) => {
        cmp_fn!($gt, simd_gt, $t, $n, ">", $alias, "false");
        cmp_fn!($lt, simd_lt, $t, $n, "<", $alias, "false");
        cmp_fn!($eq, simd_eq, $t, $n, "==", $alias, "true");
        cmp_fn!($ne, simd_ne, $t, $n, "!=", $alias, "false");
        cmp_fn!($ge, simd_ge, $t, $n, ">=", $alias, "true");
        cmp_fn!($le, simd_le, $t, $n, "<=", $alias, "true");
    };
}

cmp_family!(
    i32, 8, "I32x8", cmp_gt_i32, cmp_lt_i32, cmp_eq_i32, cmp_ne_i32, cmp_ge_i32, cmp_le_i32
);
cmp_family!(
    f32, 8, "F32x8", cmp_gt_f32, cmp_lt_f32, cmp_eq_f32, cmp_ne_f32, cmp_ge_f32, cmp_le_f32
);
cmp_family!(
    i16, 16, "I16x16", cmp_gt_i16, cmp_lt_i16, cmp_eq_i16, cmp_ne_i16, cmp_ge_i16, cmp_le_i16
);
cmp_family!(
    i8, 32, "I8x32", cmp_gt_i8, cmp_lt_i8, cmp_eq_i8, cmp_ne_i8, cmp_ge_i8, cmp_le_i8
);

/// Lanes where either operand is NaN (unordered compare, `_CMP_UNORD_Q`).
///
/// ```
/// use tpt_simd_compare::{cmp_unord_f32, mask_to_bitmask};
/// use tpt_simd_core::F32x8;
/// let a = F32x8::from_array([f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let b = F32x8::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f32::NAN]);
/// assert_eq!(mask_to_bitmask(cmp_unord_f32(a, b)), 0b1000_0001);
/// ```
#[inline]
pub fn cmp_unord_f32(a: Simd<f32, 8>, b: Simd<f32, 8>) -> SimdMask<f32, 8> {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    {
        x86::cmp_unord_f32(a, b)
    }
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )))]
    {
        a.is_nan() | b.is_nan()
    }
}

/// `true` if any lane of the mask is set.
///
/// ```
/// use tpt_simd_compare::{mask_any, SimdMask};
/// assert!(!mask_any(SimdMask::<i32, 8>::splat(false)));
/// ```
#[inline]
pub fn mask_any<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> bool {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    if N <= 64 {
        return x86::bitmask(&m.to_raw()) != 0;
    }
    m.any()
}

/// `true` if every lane of the mask is set.
///
/// ```
/// use tpt_simd_compare::{mask_all, SimdMask};
/// assert!(mask_all(SimdMask::<i32, 8>::splat(true)));
/// ```
#[inline]
pub fn mask_all<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> bool {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    if N <= 64 {
        return x86::bitmask(&m.to_raw()).count_ones() as usize == N;
    }
    m.all()
}

/// Number of set lanes.
///
/// ```
/// use tpt_simd_compare::{mask_count, SimdMask};
/// assert_eq!(mask_count(SimdMask::<i32, 8>::from_bitmask(0b1011)), 3);
/// ```
#[inline]
pub fn mask_count<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> usize {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    if N <= 64 {
        return x86::bitmask(&m.to_raw()).count_ones() as usize;
    }
    m.count()
}

/// Pack the mask into an integer, lane 0 in bit 0 (like `movmskps` /
/// `pmovmskb`).
///
/// # Panics
/// If `N > 64`.
///
/// ```
/// use tpt_simd_compare::{mask_to_bitmask, SimdMask};
/// let m = SimdMask::<i32, 4>::from_array([true, false, true, true]);
/// assert_eq!(mask_to_bitmask(m), 0b1101);
/// ```
#[inline]
pub fn mask_to_bitmask<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> u64 {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    {
        x86::bitmask(&m.to_raw())
    }
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )))]
    {
        m.to_bitmask()
    }
}

/// Unpack a mask from an integer, bit 0 to lane 0; bits above `N` ignored.
///
/// # Panics
/// If `N > 64`.
///
/// ```
/// use tpt_simd_compare::{mask_from_bitmask, mask_to_bitmask, SimdMask};
/// let m: SimdMask<i8, 32> = mask_from_bitmask(0xF0F0_F0F0);
/// assert_eq!(mask_to_bitmask(m), 0xF0F0_F0F0);
/// ```
#[inline]
pub fn mask_from_bitmask<T: SimdElement, const N: usize>(bits: u64) -> SimdMask<T, N> {
    SimdMask::from_bitmask(bits)
}

#[cfg(all(test, not(feature = "std")))]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tpt_simd_core::SimdElement;

    fn lanes<T: SimdElement + core::fmt::Debug, S: Strategy<Value = T>, const N: usize>(
        s: S,
    ) -> impl Strategy<Value = Simd<T, N>> {
        proptest::collection::vec(s, N).prop_map(|v| Simd::from_slice(&v))
    }
    use tpt_simd_core::{F32x8, I8x32, I16x16, I32x8};
    use tpt_simd_testutil::{f32_with_specials, i16_edgy, i32_edgy};

    fn bits<T: Copy, const N: usize>(a: [T; N], b: [T; N], f: impl Fn(T, T) -> bool) -> u64 {
        (0..N).fold(0u64, |m, i| m | (u64::from(f(a[i], b[i])) << i))
    }

    #[test]
    fn nan_semantics() {
        let n = f32::NAN;
        let a = F32x8::from_array([n, 1.0, n, 0.0, -0.0, 2.0, 2.0, 2.0]);
        let b = F32x8::from_array([1.0, n, n, -0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(mask_to_bitmask(cmp_gt_f32(a, b)), 0b0010_0000);
        assert_eq!(mask_to_bitmask(cmp_lt_f32(a, b)), 0b1000_0000);
        assert_eq!(mask_to_bitmask(cmp_ge_f32(a, b)), 0b0111_1000);
        assert_eq!(mask_to_bitmask(cmp_le_f32(a, b)), 0b1101_1000);
        assert_eq!(mask_to_bitmask(cmp_eq_f32(a, b)), 0b0101_1000);
        assert_eq!(mask_to_bitmask(cmp_ne_f32(a, b)), 0b1010_0111);
        assert_eq!(mask_to_bitmask(cmp_unord_f32(a, b)), 0b0000_0111);
    }

    #[test]
    fn integer_basics() {
        let a = I32x8::from_array([1, 2, 3, 4, 5, 6, 7, i32::MIN]);
        let b = I32x8::splat(4);
        assert_eq!(mask_to_bitmask(cmp_gt_i32(a, b)), 0b0111_0000);
        assert_eq!(mask_to_bitmask(cmp_le_i32(a, b)), 0b1000_1111);
        assert_eq!(mask_count(cmp_eq_i32(a, b)), 1);
        assert!(mask_all(cmp_ge_i16(I16x16::splat(0), I16x16::splat(0))));
        assert!(!mask_any(cmp_lt_i8(I8x32::splat(-1), I8x32::splat(-1))));
        // Comparisons are signed.
        assert!(mask_all(cmp_lt_i8(I8x32::splat(-1), I8x32::splat(0))));
    }

    #[test]
    fn bitmask_roundtrip() {
        let m: SimdMask<i32, 8> = mask_from_bitmask(0xA5);
        assert_eq!(mask_to_bitmask(m), 0xA5);
    }

    #[test]
    fn mask_helpers_vs_array_reference() {
        fn check<const N: usize>(seed: u64) {
            let mut s = seed;
            for _ in 0..200 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let arr: [bool; N] = core::array::from_fn(|i| (s >> (i % 60 + 3)) & 1 != 0);
                let m = SimdMask::<i32, N>::from_array(arr);
                assert_eq!(mask_to_bitmask(m), m.to_bitmask());
                assert_eq!(mask_count(m), m.count());
                assert_eq!(mask_any(m), m.any());
                assert_eq!(mask_all(m), m.all());
            }
            assert!(mask_all(SimdMask::<i32, N>::splat(true)));
            assert!(!mask_any(SimdMask::<i32, N>::splat(false)));
        }
        check::<1>(1);
        check::<4>(2);
        check::<8>(3);
        check::<12>(4);
        check::<16>(5);
        check::<24>(6);
        check::<32>(7);
        check::<40>(8);
        check::<64>(9);
    }

    #[test]
    fn mask_helpers_all_lane_widths() {
        fn check<T: SimdElement, const N: usize>(seed: u64) {
            let mut s = seed;
            for _ in 0..100 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let arr: [bool; N] = core::array::from_fn(|i| (s >> (i % 60 + 3)) & 1 != 0);
                let m = SimdMask::<T, N>::from_array(arr);
                assert_eq!(mask_to_bitmask(m), m.to_bitmask());
                assert_eq!(mask_count(m), m.count());
                assert_eq!(mask_any(m), m.any());
                assert_eq!(mask_all(m), m.all());
            }
            assert!(mask_all(SimdMask::<T, N>::splat(true)));
            assert!(!mask_any(SimdMask::<T, N>::splat(false)));
            assert_eq!(mask_count(SimdMask::<T, N>::splat(true)), N);
        }
        check::<i8, 7>(1);
        check::<i8, 32>(2);
        check::<i8, 64>(3);
        check::<i16, 3>(4);
        check::<i16, 8>(5);
        check::<i16, 16>(6);
        check::<i16, 27>(7);
        check::<i16, 32>(8);
        check::<f32, 8>(9);
        check::<f32, 5>(10);
        check::<i64, 2>(11);
        check::<i64, 4>(12);
        check::<i64, 7>(13);
    }

    #[test]
    fn cmp_produces_canonical_raw_lanes() {
        let a = I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            cmp_gt_i32(a, I32x8::splat(4)).to_raw(),
            [0, 0, 0, 0, -1, -1, -1, -1]
        );
        let f = F32x8::from_array([f32::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(cmp_ne_f32(f, f).to_raw(), [-1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(cmp_ge_f32(f, f).to_raw(), [0, -1, -1, -1, -1, -1, -1, -1]);
        let s = I16x16::from_fn(|i| i as i16);
        assert_eq!(
            cmp_lt_i16(s, I16x16::splat(2)).to_raw(),
            core::array::from_fn::<i16, 16, _>(|i| if i < 2 { -1 } else { 0 })
        );
        let t = I8x32::from_fn(|i| i as i8);
        assert_eq!(
            cmp_eq_i8(t, I8x32::splat(31)).to_raw(),
            core::array::from_fn::<i8, 32, _>(|i| if i == 31 { -1 } else { 0 })
        );
    }

    #[test]
    fn fused_counts_vs_scalar() {
        let mut s = 0x9E37_79B9u32;
        for len in [0usize, 1, 7, 8, 9, 31, 32, 33, 100, 257, 1000] {
            let ints: std::vec::Vec<i32> = (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((s >> 20) as i32 % 5) - 2
                })
                .collect();
            let mut floats: std::vec::Vec<f32> = ints.iter().map(|&x| x as f32).collect();
            for (i, f) in floats.iter_mut().enumerate() {
                if i % 7 == 3 {
                    *f = f32::NAN;
                }
            }
            for t in [-2, 0, 1] {
                assert_eq!(
                    count_gt_i32(&ints, t),
                    ints.iter().filter(|&&x| x > t).count()
                );
                assert_eq!(
                    count_lt_i32(&ints, t),
                    ints.iter().filter(|&&x| x < t).count()
                );
                assert_eq!(
                    count_eq_i32(&ints, t),
                    ints.iter().filter(|&&x| x == t).count()
                );
                let tf = t as f32;
                assert_eq!(
                    count_gt_f32(&floats, tf),
                    floats.iter().filter(|&&x| x > tf).count()
                );
                assert_eq!(
                    count_lt_f32(&floats, tf),
                    floats.iter().filter(|&&x| x < tf).count()
                );
                assert_eq!(
                    count_eq_f32(&floats, tf),
                    floats.iter().filter(|&&x| x == tf).count()
                );
            }
            assert_eq!(count_eq_f32(&floats, f32::NAN), 0);
            assert_eq!(count_gt_i32(&[i32::MAX; 40], i32::MIN), 40);
        }
    }

    proptest! {
        #[test]
        fn i32_vs_scalar(a in lanes::<i32, _, 8>(i32_edgy()), b in lanes::<i32, _, 8>(i32_edgy())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_i32(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_i32(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_i32(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_i32(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_i32(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_i32(a, b)), bits(x, y, |p, q| p <= q));
        }

        #[test]
        fn f32_vs_scalar(a in lanes::<f32, _, 8>(f32_with_specials()), b in lanes::<f32, _, 8>(f32_with_specials())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_f32(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_f32(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_f32(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_f32(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_f32(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_f32(a, b)), bits(x, y, |p, q| p <= q));
            prop_assert_eq!(
                mask_to_bitmask(cmp_unord_f32(a, b)),
                bits(x, y, |p: f32, q: f32| p.is_nan() || q.is_nan())
            );
        }

        #[test]
        fn i16_vs_scalar(a in lanes::<i16, _, 16>(i16_edgy()), b in lanes::<i16, _, 16>(i16_edgy())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_i16(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_i16(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_i16(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_i16(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_i16(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_i16(a, b)), bits(x, y, |p, q| p <= q));
        }

        #[test]
        fn i8_vs_scalar(a in lanes::<i8, _, 32>(any::<i8>()), b in lanes::<i8, _, 32>(any::<i8>())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_i8(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_i8(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_i8(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_i8(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_i8(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_i8(a, b)), bits(x, y, |p, q| p <= q));
        }
    }
}
