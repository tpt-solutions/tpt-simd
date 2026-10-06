//! SIMD non-contiguous loads (gather).
//!
//! Implemented (all operate on 8 lanes with `i32` indices):
//! * [`gather_i32`], [`gather_f32`] — `unsafe` raw-pointer gathers.
//! * [`gather_i32_portable`], [`gather_f32_portable`] — the `unsafe` scalar
//!   reference paths (identical results, no intrinsics).
//! * [`gather_checked_i32`], [`gather_checked_f32`] — safe slice gathers that
//!   **panic** if any index is out of bounds.
//! * [`try_gather_i32`], [`try_gather_f32`] — safe slice gathers returning
//!   `None` if any index is out of bounds.
//!
//! ## Index semantics
//!
//! Lane `k` of the result is `base[indices[k]]`. Indices are *element*
//! offsets and are signed: in the unchecked functions a negative index is
//! allowed as long as `base.offset(index)` is valid for reads (this matches
//! the hardware, which sign-extends indices). In the checked/`try_` variants
//! a negative index is always out of bounds. The same element may be
//! gathered by several lanes.
//!
//! ## Performance (honest note)
//!
//! `vpgatherdd` is *not* a fast instruction. On Intel Haswell/Broadwell it is
//! roughly on par with eight scalar loads; Skylake and later improved it, and
//! AMD Zen 1-3 microcode it so slowly that it is usually *slower* than
//! scalar loads (Zen 4 is better). Gather mostly wins when it feeds further
//! SIMD work and avoids a store/reload round trip; as a stand-alone
//! "load eight scattered values into an array" it often ties with or loses to
//! a plain scalar loop, and the checked variants add eight bounds compares.
//! See `benches/gather.rs`; measure on your target before relying on it.
//!
//! The AVX2 path is compiled only when `avx2` is enabled at compile time
//! (e.g. `-C target-cpu=native`) and the `scalar-only` feature is off.
//! Otherwise the portable scalar loop is used.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::Simd;

/// Portable scalar reference for [`gather_i32`].
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// reading an `i32` (in bounds of one allocation, initialised, aligned).
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_i32_portable;
/// let data = [10, 20, 30, 40];
/// let r = unsafe { gather_i32_portable(data.as_ptr(), I32x8::from_array([3, 2, 1, 0, 0, 1, 2, 3])) };
/// assert_eq!(r.to_array(), [40, 30, 20, 10, 10, 20, 30, 40]);
/// ```
#[inline]
pub unsafe fn gather_i32_portable(base: *const i32, indices: Simd<i32, 8>) -> Simd<i32, 8> {
    let mut out = [0i32; 8];
    for (o, &i) in out.iter_mut().zip(indices.0.iter()) {
        // SAFETY: caller guarantees each `base.offset(i)` is valid for reads.
        *o = unsafe { *base.offset(i as isize) };
    }
    Simd::from_array(out)
}

/// Portable scalar reference for [`gather_f32`].
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// reading an `f32` (in bounds of one allocation, initialised, aligned).
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_f32_portable;
/// let data = [1.0f32, 2.0, 3.0];
/// let r = unsafe { gather_f32_portable(data.as_ptr(), I32x8::splat(2)) };
/// assert_eq!(r.to_array(), [3.0; 8]);
/// ```
#[inline]
pub unsafe fn gather_f32_portable(base: *const f32, indices: Simd<i32, 8>) -> Simd<f32, 8> {
    let mut out = [0.0f32; 8];
    for (o, &i) in out.iter_mut().zip(indices.0.iter()) {
        // SAFETY: caller guarantees each `base.offset(i)` is valid for reads.
        *o = unsafe { *base.offset(i as isize) };
    }
    Simd::from_array(out)
}

/// Gathers `base[indices[k]]` into lane `k` (`vpgatherdd` on AVX2).
///
/// Performance: see the [crate docs](crate); the hardware gather is often no
/// faster than eight scalar loads.
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// reading an `i32`: in bounds of a single allocation, initialised and
/// aligned. Out-of-bounds indices are undefined behaviour (and may fault).
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_i32;
/// let data = [0, 10, 20, 30, 40, 50, 60, 70, 80];
/// let r = unsafe { gather_i32(data.as_ptr(), I32x8::from_array([8, 0, 4, 4, 1, 2, 3, 5])) };
/// assert_eq!(r.to_array(), [80, 0, 40, 40, 10, 20, 30, 50]);
/// ```
#[inline]
pub unsafe fn gather_i32(base: *const i32, indices: Simd<i32, 8>) -> Simd<i32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        use core::arch::x86_64::*;
        // SAFETY: `avx2` is enabled at compile time; the caller guarantees
        // every addressed element is readable. The index array is 32 bytes.
        unsafe {
            let idx = _mm256_loadu_si256(indices.0.as_ptr().cast());
            let v = _mm256_i32gather_epi32::<4>(base, idx);
            let mut out = [0i32; 8];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), v);
            Simd::from_array(out)
        }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        // SAFETY: forwarded caller contract.
        unsafe { gather_i32_portable(base, indices) }
    }
}

/// Gathers `base[indices[k]]` into lane `k` (`vgatherdps` on AVX2).
///
/// Bit patterns (including NaN payloads) are copied unchanged. Performance:
/// see the [crate docs](crate).
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// reading an `f32`: in bounds of a single allocation, initialised and
/// aligned. Out-of-bounds indices are undefined behaviour.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_f32;
/// let data = [0.5f32, 1.5, 2.5];
/// let r = unsafe { gather_f32(data.as_ptr(), I32x8::from_array([2, 1, 0, 0, 1, 2, 2, 0])) };
/// assert_eq!(r.to_array(), [2.5, 1.5, 0.5, 0.5, 1.5, 2.5, 2.5, 0.5]);
/// ```
#[inline]
pub unsafe fn gather_f32(base: *const f32, indices: Simd<i32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        use core::arch::x86_64::*;
        // SAFETY: `avx2` is enabled at compile time; the caller guarantees
        // every addressed element is readable. The index array is 32 bytes.
        unsafe {
            let idx = _mm256_loadu_si256(indices.0.as_ptr().cast());
            let v = _mm256_i32gather_ps::<4>(base, idx);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v);
            Simd::from_array(out)
        }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        // SAFETY: forwarded caller contract.
        unsafe { gather_f32_portable(base, indices) }
    }
}

/// `true` if every index is in `0..len`.
#[inline]
fn all_in_bounds(len: usize, indices: &Simd<i32, 8>) -> bool {
    indices
        .0
        .iter()
        .fold(true, |ok, &i| ok & (i >= 0) & ((i as usize) < len))
}

/// Safe gather: `None` if any index is negative or `>= data.len()`,
/// otherwise `Some` of the gathered lanes. No lane is read unless all eight
/// indices are valid.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::try_gather_i32;
/// let data = [1, 2, 3];
/// assert!(try_gather_i32(&data, I32x8::splat(2)).is_some());
/// assert!(try_gather_i32(&data, I32x8::splat(3)).is_none());
/// assert!(try_gather_i32(&data, I32x8::splat(-1)).is_none());
/// ```
#[inline]
pub fn try_gather_i32(data: &[i32], indices: Simd<i32, 8>) -> Option<Simd<i32, 8>> {
    if all_in_bounds(data.len(), &indices) {
        // SAFETY: every index was just checked to lie in `0..data.len()`.
        Some(unsafe { gather_i32(data.as_ptr(), indices) })
    } else {
        None
    }
}

/// Safe `f32` gather: `None` if any index is negative or `>= data.len()`.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::try_gather_f32;
/// let data = [1.0f32, 2.0];
/// assert_eq!(try_gather_f32(&data, I32x8::splat(1)).unwrap().to_array(), [2.0; 8]);
/// assert!(try_gather_f32(&data, I32x8::splat(2)).is_none());
/// ```
#[inline]
pub fn try_gather_f32(data: &[f32], indices: Simd<i32, 8>) -> Option<Simd<f32, 8>> {
    if all_in_bounds(data.len(), &indices) {
        // SAFETY: every index was just checked to lie in `0..data.len()`.
        Some(unsafe { gather_f32(data.as_ptr(), indices) })
    } else {
        None
    }
}

/// Bounds-checked gather.
///
/// # Panics
/// If any index is negative or `>= data.len()`.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_checked_i32;
/// let data = [5, 6, 7];
/// let r = gather_checked_i32(&data, I32x8::from_array([0, 1, 2, 2, 1, 0, 0, 1]));
/// assert_eq!(r.to_array(), [5, 6, 7, 7, 6, 5, 5, 6]);
/// ```
///
/// ```should_panic
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_checked_i32;
/// gather_checked_i32(&[1, 2, 3], I32x8::splat(3));
/// ```
#[inline]
#[track_caller]
pub fn gather_checked_i32(data: &[i32], indices: Simd<i32, 8>) -> Simd<i32, 8> {
    match try_gather_i32(data, indices) {
        Some(v) => v,
        None => panic!(
            "gather_checked_i32: index out of bounds (len {})",
            data.len()
        ),
    }
}

/// Bounds-checked `f32` gather.
///
/// # Panics
/// If any index is negative or `>= data.len()`.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_gather::gather_checked_f32;
/// assert_eq!(gather_checked_f32(&[1.5f32], I32x8::splat(0)).to_array(), [1.5; 8]);
/// ```
#[inline]
#[track_caller]
pub fn gather_checked_f32(data: &[f32], indices: Simd<i32, 8>) -> Simd<f32, 8> {
    match try_gather_f32(data, indices) {
        Some(v) => v,
        None => panic!(
            "gather_checked_f32: index out of bounds (len {})",
            data.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use proptest::prelude::*;
    use std::vec::Vec;

    fn scalar(data: &[i32], idx: [i32; 8]) -> [i32; 8] {
        idx.map(|i| data[i as usize])
    }

    #[test]
    fn basic_and_duplicates() {
        let data: Vec<i32> = (0..20).map(|x| x * 3).collect();
        let idx = [19, 0, 5, 5, 5, 19, 1, 2];
        let r = gather_checked_i32(&data, Simd::from_array(idx));
        assert_eq!(r.to_array(), scalar(&data, idx));
    }

    #[test]
    fn negative_index_unchecked() {
        let data = [1, 2, 3, 4];
        // SAFETY: base points at element 2, so offsets -2..=1 are in bounds.
        let r = unsafe {
            gather_i32(
                data[2..].as_ptr(),
                Simd::from_array([-2, -1, 0, 1, -2, -1, 0, 1]),
            )
        };
        assert_eq!(r.to_array(), [1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn try_rejects_oob() {
        let data = [1, 2, 3];
        for bad in [-1, 3, i32::MIN, i32::MAX] {
            for lane in 0..8 {
                let mut idx = [0i32; 8];
                idx[lane] = bad;
                assert!(try_gather_i32(&data, Simd::from_array(idx)).is_none());
                assert!(try_gather_f32(&[0.0; 3], Simd::from_array(idx)).is_none());
            }
        }
        assert!(try_gather_i32(&[], Simd::splat(0)).is_none());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn checked_panics_negative() {
        let _ = gather_checked_i32(&[1, 2, 3], Simd::splat(-1));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn checked_f32_panics() {
        let _ = gather_checked_f32(&[1.0], Simd::splat(1));
    }

    #[test]
    fn f32_specials_bit_exact() {
        let data = [
            f32::NAN,
            -0.0,
            f32::INFINITY,
            f32::from_bits(0x7fc0_1234),
            1.0,
        ];
        let idx = [0, 1, 2, 3, 4, 3, 1, 0];
        let r = gather_checked_f32(&data, Simd::from_array(idx));
        for k in 0..8 {
            assert_eq!(r.0[k].to_bits(), data[idx[k] as usize].to_bits());
        }
    }

    proptest! {
        #[test]
        fn i32_matches_scalar(
            data in proptest::collection::vec(any::<i32>(), 1..64),
            raw in proptest::array::uniform8(any::<u32>()),
        ) {
            let idx = raw.map(|r| (r as usize % data.len()) as i32);
            let v = Simd::from_array(idx);
            let want = scalar(&data, idx);
            prop_assert_eq!(gather_checked_i32(&data, v).to_array(), want);
            prop_assert_eq!(try_gather_i32(&data, v).map(|s| s.to_array()), Some(want));
            // SAFETY: indices were reduced modulo `data.len()`.
            let p = unsafe { gather_i32_portable(data.as_ptr(), v) };
            prop_assert_eq!(p.to_array(), want);
        }

        #[test]
        fn f32_matches_scalar(
            data in proptest::collection::vec(any::<u32>(), 1..64),
            raw in proptest::array::uniform8(any::<u32>()),
        ) {
            let data: Vec<f32> = data.into_iter().map(f32::from_bits).collect();
            let idx = raw.map(|r| (r as usize % data.len()) as i32);
            let v = Simd::from_array(idx);
            let got = gather_checked_f32(&data, v);
            // SAFETY: indices were reduced modulo `data.len()`.
            let port = unsafe { gather_f32_portable(data.as_ptr(), v) };
            for k in 0..8 {
                let w = data[idx[k] as usize].to_bits();
                prop_assert_eq!(got.0[k].to_bits(), w);
                prop_assert_eq!(port.0[k].to_bits(), w);
            }
        }

        #[test]
        fn oob_matches_predicate(
            len in 0usize..32,
            raw in proptest::array::uniform8(any::<i32>()),
        ) {
            let data = std::vec![7i32; len];
            let expect_ok = raw.iter().all(|&i| i >= 0 && (i as usize) < len);
            prop_assert_eq!(try_gather_i32(&data, Simd::from_array(raw)).is_some(), expect_ok);
        }
    }
}
