//! SIMD non-contiguous stores (scatter).
//!
//! Implemented (all operate on 8 lanes with `i32` indices):
//! * [`scatter_i32`], [`scatter_f32`] — `unsafe` raw-pointer scatters.
//! * [`scatter_i32_portable`], [`scatter_f32_portable`] — the `unsafe` scalar
//!   reference paths (identical results).
//! * [`scatter_checked_i32`], [`scatter_checked_f32`] — safe slice scatters
//!   that **panic** (before writing anything) if any index is out of bounds.
//! * [`try_scatter_i32`], [`try_scatter_f32`] — safe slice scatters returning
//!   `Err(`[`OutOfBounds`]`)` (before writing anything) on a bad index.
//!
//! ## Semantics
//!
//! `base[indices[k]] = values[k]` for `k = 0..8`.
//!
//! ### Duplicate indices: the last lane wins
//!
//! If several lanes carry the same index, the stores happen in increasing
//! lane order, so the value from the **highest-numbered** lane is the one
//! left in memory. This is guaranteed on every path: the scalar loop stores
//! in lane order, and Intel documents that AVX-512 scatter instructions
//! write overlapping elements from the least- to the most-significant lane.
//! (There is no AVX2 scatter instruction, so AVX2 builds use the scalar
//! loop.)
//!
//! ### Bounds
//!
//! Indices are signed element offsets. The unchecked functions accept a
//! negative index provided `base.offset(index)` is valid for writes (the
//! hardware sign-extends indices); the checked/`try_` variants treat a
//! negative index as out of bounds. The checked variants validate *all*
//! eight indices first, so on failure the slice is left untouched.
//!
//! ## Performance
//!
//! Without AVX-512 there is no hardware scatter, so this is eight scalar
//! stores. With AVX-512 (`avx512f` + `avx512vl` enabled at compile time,
//! e.g. `-C target-cpu=native` on a capable CPU) `vpscatterdd` is used, which
//! is not dramatically faster than eight stores either; its benefit is mostly
//! keeping data in vector registers. See `benches/scatter.rs`.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use core::fmt;
use tpt_simd_core::Simd;

/// Error returned by the `try_scatter_*` functions: lane `lane` held the
/// index `index`, which is negative or not less than the slice length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfBounds {
    /// Lane (0..8) of the first offending index.
    pub lane: usize,
    /// The offending index value.
    pub index: i32,
}

impl fmt::Display for OutOfBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "scatter index {} in lane {} is out of bounds",
            self.index, self.lane
        )
    }
}

/// Portable scalar reference for [`scatter_i32`]; stores in lane order so the
/// last lane wins on duplicate indices.
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// writing an `i32` (in bounds of one allocation, aligned, not aliased by a
/// live reference).
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_scatter::scatter_i32_portable;
/// let mut d = [0i32; 4];
/// unsafe {
///     scatter_i32_portable(d.as_mut_ptr(), I32x8::from_array([3, 2, 1, 0, 0, 0, 0, 0]), I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]));
/// }
/// assert_eq!(d, [8, 3, 2, 1]);
/// ```
#[inline]
pub unsafe fn scatter_i32_portable(base: *mut i32, indices: Simd<i32, 8>, values: Simd<i32, 8>) {
    for k in 0..8 {
        // SAFETY: caller guarantees each `base.offset(i)` is valid for writes.
        unsafe { *base.offset(indices.0[k] as isize) = values.0[k] };
    }
}

/// Portable scalar reference for [`scatter_f32`]; stores in lane order so the
/// last lane wins on duplicate indices.
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// writing an `f32` (in bounds of one allocation, aligned, not aliased by a
/// live reference).
///
/// ```
/// use tpt_simd_core::{F32x8, I32x8};
/// use tpt_simd_scatter::scatter_f32_portable;
/// let mut d = [0.0f32; 2];
/// unsafe { scatter_f32_portable(d.as_mut_ptr(), I32x8::splat(1), F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])) };
/// assert_eq!(d, [0.0, 8.0]);
/// ```
#[inline]
pub unsafe fn scatter_f32_portable(base: *mut f32, indices: Simd<i32, 8>, values: Simd<f32, 8>) {
    for k in 0..8 {
        // SAFETY: caller guarantees each `base.offset(i)` is valid for writes.
        unsafe { *base.offset(indices.0[k] as isize) = values.0[k] };
    }
}

/// Scatters `values[k]` to `base[indices[k]]`. With duplicate indices the
/// highest-numbered lane wins (see the [crate docs](crate)).
///
/// Uses `vpscatterdd` when built with `avx512f` + `avx512vl`, otherwise
/// eight scalar stores.
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// writing an `i32`: in bounds of a single allocation, aligned, and not
/// concurrently accessed. Out-of-bounds indices are undefined behaviour.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_scatter::scatter_i32;
/// let mut d = [0i32; 8];
/// // Lanes 0 and 7 both target index 2: the last lane (7) wins.
/// unsafe {
///     scatter_i32(d.as_mut_ptr(), I32x8::from_array([2, 0, 1, 3, 4, 5, 6, 2]), I32x8::from_array([10, 11, 12, 13, 14, 15, 16, 17]));
/// }
/// assert_eq!(d, [11, 12, 17, 13, 14, 15, 16, 0]);
/// ```
#[inline]
pub unsafe fn scatter_i32(base: *mut i32, indices: Simd<i32, 8>, values: Simd<i32, 8>) {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    ))]
    {
        use core::arch::x86_64::*;
        // SAFETY: `avx512f`/`avx512vl` are enabled at compile time; the
        // caller guarantees every addressed element is writable. Index and
        // value arrays are 32 bytes. Overlapping lanes are stored from the
        // least to the most significant lane (Intel SDM), so the last wins.
        unsafe {
            let idx = _mm256_loadu_si256(indices.0.as_ptr().cast());
            let v = _mm256_loadu_si256(values.0.as_ptr().cast());
            _mm256_i32scatter_epi32::<4>(base.cast(), idx, v);
        }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    )))]
    {
        // SAFETY: forwarded caller contract.
        unsafe { scatter_i32_portable(base, indices, values) }
    }
}

/// Scatters `values[k]` to `base[indices[k]]` for `f32`. With duplicate
/// indices the highest-numbered lane wins (see the [crate docs](crate)).
///
/// # Safety
/// For every lane `k`, `base.offset(indices[k] as isize)` must be valid for
/// writing an `f32`: in bounds of a single allocation, aligned, and not
/// concurrently accessed. Out-of-bounds indices are undefined behaviour.
///
/// ```
/// use tpt_simd_core::{F32x8, I32x8};
/// use tpt_simd_scatter::scatter_f32;
/// let mut d = [0.0f32; 8];
/// unsafe { scatter_f32(d.as_mut_ptr(), I32x8::from_array([7, 6, 5, 4, 3, 2, 1, 0]), F32x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])) };
/// assert_eq!(d, [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
/// ```
#[inline]
pub unsafe fn scatter_f32(base: *mut f32, indices: Simd<i32, 8>, values: Simd<f32, 8>) {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    ))]
    {
        use core::arch::x86_64::*;
        // SAFETY: see `scatter_i32`.
        unsafe {
            let idx = _mm256_loadu_si256(indices.0.as_ptr().cast());
            let v = _mm256_loadu_ps(values.0.as_ptr());
            _mm256_i32scatter_ps::<4>(base.cast(), idx, v);
        }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    )))]
    {
        // SAFETY: forwarded caller contract.
        unsafe { scatter_f32_portable(base, indices, values) }
    }
}

/// Validates all indices against `len`, returning the first bad one.
#[inline]
fn check(len: usize, indices: &Simd<i32, 8>) -> Result<(), OutOfBounds> {
    for (lane, &index) in indices.0.iter().enumerate() {
        if index < 0 || (index as usize) >= len {
            return Err(OutOfBounds { lane, index });
        }
    }
    Ok(())
}

/// Safe scatter. If every index is in `0..data.len()` the values are stored
/// (last lane wins on duplicates) and `Ok(())` is returned; otherwise
/// `data` is left untouched and the first offending lane is reported.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_scatter::{try_scatter_i32, OutOfBounds};
/// let mut d = [0i32; 4];
/// let r = try_scatter_i32(&mut d, I32x8::from_array([0, 1, 2, 3, 4, 0, 0, 0]), I32x8::splat(9));
/// assert_eq!(r, Err(OutOfBounds { lane: 4, index: 4 }));
/// assert_eq!(d, [0; 4]);
/// ```
#[inline]
pub fn try_scatter_i32(
    data: &mut [i32],
    indices: Simd<i32, 8>,
    values: Simd<i32, 8>,
) -> Result<(), OutOfBounds> {
    check(data.len(), &indices)?;
    // SAFETY: every index was just checked to lie in `0..data.len()`, and
    // `data` is an exclusive borrow.
    unsafe { scatter_i32(data.as_mut_ptr(), indices, values) };
    Ok(())
}

/// Safe `f32` scatter; see [`try_scatter_i32`].
///
/// ```
/// use tpt_simd_core::{F32x8, I32x8};
/// use tpt_simd_scatter::try_scatter_f32;
/// let mut d = [0.0f32; 2];
/// assert!(try_scatter_f32(&mut d, I32x8::splat(1), F32x8::splat(2.0)).is_ok());
/// assert_eq!(d, [0.0, 2.0]);
/// assert!(try_scatter_f32(&mut d, I32x8::splat(-1), F32x8::splat(3.0)).is_err());
/// ```
#[inline]
pub fn try_scatter_f32(
    data: &mut [f32],
    indices: Simd<i32, 8>,
    values: Simd<f32, 8>,
) -> Result<(), OutOfBounds> {
    check(data.len(), &indices)?;
    // SAFETY: every index was just checked to lie in `0..data.len()`, and
    // `data` is an exclusive borrow.
    unsafe { scatter_f32(data.as_mut_ptr(), indices, values) };
    Ok(())
}

/// Bounds-checked scatter.
///
/// # Panics
/// If any index is negative or `>= data.len()`; nothing is written in that
/// case.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_scatter::scatter_checked_i32;
/// let mut d = [0i32; 3];
/// scatter_checked_i32(&mut d, I32x8::from_array([0, 1, 2, 2, 2, 1, 0, 0]), I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]));
/// assert_eq!(d, [8, 6, 5]); // last lane wins
/// ```
///
/// ```should_panic
/// use tpt_simd_core::I32x8;
/// use tpt_simd_scatter::scatter_checked_i32;
/// scatter_checked_i32(&mut [0; 3], I32x8::splat(3), I32x8::splat(1));
/// ```
#[inline]
#[track_caller]
pub fn scatter_checked_i32(data: &mut [i32], indices: Simd<i32, 8>, values: Simd<i32, 8>) {
    let len = data.len();
    if let Err(e) = try_scatter_i32(data, indices, values) {
        panic!("scatter_checked_i32: {e} (len {len})");
    }
}

/// Bounds-checked `f32` scatter.
///
/// # Panics
/// If any index is negative or `>= data.len()`; nothing is written in that
/// case.
///
/// ```
/// use tpt_simd_core::{F32x8, I32x8};
/// use tpt_simd_scatter::scatter_checked_f32;
/// let mut d = [0.0f32; 1];
/// scatter_checked_f32(&mut d, I32x8::splat(0), F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
/// assert_eq!(d, [8.0]);
/// ```
#[inline]
#[track_caller]
pub fn scatter_checked_f32(data: &mut [f32], indices: Simd<i32, 8>, values: Simd<f32, 8>) {
    let len = data.len();
    if let Err(e) = try_scatter_f32(data, indices, values) {
        panic!("scatter_checked_f32: {e} (len {len})");
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use proptest::prelude::*;
    use std::vec::Vec;

    fn scalar(data: &mut [i32], idx: [i32; 8], vals: [i32; 8]) {
        for k in 0..8 {
            data[idx[k] as usize] = vals[k];
        }
    }

    #[test]
    fn duplicates_last_lane_wins() {
        let mut d = [0i32; 4];
        scatter_checked_i32(
            &mut d,
            Simd::from_array([1, 1, 1, 1, 1, 1, 1, 1]),
            Simd::from_array([1, 2, 3, 4, 5, 6, 7, 8]),
        );
        assert_eq!(d, [0, 8, 0, 0]);
        let mut d = [0i32; 4];
        scatter_checked_i32(
            &mut d,
            Simd::from_array([0, 3, 0, 3, 2, 2, 0, 1]),
            Simd::from_array([1, 2, 3, 4, 5, 6, 7, 8]),
        );
        assert_eq!(d, [7, 8, 6, 4]);
    }

    #[test]
    fn duplicates_f32_and_portable_agree() {
        let idx = Simd::from_array([2, 2, 0, 0, 1, 1, 2, 0]);
        let vals = Simd::from_array([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        scatter_checked_f32(&mut a, idx, vals);
        // SAFETY: all indices are < 3.
        unsafe { scatter_f32_portable(b.as_mut_ptr(), idx, vals) };
        assert_eq!(a, [8.0, 6.0, 7.0]);
        assert_eq!(a, b);
    }

    #[test]
    fn oob_leaves_data_untouched() {
        for bad in [-1, 4, i32::MIN, i32::MAX] {
            for lane in 0..8 {
                let mut idx = [0i32; 8];
                idx[lane] = bad;
                let mut d = [5i32; 4];
                let e = try_scatter_i32(&mut d, Simd::from_array(idx), Simd::splat(1)).unwrap_err();
                assert_eq!(e, OutOfBounds { lane, index: bad });
                assert_eq!(d, [5; 4]);
            }
        }
        assert!(try_scatter_i32(&mut [], Simd::splat(0), Simd::splat(1)).is_err());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn checked_panics() {
        scatter_checked_i32(&mut [0; 2], Simd::splat(-1), Simd::splat(1));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn checked_f32_panics() {
        scatter_checked_f32(&mut [0.0; 2], Simd::splat(2), Simd::splat(1.0));
    }

    #[test]
    fn display() {
        let s = std::format!("{}", OutOfBounds { lane: 3, index: -7 });
        assert_eq!(s, "scatter index -7 in lane 3 is out of bounds");
    }

    #[test]
    fn negative_index_unchecked() {
        let mut d = [0i32; 4];
        // SAFETY: base points at element 2, so offsets -2..=1 are in bounds.
        unsafe {
            scatter_i32(
                d[2..].as_mut_ptr(),
                Simd::from_array([-2, -1, 0, 1, -2, -1, 0, 1]),
                Simd::from_array([1, 2, 3, 4, 5, 6, 7, 8]),
            )
        };
        assert_eq!(d, [5, 6, 7, 8]);
    }

    proptest! {
        #[test]
        fn i32_matches_scalar(
            init in proptest::collection::vec(any::<i32>(), 1..32),
            raw in proptest::array::uniform8(any::<u32>()),
            vals in proptest::array::uniform8(any::<i32>()),
        ) {
            let idx = raw.map(|r| (r as usize % init.len()) as i32);
            let mut want = init.clone();
            scalar(&mut want, idx, vals);
            let mut a = init.clone();
            scatter_checked_i32(&mut a, Simd::from_array(idx), Simd::from_array(vals));
            prop_assert_eq!(&a, &want);
            let mut b = init.clone();
            // SAFETY: indices were reduced modulo the length.
            unsafe { scatter_i32_portable(b.as_mut_ptr(), Simd::from_array(idx), Simd::from_array(vals)) };
            prop_assert_eq!(&b, &want);
        }

        #[test]
        fn f32_matches_scalar_bits(
            init in proptest::collection::vec(any::<u32>(), 1..32),
            raw in proptest::array::uniform8(any::<u32>()),
            vals in proptest::array::uniform8(any::<u32>()),
        ) {
            let init: Vec<f32> = init.into_iter().map(f32::from_bits).collect();
            let idx = raw.map(|r| (r as usize % init.len()) as i32);
            let vals = vals.map(f32::from_bits);
            let mut want: Vec<u32> = init.iter().map(|x| x.to_bits()).collect();
            for k in 0..8 {
                want[idx[k] as usize] = vals[k].to_bits();
            }
            let mut a = init.clone();
            scatter_checked_f32(&mut a, Simd::from_array(idx), Simd::from_array(vals));
            let got: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn oob_matches_predicate(
            len in 0usize..16,
            raw in proptest::array::uniform8(any::<i32>()),
        ) {
            let mut d = std::vec![3i32; len];
            let ok = raw.iter().all(|&i| i >= 0 && (i as usize) < len);
            let r = try_scatter_i32(&mut d, Simd::from_array(raw), Simd::splat(0));
            prop_assert_eq!(r.is_ok(), ok);
            if !ok {
                prop_assert!(d.iter().all(|&x| x == 3));
            }
        }
    }
}
