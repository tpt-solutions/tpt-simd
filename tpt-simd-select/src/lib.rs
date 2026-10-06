//! SIMD lane selection and slice selection.
//!
//! * [`select_i32`] / [`select_lanes_f32`] pick lanes of a vector by index
//!   (the `permutevar8x32` pattern). Only the low 3 bits of each index are
//!   used, so every index is in range by construction.
//! * [`select_from_slice_i32`] picks elements of a slice by index using
//!   [`tpt_simd_gather`]; out-of-bounds indices panic. See
//!   [`try_select_from_slice_i32`] for the non-panicking form.
//! * [`select_f32`] is re-exported from [`tpt_simd_blend`]: it is the
//!   condition-based blend, distinct from the index-based [`select_lanes_f32`]
//!   (renamed from the spec's `select_f32` to avoid the name clash).
#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub use tpt_simd_blend::select_f32;
use tpt_simd_core::Simd;

/// Returns `out[i] = v[indices[i] & 7]`.
///
/// # Examples
/// ```
/// use tpt_simd_core::Simd;
/// let v = Simd::from_array([10, 11, 12, 13, 14, 15, 16, 17]);
/// let i = Simd::from_array([7, 0, 0, 1, 8, 9, 3, 3]);
/// assert_eq!(tpt_simd_select::select_i32(v, i).to_array(), [17, 10, 10, 11, 10, 11, 13, 13]);
/// ```
#[inline]
pub fn select_i32(v: Simd<i32, 8>, indices: Simd<i32, 8>) -> Simd<i32, 8> {
    let (v, idx) = (v.to_array(), indices.to_array());
    Simd::from_array(core::array::from_fn(|i| v[(idx[i] & 7) as usize]))
}

/// Returns `out[i] = v[indices[i] & 7]` for `f32` lanes.
///
/// # Examples
/// ```
/// use tpt_simd_core::Simd;
/// let v = Simd::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
/// let i = Simd::from_array([7, 6, 5, 4, 3, 2, 1, 0]);
/// assert_eq!(tpt_simd_select::select_lanes_f32(v, i).to_array(), [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
/// ```
#[inline]
pub fn select_lanes_f32(v: Simd<f32, 8>, indices: Simd<i32, 8>) -> Simd<f32, 8> {
    let (v, idx) = (v.to_array(), indices.to_array());
    Simd::from_array(core::array::from_fn(|i| v[(idx[i] & 7) as usize]))
}

/// Returns `out[i] = data[indices[i]]`.
///
/// # Panics
/// Panics if any index is negative or `>= data.len()`.
///
/// # Examples
/// ```
/// use tpt_simd_core::Simd;
/// let data = [5, 6, 7, 8];
/// let i = Simd::from_array([3, 2, 1, 0, 0, 1, 2, 3]);
/// assert_eq!(tpt_simd_select::select_from_slice_i32(&data, i).to_array(), [8, 7, 6, 5, 5, 6, 7, 8]);
/// ```
#[inline]
pub fn select_from_slice_i32(data: &[i32], indices: Simd<i32, 8>) -> Simd<i32, 8> {
    tpt_simd_gather::gather_checked_i32(data, indices)
}

/// Like [`select_from_slice_i32`] but returns `None` on an out-of-bounds index.
///
/// # Examples
/// ```
/// use tpt_simd_core::Simd;
/// assert!(tpt_simd_select::try_select_from_slice_i32(&[1, 2], Simd::splat(2)).is_none());
/// ```
#[inline]
pub fn try_select_from_slice_i32(data: &[i32], indices: Simd<i32, 8>) -> Option<Simd<i32, 8>> {
    tpt_simd_gather::try_gather_i32(data, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn select_matches_scalar(v in any::<[i32; 8]>(), idx in any::<[i32; 8]>()) {
            let out = select_i32(Simd::from_array(v), Simd::from_array(idx)).to_array();
            for i in 0..8 { prop_assert_eq!(out[i], v[(idx[i] & 7) as usize]); }
        }

        #[test]
        fn slice_matches_scalar(data in proptest::collection::vec(any::<i32>(), 1..64), raw in any::<[u32; 8]>()) {
            let idx: [i32; 8] = core::array::from_fn(|i| (raw[i] % data.len() as u32) as i32);
            let out = select_from_slice_i32(&data, Simd::from_array(idx)).to_array();
            for i in 0..8 { prop_assert_eq!(out[i], data[idx[i] as usize]); }
        }
    }

    #[test]
    #[should_panic]
    fn slice_oob_panics() {
        select_from_slice_i32(&[1, 2, 3], Simd::splat(3));
    }

    #[test]
    fn slice_negative_is_none() {
        assert!(try_select_from_slice_i32(&[1, 2, 3], Simd::splat(-1)).is_none());
    }
}
