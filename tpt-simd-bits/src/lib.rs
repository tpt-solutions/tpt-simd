//! Bit-level and integer-stream kernels.
//!
//! | group | functions |
//! |---|---|
//! | bitsets | [`popcount_u64`], [`bitset_and`], [`bitset_or`], [`bitset_xor`], [`bitset_andnot`] |
//! | zigzag | [`zigzag_encode_i32`], [`zigzag_decode_u32`], [`zigzag_encode_i64`], [`zigzag_decode_u64`] |
//! | streams | [`delta_encode_u32`], [`delta_decode_u32`], [`prefix_sum_u32`], [`prefix_sum_u64`] |
//!
//! ## Length policy
//!
//! Paired slices must have the same length; otherwise the function **panics**.
//! Empty input is valid.
//!
//! ## Arithmetic
//!
//! All integer arithmetic is **wrapping**, so results are the same in debug and
//! release builds. Delta and prefix-sum outputs wrap modulo 2^N.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_bits::{delta_encode_u32, delta_decode_u32};
//!
//! let mut v = [10u32, 12, 15, 15, 20];
//! delta_encode_u32(&mut v);
//! assert_eq!(v, [10, 2, 3, 0, 5]);
//! delta_decode_u32(&mut v);
//! assert_eq!(v, [10, 12, 15, 15, 20]);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cold]
#[inline(never)]
fn len_mismatch(func: &str, a: usize, b: usize) -> ! {
    panic!("{func}: slice lengths differ ({a} vs {b})");
}

/// Total number of set bits across `words`.
///
/// # Examples
///
/// ```
/// assert_eq!(tpt_simd_bits::popcount_u64(&[0b1011, u64::MAX]), 3 + 64);
/// ```
#[must_use]
pub fn popcount_u64(words: &[u64]) -> u64 {
    words.iter().map(|w| u64::from(w.count_ones())).sum()
}

macro_rules! bitset_op {
    ($(#[$doc:meta])* $name:ident, $func:literal, |$x:ident, $y:ident| $body:expr) => {
        $(#[$doc])*
        ///
        /// # Panics
        ///
        /// Panics if the three slices do not have the same length.
        pub fn $name(dst: &mut [u64], a: &[u64], b: &[u64]) {
            if dst.len() != a.len() {
                len_mismatch($func, dst.len(), a.len());
            }
            if a.len() != b.len() {
                len_mismatch($func, a.len(), b.len());
            }
            for ((d, &$x), &$y) in dst.iter_mut().zip(a).zip(b) {
                *d = $body;
            }
        }
    };
}

bitset_op!(
    /// Writes `a & b` into `dst`.
    bitset_and, "bitset_and", |x, y| x & y
);
bitset_op!(
    /// Writes `a | b` into `dst`.
    bitset_or, "bitset_or", |x, y| x | y
);
bitset_op!(
    /// Writes `a ^ b` into `dst`.
    bitset_xor, "bitset_xor", |x, y| x ^ y
);
bitset_op!(
    /// Writes `a & !b` into `dst`.
    bitset_andnot, "bitset_andnot", |x, y| x & !y
);

/// Maps a signed 32-bit value to unsigned so small magnitudes stay small.
///
/// `0, -1, 1, -2, 2, ...` map to `0, 1, 2, 3, 4, ...`.
#[must_use]
pub const fn zigzag_encode_i32(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// Inverse of [`zigzag_encode_i32`].
#[must_use]
pub const fn zigzag_decode_u32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

/// Maps a signed 64-bit value to unsigned; see [`zigzag_encode_i32`].
#[must_use]
pub const fn zigzag_encode_i64(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Inverse of [`zigzag_encode_i64`].
#[must_use]
pub const fn zigzag_decode_u64(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Replaces each element with its wrapping difference from the previous one.
///
/// The first element is unchanged. Inverse: [`delta_decode_u32`].
pub fn delta_encode_u32(values: &mut [u32]) {
    for i in (1..values.len()).rev() {
        values[i] = values[i].wrapping_sub(values[i - 1]);
    }
}

/// Inverse of [`delta_encode_u32`]: an in-place wrapping inclusive prefix sum.
pub fn delta_decode_u32(values: &mut [u32]) {
    prefix_sum_u32(values);
}

/// In-place wrapping inclusive prefix sum.
pub fn prefix_sum_u32(values: &mut [u32]) {
    let mut acc = 0u32;
    for v in values {
        acc = acc.wrapping_add(*v);
        *v = acc;
    }
}

/// In-place wrapping inclusive prefix sum over `u64`.
pub fn prefix_sum_u64(values: &mut [u64]) {
    let mut acc = 0u64;
    for v in values {
        acc = acc.wrapping_add(*v);
        *v = acc;
    }
}

/// Zigzag-encodes `src` into `dst`.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn zigzag_encode_slice_i32(src: &[i32], dst: &mut [u32]) {
    if src.len() != dst.len() {
        len_mismatch("zigzag_encode_slice_i32", src.len(), dst.len());
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = zigzag_encode_i32(s);
    }
}

/// Zigzag-decodes `src` into `dst`.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn zigzag_decode_slice_u32(src: &[u32], dst: &mut [i32]) {
    if src.len() != dst.len() {
        len_mismatch("zigzag_decode_slice_u32", src.len(), dst.len());
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = zigzag_decode_u32(s);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    #[test]
    fn popcount_matches_per_word_sum() {
        let w: Vec<u64> = (0..50u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let expected: u64 = w.iter().map(|x| u64::from(x.count_ones())).sum();
        assert_eq!(popcount_u64(&w), expected);
        assert_eq!(popcount_u64(&[]), 0);
    }

    #[test]
    fn bitset_ops_truth_table() {
        let a = [0b1100u64];
        let b = [0b1010u64];
        let mut d = [0u64];
        bitset_and(&mut d, &a, &b);
        assert_eq!(d, [0b1000]);
        bitset_or(&mut d, &a, &b);
        assert_eq!(d, [0b1110]);
        bitset_xor(&mut d, &a, &b);
        assert_eq!(d, [0b0110]);
        bitset_andnot(&mut d, &a, &b);
        assert_eq!(d, [0b0100]);
    }

    #[test]
    #[should_panic(expected = "bitset_or: slice lengths differ (1 vs 2)")]
    fn bitset_length_mismatch_panics() {
        let mut d = [0u64];
        bitset_or(&mut d, &[1], &[1, 2]);
    }

    #[test]
    fn zigzag_known_values_and_roundtrip() {
        assert_eq!(zigzag_encode_i32(0), 0);
        assert_eq!(zigzag_encode_i32(-1), 1);
        assert_eq!(zigzag_encode_i32(1), 2);
        assert_eq!(zigzag_encode_i32(i32::MIN), u32::MAX);
        assert_eq!(zigzag_encode_i32(i32::MAX), u32::MAX - 1);
        assert_eq!(zigzag_encode_i64(-2), 3);
        for v in [i32::MIN, -7, -1, 0, 1, 7, i32::MAX] {
            assert_eq!(zigzag_decode_u32(zigzag_encode_i32(v)), v);
        }
        for v in [i64::MIN, -3, 0, 3, i64::MAX] {
            assert_eq!(zigzag_decode_u64(zigzag_encode_i64(v)), v);
        }
    }

    #[test]
    fn zigzag_slices() {
        let src = [0i32, -1, 1, i32::MIN];
        let mut enc = [0u32; 4];
        zigzag_encode_slice_i32(&src, &mut enc);
        let mut dec = [0i32; 4];
        zigzag_decode_slice_u32(&enc, &mut dec);
        assert_eq!(dec, src);
    }

    #[test]
    fn delta_round_trip_wraps() {
        let orig = [u32::MAX - 1, 3, 3, 0, 100];
        let mut v = orig;
        delta_encode_u32(&mut v);
        assert_eq!(v[0], u32::MAX - 1);
        assert_eq!(v[1], 5); // 3 - (2^32 - 2) wraps to 5
        delta_decode_u32(&mut v);
        assert_eq!(v, orig);
        let mut empty: [u32; 0] = [];
        delta_encode_u32(&mut empty);
        delta_decode_u32(&mut empty);
    }

    #[test]
    fn prefix_sums() {
        let mut a = [1u32, 2, 3, u32::MAX];
        prefix_sum_u32(&mut a);
        assert_eq!(a, [1, 3, 6, 5]);
        let mut b = [u64::MAX, 2];
        prefix_sum_u64(&mut b);
        assert_eq!(b, [u64::MAX, 1]);
    }
}
