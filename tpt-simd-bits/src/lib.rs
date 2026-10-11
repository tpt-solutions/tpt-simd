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

extern crate alloc;

use alloc::vec::Vec;

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

/// Rank/select index over a bit-vector stored as `u64` words, bit `i` being
/// bit `i % 64` of word `i / 64`.
///
/// Building costs one pass and one `u32` per word. `rank1` is O(1); `select1`
/// is O(log words) plus an in-word scan.
#[derive(Clone, Debug)]
pub struct RankSelect<'a> {
    words: &'a [u64],
    len: usize,
    /// `prefix[w]` is the number of set bits in words before `w`.
    prefix: Vec<u32>,
}

impl<'a> RankSelect<'a> {
    /// Indexes the first `len` bits of `words`. Panics if `len` exceeds
    /// `words.len() * 64`.
    #[must_use]
    pub fn new(words: &'a [u64], len: usize) -> Self {
        assert!(
            len <= words.len() * 64,
            "RankSelect::new: {len} bits do not fit in {} words",
            words.len()
        );
        let mut prefix = Vec::with_capacity(words.len() + 1);
        let mut total = 0u32;
        prefix.push(0);
        for w in 0..words.len() {
            total += masked(words, len, w).count_ones();
            prefix.push(total);
        }
        Self { words, len, prefix }
    }

    /// Number of indexed bits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if no bits are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of set bits among the indexed bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.prefix[self.words.len()] as usize
    }

    /// Number of set bits in positions `[0, pos)`. Panics if `pos > len()`.
    #[must_use]
    pub fn rank1(&self, pos: usize) -> usize {
        assert!(
            pos <= self.len,
            "rank1: position {pos} past length {}",
            self.len
        );
        let w = pos / 64;
        let r = pos % 64;
        let mut count = self.prefix[w] as usize;
        if r != 0 {
            count += (masked(self.words, self.len, w) & ((1u64 << r) - 1)).count_ones() as usize;
        }
        count
    }

    /// Position of the `k`-th set bit (0-based), or `None` if there are not
    /// `k + 1` set bits.
    #[must_use]
    pub fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.count_ones() {
            return None;
        }
        let k = k as u32;
        // Largest w with prefix[w] <= k; that word holds the k-th bit.
        let w = self.prefix.partition_point(|&p| p <= k) - 1;
        let mut word = masked(self.words, self.len, w);
        for _ in 0..(k - self.prefix[w]) {
            word &= word - 1; // clear the lowest set bit
        }
        Some(w * 64 + word.trailing_zeros() as usize)
    }
}

/// Word `w` with bits at or beyond `len` cleared.
#[inline]
fn masked(words: &[u64], len: usize, w: usize) -> u64 {
    let base = w * 64;
    if base + 64 <= len {
        words[w]
    } else if base >= len {
        0
    } else {
        words[w] & ((1u64 << (len - base)) - 1)
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

#[cfg(test)]
mod rank_select_tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn bit(words: &[u64], i: usize) -> bool {
        (words[i / 64] >> (i % 64)) & 1 == 1
    }

    #[test]
    fn rank_and_select_match_naive_for_partial_lengths() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let words: Vec<u64> = (0..6)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect();
        for len in [0usize, 1, 63, 64, 65, 127, 128, 200, 384] {
            let rs = RankSelect::new(&words, len);
            let mut naive_rank = 0;
            let mut ones = Vec::new();
            for i in 0..len {
                assert_eq!(rs.rank1(i), naive_rank, "len={len} rank at {i}");
                if bit(&words, i) {
                    naive_rank += 1;
                    ones.push(i);
                }
            }
            assert_eq!(rs.rank1(len), naive_rank, "len={len}");
            assert_eq!(rs.count_ones(), ones.len(), "len={len}");
            for (k, &pos) in ones.iter().enumerate() {
                assert_eq!(rs.select1(k), Some(pos), "len={len} k={k}");
            }
            assert_eq!(rs.select1(ones.len()), None, "len={len}");
        }
    }

    #[test]
    fn empty_and_sparse_cases() {
        let rs = RankSelect::new(&[], 0);
        assert!(rs.is_empty());
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.select1(0), None);
        let words = [0u64, 0, 1 << 63];
        let rs = RankSelect::new(&words, 192);
        assert_eq!(rs.select1(0), Some(191));
        assert_eq!(rs.rank1(191), 0);
    }

    #[test]
    #[should_panic(expected = "do not fit in 1 words")]
    fn length_beyond_words_panics() {
        let _ = RankSelect::new(&[0u64], 65);
    }
}
