//! Byte search and line counting.
//!
//! | function | notes |
//! |---|---|
//! | [`find_byte`], [`rfind_byte`], [`count_byte`] | own word-at-a-time (SWAR) code; `memchr` with the `memchr` feature |
//! | [`find`] | own Two-Way substring search, linear in the haystack; `memchr::memmem` with the `memchr` feature |
//! | [`find_ascii_case_insensitive`] | own, ASCII-only case folding |
//! | [`count_lines`] | counts `\n`, plus an unterminated last line |
//!
//! There are no runtime dependencies by default. `memchr` is an optional
//! feature, and it is also a dev-dependency used as a differential-test oracle.
//!
//! The own substring search is the Two-Way algorithm, so its worst case is
//! linear in the haystack length.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_search::{count_lines, find, find_ascii_case_insensitive};
//!
//! assert_eq!(find(b"hello world", b"world"), Some(6));
//! assert_eq!(find_ascii_case_insensitive(b"Hello WORLD", b"world"), Some(6));
//! assert_eq!(count_lines(b"a\nb\nc"), 3);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]
#[cfg(feature = "runtime-dispatch")]
extern crate std;

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

/// Index of the first `needle` byte in `haystack`.
#[must_use]
pub fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(feature = "memchr")]
    {
        memchr::memchr(needle, haystack)
    }
    #[cfg(all(not(feature = "memchr"), target_arch = "x86_64"))]
    {
        if haystack.len() >= 64 && avx2::available() {
            avx2::find_byte(haystack, needle)
        } else {
            sse2::find_byte(haystack, needle)
        }
    }
    #[cfg(all(not(feature = "memchr"), not(target_arch = "x86_64")))]
    {
        find_byte_swar(haystack, needle)
    }
}

/// SSE2 kernels. SSE2 is part of the x86_64 baseline, so no runtime detection
/// is needed. Each loop reads 16-byte unaligned vectors that lie inside the slice.
#[cfg(all(not(feature = "memchr"), target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod sse2 {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_sad_epu8,
        _mm_set1_epi8, _mm_setzero_si128, _mm_storeu_si128, _mm_sub_epi8,
    };

    /// Index of the first `needle` byte, scanning 64 bytes per iteration.
    pub(crate) fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
        let mut i = 0;
        // SAFETY: every load reads 16 bytes at `i + k` with `i + 64 <= len`,
        // so all reads are inside `haystack`. SSE2 is always available on x86_64.
        unsafe {
            let v = _mm_set1_epi8(needle as i8);
            while i + 64 <= haystack.len() {
                let p = haystack.as_ptr().add(i).cast::<__m128i>();
                let c0 = _mm_cmpeq_epi8(_mm_loadu_si128(p), v);
                let c1 = _mm_cmpeq_epi8(_mm_loadu_si128(p.add(1)), v);
                let c2 = _mm_cmpeq_epi8(_mm_loadu_si128(p.add(2)), v);
                let c3 = _mm_cmpeq_epi8(_mm_loadu_si128(p.add(3)), v);
                let any = _mm_or_si128(_mm_or_si128(c0, c1), _mm_or_si128(c2, c3));
                if _mm_movemask_epi8(any) != 0 {
                    for (k, c) in [c0, c1, c2, c3].into_iter().enumerate() {
                        let m = _mm_movemask_epi8(c) as u32;
                        if m != 0 {
                            return Some(i + 16 * k + m.trailing_zeros() as usize);
                        }
                    }
                }
                i += 64;
            }
            while i + 16 <= haystack.len() {
                let c = _mm_cmpeq_epi8(
                    _mm_loadu_si128(haystack.as_ptr().add(i).cast::<__m128i>()),
                    v,
                );
                let m = _mm_movemask_epi8(c) as u32;
                if m != 0 {
                    return Some(i + m.trailing_zeros() as usize);
                }
                i += 16;
            }
        }
        haystack[i..]
            .iter()
            .position(|&b| b == needle)
            .map(|p| i + p)
    }

    /// Number of `needle` bytes in `haystack`.
    pub(crate) fn count_byte(haystack: &[u8], needle: u8) -> usize {
        // Each match is a lane of -1, so subtracting the compare masks counts matches in
        // byte lanes. A lane holds at most 255 counts, so the lanes are flushed with
        // `psadbw` every 255 blocks (4080 bytes).
        const BLOCK: usize = 16;
        const MAX_BLOCKS: usize = 255;
        let mut i = 0;
        let mut total = 0usize;
        // SAFETY: every load is 16 bytes at `i + 16 * k`, inside `haystack` by the loop bounds.
        unsafe {
            let v = _mm_set1_epi8(needle as i8);
            let zero = _mm_setzero_si128();
            let flush = |acc: __m128i| {
                let sums = _mm_sad_epu8(acc, zero);
                let mut lanes = [0u64; 2];
                _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), sums);
                (lanes[0] + lanes[1]) as usize
            };
            while i + BLOCK * MAX_BLOCKS <= haystack.len() {
                let mut acc = zero;
                for k in 0..MAX_BLOCKS {
                    let x = _mm_loadu_si128(haystack.as_ptr().add(i + BLOCK * k).cast());
                    acc = _mm_sub_epi8(acc, _mm_cmpeq_epi8(x, v));
                }
                total += flush(acc);
                i += BLOCK * MAX_BLOCKS;
            }
            let mut acc = zero;
            while i + BLOCK <= haystack.len() {
                let x = _mm_loadu_si128(haystack.as_ptr().add(i).cast());
                acc = _mm_sub_epi8(acc, _mm_cmpeq_epi8(x, v));
                i += BLOCK;
            }
            total += flush(acc);
        }
        total + haystack[i..].iter().filter(|&&b| b == needle).count()
    }
}

/// AVX2 kernels, selected at runtime under `runtime-dispatch` (or at compile time when
/// the build enables AVX2). Each loop reads 32-byte unaligned vectors inside the slice.
#[cfg(all(not(feature = "memchr"), target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
        _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_sub_epi8,
    };

    /// `true` if the AVX2 kernels may be called on this CPU.
    pub(crate) fn available() -> bool {
        #[cfg(feature = "runtime-dispatch")]
        {
            !cfg!(miri) && std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(feature = "runtime-dispatch"))]
        {
            cfg!(target_feature = "avx2")
        }
    }

    /// Index of the first `needle` byte. Requires `available()`.
    pub(crate) fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
        // SAFETY: the caller checked `available()`, so AVX2 is supported.
        unsafe { find_byte_impl(haystack, needle) }
    }

    /// Number of `needle` bytes in `haystack`. Requires `available()`.
    pub(crate) fn count_byte(haystack: &[u8], needle: u8) -> usize {
        // SAFETY: the caller checked `available()`, so AVX2 is supported.
        unsafe { count_byte_impl(haystack, needle) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn find_byte_impl(haystack: &[u8], needle: u8) -> Option<usize> {
        let len = haystack.len();
        let mut i = 0;
        // SAFETY: every load reads 32 bytes at an offset `< len` (the loop bounds keep them
        // inside `haystack`). `loadu` has no alignment requirement.
        unsafe {
            let v = _mm256_set1_epi8(needle as i8);
            while i + 256 <= len {
                let p = haystack.as_ptr().add(i).cast::<__m256i>();
                let c0 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p), v);
                let c1 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(1)), v);
                let c2 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(2)), v);
                let c3 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(3)), v);
                let c4 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(4)), v);
                let c5 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(5)), v);
                let c6 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(6)), v);
                let c7 = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(7)), v);
                // OR-reduce the eight masks, so the common no-hit case takes one branch.
                let any = _mm256_or_si256(
                    _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3)),
                    _mm256_or_si256(_mm256_or_si256(c4, c5), _mm256_or_si256(c6, c7)),
                );
                if _mm256_movemask_epi8(any) != 0 {
                    for (k, c) in [c0, c1, c2, c3, c4, c5, c6, c7].into_iter().enumerate() {
                        let m = _mm256_movemask_epi8(c) as u32;
                        if m != 0 {
                            return Some(i + 32 * k + m.trailing_zeros() as usize);
                        }
                    }
                }
                i += 256;
            }
            while i + 32 <= len {
                let c = _mm256_cmpeq_epi8(
                    _mm256_loadu_si256(haystack.as_ptr().add(i).cast::<__m256i>()),
                    v,
                );
                let m = _mm256_movemask_epi8(c) as u32;
                if m != 0 {
                    return Some(i + m.trailing_zeros() as usize);
                }
                i += 32;
            }
        }
        haystack[i..]
            .iter()
            .position(|&b| b == needle)
            .map(|p| i + p)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn count_byte_impl(haystack: &[u8], needle: u8) -> usize {
        // Byte lanes count matches (subtracting the -1 compare masks), and are flushed with
        // `psadbw` every 255 blocks of 32 bytes, before a lane can overflow.
        const BLOCK: usize = 32;
        const MAX_BLOCKS: usize = 255;
        let len = haystack.len();
        let mut i = 0;
        let mut total = 0usize;
        // SAFETY: every load reads 32 bytes at an offset inside `haystack` (loop bounds).
        unsafe {
            let v = _mm256_set1_epi8(needle as i8);
            let zero = _mm256_setzero_si256();
            while i + BLOCK * MAX_BLOCKS <= len {
                let mut acc = zero;
                for k in 0..MAX_BLOCKS {
                    let x = _mm256_loadu_si256(haystack.as_ptr().add(i + BLOCK * k).cast());
                    acc = _mm256_sub_epi8(acc, _mm256_cmpeq_epi8(x, v));
                }
                total += flush256(acc);
                i += BLOCK * MAX_BLOCKS;
            }
            let mut acc = zero;
            while i + BLOCK <= len {
                let x = _mm256_loadu_si256(haystack.as_ptr().add(i).cast());
                acc = _mm256_sub_epi8(acc, _mm256_cmpeq_epi8(x, v));
                i += BLOCK;
            }
            total += flush256(acc);
        }
        total + haystack[i..].iter().filter(|&&b| b == needle).count()
    }

    /// Sums the byte lanes of `acc` (each lane at most 255) into a count.
    #[target_feature(enable = "avx2")]
    unsafe fn flush256(acc: __m256i) -> usize {
        // SAFETY: the destination array is exactly the size of the stored vector.
        unsafe {
            let sums = _mm256_sad_epu8(acc, _mm256_setzero_si256());
            let mut lanes = [0u64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums);
            lanes.iter().sum::<u64>() as usize
        }
    }
}

/// Word-at-a-time search: a 64-bit word contains a zero byte exactly when
/// `(x - LO) & !x & HI` is non-zero, where `x` is the word XOR the needle
/// broadcast. The byte position is then found with a short scan.
#[cfg_attr(any(feature = "memchr", target_arch = "x86_64"), allow(dead_code))]
fn find_byte_swar(haystack: &[u8], needle: u8) -> Option<usize> {
    let pattern = LO * u64::from(needle);
    let mut chunks = haystack.chunks_exact(8);
    let mut offset = 0;
    for chunk in &mut chunks {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        let x = u64::from_le_bytes(word) ^ pattern;
        if x.wrapping_sub(LO) & !x & HI != 0 {
            return chunk.iter().position(|&b| b == needle).map(|i| offset + i);
        }
        offset += 8;
    }
    chunks
        .remainder()
        .iter()
        .position(|&b| b == needle)
        .map(|i| offset + i)
}

/// Index of the last `needle` byte in `haystack`.
#[must_use]
pub fn rfind_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(feature = "memchr")]
    {
        memchr::memrchr(needle, haystack)
    }
    #[cfg(not(feature = "memchr"))]
    {
        haystack.iter().rposition(|&b| b == needle)
    }
}

/// Number of occurrences of `needle` in `haystack`.
#[must_use]
pub fn count_byte(haystack: &[u8], needle: u8) -> usize {
    #[cfg(feature = "memchr")]
    {
        memchr::memchr_iter(needle, haystack).count()
    }
    #[cfg(all(not(feature = "memchr"), target_arch = "x86_64"))]
    {
        if haystack.len() >= 64 && avx2::available() {
            avx2::count_byte(haystack, needle)
        } else {
            sse2::count_byte(haystack, needle)
        }
    }
    #[cfg(all(not(feature = "memchr"), not(target_arch = "x86_64")))]
    {
        haystack.iter().filter(|&&b| b == needle).count()
    }
}

/// Index of the first occurrence of `needle` in `haystack`. An empty needle
/// matches at 0.
#[must_use]
pub fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    #[cfg(feature = "memchr")]
    {
        memchr::memmem::find(haystack, needle)
    }
    #[cfg(not(feature = "memchr"))]
    {
        prefiltered_find(haystack, needle)
    }
}

/// Candidate verifications allowed before [`prefiltered_find`] hands the rest
/// of the haystack to [`two_way_find`]. This bounds the work on adversarial input.
#[cfg(not(feature = "memchr"))]
const MAX_FAILED_CANDIDATES: usize = 32;

/// Skips to positions where the needle's last byte matches, verifies each
/// candidate, and falls back to Two-Way after too many failures.
#[cfg(not(feature = "memchr"))]
fn prefiltered_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let m = needle.len();
    if m == 0 {
        return Some(0);
    }
    let last = needle[m - 1];
    let mut pos = 0usize;
    let mut failed = 0usize;
    while pos + m <= haystack.len() {
        let end = pos + m - 1;
        let at = end + find_byte(&haystack[end..], last)?;
        let start = at + 1 - m;
        if &haystack[start..=at] == needle {
            return Some(start);
        }
        pos = start + 1;
        failed += 1;
        if failed > MAX_FAILED_CANDIDATES {
            return two_way_find(&haystack[pos..], needle).map(|i| pos + i);
        }
    }
    None
}

/// Maximal suffix of `arr` under the normal order (or the reversed order when
/// `reversed`), returned as `(start, period)`. Helper for [`critical_factorisation`].
#[cfg(not(feature = "memchr"))]
fn maximal_suffix(arr: &[u8], reversed: bool) -> (usize, usize) {
    let (mut left, mut right, mut offset, mut period) = (0usize, 1usize, 0usize, 1usize);
    while let (Some(&a), Some(&b)) = (arr.get(right + offset), arr.get(left + offset)) {
        let less = if reversed { a > b } else { a < b };
        if less {
            right += offset + 1;
            offset = 0;
            period = right - left;
        } else if a == b {
            if offset + 1 == period {
                right += offset + 1;
                offset = 0;
            } else {
                offset += 1;
            }
        } else {
            left = right;
            right += 1;
            offset = 0;
            period = 1;
        }
    }
    (left, period)
}

/// Crochemore–Perrin critical factorisation of a non-empty `needle`.
///
/// Returns `(crit, period, long_period)`. When `long_period` is false the
/// needle is periodic with `period` and the searcher keeps a memory of matched
/// bytes between windows; otherwise the shift is `max(crit, m - crit) + 1`.
#[cfg(not(feature = "memchr"))]
fn critical_factorisation(needle: &[u8]) -> (usize, usize, bool) {
    let (crit_f, period_f) = maximal_suffix(needle, false);
    let (crit_r, period_r) = maximal_suffix(needle, true);
    let (crit, period) = if crit_f > crit_r {
        (crit_f, period_f)
    } else {
        (crit_r, period_r)
    };
    let m = needle.len();
    if period + crit <= m && needle[..crit] == needle[period..period + crit] {
        (crit, period, false)
    } else {
        (crit, crit.max(m - crit) + 1, true)
    }
}

/// Two-Way substring search: linear in the haystack length for any needle.
#[cfg(not(feature = "memchr"))]
fn two_way_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let m = needle.len();
    if m == 0 {
        return Some(0);
    }
    let (crit, period, long_period) = critical_factorisation(needle);
    let mut position = 0usize;
    let mut memory = 0usize;
    while position + m <= haystack.len() {
        let window = &haystack[position..position + m];
        // Right half first, from the critical position.
        let start = if long_period { crit } else { crit.max(memory) };
        if let Some(i) = (start..m).find(|&i| needle[i] != window[i]) {
            position += i - crit + 1;
            if !long_period {
                memory = 0;
            }
            continue;
        }
        // Then the left half, right to left.
        let start = if long_period { 0 } else { memory };
        if (start..crit).rev().any(|i| needle[i] != window[i]) {
            position += period;
            if !long_period {
                memory = m - period;
            }
            continue;
        }
        return Some(position);
    }
    None
}

/// Index of the first occurrence of `needle` in `haystack`, comparing ASCII
/// letters without regard to case. Other bytes must match exactly. An empty
/// needle matches at 0.
///
/// The first needle byte is located by scanning for its upper and lower case
/// forms, so most non-matching positions are skipped quickly.
#[must_use]
pub fn find_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let Some((&first, rest)) = needle.split_first() else {
        return Some(0);
    };
    if haystack.len() < needle.len() {
        return None;
    }
    let last_start = haystack.len() - needle.len();
    let lower = first.to_ascii_lowercase();
    let upper = first.to_ascii_uppercase();
    let mut pos = 0;
    while pos <= last_start {
        let found = haystack[pos..=last_start]
            .iter()
            .position(|&b| b == lower || b == upper)?;
        let at = pos + found;
        if haystack[at + 1..at + needle.len()].eq_ignore_ascii_case(rest) {
            return Some(at);
        }
        pos = at + 1;
    }
    None
}

/// Number of lines in `bytes`: the count of `\n` bytes, plus one if the input
/// is non-empty and does not end with `\n`.
///
/// `count_lines(b"")` is 0, `count_lines(b"a\n")` is 1, and `count_lines(b"a")`
/// is 1.
#[must_use]
pub fn count_lines(bytes: &[u8]) -> usize {
    let newlines = count_byte(bytes, b'\n');
    match bytes.last() {
        Some(&b) if b != b'\n' => newlines + 1,
        _ => newlines,
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn naive_find(h: &[u8], n: &[u8]) -> Option<usize> {
        if n.is_empty() {
            return Some(0);
        }
        if h.len() < n.len() {
            return None;
        }
        (0..=h.len() - n.len()).find(|&i| &h[i..i + n.len()] == n)
    }

    fn naive_ci(h: &[u8], n: &[u8]) -> Option<usize> {
        if n.is_empty() {
            return Some(0);
        }
        if h.len() < n.len() {
            return None;
        }
        (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
    }

    #[test]
    fn byte_search_functions() {
        let h = b"abcabc";
        assert_eq!(find_byte(h, b'c'), Some(2));
        assert_eq!(rfind_byte(h, b'c'), Some(5));
        assert_eq!(count_byte(h, b'a'), 2);
        assert_eq!(find_byte(b"", b'a'), None);
    }

    #[test]
    fn count_byte_matches_naive_across_the_lane_flush() {
        // Inputs past 4080 bytes are counted in several flushes of the byte lanes.
        for len in [4079usize, 4080, 4081, 8160, 8161, 12_345, 100_000] {
            let h: Vec<u8> = (0..len)
                .map(|i| if i % 7 == 0 { b'Z' } else { b'a' })
                .collect();
            assert_eq!(
                count_byte(&h, b'Z'),
                h.iter().filter(|&&b| b == b'Z').count(),
                "len {len}"
            );
            let all = std::vec![b'Z'; len];
            assert_eq!(count_byte(&all, b'Z'), len, "all-match len {len}");
        }
    }

    #[test]
    fn byte_search_matches_naive_at_every_length_and_position() {
        // Covers the 64-byte loop, the 16-byte loop and the scalar tail.
        for len in 0..=200usize {
            for pos in 0..len {
                let mut h = std::vec![b'a'; len];
                h[pos] = b'Z';
                assert_eq!(find_byte(&h, b'Z'), Some(pos), "len {len} pos {pos}");
                assert_eq!(count_byte(&h, b'Z'), 1);
                assert_eq!(count_byte(&h, b'a'), len - 1);
            }
            let h = std::vec![b'a'; len];
            assert_eq!(find_byte(&h, b'Z'), None);
            assert_eq!(count_byte(&h, b'Z'), 0);
        }
        // High-bit bytes must not be confused with the needle.
        let h: Vec<u8> = (0..300u32).map(|i| (i * 37 % 256) as u8).collect();
        for needle in 0..=255u8 {
            let want = h.iter().position(|&b| b == needle);
            assert_eq!(find_byte(&h, needle), want);
            assert_eq!(
                count_byte(&h, needle),
                h.iter().filter(|&&b| b == needle).count()
            );
        }
    }

    #[test]
    fn swar_find_byte_matches_position_at_every_offset() {
        // A needle placed at each position of a buffer longer than a word
        // exercises the word path and the remainder path.
        for len in 0..40 {
            for at in 0..len {
                let mut h = [b'x'; 40];
                h[at] = b'Q';
                assert_eq!(
                    find_byte_swar(&h[..len], b'Q'),
                    Some(at),
                    "len={len} at={at}"
                );
            }
            let h = [b'x'; 40];
            assert_eq!(find_byte_swar(&h[..len], b'Q'), None, "len={len}");
        }
    }

    #[test]
    fn swar_handles_high_bit_bytes() {
        let h = [0x80u8, 0xFF, 0x7F, 0x00, 0x81, 0xC0, 0xFE, 0x01, 0xAA];
        for &needle in &h {
            let want = h.iter().position(|&b| b == needle);
            assert_eq!(find_byte_swar(&h, needle), want, "needle {needle:#x}");
        }
    }

    #[test]
    fn substring_find_matches_naive() {
        let hay: Vec<u8> = (0..300u32)
            .map(|i| b"abAB \n"[(i * 7 % 6) as usize])
            .collect();
        for needle in [&b""[..], b"a", b"ab", b"AB \n", b"zz", b"abAB \nab"] {
            assert_eq!(find(&hay, needle), naive_find(&hay, needle), "{needle:?}");
        }
    }

    #[test]
    fn substring_find_matches_naive_for_periodic_needles() {
        let hays: [&[u8]; 3] = [
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            b"abababababbababababaabababab",
            b"aabaabaaabaabaaaabaabaaabaa",
        ];
        let needles: [&[u8]; 9] = [
            b"a",
            b"aa",
            b"aaa",
            b"aaaaaaaZ",
            b"ab",
            b"aba",
            b"abab",
            b"aabaa",
            b"baab",
        ];
        for hay in hays {
            for needle in needles {
                assert_eq!(
                    find(hay, needle),
                    naive_find(hay, needle),
                    "{needle:?} in {hay:?}"
                );
            }
        }
    }

    #[test]
    fn substring_find_falls_back_on_many_false_candidates() {
        // The last needle byte 'b' occurs everywhere, so most candidates fail
        // and the search must switch to Two-Way partway through.
        let mut hay = std::vec![b'a'; 2000];
        for i in (0..hay.len()).step_by(3) {
            hay[i] = b'b';
        }
        let needle = b"aaab";
        assert_eq!(find(&hay, needle), naive_find(&hay, needle));
        hay.extend_from_slice(b"XaaaaZ");
        assert_eq!(find(&hay, b"aaZ"), naive_find(&hay, b"aaZ"));
        assert_eq!(find(&hay, b"aaaaaZ"), naive_find(&hay, b"aaaaaZ"));
        assert_eq!(find(&hay, b"zzz"), None);
    }

    #[test]
    fn substring_find_matches_naive_on_random_inputs() {
        // A small alphabet makes periodic and near-periodic needles common.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = |k: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % k
        };
        for _ in 0..3000 {
            let hay: Vec<u8> = (0..next(80) as usize)
                .map(|_| b"ab"[next(2) as usize])
                .collect();
            let m = next(12) as usize;
            let needle: Vec<u8> = (0..m).map(|_| b"abc"[next(3) as usize]).collect();
            assert_eq!(
                find(&hay, &needle),
                naive_find(&hay, &needle),
                "{needle:?} in {hay:?}"
            );
        }
    }

    #[test]
    fn case_insensitive_find_matches_naive() {
        let hay: Vec<u8> = (0..400u32)
            .map(|i| b"aAbB_-xY z\n"[(i * 5 % 11) as usize])
            .collect();
        for needle in [&b""[..], b"a", b"AB", b"ab_", b"xy z", b"ZZZ", b"Ab_-x"] {
            assert_eq!(
                find_ascii_case_insensitive(&hay, needle),
                naive_ci(&hay, needle),
                "{needle:?}"
            );
        }
    }

    #[test]
    fn case_insensitive_is_ascii_only() {
        // Non-ASCII bytes must match exactly, not fold.
        assert_eq!(
            find_ascii_case_insensitive("É".as_bytes(), "é".as_bytes()),
            None
        );
        assert_eq!(find_ascii_case_insensitive(b"Hello", b"HELLO!"), None);
        assert_eq!(find_ascii_case_insensitive(b"Hello", b"hello"), Some(0));
    }

    #[test]
    fn line_counts() {
        assert_eq!(count_lines(b""), 0);
        assert_eq!(count_lines(b"\n"), 1);
        assert_eq!(count_lines(b"a"), 1);
        assert_eq!(count_lines(b"a\n"), 1);
        assert_eq!(count_lines(b"a\nb\nc"), 3);
        assert_eq!(count_lines(b"\n\n\n"), 3);
    }
}
