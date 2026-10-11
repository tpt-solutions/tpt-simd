//! Byte search and line counting.
//!
//! | function | notes |
//! |---|---|
//! | [`find_byte`], [`rfind_byte`], [`count_byte`] | own word-at-a-time (SWAR) code; `memchr` with the `memchr` feature |
//! | [`find`] | own substring search with a first-byte scan; `memchr::memmem` with the `memchr` feature |
//! | [`find_ascii_case_insensitive`] | own, ASCII-only case folding |
//! | [`count_lines`] | counts `\n`, plus an unterminated last line |
//!
//! There are no runtime dependencies by default. `memchr` is an optional
//! feature, and it is also a dev-dependency used as a differential-test oracle.
//!
//! The own substring search is worst-case O(n * m). Use the `memchr` feature
//! if adversarial needles are a concern.
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

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

/// Index of the first `needle` byte in `haystack`.
#[must_use]
pub fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(feature = "memchr")]
    {
        memchr::memchr(needle, haystack)
    }
    #[cfg(not(feature = "memchr"))]
    {
        find_byte_swar(haystack, needle)
    }
}

/// Word-at-a-time search: a 64-bit word contains a zero byte exactly when
/// `(x - LO) & !x & HI` is non-zero, where `x` is the word XOR the needle
/// broadcast. The byte position is then found with a short scan.
#[cfg_attr(feature = "memchr", allow(dead_code))]
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
    #[cfg(not(feature = "memchr"))]
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
        let Some((&first, rest)) = needle.split_first() else {
            return Some(0);
        };
        if haystack.len() < needle.len() {
            return None;
        }
        let last_start = haystack.len() - needle.len();
        let mut pos = 0;
        while pos <= last_start {
            let at = pos + find_byte_swar(&haystack[pos..=last_start], first)?;
            if &haystack[at + 1..at + needle.len()] == rest {
                return Some(at);
            }
            pos = at + 1;
        }
        None
    }
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
