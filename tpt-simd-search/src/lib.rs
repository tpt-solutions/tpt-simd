//! Byte search and line counting.
//!
//! | function | source | notes |
//! |---|---|---|
//! | [`find_byte`], [`rfind_byte`], [`count_byte`] | wraps `memchr` | SIMD byte search |
//! | [`find`] | wraps `memchr::memmem` | substring search |
//! | [`find_ascii_case_insensitive`] | built here | ASCII-only case folding |
//! | [`count_lines`] | built here | counts `\n`, plus an unterminated last line |
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

/// Index of the first `needle` byte in `haystack`.
#[must_use]
pub fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    memchr::memchr(needle, haystack)
}

/// Index of the last `needle` byte in `haystack`.
#[must_use]
pub fn rfind_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    memchr::memrchr(needle, haystack)
}

/// Number of occurrences of `needle` in `haystack`.
#[must_use]
pub fn count_byte(haystack: &[u8], needle: u8) -> usize {
    memchr::memchr_iter(needle, haystack).count()
}

/// Index of the first occurrence of `needle` in `haystack`. An empty needle
/// matches at 0.
#[must_use]
pub fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memchr::memmem::find(haystack, needle)
}

/// Index of the first occurrence of `needle` in `haystack`, comparing ASCII
/// letters without regard to case. Other bytes must match exactly. An empty
/// needle matches at 0.
///
/// The first needle byte is located with a two-way byte search for its upper
/// and lower case forms, so most non-matching positions are skipped quickly.
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
        let found = memchr::memchr2(lower, upper, &haystack[pos..=last_start])?;
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
        (0..=h.len().saturating_sub(n.len()))
            .find(|&i| h.len() >= n.len() && &h[i..i + n.len()] == n)
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
