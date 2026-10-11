//! Text kernels.
//!
//! | function | notes |
//! |---|---|
//! | [`is_valid_utf8`] | `core` validation by default; `simdutf8` with the `simdutf8` feature |
//! | [`hex_encode_lower`], [`hex_decode`] | own implementation, no allocation |
//! | [`utf8_to_utf16`], [`utf16_to_utf8`] | own implementation, surrogate checks, exact sizes |
//!
//! There are no runtime dependencies by default. `simdutf8` is optional, and
//! `simdutf8` and `faster-hex` are used only as dev-dependency oracles in tests.
//!
//! Every function that writes into a caller buffer validates its input and size
//! before writing, so an error leaves the buffer untouched.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_text::{utf8_to_utf16, utf16_to_utf8};
//!
//! let mut units = [0u16; 4];
//! let n = utf8_to_utf16("h€😀", &mut units).unwrap();
//! assert_eq!(&units[..n], &[0x68, 0x20AC, 0xD83D, 0xDE00]);
//!
//! let mut bytes = [0u8; 8];
//! let m = utf16_to_utf8(&units[..n], &mut bytes).unwrap();
//! assert_eq!(core::str::from_utf8(&bytes[..m]).unwrap(), "h€😀");
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Errors from the functions in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextError {
    /// The output buffer cannot hold the result.
    BufferTooSmall {
        /// Elements the output needs.
        needed: usize,
        /// Elements the buffer has.
        got: usize,
    },
    /// UTF-16 contains a lone surrogate. `index` is the code unit where it starts.
    UnpairedSurrogate {
        /// Index into the UTF-16 input.
        index: usize,
    },
    /// Hex input contains a byte that is not a hex digit.
    InvalidHexDigit,
    /// Hex input has an odd number of digits.
    OddHexLength,
}

impl core::fmt::Display for TextError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::BufferTooSmall { needed, got } => {
                write!(f, "output buffer holds {got} elements, needs {needed}")
            }
            Self::UnpairedSurrogate { index } => {
                write!(f, "unpaired UTF-16 surrogate at code unit {index}")
            }
            Self::InvalidHexDigit => f.write_str("invalid hex digit"),
            Self::OddHexLength => f.write_str("hex input has an odd length"),
        }
    }
}

/// Returns `true` if `bytes` is valid UTF-8.
///
/// Uses `core::str::from_utf8` unless the `simdutf8` feature is enabled, in
/// which case the `simdutf8` SIMD validator is used. Both give the same answer.
#[must_use]
pub fn is_valid_utf8(bytes: &[u8]) -> bool {
    #[cfg(feature = "simdutf8")]
    {
        simdutf8::basic::from_utf8(bytes).is_ok()
    }
    #[cfg(not(feature = "simdutf8"))]
    {
        core::str::from_utf8(bytes).is_ok()
    }
}

fn check_size(needed: usize, got: usize) -> Result<(), TextError> {
    if needed > got {
        Err(TextError::BufferTooSmall { needed, got })
    } else {
        Ok(())
    }
}

/// Number of UTF-16 code units needed to encode `s`.
#[must_use]
pub fn utf16_len(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// Encodes `src` as UTF-16 into `dst`. Returns the number of code units written.
///
/// # Errors
///
/// Returns [`TextError::BufferTooSmall`] if `dst` is shorter than
/// [`utf16_len`]`(src)`; `dst` is not modified in that case.
pub fn utf8_to_utf16(src: &str, dst: &mut [u16]) -> Result<usize, TextError> {
    let needed = utf16_len(src);
    check_size(needed, dst.len())?;
    let mut pos = 0;
    for c in src.chars() {
        pos += c.encode_utf16(&mut dst[pos..]).len();
    }
    Ok(pos)
}

/// Number of UTF-8 bytes needed to encode the UTF-16 input `src`.
///
/// # Errors
///
/// Returns [`TextError::UnpairedSurrogate`] if `src` contains a lone surrogate.
pub fn utf8_len_of_utf16(src: &[u16]) -> Result<usize, TextError> {
    let mut bytes = 0;
    let mut index = 0;
    for r in char::decode_utf16(src.iter().copied()) {
        match r {
            Ok(c) => {
                bytes += c.len_utf8();
                index += c.len_utf16();
            }
            Err(_) => return Err(TextError::UnpairedSurrogate { index }),
        }
    }
    Ok(bytes)
}

/// Decodes UTF-16 `src` into UTF-8 in `dst`. Returns the number of bytes written.
///
/// # Errors
///
/// Returns [`TextError::UnpairedSurrogate`] for a lone surrogate, or
/// [`TextError::BufferTooSmall`] if `dst` is too short. `dst` is not modified
/// on error.
pub fn utf16_to_utf8(src: &[u16], dst: &mut [u8]) -> Result<usize, TextError> {
    let needed = utf8_len_of_utf16(src)?;
    check_size(needed, dst.len())?;
    let mut pos = 0;
    for r in char::decode_utf16(src.iter().copied()) {
        // Validated by utf8_len_of_utf16, so every item is Ok.
        let c = r.unwrap_or(char::REPLACEMENT_CHARACTER);
        pos += c.encode_utf8(&mut dst[pos..]).len();
    }
    Ok(pos)
}

/// Encodes `src` as lowercase hex into `dst`, which must hold `2 * src.len()`
/// bytes. Returns the number of bytes written.
///
/// # Errors
///
/// Returns [`TextError::BufferTooSmall`] if `dst` is too short.
pub fn hex_encode_lower(src: &[u8], dst: &mut [u8]) -> Result<usize, TextError> {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let needed = src.len() * 2;
    check_size(needed, dst.len())?;
    for (pair, &b) in dst[..needed].chunks_exact_mut(2).zip(src) {
        pair[0] = DIGITS[usize::from(b >> 4)];
        pair[1] = DIGITS[usize::from(b & 0x0F)];
    }
    Ok(needed)
}

/// Value of one hex digit, either case.
#[inline]
fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Decodes hex digits (either case) from `src` into `dst`. `src` must have an
/// even length and `dst` must hold `src.len() / 2` bytes. Returns the number of
/// bytes written.
///
/// # Errors
///
/// Returns [`TextError::OddHexLength`], [`TextError::InvalidHexDigit`], or
/// [`TextError::BufferTooSmall`].
pub fn hex_decode(src: &[u8], dst: &mut [u8]) -> Result<usize, TextError> {
    if !src.len().is_multiple_of(2) {
        return Err(TextError::OddHexLength);
    }
    let needed = src.len() / 2;
    check_size(needed, dst.len())?;
    // Validate every digit before writing, so an error leaves dst untouched.
    if src.iter().any(|&c| hex_nibble(c).is_none()) {
        return Err(TextError::InvalidHexDigit);
    }
    for (out, pair) in dst[..needed].iter_mut().zip(src.chunks_exact(2)) {
        // Digits were checked above, so both lookups succeed.
        let hi = hex_nibble(pair[0]).unwrap_or(0);
        let lo = hex_nibble(pair[1]).unwrap_or(0);
        *out = (hi << 4) | lo;
    }
    Ok(needed)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::string::String;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn utf8_validation_basic() {
        assert!(is_valid_utf8("héllo 😀".as_bytes()));
        assert!(is_valid_utf8(b""));
        assert!(!is_valid_utf8(&[0xC3]));
        assert!(!is_valid_utf8(&[0xED, 0xA0, 0x80])); // encoded surrogate
    }

    #[test]
    fn utf8_to_utf16_matches_std() {
        for s in ["", "a", "ß€", "😀x", "mixed ✓ 😀 end", &"é".repeat(100)] {
            let want: Vec<u16> = s.encode_utf16().collect();
            let mut got = vec![0u16; want.len()];
            let n = utf8_to_utf16(s, &mut got).unwrap();
            assert_eq!(&got[..n], &want[..], "{s:?}");
        }
    }

    #[test]
    fn utf16_round_trip_and_std_agreement() {
        let s = "a€😀 b\u{10FFFF}";
        let units: Vec<u16> = s.encode_utf16().collect();
        let mut bytes = vec![0u8; s.len()];
        let n = utf16_to_utf8(&units, &mut bytes).unwrap();
        assert_eq!(&bytes[..n], s.as_bytes());
        let mut back = vec![0u16; units.len()];
        let m = utf8_to_utf16(core::str::from_utf8(&bytes[..n]).unwrap(), &mut back).unwrap();
        assert_eq!(&back[..m], &units[..]);
        let mut again = String::new();
        again.push_str(s);
        assert_eq!(again, s);
    }

    #[test]
    fn utf16_lone_surrogates_are_reported_with_index() {
        assert_eq!(
            utf8_len_of_utf16(&[0x61, 0xD83D]),
            Err(TextError::UnpairedSurrogate { index: 1 })
        );
        assert_eq!(
            utf8_len_of_utf16(&[0xDE00, 0x61]),
            Err(TextError::UnpairedSurrogate { index: 0 })
        );
        let mut out = [0xAAu8; 8];
        assert!(utf16_to_utf8(&[0x61, 0xD83D, 0x62], &mut out).is_err());
        assert_eq!(out, [0xAA; 8], "nothing written on error");
    }

    #[test]
    fn small_buffers_are_rejected_without_partial_writes() {
        let mut units = [0xFFFFu16; 1];
        assert_eq!(
            utf8_to_utf16("😀", &mut units),
            Err(TextError::BufferTooSmall { needed: 2, got: 1 })
        );
        assert_eq!(units, [0xFFFF]);
        let mut bytes = [0u8; 3];
        assert_eq!(
            utf16_to_utf8(&[0x20AC, 0x20AC], &mut bytes),
            Err(TextError::BufferTooSmall { needed: 6, got: 3 })
        );
    }

    #[test]
    fn hex_round_trip_and_errors() {
        let data = [0x00u8, 0x7f, 0xab, 0xff];
        let mut enc = [0u8; 8];
        let n = hex_encode_lower(&data, &mut enc).unwrap();
        assert_eq!(&enc[..n], b"007fabff");
        let mut dec = [0u8; 4];
        assert_eq!(hex_decode(&enc[..n], &mut dec), Ok(4));
        assert_eq!(dec, data);
        assert_eq!(hex_decode(b"ABcd", &mut dec[..2]), Ok(2));
        assert_eq!(hex_decode(b"abc", &mut dec), Err(TextError::OddHexLength));
        assert_eq!(hex_decode(b"zz", &mut dec), Err(TextError::InvalidHexDigit));
        assert_eq!(
            hex_encode_lower(&data, &mut [0u8; 7]),
            Err(TextError::BufferTooSmall { needed: 8, got: 7 })
        );
    }
}
