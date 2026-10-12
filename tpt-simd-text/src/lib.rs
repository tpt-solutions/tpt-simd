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
#[cfg(feature = "runtime-dispatch")]
extern crate std;

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
/// Inputs of 64 bytes or more use the AVX2 validator in this crate when AVX2 is available
/// (chosen at runtime under `runtime-dispatch`). Other inputs, and targets without AVX2, use
/// `core::str::from_utf8`. The `simdutf8` feature, if enabled, takes precedence. Every path
/// gives the same answer.
#[must_use]
pub fn is_valid_utf8(bytes: &[u8]) -> bool {
    #[cfg(feature = "simdutf8")]
    {
        simdutf8::basic::from_utf8(bytes).is_ok()
    }
    #[cfg(not(feature = "simdutf8"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if bytes.len() >= 64 && utf8_avx2::available() {
                return utf8_avx2::is_valid(bytes);
            }
        }
        core::str::from_utf8(bytes).is_ok()
    }
}

/// AVX2 UTF-8 validator. Each byte is checked against its predecessors: a continuation byte
/// must follow exactly the lead bytes that expect one, and the second byte of `E0`, `ED`,
/// `F0` and `F4` must lie in the range that excludes overlong, surrogate and too-large
/// sequences. The lead bytes `C0`, `C1` and `F5..=FF` are never valid.
///
/// Whole 32-byte blocks go through the vector check. The tail starts at a character
/// boundary, so `core::str::from_utf8` can validate it on its own.
#[cfg(all(target_arch = "x86_64", not(feature = "simdutf8")))]
#[allow(unsafe_code)]
mod utf8_avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_alignr_epi8, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_max_epu8, _mm256_min_epu8, _mm256_or_si256, _mm256_permute2x128_si256,
        _mm256_set1_epi8, _mm256_setzero_si256, _mm256_testz_si256, _mm256_xor_si256,
    };

    /// `true` if the AVX2 validator may be called on this CPU.
    pub(super) fn available() -> bool {
        #[cfg(feature = "runtime-dispatch")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(feature = "runtime-dispatch"))]
        {
            cfg!(target_feature = "avx2")
        }
    }

    /// Validates `bytes`. Requires `available()`.
    pub(super) fn is_valid(bytes: &[u8]) -> bool {
        let n = bytes.len();
        let n32 = n / 32 * 32;
        // SAFETY: the caller checked `available()`.
        if !unsafe { blocks_valid(&bytes[..n32]) } {
            return false;
        }
        // Start the tail at a character boundary. A character has at most three
        // continuation bytes, so a boundary exists within the last three bytes of the blocks.
        let mut start = n32;
        if start < n && is_continuation(bytes[start]) {
            match (n32.saturating_sub(3)..n32)
                .rev()
                .find(|&i| !is_continuation(bytes[i]))
            {
                Some(i) => start = i,
                None => return false,
            }
        }
        if start == n32 && n32 > 0 && pending_lead(&bytes[..n32]) {
            return false;
        }
        core::str::from_utf8(&bytes[start..]).is_ok()
    }

    fn is_continuation(b: u8) -> bool {
        b & 0xC0 == 0x80
    }

    /// `true` if the last lead byte in `prefix` still needs continuation bytes past its end.
    fn pending_lead(prefix: &[u8]) -> bool {
        for k in 1..=3.min(prefix.len()) {
            let need = match prefix[prefix.len() - k] {
                0xC2..=0xDF => 1,
                0xE0..=0xEF => 2,
                0xF0..=0xF4 => 3,
                _ => 0,
            };
            if need >= k {
                return true;
            }
        }
        false
    }

    /// `true` if every 32-byte block of `data` passes the vector check.
    #[target_feature(enable = "avx2")]
    unsafe fn blocks_valid(data: &[u8]) -> bool {
        let mut prev = _mm256_setzero_si256();
        let mut errors = _mm256_setzero_si256();
        for chunk in data.chunks_exact(32) {
            // SAFETY: `chunks_exact(32)` yields 32 readable bytes per chunk.
            let cur = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast::<__m256i>()) };
            errors = _mm256_or_si256(errors, block_errors(cur, prev));
            prev = cur;
        }
        _mm256_testz_si256(errors, errors) != 0
    }

    /// Lanes that are wrong in `cur`, given the previous block `prev` (zeros before the input).
    #[target_feature(enable = "avx2")]
    fn block_errors(cur: __m256i, prev: __m256i) -> __m256i {
        // `joined` holds the previous block's upper half and the current block's lower half.
        // `alignr` then yields the bytes one, two and three positions before each byte.
        let joined = _mm256_permute2x128_si256::<0x21>(prev, cur);
        let p1 = _mm256_alignr_epi8::<15>(cur, joined);
        let p2 = _mm256_alignr_epi8::<14>(cur, joined);
        let p3 = _mm256_alignr_epi8::<13>(cur, joined);

        // A continuation is needed at this byte if a lead byte one, two or three positions
        // back still expects one.
        // Two-byte leads expect one continuation, three-byte leads two, four-byte leads three.
        let needed = _mm256_or_si256(
            or3(
                in_range(p1, 0xC2, 0xDF),
                in_range(p1, 0xE0, 0xEF),
                in_range(p2, 0xE0, 0xEF),
            ),
            or3(
                in_range(p1, 0xF0, 0xF4),
                in_range(p2, 0xF0, 0xF4),
                in_range(p3, 0xF0, 0xF4),
            ),
        );
        let continuation = in_range(cur, 0x80, 0xBF);
        let mismatch = _mm256_xor_si256(needed, continuation);

        let bad_lead = _mm256_or_si256(in_range(cur, 0xC0, 0xC1), in_range(cur, 0xF5, 0xFF));
        let special = or3(
            _mm256_and_si256(eq(p1, 0xE0), in_range(cur, 0x80, 0x9F)),
            _mm256_and_si256(eq(p1, 0xED), in_range(cur, 0xA0, 0xBF)),
            or3(
                _mm256_and_si256(eq(p1, 0xF0), in_range(cur, 0x80, 0x8F)),
                _mm256_and_si256(eq(p1, 0xF4), in_range(cur, 0x90, 0xBF)),
                bad_lead,
            ),
        );
        _mm256_or_si256(mismatch, special)
    }

    /// Lanes whose unsigned value lies in `lo..=hi`.
    #[target_feature(enable = "avx2")]
    fn in_range(x: __m256i, lo: u8, hi: u8) -> __m256i {
        let clamped = _mm256_min_epu8(
            _mm256_max_epu8(x, _mm256_set1_epi8(lo as i8)),
            _mm256_set1_epi8(hi as i8),
        );
        _mm256_cmpeq_epi8(clamped, x)
    }

    #[target_feature(enable = "avx2")]
    fn eq(x: __m256i, value: u8) -> __m256i {
        _mm256_cmpeq_epi8(x, _mm256_set1_epi8(value as i8))
    }

    #[target_feature(enable = "avx2")]
    fn or3(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
        _mm256_or_si256(a, _mm256_or_si256(b, c))
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
    let needed = src.len() * 2;
    check_size(needed, dst.len())?;
    let done = hex_encode_blocks(src, &mut dst[..needed]);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for (pair, &b) in dst[done..needed].chunks_exact_mut(2).zip(&src[done / 2..]) {
        pair[0] = DIGITS[usize::from(b >> 4)];
        pair[1] = DIGITS[usize::from(b & 0x0F)];
    }
    Ok(needed)
}

/// Encodes the SIMD prefix of `src` into `dst` and returns the number of output bytes
/// written. AVX2 takes the 32-byte blocks when available; SSE2 takes the remaining 16-byte
/// blocks.
#[cfg(target_arch = "x86_64")]
fn hex_encode_blocks(src: &[u8], dst: &mut [u8]) -> usize {
    let wide = src.len() / 32 * 32;
    if wide > 0 && hex_avx2::available() {
        hex_avx2::encode(&src[..wide], &mut dst[..2 * wide]);
        return 2 * wide + hex_encode_sse2(&src[wide..], &mut dst[2 * wide..]);
    }
    hex_encode_sse2(src, dst)
}

/// Encodes whole 16-byte blocks of `src` into `dst` and returns the number of output
/// bytes written. SSE2 is part of the x86_64 baseline, so no detection is needed.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn hex_encode_sse2(src: &[u8], dst: &mut [u8]) -> usize {
    use core::arch::x86_64::{
        _mm_add_epi8, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128, _mm_set1_epi8,
        _mm_srli_epi16, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };
    let blocks = src.len() / 16;
    // SAFETY: each load reads 16 input bytes and each pair of stores writes 32 output bytes,
    // both inside the slices because `blocks * 16 <= src.len()` and `blocks * 32 <= dst.len()`.
    unsafe {
        let mask = _mm_set1_epi8(0x0F);
        let nine = _mm_set1_epi8(9);
        let zero_digit = _mm_set1_epi8(b'0' as i8);
        // For nibbles above 9, add the distance from '9' to 'a' in ASCII.
        let letter_gap = _mm_set1_epi8((b'a' - b'0' - 10) as i8);
        let to_ascii = |n: core::arch::x86_64::__m128i| {
            let is_letter = _mm_cmpgt_epi8(n, nine);
            _mm_add_epi8(
                _mm_add_epi8(n, zero_digit),
                _mm_and_si128(is_letter, letter_gap),
            )
        };
        for i in 0..blocks {
            let x = _mm_loadu_si128(src.as_ptr().add(i * 16).cast());
            let hi = to_ascii(_mm_and_si128(_mm_srli_epi16::<4>(x), mask));
            let lo = to_ascii(_mm_and_si128(x, mask));
            let out_lo = _mm_unpacklo_epi8(hi, lo);
            let out_hi = _mm_unpackhi_epi8(hi, lo);
            _mm_storeu_si128(dst.as_mut_ptr().add(i * 32).cast(), out_lo);
            _mm_storeu_si128(dst.as_mut_ptr().add(i * 32 + 16).cast(), out_hi);
        }
    }
    blocks * 32
}

/// Portable encoder: nothing is written here, so the caller encodes every byte.
#[cfg(not(target_arch = "x86_64"))]
fn hex_encode_blocks(_src: &[u8], _dst: &mut [u8]) -> usize {
    0
}

/// Maps each ASCII byte to its hex digit value (either case), or `0xFF` if it is not a digit.
const HEX_VALUE: [u8; 256] = {
    let mut table = [0xFFu8; 256];
    let mut i = 0;
    while i < 10 {
        table[b'0' as usize + i] = i as u8;
        i += 1;
    }
    let mut i = 0;
    while i < 6 {
        table[b'a' as usize + i] = 10 + i as u8;
        table[b'A' as usize + i] = 10 + i as u8;
        i += 1;
    }
    table
};

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
    // Validate every digit before writing, so an error leaves dst untouched. The SIMD
    // blocks are checked first, then the scalar tail.
    let head_len = simd_head_len(src.len());
    let (head, tail) = src.split_at(head_len);
    let tail_invalid = tail
        .iter()
        .fold(0u8, |acc, &c| acc | HEX_VALUE[usize::from(c)]);
    if !simd_head_valid(head) || tail_invalid & 0xF0 != 0 {
        return Err(TextError::InvalidHexDigit);
    }
    simd_head_decode(head, &mut dst[..head_len / 2]);
    for (out, pair) in dst[head_len / 2..needed]
        .iter_mut()
        .zip(tail.chunks_exact(2))
    {
        *out = (HEX_VALUE[usize::from(pair[0])] << 4) | HEX_VALUE[usize::from(pair[1])];
    }
    Ok(needed)
}

/// Length of the prefix of `len` input bytes that the SIMD decoder handles (whole 16-byte blocks).
#[cfg(target_arch = "x86_64")]
fn simd_head_len(len: usize) -> usize {
    len / 16 * 16
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_head_len(_len: usize) -> usize {
    0
}

/// `true` if every byte of `head` is a hex digit. `head` is a whole number of 16-byte blocks.
#[cfg(target_arch = "x86_64")]
fn simd_head_valid(head: &[u8]) -> bool {
    let wide = head.len() / 32 * 32;
    if wide > 0 && hex_avx2::available() {
        return hex_avx2::validate(&head[..wide]) && hex_sse2::validate(&head[wide..]);
    }
    hex_sse2::validate(head)
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_head_valid(_head: &[u8]) -> bool {
    true
}

/// Decodes `head` (validated, whole 16-byte blocks) into `dst` (`head.len() / 2` bytes).
#[cfg(target_arch = "x86_64")]
fn simd_head_decode(head: &[u8], dst: &mut [u8]) {
    let wide = head.len() / 32 * 32;
    if wide > 0 && hex_avx2::available() {
        hex_avx2::decode(&head[..wide], &mut dst[..wide / 2]);
        hex_sse2::decode(&head[wide..], &mut dst[wide / 2..]);
        return;
    }
    hex_sse2::decode(head, dst);
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_head_decode(_head: &[u8], _dst: &mut [u8]) {}

/// SSE2 hex decoder. SSE2 is part of the x86_64 baseline, so no detection is needed.
///
/// A byte is a digit when it lies in `0x30..=0x39`. A letter is a byte whose lowercase form
/// lies in `a..=f`. The value of a digit is `c & 0x0F`, and of a letter `(c & 0x0F) + 9`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod hex_sse2 {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi8, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128, _mm_movemask_epi8,
        _mm_or_si128, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_setzero_si128,
        _mm_slli_epi16, _mm_srli_epi16, _mm_storel_epi64,
    };

    /// Digit values of 16 bytes, and the mask of bytes that are hex digits (`0xFF` for valid).
    #[inline]
    fn values(x: __m128i) -> (__m128i, __m128i) {
        // SAFETY: SSE2 is part of the x86_64 baseline, so these intrinsics are always available.
        unsafe {
            let is_digit = _mm_and_si128(
                _mm_cmpgt_epi8(x, _mm_set1_epi8(0x2F)),
                _mm_cmpgt_epi8(_mm_set1_epi8(0x3A), x),
            );
            let lower = _mm_or_si128(x, _mm_set1_epi8(0x20));
            let is_letter = _mm_and_si128(
                _mm_cmpgt_epi8(lower, _mm_set1_epi8(0x60)),
                _mm_cmpgt_epi8(_mm_set1_epi8(0x67), lower),
            );
            let nibble = _mm_and_si128(x, _mm_set1_epi8(0x0F));
            let value = _mm_add_epi8(nibble, _mm_and_si128(is_letter, _mm_set1_epi8(9)));
            (value, _mm_or_si128(is_digit, is_letter))
        }
    }

    /// `true` if every byte of `src` (a whole number of 16-byte blocks) is a hex digit.
    pub(super) fn validate(src: &[u8]) -> bool {
        // SAFETY: each load reads the 16 bytes of one `chunks_exact(16)` chunk.
        // The masks are AND-ed across blocks, so there is one branch at the end, not one per block.
        unsafe {
            let all_valid = src.chunks_exact(16).fold(_mm_set1_epi8(-1), |ok, chunk| {
                let x = _mm_loadu_si128(chunk.as_ptr().cast());
                _mm_and_si128(ok, values(x).1)
            });
            _mm_movemask_epi8(all_valid) == 0xFFFF
        }
    }

    /// Decodes `src` (validated, a whole number of 16-byte blocks) into `dst`.
    pub(super) fn decode(src: &[u8], dst: &mut [u8]) {
        // SAFETY: each load reads 16 input bytes and each store writes 8 output bytes, both
        // inside the slices because `dst.len() == src.len() / 2`.
        unsafe {
            let low_byte = _mm_set1_epi16(0x00FF);
            for (i, chunk) in src.chunks_exact(16).enumerate() {
                let x = _mm_loadu_si128(chunk.as_ptr().cast());
                let (v, _) = values(x);
                // Byte 2k is the high nibble of output byte k, byte 2k+1 the low nibble.
                let high = _mm_slli_epi16(_mm_and_si128(v, low_byte), 4);
                let low = _mm_srli_epi16(v, 8);
                let packed = _mm_packus_epi16(_mm_or_si128(high, low), _mm_setzero_si128());
                _mm_storel_epi64(dst.as_mut_ptr().add(i * 8).cast(), packed);
            }
        }
    }
}

/// AVX2 hex kernels over 32-byte blocks, selected at runtime under `runtime-dispatch` (or at
/// compile time when the build enables AVX2). The callers pass whole 32-byte blocks.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod hex_avx2 {
    use core::arch::x86_64::{
        __m256i, _mm_storeu_si128, _mm256_add_epi8, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
        _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi8,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16,
        _mm256_srli_epi16, _mm256_storeu_si256, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
    };

    /// `true` if the AVX2 kernels may be called on this CPU.
    pub(super) fn available() -> bool {
        #[cfg(feature = "runtime-dispatch")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(feature = "runtime-dispatch"))]
        {
            cfg!(target_feature = "avx2")
        }
    }

    /// Encodes `src` (a whole number of 32-byte blocks) into `dst` (twice as long).
    /// Requires `available()`.
    pub(super) fn encode(src: &[u8], dst: &mut [u8]) {
        // SAFETY: the caller checked `available()`, and the slices are whole blocks.
        unsafe { encode_impl(src, dst) }
    }

    /// Validates `src` (a whole number of 32-byte blocks). Requires `available()`.
    pub(super) fn validate(src: &[u8]) -> bool {
        // SAFETY: the caller checked `available()`, and the slice is whole blocks.
        unsafe { validate_impl(src) }
    }

    /// Decodes `src` (validated, whole 32-byte blocks) into `dst` (half as long).
    /// Requires `available()`.
    pub(super) fn decode(src: &[u8], dst: &mut [u8]) {
        // SAFETY: the caller checked `available()`, and the slices are whole blocks.
        unsafe { decode_impl(src, dst) }
    }

    /// Digit values of 32 bytes, and the mask of bytes that are hex digits.
    #[target_feature(enable = "avx2")]
    unsafe fn values(x: __m256i) -> (__m256i, __m256i) {
        let is_digit = _mm256_and_si256(
            _mm256_cmpgt_epi8(x, _mm256_set1_epi8(0x2F)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x3A), x),
        );
        let lower = _mm256_or_si256(x, _mm256_set1_epi8(0x20));
        let is_letter = _mm256_and_si256(
            _mm256_cmpgt_epi8(lower, _mm256_set1_epi8(0x60)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x67), lower),
        );
        let nibble = _mm256_and_si256(x, _mm256_set1_epi8(0x0F));
        let value = _mm256_add_epi8(nibble, _mm256_and_si256(is_letter, _mm256_set1_epi8(9)));
        (value, _mm256_or_si256(is_digit, is_letter))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn encode_impl(src: &[u8], dst: &mut [u8]) {
        // SAFETY: each load reads 32 bytes inside `src`, and each store writes 32 bytes inside
        // `dst`, because `src` is whole 32-byte blocks and `dst` is twice its length.
        unsafe {
            // The 16 digits, twice, so each 128-bit lane has its own copy for `vpshufb`.
            let digits = *b"0123456789abcdef0123456789abcdef";
            let table = _mm256_loadu_si256(digits.as_ptr().cast());
            let mask = _mm256_set1_epi8(0x0F);
            for (i, chunk) in src.chunks_exact(32).enumerate() {
                let x = _mm256_loadu_si256(chunk.as_ptr().cast());
                let hi =
                    _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
                let lo = _mm256_shuffle_epi8(table, _mm256_and_si256(x, mask));
                // The unpacks interleave within each 128-bit lane, so the bytes come out in the
                // order (inputs 0..7, 16..23 | 8..15, 24..31). Two lane permutes restore the order.
                let a = _mm256_unpacklo_epi8(hi, lo);
                let b = _mm256_unpackhi_epi8(hi, lo);
                let first = _mm256_permute2x128_si256::<0x20>(a, b);
                let second = _mm256_permute2x128_si256::<0x31>(a, b);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i * 64).cast(), first);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i * 64 + 32).cast(), second);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn validate_impl(src: &[u8]) -> bool {
        let mut all_valid = _mm256_set1_epi8(-1);
        for chunk in src.chunks_exact(32) {
            // SAFETY: `chunks_exact(32)` yields 32 readable bytes per chunk.
            let x = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
            // SAFETY: this function is only reached with AVX2, which `values` requires.
            all_valid = _mm256_and_si256(all_valid, unsafe { values(x) }.1);
        }
        _mm256_movemask_epi8(all_valid) == -1
    }

    #[target_feature(enable = "avx2")]
    unsafe fn decode_impl(src: &[u8], dst: &mut [u8]) {
        let low_byte = _mm256_set1_epi16(0x00FF);
        for (i, chunk) in src.chunks_exact(32).enumerate() {
            // SAFETY: `chunks_exact(32)` yields 32 readable bytes per chunk; the 16-byte store
            // at `i * 16` is inside `dst`, which is half the length of `src`.
            let x = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
            // SAFETY: this function is only reached with AVX2, which `values` requires.
            let (v, _) = unsafe { values(x) };
            // Byte 2k is the high nibble of output byte k, byte 2k+1 the low nibble.
            let high = _mm256_slli_epi16::<4>(_mm256_and_si256(v, low_byte));
            let low = _mm256_srli_epi16::<8>(v);
            let packed = _mm256_packus_epi16(_mm256_or_si256(high, low), _mm256_setzero_si256());
            // Each lane's eight result bytes sit in its low qword (qwords 0 and 2). Gather
            // them into the low 128 bits, in order.
            let ordered = _mm256_permute4x64_epi64::<0x08>(packed);
            // SAFETY: the 16-byte store at `i * 16` is inside `dst`, which is half of `src`.
            unsafe {
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(i * 16).cast(),
                    _mm256_castsi256_si128(ordered),
                );
            }
        }
    }
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
