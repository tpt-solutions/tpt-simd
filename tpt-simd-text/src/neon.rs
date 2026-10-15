//! aarch64 NEON UTF-8 validation and hex encoding.
//!
//! The validator uses the same lookup tables and error flags as the AVX2 path (`utf8_tables`),
//! and the same 32-byte blocks and tail check, so both accept exactly the same inputs. `vextq_u8`
//! supplies the one, two and three bytes before each byte, and `vqtbl1q_u8` is the byte lookup
//! (`pshufb` with nibble indices, which never set the high bit).

#![allow(unsafe_code)]

use crate::utf8_tables::{BYTE_1_HIGH, BYTE_1_LOW, BYTE_2_HIGH, TWO_CONTS, tail_valid};
use core::arch::aarch64::*;

/// The three lookup tables as vectors.
#[derive(Clone, Copy)]
struct Tables {
    byte_1_high: uint8x16_t,
    byte_1_low: uint8x16_t,
    byte_2_high: uint8x16_t,
}

/// Error bits of one 16-byte block, given the previous block (zeros before the input).
#[inline(always)]
fn block_errors(cur: uint8x16_t, prev: uint8x16_t, t: Tables) -> uint8x16_t {
    // SAFETY: NEON is part of the aarch64 baseline; all operands are register values.
    unsafe {
        // The bytes one, two and three positions before each byte of `cur`.
        let p1 = vextq_u8::<15>(prev, cur);
        let p2 = vextq_u8::<14>(prev, cur);
        let p3 = vextq_u8::<13>(prev, cur);

        let nibble = vdupq_n_u8(0x0F);
        let p1_high = vandq_u8(vshrq_n_u8::<4>(p1), nibble);
        let p1_low = vandq_u8(p1, nibble);
        let cur_high = vandq_u8(vshrq_n_u8::<4>(cur), nibble);
        let special = vandq_u8(
            vandq_u8(
                vqtbl1q_u8(t.byte_1_high, p1_high),
                vqtbl1q_u8(t.byte_1_low, p1_low),
            ),
            vqtbl1q_u8(t.byte_2_high, cur_high),
        );

        // A 3- or 4-byte lead two bytes back, or a 4-byte lead three bytes back, needs a
        // continuation here. Those leads are the bytes >= 0xE0 and >= 0xF0.
        let third = vceqq_u8(vmaxq_u8(p2, vdupq_n_u8(0xE0)), p2);
        let fourth = vceqq_u8(vmaxq_u8(p3, vdupq_n_u8(0xF0)), p3);
        let must = vandq_u8(vorrq_u8(third, fourth), vdupq_n_u8(TWO_CONTS));
        veorq_u8(special, must)
    }
}

/// `true` if every 32-byte block of `data` passes the vector check.
fn blocks_valid(data: &[u8]) -> bool {
    // SAFETY: NEON is part of the aarch64 baseline. Each chunk holds 32 bytes, and each 16-byte
    // load stays inside its chunk.
    unsafe {
        let tables = Tables {
            byte_1_high: vld1q_u8(BYTE_1_HIGH.as_ptr()),
            byte_1_low: vld1q_u8(BYTE_1_LOW.as_ptr()),
            byte_2_high: vld1q_u8(BYTE_2_HIGH.as_ptr()),
        };
        let mut prev = vdupq_n_u8(0);
        let mut errors = vdupq_n_u8(0);
        for chunk in data.chunks_exact(32) {
            for half in [0usize, 16] {
                let cur = vld1q_u8(chunk.as_ptr().add(half));
                errors = vorrq_u8(errors, block_errors(cur, prev, tables));
                prev = cur;
            }
        }
        vmaxvq_u8(errors) == 0
    }
}

/// Validates `bytes` as UTF-8.
pub(crate) fn is_valid(bytes: &[u8]) -> bool {
    let n32 = bytes.len() / 32 * 32;
    if !blocks_valid(&bytes[..n32]) {
        return false;
    }
    tail_valid(bytes, n32)
}

/// Encodes whole 16-byte blocks of `src` into `dst` (two output bytes per input byte) and returns
/// the number of output bytes written. Matches `hex_encode_sse2`.
pub(crate) fn hex_encode_blocks(src: &[u8], dst: &mut [u8]) -> usize {
    let blocks = src.len() / 16;
    // SAFETY: each load reads 16 input bytes and each pair of stores writes 32 output bytes, both
    // inside the slices because `blocks * 16 <= src.len()` and `blocks * 32 <= dst.len()`.
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let nine = vdupq_n_u8(9);
        let zero_digit = vdupq_n_u8(b'0');
        // For nibbles above 9, add the distance from '9' to 'a' in ASCII.
        let letter_gap = vdupq_n_u8(b'a' - b'0' - 10);
        let to_ascii = |n: uint8x16_t| {
            let is_letter = vcgtq_u8(n, nine);
            vaddq_u8(vaddq_u8(n, zero_digit), vandq_u8(is_letter, letter_gap))
        };
        for i in 0..blocks {
            let x = vld1q_u8(src.as_ptr().add(i * 16));
            let hi = to_ascii(vandq_u8(vshrq_n_u8::<4>(x), mask));
            let lo = to_ascii(vandq_u8(x, mask));
            vst1q_u8(dst.as_mut_ptr().add(i * 32), vzip1q_u8(hi, lo));
            vst1q_u8(dst.as_mut_ptr().add(i * 32 + 16), vzip2q_u8(hi, lo));
        }
    }
    blocks * 32
}
