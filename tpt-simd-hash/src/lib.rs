//! Checksums and hashes.
//!
//! | function / type | notes |
//! |---|---|
//! | [`crc32`], [`Crc32`] | IEEE CRC-32, slicing-by-8, streaming |
//! | [`adler32`], [`Adler32`] | Adler-32 with deferred modulo, streaming |
//! | [`fnv1a_64`] | FNV-1a, 64-bit |
//! | [`Rolling`] | polynomial rolling hash over a fixed window |
//!
//! Everything is implemented here with no runtime dependencies. The `crc32fast`
//! and `simd-adler32` crates are used only as dev-dependencies, as differential
//! test oracles. Nothing here uses `unsafe`.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_hash::{crc32, fnv1a_64};
//!
//! assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
//! assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Reflected IEEE 802.3 polynomial.
const CRC_POLY: u32 = 0xEDB8_8320;

/// Slicing-by-8 tables: `TABLES[k][i]` is the CRC of byte `i` followed by `k`
/// zero bytes. Built at compile time.
const TABLES: [[u32; 256]; 8] = build_crc_tables();

const fn build_crc_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            c = if c & 1 != 0 {
                CRC_POLY ^ (c >> 1)
            } else {
                c >> 1
            };
            bit += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut k = 1;
    while k < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[k - 1][i];
            t[k][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            i += 1;
        }
        k += 1;
    }
    t
}

/// Advances the raw (non-inverted) CRC state over `data`.
fn crc_update(mut state: u32, data: &[u8]) -> u32 {
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let lo = state ^ u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        state = TABLES[7][(lo & 0xFF) as usize]
            ^ TABLES[6][((lo >> 8) & 0xFF) as usize]
            ^ TABLES[5][((lo >> 16) & 0xFF) as usize]
            ^ TABLES[4][(lo >> 24) as usize]
            ^ TABLES[3][c[4] as usize]
            ^ TABLES[2][c[5] as usize]
            ^ TABLES[1][c[6] as usize]
            ^ TABLES[0][c[7] as usize];
    }
    for &b in chunks.remainder() {
        state = TABLES[0][((state ^ u32::from(b)) & 0xFF) as usize] ^ (state >> 8);
    }
    state
}

/// IEEE CRC-32 of `data`.
///
/// # Examples
///
/// ```
/// assert_eq!(tpt_simd_hash::crc32(b"123456789"), 0xCBF4_3926);
/// ```
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    !crc_update(!0, data)
}

/// Streaming IEEE CRC-32.
#[derive(Clone, Copy, Debug)]
pub struct Crc32 {
    /// Raw state, before the final inversion.
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Starts a new checksum.
    #[must_use]
    pub fn new() -> Self {
        Self { state: !0 }
    }

    /// Feeds more bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.state = crc_update(self.state, data);
    }

    /// Returns the checksum of all bytes fed so far. Does not consume the state.
    #[must_use]
    pub fn finish(&self) -> u32 {
        !self.state
    }
}

/// Modulus for Adler-32.
const ADLER_MOD: u32 = 65_521;
/// Largest block that can be summed before the 32-bit accumulators need a
/// modulo (zlib's NMAX).
const ADLER_NMAX: usize = 5552;

/// Adler-32 of `data`.
///
/// # Examples
///
/// ```
/// assert_eq!(tpt_simd_hash::adler32(b"Wikipedia"), 0x11E6_0398);
/// ```
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    let mut a = Adler32::new();
    a.update(data);
    a.finish()
}

/// Streaming Adler-32.
#[derive(Clone, Copy, Debug)]
pub struct Adler32 {
    a: u32,
    b: u32,
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Adler32 {
    /// Starts a new checksum.
    #[must_use]
    pub fn new() -> Self {
        Self { a: 1, b: 0 }
    }

    /// Feeds more bytes.
    pub fn update(&mut self, data: &[u8]) {
        for block in data.chunks(ADLER_NMAX) {
            // Accumulators stay below 2^32 for a whole block, so the modulo
            // is taken once per block instead of once per byte.
            for &byte in block {
                self.a += u32::from(byte);
                self.b += self.a;
            }
            self.a %= ADLER_MOD;
            self.b %= ADLER_MOD;
        }
    }

    /// Returns the checksum of all bytes fed so far.
    #[must_use]
    pub fn finish(&self) -> u32 {
        (self.b << 16) | self.a
    }
}

const FNV_OFFSET_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a, 64-bit. Fast for short keys; not a checksum and not collision
/// resistant.
#[must_use]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_64;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

/// Polynomial rolling hash over a fixed window of bytes, modulo 2^64.
///
/// Holds the hash of the last `window` bytes. Feed the next byte with
/// [`Rolling::roll`], which also takes the byte that leaves the window. Until
/// the window is full, call [`Rolling::push`] instead.
///
/// Use an odd `base` (for example a large prime): an even base loses high
/// bits of the input.
#[derive(Clone, Copy, Debug)]
pub struct Rolling {
    base: u64,
    /// `base^(window - 1)`, the weight of the byte leaving the window.
    lead: u64,
    hash: u64,
}

impl Rolling {
    /// Creates an empty rolling hash for windows of `window` bytes.
    /// Panics if `window` is zero.
    #[must_use]
    pub fn new(window: usize, base: u64) -> Self {
        assert!(window != 0, "Rolling::new: window must be non-zero");
        let mut lead = 1u64;
        for _ in 1..window {
            lead = lead.wrapping_mul(base);
        }
        Self {
            base,
            lead,
            hash: 0,
        }
    }

    /// Appends `byte` while the window is still filling.
    pub fn push(&mut self, byte: u8) {
        self.hash = self
            .hash
            .wrapping_mul(self.base)
            .wrapping_add(u64::from(byte));
    }

    /// Slides the window: removes `out` (the oldest byte) and appends `inb`.
    pub fn roll(&mut self, out: u8, inb: u8) {
        let removed = u64::from(out).wrapping_mul(self.lead);
        self.hash = self
            .hash
            .wrapping_sub(removed)
            .wrapping_mul(self.base)
            .wrapping_add(u64::from(inb));
    }

    /// Current hash.
    #[must_use]
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn sample(n: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect()
    }

    #[test]
    fn crc32_check_value_and_streaming() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        let mut c = Crc32::new();
        c.update(b"1234");
        c.update(b"56789");
        assert_eq!(c.finish(), 0xCBF4_3926);
    }

    #[test]
    fn crc32_matches_bitwise_reference_for_all_tail_lengths() {
        fn reference(data: &[u8]) -> u32 {
            let mut c = !0u32;
            for &b in data {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = if c & 1 != 0 {
                        CRC_POLY ^ (c >> 1)
                    } else {
                        c >> 1
                    };
                }
            }
            !c
        }
        for n in 0..70 {
            let d = sample(n, n as u32);
            assert_eq!(crc32(&d), reference(&d), "n={n}");
        }
    }

    #[test]
    fn adler32_known_value_and_streaming() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
        let data = sample(10_000, 9);
        let whole = adler32(&data);
        let mut a = Adler32::new();
        for chunk in data.chunks(777) {
            a.update(chunk);
        }
        assert_eq!(a.finish(), whole);
    }

    #[test]
    fn adler32_all_ff_crosses_the_deferred_modulo() {
        // Long runs of 0xFF push the accumulators past NMAX, exercising the
        // per-block reduction.
        let data = [0xFFu8; 20_000];
        let mut a = 1u32;
        let mut b = 0u32;
        for &x in &data {
            a = (a + u32::from(x)) % ADLER_MOD;
            b = (b + a) % ADLER_MOD;
        }
        assert_eq!(adler32(&data), (b << 16) | a);
    }

    #[test]
    fn fnv1a_known_values() {
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn rolling_matches_direct_window_hash() {
        const B: u64 = 1_000_003;
        let data: Vec<u8> = (0..200u32).map(|i| (i * 17 % 256) as u8).collect();
        let w = 8;
        let direct = |s: &[u8]| {
            s.iter()
                .fold(0u64, |h, &b| h.wrapping_mul(B).wrapping_add(u64::from(b)))
        };
        let mut r = Rolling::new(w, B);
        for &b in &data[..w] {
            r.push(b);
        }
        assert_eq!(r.hash(), direct(&data[..w]));
        for i in w..data.len() {
            r.roll(data[i - w], data[i]);
            assert_eq!(r.hash(), direct(&data[i + 1 - w..=i]), "i={i}");
        }
    }

    #[test]
    #[should_panic(expected = "window must be non-zero")]
    fn rolling_zero_window_panics() {
        let _ = Rolling::new(0, 3);
    }
}
