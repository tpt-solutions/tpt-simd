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
#[cfg(feature = "runtime-dispatch")]
extern crate std;

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

/// Advances the raw (non-inverted) CRC state over `data`, using the PCLMULQDQ
/// kernel when it is available.
fn crc_update(state: u32, data: &[u8]) -> u32 {
    #[cfg(all(
        target_arch = "x86_64",
        any(
            feature = "runtime-dispatch",
            all(
                target_feature = "pclmulqdq",
                target_feature = "sse2",
                target_feature = "sse4.1",
                target_feature = "ssse3"
            )
        )
    ))]
    {
        if pclmul::available() {
            return !pclmul::update(!state, data);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon::available() {
            return !neon::update(!state, data);
        }
    }
    crc_update_scalar(state, data)
}

#[cfg(target_arch = "aarch64")]
mod neon;

/// Portable slicing-by-8 update of the raw CRC state over `data`.
fn crc_update_scalar(mut state: u32, data: &[u8]) -> u32 {
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

/// PCLMULQDQ and VPCLMULQDQ folding CRC-32: fold by 4 over 128-bit lanes, an
/// eight-stream 256-bit kernel where VPCLMULQDQ is available, and a Barrett
/// reduction. This follows Gopal et al., "Fast CRC computation for generic
/// polynomials using PCLMULQDQ instruction" (Intel, 2009); the fold constants are
/// the reflected IEEE constants from that paper.
#[cfg(all(
    target_arch = "x86_64",
    any(
        feature = "runtime-dispatch",
        all(
            target_feature = "pclmulqdq",
            target_feature = "sse2",
            target_feature = "sse4.1",
            target_feature = "ssse3"
        )
    )
))]
#[allow(unsafe_code)]
mod pclmul {
    use super::crc_update_scalar;
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_and_si128, _mm_clmulepi64_si128, _mm_cvtsi32_si128,
        _mm_extract_epi32, _mm_loadu_si128, _mm_set_epi32, _mm_set_epi64x, _mm_srli_si128,
        _mm_xor_si128, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_clmulepi64_epi128,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set_epi64x, _mm256_xor_si256,
    };

    const K1: i64 = 0x1_5444_2bd4;
    const K2: i64 = 0x1_c6e4_1596;
    const K3: i64 = 0x1_7519_97d0;
    const K4: i64 = 0x0_ccaa_009e;
    const K5: i64 = 0x1_63cd_6124;
    const P_X: i64 = 0x1_DB71_0641;
    const U_PRIME: i64 = 0x1_F701_1641;

    /// Inputs of 16..128 bytes use a single-accumulator fold; the fold-by-4 loop needs 128.
    const MIN_FOLD_BYTES: usize = 16;
    const MIN_FOLD_BY_4_BYTES: usize = 128;
    /// Below this the eight-stream setup and collapse cost more than they save.
    const MIN_WIDE_BYTES: usize = 256;
    /// Fold constants for the 1024-bit stride (eight streams).
    const K_1024_LOW: i64 = 0x1_e88e_f372;
    const K_1024_HIGH: i64 = 0x1_4a7f_e880;

    /// `true` if the 256-bit VPCLMULQDQ kernel may be called on this CPU.
    fn wide_available() -> bool {
        #[cfg(feature = "runtime-dispatch")]
        {
            !cfg!(miri)
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("vpclmulqdq")
        }
        #[cfg(not(feature = "runtime-dispatch"))]
        {
            cfg!(all(target_feature = "avx2", target_feature = "vpclmulqdq"))
        }
    }

    /// `true` if the folding kernels may be called on this CPU.
    pub(super) fn available() -> bool {
        #[cfg(all(
            target_feature = "pclmulqdq",
            target_feature = "sse2",
            target_feature = "sse4.1",
            target_feature = "ssse3"
        ))]
        {
            true
        }
        #[cfg(not(all(
            target_feature = "pclmulqdq",
            target_feature = "sse2",
            target_feature = "sse4.1",
            target_feature = "ssse3"
        )))]
        {
            !cfg!(miri)
                && std::is_x86_feature_detected!("pclmulqdq")
                && std::is_x86_feature_detected!("sse2")
                && std::is_x86_feature_detected!("sse4.1")
                && std::is_x86_feature_detected!("ssse3")
        }
    }

    /// Advances the finalised CRC `crc` over `data`.
    pub(super) fn update(crc: u32, data: &[u8]) -> u32 {
        if data.len() < MIN_FOLD_BYTES || !available() {
            return !crc_update_scalar(!crc, data);
        }
        if data.len() >= MIN_WIDE_BYTES && wide_available() {
            // SAFETY: `wide_available()` confirmed AVX2 and VPCLMULQDQ, and `available()` the rest.
            return unsafe { calculate_wide(crc, data) };
        }
        // SAFETY: `available()` confirmed the CPU supports every feature `calculate` enables.
        unsafe { calculate(crc, data) }
    }

    /// 256-bit VPCLMULQDQ kernel: eight streams (four YMM registers, two lanes each), 128 bytes
    /// per iteration. The streams are collapsed into one 128-bit state, and the remainder goes
    /// through `update`, which is valid because CRC updates compose.
    #[target_feature(enable = "avx,avx2,vpclmulqdq,pclmulqdq,sse2,sse4.1,ssse3")]
    unsafe fn calculate_wide(crc: u32, mut data: &[u8]) -> u32 {
        // SAFETY: the caller checked CPU support and `data.len() >= MIN_WIDE_BYTES`, so the
        // four initial 32-byte loads are in range; each later load is guarded by a length check.
        unsafe {
            let mut v0 = get256(&mut data);
            let mut v1 = get256(&mut data);
            let mut v2 = get256(&mut data);
            let mut v3 = get256(&mut data);
            v0 = _mm256_xor_si256(v0, _mm256_castsi128_si256(_mm_cvtsi32_si128(!crc as i32)));

            let k = _mm256_set_epi64x(K_1024_HIGH, K_1024_LOW, K_1024_HIGH, K_1024_LOW);
            while data.len() >= 128 {
                v0 = reduce256(v0, get256(&mut data), k);
                v1 = reduce256(v1, get256(&mut data), k);
                v2 = reduce256(v2, get256(&mut data), k);
                v3 = reduce256(v3, get256(&mut data), k);
            }

            // Collapse the eight streams in increasing byte offset.
            let k3k4 = _mm_set_epi64x(K4, K3);
            let mut x = _mm256_castsi256_si128(v0);
            x = reduce128(x, _mm256_extracti128_si256::<1>(v0), k3k4);
            x = reduce128(x, _mm256_castsi256_si128(v1), k3k4);
            x = reduce128(x, _mm256_extracti128_si256::<1>(v1), k3k4);
            x = reduce128(x, _mm256_castsi256_si128(v2), k3k4);
            x = reduce128(x, _mm256_extracti128_si256::<1>(v2), k3k4);
            x = reduce128(x, _mm256_castsi256_si128(v3), k3k4);
            x = reduce128(x, _mm256_extracti128_si256::<1>(v3), k3k4);

            update(reduce_to_crc(x), data)
        }
    }

    /// Single-accumulator fold for 16..128 bytes.
    #[target_feature(enable = "pclmulqdq,sse2,sse4.1,ssse3")]
    unsafe fn calculate_short(crc: u32, mut data: &[u8]) -> u32 {
        // SAFETY: the caller checked CPU support and `data.len() >= 16`.
        unsafe {
            let k3k4 = _mm_set_epi64x(K4, K3);
            let mut x = get(&mut data);
            x = _mm_xor_si128(x, _mm_cvtsi32_si128(!crc as i32));
            while data.len() >= 16 {
                x = reduce128(x, get(&mut data), k3k4);
            }
            let crc = reduce_to_crc(x);
            !crc_update_scalar(!crc, data)
        }
    }

    #[target_feature(enable = "pclmulqdq,sse2,sse4.1,ssse3")]
    unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
        if data.len() < MIN_FOLD_BY_4_BYTES {
            // SAFETY: as for `calculate_short`.
            return unsafe { calculate_short(crc, data) };
        }
        // SAFETY: the caller checked CPU support; `data` has at least 128 bytes, so the
        // four initial loads are in range, and every later load is guarded by a length check.
        unsafe {
            let mut x3 = get(&mut data);
            let mut x2 = get(&mut data);
            let mut x1 = get(&mut data);
            let mut x0 = get(&mut data);
            x3 = _mm_xor_si128(x3, _mm_cvtsi32_si128(!crc as i32));

            let k1k2 = _mm_set_epi64x(K2, K1);
            while data.len() >= 64 {
                x3 = reduce128(x3, get(&mut data), k1k2);
                x2 = reduce128(x2, get(&mut data), k1k2);
                x1 = reduce128(x1, get(&mut data), k1k2);
                x0 = reduce128(x0, get(&mut data), k1k2);
            }

            let k3k4 = _mm_set_epi64x(K4, K3);
            let mut x = reduce128(x3, x2, k3k4);
            x = reduce128(x, x1, k3k4);
            x = reduce128(x, x0, k3k4);
            while data.len() >= 16 {
                x = reduce128(x, get(&mut data), k3k4);
            }

            // Continue the scalar kernel over the final 0..=15 bytes from the reduced state.
            let crc = reduce_to_crc(x);
            !crc_update_scalar(!crc, data)
        }
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq,sse2")]
    unsafe fn reduce128(a: __m128i, b: __m128i, keys: __m128i) -> __m128i {
        let t1 = _mm_clmulepi64_si128(a, keys, 0x00);
        let t2 = _mm_clmulepi64_si128(a, keys, 0x11);
        _mm_xor_si128(_mm_xor_si128(b, t1), t2)
    }

    #[inline]
    #[target_feature(enable = "avx,avx2,vpclmulqdq")]
    unsafe fn reduce256(a: __m256i, b: __m256i, keys: __m256i) -> __m256i {
        let t1 = _mm256_clmulepi64_epi128(a, keys, 0x00);
        let t2 = _mm256_clmulepi64_epi128(a, keys, 0x11);
        _mm256_xor_si256(_mm256_xor_si256(b, t1), t2)
    }

    /// Loads 16 bytes and advances `a` past them. The caller guarantees `a.len() >= 16`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn get(a: &mut &[u8]) -> __m128i {
        // SAFETY: the caller guarantees 16 readable bytes, and `loadu` has no alignment requirement.
        let r = unsafe { _mm_loadu_si128(a.as_ptr().cast::<__m128i>()) };
        *a = &a[16..];
        r
    }

    /// Loads 32 bytes and advances `a` past them. The caller guarantees `a.len() >= 32`.
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn get256(a: &mut &[u8]) -> __m256i {
        // SAFETY: the caller guarantees 32 readable bytes, and `loadu` has no alignment requirement.
        let r = unsafe { _mm256_loadu_si256(a.as_ptr().cast::<__m256i>()) };
        *a = &a[32..];
        r
    }

    /// Reduces the 128-bit folded state to a finalised 32-bit CRC.
    #[target_feature(enable = "pclmulqdq,sse2,sse4.1,ssse3")]
    unsafe fn reduce_to_crc(x: __m128i) -> u32 {
        let k3k4 = _mm_set_epi64x(K4, K3);
        let low32 = _mm_set_epi32(0, 0, 0, !0);

        let x = _mm_xor_si128(_mm_clmulepi64_si128(x, k3k4, 0x10), _mm_srli_si128(x, 8));
        let x = _mm_xor_si128(
            _mm_clmulepi64_si128(_mm_and_si128(x, low32), _mm_set_epi64x(0, K5), 0x00),
            _mm_srli_si128(x, 4),
        );

        // Barrett reduction for the bit-reflected variant.
        let pu = _mm_set_epi64x(U_PRIME, P_X);
        let t1 = _mm_clmulepi64_si128(_mm_and_si128(x, low32), pu, 0x10);
        let t2 = _mm_clmulepi64_si128(_mm_and_si128(t1, low32), pu, 0x00);
        !(_mm_extract_epi32(_mm_xor_si128(x, t2), 1) as u32)
    }
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
            // The modulo is taken once per NMAX block, not once per byte.
            (self.a, self.b) = adler_block(self.a, self.b, block);
        }
    }

    /// Returns the checksum of all bytes fed so far.
    #[must_use]
    pub fn finish(&self) -> u32 {
        (self.b << 16) | self.a
    }
}

/// Adler-32 update of `(a, b)` over one block of at most [`ADLER_NMAX`] bytes.
/// Returns the reduced values.
#[cfg(target_arch = "x86_64")]
fn adler_block(a: u32, b: u32, block: &[u8]) -> (u32, u32) {
    let whole = block.len() / 32 * 32;
    if whole > 0 && adler_avx2::available() {
        let (a, b) = adler_avx2::block(a, b, &block[..whole]);
        return adler_sse2::block(a, b, &block[whole..]);
    }
    adler_sse2::block(a, b, block)
}

/// AVX2 Adler-32 kernel over 32-byte blocks, with the same bookkeeping as [`adler_sse2`].
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod adler_avx2 {
    use super::ADLER_MOD;
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_sad_epu8, _mm256_set1_epi16, _mm256_setr_epi8,
        _mm256_setzero_si256, _mm256_storeu_si256,
    };

    /// `true` if the AVX2 kernel may be called on this CPU.
    pub(super) fn available() -> bool {
        #[cfg(feature = "runtime-dispatch")]
        {
            !cfg!(miri) && std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(feature = "runtime-dispatch"))]
        {
            cfg!(target_feature = "avx2")
        }
    }

    /// Adler-32 update of `(a, b)` over `data`, whose length is a non-zero multiple of 32.
    /// Requires `available()`.
    pub(super) fn block(a: u32, b: u32, data: &[u8]) -> (u32, u32) {
        // SAFETY: the caller checked `available()`, and `data` is a whole number of 32-byte blocks.
        unsafe { block_impl(a, b, data) }
    }

    /// Byte `i` of a 32-byte block has weight `32 - i`; the two 16-byte halves use the
    /// weight vectors below.
    #[target_feature(enable = "avx2")]
    unsafe fn block_impl(a: u32, b: u32, data: &[u8]) -> (u32, u32) {
        let (mut a, mut b) = (u64::from(a), u64::from(b));
        let zero = _mm256_setzero_si256();
        // Byte `i` of a block has weight `32 - i`. `pmaddubsw` multiplies unsigned bytes by these
        // signed weights and sums adjacent pairs, so each pair sum is at most 255 * 63 and
        // cannot saturate i16. `pmaddwd` with ones then sums the pairs into i32 lanes.
        let weights = _mm256_setr_epi8(
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
            10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        );
        let ones = _mm256_set1_epi16(1);
        let mut bytes = zero;
        let mut running = zero;
        let mut weighted = zero;
        for chunk in data.chunks_exact(32) {
            // SAFETY: each load reads the 32 bytes of one `chunks_exact(32)` chunk.
            let x = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast::<__m256i>()) };
            running = _mm256_add_epi64(running, bytes);
            bytes = _mm256_add_epi64(bytes, _mm256_sad_epu8(x, zero));
            let pairs = _mm256_maddubs_epi16(x, weights);
            weighted = _mm256_add_epi32(weighted, _mm256_madd_epi16(pairs, ones));
        }

        let mut b64 = [0u64; 4];
        let mut r64 = [0u64; 4];
        let mut w32 = [0i32; 8];
        // SAFETY: each destination is exactly the size of the vector being stored.
        unsafe {
            _mm256_storeu_si256(b64.as_mut_ptr().cast::<__m256i>(), bytes);
            _mm256_storeu_si256(r64.as_mut_ptr().cast::<__m256i>(), running);
            _mm256_storeu_si256(w32.as_mut_ptr().cast::<__m256i>(), weighted);
        }
        let sum_bytes: u64 = b64.iter().sum();
        let sum_running: u64 = r64.iter().sum();
        let sum_weighted: u64 = w32.iter().map(|&v| v as u64).sum();
        let blocks = (data.len() / 32) as u64;
        b += 32 * blocks * a + 32 * sum_running + sum_weighted;
        a += sum_bytes;
        (
            (a % u64::from(ADLER_MOD)) as u32,
            (b % u64::from(ADLER_MOD)) as u32,
        )
    }
}

/// Portable Adler-32 update of `(a, b)` over one block of at most [`ADLER_NMAX`] bytes.
#[cfg(not(target_arch = "x86_64"))]
fn adler_block(a: u32, b: u32, block: &[u8]) -> (u32, u32) {
    let (mut a, mut b) = (a, b);
    // Accumulators stay below 2^32 for a whole NMAX block.
    for &byte in block {
        a += u32::from(byte);
        b += a;
    }
    (a % ADLER_MOD, b % ADLER_MOD)
}

/// SSE2 Adler-32 kernel. SSE2 is part of the x86_64 baseline, so no runtime detection is needed.
///
/// For a run of 16-byte blocks with running sum `a` and weighted sum `b`, each block adds
/// `16 * a_before + sum_i (16 - i) * byte_i` to `b`. The running sums are kept in vector
/// registers, so only one horizontal reduction is needed per call.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod adler_sse2 {
    use super::ADLER_MOD;
    use core::arch::x86_64::{
        _mm_add_epi32, _mm_add_epi64, _mm_loadu_si128, _mm_madd_epi16, _mm_sad_epu8, _mm_set_epi16,
        _mm_setzero_si128, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    /// Adler-32 update of `(a, b)` over `data` (at most one NMAX block). Returns reduced values.
    pub(super) fn block(a: u32, b: u32, data: &[u8]) -> (u32, u32) {
        let full = data.len() / 16 * 16;
        let (mut a, mut b) = (u64::from(a), u64::from(b));
        if full > 0 {
            let (sum_bytes, sum_running, weighted) = sums(&data[..full]);
            let blocks = (full / 16) as u64;
            b += 16 * blocks * a + 16 * sum_running + weighted;
            a += sum_bytes;
        }
        for &byte in &data[full..] {
            a += u64::from(byte);
            b += a;
        }
        (
            (a % u64::from(ADLER_MOD)) as u32,
            (b % u64::from(ADLER_MOD)) as u32,
        )
    }

    /// For the 16-byte blocks of `data` (a multiple of 16 bytes, non-empty), returns
    /// `(total byte sum, sum over blocks of the byte sum before each block, weighted sum)`,
    /// where byte `i` of a block has weight `16 - i`.
    fn sums(data: &[u8]) -> (u64, u64, u64) {
        // SAFETY: every load reads the 16 bytes of one `chunks_exact(16)` chunk, which lie
        // inside `data`.
        unsafe {
            let zero = _mm_setzero_si128();
            // Lane 0 holds weight 16 (the first byte of a block), lane 7 holds 9.
            let w_lo = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16);
            let w_hi = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
            let mut bytes = zero;
            let mut running = zero;
            let mut weighted = zero;
            for chunk in data.chunks_exact(16) {
                let x = _mm_loadu_si128(chunk.as_ptr().cast());
                running = _mm_add_epi64(running, bytes);
                bytes = _mm_add_epi64(bytes, _mm_sad_epu8(x, zero));
                let lo = _mm_unpacklo_epi8(x, zero);
                let hi = _mm_unpackhi_epi8(x, zero);
                let w = _mm_add_epi32(_mm_madd_epi16(lo, w_lo), _mm_madd_epi16(hi, w_hi));
                weighted = _mm_add_epi32(weighted, w);
            }
            let mut b64 = [0u64; 2];
            let mut r64 = [0u64; 2];
            let mut w32 = [0i32; 4];
            _mm_storeu_si128(b64.as_mut_ptr().cast(), bytes);
            _mm_storeu_si128(r64.as_mut_ptr().cast(), running);
            _mm_storeu_si128(w32.as_mut_ptr().cast(), weighted);
            (
                b64[0] + b64[1],
                r64[0] + r64[1],
                w32.iter().map(|&v| v as u64).sum(),
            )
        }
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
    fn crc32_dispatch_matches_scalar_kernel() {
        // Covers the scalar tail, the single-accumulator fold, fold-by-4, the 512-bit kernel
        // (from 2048 bytes) and the remainders after each.
        let data = sample(6000, 7);
        for len in (0..2300).chain([2304, 2560, 3000, 4096, 4352, 5000, 6000]) {
            let want = !crc_update_scalar(!0, &data[..len]);
            assert_eq!(crc32(&data[..len]), want, "len {len}");
        }
        for split in [1usize, 15, 16, 63, 64, 129, 500, 2100, 4000] {
            let mut c = Crc32::new();
            c.update(&data[..split]);
            c.update(&data[split..]);
            assert_eq!(c.finish(), crc32(&data));
        }
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
    fn adler32_matches_scalar_reference() {
        fn reference(data: &[u8]) -> u32 {
            let (mut a, mut b) = (1u32, 0u32);
            for &x in data {
                a = (a + u32::from(x)) % ADLER_MOD;
                b = (b + a) % ADLER_MOD;
            }
            (b << 16) | a
        }
        // Covers the 16-byte blocks, the byte tail, and inputs past one NMAX block.
        let data = sample(12_010, 3);
        for len in (0..300).chain([1000, 5551, 5552, 5553, 11_111, 12_000]) {
            for offset in 0..3 {
                let slice = &data[offset..offset + len];
                assert_eq!(
                    adler32(slice),
                    reference(slice),
                    "len {len} offset {offset}"
                );
            }
        }
        let ff = std::vec![0xFFu8; 40_000];
        assert_eq!(adler32(&ff), reference(&ff));
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
