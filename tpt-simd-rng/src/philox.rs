//! Philox4x32-10 counter-based generator (Salmon et al., "Parallel random
//! numbers: as easy as 1, 2, 3"), eight blocks per step.

use crate::Rng8;

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// One Philox4x32-10 block: encrypts `counter` under `key`.
///
/// This is the scalar reference; [`Philox4x32X8`] computes eight of them at
/// once.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for _ in 0..10 {
        let p0 = u64::from(M0) * u64::from(c[0]);
        let p1 = u64::from(M1) * u64::from(c[2]);
        c = [
            ((p1 >> 32) as u32) ^ c[1] ^ k[0],
            p1 as u32,
            ((p0 >> 32) as u32) ^ c[3] ^ k[1],
            p0 as u32,
        ];
        k[0] = k[0].wrapping_add(W0);
        k[1] = k[1].wrapping_add(W1);
    }
    c
}

/// Eight Philox4x32-10 blocks per step, with a 64-bit block counter and a
/// 64-bit stream id.
///
/// Block `n` of stream `s` under key `k` is
/// `philox4x32_10([n as u32, (n >> 32) as u32, s as u32, (s >> 32) as u32], k)`.
/// Different `(key, stream)` pairs never overlap (`2^64` blocks each), so a
/// stream id is the natural way to hand independent sequences to workers,
/// and [`seek`](Self::seek) gives random access.
///
/// [`Rng8::next_u64x8`] consumes eight consecutive blocks; lane `b` of the
/// first call returns `words[0] | words[1] << 32` of block `b`, the second
/// call returns `words[2] | words[3] << 32`, then the counter advances by 8.
/// (So fills use all 128 output bits of every block.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Philox4x32X8 {
    key: [u32; 2],
    stream: u64,
    counter: u64,
    pending: Option<[u64; 8]>,
}

impl Philox4x32X8 {
    /// Stream 0, counter 0; the 64-bit `seed` is the key.
    pub fn from_seed(seed: u64) -> Self {
        Self::new(seed, 0)
    }

    /// Stream `stream` of key `seed`, counter 0.
    pub fn new(seed: u64, stream: u64) -> Self {
        Self {
            key: [seed as u32, (seed >> 32) as u32],
            stream,
            counter: 0,
            pending: None,
        }
    }

    /// Positions the generator at block `block` (a multiple of 8 is not
    /// required; the 8-block step starts there). Clears buffered output.
    pub fn seek(&mut self, block: u64) {
        self.counter = block;
        self.pending = None;
    }

    /// Computes blocks `counter .. counter + 8` and advances the counter.
    /// Returns the four output words as `[word][block]`.
    #[inline(always)]
    pub fn blocks(&mut self) -> [[u32; 8]; 4] {
        let mut c0: [u32; 8] = core::array::from_fn(|i| self.counter.wrapping_add(i as u64) as u32);
        let mut c1: [u32; 8] =
            core::array::from_fn(|i| (self.counter.wrapping_add(i as u64) >> 32) as u32);
        let mut c2 = [self.stream as u32; 8];
        let mut c3 = [(self.stream >> 32) as u32; 8];
        let (mut k0, mut k1) = (self.key[0], self.key[1]);
        for _ in 0..10 {
            let mut n0 = [0u32; 8];
            let mut n1 = [0u32; 8];
            let mut n2 = [0u32; 8];
            let mut n3 = [0u32; 8];
            for i in 0..8 {
                let p0 = u64::from(M0) * u64::from(c0[i]);
                let p1 = u64::from(M1) * u64::from(c2[i]);
                n0[i] = ((p1 >> 32) as u32) ^ c1[i] ^ k0;
                n1[i] = p1 as u32;
                n2[i] = ((p0 >> 32) as u32) ^ c3[i] ^ k1;
                n3[i] = p0 as u32;
            }
            (c0, c1, c2, c3) = (n0, n1, n2, n3);
            k0 = k0.wrapping_add(W0);
            k1 = k1.wrapping_add(W1);
        }
        self.counter = self.counter.wrapping_add(8);
        [c0, c1, c2, c3]
    }
}

impl Rng8 for Philox4x32X8 {
    #[inline(always)]
    fn next_u64x8(&mut self) -> [u64; 8] {
        if let Some(p) = self.pending.take() {
            return p;
        }
        let w = self.blocks();
        let lo = core::array::from_fn(|i| u64::from(w[0][i]) | (u64::from(w[1][i]) << 32));
        self.pending = Some(core::array::from_fn(|i| {
            u64::from(w[2][i]) | (u64::from(w[3][i]) << 32)
        }));
        lo
    }
}
