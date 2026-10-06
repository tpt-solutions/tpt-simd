//! SplitMix64, scalar xoshiro256++ and the 8-lane [`Xoshiro256ppX8`].

use crate::Rng8;
use tpt_simd_core::Simd;

/// SplitMix64 (Vigna), used for seeding. Any `u64` seed is valid (including
/// 0); every output is a bijective mix of the counter, so equal seeds give
/// equal sequences and distinct seeds never collide on the first output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator whose first output is derived from `seed`.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const JUMP: [u64; 4] = [
    0x180e_c6d3_3cfd_0aba,
    0xd5a6_1266_f0c9_392c,
    0xa958_2618_e03f_c9aa,
    0x39ab_dc45_29b1_661c,
];
const LONG_JUMP: [u64; 4] = [
    0x76e1_5d3e_fefd_cbbf,
    0xc500_4e44_1c52_2fb3,
    0x7771_0069_854e_e241,
    0x3910_9bb0_2acb_e635,
];

/// Scalar xoshiro256++ (Blackman and Vigna). Period `2^256 - 1`.
///
/// This is the reference single stream; [`Xoshiro256ppX8`] runs eight of
/// them in lock step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Builds a generator from an explicit state.
    ///
    /// # Panics
    /// If the state is all zero (the one invalid state).
    pub fn from_state(s: [u64; 4]) -> Self {
        assert!(s != [0; 4], "xoshiro256++ state must not be all zero");
        Self { s }
    }

    /// Seeds from a `u64` by drawing four words from [`SplitMix64`] (the
    /// authors' recommended procedure).
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        if s == [0; 4] {
            s[0] = 1; // unreachable in practice; keeps the state valid
        }
        Self { s }
    }

    /// The current state words.
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    fn apply_jump(&mut self, poly: &[u64; 4]) {
        let mut acc = [0u64; 4];
        for &word in poly {
            for b in 0..64 {
                if (word >> b) & 1 != 0 {
                    for k in 0..4 {
                        acc[k] ^= self.s[k];
                    }
                }
                self.next_u64();
            }
        }
        self.s = acc;
    }

    /// Advances the stream by exactly `2^128` steps. Calling it repeatedly
    /// yields up to `2^128` non-overlapping subsequences of length `2^128`.
    pub fn jump(&mut self) {
        self.apply_jump(&JUMP);
    }

    /// Advances the stream by exactly `2^192` steps (`2^64` non-overlapping
    /// groups of `2^64` [`jump`](Self::jump) streams).
    pub fn long_jump(&mut self) {
        self.apply_jump(&LONG_JUMP);
    }
}

/// Eight xoshiro256++ streams advanced in lock step (structure of arrays, so
/// each state word update is one vector operation).
///
/// ## Streams
///
/// A generator is the *block* `b` of a seed: lane `i` of block `b` is the
/// scalar stream obtained by seeding [`Xoshiro256pp::from_seed`] with the
/// seed and calling [`Xoshiro256pp::jump`] `8*b + i` times. Streams are
/// therefore `2^128` steps apart (they overlap only after `2^128` outputs
/// per lane) and distinct `(seed, block, lane)` triples with the same seed
/// never overlap. Different seeds are unrelated random starting points on
/// the same `2^256` cycle (overlap probability negligible). Use
/// [`from_seed_block`](Self::from_seed_block) to hand block `b` to thread or
/// worker `b`, or [`jump_blocks`](Self::jump_blocks) to advance in place.
///
/// Output order of [`Rng8`] fills is lane-interleaved: element `k` comes
/// from lane `k % 8` at step `k / 8`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro256ppX8 {
    s: [[u64; 8]; 4],
}

impl Xoshiro256ppX8 {
    /// Lanes (independent streams) per generator.
    pub const LANES: usize = 8;

    /// Block 0 of `seed`: lanes are streams `0..8` of the seed.
    pub fn from_seed(seed: u64) -> Self {
        Self::from_seed_block(seed, 0)
    }

    /// Block `block` of `seed`: lanes are streams `8*block .. 8*block + 8`.
    /// Costs `8*block + 8` jumps (each a few hundred steps).
    pub fn from_seed_block(seed: u64, block: u64) -> Self {
        let mut base = Xoshiro256pp::from_seed(seed);
        for _ in 0..block.wrapping_mul(8) {
            base.jump();
        }
        let mut lanes: [Xoshiro256pp; 8] = core::array::from_fn(|_| base.clone());
        for i in 1..8 {
            lanes[i] = lanes[i - 1].clone();
            lanes[i].jump();
        }
        Self::from_lanes(&lanes)
    }

    /// Builds a generator from eight scalar streams (lane `i` = `lanes[i]`).
    pub fn from_lanes(lanes: &[Xoshiro256pp; 8]) -> Self {
        let mut s = [[0u64; 8]; 4];
        for (i, l) in lanes.iter().enumerate() {
            for (k, row) in s.iter_mut().enumerate() {
                row[i] = l.s[k];
            }
        }
        Self { s }
    }

    /// Extracts lane `i` as a scalar generator at its current position.
    ///
    /// # Panics
    /// If `i >= 8`.
    pub fn lane(&self, i: usize) -> Xoshiro256pp {
        Xoshiro256pp {
            s: [self.s[0][i], self.s[1][i], self.s[2][i], self.s[3][i]],
        }
    }

    /// Moves every lane forward by `8 * n` jumps (`n` whole blocks of
    /// `2^128`-step streams), i.e. turns block `b` into block `b + n`.
    pub fn jump_blocks(&mut self, n: u64) {
        let mut lanes: [Xoshiro256pp; 8] = core::array::from_fn(|i| self.lane(i));
        for l in &mut lanes {
            for _ in 0..n.wrapping_mul(8) {
                l.jump();
            }
        }
        *self = Self::from_lanes(&lanes);
    }

    /// One step of all eight lanes.
    #[inline(always)]
    pub fn step(&mut self) -> [u64; 8] {
        let s = &mut self.s;
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = s[0][i].wrapping_add(s[3][i]).rotate_left(23).wrapping_add(s[0][i]);
            let t = s[1][i] << 17;
            s[2][i] ^= s[0][i];
            s[3][i] ^= s[1][i];
            s[1][i] ^= s[2][i];
            s[0][i] ^= s[3][i];
            s[2][i] ^= t;
            s[3][i] = s[3][i].rotate_left(45);
        }
        out
    }

    /// One step as a [`Simd<u64, 8>`].
    #[inline]
    pub fn next_simd(&mut self) -> Simd<u64, 8> {
        Simd::from_array(self.step())
    }
}

impl Rng8 for Xoshiro256ppX8 {
    #[inline(always)]
    fn next_u64x8(&mut self) -> [u64; 8] {
        self.step()
    }
}
