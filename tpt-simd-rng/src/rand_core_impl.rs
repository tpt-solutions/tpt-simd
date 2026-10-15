//! `rand_core` 0.9 traits for the scalar xoshiro256++ generator (feature `rand_core`).
//!
//! Seeding follows the `rand_xoshiro` convention, so a generator built here and one built
//! there from the same seed produce the same stream: `seed_from_u64` runs SplitMix64, `next_u32`
//! returns the upper 32 bits, and an all-zero seed falls back to `seed_from_u64(0)`.

use crate::Xoshiro256pp;
use rand_core::{RngCore, SeedableRng, impls};

impl RngCore for Xoshiro256pp {
    fn next_u32(&mut self) -> u32 {
        // The low bits of xoshiro256++ have linear dependencies, so take the upper half.
        (Xoshiro256pp::next_u64(self) >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        Xoshiro256pp::next_u64(self)
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        impls::fill_bytes_via_next(self, dst)
    }
}

impl SeedableRng for Xoshiro256pp {
    type Seed = [u8; 32];

    fn from_seed(seed: [u8; 32]) -> Self {
        let mut state = [0u64; 4];
        for (word, bytes) in state.iter_mut().zip(seed.chunks_exact(8)) {
            let mut b = [0u8; 8];
            b.copy_from_slice(bytes);
            *word = u64::from_le_bytes(b);
        }
        if state == [0; 4] {
            return Self::from_seed(0u64);
        }
        Self::from_state(state)
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self::from_seed(seed)
    }
}
