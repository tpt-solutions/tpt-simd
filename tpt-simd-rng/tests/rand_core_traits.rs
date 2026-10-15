//! The `rand_core` impls on `Xoshiro256pp` against `rand_xoshiro`, which implements the same traits.
#![cfg(feature = "rand_core")]

use rand_core::{RngCore, SeedableRng};
use tpt_simd_rng::Xoshiro256pp;

const SEEDS: [u64; 4] = [0, 7, 0xDEAD_BEEF, u64::MAX];

#[test]
fn seed_from_u64_matches_rand_xoshiro() {
    for seed in SEEDS {
        let mut ours = <Xoshiro256pp as SeedableRng>::seed_from_u64(seed);
        let mut theirs = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        for _ in 0..64 {
            assert_eq!(ours.next_u64(), theirs.next_u64());
        }
    }
}

#[test]
fn next_u32_is_the_upper_half_like_rand_xoshiro() {
    for seed in SEEDS {
        let mut ours = <Xoshiro256pp as SeedableRng>::seed_from_u64(seed);
        let mut theirs = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        for _ in 0..64 {
            assert_eq!(ours.next_u32(), theirs.next_u32());
        }
    }
}

#[test]
fn from_seed_bytes_match_rand_xoshiro() {
    for seed in SEEDS {
        let mut bytes = [0u8; 32];
        for (i, chunk) in bytes.chunks_exact_mut(8).enumerate() {
            chunk.copy_from_slice(&seed.wrapping_add(i as u64 * 0x1234_5678).to_le_bytes());
        }
        let mut ours = <Xoshiro256pp as SeedableRng>::from_seed(bytes);
        let mut theirs = rand_xoshiro::Xoshiro256PlusPlus::from_seed(bytes);
        for _ in 0..64 {
            assert_eq!(ours.next_u64(), theirs.next_u64());
        }
    }
}

#[test]
fn all_zero_seed_falls_back_like_rand_xoshiro() {
    let mut ours = <Xoshiro256pp as SeedableRng>::from_seed([0u8; 32]);
    let mut theirs = rand_xoshiro::Xoshiro256PlusPlus::from_seed([0u8; 32]);
    for _ in 0..16 {
        assert_eq!(ours.next_u64(), theirs.next_u64());
    }
}

#[test]
fn fill_bytes_is_little_endian_words_then_tail() {
    let mut ours = <Xoshiro256pp as SeedableRng>::seed_from_u64(3);
    let mut theirs = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let mut a = [0u8; 21];
    let mut b = [0u8; 21];
    ours.fill_bytes(&mut a);
    theirs.fill_bytes(&mut b);
    assert_eq!(a, b);
}
