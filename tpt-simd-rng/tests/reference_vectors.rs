//! Output checked against `rand_xoshiro`, an independent implementation of the same generators.
//!
//! The seeding is the same (SplitMix64 fills the four state words), so the streams must match
//! word for word, including the jumped streams used by the eight-lane generator.

use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::{SplitMix64 as RefSplitMix64, Xoshiro256PlusPlus};
use tpt_simd_rng::{Rng8, SplitMix64, Xoshiro256pp, Xoshiro256ppX8};

const SEEDS: [u64; 5] = [0, 1, 42, 0xDEAD_BEEF_CAFE_F00D, u64::MAX];

#[test]
fn splitmix64_matches_rand_xoshiro() {
    for seed in SEEDS {
        let mut ours = SplitMix64::new(seed);
        let mut theirs = RefSplitMix64::seed_from_u64(seed);
        for i in 0..64 {
            assert_eq!(ours.next_u64(), theirs.next_u64(), "seed {seed}, word {i}");
        }
    }
}

#[test]
fn xoshiro256pp_matches_rand_xoshiro() {
    for seed in SEEDS {
        let mut ours = Xoshiro256pp::from_seed(seed);
        let mut theirs = Xoshiro256PlusPlus::seed_from_u64(seed);
        for i in 0..256 {
            assert_eq!(ours.next_u64(), theirs.next_u64(), "seed {seed}, word {i}");
        }
    }
}

#[test]
fn jump_matches_rand_xoshiro() {
    for seed in SEEDS {
        let mut ours = Xoshiro256pp::from_seed(seed);
        let mut theirs = Xoshiro256PlusPlus::seed_from_u64(seed);
        ours.jump();
        theirs.jump();
        for i in 0..64 {
            assert_eq!(
                ours.next_u64(),
                theirs.next_u64(),
                "seed {seed}, word {i} after jump"
            );
        }
    }
}

#[test]
fn x8_lanes_are_jumped_streams_of_the_reference() {
    for seed in SEEDS {
        // Lane i is the reference stream jumped i times.
        let mut lanes: Vec<Xoshiro256PlusPlus> = Vec::with_capacity(8);
        let mut base = Xoshiro256PlusPlus::seed_from_u64(seed);
        for _ in 0..8 {
            lanes.push(base.clone());
            base.jump();
        }

        let mut x8 = Xoshiro256ppX8::from_seed(seed);
        for step in 0..16 {
            let words = x8.next_u64x8();
            for (lane, reference) in lanes.iter_mut().enumerate() {
                assert_eq!(
                    words[lane],
                    reference.next_u64(),
                    "seed {seed}, lane {lane}, step {step}"
                );
            }
        }
    }
}
