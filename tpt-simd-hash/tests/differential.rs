//! Differential tests: our CRC-32 and Adler-32 against the `crc32fast` and
//! `simd-adler32` crates, which are dev-dependencies only.

use tpt_simd_hash::{Adler32, Crc32, adler32, crc32};

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
fn crc32_matches_crc32fast_over_lengths_and_alignments() {
    for n in (0..300).chain([1023, 1024, 4096, 65_537]) {
        let data = sample(n, n as u32);
        assert_eq!(crc32(&data), crc32fast::hash(&data), "n={n}");
        // Misaligned sub-slice exercises the unaligned slicing path.
        if n > 3 {
            assert_eq!(crc32(&data[3..]), crc32fast::hash(&data[3..]), "tail n={n}");
        }
    }
}

#[test]
fn crc32_streaming_matches_one_shot_and_crc32fast() {
    let data = sample(10_007, 42);
    for step in [1usize, 3, 7, 8, 9, 64, 1000] {
        let mut ours = Crc32::new();
        let mut theirs = crc32fast::Hasher::new();
        for chunk in data.chunks(step) {
            ours.update(chunk);
            theirs.update(chunk);
        }
        assert_eq!(ours.finish(), theirs.clone().finalize(), "step={step}");
        assert_eq!(ours.finish(), crc32(&data), "step={step}");
    }
}

#[test]
fn adler32_matches_simd_adler32_over_lengths() {
    for n in (0..300).chain([5551, 5552, 5553, 11_104, 100_000]) {
        let data = sample(n, n as u32 + 7);
        assert_eq!(adler32(&data), simd_adler(&data), "n={n}");
    }
}

#[test]
fn adler32_streaming_matches_simd_adler32() {
    let data = sample(50_000, 3);
    for step in [1usize, 5, 4096, 5552, 6000] {
        let mut ours = Adler32::new();
        let mut theirs = simd_adler32::Adler32::new();
        for chunk in data.chunks(step) {
            ours.update(chunk);
            theirs.write(chunk);
        }
        assert_eq!(ours.finish(), theirs.finish(), "step={step}");
    }
}

#[test]
fn extreme_inputs_match() {
    for byte in [0x00u8, 0x7F, 0x80, 0xFF] {
        let data = vec![byte; 40_000];
        assert_eq!(crc32(&data), crc32fast::hash(&data), "byte={byte:#x}");
        assert_eq!(adler32(&data), simd_adler(&data), "byte={byte:#x}");
    }
}

fn simd_adler(data: &[u8]) -> u32 {
    let mut h = simd_adler32::Adler32::new();
    h.write(data);
    h.finish()
}
