//! Property tests: checksums and hashes against bit-by-bit or byte-by-byte references.

use proptest::prelude::*;
use tpt_simd_hash::{Adler32, Crc32, adler32, crc32, fnv1a_64};

/// CRC-32 (IEEE, reflected), one bit at a time.
fn crc32_reference(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32_reference(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

fn fnv1a_reference(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

proptest! {
    #[test]
    fn crc32_matches_bitwise_reference(data in prop::collection::vec(any::<u8>(), 0..600)) {
        prop_assert_eq!(crc32(&data), crc32_reference(&data));
    }

    #[test]
    fn crc32_streaming_equals_one_shot(data in prop::collection::vec(any::<u8>(), 0..600), split in any::<usize>()) {
        let cut = if data.is_empty() { 0 } else { split % (data.len() + 1) };
        let mut hasher = Crc32::new();
        hasher.update(&data[..cut]);
        hasher.update(&data[cut..]);
        prop_assert_eq!(hasher.finish(), crc32(&data));
    }

    #[test]
    fn adler32_matches_reference(data in prop::collection::vec(any::<u8>(), 0..2000)) {
        prop_assert_eq!(adler32(&data), adler32_reference(&data));
    }

    #[test]
    fn adler32_streaming_equals_one_shot(data in prop::collection::vec(any::<u8>(), 0..2000), split in any::<usize>()) {
        let cut = if data.is_empty() { 0 } else { split % (data.len() + 1) };
        let mut hasher = Adler32::new();
        hasher.update(&data[..cut]);
        hasher.update(&data[cut..]);
        prop_assert_eq!(hasher.finish(), adler32(&data));
    }

    #[test]
    fn fnv1a_matches_reference(data in prop::collection::vec(any::<u8>(), 0..600)) {
        prop_assert_eq!(fnv1a_64(&data), fnv1a_reference(&data));
    }
}
