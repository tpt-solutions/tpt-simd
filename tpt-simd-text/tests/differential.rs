//! Differential tests against oracles that are dev-dependencies only:
//! `faster-hex` for hex, `simdutf8` for UTF-8 validation, and `std` for UTF-16.

use tpt_simd_text::{
    TextError, hex_decode, hex_encode_lower, is_valid_utf8, utf8_to_utf16, utf16_to_utf8,
};

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
fn hex_encode_matches_faster_hex() {
    for n in 0..80 {
        let data = sample(n, n as u32);
        let mut ours = vec![0u8; 2 * n];
        let len = hex_encode_lower(&data, &mut ours).unwrap();
        let mut theirs = vec![0u8; 2 * n];
        let want = faster_hex::hex_encode(&data, &mut theirs).unwrap();
        assert_eq!(&ours[..len], want.as_bytes(), "n={n}");
    }
}

#[test]
fn hex_decode_matches_faster_hex_on_valid_input_both_cases() {
    for n in 0..80 {
        let data = sample(n, n as u32 + 100);
        let mut enc = vec![0u8; 2 * n];
        let len = hex_encode_lower(&data, &mut enc).unwrap();
        // Mixed case: upper-case every other digit.
        for (i, c) in enc[..len].iter_mut().enumerate() {
            if i % 2 == 1 {
                *c = c.to_ascii_uppercase();
            }
        }
        let mut ours = vec![0u8; n];
        assert_eq!(hex_decode(&enc[..len], &mut ours), Ok(n), "n={n}");
        assert_eq!(ours, data);
        let mut theirs = vec![0u8; n];
        faster_hex::hex_decode(&enc[..len], &mut theirs).unwrap();
        assert_eq!(ours, theirs);
    }
}

#[test]
fn hex_decode_rejects_what_faster_hex_rejects() {
    let cases: [&[u8]; 4] = [b"zz", b"0g", b"g0", b"12\n4"];
    for case in cases {
        let mut ours = [0u8; 2];
        let mut theirs = [0u8; 2];
        assert!(hex_decode(case, &mut ours).is_err(), "{case:?}");
        assert!(
            faster_hex::hex_decode(case, &mut theirs).is_err(),
            "{case:?}"
        );
    }
    let mut out = [0xAAu8; 2];
    assert_eq!(
        hex_decode(b"ab0g", &mut out),
        Err(TextError::InvalidHexDigit)
    );
    assert_eq!(out, [0xAA, 0xAA], "no partial write on invalid digit");
}

#[test]
fn utf8_validation_matches_simdutf8_over_random_bytes() {
    for seed in 0..2000u32 {
        let n = (seed % 70) as usize;
        let data = sample(n, seed);
        assert_eq!(
            is_valid_utf8(&data),
            simdutf8::basic::from_utf8(&data).is_ok(),
            "seed={seed}"
        );
    }
}

#[test]
fn utf8_validation_known_edge_cases() {
    let cases: [&[u8]; 6] = [
        &[0xC0, 0x80],             // overlong NUL
        &[0xED, 0xA0, 0x80],       // encoded surrogate
        &[0xF4, 0x90, 0x80, 0x80], // above U+10FFFF
        &[0xE2, 0x82],             // truncated
        "€😀".as_bytes(),
        &[],
    ];
    for case in cases {
        assert_eq!(
            is_valid_utf8(case),
            simdutf8::basic::from_utf8(case).is_ok(),
            "{case:?}"
        );
    }
}

#[test]
fn utf16_transcoding_matches_std_both_directions() {
    let strings = [
        String::new(),
        "a".to_string(),
        "ß€😀 mixed ✓".to_string(),
        "\u{10FFFF}\u{FFFD}".to_string(),
        "é".repeat(300),
    ];
    for s in &strings {
        let want: Vec<u16> = s.encode_utf16().collect();
        let mut got = vec![0u16; want.len()];
        let n = utf8_to_utf16(s, &mut got).unwrap();
        assert_eq!(&got[..n], &want[..]);

        let mut bytes = vec![0u8; s.len()];
        let m = utf16_to_utf8(&want, &mut bytes).unwrap();
        assert_eq!(&bytes[..m], s.as_bytes());
    }
}
