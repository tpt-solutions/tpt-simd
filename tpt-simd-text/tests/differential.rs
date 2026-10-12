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

#[test]
fn hex_decode_agrees_with_faster_hex_for_every_byte_value_in_every_lane() {
    // Every byte value at every position of 32-, 64- and 96-byte inputs (whole AVX2 blocks,
    // with the SSE2 and scalar tails after them). The rest of the input is valid digits, so the
    // only difference between cases is the byte under test.
    for n in [32usize, 64, 96] {
        for pos in 0..n {
            for b in 0..=255u8 {
                let mut src: Vec<u8> = (0..n).map(|i| b"0aF9bE"[i % 6]).collect();
                src[pos] = b;
                let mut ours = vec![0xAAu8; n / 2];
                let mut theirs = vec![0xAAu8; n / 2];
                let got = hex_decode(&src, &mut ours);
                let want = faster_hex::hex_decode(&src, &mut theirs);
                assert_eq!(got.is_ok(), want.is_ok(), "n={n} pos={pos} b={b:#04x}");
                if got.is_ok() {
                    assert_eq!(ours, theirs, "n={n} pos={pos} b={b:#04x}");
                }
            }
        }
    }
}

#[test]
fn hex_decode_rejects_invalid_byte_at_every_position_without_writing() {
    // Each length crosses the 16-byte SIMD blocks, and the bad byte is placed in the blocks and
    // in the scalar tail. Bytes near the digit ranges are the easiest to misclassify.
    let bad_bytes = [
        b'g', b'G', b'/', b':', b'`', b'@', b'Z', 0x00, 0x7F, 0x80, 0xFF,
    ];
    for n in (2..=70).step_by(2) {
        for pos in 0..n {
            for &bad in &bad_bytes {
                let mut src: Vec<u8> = (0..n).map(|i| b"0aF9bE"[i % 6]).collect();
                src[pos] = bad;
                let mut out = vec![0xAAu8; n / 2];
                assert_eq!(
                    hex_decode(&src, &mut out),
                    Err(TextError::InvalidHexDigit),
                    "n={n} pos={pos} bad={bad:#x}"
                );
                assert!(
                    out.iter().all(|&b| b == 0xAA),
                    "partial write n={n} pos={pos}"
                );
            }
        }
    }
}

fn core_valid(bytes: &[u8]) -> bool {
    core::str::from_utf8(bytes).is_ok()
}

#[test]
fn utf8_validation_matches_core_for_all_two_byte_pairs() {
    // Each pair sits inside a 96-byte ASCII buffer, at positions around the 32-byte blocks.
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            for pos in [0usize, 1, 30, 31, 32, 33, 62] {
                let mut buf = vec![b'x'; 96];
                buf[pos] = a;
                buf[pos + 1] = b;
                assert_eq!(
                    is_valid_utf8(&buf),
                    core_valid(&buf),
                    "a={a:#x} b={b:#x} pos={pos}"
                );
            }
        }
    }
}

#[test]
fn utf8_validation_matches_core_for_structured_three_and_four_byte_sequences() {
    // Boundary bytes of every UTF-8 class: ASCII, stray continuations, all leads, and the
    // second-byte limits of the overlong, surrogate and too-large ranges.
    const INTERESTING: [u8; 21] = [
        0x00, 0x41, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1,
        0xED, 0xEF, 0xF0, 0xF4, 0xF5, 0xFF,
    ];
    let n = INTERESTING.len();
    for &pos in &[30usize, 31, 61] {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let seq = [
                            INTERESTING[i],
                            INTERESTING[j],
                            INTERESTING[k],
                            INTERESTING[l],
                        ];
                        let mut buf = vec![b'x'; 96];
                        buf[pos..pos + 4].copy_from_slice(&seq);
                        assert_eq!(is_valid_utf8(&buf), core_valid(&buf), "{seq:x?} pos={pos}");
                    }
                }
            }
        }
    }
}

#[test]
fn utf8_validation_matches_core_for_every_prefix_and_corruption() {
    // Truncation at every length exercises the tail and the pending-lead check at block ends.
    let text = "héllo wörld 😀 日本語テキスト ✓ ".repeat(8);
    let bytes = text.as_bytes();
    for len in 0..=bytes.len() {
        assert_eq!(
            is_valid_utf8(&bytes[..len]),
            core_valid(&bytes[..len]),
            "prefix {len}"
        );
    }
    // Corrupting any single byte, with a few replacement values.
    for pos in 0..bytes.len() {
        for &value in &[0x80u8, 0xBF, 0xC0, 0xE0, 0xF4, 0xFF, b'a'] {
            let mut bad = bytes.to_vec();
            bad[pos] = value;
            assert_eq!(
                is_valid_utf8(&bad),
                core_valid(&bad),
                "pos {pos} value {value:#x}"
            );
        }
    }
}

#[test]
fn utf8_validation_matches_core_on_pseudo_random_bytes() {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..20_000 {
        let len = 64 + (next() % 300) as usize;
        // Bias towards the multi-byte ranges so that many inputs are nearly valid.
        let bytes: Vec<u8> = (0..len)
            .map(|_| match next() % 4 {
                0 => (next() % 256) as u8,
                1 => 0x80 + (next() % 64) as u8,
                2 => b"\xC2\xE0\xED\xF0\xF4"[(next() % 5) as usize],
                _ => b'a' + (next() % 26) as u8,
            })
            .collect();
        assert_eq!(is_valid_utf8(&bytes), core_valid(&bytes), "{bytes:x?}");
    }
}
