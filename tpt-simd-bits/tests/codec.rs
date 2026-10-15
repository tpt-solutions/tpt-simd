//! Varint and bitpacking: known answers, round trips, and malformed input.

use proptest::prelude::*;
use tpt_simd_bits::{
    bitpack_u32, bitpacked_len, bitunpack_u32, varint_decode_u32, varint_encode_u32,
    varint_encoded_len_u32, varint_len_u64,
};

#[test]
fn varint_known_answers() {
    let mut out = [0u8; 16];
    assert_eq!(varint_encode_u32(&[0], &mut out), 1);
    assert_eq!(out[0], 0x00);
    assert_eq!(varint_encode_u32(&[127], &mut out), 1);
    assert_eq!(out[0], 0x7f);
    assert_eq!(varint_encode_u32(&[128], &mut out), 2);
    assert_eq!(&out[..2], &[0x80, 0x01]);
    assert_eq!(varint_encode_u32(&[u32::MAX], &mut out), 5);
    assert_eq!(&out[..5], &[0xff, 0xff, 0xff, 0xff, 0x0f]);
    assert_eq!(varint_len_u64(0), 1);
    assert_eq!(varint_len_u64(u64::MAX), 10);
}

#[test]
fn varint_rejects_malformed_input() {
    let mut out = [0u32; 1];
    // Truncated: continuation bit set on the last byte.
    assert_eq!(varint_decode_u32(&[0x80], &mut out), None);
    // Six bytes: longer than any u32 varint.
    assert_eq!(
        varint_decode_u32(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x00], &mut out),
        None
    );
    // Five bytes, but the value needs 33 bits.
    assert_eq!(
        varint_decode_u32(&[0xff, 0xff, 0xff, 0xff, 0x1f], &mut out),
        None
    );
}

#[test]
fn bitpack_known_answer() {
    // Values 1, 2, 3 at 2 bits: bit stream 01 10 11 read LSB first is 0b11_10_01 = 0x39
    // when packed as 1 | 2 << 2 | 3 << 4.
    let mut out = [0u8; 2];
    let n = bitpack_u32(&[1, 2, 3], 2, &mut out);
    assert_eq!(n, 1);
    assert_eq!(out[0], 0b11_10_01);
}

proptest! {
    #[test]
    fn varint_round_trips(values in prop::collection::vec(any::<u32>(), 0..300)) {
        let mut buf = vec![0u8; varint_encoded_len_u32(&values)];
        let written = varint_encode_u32(&values, &mut buf);
        prop_assert_eq!(written, buf.len());
        let mut back = vec![0u32; values.len()];
        let (count, used) = varint_decode_u32(&buf, &mut back).expect("valid stream");
        prop_assert_eq!(count, values.len());
        prop_assert_eq!(used, written);
        prop_assert_eq!(back, values);
    }

    #[test]
    fn bitpack_round_trips(
        bits in 1u32..=32,
        raw in prop::collection::vec(any::<u32>(), 0..300),
    ) {
        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let values: Vec<u32> = raw.iter().map(|v| v & mask).collect();
        let mut packed = vec![0u8; bitpacked_len(values.len(), bits)];
        let written = bitpack_u32(&values, bits, &mut packed);
        prop_assert_eq!(written, packed.len());
        let mut back = vec![0u32; values.len()];
        prop_assert_eq!(bitunpack_u32(&packed, bits, &mut back), Some(values.len()));
        prop_assert_eq!(back, values);
    }

    #[test]
    fn bitunpack_rejects_short_input(bits in 1u32..=32, n in 1usize..200) {
        let short = vec![0u8; bitpacked_len(n, bits) - 1];
        let mut out = vec![0u32; n];
        prop_assert_eq!(bitunpack_u32(&short, bits, &mut out), None);
    }
}
