//! Bitsets, zigzag coding, and delta / prefix-sum streams, as used in codecs.
//!
//! Run with: `cargo run -p tpt-simd-bits --example bits`

use tpt_simd_bits::{
    delta_decode_u32, delta_encode_u32, popcount_u64, prefix_sum_u32, zigzag_decode_slice_u32,
    zigzag_encode_slice_i32,
};

fn main() {
    // Count set bits across a bitmap.
    let words = [0xFFu64, 0b1011, u64::MAX];
    assert_eq!(popcount_u64(&words), 8 + 3 + 64);

    // Delta-encode a sorted id list, which makes it compress well; decode restores it.
    let mut ids = [10u32, 12, 15, 15, 20];
    delta_encode_u32(&mut ids);
    assert_eq!(ids, [10, 2, 3, 0, 5]);
    delta_decode_u32(&mut ids);
    assert_eq!(ids, [10, 12, 15, 15, 20]);

    // Zigzag maps small signed values to small unsigned ones: 0, -1, 1, -2 -> 0, 1, 2, 3.
    let signed = [0i32, -1, 1, -2];
    let mut unsigned = [0u32; 4];
    zigzag_encode_slice_i32(&signed, &mut unsigned);
    assert_eq!(unsigned, [0, 1, 2, 3]);
    let mut back = [0i32; 4];
    zigzag_decode_slice_u32(&unsigned, &mut back);
    assert_eq!(back, signed);

    // Prefix sum (wrapping) turns per-block counts into offsets.
    let mut offsets = [3u32, 1, 4, 1];
    prefix_sum_u32(&mut offsets);
    assert_eq!(offsets, [3, 4, 8, 9]);

    println!("bits: all checks passed");
}
