#![no_main]

//! The AVX2 / AVX-512BW / portable `i16` dot products must agree exactly
//! (wrapping `i32` accumulation is order-independent).

use libfuzzer_sys::fuzz_target;
use tpt_simd_dot::{dot_product_i16, portable};

fuzz_target!(|data: &[u8]| {
    let v: Vec<i16> = data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let n = v.len() / 2;
    let (a, b) = (&v[..n], &v[n..2 * n]);
    assert_eq!(dot_product_i16(a, b), portable::dot_product_i16(a, b), "n={n}");
});
