#![no_main]

//! The SSE2 / AVX2 8x8 `i16` transpose must equal the portable transpose.

use libfuzzer_sys::fuzz_target;
use tpt_simd_permute::{transpose_8x8_i16, transpose_8x8_i16_portable};

fuzz_target!(|data: &[u8]| {
    let mut m = [[0i16; 8]; 8];
    for (i, cell) in m.iter_mut().flatten().enumerate() {
        let b = (i * 2) % data.len().max(1);
        *cell = i16::from_le_bytes([data.get(b).copied().unwrap_or(0), data.get(b + 1).copied().unwrap_or(0)]);
    }
    let mut want = m;
    transpose_8x8_i16_portable(&mut want);
    transpose_8x8_i16(&mut m);
    assert_eq!(m, want);
});
