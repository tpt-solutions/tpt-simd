#![no_main]

//! `gemm_f32` (packed, blocked, FMA where compiled in) against the naive
//! reference. Results may differ by reassociation, so the check is relative
//! to the magnitude of the terms (ADR 0003 tolerance tier).

use libfuzzer_sys::fuzz_target;
use tpt_simd_blas::{gemm_f32, reference};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let m = 1 + data[0] as usize % 40;
    let n = 1 + data[1] as usize % 40;
    let k = 1 + data[2] as usize % 40;
    // Finite values in [-1, 1): every byte maps to one value, so no NaN/inf.
    let val = |i: usize| (data.get(i % data.len()).copied().unwrap_or(0) as f32 - 127.5) / 127.5;
    let a: Vec<f32> = (0..m * k).map(|i| val(i + 3)).collect();
    let b: Vec<f32> = (0..k * n).map(|i| val(i + 3 + m * k)).collect();
    let mut c = vec![0.0f32; m * n];
    let mut want = vec![0.0f32; m * n];
    gemm_f32(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m);
    reference::gemm_f32(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut want, m);
    let tol = 1e-5 * (k as f32) * 2.0;
    for i in 0..m * n {
        assert!((c[i] - want[i]).abs() <= tol, "({i}) got {} want {}", c[i], want[i]);
    }
});
