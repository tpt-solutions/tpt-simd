#![no_main]

//! Checked gather and scatter against a scalar reference. Indices are reduced
//! into range; out-of-range indices are covered by the `*_checked` panics
//! and are not fuzzed here. Duplicate scatter indices: the last lane wins.

use libfuzzer_sys::fuzz_target;
use tpt_simd_core::Simd;
use tpt_simd_gather::gather_checked_i32;
use tpt_simd_scatter::scatter_checked_i32;

fuzz_target!(|data: &[u8]| {
    if data.len() < 9 {
        return;
    }
    let len = 1 + data[0] as usize;
    let table: Vec<i32> = (0..len).map(|i| (i as i32).wrapping_mul(0x9E37) ^ data[1] as i32).collect();
    let mut idx = [0i32; 8];
    let mut val = [0i32; 8];
    for k in 0..8 {
        let b = data[1 + k] as usize;
        idx[k] = (b % len) as i32;
        val[k] = (data[(k + 1) % data.len()] as i32) << 16 | k as i32;
    }

    let got = gather_checked_i32(&table, Simd::from_array(idx)).to_array();
    for k in 0..8 {
        assert_eq!(got[k], table[idx[k] as usize], "gather lane {k}");
    }

    let mut out = table.clone();
    scatter_checked_i32(&mut out, Simd::from_array(idx), Simd::from_array(val));
    let mut expect = table.clone();
    for k in 0..8 {
        expect[idx[k] as usize] = val[k];
    }
    assert_eq!(out, expect, "scatter");
});
