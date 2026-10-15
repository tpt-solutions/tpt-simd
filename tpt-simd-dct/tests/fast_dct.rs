//! FFT-based DCT against the table-based `Dct`, and round trips.

use proptest::prelude::*;
use tpt_simd_dct::{Dct, FastDct};

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(1.0f32, |m, x| m.max(x.abs()))
}

#[test]
fn matches_table_dct_for_every_power_of_two() {
    for log in 0..=9 {
        let n = 1usize << log;
        let input: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0)
            .collect();
        let table = Dct::new(n);
        let fast = FastDct::new(n);

        let mut want = vec![0f32; n];
        let mut got = vec![0f32; n];
        table.forward(&input, &mut want);
        fast.forward(&input, &mut got);
        let tol = 1e-4 * scale_of(&want);
        for k in 0..n {
            assert!(
                (got[k] - want[k]).abs() <= tol,
                "forward n={n} k={k}: {} vs {}",
                got[k],
                want[k]
            );
        }

        table.inverse(&input, &mut want);
        fast.inverse(&input, &mut got);
        let tol = 1e-4 * scale_of(&want);
        for i in 0..n {
            assert!(
                (got[i] - want[i]).abs() <= tol,
                "inverse n={n} i={i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}

proptest! {
    #[test]
    fn inverse_undoes_forward(log in 0u32..=8, seed in prop::collection::vec(-1.0f32..1.0, 256)) {
        let n = 1usize << log;
        let x = &seed[..n];
        let fast = FastDct::new(n);
        let mut coeffs = vec![0f32; n];
        let mut back = vec![0f32; n];
        fast.forward(x, &mut coeffs);
        fast.inverse(&coeffs, &mut back);
        for i in 0..n {
            prop_assert!((back[i] - x[i]).abs() <= 1e-4, "i={}: {} vs {}", i, back[i], x[i]);
        }
    }
}

#[test]
#[should_panic(expected = "power of two")]
fn rejects_non_power_of_two() {
    let _ = FastDct::new(12);
}

#[test]
fn size_one_is_identity() {
    let fast = FastDct::new(1);
    let mut out = [0f32; 1];
    fast.forward(&[0.75], &mut out);
    assert!((out[0] - 0.75).abs() < 1e-7);
}
