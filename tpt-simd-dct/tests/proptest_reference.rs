//! Property tests: DCT against the orthonormal DCT-II formula, and inverse round trip.

use proptest::prelude::*;
use tpt_simd_dct::Dct;

/// Orthonormal DCT-II, straight from the definition.
fn dct2_reference(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let scale = if k == 0 {
                (1.0 / n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
            };
            let sum: f64 = x
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    f64::from(v)
                        * (std::f64::consts::PI / n as f64 * (i as f64 + 0.5) * k as f64).cos()
                })
                .sum();
            (scale * sum) as f32
        })
        .collect()
}

proptest! {
    #[test]
    fn forward_matches_orthonormal_dct2(
        n in 1usize..=48,
        seed in prop::collection::vec(-1.0f32..1.0, 48),
    ) {
        let x = &seed[..n];
        let dct = Dct::new(n);
        let mut out = vec![0f32; n];
        dct.forward(x, &mut out);
        let want = dct2_reference(x);
        for k in 0..n {
            prop_assert!((out[k] - want[k]).abs() <= 1e-4, "k = {}: {} vs {}", k, out[k], want[k]);
        }
    }

    #[test]
    fn inverse_undoes_forward(
        n in 1usize..=48,
        seed in prop::collection::vec(-1.0f32..1.0, 48),
    ) {
        let x = &seed[..n];
        let dct = Dct::new(n);
        let mut coeffs = vec![0f32; n];
        let mut back = vec![0f32; n];
        dct.forward(x, &mut coeffs);
        dct.inverse(&coeffs, &mut back);
        for i in 0..n {
            prop_assert!((back[i] - x[i]).abs() <= 1e-4);
        }
    }
}
