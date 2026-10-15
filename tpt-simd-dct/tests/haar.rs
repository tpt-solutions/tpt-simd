//! Haar wavelet transform: one-level values, energy, perfect reconstruction.

use proptest::prelude::*;
use tpt_simd_dct::{haar_forward, haar_inverse};

#[test]
fn one_level_matches_the_definition() {
    let mut x = [3.0f32, 1.0, -2.0, 4.0];
    haar_forward(&mut x, 1);
    let r = core::f32::consts::FRAC_1_SQRT_2;
    let want = [
        (3.0 + 1.0) * r,
        (-2.0 + 4.0) * r,
        (3.0 - 1.0) * r,
        (-2.0 - 4.0) * r,
    ];
    for (got, want) in x.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

#[test]
fn two_levels_recurse_on_the_approximations() {
    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    haar_forward(&mut x, 2);
    // Level 1 gives approximations [3r, 7r] and details [-r, -r]; level 2 on [3r, 7r].
    let r = core::f32::consts::FRAC_1_SQRT_2;
    assert!((x[0] - (3.0 * r + 7.0 * r) * r).abs() < 1e-6);
    assert!((x[1] - (3.0 * r - 7.0 * r) * r).abs() < 1e-6);
    assert!((x[2] + r).abs() < 1e-6 && (x[3] + r).abs() < 1e-6);
}

proptest! {
    #[test]
    fn preserves_energy_and_reconstructs(
        levels in 0u32..=6,
        seed in prop::collection::vec(-100.0f32..100.0, 64),
    ) {
        let n = 64usize;
        let mut x = seed.clone();
        let energy_before: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
        haar_forward(&mut x, levels);
        let energy_after: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
        prop_assert!((energy_after - energy_before).abs() <= 1e-6 * energy_before.max(1.0));
        haar_inverse(&mut x, levels);
        for i in 0..n {
            prop_assert!((x[i] - seed[i]).abs() <= 1e-3 * seed[i].abs().max(1.0));
        }
    }
}

#[test]
#[should_panic(expected = "not divisible")]
fn rejects_lengths_not_divisible_by_the_levels() {
    let mut x = [0.0f32; 6];
    haar_forward(&mut x, 2);
}
