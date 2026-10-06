//! Shared test helpers for the tpt-simd workspace (unpublished).
//!
//! * proptest strategies for lanes, vectors and slices (including NaN/inf
//!   edge values and every tail length).
//! * tolerant float comparison for reductions that reassociate.
#![forbid(unsafe_code)]

use proptest::collection::vec;
use proptest::prelude::*;
use tpt_simd_vector::{Simd, SimdElement};

/// Slice lengths that exercise empty input, sub-vector tails and several
/// full vectors (0..=`max`).
pub fn lens(max: usize) -> impl Strategy<Value = usize> {
    0..=max
}

/// Finite `f32` in a range where sums of a few hundred values stay exact enough to compare.
pub fn finite_f32() -> impl Strategy<Value = f32> {
    -1000.0f32..1000.0
}

/// `f32` including NaN, ±inf, ±0 and extremes alongside ordinary values.
pub fn f32_with_specials() -> impl Strategy<Value = f32> {
    prop_oneof![
        8 => -1.0e6f32..1.0e6,
        1 => Just(f32::NAN),
        1 => Just(f32::INFINITY),
        1 => Just(f32::NEG_INFINITY),
        1 => Just(0.0f32),
        1 => Just(-0.0f32),
        1 => Just(f32::MAX),
        1 => Just(f32::MIN),
        1 => Just(f32::MIN_POSITIVE),
    ]
}

/// `i16` biased toward the extremes where saturation/overflow bugs hide.
pub fn i16_edgy() -> impl Strategy<Value = i16> {
    prop_oneof![
        6 => any::<i16>(),
        1 => Just(i16::MIN),
        1 => Just(i16::MAX),
        1 => Just(0i16),
        1 => Just(-1i16),
        1 => Just(1i16),
    ]
}

/// `i32` biased toward the extremes.
pub fn i32_edgy() -> impl Strategy<Value = i32> {
    prop_oneof![
        6 => any::<i32>(),
        1 => Just(i32::MIN),
        1 => Just(i32::MAX),
        1 => Just(0i32),
        1 => Just(-1i32),
        1 => Just(1i32),
    ]
}

/// A `Vec` of `len` in `0..=max_len` drawn from `elem`.
pub fn vec_of<S: Strategy>(elem: S, max_len: usize) -> impl Strategy<Value = Vec<S::Value>> {
    vec(elem, 0..=max_len)
}

/// Two equal-length vectors (for binary slice operations).
pub fn vec_pair<S: Strategy + Clone>(
    elem: S,
    max_len: usize,
) -> impl Strategy<Value = (Vec<S::Value>, Vec<S::Value>)>
where
    S::Value: Clone + core::fmt::Debug,
{
    (0..=max_len).prop_flat_map(move |n| (vec(elem.clone(), n), vec(elem.clone(), n)))
}

/// A `Simd<T, N>` whose lanes are drawn from `elem`.
pub fn simd_of<T, S, const N: usize>(elem: S) -> impl Strategy<Value = Simd<T, N>>
where
    T: SimdElement,
    S: Strategy<Value = T> + Clone,
{
    vec(elem, N).prop_map(|v| Simd::from_slice(&v))
}

/// `true` if `a` and `b` are equal, both NaN, or within the given tolerances.
pub fn approx_eq_f32(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
        return true; // also covers equal infinities
    }
    let diff = (a - b).abs();
    diff <= abs || diff <= rel * a.abs().max(b.abs())
}

/// Assert two `f32` slices are element-wise approximately equal.
#[track_caller]
pub fn assert_slice_close(actual: &[f32], expected: &[f32], rel: f32, abs: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(approx_eq_f32(a, e, rel, abs), "index {i}: got {a}, expected {e}");
    }
}
