//! Distance and similarity kernels for vector search and clustering.
//!
//! | function | element | accumulator | notes |
//! |---|---|---|---|
//! | [`inner_product_f32`] | `f32` | `f32`, 8 lanes | fixed order, see below |
//! | [`l2_squared_f32`] | `f32` | `f32`, 8 lanes | fixed order, see below |
//! | [`cosine_distance_f32`] | `f32` | `f32`, 8 lanes | `1 - cos(a, b)` |
//! | [`inner_product_i8`] / [`l2_squared_i8`] | `i8` | `i64` | exact |
//! | [`inner_product_u8`] / [`l2_squared_u8`] | `u8` | `i64` | exact |
//! | [`hamming_distance_u64`] | `u64` bit-vectors | `u32` | exact |
//!
//! ## Length policy
//!
//! Paired slices must have the same length; otherwise the function **panics**
//! naming the function and both lengths. Empty input is valid and returns zero
//! (for cosine distance, see its docs).
//!
//! ## Determinism
//!
//! The `f32` kernels keep eight independent partial sums filled in a fixed
//! order, reduce them in a fixed pairwise order, then add the tail
//! sequentially. Separate multiply and add are used (never FMA), so the result
//! is the same on every target and build configuration. It is **not** equal to
//! a naive left-to-right sum in general; the difference is the usual float
//! reordering error.
//!
//! The integer kernels are exact.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_distance::{hamming_distance_u64, l2_squared_u8};
//!
//! assert_eq!(l2_squared_u8(&[0, 3], &[4, 0]), 16 + 9);
//! assert_eq!(hamming_distance_u64(&[0b1011], &[0b0010]), 2);
//! ```
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_code)]
#![warn(missing_docs)]

const LANES: usize = 8;

#[cold]
#[inline(never)]
fn len_mismatch(func: &str, a: usize, b: usize) -> ! {
    panic!("{func}: slice lengths differ ({a} vs {b})");
}

/// Sums eight partial accumulators in a fixed pairwise order.
#[inline]
fn reduce8(acc: [f32; LANES]) -> f32 {
    let lo = (acc[0] + acc[4]) + (acc[1] + acc[5]);
    let hi = (acc[2] + acc[6]) + (acc[3] + acc[7]);
    lo + hi
}

/// Inner product of two `f32` slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Examples
///
/// ```
/// assert_eq!(tpt_simd_distance::inner_product_f32(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
/// ```
#[must_use]
pub fn inner_product_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        len_mismatch("inner_product_f32", a.len(), b.len());
    }
    let mut acc = [0.0f32; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let (ta, tb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = reduce8(acc);
    for (x, y) in ta.iter().zip(tb) {
        s += x * y;
    }
    s
}

/// Squared Euclidean distance between two `f32` slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Examples
///
/// ```
/// assert_eq!(tpt_simd_distance::l2_squared_f32(&[0.0, 3.0], &[4.0, 0.0]), 25.0);
/// ```
#[must_use]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        len_mismatch("l2_squared_f32", a.len(), b.len());
    }
    let mut acc = [0.0f32; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let (ta, tb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut s = reduce8(acc);
    for (x, y) in ta.iter().zip(tb) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Cosine distance `1 - cos(a, b)` between two `f32` slices.
///
/// The result lies in `[0, 2]` up to rounding. If either vector has zero norm
/// the cosine is undefined, so this returns `1.0` (treated as orthogonal).
/// That includes two empty slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Examples
///
/// ```
/// let d = tpt_simd_distance::cosine_distance_f32(&[1.0, 0.0], &[0.0, 2.0]);
/// assert!((d - 1.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        len_mismatch("cosine_distance_f32", a.len(), b.len());
    }
    let mut dot = [0.0f32; LANES];
    let mut na = [0.0f32; LANES];
    let mut nb = [0.0f32; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let (ta, tb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            dot[i] += x[i] * y[i];
            na[i] += x[i] * x[i];
            nb[i] += y[i] * y[i];
        }
    }
    let (mut d, mut sa, mut sb) = (reduce8(dot), reduce8(na), reduce8(nb));
    for (x, y) in ta.iter().zip(tb) {
        d += x * y;
        sa += x * x;
        sb += y * y;
    }
    if sa == 0.0 || sb == 0.0 {
        return 1.0;
    }
    1.0 - d / (libm::sqrtf(sa) * libm::sqrtf(sb))
}

/// Inner product of two `i8` slices, exact.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn inner_product_i8(a: &[i8], b: &[i8]) -> i64 {
    if a.len() != b.len() {
        len_mismatch("inner_product_i8", a.len(), b.len());
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// Squared Euclidean distance between two `i8` slices, exact.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn l2_squared_i8(a: &[i8], b: &[i8]) -> i64 {
    if a.len() != b.len() {
        len_mismatch("l2_squared_i8", a.len(), b.len());
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            d * d
        })
        .sum()
}

/// Inner product of two `u8` slices, exact.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn inner_product_u8(a: &[u8], b: &[u8]) -> i64 {
    if a.len() != b.len() {
        len_mismatch("inner_product_u8", a.len(), b.len());
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// Squared Euclidean distance between two `u8` slices, exact.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn l2_squared_u8(a: &[u8], b: &[u8]) -> i64 {
    if a.len() != b.len() {
        len_mismatch("l2_squared_u8", a.len(), b.len());
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            d * d
        })
        .sum()
}

/// Hamming distance (number of differing bits) between two packed bit-vectors.
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Examples
///
/// ```
/// assert_eq!(tpt_simd_distance::hamming_distance_u64(&[u64::MAX], &[0]), 64);
/// ```
#[must_use]
pub fn hamming_distance_u64(a: &[u64], b: &[u64]) -> u32 {
    if a.len() != b.len() {
        len_mismatch("hamming_distance_u64", a.len(), b.len());
    }
    a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

/// Finds the `k` rows of a row-major `f32` matrix nearest to `query` by squared
/// L2 distance, writing them nearest first into `out_idx` / `out_dist`.
///
/// `rows` holds `rows.len() / dim` vectors of length `dim`, back to back.
/// Returns the number of results written, `min(k, number of rows)`, where `k`
/// is the shorter of the two output slices. Ties are broken by lower row index.
/// Distances are computed with [`l2_squared_f32`], so they match it exactly.
///
/// The selection keeps a sorted buffer of at most `k` entries, so the cost is
/// `O(rows * dim + rows * k)` with no allocation.
///
/// # Panics
///
/// Panics if `dim` is zero, `query.len() != dim`, or `rows.len()` is not a
/// multiple of `dim`.
///
/// # Examples
///
/// ```
/// let rows = [0.0, 0.0, 5.0, 5.0, 1.0, 0.0];
/// let mut idx = [0usize; 2];
/// let mut dist = [0.0f32; 2];
/// let n = tpt_simd_distance::top_k_l2_f32(&[0.0, 0.0], &rows, 2, &mut idx, &mut dist);
/// assert_eq!(n, 2);
/// assert_eq!(idx, [0, 2]);
/// assert_eq!(dist, [0.0, 1.0]);
/// ```
pub fn top_k_l2_f32(
    query: &[f32],
    rows: &[f32],
    dim: usize,
    out_idx: &mut [usize],
    out_dist: &mut [f32],
) -> usize {
    assert!(dim != 0, "top_k_l2_f32: dim must be non-zero");
    assert!(
        query.len() == dim,
        "top_k_l2_f32: query length {} does not match dim {dim}",
        query.len()
    );
    assert!(
        rows.len().is_multiple_of(dim),
        "top_k_l2_f32: rows length {} is not a multiple of dim {dim}",
        rows.len()
    );
    let k = out_idx.len().min(out_dist.len());
    if k == 0 {
        return 0;
    }
    let mut len = 0;
    for (row, v) in rows.chunks_exact(dim).enumerate() {
        let d = l2_squared_f32(query, v);
        // Rows arrive in index order, so an equal distance never displaces an
        // earlier row: insert only when strictly closer than the current last.
        if len == k && d >= out_dist[k - 1] {
            continue;
        }
        let mut pos = if len < k { len } else { k - 1 };
        while pos > 0 && out_dist[pos - 1] > d {
            out_dist[pos] = out_dist[pos - 1];
            out_idx[pos] = out_idx[pos - 1];
            pos -= 1;
        }
        out_dist[pos] = d;
        out_idx[pos] = row;
        if len < k {
            len += 1;
        }
    }
    len
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn wide_sum(terms: impl Iterator<Item = f64>) -> f64 {
        terms.sum()
    }

    fn sample(n: usize, mul: usize, off: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * mul % 101) as f32 - off) / 7.0)
            .collect()
    }

    #[test]
    fn f32_kernels_match_f64_within_tolerance() {
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 100, 1000] {
            let a = sample(n, 37, 50.0);
            let b = sample(n, 53, 48.0);
            let exp_ip = wide_sum(a.iter().zip(&b).map(|(&x, &y)| f64::from(x) * f64::from(y)));
            let ip = f64::from(inner_product_f32(&a, &b));
            assert!(
                (ip - exp_ip).abs() <= 1e-3 * (1.0 + exp_ip.abs()),
                "ip n={n}"
            );

            let exp_l2 = wide_sum(
                a.iter()
                    .zip(&b)
                    .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)),
            );
            let l2 = f64::from(l2_squared_f32(&a, &b));
            assert!(
                (l2 - exp_l2).abs() <= 1e-3 * (1.0 + exp_l2.abs()),
                "l2 n={n}"
            );
        }
    }

    #[test]
    fn f32_empty_and_tail() {
        assert_eq!(inner_product_f32(&[], &[]), 0.0);
        assert_eq!(l2_squared_f32(&[], &[]), 0.0);
        assert_eq!(inner_product_f32(&[2.0; 9], &[3.0; 9]), 54.0);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "libm x86 routine uses inline asm, which Miri does not support"
    )]
    fn cosine_special_cases() {
        assert_eq!(cosine_distance_f32(&[], &[]), 1.0);
        assert_eq!(cosine_distance_f32(&[0.0; 10], &[1.0; 10]), 1.0);
        assert!(cosine_distance_f32(&[1.0; 16], &[2.0; 16]).abs() < 1e-6);
        assert!((cosine_distance_f32(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn integer_kernels_exact() {
        let a: Vec<i8> = (0..300).map(|i| (i % 255 - 127) as i8).collect();
        let b: Vec<i8> = (0..300).map(|i| (i * 7 % 255 - 127) as i8).collect();
        let ip: i64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum();
        assert_eq!(inner_product_i8(&a, &b), ip);
        assert_eq!(l2_squared_i8(&[-128], &[127]), 255 * 255);
        assert_eq!(
            inner_product_u8(&[255; 1000], &[255; 1000]),
            255 * 255 * 1000
        );
        assert_eq!(l2_squared_u8(&[0; 3], &[255; 3]), 3 * 255 * 255);
    }

    #[test]
    fn hamming_counts_bits() {
        assert_eq!(hamming_distance_u64(&[], &[]), 0);
        assert_eq!(hamming_distance_u64(&[0, 0, 0], &[0, 0, u64::MAX]), 64);
    }

    #[test]
    fn top_k_matches_full_sort() {
        let dim = 5;
        let n = 200;
        let rows: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 31 % 97) as f32 - 40.0) / 9.0)
            .collect();
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        let mut all: Vec<(f32, usize)> = (0..n)
            .map(|r| (l2_squared_f32(&query, &rows[r * dim..(r + 1) * dim]), r))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        for k in [1usize, 3, 10, 200, 250] {
            let mut idx = vec![0usize; k];
            let mut dist = vec![0.0f32; k];
            let got = top_k_l2_f32(&query, &rows, dim, &mut idx, &mut dist);
            assert_eq!(got, k.min(n));
            for i in 0..got {
                assert_eq!(idx[i], all[i].1, "k={k} i={i}");
                assert_eq!(dist[i], all[i].0);
            }
        }
    }

    #[test]
    fn top_k_ties_prefer_lower_index_and_empty_inputs() {
        let rows = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut idx = [9usize; 2];
        let mut dist = [0.0f32; 2];
        assert_eq!(top_k_l2_f32(&[0.0], &rows, 1, &mut idx, &mut dist), 2);
        assert_eq!(idx, [0, 1]);
        assert_eq!(top_k_l2_f32(&[0.0], &[], 1, &mut idx, &mut dist), 0);
        assert_eq!(top_k_l2_f32(&[0.0], &rows, 1, &mut [], &mut []), 0);
    }

    #[test]
    #[should_panic(expected = "rows length 5 is not a multiple of dim 2")]
    fn top_k_rejects_ragged_rows() {
        let _ = top_k_l2_f32(
            &[0.0, 0.0],
            &[0.0; 5],
            2,
            &mut [0usize; 1],
            &mut [0.0f32; 1],
        );
    }

    #[test]
    #[should_panic(expected = "l2_squared_u8: slice lengths differ (2 vs 3)")]
    fn length_mismatch_panics() {
        let _ = l2_squared_u8(&[1, 2], &[1, 2, 3]);
    }
}
