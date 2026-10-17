//! Portable reference lane kernels.
//!
//! Each function keeps eight partial sums. Element `i` of the input goes to
//! lane `i % 8`, and products use a separate multiply and add (never FMA). The
//! AVX2 path in this crate reproduces these lane sums bit for bit, so the
//! public functions give the same result on every build.
//!
//! Inputs must have equal lengths that are a multiple of eight. The public
//! functions pass only that whole-vector body and sum the tail themselves.

use crate::LANES;

/// Eight lane sums of `a[i] * b[i]`.
#[inline]
#[must_use]
pub fn inner_product_lanes(a: &[f32], b: &[f32]) -> [f32; LANES] {
    let mut acc = [0.0f32; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    acc
}

/// Eight lane sums of `(a[i] - b[i])^2`.
#[inline]
#[must_use]
pub fn l2_squared_lanes(a: &[f32], b: &[f32]) -> [f32; LANES] {
    let mut acc = [0.0f32; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    acc
}

/// Eight lane sums for cosine: `(a·b, a·a, b·b)`, in that order.
#[inline]
#[must_use]
pub fn cosine_lanes(a: &[f32], b: &[f32]) -> ([f32; LANES], [f32; LANES], [f32; LANES]) {
    let mut dot = [0.0f32; LANES];
    let mut na = [0.0f32; LANES];
    let mut nb = [0.0f32; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for i in 0..LANES {
            dot[i] += x[i] * y[i];
            na[i] += x[i] * x[i];
            nb[i] += y[i] * y[i];
        }
    }
    (dot, na, nb)
}
