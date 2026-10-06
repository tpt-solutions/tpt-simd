//! SpMV kernels (CSR/CSC, plain and transposed).

use crate::real::Real;
use crate::view::{Compressed, CscView, CsrView};

/// How each compressed line is reduced in the gather-style kernels
/// ([`spmv_csr`], [`spmv_csc_t`]). See the crate docs, "Measured findings".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowStrategy {
    /// One accumulator, strictly left to right (latency-bound reference
    /// strategy; the order of the naive loop).
    Scalar,
    /// Four independent accumulators for every line. Fastest or tied on
    /// every benchmarked shape, so it is the default ([`DEFAULT_STRATEGY`]).
    Lanes4,
    /// Eight independent accumulators for every line.
    Lanes8,
    /// Lines shorter than 8 use the scalar loop, longer ones eight
    /// accumulators (measured no better than [`Lanes4`](Self::Lanes4)).
    Hybrid,
}

/// Strategy used by [`spmv_csr`] and [`spmv_csc_t`].
pub const DEFAULT_STRATEGY: RowStrategy = RowStrategy::Lanes4;

/// Reduces the `W` lane accumulators with a fixed halving tree.
#[inline(always)]
fn reduce<T: Real, const W: usize>(mut acc: [T; W]) -> T {
    let mut w = W;
    while w > 1 {
        w /= 2;
        for i in 0..w {
            acc[i] = acc[i] + acc[i + w];
        }
    }
    acc[0]
}

/// Loads `x[i]` without a bounds check.
///
/// # Safety
/// `i < x.len()`.
#[inline(always)]
unsafe fn ld<T: Real>(x: &[T], i: u32) -> T {
    // SAFETY: forwarded caller contract.
    unsafe { *x.get_unchecked(i as usize) }
}

/// Sparse dot product of one line with the dense vector `x`: scalar loads
/// of `x`, `W` independent accumulators for lines of at least `SHORT`
/// entries, a single accumulator otherwise.
///
/// # Safety
/// Every `idx[k] < x.len()`; `idx.len() == d.len()`.
#[inline(always)]
unsafe fn line_dot<T: Real, const W: usize, const SHORT: usize>(
    idx: &[u32],
    d: &[T],
    x: &[T],
) -> T {
    debug_assert_eq!(idx.len(), d.len());
    let mut t = T::ZERO;
    if idx.len() < SHORT {
        for (&i, &v) in idx.iter().zip(d) {
            // SAFETY: caller contract.
            t = t + v * unsafe { ld(x, i) };
        }
        return t;
    }
    let ic = idx.chunks_exact(W);
    let dc = d.chunks_exact(W);
    let (it, dt) = (ic.remainder(), dc.remainder());
    let mut acc = [T::ZERO; W];
    for (i, v) in ic.zip(dc) {
        for j in 0..W {
            // SAFETY: caller contract.
            acc[j] = acc[j] + v[j] * unsafe { ld(x, i[j]) };
        }
    }
    t = reduce(acc);
    for (&i, &v) in it.iter().zip(dt) {
        // SAFETY: caller contract.
        t = t + v * unsafe { ld(x, i) };
    }
    t
}

/// `y[r] = alpha * (line r . x) + beta * y[r]` for every line.
fn gather_lines<T: Real, const W: usize, const SHORT: usize>(
    c: &Compressed<'_, T>,
    alpha: T,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    assert_eq!(x.len(), c.minor, "spmv: x has wrong length");
    assert_eq!(y.len(), c.major, "spmv: y has wrong length");
    let overwrite = beta == T::ZERO;
    for (yr, w) in y.iter_mut().zip(c.indptr.windows(2)) {
        let (s, e) = (w[0], w[1]);
        // SAFETY: `c` is validated (every index < minor == x.len(), checked
        // above) and the index/data slices have equal length.
        let t = unsafe { line_dot::<T, W, SHORT>(&c.indices[s..e], &c.data[s..e], x) };
        *yr = if overwrite {
            alpha * t
        } else {
            alpha * t + beta * *yr
        };
    }
}

/// `y = beta * y`, overwriting (not multiplying) when `beta == 0`.
#[inline]
fn scale_y<T: Real>(beta: T, y: &mut [T]) {
    if beta == T::ZERO {
        y.fill(T::ZERO);
    } else if beta != T::ONE {
        for v in y.iter_mut() {
            *v = beta * *v;
        }
    }
}

/// `y = alpha * A * x + beta * y` where line `k` of `c` is column `k` of
/// the operator: for each line, `y[idx] += data * (alpha * x[k])`.
fn scatter_lines<T: Real>(c: &Compressed<'_, T>, alpha: T, x: &[T], beta: T, y: &mut [T]) {
    assert_eq!(x.len(), c.major, "spmv: x has wrong length");
    assert_eq!(y.len(), c.minor, "spmv: y has wrong length");
    scale_y(beta, y);
    for (&xk, w) in x.iter().zip(c.indptr.windows(2)) {
        let a = alpha * xk;
        let (s, e) = (w[0], w[1]);
        for (&i, &v) in c.indices[s..e].iter().zip(&c.data[s..e]) {
            // SAFETY: `c` is validated: every index < minor == y.len()
            // (checked above).
            let yi = unsafe { y.get_unchecked_mut(i as usize) };
            *yi = *yi + v * a;
        }
    }
}

fn gather_dispatch<T: Real>(
    s: RowStrategy,
    c: &Compressed<'_, T>,
    alpha: T,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    match s {
        RowStrategy::Scalar => gather_lines::<T, 1, { usize::MAX }>(c, alpha, x, beta, y),
        RowStrategy::Lanes4 => gather_lines::<T, 4, 0>(c, alpha, x, beta, y),
        RowStrategy::Lanes8 => gather_lines::<T, 8, 0>(c, alpha, x, beta, y),
        RowStrategy::Hybrid => gather_lines::<T, 8, 8>(c, alpha, x, beta, y),
    }
}

/// `y = alpha * A * x + beta * y` for a CSR matrix `A` (`nrows x ncols`).
///
/// If `beta == 0`, `y` is overwritten without being read, so NaN or
/// infinity already in `y` does not leak into the result. `alpha == 0` is
/// **not** special-cased (`0 * NaN` stays NaN).
///
/// Rows are reduced with [`DEFAULT_STRATEGY`]; the summation order within a
/// row is fixed and independent of the target, so results are
/// bit-reproducible.
///
/// # Panics
/// If `x.len() != ncols` or `y.len() != nrows`.
pub fn spmv_csr<T: Real>(alpha: T, a: &CsrView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    gather_dispatch(DEFAULT_STRATEGY, &a.c, alpha, x, beta, y);
}

/// [`spmv_csr`] with an explicit [`RowStrategy`] (for benchmarking and
/// experiments).
///
/// # Panics
/// If `x.len() != ncols` or `y.len() != nrows`.
pub fn spmv_csr_with<T: Real>(
    strategy: RowStrategy,
    alpha: T,
    a: &CsrView<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    gather_dispatch(strategy, &a.c, alpha, x, beta, y);
}

/// `y = alpha * Aᵀ * x + beta * y` for a CSR matrix `A` (`nrows x ncols`),
/// so `x` has `nrows` and `y` has `ncols` elements. A scatter kernel: `y` is
/// scaled by `beta` first (overwritten with zeros when `beta == 0`), then
/// each row adds `a_ij * (alpha * x_i)` into `y[j]`.
///
/// # Panics
/// If `x.len() != nrows` or `y.len() != ncols`.
pub fn spmv_csr_t<T: Real>(alpha: T, a: &CsrView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    scatter_lines(&a.c, alpha, x, beta, y);
}

/// `y = alpha * A * x + beta * y` for a CSC matrix `A` (`nrows x ncols`):
/// the scatter kernel, one column at a time. Same `beta == 0` rule as
/// [`spmv_csr`].
///
/// # Panics
/// If `x.len() != ncols` or `y.len() != nrows`.
pub fn spmv_csc<T: Real>(alpha: T, a: &CscView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    scatter_lines(&a.c, alpha, x, beta, y);
}

/// `y = alpha * Aᵀ * x + beta * y` for a CSC matrix `A`: one sparse dot per
/// column (the same kernel as [`spmv_csr`]).
///
/// # Panics
/// If `x.len() != nrows` or `y.len() != ncols`.
pub fn spmv_csc_t<T: Real>(alpha: T, a: &CscView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    gather_dispatch(DEFAULT_STRATEGY, &a.c, alpha, x, beta, y);
}
