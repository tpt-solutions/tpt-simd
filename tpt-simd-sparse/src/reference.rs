//! Naive scalar reference implementations (single accumulator, safe
//! indexing). Used by the tests and benchmarks as ground truth.

use crate::real::Real;
use crate::view::{CscView, CsrView};

/// Reference `y = alpha * A * x + beta * y` (CSR); `beta == 0` overwrites.
///
/// # Panics
/// On length mismatch.
pub fn spmv_csr<T: Real>(alpha: T, a: &CsrView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    assert_eq!(x.len(), a.ncols());
    assert_eq!(y.len(), a.nrows());
    for r in 0..a.nrows() {
        let mut t = T::ZERO;
        for k in a.indptr()[r]..a.indptr()[r + 1] {
            t = t + a.data()[k] * x[a.indices()[k] as usize];
        }
        y[r] = if beta == T::ZERO {
            alpha * t
        } else {
            alpha * t + beta * y[r]
        };
    }
}

/// Reference `y = alpha * Aᵀ * x + beta * y` (CSR).
///
/// # Panics
/// On length mismatch.
pub fn spmv_csr_t<T: Real>(alpha: T, a: &CsrView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    assert_eq!(x.len(), a.nrows());
    assert_eq!(y.len(), a.ncols());
    for v in y.iter_mut() {
        *v = if beta == T::ZERO { T::ZERO } else { beta * *v };
    }
    for r in 0..a.nrows() {
        for k in a.indptr()[r]..a.indptr()[r + 1] {
            let j = a.indices()[k] as usize;
            y[j] = y[j] + alpha * (a.data()[k] * x[r]);
        }
    }
}

/// Reference `y = alpha * A * x + beta * y` (CSC).
///
/// # Panics
/// On length mismatch.
pub fn spmv_csc<T: Real>(alpha: T, a: &CscView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    assert_eq!(x.len(), a.ncols());
    assert_eq!(y.len(), a.nrows());
    for v in y.iter_mut() {
        *v = if beta == T::ZERO { T::ZERO } else { beta * *v };
    }
    for c in 0..a.ncols() {
        for k in a.indptr()[c]..a.indptr()[c + 1] {
            let i = a.indices()[k] as usize;
            y[i] = y[i] + alpha * (a.data()[k] * x[c]);
        }
    }
}

/// Reference `y = alpha * Aᵀ * x + beta * y` (CSC).
///
/// # Panics
/// On length mismatch.
pub fn spmv_csc_t<T: Real>(alpha: T, a: &CscView<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    assert_eq!(x.len(), a.nrows());
    assert_eq!(y.len(), a.ncols());
    for c in 0..a.ncols() {
        let mut t = T::ZERO;
        for k in a.indptr()[c]..a.indptr()[c + 1] {
            t = t + a.data()[k] * x[a.indices()[k] as usize];
        }
        y[c] = if beta == T::ZERO {
            alpha * t
        } else {
            alpha * t + beta * y[c]
        };
    }
}

/// Reference dot product (single accumulator, left to right).
///
/// # Panics
/// On length mismatch.
#[must_use]
pub fn dot<T: Real>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len());
    let mut t = T::ZERO;
    for (&a, &b) in x.iter().zip(y) {
        t = t + a * b;
    }
    t
}

/// Reference `y += alpha * x`.
///
/// # Panics
/// On length mismatch.
pub fn axpy<T: Real>(alpha: T, x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len());
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = *yi + alpha * xi;
    }
}
