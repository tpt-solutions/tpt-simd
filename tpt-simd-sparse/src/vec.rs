//! Dense vector kernels for Krylov solvers: single-pass fused updates with
//! several independent accumulators.

use crate::real::Real;

/// Lanes per accumulator vector.
const W: usize = 8;
/// Elements per unrolled block.
const B: usize = W * 4;

type Acc<T> = [T; B];

#[inline(always)]
fn combine<T: Real>(acc: Acc<T>) -> T {
    let mut v = [T::ZERO; W];
    for j in 0..W {
        v[j] = (acc[j] + acc[W + j]) + (acc[2 * W + j] + acc[3 * W + j]);
    }
    let mut w = W;
    while w > 1 {
        w /= 2;
        for j in 0..w {
            v[j] = v[j] + v[j + w];
        }
    }
    v[0]
}

#[inline(always)]
fn same_len(a: usize, b: usize, what: &str) {
    assert_eq!(a, b, "{what}: length mismatch ({a} vs {b})");
}

/// `x . y`.
///
/// Accumulation order is fixed (see the [crate docs](crate#accumulation-order)).
/// Empty input gives `0`.
///
/// # Panics
/// If the lengths differ.
#[must_use]
pub fn dot<T: Real>(x: &[T], y: &[T]) -> T {
    same_len(x.len(), y.len(), "dot");
    let mut acc: Acc<T> = [T::ZERO; B];
    let (xb, xt) = x.as_chunks::<B>();
    let (yb, yt) = y.as_chunks::<B>();
    for (a, b) in xb.iter().zip(yb) {
        for i in 0..B {
            acc[i] = acc[i] + a[i] * b[i];
        }
    }
    let mut t = combine(acc);
    for (&a, &b) in xt.iter().zip(yt) {
        t = t + a * b;
    }
    t
}

/// `x . x` (squared 2-norm).
#[must_use]
pub fn sqnorm<T: Real>(x: &[T]) -> T {
    dot(x, x)
}

/// `y += alpha * x`.
///
/// # Panics
/// If the lengths differ.
pub fn axpy<T: Real>(alpha: T, x: &[T], y: &mut [T]) {
    same_len(x.len(), y.len(), "axpy");
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = *yi + alpha * xi;
    }
}

/// `y = x + alpha * y` (the CG direction update `p = r + beta * p`).
///
/// # Panics
/// If the lengths differ.
pub fn xpay<T: Real>(x: &[T], alpha: T, y: &mut [T]) {
    same_len(x.len(), y.len(), "xpay");
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = xi + alpha * *yi;
    }
}

/// BiCGSTAB direction update `p = r + beta * (p - omega * v)`.
///
/// # Panics
/// If the lengths differ.
pub fn bicgstab_p_update<T: Real>(beta: T, omega: T, r: &[T], v: &[T], p: &mut [T]) {
    same_len(r.len(), p.len(), "bicgstab_p_update");
    same_len(v.len(), p.len(), "bicgstab_p_update");
    for ((pi, &ri), &vi) in p.iter_mut().zip(r).zip(v) {
        *pi = ri + beta * (*pi - omega * vi);
    }
}

/// Fused `y += alpha * x` and `y_new . z` in one pass over memory.
///
/// Returns the dot product of the **updated** `y` with `z`. Equivalent to
/// [`axpy`] followed by [`dot`], with identical accumulation order, so the
/// result is bit-identical to the unfused pair.
///
/// # Panics
/// If the lengths differ.
pub fn axpy_dot<T: Real>(alpha: T, x: &[T], y: &mut [T], z: &[T]) -> T {
    same_len(x.len(), y.len(), "axpy_dot");
    same_len(z.len(), y.len(), "axpy_dot");
    let mut acc: Acc<T> = [T::ZERO; B];
    let (yb, yt) = y.as_chunks_mut::<B>();
    let (xb, xt) = x.as_chunks::<B>();
    let (zb, zt) = z.as_chunks::<B>();
    for ((yc, xc), zc) in yb.iter_mut().zip(xb).zip(zb) {
        for i in 0..B {
            let ynew = yc[i] + alpha * xc[i];
            yc[i] = ynew;
            acc[i] = acc[i] + ynew * zc[i];
        }
    }
    let mut t = combine(acc);
    for ((yi, &xi), &zi) in yt.iter_mut().zip(xt).zip(zt) {
        *yi = *yi + alpha * xi;
        t = t + *yi * zi;
    }
    t
}

/// Fused `y += alpha * x` and `y_new . y_new` in one pass: the residual
/// update `r -= alpha * q` (pass `-alpha`) together with the new `r . r`.
///
/// Bit-identical to [`axpy`] followed by [`sqnorm`].
///
/// # Panics
/// If the lengths differ.
pub fn axpy_sqnorm<T: Real>(alpha: T, x: &[T], y: &mut [T]) -> T {
    same_len(x.len(), y.len(), "axpy_sqnorm");
    let mut acc: Acc<T> = [T::ZERO; B];
    let (yb, yt) = y.as_chunks_mut::<B>();
    let (xb, xt) = x.as_chunks::<B>();
    for (yc, xc) in yb.iter_mut().zip(xb) {
        for i in 0..B {
            let ynew = yc[i] + alpha * xc[i];
            yc[i] = ynew;
            acc[i] = acc[i] + ynew * ynew;
        }
    }
    let mut t = combine(acc);
    for (yi, &xi) in yt.iter_mut().zip(xt) {
        *yi = *yi + alpha * xi;
        t = t + *yi * *yi;
    }
    t
}

/// The complete vector part of a CG step in one pass:
/// `x += alpha * p`, `r -= alpha * q`, returns the new `r . r`.
///
/// Replaces three passes (two [`axpy`] and one [`sqnorm`]); the returned
/// value is bit-identical to computing them separately.
///
/// # Panics
/// If the lengths differ.
pub fn cg_update<T: Real>(alpha: T, p: &[T], q: &[T], x: &mut [T], r: &mut [T]) -> T {
    same_len(p.len(), x.len(), "cg_update");
    same_len(q.len(), x.len(), "cg_update");
    same_len(r.len(), x.len(), "cg_update");
    let mut acc: Acc<T> = [T::ZERO; B];
    let (xb, xt) = x.as_chunks_mut::<B>();
    let (rb, rt) = r.as_chunks_mut::<B>();
    let (pb, pt) = p.as_chunks::<B>();
    let (qb, qt) = q.as_chunks::<B>();
    for (((xc, rc), pc), qc) in xb.iter_mut().zip(rb.iter_mut()).zip(pb).zip(qb) {
        for i in 0..B {
            xc[i] = xc[i] + alpha * pc[i];
            let rnew = rc[i] - alpha * qc[i];
            rc[i] = rnew;
            acc[i] = acc[i] + rnew * rnew;
        }
    }
    let mut t = combine(acc);
    for (((xi, ri), &pi), &qi) in xt.iter_mut().zip(rt.iter_mut()).zip(pt).zip(qt) {
        *xi = *xi + alpha * pi;
        *ri = *ri - alpha * qi;
        t = t + *ri * *ri;
    }
    t
}
