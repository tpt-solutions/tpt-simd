//! Dense vector kernels for Krylov solvers: single-pass fused updates with
//! several independent accumulators.

use crate::real::Real;

/// Lanes per accumulator vector.
const W: usize = 8;
/// Independent accumulator vectors.
const U: usize = 4;
/// Elements per unrolled block.
const B: usize = W * U;

type Acc<T> = [[T; W]; U];

#[inline(always)]
fn combine<T: Real>(acc: Acc<T>) -> T {
    let mut v = [T::ZERO; W];
    for j in 0..W {
        v[j] = (acc[0][j] + acc[1][j]) + (acc[2][j] + acc[3][j]);
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
    let mut acc: Acc<T> = [[T::ZERO; W]; U];
    let (xc, yc) = (x.chunks_exact(B), y.chunks_exact(B));
    let (xt, yt) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        for k in 0..U {
            for j in 0..W {
                acc[k][j] = acc[k][j] + a[k * W + j] * b[k * W + j];
            }
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
    let mut acc: Acc<T> = [[T::ZERO; W]; U];
    let n = y.len() / B * B;
    let (yb, yt) = y.split_at_mut(n);
    for ((yc, xc), zc) in yb.chunks_exact_mut(B).zip(x.chunks_exact(B)).zip(z.chunks_exact(B)) {
        for k in 0..U {
            for j in 0..W {
                let i = k * W + j;
                let ynew = yc[i] + alpha * xc[i];
                yc[i] = ynew;
                acc[k][j] = acc[k][j] + ynew * zc[i];
            }
        }
    }
    let mut t = combine(acc);
    for ((yi, &xi), &zi) in yt.iter_mut().zip(&x[n..]).zip(&z[n..]) {
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
    let mut acc: Acc<T> = [[T::ZERO; W]; U];
    let n = y.len() / B * B;
    let (yb, yt) = y.split_at_mut(n);
    for (yc, xc) in yb.chunks_exact_mut(B).zip(x.chunks_exact(B)) {
        for k in 0..U {
            for j in 0..W {
                let i = k * W + j;
                let ynew = yc[i] + alpha * xc[i];
                yc[i] = ynew;
                acc[k][j] = acc[k][j] + ynew * ynew;
            }
        }
    }
    let mut t = combine(acc);
    for (yi, &xi) in yt.iter_mut().zip(&x[n..]) {
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
    let mut acc: Acc<T> = [[T::ZERO; W]; U];
    let n = x.len() / B * B;
    let (xb, xt) = x.split_at_mut(n);
    let (rb, rt) = r.split_at_mut(n);
    for (((xc, rc), pc), qc) in xb
        .chunks_exact_mut(B)
        .zip(rb.chunks_exact_mut(B))
        .zip(p.chunks_exact(B))
        .zip(q.chunks_exact(B))
    {
        for k in 0..U {
            for j in 0..W {
                let i = k * W + j;
                xc[i] = xc[i] + alpha * pc[i];
                let rnew = rc[i] - alpha * qc[i];
                rc[i] = rnew;
                acc[k][j] = acc[k][j] + rnew * rnew;
            }
        }
    }
    let mut t = combine(acc);
    for (((xi, ri), &pi), &qi) in xt.iter_mut().zip(rt.iter_mut()).zip(&p[n..]).zip(&q[n..]) {
        *xi = *xi + alpha * pi;
        *ri = *ri - alpha * qi;
        t = t + *ri * *ri;
    }
    t
}
