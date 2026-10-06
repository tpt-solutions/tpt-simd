//! SIMD reductions over slices: sums, compensated sums, mean, variance,
//! covariance, min/max/argmin/argmax and norms for `f32`/`f64` (plus `i32`
//! sums and extrema).
//!
//! Built on [`tpt_simd_core::Simd`] and the fixed halving tree of
//! [`tpt_simd_horizontal`]. Everything is safe, portable code; LLVM maps it
//! to vector instructions for whatever target features the build enables
//! (ADR 0001: compile-time dispatch, so build with `-C target-cpu=native`
//! or equivalent for AVX2 speed).
//!
//! Every function exists as `<op>_f32` and `<op>_f64`; integer variants are
//! suffixed `_i32`.
//!
//! | Function | Meaning |
//! |---|---|
//! | `sum_*` | multi-accumulator vectorised sum |
//! | `pairwise_sum_*` | recursive blocked sum, O(log n) error growth |
//! | `sum_compensated_*` | vectorised Neumaier/TwoSum sum, error ~ 1 ulp of the result for sane inputs |
//! | `mean_*` | arithmetic mean (pairwise) |
//! | `variance_*`, `sample_variance_*` | two-pass population / sample variance |
//! | `variance_welford_*` | scalar single-pass Welford (population) |
//! | `covariance_*`, `sample_covariance_*` | two-pass population / sample covariance |
//! | `min_*`, `max_*`, `argmin_*`, `argmax_*` | extrema, NaN-ignoring |
//! | `sum_squares_*`, `norm_*` | sum of squares and its square root |
//! | `sum_i32`, `sum_i32_wide`, `min_i32`, `max_i32`, `argmin_i32`, `argmax_i32` | integer variants |
//!
//! ## Accumulation order (determinism)
//!
//! All orders are fixed by the code and identical on every target and
//! feature set, so results are bit-for-bit reproducible (no FMA is used).
//! Let `N` be the lane count (`8` for `f32`, `4` for `f64`) and `B = 4N`.
//!
//! 1. The slice is cut into blocks of `B` elements. Element `k*N + j` of a
//!    block is added to lane `j` of accumulator `k` (`k = 0..4`).
//! 2. The remaining whole `N`-element vectors are added, in order, to
//!    accumulator 0.
//! 3. The accumulators combine as `(a0 + a1) + (a2 + a3)`, then the lanes
//!    reduce with the halving tree of [`tpt_simd_horizontal::reduce_sum`].
//! 4. The final `< N` tail elements are added to that result one at a time,
//!    left to right.
//!
//! [`pairwise_sum_f32`] splits slices longer than 1024 elements in half
//! (`len / 2`) recursively and uses the order above for each leaf.
//!
//! ## Tolerance versus a scalar reference
//!
//! A left-to-right scalar loop has worst-case relative error `(n-1)·ε`
//! against `Σ|x|`; the multi-accumulator sum has roughly
//! `(n/(4N) + log2(4N) + N)·ε`, the pairwise sum `~log2(n)·ε`, and the
//! compensated sum `~ε + O(n·ε²)`. (`ε` is `2^-24` for `f32`, `2^-53` for
//! `f64`.) Results therefore differ from a scalar loop by a few ulps of
//! `Σ|x|`; compare with a tolerance relative to `Σ|x|`, not to the result,
//! when terms cancel. `sum_compensated_*` is the one to use when
//! cancellation matters (see the example above).
//!
//! ## NaN and infinity policy
//!
//! * Sums, means, variances, covariances and norms propagate NaN and
//!   infinity as ordinary IEEE arithmetic would (`inf - inf = NaN`).
//! * `min_*` / `max_*` **ignore NaN** (same as `tpt-simd-horizontal`): the
//!   result is the extremum of the non-NaN elements, and `Some(NaN)` if
//!   every element is NaN. The sign of a zero result is unspecified if both
//!   `+0.0` and `-0.0` are present.
//! * `argmin_*` / `argmax_*` return the index of the **first** element equal
//!   to the extremum, and `None` if the slice is empty or all NaN.
//!
//! ## Empty slices and panics
//!
//! Functions with no sensible value on empty input return `Option`
//! (`mean`, `variance`, `covariance`, `min`, `max`, `argmin`, `argmax`;
//! sample variants need at least two elements). Sums of an empty slice are
//! `0`. Mismatched slice lengths in `covariance` panic (ADR 0002).
//!
//! # Examples
//! ```
//! use tpt_simd_reduce::*;
//! let xs = [1.0f32, 2.0, 3.0, 4.0];
//! assert_eq!(sum_f32(&xs), 10.0);
//! assert_eq!(mean_f32(&xs), Some(2.5));
//! assert_eq!(variance_f32(&xs), Some(1.25));
//! assert_eq!(argmax_f32(&[1.0, f32::NAN, 7.0, 3.0]), Some(2));
//! assert_eq!(min_f32(&[]), None);
//! // Compensated summation survives catastrophic cancellation:
//! let big = [1.0e8f32, 1.0, -1.0e8];
//! assert_eq!(sum_f32(&big), 0.0);
//! assert_eq!(sum_compensated_f32(&big), 1.0);
//! ```
#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(test)]
mod tests;

macro_rules! float_kernels {
    ($m:ident, $t:ident, $n:literal) => {
        pub(crate) mod $m {
            use tpt_simd_core::Simd;
            use tpt_simd_horizontal::reduce_sum;

            type V = Simd<$t, $n>;
            const N: usize = $n;
            const BLOCK: usize = 4 * N;
            const PAIR_BASE: usize = 1024;

            /// Generic accumulate: `sum f(x[i], y[i])` in the documented order.
            #[inline(always)]
            fn sum_map2(
                xs: &[$t],
                ys: &[$t],
                mv: impl Fn(V, V) -> V,
                ms: impl Fn($t, $t) -> $t,
            ) -> $t {
                debug_assert_eq!(xs.len(), ys.len());
                let mut a = [V::splat(0.0); 4];
                let mut xb = xs.chunks_exact(BLOCK);
                let mut yb = ys.chunks_exact(BLOCK);
                for (x, y) in (&mut xb).zip(&mut yb) {
                    for k in 0..4 {
                        a[k] += mv(V::from_slice(&x[k * N..]), V::from_slice(&y[k * N..]));
                    }
                }
                let mut xv = xb.remainder().chunks_exact(N);
                let mut yv = yb.remainder().chunks_exact(N);
                for (x, y) in (&mut xv).zip(&mut yv) {
                    a[0] += mv(V::from_slice(x), V::from_slice(y));
                }
                let mut r = reduce_sum((a[0] + a[1]) + (a[2] + a[3]));
                for (&x, &y) in xv.remainder().iter().zip(yv.remainder()) {
                    r += ms(x, y);
                }
                r
            }

            #[inline(always)]
            fn sum_map(xs: &[$t], mv: impl Fn(V) -> V, ms: impl Fn($t) -> $t) -> $t {
                sum_map2(xs, xs, |a, _| mv(a), |a, _| ms(a))
            }

            /// Multi-accumulator sum (see the crate docs for the exact order).
            #[inline]
            pub fn sum(xs: &[$t]) -> $t {
                sum_map(xs, |v| v, |x| x)
            }

            /// Recursive pairwise sum: halves the slice until it has at most
            /// 1024 elements, sums each leaf with the multi-accumulator
            /// kernel and adds the halves. Worst-case error grows like
            /// `log2(n)·ε` instead of `n·ε`.
            #[inline]
            pub fn pairwise_sum(xs: &[$t]) -> $t {
                if xs.len() <= PAIR_BASE {
                    sum(xs)
                } else {
                    let (a, b) = xs.split_at(xs.len() / 2);
                    pairwise_sum(a) + pairwise_sum(b)
                }
            }

            /// Branch-free Knuth TwoSum: returns `(s + x, rounding error)`.
            #[inline(always)]
            fn two_sum_v(s: V, x: V) -> (V, V) {
                let t = s + x;
                let bp = t - s;
                (t, (s - (t - bp)) + (x - bp))
            }

            #[inline(always)]
            fn two_sum(s: $t, x: $t) -> ($t, $t) {
                let t = s + x;
                let bp = t - s;
                (t, (s - (t - bp)) + (x - bp))
            }

            /// Compensated (Neumaier/TwoSum) sum, vectorised.
            ///
            /// Every lane keeps a running sum and a running error term using
            /// an exact TwoSum; the lanes and tail are merged with a scalar
            /// Neumaier loop. The result is accurate to about one ulp of the
            /// exact sum for any ordering of the data (unlike the plain sum,
            /// whose error scales with the sum of absolute values). If the
            /// compensated result is not finite (an infinity or NaN input, or
            /// overflow) it falls back to the plain sum so IEEE special-value
            /// semantics are kept. Roughly 3-4x the arithmetic of the plain sum.
            #[inline]
            pub fn sum_compensated(xs: &[$t]) -> $t {
                let mut s = [V::splat(0.0); 4];
                let mut c = [V::splat(0.0); 4];
                let mut blocks = xs.chunks_exact(BLOCK);
                for b in &mut blocks {
                    for k in 0..4 {
                        let (t, e) = two_sum_v(s[k], V::from_slice(&b[k * N..]));
                        s[k] = t;
                        c[k] += e;
                    }
                }
                let mut vecs = blocks.remainder().chunks_exact(N);
                for v in &mut vecs {
                    let (t, e) = two_sum_v(s[0], V::from_slice(v));
                    s[0] = t;
                    c[0] += e;
                }
                let mut total: $t = 0.0;
                let mut comp: $t = 0.0;
                for sv in &s {
                    for &l in sv.as_array() {
                        let (t, e) = two_sum(total, l);
                        total = t;
                        comp += e;
                    }
                }
                for &x in vecs.remainder() {
                    let (t, e) = two_sum(total, x);
                    total = t;
                    comp += e;
                }
                for cv in &c {
                    for &l in cv.as_array() {
                        comp += l;
                    }
                }
                let r = total + comp;
                if r.is_finite() { r } else { sum(xs) }
            }

            /// Arithmetic mean (pairwise sum divided by the length); `None` if empty.
            #[inline]
            pub fn mean(xs: &[$t]) -> Option<$t> {
                if xs.is_empty() {
                    None
                } else {
                    Some(pairwise_sum(xs) / xs.len() as $t)
                }
            }

            /// Sum of squared deviations from the (pairwise) mean.
            #[inline]
            fn m2(xs: &[$t]) -> Option<$t> {
                let m = mean(xs)?;
                let mv = V::splat(m);
                Some(sum_map(
                    xs,
                    |v| {
                        let d = v - mv;
                        d * d
                    },
                    |x| {
                        let d = x - m;
                        d * d
                    },
                ))
            }

            /// Population variance `sum((x-mean)^2)/n`, two-pass (mean first,
            /// then squared deviations with the multi-accumulator kernel).
            /// Numerically robust for data whose mean is large relative to
            /// its spread. `None` if empty.
            #[inline]
            pub fn variance(xs: &[$t]) -> Option<$t> {
                Some(m2(xs)? / xs.len() as $t)
            }

            /// Sample variance `sum((x-mean)^2)/(n-1)`, two-pass. `None` if `len < 2`.
            #[inline]
            pub fn sample_variance(xs: &[$t]) -> Option<$t> {
                if xs.len() < 2 {
                    return None;
                }
                Some(m2(xs)? / (xs.len() - 1) as $t)
            }

            /// Population variance by scalar single-pass Welford updating.
            /// Streaming-friendly and stable, but serial (not vectorised);
            /// use it as a reference or when the data cannot be revisited.
            /// `None` if empty.
            #[inline]
            pub fn variance_welford(xs: &[$t]) -> Option<$t> {
                if xs.is_empty() {
                    return None;
                }
                let mut mean: $t = 0.0;
                let mut m2: $t = 0.0;
                let mut n: $t = 0.0;
                for &x in xs {
                    n += 1.0;
                    let d = x - mean;
                    mean += d / n;
                    m2 += d * (x - mean);
                }
                Some(m2 / n)
            }

            #[inline]
            fn cross(xs: &[$t], ys: &[$t]) -> Option<$t> {
                assert_eq!(xs.len(), ys.len(), "covariance: slice length mismatch");
                let mx = mean(xs)?;
                let my = mean(ys)?;
                let (vx, vy) = (V::splat(mx), V::splat(my));
                Some(sum_map2(
                    xs,
                    ys,
                    |a, b| (a - vx) * (b - vy),
                    |a, b| (a - mx) * (b - my),
                ))
            }

            /// Population covariance `sum((x-mx)(y-my))/n`, two-pass.
            /// `None` if empty.
            ///
            /// # Panics
            /// If `xs.len() != ys.len()`.
            #[inline]
            pub fn covariance(xs: &[$t], ys: &[$t]) -> Option<$t> {
                Some(cross(xs, ys)? / xs.len() as $t)
            }

            /// Sample covariance `sum((x-mx)(y-my))/(n-1)`. `None` if `len < 2`.
            ///
            /// # Panics
            /// If `xs.len() != ys.len()`.
            #[inline]
            pub fn sample_covariance(xs: &[$t], ys: &[$t]) -> Option<$t> {
                assert_eq!(xs.len(), ys.len(), "covariance: slice length mismatch");
                if xs.len() < 2 {
                    return None;
                }
                Some(cross(xs, ys)? / (xs.len() - 1) as $t)
            }
            /// Plain fold: LLVM already turns the NaN-ignoring `min`/`max`
            /// fold into vector min/max with several accumulators, and beats
            /// every hand-blocked variant we measured (see the benches).
            #[inline(always)]
            fn extreme(xs: &[$t], sf: impl Fn($t, $t) -> $t) -> Option<$t> {
                if xs.is_empty() {
                    return None;
                }
                Some(xs.iter().copied().fold($t::NAN, sf))
            }

            /// Minimum, ignoring NaN; `Some(NaN)` if all NaN, `None` if empty.
            #[inline]
            pub fn min(xs: &[$t]) -> Option<$t> {
                extreme(xs, |a, b| a.min(b))
            }

            /// Maximum, ignoring NaN; `Some(NaN)` if all NaN, `None` if empty.
            #[inline]
            pub fn max(xs: &[$t]) -> Option<$t> {
                extreme(xs, |a, b| a.max(b))
            }

            /// Index of the first minimum, ignoring NaN; `None` if empty or all NaN.
            #[inline]
            pub fn argmin(xs: &[$t]) -> Option<usize> {
                let m = min(xs)?;
                xs.iter().position(|&x| x == m)
            }

            /// Index of the first maximum, ignoring NaN; `None` if empty or all NaN.
            #[inline]
            pub fn argmax(xs: &[$t]) -> Option<usize> {
                let m = max(xs)?;
                xs.iter().position(|&x| x == m)
            }

            /// Sum of squares (unfused multiply-add; multi-accumulator order).
            #[inline]
            pub fn sum_squares(xs: &[$t]) -> $t {
                sum_map(xs, |v| v * v, |x| x * x)
            }

            /// Euclidean norm `sqrt(sum of squares)`. No overflow/underflow
            /// scaling: intermediate squares overflow for `|x|` above about
            /// `sqrt(MAX)`.
            #[inline]
            pub fn norm(xs: &[$t]) -> $t {
                Simd::<$t, 1>::splat(sum_squares(xs)).sqrt().to_array()[0]
            }
        }
    };
}

float_kernels!(f32k, f32, 8);
float_kernels!(f64k, f64, 4);

pub use f32k::{
    argmax as argmax_f32, argmin as argmin_f32, covariance as covariance_f32, max as max_f32,
    mean as mean_f32, min as min_f32, norm as norm_f32, pairwise_sum as pairwise_sum_f32,
    sample_covariance as sample_covariance_f32, sample_variance as sample_variance_f32,
    sum as sum_f32, sum_compensated as sum_compensated_f32, sum_squares as sum_squares_f32,
    variance as variance_f32, variance_welford as variance_welford_f32,
};
pub use f64k::{
    argmax as argmax_f64, argmin as argmin_f64, covariance as covariance_f64, max as max_f64,
    mean as mean_f64, min as min_f64, norm as norm_f64, pairwise_sum as pairwise_sum_f64,
    sample_covariance as sample_covariance_f64, sample_variance as sample_variance_f64,
    sum as sum_f64, sum_compensated as sum_compensated_f64, sum_squares as sum_squares_f64,
    variance as variance_f64, variance_welford as variance_welford_f64,
};

mod int {

    /// Wrapping sum of all elements (order is irrelevant for wrapping adds).
    /// A plain fold: LLVM vectorises it better than a hand-blocked version.
    #[inline]
    pub fn sum_i32(xs: &[i32]) -> i32 {
        xs.iter().fold(0i32, |a, &x| a.wrapping_add(x))
    }

    /// Exact sum widened to `i64` (cannot overflow for slices shorter than 2^32).
    #[inline]
    pub fn sum_i32_wide(xs: &[i32]) -> i64 {
        let mut a = [0i64; 8];
        let mut c = xs.chunks_exact(8);
        for ch in &mut c {
            for k in 0..8 {
                a[k] += i64::from(ch[k]);
            }
        }
        let mut r: i64 = a.iter().sum();
        for &x in c.remainder() {
            r += i64::from(x);
        }
        r
    }

    #[inline(always)]
    fn extreme(xs: &[i32], init: i32, sf: impl Fn(i32, i32) -> i32) -> Option<i32> {
        if xs.is_empty() {
            return None;
        }
        Some(xs.iter().copied().fold(init, sf))
    }

    /// Minimum; `None` if empty.
    #[inline]
    pub fn min_i32(xs: &[i32]) -> Option<i32> {
        extreme(xs, i32::MAX, i32::min)
    }

    /// Maximum; `None` if empty.
    #[inline]
    pub fn max_i32(xs: &[i32]) -> Option<i32> {
        extreme(xs, i32::MIN, i32::max)
    }

    /// Index of the first minimum; `None` if empty.
    #[inline]
    pub fn argmin_i32(xs: &[i32]) -> Option<usize> {
        let m = min_i32(xs)?;
        xs.iter().position(|&x| x == m)
    }

    /// Index of the first maximum; `None` if empty.
    #[inline]
    pub fn argmax_i32(xs: &[i32]) -> Option<usize> {
        let m = max_i32(xs)?;
        xs.iter().position(|&x| x == m)
    }
}

pub use int::{argmax_i32, argmin_i32, max_i32, min_i32, sum_i32, sum_i32_wide};
