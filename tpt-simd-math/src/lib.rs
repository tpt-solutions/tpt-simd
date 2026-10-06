//! Vectorised `f32` elementary functions: [`exp`](exp_f32), [`ln`](ln_f32),
//! [`sin`](sin_f32), [`cos`](cos_f32), [`tanh`](tanh_f32) and
//! [`erf`](erf_f32).
//!
//! Each function is implemented as a branch-free scalar kernel (range
//! reduction + polynomial, every conditional a value select) applied across
//! 8 lanes. Only plain `+ - * /` are used: no per-lane `libm` calls, no
//! `mul_add`. LLVM turns the 8-lane loops into straight vector code for
//! whatever target features the build enables (ADR 0001: compile-time
//! dispatch; build with `-C target-cpu=native` for AVX2 speed). Results are
//! **bit-identical on every target and feature set** (no FMA contraction),
//! and identical between the scalar, `Simd<f32, 8>` and slice entry points,
//! including slice tails.
//!
//! For every function `f` there are four entry points:
//!
//! | Entry point | Signature |
//! |---|---|
//! | `f_scalar` | `fn(f32) -> f32` (the shared kernel; one lane) |
//! | `f_f32x8` | `fn(Simd<f32, 8>) -> Simd<f32, 8>` |
//! | `f_f32` | `fn(&[f32], &mut [f32])` (panics if lengths differ) |
//! | `f_f32_inplace` | `fn(&mut [f32])` |
//!
//! ```
//! use tpt_simd_math::{exp_f32, exp_f32x8};
//! use tpt_simd_core::Simd;
//!
//! let x = [0.0f32, 1.0, -1.0, 10.0, 0.5, 2.0, -20.0, 88.0, 3.0];
//! let mut y = [0.0f32; 9];
//! exp_f32(&x, &mut y);
//! assert_eq!(y[0], 1.0);
//! assert!((y[1] - core::f32::consts::E).abs() < 1e-6);
//!
//! let v = exp_f32x8(Simd::splat(0.0));
//! assert_eq!(v.to_array(), [1.0; 8]);
//! ```
//!
//! ## Accuracy (Tier 2, ADR 0003)
//!
//! Accuracy is a documented ULP bound, not bit equality with `libm` or
//! `std`. "ULP" below is the error divided by the spacing of `f32` at the
//! exact (`f64`-computed) result, so 0.5 would be a correctly rounded
//! result. Maxima were measured by `cargo test` over dense sweeps and
//! random bit patterns (see `src/tests.rs`, which asserts these bounds):
//!
//! | Function | Range | Max ULP measured | Bound asserted |
//! |---|---|---|---|
//! | `exp` | `[-87.3, 88.72]` (normal results) | EXP_NORMAL | EXP_NORMAL_B |
//! | `exp` | `[-103.97, -87.3]` (subnormal results) | EXP_SUB | EXP_SUB_B |
//! | `ln` | all positive finite, incl. subnormals | LN | LN_B |
//! | `sin`, `cos` | `abs(x) <= 8192` | TRIG | TRIG_B |
//! | `sin`, `cos` | `8192 < abs(x) <= 100000` | TRIG_BIG | TRIG_BIG_B |
//! | `tanh` | all finite | TANH | TANH_B |
//! | `erf` | all finite | ERF | ERF_B |
//!
//! `sin`/`cos` are relative-error limited near the zeros of the function
//! (the reduced argument carries an absolute error of a few `1e-8`); the
//! absolute error is below `2e-7` over `abs(x) <= 8192`. The ULP numbers
//! above are measured against the exact value over the whole range, zeros
//! included.
//!
//! ## Special values
//!
//! | Input | `exp` | `ln` | `sin` / `cos` | `tanh` | `erf` |
//! |---|---|---|---|---|---|
//! | NaN | NaN | NaN | NaN | NaN | NaN |
//! | `+inf` | `+inf` | `+inf` | NaN | `1` | `1` |
//! | `-inf` | `0` | NaN | NaN | `-1` | `-1` |
//! | `+0` / `-0` | `1` | `-inf` | `sin`: `+0` / `-0`, `cos`: `1` | `+0` / `-0` | `+0` / `-0` |
//! | negative | n/a | NaN | n/a | n/a | n/a |
//! | subnormal | `1` | exact-ish (see table) | `sin`: `x`; `cos`: `1` | `x` | `~1.128 x` |
//!
//! * `exp(x)` is `+inf` for `x > 88.7228` (the first `f32` overflow
//!   threshold, `ln(f32::MAX)`), and underflows gradually through the
//!   subnormals to `+0` for `x < -103.97`.
//! * `ln(+0)` and `ln(-0)` are `-inf`; `ln(x)` is NaN for every `x < 0`
//!   (including `-inf`). `ln(1)` is exactly `0`.
//! * `sin`/`cos` return NaN for `abs(x) > 100000` (outside the supported
//!   range, no Payne-Hanek reduction) as well as for NaN and infinities.
//! * A NaN input is returned unchanged (payload and sign preserved).
//! * `tanh` and `erf` are odd functions: the sign of zero and of the result
//!   follows the input.
//!
//! ## Cost note
//!
//! `tanh` and `erf` evaluate several polynomials and an `exp` and select;
//! the cost is the sum of all branches, which is what branch-free code
//! costs. See `benches/math.rs` and `docs/benchmarks.md` for measured speed
//! versus `libm` and `std`.
#![no_std]
#![forbid(unsafe_code)]

#[cfg(any(feature = "std", test))]
extern crate std;

mod kernels;

pub use tpt_simd_core::Simd;

/// Lanes per vector for the `f32x8` entry points.
const LANES: usize = 8;

#[inline(always)]
fn apply8(x: [f32; LANES], f: impl Fn(f32) -> f32) -> [f32; LANES] {
    let mut o = [0.0f32; LANES];
    for i in 0..LANES {
        o[i] = f(x[i]);
    }
    o
}

#[inline(always)]
fn map_slice(x: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32 + Copy) {
    assert_eq!(x.len(), out.len(), "input and output lengths must match");
    let mut xc = x.chunks_exact(LANES);
    let mut oc = out.chunks_exact_mut(LANES);
    for (xi, oi) in (&mut xc).zip(&mut oc) {
        let mut v = [0.0f32; LANES];
        v.copy_from_slice(xi);
        oi.copy_from_slice(&apply8(v, f));
    }
    let (xr, or) = (xc.remainder(), oc.into_remainder());
    if !xr.is_empty() {
        let mut v = [0.0f32; LANES];
        v[..xr.len()].copy_from_slice(xr);
        let r = apply8(v, f);
        or.copy_from_slice(&r[..xr.len()]);
    }
}

#[inline(always)]
fn map_slice_inplace(buf: &mut [f32], f: impl Fn(f32) -> f32 + Copy) {
    let mut chunks = buf.chunks_exact_mut(LANES);
    for c in &mut chunks {
        let mut v = [0.0f32; LANES];
        v.copy_from_slice(c);
        c.copy_from_slice(&apply8(v, f));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
        let mut v = [0.0f32; LANES];
        v[..rem.len()].copy_from_slice(rem);
        let r = apply8(v, f);
        rem.copy_from_slice(&r[..rem.len()]);
    }
}

macro_rules! api {
    ($kernel:ident, $scalar:ident, $simd:ident, $slice:ident, $inplace:ident, $what:literal) => {
        #[doc = concat!("Scalar `", $what, "`: the single-lane kernel shared by every entry point.")]
        #[doc = ""]
        #[doc = "Branch-free; see the crate docs for accuracy and special values."]
        #[inline]
        pub fn $scalar(x: f32) -> f32 {
            kernels::$kernel(x)
        }

        #[doc = concat!("`", $what, "` of 8 lanes.")]
        #[doc = ""]
        #[doc = "Bit-identical to applying the scalar kernel to each lane."]
        #[inline]
        pub fn $simd(x: Simd<f32, 8>) -> Simd<f32, 8> {
            Simd::from_array(apply8(x.to_array(), kernels::$kernel))
        }

        #[doc = concat!("`out[i] = ", $what, "(x[i])` over a slice.")]
        #[doc = ""]
        #[doc = "Bit-identical to the scalar kernel for every element (tails included)."]
        #[doc = ""]
        #[doc = "# Panics"]
        #[doc = ""]
        #[doc = "Panics if `x.len() != out.len()`."]
        pub fn $slice(x: &[f32], out: &mut [f32]) {
            map_slice(x, out, kernels::$kernel);
        }

        #[doc = concat!("`buf[i] = ", $what, "(buf[i])` in place.")]
        #[doc = ""]
        #[doc = "Bit-identical to the scalar kernel for every element (tails included)."]
        pub fn $inplace(buf: &mut [f32]) {
            map_slice_inplace(buf, kernels::$kernel);
        }
    };
}

api!(exp, exp_scalar, exp_f32x8, exp_f32, exp_f32_inplace, "exp");
api!(ln, ln_scalar, ln_f32x8, ln_f32, ln_f32_inplace, "ln");
api!(sin, sin_scalar, sin_f32x8, sin_f32, sin_f32_inplace, "sin");
api!(cos, cos_scalar, cos_f32x8, cos_f32, cos_f32_inplace, "cos");
api!(
    tanh,
    tanh_scalar,
    tanh_f32x8,
    tanh_f32,
    tanh_f32_inplace,
    "tanh"
);
api!(erf, erf_scalar, erf_f32x8, erf_f32, erf_f32_inplace, "erf");

#[cfg(test)]
mod tests;
