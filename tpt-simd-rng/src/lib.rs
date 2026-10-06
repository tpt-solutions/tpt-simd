//! Lane-parallel pseudo-random numbers: eight xoshiro256++ streams (or eight
//! Philox4x32-10 blocks) advanced in lock step, with slice fills for `u32`,
//! `u64`, uniform `f32`/`f64` in `[0, 1)` and standard-normal `f32`/`f64`.
//!
//! **Not cryptographically secure.**
//!
//! Everything is safe, portable code on `[T; 8]` arrays (wrapping integer
//! ops, shifts, rotates, bit tricks); LLVM maps it to vector instructions for
//! whatever target features the build enables (ADR 0001: compile-time
//! dispatch, so build with `-C target-cpu=native` for AVX2 speed). The `std`
//! feature only forwards to `tpt-simd-core/std`; results are bit-identical
//! with and without it, and on every target (no FMA, no libm).
//!
//! ```
//! use tpt_simd_rng::{Rng8, Xoshiro256ppX8};
//!
//! let mut rng = Xoshiro256ppX8::from_seed(42);
//! let mut u = [0.0f32; 100];
//! rng.fill_f32(&mut u);
//! assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));
//! let mut z = [0.0f64; 100];
//! rng.fill_normal_f64(&mut z);
//! ```
//!
//! ## Generators
//!
//! * [`SplitMix64`]: seeding generator (any seed, including 0).
//! * [`Xoshiro256pp`]: scalar reference with [`jump`](Xoshiro256pp::jump)
//!   (`2^128` steps) and [`long_jump`](Xoshiro256pp::long_jump) (`2^192`).
//! * [`Xoshiro256ppX8`]: **8 independent `u64` streams** (not 8 x `u32`
//!   lanes), structure-of-arrays. Seeding: [`SplitMix64`] -> scalar state ->
//!   lane `i` is that state jumped `i` times, so lanes are `2^128` steps
//!   apart and cannot overlap in any realistic run; whole generators for
//!   parallel workers come from [`Xoshiro256ppX8::from_seed_block`].
//! * [`Philox4x32X8`]: counter-based; eight blocks per step; independent
//!   streams by `(key, stream id)`, random access by
//!   [`seek`](Philox4x32X8::seek).
//!
//! ## Fill semantics (the [`Rng8`] trait)
//!
//! Every fill consumes whole steps of 8 `u64` words (xoshiro: one step; the
//! partial last step's surplus is discarded). Consequently a fill of `n`
//! elements equals the first `n` elements of a longer fill from the same
//! generator state, and splitting one fill into several calls gives
//! different (but equally good) numbers unless every chunk length is a
//! multiple of the step size (8 for `u64`/`f64`, 16 for `u32`/`f32`/normals).
//!
//! | fill | step size | element from |
//! |---|---|---|
//! | `fill_u64` | 8 | one word |
//! | `fill_u32` | 16 | low halves of the 8 words, then high halves |
//! | `fill_f32` | 16 | a `u32` half `h`: `(h >> 8) * 2^-24`, 24 bits, in `[0, 1 - 2^-24]` |
//! | `fill_f64` | 8 | a word `w`: `from_bits(1.0 bits \| w >> 12) - 1.0`, 52 bits, in `[0, 1 - 2^-52]` |
//! | `fill_normal_f32` | 16 | Box-Muller pairs from the 16 halves |
//! | `fill_normal_f64` | 16 (two steps) | Box-Muller pairs |
//!
//! Uniform values are never 1.0. Note that `f64` uniforms carry 52 (not 53)
//! random bits, which is what lets the conversion vectorise on AVX2.
//!
//! ## Normal variates
//!
//! Box-Muller: `u1 = 1 - u` in `(0, 1]`, `u2 = u` in `[0, 1)`,
//! `(z0, z1) = (r cos 2 pi u2, r sin 2 pi u2)`, `r = sqrt(-2 ln u1)`, using
//! the branch-free polynomial approximations of [`math`] (accuracy table
//! there; end-to-end normal error is a few ulp, far below sampling noise).
//! The support is truncated at `|z| <= 5.77` for `f32` and `<= 8.5` for
//! `f64` (tail probability `~1e-8` and `~1e-17` beyond the cut). Per call the
//! output order is `z0` for the 8 lanes of a half step, then `z1`.
//!
//! ## Vectorisation
//!
//! The xoshiro step and the `u32`/`f32`/`f64` conversions compile to
//! `vpaddq`/`vpsllq`/`vpsrlq`/`vpxor`/`vcvtdq2ps` on AVX2 (64-bit rotates are
//! shift+or without AVX-512). See `docs/benchmarks.md`-style numbers in the
//! crate README / benches (`cargo bench -p tpt-simd-rng --bench rng`).

#![no_std]
#![forbid(unsafe_code)]

pub mod math;
mod philox;
mod xoshiro;

pub use philox::{Philox4x32X8, philox4x32_10};
pub use xoshiro::{SplitMix64, Xoshiro256pp, Xoshiro256ppX8};

use math::{box_muller_f32, box_muller_f64};

#[cfg(test)]
mod tests;

const INV_2_24: f32 = 1.0 / 16_777_216.0;

/// Uniform `f32` in `[0, 1)` (24 random bits) from a `u32`.
#[inline(always)]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    ((x >> 8) as i32 as f32) * INV_2_24
}

/// Uniform `f64` in `[0, 1)` (52 random bits) from a `u64`.
#[inline(always)]
pub fn u64_to_unit_f64(x: u64) -> f64 {
    f64::from_bits(0x3FF0_0000_0000_0000 | (x >> 12)) - 1.0
}

#[inline(always)]
fn halves(x: [u64; 8]) -> [u32; 16] {
    let mut out = [0u32; 16];
    for i in 0..8 {
        out[i] = x[i] as u32;
        out[8 + i] = (x[i] >> 32) as u32;
    }
    out
}

#[inline(always)]
fn fill_with<T: Copy + Default, const C: usize>(out: &mut [T], mut gen_chunk: impl FnMut() -> [T; C]) {
    let mut it = out.chunks_exact_mut(C);
    for c in &mut it {
        c.copy_from_slice(&gen_chunk());
    }
    let rem = it.into_remainder();
    if !rem.is_empty() {
        let tmp = gen_chunk();
        rem.copy_from_slice(&tmp[..rem.len()]);
    }
}

/// A source of eight `u64` words per step, with vectorised slice fills.
///
/// See the [crate docs](crate) for the exact output layout and step sizes.
pub trait Rng8 {
    /// Next eight 64-bit words (one per lane / block).
    fn next_u64x8(&mut self) -> [u64; 8];

    /// Fills `out` with random `u64`s.
    fn fill_u64(&mut self, out: &mut [u64]) {
        fill_with(out, || self.next_u64x8());
    }

    /// Fills `out` with random `u32`s.
    fn fill_u32(&mut self, out: &mut [u32]) {
        fill_with(out, || halves(self.next_u64x8()));
    }

    /// Fills `out` with uniform `f32` in `[0, 1)` (never 1.0).
    fn fill_f32(&mut self, out: &mut [f32]) {
        fill_with(out, || {
            let h = halves(self.next_u64x8());
            let mut o = [0.0f32; 16];
            for i in 0..16 {
                o[i] = u32_to_unit_f32(h[i]);
            }
            o
        });
    }

    /// Fills `out` with uniform `f64` in `[0, 1)` (never 1.0).
    fn fill_f64(&mut self, out: &mut [f64]) {
        fill_with(out, || {
            let w = self.next_u64x8();
            let mut o = [0.0f64; 8];
            for i in 0..8 {
                o[i] = u64_to_unit_f64(w[i]);
            }
            o
        });
    }

    /// Fills `out` with standard normal `f32` (Box-Muller, see crate docs).
    fn fill_normal_f32(&mut self, out: &mut [f32]) {
        fill_with(out, || {
            let h = halves(self.next_u64x8());
            let mut u1 = [0.0f32; 8];
            let mut u2 = [0.0f32; 8];
            for i in 0..8 {
                u1[i] = 1.0 - u32_to_unit_f32(h[i]);
                u2[i] = u32_to_unit_f32(h[8 + i]);
            }
            let (z0, z1) = box_muller_f32(u1, u2);
            let mut o = [0.0f32; 16];
            o[..8].copy_from_slice(&z0);
            o[8..].copy_from_slice(&z1);
            o
        });
    }

    /// Fills `out` with standard normal `f64` (Box-Muller, see crate docs).
    fn fill_normal_f64(&mut self, out: &mut [f64]) {
        fill_with(out, || {
            let a = self.next_u64x8();
            let b = self.next_u64x8();
            let mut u1 = [0.0f64; 8];
            let mut u2 = [0.0f64; 8];
            for i in 0..8 {
                u1[i] = 1.0 - u64_to_unit_f64(a[i]);
                u2[i] = u64_to_unit_f64(b[i]);
            }
            let (z0, z1) = box_muller_f64(u1, u2);
            let mut o = [0.0f64; 16];
            o[..8].copy_from_slice(&z0);
            o[8..].copy_from_slice(&z1);
            o
        });
    }
}
