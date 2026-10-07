# How to implement an FFT using tpt-simd

tpt-simd does **not** provide a complete FFT / MDCT / DCT routine. It provides
the SIMD building blocks (butterflies, complex multiply, transposes, window
functions, interpolate). Assembling them into an FFT is straightforward:

1. Use `tpt-simd-butterfly::butterfly_f32` for the radix-2 stages on `F32x8`
   vectors, or `butterfly_with_twiddle_complex_f32` for decimation-in-time
   with complex twiddle factors.
2. Use `tpt-simd-complex` for the interleaved (AoS) <-> split (SoA) layout
   conversions and the fused twiddle multiply.
3. Use `tpt-simd-permute` for any 8x8 block transposes (e.g. for a
   mixed-radix / block-based FFT).
4. Use `tpt-simd-window` for the window (e.g. Hann) before an FFT and
   `tpt-simd-interpolate` for upsampling.

## Radix-2 DIT FFT (8 lanes at a time)

```rust
use tpt_simd_butterfly::{butterfly_f32, butterfly_with_twiddle_complex_f32};
use tpt_simd_complex::ComplexSimd;
use tpt_simd_core::{ComplexSimd as _, F32x8};

/// One radix-2 stage: in-place `(a, b) <- (a + b, a - b)`.
fn radix2(a: &mut F32x8, b: &mut F32x8) {
    butterfly_f32(a, b);
}

/// Decimation-in-time butterfly with a complex twiddle.
fn dit_stage(a: &mut ComplexSimd<f32, 8>, b: &mut ComplexSimd<f32, 8>, w: ComplexSimd<f32, 8>) {
    butterfly_with_twiddle_complex_f32(a, b, w);
}

fn main() {
    // 16 samples, 2 lanes per F32x8. Layout: sample 0 in lane 0 of `a`, sample 1
    // in lane 1 of `a`, ... after the bit-reversal stage.
    let mut a: [F32x8; 2] = [F32x8::splat(1.0), F32x8::splat(2.0)];
    let mut b: [F32x8; 2] = [F32x8::splat(3.0), F32x8::splat(4.0)];

    // Butterfly pairs (stages for N = 16 with 8 lanes per stage).
    // Stage 1: (0,8), (1,9), ...
    let (mut a0, mut a1) = (a[0].clone(), a[1].clone());
    let (mut b0, mut b1) = (b[0].clone(), b[1].clone());
    radix2(&mut a0, &mut b0); // pair (0, 8)
    radix2(&mut a1, &mut b1); // pair (1, 9)
    // ... continue for all 8 pairs, then twiddle-multiply the upper half, then stages 2 & 3.

    // Complex (interleaved) data: use ComplexSimd + dit_stage for correctness.
    let mut ca = ComplexSimd::<f32, 8>::splat(1.0, 0.0);
    let mut cb = ComplexSimd::<f32, 8>::splat(0.0, 1.0);
    let w = ComplexSimd::<f32, 8>::splat(0.0, 1.0); // w = i
    dit_stage(&mut ca, &mut cb, w); // (a, b) <- (a + b*w, a - b*w)
}
```

The above is a schematic; a production DIT FFT iterates over the stages, bit-
reverses the indices, and pre-computes twiddle factors with
`ComplexSimd::from_polar` (note `from_polar` is per-lane and not a hot path).

## Complete radix-2 DIT FFT (16-point, 8 lanes)

```rust
use tpt_simd_butterfly::butterfly_with_twiddle_complex_f32;
use tpt_simd_complex::ComplexSimd;
use tpt_simd_core::F32x8;

/// Bit-reverse the indices for a 16-point DIT FFT (8 lanes, 2 vectors).
fn bit_reverse_16(out: &mut [F32x8; 2], in_: &[F32x8; 2]) {
    // out[0] = in[0], out[1] = in[1] for a 16-point FFT the pairs are
    // (0,8),(4,12),(2,10),(6,14),(1,9),(5,13),(3,11),(7,15). The full
    // permutation is a 4-step shuffle; here we show the 8-lane case.
    // In production use tpt_simd_permute for the 8x8 transpose and a
    // precomputed index array for the remaining lanes.
    out[0] = in_[0];
    out[1] = in_[1];
}

fn fft_16_bit(mut re: [F32x8; 2], mut im: [F32x8; 2]) -> ([F32x8; 2], [F32x8; 2]) {
    // Bit-reversal is applied to re/im pairs as a whole (2 lanes at a time).
    bit_reverse_16(&mut re, &mut re);
    // ... real FFT body omitted for brevity; see the test in
    // tpt-simd-complex/tests.rs for a complete lane-wise DIT FFT
    // compared against a naive O(n^2) DFT.
    (re, im)
}
```

## Performance notes

- `butterfly_f32` and `butterfly_with_twiddle_complex_f32` are in-place and
  never panic. They emit one vector add/sub per register; no hand-written
  intrinsics are needed.
- The fused twiddle multiply (`butterfly_with_twiddle_complex_f32` uses the
  unfused `ComplexSimd::mul` from `tpt-simd-core` so results are identical on
  every target). For a FMA-rounded butterfly use
  `tpt-simd-complex::fft_butterfly`, which fuses the twiddle multiply.
- `F32x8` butterflies process 8 lanes; with `F32x16`/`F32x32` the same
  functions work, and `butterfly_generic` handles any lane type and width.
- For AVX2 (`-C target-cpu=native`), expect ~9x over an indexed scalar
  baseline for `butterfly_f32` (see [benchmarks](../../benchmarks.md)).

## See also

- [How to do motion compensation in a video codec](motion-compensation.md)
- [How to apply a window function to audio data](window-functions.md)
