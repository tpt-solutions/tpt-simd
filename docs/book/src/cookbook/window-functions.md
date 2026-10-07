# How to apply a window function to audio data

A window (Hann, Hamming, Blackman, Kaiser) tapers audio samples before a
FFT or filter. tpt-simd provides symmetric window generators, SIMD application,
and the polynomial cosine used to compute the windows.

## Plain samples (no SIMD needed)

```rust
use tpt_simd_window::{hanning_window_into_f32, apply_window_f32};

let mut window = [0.0f32; 8];
hanning_window_into_f32(&mut window);
// symmetric, zero endpoints: window[0] ~= window[7] ~= 0

let mut data = [1.0f32; 8];
apply_window_f32(&mut data, &window);
```

## SIMD application

```rust
use tpt_simd_window::{apply_window_simd_f32, hamming_window_into_f32};
use tpt_simd_core::F32x8;

let mut window = [0.0f32; 8];
hamming_window_into_f32(&mut window);

// data holds 16 samples; window holds 8 (apply in two vector passes).
let mut data = [1.0f32; 16];
apply_window_simd_f32(&mut data[..8], &window);
apply_window_simd_f32(&mut data[8..], &window);
```

## Selecting a window

| Window | Formula (symmetric, denominator N-1) | Use |
| --- | --- | --- |
| Hann (`hanning`) | `0.5 - 0.5 cos(2 pi n / (N-1))` | general FFT analysis |
| Hamming (`hamming`) | `0.54 - 0.46 cos(2 pi n / (N-1))` | spectrum estimation |
| Blackman (`blackman`) | `0.42 - 0.5 cos(2 pi n/(N-1)) + 0.08 cos(4 pi n/(N-1))` | narrowband |
| Kaiser (`kaiser`) | modified Bessel I0, `beta` shapes the taper | flexible, `beta=0` is rectangular |

```rust
use tpt_simd_window::{hanning_window_f32, kaiser_window_f32};

let w_hann = hanning_window_f32(64);
let w_kaiser = kaiser_window_f32(64, 8.0); // beta = 8
```

## Performance notes

- Windows are **symmetric** (`w[0] == w[N-1]`); size 0 is empty, size 1 is
  `[1.0]`.
- The cosine-sum windows evaluate 8 lanes at a time with a plain
  multiply/add polynomial; no hand-written intrinsics and no `unsafe`.
- `apply_window_f32`/`apply_window_simd_f32` are bit-identical to the scalar
  reference; they panic if the slice lengths differ.
- `tpt-simd-window` is `#![no_std]`; the allocating generators need the
  `alloc` (or `std`) feature. Benchmarks: Hamming (4096) is ~5.3x over a
  `libm` `cosf` scalar baseline.

## See also

- [How to implement an FFT using tpt-simd](fft.md)
- [How to do motion compensation in a video codec](motion-compensation.md)
