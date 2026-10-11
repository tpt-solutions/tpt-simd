# tpt-simd-audio

Audio kernels on `f32` samples:

- `mix_into`: `dst += gain * src`
- `interleave` / `deinterleave`: planar to interleaved and back
- `peak` / `rms`: per-buffer level measurement
- `fir`: causal direct-form FIR filter
- `Biquad`: RBJ low-pass and high-pass sections, direct form II transposed
- `resample_sinc`: windowed-sinc (Blackman) sample-rate conversion

No runtime dependencies beyond `libm` for trigonometry. MIT OR Apache-2.0.
