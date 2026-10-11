# tpt-simd

Portable, zero-cost SIMD abstractions for Rust DSP and codec development.

tpt-simd is a workspace of small, focused crates that wrap low-level SIMD
intrinsics into the building blocks every codec and DSP project ends up
rewriting: complex multiply, fixed-point arithmetic, FFT butterflies,
horizontal reductions, transposes, saturating arithmetic and more.

> **Status:** pre-release (v0.1.0). The crates below are implemented with
> tests and benchmarks; the API may still change before publishing. See
> [spec.txt](spec.txt) for the design and [todo.md](todo.md) for the roadmap.

## Goals

- 40 focused crates plus a `tpt-simd` umbrella (and the unpublished `testutil`), each solving one common SIMD problem
- `no_std` + `alloc` (`std` optional)
- Targets: x86_64 (SSE/AVX/AVX2/AVX-512), ARM64 (NEON/SVE), RISC-V RVV, scalar fallback
- Stable backend via `tpt-simd-vector`; `core::simd` behind the `nightly` feature
- Direct integration path into tpt-kinetix and tpt-cadence

## Crates

`tpt-simd` re-exports everything below; depend on individual crates to keep
the dependency graph small.

| Area | Crates |
| --- | --- |
| Foundation | `core` (traits, feature detection), `vector` (stable `Simd`/`SimdMask` backend), `aligned`, `testutil` (unpublished) |
| Arithmetic | `mul`, `fixed`, `saturate`, `rounding`, `shift`, `math` (exp/ln/sin/cos/tanh/erf) |
| Complex / transforms | `complex`, `butterfly`, `window`, `convolve`, `interpolate` |
| Data movement | `permute`, `gather`, `scatter`, `select`, `blend`, `compare` |
| Reductions | `horizontal`, `reduce`, `dot` |
| Linear algebra | `matrix`, `blas`, `sparse` |
| Random numbers | `rng` (xoshiro256++, Philox4x32-10) |
| Transforms and signal | `dct` (DCT-II/III, MDCT), `ntt` (modular NTT), `audio` (mix, biquad, resampler), `window` |
| Bits and integers | `bits` (popcount, bitset, delta, rank/select), `hash` (CRC-32, Adler-32, FNV-1a) |
| Text and bytes | `text` (hex, UTF-8, UTF-16 transcoding), `search` (byte and substring search), `half` (f16/bf16) |
| Geometry and images | `geom` (SoA distance, AABB, quaternions), `transpose`, `pixel` (RGB/HSV, histograms) |
| Similarity and sorting | `distance` (dot, L2, cosine, Hamming, top-k), `sort` (sort, select_nth, argsort) |

All crate names are prefixed `tpt-simd-`.

### Which crate do I need?

| Task | Crate |
| --- | --- |
| FIR filter, convolution | `convolve` (f32, i16) |
| FFT butterflies, complex math | `complex`, `butterfly` |
| Dot product, matrix multiply | `dot`, `blas` (f32/f64 gemm, gemv, LAPACK-style LU/Cholesky) |
| Sum, mean, variance | `reduce` |
| Sparse CG / BiCGSTAB | `sparse` |
| Checksums | `hash` |
| Hex or UTF-8 validation | `text` |
| Substring or byte search | `search` |
| Sort or top-k | `sort` |
| Nearest-neighbour distances | `distance` |
| Random numbers | `rng` |
| Fixed-point arithmetic | `fixed` |

`hash`, `sort`, `search`, `text` and `pixel` have no runtime dependencies by default; `audio` needs only `libm`. All of them build for `no_std` targets without features.

## Getting full performance

Three opt-ins, all off by default so `no_std` users are unaffected:

- **`std` feature** (e.g. `tpt-simd = { version = "0.1", features = ["std"] }`):
  float `mul_add`/`round`/`floor`/`ceil`/`trunc`/`sqrt` use the `std` intrinsics
  instead of per-lane `libm` calls. Results are bit-identical; the portable
  complex multiply is ~5x faster. Without it, those ops fall back to `libm`.
- **`-C target-cpu=native`** (or specific `-C target-feature=+avx2,+fma`):
  dispatch is compile-time (see [ADR 0001](docs/adr/0001-backend-and-dispatch.md)),
  so the AVX2/FMA paths are only used when the target features are enabled.
- **`runtime-dispatch` feature** (implies `std`; currently affects `tpt-simd-blas` gemm):
  detects AVX2+FMA at runtime so no `-C target-cpu` is needed (gemm f32 ~1.8x
  faster than the plain build; still below a fully native build). See [ADR 0003](docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

Measured numbers and caveats: [docs/benchmarks.md](docs/benchmarks.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
