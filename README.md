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

- 39 focused crates plus a `tpt-simd` umbrella (and the unpublished `testutil`), each solving one common SIMD problem
- `no_std` + `alloc` (`std` optional)
- Targets: x86_64 (SSE2/AVX2, AVX-512 where a kernel has a path), aarch64 (via LLVM lowering of the portable backend), and a portable fallback everywhere else
- Stable backend via `tpt-simd-vector`; the `nightly` feature is reserved for a future `core::simd` backend and is currently a no-op
- Direct integration path into tpt-kinetix and tpt-cadence

## Crates

`tpt-simd` re-exports most of the crates below (not `vector`, `dct`, `ntt`,
`audio`, `bits`, `geom`, `transpose`, `pixel`, `distance` or `half`); depend on
individual crates to keep the dependency graph small.

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

Three opt-ins. The first two are off by default so `no_std` users are unaffected;
`runtime-dispatch` is on by default in the `tpt-simd` umbrella (use
`default-features = false` to drop it):

- **`std` feature** (e.g. `tpt-simd = { version = "0.1", features = ["std"] }`):
  float `mul_add`/`round`/`floor`/`ceil`/`trunc`/`sqrt` use the `std` intrinsics
  instead of per-lane `libm` calls. Results are bit-identical; the portable
  complex multiply is ~5x faster. Without it, those ops fall back to `libm`.
- **`-C target-cpu=native`** (or specific `-C target-feature=+avx2,+fma`):
  dispatch is compile-time (see [ADR 0001](docs/adr/0001-backend-and-dispatch.md)),
  so the AVX2/FMA paths are only used when the target features are enabled.
- **`runtime-dispatch` feature** (implies `std`; affects `tpt-simd-blas` gemm and the `hash`, `text` and `search` kernels):
  detects AVX2+FMA at runtime so no `-C target-cpu` is needed (gemm f32 ~1.8x
  faster than the plain build; still below a fully native build). See [ADR 0003](docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

Measured numbers and caveats: [docs/benchmarks.md](docs/benchmarks.md).

## Performance against incumbents

Head-to-head benchmarks against established crates (`crc32fast`, `simd-adler32`,
`memchr`, `faster-hex`, `simdutf8`, `glam`, `gemm`, `faer`, std sort). Ratio = tpt-simd
time ÷ incumbent time, measured in the same run; below 1.0 is faster. Measured on an
Intel Core i5-13500 (2026-10-12), two same-run passes, shown as a range. Summary:

| Area | Operation | Size | Ratio | Notes |
|---|---|---|---|---|
| hash | CRC-32 | 4 KB | 1.00 | parity with `crc32fast` |
| search | `find_byte` | 1 MB | 0.92–0.93 | faster than `memchr` |
| search | `count_byte` | 1 MB | 0.43 | faster than `memchr_iter` |
| text | UTF-8 validate | 4 KB / 256 KB | 1.04–1.09 | close to `simdutf8` |
| text | hex decode | 4 KB | 1.64–1.65 | slower than `faster-hex` (lookup-table validator) |
| sort | radix `sort_u32` | 1 M | 0.40–0.41 | faster than std |
| sort | generic `sort_unstable` | 1 M | 1.19–1.47 | slower than std |
| matrix | mat3 inverse | 3×3 | 1.74–1.77 | slower than `glam` |
| gemm f32 | 512 | 1.03–1.05 | parity with `gemm` / `faer` |

Full tables, the method, and the caveats (noisy rows, and what is not yet compared):
[docs/incumbent-benchmarks.md](docs/incumbent-benchmarks.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
