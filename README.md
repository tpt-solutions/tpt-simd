# tpt-simd

Portable, zero-cost SIMD abstractions for Rust DSP and codec development.

tpt-simd is a workspace of small, focused crates that wrap low-level SIMD
intrinsics into the building blocks every codec and DSP project ends up
rewriting: complex multiply, fixed-point arithmetic, FFT butterflies,
horizontal reductions, transposes, saturating arithmetic and more.

> **Status:** design phase (v0.1.0-draft). See [spec.txt](spec.txt) for the
> design and [todo.md](todo.md) for the roadmap.

## Goals

- 22 focused crates, each solving one common SIMD problem
- `no_std` + `alloc` (`std` optional)
- Targets: x86_64 (SSE/AVX/AVX2/AVX-512), ARM64 (NEON/SVE), RISC-V RVV, scalar fallback
- Stable backend via `tpt-simd-vector`; `core::simd` behind the `nightly` feature
- Direct integration path into tpt-kinetix and tpt-cadence

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
