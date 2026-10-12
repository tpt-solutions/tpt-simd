# tpt-simd-fuzz

libFuzzer targets for tpt-simd. Unpublished (`publish = false`).

Each target checks a SIMD kernel against a portable or naive reference:

| Target | Checks |
|---|---|
| `dot_i16` | AVX2, AVX-512BW and portable `i16` dot products agree exactly |
| `gather_scatter_i32` | Checked gather and scatter against a scalar reference; indices are reduced into range |
| `transpose_8x8_i16` | SSE2 and AVX2 8x8 `i16` transpose equals the portable transpose |
| `gemm_f32` | Packed, blocked `f32` GEMM against the naive reference, within the ADR 0003 tolerance |

## Running

The targets use [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz), which needs a nightly toolchain:

```sh
cargo +nightly fuzz run dot_i16
```

## License

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE)) or MIT license ([LICENSE-MIT](../LICENSE-MIT)), at your option.
