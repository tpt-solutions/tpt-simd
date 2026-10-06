# tpt-simd-blas

BLAS-style f32/f64 kernels: level-1, gemv and a packed, blocked gemm

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
tpt-simd-blas = "0.1.0"
```

## Features

- `alloc`: Enable allocator support (default)
- `std`: Enable standard library support (default)
- `nightly`: Enable nightly-only features
- `scalar-only`: Use scalar implementations only (no SIMD)

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.