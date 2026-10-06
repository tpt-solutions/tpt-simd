# tpt-simd-complex

SIMD complex number operations for FFT/MDCT

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
tpt-simd-complex = "0.1.0"
```

## Features

- `std`: Enable standard library support (default) (includes tpt-simd-mul/std)
- `nightly`: Enable nightly-only features (includes tpt-simd-mul/nightly)
- `scalar-only`: Use scalar implementations only (no SIMD) (includes tpt-simd-mul/scalar-only)

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.