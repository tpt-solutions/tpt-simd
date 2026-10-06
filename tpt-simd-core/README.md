# tpt-simd-core

Core traits, type aliases and feature detection for tpt-simd

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
tpt-simd-core = "0.1.0"
```

## Features

- `std`: Enable standard library support (default) (includes tpt-simd-vector/std)
- `nightly`: Enable nightly-only features (includes tpt-simd-vector/nightly)
- `scalar-only`: Use scalar implementations only (no SIMD) (includes tpt-simd-vector/scalar-only)

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.