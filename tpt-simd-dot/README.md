# tpt-simd-dot

SIMD dot product variants

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
tpt-simd-dot = "0.1.0"
```

## Features

- `std`: Enable standard library support (default) (includes tpt-simd-core/std and tpt-simd-horizontal/std)
- `nightly`: Enable nightly-only features (includes tpt-simd-core/nightly and tpt-simd-horizontal/nightly)
- `scalar-only`: Use scalar implementations only (no SIMD) (includes tpt-simd-core/scalar-only and tpt-simd-horizontal/scalar-only)

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.