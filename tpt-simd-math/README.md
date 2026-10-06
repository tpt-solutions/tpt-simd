# tpt-simd-math

Vectorised `f32` `exp`, `ln`, `sin`, `cos`, `tanh` and `erf` over slices and
`Simd<f32, 8>`: branch-free polynomial approximations with range reduction,
plain `+ - * /` only (no per-lane `libm`), bit-identical on every target.
Documented max ULP error and special-value behaviour are in the crate docs.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
tpt-simd-math = "0.1.0"
```

```rust
let x = [0.0f32, 1.0, -1.0, 10.0];
let mut y = [0.0f32; 4];
tpt_simd_math::exp_f32(&x, &mut y);
```

Build with `-C target-cpu=native` (or `+avx2`) for AVX2 speed; compile-time
dispatch, see ADR 0001. Accuracy is Tier 2 (documented ULP bounds, ADR 0003).

## Features

- `std`: Enable standard library support (includes tpt-simd-core/std)
- `nightly`: Enable nightly-only features (includes tpt-simd-core/nightly)
- `scalar-only`: Use scalar implementations only (no SIMD) (includes tpt-simd-core/scalar-only)

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.