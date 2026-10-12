# tpt-simd-transpose

Cache-blocked matrix transpose and strided 2D copies, generic over any `Copy` element type.

## Overview

| Function | Operation |
|---|---|
| `transpose` | out-of-place `rows x cols` row-major to `cols x rows` |
| `transpose_square_in_place` | in-place transpose of an `n x n` matrix |
| `copy_strided` | 2D copy between buffers with independent row strides |

Work is done in `BLOCK x BLOCK` tiles, so reads and writes stay within a few cache lines per tile. The element type only needs `Copy`; the inner loops are plain copies that compilers vectorise for small types.

Fixed-size 4x4 and 8x8 register tiles live in `tpt-simd-permute`.

The crate is `#![no_std]` and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-transpose = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_transpose::{transpose, transpose_square_in_place};

// Out-of-place: a 2 x 3 row-major matrix becomes 3 x 2.
let src = [1, 2, 3, 4, 5, 6];
let mut dst = [0; 6];
transpose(&src, 2, 3, &mut dst);
assert_eq!(dst, [1, 4, 2, 5, 3, 6]);

// In place for a square matrix.
let mut m = [1, 2, 3, 4];
transpose_square_in_place(&mut m, 2);
assert_eq!(m, [1, 3, 2, 4]);
```

## Platform support and semantics

- Buffers must hold the matrix they describe; a mismatch causes a **panic** before any data is touched.
- Zero rows or columns is valid and does nothing.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/transpose.rs](examples/transpose.rs). Run it with `cargo run -p tpt-simd-transpose --example transpose`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
