# tpt-simd-distance

Distance and similarity kernels for vector search and clustering: inner product, squared L2 and cosine distance over `f32`, `i8` and `u8`, Hamming distance over packed `u64` bit-vectors, and top-k nearest neighbours.

## Overview

| Function | Element | Accumulator | Notes |
|---|---|---|---|
| `inner_product_f32` | `f32` | `f32`, 8 lanes | fixed order |
| `l2_squared_f32` | `f32` | `f32`, 8 lanes | fixed order |
| `cosine_distance_f32` | `f32` | `f32`, 8 lanes | `1 - cos(a, b)` |
| `inner_product_i8`, `l2_squared_i8` | `i8` | `i64` | exact |
| `inner_product_u8`, `l2_squared_u8` | `u8` | `i64` | exact |
| `hamming_distance_u64` | `u64` bit-vectors | `u32` | exact |
| `top_k_l2_f32` | `f32` rows | `f32` | k nearest rows by squared L2 |

The crate is `#![no_std]` and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-distance = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_distance::{cosine_distance_f32, hamming_distance_u64, l2_squared_u8, top_k_l2_f32};

let query = [1.0f32, 0.0, 0.0];
assert!((cosine_distance_f32(&query, &[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-6); // orthogonal

// Integer codes (e.g. quantised embeddings) are exact.
assert_eq!(l2_squared_u8(&[0, 3], &[4, 0]), 16 + 9);

// Binary signatures: number of differing bits.
assert_eq!(hamming_distance_u64(&[0b1011], &[0b0010]), 2);

// Nearest neighbours. Rows are stored flat, `dim` values per row.
let rows = [0.0f32, 0.0, 5.0, 5.0, 1.0, 0.0, 9.0, 9.0];
let mut idx = [0usize; 2];
let mut dist = [0.0f32; 2];
let n = top_k_l2_f32(&[0.9, 0.1], &rows, 2, &mut idx, &mut dist);
assert_eq!(n, 2);
assert_eq!(idx, [2, 0]);
```

## Platform support and semantics

- Paired slices must have equal lengths; a mismatch causes a **panic**, naming the function and both lengths. Empty input is valid and returns zero.
- The `f32` kernels keep eight independent partial sums in a fixed order, then reduce them in a fixed order. Separate multiply and add are used, never FMA, so results are identical on every target. They are not equal to a naive left-to-right sum; the difference is ordinary float reordering error.
- The integer kernels are exact.
- `top_k_l2_f32` panics if `dim` is zero, if the query length differs from `dim`, or if the row buffer is not a multiple of `dim`.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/distance.rs](examples/distance.rs). Run it with `cargo run -p tpt-simd-distance --example distance`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
