# tpt-simd-geom

Batched 3D geometry over structure-of-arrays (SoA) `f32` data: distances, dot products, AABB overlap, ray casts, quaternion products and rotations, and 4x4 matrix transforms of points.

## Overview

| Function | Input | Output |
|---|---|---|
| `distance_sq` / `distance` | points, query point | squared / Euclidean distance per point |
| `dot` | points, direction | dot product per point |
| `aabb_overlap` | boxes, query box | `bool` per box |
| `ray_aabb` | boxes, ray origin and direction, `t_max` | entry distance per box, or `+inf` on a miss |
| `quat_mul` | two quaternion batches | batched Hamilton product |
| `quat_rotate` | quaternions, vectors | rotated vectors |
| `mat4_transform_points` | one 4x4 matrix, points | transformed points |

Coordinates are split into one slice per axis (`Soa3`, `Soa3Mut`). Boxes use two of these, minimum and maximum corners (`BoxesSoa`). Quaternions use `QuatSoa` and `QuatSoaMut`, with `w` as the scalar part. Inputs are borrowed slices, so nothing is allocated.

The crate is `#![no_std]` and forbids `unsafe` code. Its only dependency is `libm`.

## Installation

```toml
[dependencies]
tpt-simd-geom = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_geom::{Soa3, distance_sq};

let x = [0.0, 3.0];
let y = [0.0, 4.0];
let z = [0.0, 0.0];
let pts = Soa3::new(&x, &y, &z);
let mut out = [0.0f32; 2];
distance_sq(pts, [0.0, 0.0, 0.0], &mut out);
assert_eq!(out, [0.0, 25.0]);
```

## Platform support and semantics

- Slices that describe the same set of items must have the same length, and output slices must match the item count. A mismatch causes a **panic**. Empty input is valid.
- Plain `f32` arithmetic with separate multiply and add (no FMA), so results are the same on every target.
- NaN coordinates are not supported. A NaN in a box or ray gives an unspecified hit or miss.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/geom.rs](examples/geom.rs). Run it with `cargo run -p tpt-simd-geom --example geom`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
