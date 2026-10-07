# tpt-simd-select

SIMD element selection by index: lane selection within a vector, and slice selection through a gather.

## Overview

This crate provides three index-driven selection operations plus a re-export:

- `select_i32` and `select_lanes_f32` pick lanes of an 8-lane vector by index (the `permutevar8x32` pattern). Only the low 3 bits of each index are used, so every index is in range by construction and the functions never panic.
- `select_from_slice_i32` picks elements of a slice by index using `tpt-simd-gather`; an out-of-bounds or negative index panics. `try_select_from_slice_i32` is the non-panicking form.
- `select_f32` is re-exported from `tpt-simd-blend`. It is the condition-based blend (a lane takes `a` when the condition lane's sign bit is set), distinct from the index-based `select_lanes_f32`, which was renamed from the spec's `select_f32` to avoid the name clash.

Scalar vs SIMD: the lane-selection functions in this crate are implemented as portable lane loops (`core::array::from_fn`), with no hand-written intrinsics. `select_f32` and the slice selection delegate to the SIMD-capable code in `tpt-simd-blend` and `tpt-simd-gather`, which dispatch at compile time (see `docs/adr`). The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-select = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::Simd;
use tpt_simd_select::{select_from_slice_i32, select_i32, try_select_from_slice_i32};

// Index-based lane selection; only the low 3 bits of each index matter.
let v = Simd::from_array([10, 11, 12, 13, 14, 15, 16, 17]);
let i = Simd::from_array([7, 0, 0, 1, 8, 9, 3, 3]);
assert_eq!(select_i32(v, i).to_array(), [17, 10, 10, 11, 10, 11, 13, 13]);

// Selection from a slice.
let data = [5, 6, 7, 8];
let i = Simd::from_array([3, 2, 1, 0, 0, 1, 2, 3]);
assert_eq!(select_from_slice_i32(&data, i).to_array(), [8, 7, 6, 5, 5, 6, 7, 8]);

// Non-panicking form.
assert!(try_select_from_slice_i32(&[1, 2], Simd::splat(2)).is_none());
```

## API overview

| Function | Description |
|---|---|
| `select_i32(v, indices)` | `out[i] = v[indices[i] & 7]` for `i32` lanes. |
| `select_lanes_f32(v, indices)` | `out[i] = v[indices[i] & 7]` for `f32` lanes. |
| `select_from_slice_i32(data, indices)` | `out[i] = data[indices[i]]`; panics if any index is negative or `>= data.len()`. |
| `try_select_from_slice_i32(data, indices)` | Same, returning `None` on an out-of-bounds index. |
| `select_f32(condition, a, b)` | Re-export from `tpt-simd-blend`: `a` where the condition lane's sign bit is set, else `b`. |

All index arguments are `Simd<i32, 8>`; vectors are eight lanes.

## Feature flags

The crate is `#![no_std]` in every configuration. The features only forward to `tpt-simd-core`.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only`. It is not forwarded to `tpt-simd-blend` or `tpt-simd-gather`, so it does not disable their AVX2 paths. |

## Platform and target support

- All targets: the lane-selection functions use portable code.
- Slice selection and `select_f32` use AVX2 (`vpgatherdd`, `vblendvps`) when `avx2` is enabled at compile time, through their home crates; otherwise they use portable code.
- There is no hand-written NEON code. All public functions are safe; the crate itself contains no `unsafe`.

## Performance

This crate has no benchmarks of its own. For the underlying operations see `cargo bench -p tpt-simd-gather` and `cargo bench -p tpt-simd-blend`, and [../docs/benchmarks.md](../docs/benchmarks.md). Note that hardware gather is often no faster than scalar loads.

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-blend`: home of `select_f32` (dependency).
- `tpt-simd-gather`: provides the checked gathers used for slice selection (dependency).
- `tpt-simd-permute`: general lane permutation (`permute_f32`, `permute_i32`, `permute`).
- `tpt-simd`: umbrella crate re-exporting the workspace.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
