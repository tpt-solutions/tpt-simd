# tpt-simd-testutil

Unpublished proptest strategies and scalar-reference helpers shared by the tpt-simd test suites.

This is an internal workspace helper crate. It is not published to crates.io (`publish = false` is set in its `Cargo.toml`) and carries no stability guarantee; do not depend on it from outside this repository.

## Overview

The kernel crates test SIMD results against a scalar reference over awkward inputs. This crate collects the reusable pieces:

- proptest strategies for lanes, vectors and slices, including NaN, infinities, signed zeros and extreme values, and every tail length;
- a tolerant `f32` comparison for reductions that reassociate (see ADR 0003 for the tolerance tiers).

It is `#![forbid(unsafe_code)]`, uses `std` (via `proptest`), and is not `no_std`.

## Usage (inside the workspace)

Workspace crates declare it as a dev-dependency:

```toml
[dev-dependencies]
proptest.workspace = true
tpt-simd-testutil.workspace = true
```

Example:

```rust
use proptest::prelude::*;
use tpt_simd_testutil::{assert_slice_close, f32_with_specials, vec_pair};

proptest! {
    #[test]
    fn add_matches_scalar((a, b) in vec_pair(f32_with_specials(), 200)) {
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        // Replace this with a call to the kernel under test.
        let actual: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_slice_close(&actual, &expected, 1e-5, 1e-6);
    }
}
```

## API overview

| Function | Description |
| --- | --- |
| `lens(max)` | Strategy for slice lengths `0..=max` |
| `finite_f32()` | Finite `f32` in `-1000.0..1000.0` |
| `f32_with_specials()` | `f32` mixing ordinary values with NaN, +/-inf, +/-0, `MAX`, `MIN`, `MIN_POSITIVE` |
| `i16_edgy()`, `i32_edgy()` | Integers biased toward `MIN`, `MAX`, 0, -1, 1 |
| `vec_of(elem, max_len)` | `Vec` of length `0..=max_len` |
| `vec_pair(elem, max_len)` | Two equal-length `Vec`s for binary slice operations |
| `simd_of::<T, _, N>(elem)` | Strategy for a `Simd<T, N>` |
| `approx_eq_f32(a, b, rel, abs)` | Equal, both NaN, or within relative / absolute tolerance |
| `assert_slice_close(actual, expected, rel, abs)` | Element-wise `approx_eq_f32` over slices, with the index in the failure message |

## Feature flags

None.

## Related crates

- `tpt-simd-vector` (dependency): `Simd` and `SimdElement`.
- `proptest` (dependency).
- Used as a dev-dependency by `tpt-simd-core`, `tpt-simd-aligned` and the kernel crates.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
