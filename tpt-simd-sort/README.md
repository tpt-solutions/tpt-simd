# tpt-simd-sort

Unstable sort, quickselect, argsort, partition and top-k over slices, with no runtime dependencies.

## Overview

| Function | Notes |
|---|---|
| `sort_unstable`, `sort_by` | introsort with 3-way partitioning, insertion sort for small slices, heapsort fallback |
| `sort_u32` | sort for `u32` slices |
| `select_nth` | quickselect with the same fallback |
| `top_k_smallest` | moves the k smallest elements to the front, sorted ascending |
| `argsort` | sorts indices by keys, leaving the data untouched; not stable |
| `partition` | in-place, unstable predicate partition; returns the count of matches |

Worst case is O(n log n) for every sort and select function. Equal elements are handled in one pass by the 3-way partition, so many duplicates are cheap.

Sorting networks for fixed small blocks are not built yet.

The crate is `#![no_std]` and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-sort = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_sort::{argsort, select_nth, sort_unstable, top_k_smallest};

let mut v = [5, 1, 4, 1, 3];
sort_unstable(&mut v);
assert_eq!(v, [1, 1, 3, 4, 5]);

// Quickselect: put the element of rank k in place without a full sort.
let mut w = [9, 2, 7, 4];
select_nth(&mut w, 1);
assert_eq!(w[1], 4);

// Sort indices by key, leaving the data untouched.
let keys = [30, 10, 20];
let mut order = [0usize, 1, 2];
argsort(&keys, &mut order);
assert_eq!(order, [1, 2, 0]);

// Top-k: the two smallest move to the front.
let mut data = [8, 3, 9, 1, 7];
top_k_smallest(&mut data, 2);
assert_eq!(&data[..2], &[1, 3]);
```

## Platform support and semantics

- `argsort` requires `out.len() == keys.len()`; a mismatch causes a **panic**.
- `select_nth` panics if the index is out of range.
- Unstable: elements that compare equal may be reordered.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/sort.rs](examples/sort.rs). Run it with `cargo run -p tpt-simd-sort --example sort`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
