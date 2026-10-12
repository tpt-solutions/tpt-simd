# tpt-simd-bits

Bit-level and integer-stream kernels used by codecs and index structures: bitsets, popcount, zigzag coding, delta and prefix-sum streams, and a rank/select index.

## Overview

| Group | Functions and types |
|---|---|
| Bitsets | `popcount_u64`, `bitset_and`, `bitset_or`, `bitset_xor`, `bitset_andnot` |
| Zigzag | `zigzag_encode_i32`, `zigzag_decode_u32`, `zigzag_encode_i64`, `zigzag_decode_u64`, and slice forms `zigzag_encode_slice_i32`, `zigzag_decode_slice_u32` |
| Streams | `delta_encode_u32`, `delta_decode_u32`, `prefix_sum_u32`, `prefix_sum_u64` |
| Index | `RankSelect`: `rank1` in O(1), `select1` in O(log words) plus an in-word scan |

The crate is `#![no_std]` (it uses `alloc` only where noted) and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-bits = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_bits::{delta_decode_u32, delta_encode_u32, popcount_u64, prefix_sum_u32};

assert_eq!(popcount_u64(&[0xFF, 0b1011]), 8 + 3);

// Delta-encode a sorted id list, which makes it compress well; decode restores it.
let mut ids = [10u32, 12, 15, 15, 20];
delta_encode_u32(&mut ids);
assert_eq!(ids, [10, 2, 3, 0, 5]);
delta_decode_u32(&mut ids);
assert_eq!(ids, [10, 12, 15, 15, 20]);

// Prefix sum turns per-block counts into offsets.
let mut offsets = [3u32, 1, 4, 1];
prefix_sum_u32(&mut offsets);
assert_eq!(offsets, [3, 4, 8, 9]);
```

## Platform support and semantics

- Paired slices must have equal lengths; a mismatch causes a **panic**. Empty input is valid.
- All integer arithmetic is wrapping, so debug and release builds give the same results. Delta and prefix-sum outputs wrap modulo 2^N.
- Zigzag maps small signed values to small unsigned ones: 0, -1, 1, -2 become 0, 1, 2, 3.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/bits.rs](examples/bits.rs). Run it with `cargo run -p tpt-simd-bits --example bits`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
