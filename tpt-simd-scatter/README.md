# tpt-simd-scatter

Eight-lane SIMD scatter (non-contiguous stores) for `i32` and `f32`, with unsafe, panicking and `Result`-returning slice APIs.

## Overview

`base[indices[k]] = values[k]` for `k = 0..8`, with signed `i32` element indices. Each element type comes in four flavours:

- an `unsafe` raw-pointer scatter (`vpscatterdd`/`vscatterdps` when AVX-512 is enabled at compile time),
- an `unsafe` portable scalar reference with identical results,
- a safe bounds-checked slice scatter that panics on a bad index (before writing anything),
- a safe slice scatter that returns `Err(OutOfBounds)` on a bad index (before writing anything).

Duplicate indices: the last lane wins. Stores happen in increasing lane order, so the highest-numbered lane's value is the one left in memory. This holds on every path: the scalar loop stores in lane order, and Intel documents that AVX-512 scatters write overlapping elements from the least to the most significant lane.

Performance caveat: AVX2 has no scatter instruction, so without AVX-512 this is eight scalar stores. With AVX-512 `vpscatterdd` is not dramatically faster than eight stores either; its benefit is mostly keeping data in vector registers.

Dispatch is compile-time (see `docs/adr`): the AVX-512 path is compiled in only when `avx512f` and `avx512vl` are enabled for an `x86_64` target (for example `RUSTFLAGS="-C target-cpu=native"` on a capable CPU) and the `scalar-only` feature is off. There is no runtime detection. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-scatter = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::I32x8;
use tpt_simd_scatter::{scatter_checked_i32, try_scatter_i32, OutOfBounds};

let mut d = [0i32; 3];
scatter_checked_i32(
    &mut d,
    I32x8::from_array([0, 1, 2, 2, 2, 1, 0, 0]),
    I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]),
);
assert_eq!(d, [8, 6, 5]); // last lane wins on duplicates

// Non-panicking form: nothing is written if any index is bad.
let mut d = [0i32; 4];
let r = try_scatter_i32(&mut d, I32x8::from_array([0, 1, 2, 3, 4, 0, 0, 0]), I32x8::splat(9));
assert_eq!(r, Err(OutOfBounds { lane: 4, index: 4 }));
assert_eq!(d, [0; 4]);
```

## API overview

| Item | Description |
|---|---|
| `scatter_i32(base, indices, values)` | `unsafe` raw-pointer scatter of `i32`; AVX-512 `vpscatterdd` when enabled. |
| `scatter_f32(base, indices, values)` | `unsafe` raw-pointer scatter of `f32`. |
| `scatter_i32_portable(base, indices, values)` | `unsafe` scalar reference; stores in lane order. |
| `scatter_f32_portable(base, indices, values)` | `unsafe` scalar reference for `f32`. |
| `scatter_checked_i32(data, indices, values)` | Safe; panics if any index is negative or `>= data.len()`; nothing is written in that case. |
| `scatter_checked_f32(data, indices, values)` | Safe; same panic rule for `f32`. |
| `try_scatter_i32(data, indices, values)` | Safe; returns `Result<(), OutOfBounds>`; `data` is untouched on error. |
| `try_scatter_f32(data, indices, values)` | Safe; same for `f32`. |
| `OutOfBounds { lane, index }` | Error type: the first offending lane and its index. Implements `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq` and `Display`. |

`indices` is `Simd<i32, 8>` (`I32x8`); `values` is `I32x8` or `F32x8`.

Index semantics: indices are signed element offsets. The unsafe functions accept a negative index provided `base.offset(index)` is valid for writes (the hardware sign-extends indices); the checked and `try_` variants treat a negative index as out of bounds and validate all eight indices before writing.

## Feature flags

The crate is `#![no_std]` in every configuration.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only` and disables this crate's AVX-512 scatter, so the portable loop is always used. |

## Platform and target support

- x86_64 with `avx512f` and `avx512vl` enabled at compile time (and `scalar-only` off): AVX-512 scatter instructions.
- Every other target, including AVX2-only x86_64 and aarch64: portable scalar loop.
- Safety: the `unsafe` functions require that every addressed element be writable, in bounds of a single allocation, aligned, and not aliased by a live reference or accessed concurrently; violating this is undefined behaviour. The checked and `try_` variants are safe.

## Performance

Benchmarks live in `benches/scatter.rs`; run `cargo bench -p tpt-simd-scatter`, preferably with `RUSTFLAGS="-C target-cpu=native"`. See [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-gather`: the load counterpart.
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
