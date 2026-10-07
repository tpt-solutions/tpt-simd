# tpt-simd-gather

Eight-lane SIMD gather (non-contiguous loads) for `i32` and `f32`, with unsafe, panicking and non-panicking slice APIs.

## Overview

Lane `k` of the result is `base[indices[k]]`, with eight signed `i32` element indices per call. The same element may be gathered by several lanes. Each element type comes in four flavours:

- an `unsafe` raw-pointer gather (`vpgatherdd`/`vgatherdps` on AVX2),
- an `unsafe` portable scalar reference with identical results,
- a safe bounds-checked slice gather that panics on a bad index,
- a safe slice gather that returns `None` on a bad index.

Performance caveat (stated in the crate docs): the hardware gather is not a fast instruction. It is roughly on par with eight scalar loads on Haswell/Broadwell, better on Skylake and later, and usually slower than scalar loads on AMD Zen 1-3 (Zen 4 is better). A stand-alone gather often ties with or loses to a plain scalar loop, and the checked variants add eight bounds compares. It tends to win when it feeds further SIMD work. Measure on your target.

Dispatch is compile-time (see `docs/adr`): the AVX2 path is compiled in only when `avx2` is enabled for an `x86_64` target (for example `RUSTFLAGS="-C target-cpu=native"`) and the `scalar-only` feature is off. There is no runtime detection. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-gather = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::I32x8;
use tpt_simd_gather::{gather_checked_i32, try_gather_i32};

let data = [5, 6, 7];
let r = gather_checked_i32(&data, I32x8::from_array([0, 1, 2, 2, 1, 0, 0, 1]));
assert_eq!(r.to_array(), [5, 6, 7, 7, 6, 5, 5, 6]);

// Non-panicking form: any index outside 0..len yields None.
assert!(try_gather_i32(&data, I32x8::splat(3)).is_none());
assert!(try_gather_i32(&data, I32x8::splat(-1)).is_none());
```

Raw-pointer form:

```rust
use tpt_simd_core::I32x8;
use tpt_simd_gather::gather_i32;

let data = [0, 10, 20, 30, 40, 50, 60, 70, 80];
// SAFETY: every index is in bounds of `data`.
let r = unsafe { gather_i32(data.as_ptr(), I32x8::from_array([8, 0, 4, 4, 1, 2, 3, 5])) };
assert_eq!(r.to_array(), [80, 0, 40, 40, 10, 20, 30, 50]);
```

## API overview

| Function | Element | Safety / behaviour |
|---|---|---|
| `gather_i32(base, indices)` | `i32` | `unsafe`; `vpgatherdd` on AVX2. |
| `gather_f32(base, indices)` | `f32` | `unsafe`; `vgatherdps` on AVX2; bit patterns (including NaN payloads) are copied unchanged. |
| `gather_i32_portable(base, indices)` | `i32` | `unsafe`; scalar reference, no intrinsics. |
| `gather_f32_portable(base, indices)` | `f32` | `unsafe`; scalar reference, no intrinsics. |
| `gather_checked_i32(data, indices)` | `i32` | Safe; panics if any index is negative or `>= data.len()`. |
| `gather_checked_f32(data, indices)` | `f32` | Safe; same panic rule. |
| `try_gather_i32(data, indices)` | `i32` | Safe; returns `Option<Simd<i32, 8>>`, `None` on any bad index. No lane is read unless all eight indices are valid. |
| `try_gather_f32(data, indices)` | `f32` | Safe; same as above for `f32`. |

All functions take `indices: Simd<i32, 8>` (`I32x8`) and return an eight-lane vector.

Index semantics: indices are signed element offsets. In the unsafe functions a negative index is allowed as long as `base.offset(index)` is valid for reads (the hardware sign-extends indices). In the checked and `try_` variants a negative index is always out of bounds.

## Feature flags

The crate is `#![no_std]` in every configuration.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only` and disables this crate's AVX2 gather, so the portable loop is always used. |

## Platform and target support

- x86_64 with `avx2` enabled at compile time (and `scalar-only` off): `_mm256_i32gather_epi32` / `_mm256_i32gather_ps`.
- Every other target (including 32-bit x86 and aarch64): portable scalar loop; there is no NEON or AVX-512 gather path.
- Safety: the `unsafe` functions require that every addressed element be in bounds of a single allocation, initialised and aligned; violating this is undefined behaviour and may fault. The checked and `try_` variants validate all eight indices before any read and are safe.

## Performance

Benchmarks live in `benches/gather.rs`; run `cargo bench -p tpt-simd-gather`, preferably with `RUSTFLAGS="-C target-cpu=native"`. See the performance note above and [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-scatter`: the store counterpart.
- `tpt-simd-select`: uses `gather_checked_i32` / `try_gather_i32` for slice selection by index.
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
