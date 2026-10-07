# tpt-simd-rng

Lane-parallel pseudo-random number generation for `no_std`: eight independent xoshiro256++ streams (or eight Philox4x32-10 blocks) advanced in lock step, with slice fills for `u32`, `u64`, uniform `f32`/`f64` in `[0, 1)` and standard-normal `f32`/`f64`.

## Overview

The crate provides bulk random-number generation where throughput matters: filling large buffers for Monte Carlo simulation, noise generation, dithering or test data. All generators work on `[T; 8]` arrays (structure-of-arrays state), and every fill advances all eight lanes together.

- Generators: `SplitMix64` (seeding), scalar `Xoshiro256pp` (reference, with `jump` / `long_jump`), `Xoshiro256ppX8` (8 independent `u64` streams), and the counter-based `Philox4x32X8` (8 blocks per step, random access via `seek`).
- Fills (trait `Rng8`): `u64`, `u32`, uniform `f32`/`f64`, and standard normals (Box-Muller using branch-free polynomial `ln` / `sin` / `cos` and Newton `sqrt`, no libm in the hot loop).
- Branch-free lane-wise math helpers are public in the `math` module.

Scalar versus SIMD: the code is safe, portable Rust on arrays of 8 elements (wrapping integer ops, shifts, rotates). There is no separate scalar/intrinsic code path; LLVM maps the array code to vector instructions for whichever target features the build enables. Dispatch is compile-time (see [ADR 0001](../docs/adr/0001-backend-and-dispatch.md) and [ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md)), so build with `-C target-cpu=native` (or `+avx2`) to get AVX2 code. Results are bit-identical on every target and feature set (no FMA, no libm).

`no_std`: the crate is `#![no_std]` and `#![forbid(unsafe_code)]`. It is not cryptographically secure.

## Installation

```toml
[dependencies]
tpt-simd-rng = "0.1.0"
```

The same functionality is available through the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_rng::{Rng8, Xoshiro256ppX8};

let mut rng = Xoshiro256ppX8::from_seed(42);

let mut u = [0.0f32; 100];
rng.fill_f32(&mut u);
assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));

let mut z = [0.0f64; 100];
rng.fill_normal_f64(&mut z);
```

Counter-based streams with random access:

```rust
use tpt_simd_rng::{Philox4x32X8, Rng8};

let mut g = Philox4x32X8::new(1234, 7); // key 1234, stream 7
let mut buf = [0u32; 64];
g.fill_u32(&mut buf);
g.seek(0); // back to block 0
```

## API overview

Generators

| Item | Description |
|---|---|
| `SplitMix64` | Seeding generator (`new`, `next_u64`); any seed including 0 |
| `Xoshiro256pp` | Scalar xoshiro256++ reference: `from_state`, `from_seed`, `state`, `next_u64`, `jump` (2^128 steps), `long_jump` (2^192 steps) |
| `Xoshiro256ppX8` | 8 independent streams: `LANES`, `from_seed`, `from_seed_block`, `from_lanes`, `lane`, `jump_blocks`, `step`, `next_simd` |
| `Philox4x32X8` | Counter-based, 8 blocks per step: `from_seed`, `new(seed, stream)`, `seek`, `blocks` |
| `philox4x32_10` | Scalar single-block Philox4x32-10 reference |

`Rng8` trait (implemented by `Xoshiro256ppX8` and `Philox4x32X8`)

| Method | Description |
|---|---|
| `next_u64x8` | Next eight 64-bit words |
| `fill_u64`, `fill_u32` | Raw integer fills |
| `fill_f32`, `fill_f64` | Uniform in `[0, 1)` (24 and 52 random bits; never 1.0) |
| `fill_normal_f32`, `fill_normal_f64` | Standard normal (Box-Muller) |

Conversions: `u32_to_unit_f32`, `u64_to_unit_f64`.

`math` module (8-lane arrays): `ln_f32`, `ln_f64`, `sincos_turns_f32`, `sincos_turns_f64`, `sqrt_f32`, `sqrt_f64`, `box_muller_f32`, `box_muller_f64`.

Fill semantics: every fill consumes whole steps (8 `u64` words); a partial last step's surplus is discarded. A fill of `n` elements equals the first `n` of a longer fill from the same state, but splitting a fill into several calls changes the numbers unless every chunk is a multiple of the step size (8 for `u64`/`f64`, 16 for `u32`/`f32`/normals). See the crate docs for the full table.

## Feature flags

| Feature | Effect |
|---|---|
| `default` | empty |
| `std` | Forwards to `tpt-simd-core/std`; results are unchanged |
| `nightly` | Forwards to `tpt-simd-core/nightly` |
| `scalar-only` | Forwards to `tpt-simd-core/scalar-only` |

## Platform support and accuracy

- Works on all targets; vector code generation depends on the enabled target features (x86 SSE/AVX2, AVX-512, aarch64 NEON) as chosen by LLVM. There are no hand-written intrinsics and no `unsafe`.
- 64-bit vector multiply/rotate do not exist before AVX-512, so raw `u64` output gains little over a pipelined scalar loop, while `u32`/`f32` output and normals benefit most.
- Normal variates have a few ulp end-to-end error and a truncated support (`|z| <= 5.77` for `f32`, `<= 8.5` for `f64`). `f64` uniforms carry 52 random bits.
- `math` functions are documented with accuracy tables (for example `ln_f32` 1 ulp, `ln_f64` 2 ulp); inputs outside the stated domains give unspecified but memory-safe results.

## Performance

Benchmarks compare against a scalar xoshiro256++ loop and a libm Box-Muller: `cargo bench -p tpt-simd-rng`. See [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

Depends on `tpt-simd-core`. Siblings: `tpt-simd-math`, `tpt-simd-sparse`, `tpt-simd-blas`. Umbrella: `tpt-simd`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
