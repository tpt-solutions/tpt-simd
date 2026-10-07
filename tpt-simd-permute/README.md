# tpt-simd-permute

SIMD transposes, stereo and YUV interleaving, byte widening and arbitrary lane permutation.

## Overview

`tpt-simd-permute` collects data-shuffling building blocks:

- in-place block transposes (`8x8` of `i16`, `4x4` of `f32`), the core of 2-D video DCTs and matrix kernels,
- planar to packed conversions for audio (`LLLL/RRRR` to `LRLRLR`) and video (I420 to NV12),
- sign-extending widening of a byte slice,
- arbitrary lane permutation (the `vpermps` / `vpermd` pattern), plus a generic version for any `Simd<T, N>`,
- re-exports of the saturating pack functions from `tpt-simd-saturate`, their home crate.

Scalar vs SIMD: transposes have a hand-written SSE2 unpack network on x86_64 and a portable swap loop elsewhere. `permute_f32` and `permute_i32` use `vpermps`/`vpermd` when AVX2 is enabled at compile time and a portable index loop otherwise. The interleave, deinterleave, YUV and unpack functions are plain loops written so LLVM can auto-vectorise them. All paths give identical results.

Dispatch is compile-time (see `docs/adr`): SSE2 is part of the x86_64 baseline, so the transposes use it by default; the AVX2 permutes need `avx2` enabled (for example `RUSTFLAGS="-C target-cpu=native"`). There is no runtime detection. The crate is `#![no_std]`.

Length policy (ADR 0002): slice functions never return errors. Every length relationship is checked up front with `assert!` and a mismatch panics with a descriptive message (documented under `# Panics` on each function). Lengths must match exactly; there is no silent truncation. Empty slices are valid.

## Installation

```toml
[dependencies]
tpt-simd-permute = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::{F32x8, Simd};
use tpt_simd_permute::{interleave_stereo_i16, permute_f32, transpose_8x8_i16};

// Reverse the lanes of a vector.
let v = F32x8::from_fn(|i| i as f32);
let r = permute_f32(v, Simd::from_array([7, 6, 5, 4, 3, 2, 1, 0]));
assert_eq!(r.to_array(), [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);

// Interleave two mono channels into packed stereo.
let mut out = [0i16; 6];
interleave_stereo_i16(&[1, 2, 3], &[-1, -2, -3], &mut out);
assert_eq!(out, [1, -1, 2, -2, 3, -3]);

// In-place 8x8 transpose.
let mut m = [[0i16; 8]; 8];
m[1][2] = 7;
transpose_8x8_i16(&mut m);
assert_eq!(m[2][1], 7);
```

## API overview

Transposes:

| Function | Description |
|---|---|
| `transpose_8x8_i16(&mut [[i16; 8]; 8])` | In-place transpose; SSE2 unpack network on x86_64. |
| `transpose_8x8_i16_portable` | Portable reference. |
| `transpose_4x4_f32(&mut [[f32; 4]; 4])` | In-place transpose; SSE unpack/move network on x86_64. |
| `transpose_4x4_f32_portable` | Portable reference. |

Interleaving and conversion:

| Function | Description |
|---|---|
| `interleave_stereo_i16(left, right, output)` | Planar to `[l0, r0, l1, r1, ...]`. Panics unless `left.len() == right.len()` and `output.len() == 2 * left.len()`. |
| `deinterleave_stereo_i16(input, left, right)` | Inverse of the above. Panics on odd `input.len()` or any length mismatch. |
| `interleave_yuv420(y, u, v, output)` | I420 (Y, U, V planes) to NV12: Y copied, then `u0, v0, u1, v1, ...`. Requires `y.len() == 4 * u.len()`, `u.len() == v.len()`, `output.len() == y.len() + 2 * u.len()`. |
| `unpack_i8_to_i16(data, output)` | Sign-extending widen of a byte slice. Panics unless lengths match. |

Lane permutation:

| Function | Description |
|---|---|
| `permute_f32(v, idx)` | `result[i] = v[idx[i] & 7]` for `F32x8`; indices are `Simd<u32, 8>`; lanes move bit-for-bit. |
| `permute_i32(v, idx)` | Same for `I32x8`. |
| `permute_f32_portable`, `permute_i32_portable` | Portable references. |
| `permute(v, idx)` | Generic `Simd<T, N>`: `result[i] = v[idx[i] % N]`; out-of-range indices wrap modulo `N`; never panics. |

Re-exports from `tpt-simd-saturate`:

| Function | Description |
|---|---|
| `pack_i16_to_i8` | Saturating narrow of 16 `i16` lanes to `i8`. |
| `pack_i16_to_i8_slice` | Slice form of the above. |

## Feature flags

The crate is `#![no_std]` in every configuration.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std` and `tpt-simd-saturate/std`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` and `tpt-simd-saturate/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only` and `tpt-simd-saturate/scalar-only`, and disables this crate's SSE2 transposes and AVX2 permutes so the portable code is always used. |

## Platform and target support

- x86_64 with SSE2 (baseline): `transpose_8x8_i16` and `transpose_4x4_f32` use unpack networks.
- x86_64 with `avx2` enabled at compile time: `permute_f32` (`vpermps`) and `permute_i32` (`vpermd`).
- Every other target, including aarch64 and 32-bit x86: portable code; there is no hand-written NEON path.
- Unsafe code is limited to the intrinsic paths, each with a `SAFETY` comment. All public functions are safe.

## Performance

Benchmarks live in `benches/permute.rs`; run `cargo bench -p tpt-simd-permute`, preferably with `RUSTFLAGS="-C target-cpu=native"`. Results for the workspace are collected in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-saturate`: home of `pack_i16_to_i8` and `pack_i16_to_i8_slice` (dependency).
- `tpt-simd-select`: index-based lane selection (`permutevar8x32` pattern with an `& 7` rule).
- `tpt-simd-gather`, `tpt-simd-scatter`: non-contiguous loads and stores.
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
