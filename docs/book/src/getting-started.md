# Getting started

## Install

Add one (or more) of the member crates to your `Cargo.toml`. All crates are
`no_std` by default; enable the `std` feature where you need `std`-based
layout allocation, runtime CPU detection or `std`-internals for float
rounding.

```toml
[dependencies]
tpt-simd-core = "0.1.0"
tpt-simd-complex = "0.1.0"
tpt-simd-fixed = "0.1.0"
```

Use the umbrella crate to re-export everything under `tpt_simd`:

```toml
[dependencies]
tpt-simd = "0.1.0"
```

```rust
use tpt_simd_fixed::Fixed;
use tpt_simd_core::F32x8;

let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));
let b: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(2.5));
let result = a * b; // 1.5 * 2.5 = 3.75 with correct rounding
```

## Backend and dispatch

The workspace uses **compile-time dispatch** (`cfg(target_feature)`) as the
default. What "native" means for your build is decided by the flags you pass
to the compiler, not by a runtime probe inside the library:

- Plain `cargo build` on x86_64 gives you SSE2 and whatever LLVM auto-vectorises.
- `RUSTFLAGS="-C target-cpu=native"` (or `-C target-feature=+avx2,+fma`)
  upgrades the fast paths. This is the recommended way to reach the documented
  speedups.
- `tpt-simd-blas` and `tpt-simd-core` additionally honour the `runtime-dispatch`
  feature (implies `std`), which probes for AVX2+FMA once per call and caches
  the result. This is a kernel-crate-only feature.

See [ADR 0001](crate-overview.md#backend-dispatch) for the full decision
record.

### Check which path your machine takes

`cargo xtask cpu` (run from the repository) prints which ISA extensions this
build was compiled with and which the CPU reports. The output below is from a
Windows x86_64 machine built with plain `cargo xtask cpu`, so the SSE2 path is
compiled and the AVX2 path is available but not used:

```text
tpt-simd CPU report
  native vector width: 128 bits (4 x f32 lanes)
  x86 / x86_64:
  sse2     compiled + available
  sse4.1   available, not compiled (portable path)
  avx      available, not compiled (portable path)
  avx2     available, not compiled (portable path)
  fma      available, not compiled (portable path)
  avx512f  -
  aarch64:
  neon     -
  sve      -
  riscv:
  rvv      -
```

Rebuild with `RUSTFLAGS="-C target-cpu=native"` and the same machine reports
a 256-bit native width, with `sse4.1`, `avx`, `avx2` and `fma` moving to
`compiled + available`.

## Try the examples

The crate READMEs and the [cookbook](cookbook/index.md) contain runnable
examples. Every public item carries a doc comment with an example, and the
spec's Appendix B examples are verified to compile as doctests (see
[the verification log](known-issues.md#appendix-b-doctests)).

## Contributing

This project accepts **issues only**; pull requests are not accepted and will
be closed unmerged. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the rules.

## Licensing

Dual MIT OR Apache-2.0. Contributions are dual-licensed as above unless
stated otherwise.
