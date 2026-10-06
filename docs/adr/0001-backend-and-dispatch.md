# ADR 0001: Backend and dispatch strategy

Status: accepted

## Context

The spec assumed `core::simd` was stable. It is nightly-only. We still need a
stable, `no_std`, MIT-compatible foundation, and a way to use architecture
intrinsics where LLVM auto-vectorisation is not enough.

## Decisions

1. **Backend.** `tpt-simd-vector` provides `Simd<T, N>` / `SimdMask<T, N>` as
   array wrappers (`#[repr(transparent)]` over `[T; N]`) with lane-loop
   operations. It is safe (`forbid(unsafe_code)`), `no_std`, and stable.
   `tpt-simd-core` re-exports it. The `nightly` feature is **reserved** for a
   `core::simd`-backed implementation of the same API; it currently changes
   nothing.
2. **Lane traits live in `tpt-simd-vector`**, not `tpt-simd-core` as the spec
   says, because core depends on vector (not the other way round). Core
   re-exports them, so users see the spec's paths.
3. **Compile-time dispatch.** Fast paths are selected with
   `cfg(all(target_arch = "...", target_feature = "..."))` inside each crate,
   always next to a portable scalar/lane-loop implementation that is the
   behavioural reference. There are no per-architecture Cargo features;
   build with `-C target-cpu=...` / `-C target-feature=...` to enable them.
   The `scalar-only` feature forces the portable path everywhere (used to
   test the fallback on any machine).
4. **Runtime detection** (`tpt_simd_core::detect::Features::runtime`, `std`
   only) exists for applications that ship several builds; crates themselves
   do not dispatch at runtime.
5. **Semantics are identical across paths.** Every fast path must be
   bit-identical to the portable path (floats: documented tolerance only for
   reductions that reassociate). Tests compare both.
6. **Float math** uses `libm` so results do not depend on the target's libm.

## Consequences

* Default builds (x86_64 baseline = SSE2) get whatever LLVM auto-vectorises.
  Users opt into AVX2/FMA/NEON by compiling with the right target features.
* Adding a nightly backend later does not change the public API.
