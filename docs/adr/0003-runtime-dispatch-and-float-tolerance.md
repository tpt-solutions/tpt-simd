# ADR 0003: Runtime dispatch for kernel crates and float-determinism tiers

Status: **accepted** (amends ADR 0001 for the Phase 10 numeric crates only). `runtime-dispatch` is implemented in `tpt-simd-blas` and forwarded by the umbrella crate; see Implementation status.

## Context

ADR 0001 chose compile-time dispatch (`cfg(target_feature)`). Consequences measured
in `docs/benchmarks.md`:

* A library user who does not build with `-C target-cpu=native` (or explicit
  `+avx2,+fma`) silently gets the portable path: e.g. `gemm` without FMA/AVX2 is
  an untimed, much slower kernel, and FMA-based paths fall back to per-lane `libm`.
* Phase 10 targets are libraries (tpt-math) whose *consumers* control build flags,
  not tpt-math itself.

ADR 0001 also requires bit-identical results to the scalar reference. The Phase 10
crates (`blas`, `reduce`, `math`, `rng`, `sparse`) reassociate reductions and use
polynomial approximations; they already document tolerances.

## Decision (proposed)

1. **Keep compile-time dispatch as the default** everywhere. No behaviour change
   for `no_std` users or for existing crates.
2. **Add an opt-in `runtime-dispatch` feature (implies `std`) to coarse-grained
   kernel crates only** (`blas`, `reduce`, `math` slice APIs, `sparse`, later
   `convolve`). Dispatch happens **once per call** at the slice-level entry point,
   never per vector/lane function:
   * x86_64: `is_x86_feature_detected!("avx2") && ("fma")`, result cached in an
     atomic; the AVX2+FMA kernel is a separate `#[target_feature(enable = "avx2,fma")]`
     `unsafe fn`, called only after the check.
   * Other targets: unchanged (compile-time). aarch64 NEON is baseline so needs no
     runtime check; SVE/RVV are decided when implemented.
   * `scalar-only` forces the portable path regardless.
3. **Fine-grained vector functions (`Simd` ops, `cmp_*`, `blend_*`, shifts, ...)
   stay compile-time.** A runtime branch per 8-lane op costs more than it saves;
   such code gets speed from the build flags (documented in the README).
4. **Float-determinism tiers**, stated in each crate's docs:
   * *Tier 1 (bit-exact vs scalar reference, cross-target)*: all Phase 1-6 crates.
   * *Tier 2 (documented ulp/relative tolerance, may differ across targets or
     dispatch path)*: `blas`, `reduce` (except `min/max/arg*`, which are exact),
     `math`, `sparse`. Results are deterministic for a fixed build *and* CPU class.
     With `runtime-dispatch` the same binary can give slightly different results
     on different machines (FMA vs non-FMA rounding), so do not compare these
     outputs bit-for-bit across machines.
   * *Tier 3 (statistical)*: `rng` outputs are bit-reproducible for a seed, but
     normal/uniform-float transforms use approximations with documented error.
5. tpt-math adopts the kernels behind its own optional `simd` feature, which is
   **off by default** (binding rule), with the scalar path kept as the reference
   and used by tests that need exact values. Formal-verification consumers must
   stay on the scalar path (Tier 1 only).

## Consequences

* Users get full speed from a plain `cargo build --features "std runtime-dispatch"`.
* Adds `unsafe` + a detection cache to the kernel crates (covered by the unsafe policy
  in ADR 0002: `# Safety` docs, tests for both paths, Miri skips intrinsics).
* Tests must run both paths (force via `scalar-only` and via the dispatched path).
* Open question: expose a public `dispatch::active_backend()` for diagnostics.

## Alternatives considered

* Runtime dispatch everywhere: rejected (per-op branch cost, `no_std` breakage).
* Document build flags only: rejected as the sole answer (silent 3-30x slowdowns).
* `multiversion`/`pulp` crates: a dependency, and licences/maintenance to vet;
  revisit if hand-rolled dispatch grows.

## Implementation status

* `tpt-simd-blas`: `runtime-dispatch` feature done. `x86::available()` returns
  `true` under static AVX2+FMA, else detects once via `is_x86_feature_detected!`
  and caches in an `AtomicU8`; checked once per microkernel call (thousands of
  flops), and `scalar-only` still forces the portable path. Tests pass with the
  feature, with `scalar-only`, and with native flags.
  Measured gemm f32 256, plain build (no target flags): 2.06 ms -> 1.12 ms
  (about 1.8x); fully native build is 0.83 ms (the rest of the gemm code also
  benefits from AVX2 codegen).
* Umbrella `tpt-simd` forwards `runtime-dispatch`.
* `reduce`, `math`, `sparse`, `rng` use portable code only (no intrinsics), so
  they have nothing to dispatch yet; they gain from `-C target-cpu`/LLVM only.
