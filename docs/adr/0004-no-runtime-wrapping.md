# ADR 0004: Own implementations by default; incumbents only as test oracles

Status: **accepted**. Applies to Phase 11 and later crates. Earlier crates are unaffected.

## Context

Phase 11 added crates that wrap existing ones (`tpt-simd-hash` over `crc32fast` and `simd-adler32`, `tpt-simd-text` over `simdutf8` and `faster-hex`, `tpt-simd-search` over `memchr`). Comparing them against the wrapped crates showed no meaningful speed or correctness gain. Our kernels match the incumbents, so the wrappers mostly added dependencies.

Costs of a runtime wrapper: another repository to track, licence drift (three candidates were already off the allow-list), API churn, MSRV coupling, and a supply-chain surface.

Maintenance: SIMD instruction sets change slowly. SSE2, AVX2 and NEON have been stable for a decade or more. What does change is the arrival of new ISAs and per-CPU tuning, which our own kernels must track anyway.

## Decision

1. Each crate ships its own implementation and has **no runtime dependencies** by default.
2. An incumbent may be used in two ways only:
   - as a **dev-dependency** test oracle for differential tests and fuzzing, or
   - as an **optional, default-off** cargo feature where reimplementing is costly (currently `simdutf8` in `tpt-simd-text` and `memchr` in `tpt-simd-search`).
3. `tpt-simd-half` keeps the `half` crate as its one runtime dependency. `f16` and `bf16` are ecosystem interchange types, and reimplementing them would break interop.
4. Speed claims require a criterion bench against the incumbent. A missing benchmark means no claim.
5. Licence rules are unchanged: an incumbent must pass `cargo deny check licenses`, even when it is only a dev-dependency.

## Consequences

- Dependency graphs stay small, and the licence gate covers only what ships.
- Each own kernel needs differential tests against its incumbent so the oracle keeps it honest.
- Some own kernels will be slower than the best incumbent. Those gaps are recorded in `docs/benchmarks.md`, not hidden.
- The `Cargo.lock` shrinks as runtime wrappers are removed.
