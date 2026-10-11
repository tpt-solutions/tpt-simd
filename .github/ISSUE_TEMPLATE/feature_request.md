---
name: Feature request
about: Propose a new kernel, an API change or a new target
labels: enhancement
---

## Problem

<!-- The DSP, codec or numeric task you are trying to solve. -->

## Proposed API

```rust
// sketch of the function or type you would like to call
```

## Scalar fallback

<!-- Every SIMD kernel needs a scalar reference that is used for tests and as the fallback. Say what it should compute, including edge cases (empty input, NaN, overflow, tail lengths). -->

## Alternatives considered

## Target architectures affected

- [ ] x86_64 (SSE / AVX2 / AVX-512)
- [ ] aarch64 (NEON / SVE)
- [ ] RISC-V (RVV)
- [ ] Scalar / `no_std`
