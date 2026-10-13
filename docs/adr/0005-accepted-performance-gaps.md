# ADR 0005: Accepted performance gaps against incumbents

Status: **accepted**.

## Context

Round 5 of [incumbent-benchmarks.md](../incumbent-benchmarks.md) measured the crates on a new
machine against the incumbent crates. Several kernels are behind, and some of the gaps are
the price of a deliberate design decision rather than unfinished work. Those gaps should be
recorded as decisions, so that nobody reopens them by accident.

## Decision

1. **`hex_decode` validates the whole input before it writes.** An error leaves `dst`
   untouched. That needs a validation pass and a write pass over the input. A single-pass
   decoder would be faster at small sizes, but would leave `dst` partly written on error.
   The guarantee is part of the public contract and is covered by
   `hex_decode_rejects_invalid_byte_at_every_position_without_writing`. We keep it.
   At 4 KB and 256 KB the cost is about 1.1–1.4× `faster-hex`. At 64 B it is about 1.7×,
   mostly fixed per-call cost.

2. **`tpt-simd-sort` stays `#![deny(unsafe_code)]`.** A competitive generic `sort_unstable`
   needs unchecked indexing in its partition, which this crate does not allow. The generic sort
   is 1.5× behind the standard library at 100 K elements and above. The radix `sort_u32`,
   which needs no unsafe code, is ahead of the standard library by about 2.5×. We keep the
   safe policy and document the generic gap. This decision should be revisited if the sort
   becomes a bottleneck for a real user.

3. **`tpt-simd-matrix` keeps the array-typed API, and adds a register-typed one.** `[[f32; N]; N]`
   is returned through memory, which cost 1.5–2.3× against `glam` on transpose and 1.7× on the
   3×3 inverse. The maintainer chose to add `Mat4` and `Mat3` (rows held as 128-bit lane values)
   alongside the array functions, not to replace them. The array functions and their results
   are unchanged. Round 9 in [incumbent-benchmarks.md](../incumbent-benchmarks.md) has the
   measurements: the register 3×3 inverse is ahead of `glam`, the register 4×4 multiply is
   about 1.07×, and the register 4×4 transpose is still about 1.6× behind.

4. **`tpt-simd-blas` gemm at very small sizes is not a goal yet.** At 64 the gap is about
   1.4× to `faer`, which comes from packing overhead. Closing it means changing the packing
   strategy, which is a substantial change for a small-size case.

## Consequences

- The benchmark tables must show these rows as "behind, by decision", not as open bugs.
- Re-measuring on another machine does not reopen these decisions. Only a change in the
  contract, the crate policy, or the API does.
- Each decision names the thing that would reopen it, so a future change can be judged
  against it.
