---
name: Bug report
about: Wrong result, panic, build failure, or a performance regression
labels: bug
---

<!-- Minimal reproduction first. A failing unit test or a 10-line `main` is ideal. -->

## What happened

<!-- Observed behaviour, next to what you expected. For accuracy issues, include the input values and both results. -->

## Reproduction

```rust
// minimal code here
```

## Environment

- tpt-simd crate(s) and version:
- `rustc -Vv` output:
- Target triple:
- Features enabled (`std`, `scalar-only`, `nightly`, `runtime-dispatch`):
- `RUSTFLAGS` / `-C target-cpu`, if set:

## CPU report

Paste the output of `cargo xtask cpu` (or `cargo run -p tpt-simd --features std --example cpu_report`) from the machine where the problem occurs.

```text

```

## Performance reports only

- How you measured (criterion bench name, or your own timing loop):
- Scalar baseline vs tpt-simd numbers:
