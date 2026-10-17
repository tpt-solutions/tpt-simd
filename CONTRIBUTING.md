# Contributing to tpt-simd

Thanks for your interest in tpt-simd!

> **Issues only.** tpt-simd accepts issues, not pull requests. Pull requests
> will be closed unmerged. Please [open an issue](../../issues) to report a bug,
> request a feature or propose a change. See [Workflow](#workflow) for what to include.

## Licensing of contributions and suggestions

Unless you explicitly state otherwise, any code or text intentionally submitted
to this project by you (for example in an issue), as defined in the Apache-2.0 license,
shall be dual licensed under `MIT OR Apache-2.0`, without any additional terms
or conditions.

Dependencies must be MIT-compatible (permissive). Apache-2.0-only, GPL, LGPL
and AGPL dependencies are not accepted.

Crates implement their own code by default and have no runtime dependencies.
An external crate may be used as a dev-dependency test oracle, or behind a
default-off feature, but not as an unconditional runtime dependency unless an
ADR says so (see `docs/adr/0004-no-runtime-wrapping.md`).

## Per-crate definition of done

Every crate needs:

- A scalar reference implementation and SIMD implementation(s) for each in-scope target
- Unit tests against the scalar reference (NaN/inf, overflow, empty input, tail lengths)
- proptest property tests (SIMD == scalar)
- A doc comment on every public item, with an example and `# Safety` on unsafe fns
- A criterion benchmark against the scalar version
- A passing `no_std` build; clippy, rustfmt and `cargo doc` clean

See [todo.md](todo.md) for the full checklist.

## Local checks

The workspace ships a small task runner (`xtask`, no dependencies) that runs
the same checks as CI:

```sh
cargo xtask ci       # fmt, clippy -D warnings, cargo doc, tests (std and scalar-only)
cargo xtask no-std   # no_std builds for thumbv7em-none-eabihf and wasm32 (if installed)
cargo xtask msrv     # cargo check on Rust 1.95 (if installed)
cargo xtask cpu      # which SIMD features this build and this CPU support
```

## Workflow

tpt-simd accepts **issues only**. Pull requests are not accepted and will be
closed unmerged.

- Report bugs, request features and propose changes by opening an issue.
- Include a minimal reproduction, the target CPU / feature set
  (`-C target-cpu=...`, `std` / `scalar-only` / `nightly`) and your Rust version.
- For accuracy or performance reports, include the input, the observed and
  expected results (or benchmark numbers) and how you measured them.
