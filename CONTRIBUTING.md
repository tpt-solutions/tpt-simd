# Contributing to tpt-simd

Thanks for your interest in tpt-simd!

## Licensing of contributions and suggestions

Unless you explicitly state otherwise, any code or text intentionally submitted
to this project by you (for example in an issue), as defined in the Apache-2.0 license,
shall be dual licensed under `MIT OR Apache-2.0`, without any additional terms
or conditions.

Dependencies must be MIT-compatible (permissive). Apache-2.0-only, GPL, LGPL
and AGPL dependencies are not accepted.

## Per-crate definition of done

Every crate needs:

- A scalar reference implementation and SIMD implementation(s) for each in-scope target
- Unit tests against the scalar reference (NaN/inf, overflow, empty input, tail lengths)
- proptest property tests (SIMD == scalar)
- A doc comment on every public item, with an example and `# Safety` on unsafe fns
- A criterion benchmark against the scalar version
- A passing `no_std` build; clippy, rustfmt and `cargo doc` clean

See [todo.md](todo.md) for the full checklist.

## Workflow

tpt-simd accepts **issues only**. Pull requests are not accepted and will be
closed unmerged.

- Report bugs, request features and propose changes by opening an issue.
- Include a minimal reproduction, the target CPU / feature set
  (`-C target-cpu=...`, `std` / `scalar-only` / `nightly`) and your Rust version.
- For accuracy or performance reports, include the input, the observed and
  expected results (or benchmark numbers) and how you measured them.

## Conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
