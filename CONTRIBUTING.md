# Contributing to tpt-simd

Thanks for your interest in contributing!

## Licensing of contributions

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
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

1. Open an issue for non-trivial changes before starting.
2. Branch from `master` and keep PRs focused.
3. Run `cargo fmt`, `cargo clippy -- -D warnings` and `cargo test` before pushing.

## Conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
