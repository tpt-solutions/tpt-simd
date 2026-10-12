# Semver policy and release process

Written for: the maintainer cutting a release. Draft; the policy below is the proposed default and
should be confirmed before the 0.1.0 publish. The step-by-step publish commands are in
[PUBLISH.md](../PUBLISH.md).

## Versioning

- All 28 published crates share one version (`[workspace.package] version`). A release bumps every
  crate, even when only one changed, so the set stays easy to reason about.
- Until 1.0.0, a minor bump (`0.x` -> `0.(x+1)`) may contain breaking changes; a patch bump
  (`0.x.y` -> `0.x.(y+1)`) must not.
- From 1.0.0: semver as written in the [Cargo book](https://doc.rust-lang.org/cargo/reference/semver.html).

## What counts as breaking

Breaking (needs a minor bump before 1.0, a major after):

- removing or renaming a public item, or changing its signature, bounds or return type;
- changing a `#[repr]`, a public field, or the set of variants of a public enum;
- adding a required method to a public trait, or a new item that makes a glob import ambiguous;
- raising `rust-version` (MSRV). Raising it is a minor-bump change and must be listed in the CHANGELOG;
- changing a documented result beyond its stated tolerance (see the float-determinism tiers in
  [ADR 0003](adr/0003-runtime-dispatch-and-float-tolerance.md)). Tier 1 results are bit-exact, so any
  change to them is breaking; Tier 2 changes within the documented ulp bound are not.

Not breaking:

- adding a function, type, or trait method with a default implementation;
- adding a Cargo feature that is off by default; adding a feature that is on by default is breaking
  only if it changes behaviour;
- faster code, or a different SIMD path, with the same results;
- changing private items, docs, or examples.

Feature flags: removing a feature is breaking. The `nightly` and `scalar-only` features are kept
for this reason (see the book's feature table).

## Checks before a release

Run from a clean tree at the commit being published:

1. `cargo xtask ci` (fmt, clippy `-D warnings`, doc, tests, scalar-only).
2. `cargo xtask no-std` (bare-metal and wasm targets).
3. `cargo xtask msrv` or `cargo +<rust-version> check --workspace --all-targets`.
4. `cargo hack check --workspace --feature-powerset` (the `feature-powerset` CI job).
5. `cargo semver-checks check-release --workspace` against the last published version
   (not yet run for 0.1.0, since nothing is published; for 0.1.0 this step is skipped and recorded).
6. `cargo publish --workspace --dry-run`.
7. Each crate's CHANGELOG has an entry for the release; `[Unreleased]` is renamed to the version and date.

## Release steps

1. Confirm the checks above pass on `master`, and that the CI runs on the release commit are green.
2. Bump `[workspace.package] version` and the path versions in the workspace dependency table.
3. Move `[Unreleased]` to `[x.y.z] - YYYY-MM-DD` in every crate's CHANGELOG.
4. Commit and tag `vX.Y.Z` on `master`.
5. Publish in the order listed in [PUBLISH.md](../PUBLISH.md), waiting for each crate to reach the index.
6. Create the GitHub release from the tag with the CHANGELOG entries as notes.
7. Announce only after step 5 completes, so the links in the announcement resolve.

## Yanking

A bad release is yanked (`cargo yank --version X.Y.Z <crate>`) rather than deleted. A fix ships as the
next patch release; yanking does not replace a release note.
