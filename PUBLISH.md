# Publishing tpt-simd to crates.io

Release checklist for the 0.1.0 publish. Publishing is manual (see `.github/workflows/release.yml`):
the release workflow only verifies and packages; nothing is uploaded automatically.

Written for: the maintainer doing the release.

## Status (as of 2026-10-12)

| Check | Result |
| --- | --- |
| `cargo check --workspace --all-targets` | Pass |
| `cargo test --workspace --features std` | Pass: 729 passed, 0 failed |
| `cargo publish --workspace --dry-run` | Pass: 28 of 28 crates packaged and verified, no warnings |
| crates.io name availability (28 names) | All free (HTTP 404 from the crates.io API) |

Publishable crates: 28. Unpublished: `tpt-simd-testutil` (`publish = false`, dev-dependency only).

### Fix made while preparing

`tpt-simd-core` and other crates list `tpt-simd-testutil` as a dev-dependency. The workspace entry
had `version = "0.1.0"`, so `cargo publish` tried to resolve it from crates.io and failed with
`no matching package named tpt-simd-testutil found`. The workspace entry is now path-only
(`tpt-simd-testutil = { path = "tpt-simd-testutil" }`), so Cargo strips the dev-dependency from the
published manifests. No normal (non-dev) dependency on `testutil` exists.

## Before you publish

- [ ] Working tree is clean and committed (the `Cargo.toml` change above is not yet committed).
- [ ] Decide the version. Everything is `0.1.0` via `[workspace.package]`.
- [ ] `cargo login` with a crates.io API token (token scope: `publish-new` and `publish-update`).
- [ ] Re-run the dry run from the commit you will publish:
      `cargo publish --workspace --dry-run`
- [ ] Optional but recommended: add per-crate `LICENSE-MIT` and `LICENSE-APACHE` files (currently only at
      the repo root, so the `.crate` tarballs contain only the SPDX `license` field, not the license text).
- [ ] Open items in `todo.md` under "Release": CHANGELOG/semver policy and release process are not written.

## Publish order

Dependency order, as Cargo itself computed it during the dry run. Publish one crate at a time, and wait
for each to appear in the index before the next (`cargo publish` does this automatically for each call).

 1. tpt-simd-vector
 2. tpt-simd-core
 3. tpt-simd-aligned
 4. tpt-simd-blas
 5. tpt-simd-blend
 6. tpt-simd-butterfly
 7. tpt-simd-compare
 8. tpt-simd-convolve
 9. tpt-simd-fixed
10. tpt-simd-gather
11. tpt-simd-horizontal
12. tpt-simd-interpolate
13. tpt-simd-math
14. tpt-simd-mul
15. tpt-simd-rng
16. tpt-simd-rounding
17. tpt-simd-saturate
18. tpt-simd-scatter
19. tpt-simd-shift
20. tpt-simd-window
21. tpt-simd-complex
22. tpt-simd-dot
23. tpt-simd-permute
24. tpt-simd-reduce
25. tpt-simd-select
26. tpt-simd-sparse
27. tpt-simd-matrix
28. tpt-simd

## Commands

Dry run (no upload):

```sh
cargo publish --workspace --dry-run
```

Publish everything in order (uploads; cannot be undone, only yanked):

```sh
cargo publish --workspace
```

Or publish one crate at a time, if you prefer to stop between steps:

```sh
cargo publish -p tpt-simd-vector
cargo publish -p tpt-simd-core
# ... continue down the list above
```

If a publish fails partway, the crates already published stay published. Re-run only the remaining
crates from the list. Do not republish a version that already exists; bump the version instead.

## After publishing

- [ ] Verify each crate page on crates.io and that `docs.rs` builds (docs.rs builds run after upload).
- [ ] Tag the release: `git tag v0.1.0 && git push origin v0.1.0`. This triggers `release.yml`, which
      re-runs the test and package checks against the tagged commit.
- [ ] Create the GitHub release from the tag.
- [ ] Tick the publish items in `todo.md`.

## Rollback

`cargo yank --version 0.1.0 -p <crate>` stops new projects from resolving the version. It does not
delete the crate, and the name stays taken. Yank per crate, in reverse dependency order if you need to
back out the whole set.
