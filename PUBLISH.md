# Publishing tpt-simd to crates.io

Release checklist for the 0.1.0 publish. Publishing is manual (see `.github/workflows/release.yml`):
the release workflow only verifies and packages; nothing is uploaded automatically.

Written for: the maintainer doing the release.

## Status (as of 2026-10-12)

| Check | Result |
| --- | --- |
| `cargo check --workspace --all-targets` | Pass (last recorded run) |
| `cargo test --workspace --features std` | Pass: 890 passed, 0 failed, 2 ignored (last recorded run, not re-run for the 0857dce dry run) |
| `cargo publish --workspace --dry-run` | Pass at `0857dce`, without `--allow-dirty`: 41 of 41 crates packaged and verified, no errors or warnings |
| crates.io name availability (41 names) | All free (HTTP 404 from the crates.io API, re-checked at `0857dce`) |
| Working tree | Clean at `0857dce` |
| Release tag | `v0.1.0` does not exist yet |
| Per-crate docs | Every publishable crate has a README, a CHANGELOG, keywords, categories and an `examples/` program |

Publishable crates: 41 (the `tpt-simd` umbrella plus 40 focused crates).
Unpublished (`publish = false`): `tpt-simd-testutil` (dev-dependency only), `xtask`, `tpt-simd-fuzz` (`fuzz/`), and `bench-vs`.

### Fix made while preparing

`tpt-simd-core` and other crates list `tpt-simd-testutil` as a dev-dependency. The workspace entry
had `version = "0.1.0"`, so `cargo publish` tried to resolve it from crates.io and failed with
`no matching package named tpt-simd-testutil found`. The workspace entry is now path-only
(`tpt-simd-testutil = { path = "tpt-simd-testutil" }`), so Cargo strips the dev-dependency from the
published manifests. No normal (non-dev) dependency on `testutil` exists.

## Before you publish

- [x] Working tree is clean and committed (`0857dce`).
- [x] Dry run passes without `--allow-dirty` from `0857dce`.
- [ ] Decide the version. Everything is `0.1.0` via `[workspace.package]`.
- [ ] `cargo login` with a crates.io API token (token scope: `publish-new` and `publish-update`).
- [ ] If any commit lands before publishing, re-run the dry run from the new commit:
      `cargo publish --workspace --dry-run`
- [ ] Semver policy and release process are drafted in `docs/release.md`. Maintainer review is still
      needed; see `todo.md` under "Release".

## Publish order

Dependency order, computed from the normal (non-dev) dependencies between the 41 publishable crates. Dev-dependencies
are not part of the order. Publish one crate at a time, and wait for each to appear in the index before the next
(`cargo publish` does this automatically for each call). The umbrella `tpt-simd` is last because it depends on
almost everything else.

Tick a box once the crate is live on crates.io (check with `https://crates.io/api/v1/crates/<name>`, which returns
HTTP 200 when the crate exists). Update this list as you go.

Progress as of 2026-10-12 (41 of 41 live):

- [x] 1. tpt-simd-audio
- [x] 2. tpt-simd-bits
- [x] 3. tpt-simd-dct
- [x] 4. tpt-simd-distance
- [x] 5. tpt-simd-geom
- [x] 6. tpt-simd-half
- [x] 7. tpt-simd-hash
- [x] 8. tpt-simd-ntt
- [x] 9. tpt-simd-pixel
- [x] 10. tpt-simd-search
- [x] 11. tpt-simd-sort
- [x] 12. tpt-simd-text
- [x] 13. tpt-simd-transpose
- [x] 14. tpt-simd-vector
- [x] 15. tpt-simd-core
- [x] 16. tpt-simd-aligned
- [x] 17. tpt-simd-blas
- [x] 18. tpt-simd-blend
- [x] 19. tpt-simd-butterfly
- [x] 20. tpt-simd-compare
- [x] 21. tpt-simd-convolve
- [x] 22. tpt-simd-fixed
- [x] 23. tpt-simd-gather
- [x] 24. tpt-simd-horizontal
- [x] 25. tpt-simd-interpolate
- [x] 26. tpt-simd-math
- [x] 27. tpt-simd-mul
- [x] 28. tpt-simd-rng
- [x] 29. tpt-simd-rounding
- [x] 30. tpt-simd-saturate
- [x] 31. tpt-simd-scatter
- [x] 32. tpt-simd-shift
- [x] 33. tpt-simd-sparse
- [x] 34. tpt-simd-window
- [x] 35. tpt-simd-complex
- [x] 36. tpt-simd-dot
- [x] 37. tpt-simd-permute
- [x] 38. tpt-simd-reduce
- [x] 39. tpt-simd-select
- [x] 40. tpt-simd-matrix
- [x] 41. tpt-simd

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
cargo publish -p tpt-simd-audio
cargo publish -p tpt-simd-bits
# ... continue down the list above
```

If a publish fails partway, the crates already published stay published. Re-run only the remaining
crates from the list. Do not republish a version that already exists; bump the version instead.

## After publishing

- [x] Crates live: all 41 return HTTP 200 at version 0.1.0 on the crates.io API (checked 2026-10-13).
- [ ] Verify docs.rs builds for each crate (not yet checked).
- [x] Tagged `v0.1.0` at `1369dea`, the commit whose source is in the published tarballs (checked against
      `tpt-simd-sort` and `tpt-simd-text`). Pushed to origin, which triggers `release.yml`. Later commits
      (through `7440c7c`) change `tpt-simd-matrix`, `tpt-simd-sort`, `tpt-simd-text` and `tpt-simd-blas` and are
      unreleased; they belong in 0.1.1.
- [ ] Create the GitHub release from `v0.1.0`. Notes are in the scratchpad file `release-notes-v0.1.0.md`.
      `gh` is not installed here, so create it at https://github.com/tpt-solutions/tpt-simd/releases/new?tag=v0.1.0.
- [ ] Tick the publish items in `todo.md`.

## Rollback

`cargo yank --version 0.1.0 -p <crate>` stops new projects from resolving the version. It does not
delete the crate, and the name stays taken. Yank per crate, in reverse dependency order if you need to
back out the whole set.
