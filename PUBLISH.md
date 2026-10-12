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
- [ ] Optional but recommended: add per-crate `LICENSE-MIT` and `LICENSE-APACHE` files. The READMEs link to the
      root license files (`../LICENSE-APACHE`), and those are not in the `.crate` tarballs, so the tarballs
      carry only the SPDX `license` field, not the license text.
- [ ] Semver policy and release process are drafted in `docs/release.md`. Maintainer review is still
      needed; see `todo.md` under "Release".

## Publish order

Dependency order, computed from the normal (non-dev) dependencies between the 41 publishable crates. Dev-dependencies
are not part of the order. Publish one crate at a time, and wait for each to appear in the index before the next
(`cargo publish` does this automatically for each call). The umbrella `tpt-simd` is last because it depends on
almost everything else.

 1. tpt-simd-audio
 2. tpt-simd-bits
 3. tpt-simd-dct
 4. tpt-simd-distance
 5. tpt-simd-geom
 6. tpt-simd-half
 7. tpt-simd-hash
 8. tpt-simd-ntt
 9. tpt-simd-pixel
10. tpt-simd-search
11. tpt-simd-sort
12. tpt-simd-text
13. tpt-simd-transpose
14. tpt-simd-vector
15. tpt-simd-core
16. tpt-simd-aligned
17. tpt-simd-blas
18. tpt-simd-blend
19. tpt-simd-butterfly
20. tpt-simd-compare
21. tpt-simd-convolve
22. tpt-simd-fixed
23. tpt-simd-gather
24. tpt-simd-horizontal
25. tpt-simd-interpolate
26. tpt-simd-math
27. tpt-simd-mul
28. tpt-simd-rng
29. tpt-simd-rounding
30. tpt-simd-saturate
31. tpt-simd-scatter
32. tpt-simd-shift
33. tpt-simd-sparse
34. tpt-simd-window
35. tpt-simd-complex
36. tpt-simd-dot
37. tpt-simd-permute
38. tpt-simd-reduce
39. tpt-simd-select
40. tpt-simd-matrix
41. tpt-simd

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

- [ ] Verify each crate page on crates.io and that `docs.rs` builds (docs.rs builds run after upload).
- [ ] Tag the release: `git tag v0.1.0 && git push origin v0.1.0`. This triggers `release.yml`, which
      re-runs the test and package checks against the tagged commit.
- [ ] Create the GitHub release from the tag, using the CHANGELOG entries as the notes.
- [ ] Tick the publish items in `todo.md`.

## Rollback

`cargo yank --version 0.1.0 -p <crate>` stops new projects from resolving the version. It does not
delete the crate, and the name stays taken. Yank per crate, in reverse dependency order if you need to
back out the whole set.
