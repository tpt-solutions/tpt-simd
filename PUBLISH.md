# Publishing tpt-simd to crates.io

Release checklist. 0.1.0 is published (41 of 41 live); this file now tracks the 0.1.1 patch release, and the 0.1.0 record is kept below. Publishing is manual (see `.github/workflows/release.yml`):
the release workflow only verifies and packages; nothing is uploaded automatically.

Written for: the maintainer doing the release.

## 0.1.1 release (in preparation)

Why: 51 commits since `v0.1.0` (`1369dea`) are unpublished, and one is a correctness fix. **`tpt-simd-text` 0.1.0 has a
UTF-8 validator that accepts some invalid input** (lead byte `0xFD`/`F5..FF` before a continuation byte, and
`C0`/`C1`/`F5..FF` at the end of a 32-byte block, on both the AVX2 and NEON paths). The fix is `b60f2a1`. Also in
0.1.1: aarch64 NEON paths in 14 crates, `rand_core` impls for `Xoshiro256pp`, new `Mat4`/`Mat3`, varint and
bitpacking codecs, f16/bf16 arithmetic, FFT-based DCT, and a faster `select_nth`. The per-crate list is in each
CHANGELOG under `[0.1.1]`.

| Check | Result |
| --- | --- |
| Semver (`cargo semver-checks check-release --workspace --baseline-version 0.1.0`), re-run 2026-10-18 on the working tree | Pass: 41 of 41 "no semver update required". The tree adds public items (Phase 11 work, see the decision below), which are non-breaking |
| `cargo test --workspace --features std` | Pass: 1065 passed, 0 failed, 2 ignored (2026-10-18, default build); same count with `-C target-cpu=x86-64-v3 -C target-feature=+pclmulqdq` (AVX2 paths) |
| `cargo xtask ci`, `cargo xtask no-std`, `cargo xtask msrv` (Rust 1.95), `cargo hack check --workspace --feature-powerset` with `-D warnings` | Pass on 2026-10-18 (exit 0 each; 523 feature combinations) |
| `cargo publish --workspace --dry-run --allow-dirty` | Pass 2026-10-18: 41 of 41 packaged and verified at 0.1.1, no warnings. Re-run without `--allow-dirty` after committing |
| CI and bench on GitHub (master) | Green as reported by the maintainer, 2026-10-17; re-check on the release commit |
| Version bump | Done: `[workspace.package]` and every `version = "0.1.x"` in the dependency tables |
| CHANGELOGs | Done: `[Unreleased]` split into `[0.1.1] - 2026-10-17` and `[0.1.0] - 2026-10-12` in all 41 crates. On 2026-10-18 the Phase 13 fixes (`tpt-simd`, `tpt-simd-hash`, `tpt-simd-window`) were moved into `[0.1.1]`. The date still says 2026-10-17; set it to the real release date |
| Working tree | Uncommitted. Commit, wait for green CI, then tag |

Before publishing 0.1.1:

- [ ] Commit the bump and CHANGELOGs; push; confirm CI and bench are green on that commit.
- [ ] `cargo publish --workspace --dry-run` without `--allow-dirty`.
- [ ] Publish in the order below (reset to unticked for 0.1.1), then tag `v0.1.1` on the published commit.
- [ ] After publishing, yank `tpt-simd-text` 0.1.0 (`cargo yank --version 0.1.0 tpt-simd-text`) so new projects do not resolve the
      validator bug. Existing lockfiles keep working. Decide whether to yank the other 0.1.0 crates (not needed for correctness).
- [ ] Switch the `tpt-math` `simd` path dependencies (`tpt-simd-blas`, `tpt-simd-sparse`) to `version = "0.1.1"`.

### Commit split (2026-10-18)

The 0.1.1 release is committed separately from the Phase 11 work. The release commit holds the version bump, the `[0.1.1]` CHANGELOG entries, the Phase 13 fixes, and the release-gate fixes (a formatting pass and two blas lint/doc fixes). The Phase 11 work is the next commit. It is not part of 0.1.1, and its CHANGELOG entries sit under `[Unreleased]` for the next release. Publish and tag 0.1.1 from the release commit. Nothing has been pushed, published or tagged.

## 0.1.0 record

### Status (as of 2026-10-12)

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

#### Fix made while preparing

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

Progress for 0.1.1 (0 of 41 live). The 0.1.0 run was 41 of 41, every box below ticked:

- [ ] 1. tpt-simd-audio
- [ ] 2. tpt-simd-bits
- [ ] 3. tpt-simd-dct
- [ ] 4. tpt-simd-distance
- [ ] 5. tpt-simd-geom
- [ ] 6. tpt-simd-half
- [ ] 7. tpt-simd-hash
- [ ] 8. tpt-simd-ntt
- [ ] 9. tpt-simd-pixel
- [ ] 10. tpt-simd-search
- [ ] 11. tpt-simd-sort
- [ ] 12. tpt-simd-text
- [ ] 13. tpt-simd-transpose
- [ ] 14. tpt-simd-vector
- [ ] 15. tpt-simd-core
- [ ] 16. tpt-simd-aligned
- [ ] 17. tpt-simd-blas
- [ ] 18. tpt-simd-blend
- [ ] 19. tpt-simd-butterfly
- [ ] 20. tpt-simd-compare
- [ ] 21. tpt-simd-convolve
- [ ] 22. tpt-simd-fixed
- [ ] 23. tpt-simd-gather
- [ ] 24. tpt-simd-horizontal
- [ ] 25. tpt-simd-interpolate
- [ ] 26. tpt-simd-math
- [ ] 27. tpt-simd-mul
- [ ] 28. tpt-simd-rng
- [ ] 29. tpt-simd-rounding
- [ ] 30. tpt-simd-saturate
- [ ] 31. tpt-simd-scatter
- [ ] 32. tpt-simd-shift
- [ ] 33. tpt-simd-sparse
- [ ] 34. tpt-simd-window
- [ ] 35. tpt-simd-complex
- [ ] 36. tpt-simd-dot
- [ ] 37. tpt-simd-permute
- [ ] 38. tpt-simd-reduce
- [ ] 39. tpt-simd-select
- [ ] 40. tpt-simd-matrix
- [ ] 41. tpt-simd

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
