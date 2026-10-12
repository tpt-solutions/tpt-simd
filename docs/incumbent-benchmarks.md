# Head-to-head benchmarks against incumbents

This page records how tpt-simd compares with established crates for the same
operations. The goal is parity: a kernel is at parity when its median time is
within about 10% of the fastest incumbent measured in the same run, or faster.

The `bench-vs` crate (unpublished) holds the benches. Per-crate kernel history
is in [benchmarks.md](benchmarks.md).

## How to read the numbers

- **Compare ratios within one run, not absolute times across runs.** On this
  machine the same incumbent moved by up to 3–4× between sessions (for example
  `gemm` f32 at 64 was 13.0 µs in one run and 4.1 µs in another), so only the
  ratio of two rows measured together means anything.
- **Ratio** = tpt-simd time ÷ incumbent time. Below 1.0 is faster.
- Changes under about 20% between runs are noise.

## Method

- **Machine:** Intel Core i5-1135G7 (Tiger Lake), Windows 11. It reports AVX2,
  FMA and VPCLMULQDQ to Rust's feature detection. It does not report `avx512f`,
  so AVX-512 kernels are not used here.
- **Harness:** Criterion. Release profile with `opt-level = 3`, `lto = "thin"`,
  `codegen-units = 1`.
- **Build flags:** none (`target-cpu` is not set). Kernels that need a CPU
  feature are chosen at runtime through the `runtime-dispatch` feature, which
  `bench-vs` enables for `tpt-simd-blas` and `tpt-simd-hash`. The umbrella crate
  `tpt-simd` enables it by default. The incumbents also choose kernels at runtime,
  so this is the fair comparison.
- **Reported value:** the criterion median of the run.

Reproduce a suite with, for example:

```sh
cargo bench -p bench-vs --bench hash_vs
```

## Summary (refresh run)

| Area | Operation | Size | tpt-simd | Incumbent | Ratio | Status |
|---|---|---|---|---|---|---|
| blas | f32 gemm | 64 | 6.2 µs | faer 4.0 µs | 1.54 | behind |
| blas | f32 gemm | 256 | 271 µs | faer 245 µs | 1.11 | behind |
| blas | f32 gemm | 512 | 2.10 ms | faer 1.94 ms | 1.08 | behind |
| blas | f64 gemm | 64 | 11.3 µs | faer 8.7 µs | 1.29 | behind |
| blas | f64 gemm | 256 | 633 µs | faer 492 µs | 1.29 | behind |
| blas | f64 gemm | 512 | 4.22 ms | faer 4.15 ms | 1.02 | parity |
| hash | CRC-32 | 64 B | 6.1 ns | crc32fast 6.5 ns | 0.93 | ahead |
| hash | CRC-32 | 4 KB | 59.5 ns | crc32fast 59.8 ns | 1.00 | parity |
| hash | CRC-32 | 1 MB | 14.6 µs | crc32fast 14.2 µs | 1.03 | parity |
| hash | Adler-32 | 64 B | 7.4 ns | simd-adler32 4.7 ns | 1.57 | behind |
| hash | Adler-32 | 4 KB | 105 ns | simd-adler32 70 ns | 1.49 | behind |
| hash | Adler-32 | 1 MB | 26 µs | simd-adler32 17.5 µs | 1.49 | behind |
| matrix | mat4 mul | 4×4 | 3.7 ns | glam 3.4 ns | 1.10 | behind |
| matrix | mat4 transpose, by value | 4×4 | 1.8 ns | glam 1.1 ns | 1.62 | behind |
| matrix | mat4 transpose, in place | 4×4 | 2.5 ns | glam 1.1 ns | 2.3 | behind |
| matrix | mat3 inverse | 3×3 | 22 ns | glam 14 ns | 1.54 | behind |
| search | find_byte | 64 B | 1.5 ns | memchr 3.5 ns | 0.43 | ahead |
| search | find_byte | 4 KB | 47 ns | memchr 26 ns | 1.82 | behind |
| search | find_byte | 1 MB | 12.0 µs | memchr 8.4 µs | 1.43 | behind |
| search | count_byte | 64 B | 2.2 ns | memchr_iter 6.3 ns | 0.35 | ahead |
| search | count_byte | 4 KB | 48 ns | memchr_iter 82 ns | 0.59 | ahead |
| search | count_byte | 1 MB | 13.0 µs | memchr_iter 20.7 µs | 0.63 | ahead |
| search | substring | 64 B | 9.8 ns | memmem 26 ns | 0.37 | ahead |
| search | substring | 4 KB | 61 ns | memmem 80 ns | 0.76 | ahead* |
| search | substring | 1 MB | 14.4 µs | memmem 21.8 µs | 0.66 | ahead* |
| text | hex encode | 64 B | 3.9 ns | faster-hex 2.6 ns | 1.50 | behind |
| text | hex encode | 4 KB | 241 ns | faster-hex 87 ns | 2.78 | behind |
| text | hex encode | 256 KB | 16.0 µs | faster-hex 9.0 µs | 1.78 | behind |
| text | hex decode | 64 B | 13.6 ns | faster-hex 4.6 ns | 2.96 | behind |
| text | hex decode | 4 KB | 839 ns | faster-hex 271 ns | 3.10 | behind |
| text | hex decode | 256 KB | 53.8 µs | faster-hex 17.6 µs | 3.06 | behind |
| text | UTF-8 validate | 64 B | 44.8 ns | simdutf8 4.8 ns | 9.3 | behind (core, not SIMD) |
| text | UTF-8 validate | 4 KB | 2.56 µs | simdutf8 212 ns | 12.1 | behind (core, not SIMD) |
| text | UTF-8 validate | 256 KB | 159 µs | simdutf8 13.2 µs | 12.1 | behind (core, not SIMD) |
| sort | `sort_unstable` (u32) | 1 K | 6.7 µs | std 5.0 µs | 1.34 | behind |
| sort | `sort_unstable` | 100 K | 1.72 ms | std 0.98 ms | 1.75 | behind |
| sort | `sort_unstable` | 1 M | 21.3 ms | std 12.0 ms | 1.78 | behind |
| sort | `select_nth` | 1 M | 1.48 ms | std 1.19 ms | 1.24 | behind |
| sort | `argsort` | 1 M | 33.9 ms | std 28.6 ms | 1.19 | behind |

\* Substring search is measured on a favourable pattern: the needle's last byte
occurs once in the haystack, which the last-byte prefilter handles well. It does
not show parity on inputs where the last needle byte is common (those fall back
to Two-Way, which is linear but slower than memmem's skip loop).

## Detail

### blas: gemm

Single-threaded, column-major, `gemm_with_workspace_*` (no per-call allocation).
Incumbents: `gemm` 0.19, `faer` 0.24, `matrixmultiply` 0.3.

| Size | tpt-simd | gemm | faer | matrixmultiply |
|---|---|---|---|---|
| f32 64 | 6.2 µs | 4.1 µs | 4.0 µs | 5.4 µs |
| f32 256 | 271 µs | 247 µs | 245 µs | 304 µs |
| f32 512 | 2.10 ms | 1.91 ms | 1.94 ms | 2.29 ms |
| f64 64 | 11.3 µs | 8.3 µs | 8.7 µs | 10.0 µs |
| f64 256 | 633 µs | 491 µs | 492 µs | 614 µs |
| f64 512 | 4.22 ms | 3.92 ms | 4.14 ms | 4.92 ms |

Without `runtime-dispatch` (the default for the individual crate), the f32
kernel is compiled out on the default target and tpt-simd is several times
slower. Enabling the feature, or building with `-C target-cpu=native`, is
required to see these numbers.

### hash: CRC-32

| Size | tpt-simd | crc32fast |
|---|---|---|
| 64 B | 6.1 ns | 6.5 ns |
| 4 KB | 59.5 ns | 59.8 ns |
| 1 MB | 14.6 µs | 14.2 µs |

The kernel is PCLMULQDQ folding (128-bit fold-by-4 over the input, with the
folded state reduced by Barrett), and when AVX2 and VPCLMULQDQ are detected, an
eight-stream 256-bit kernel. Inputs under 128 bytes use a single-accumulator fold.

### hash: Adler-32

| Size | tpt-simd | simd-adler32 |
|---|---|---|
| 64 B | 7.4 ns | 4.7 ns |
| 4 KB | 105 ns | 70 ns |
| 1 MB | 26 µs | 17.5 µs |

The kernel accumulates per-block byte sums and position-weighted sums in vector
registers (SSE2 by default, AVX2 under runtime dispatch). It is still about 1.5×
behind `simd-adler32` at every size in this run; the cause has not been profiled.
Earlier notes that said this was at parity were from a different run and are
withdrawn.

### matrix: 3×3 and 4×4

Incumbent: `glam` 0.30.

| Operation | tpt-simd | glam |
|---|---|---|
| mat4 mul | 3.7 ns | 3.4 ns |
| mat4 transpose, by value (`mat4x4_transposed_f32`) | 1.8 ns | 1.1 ns |
| mat4 transpose, in place (`mat4x4_transpose_f32`) | 2.5 ns | 1.1 ns |
| mat3 inverse | 22 ns | 14 ns |

The in-place transpose must store and reload the matrix, which costs about 0.7 ns
over the by-value form. The 3×3 inverse returns an `Option` of 36 bytes, which
goes through memory; `glam` returns registers.

### search

| Operation | Size | tpt-simd | Incumbent |
|---|---|---|---|
| find_byte | 64 B | 1.5 ns | memchr 3.5 ns |
| find_byte | 4 KB | 47 ns | memchr 26 ns |
| find_byte | 1 MB | 12.0 µs | memchr 8.4 µs |
| count_byte | 64 B | 2.2 ns | memchr_iter 6.3 ns |
| count_byte | 4 KB | 48 ns | memchr_iter 82 ns |
| count_byte | 1 MB | 13.0 µs | memchr_iter 20.7 µs |
| substring | 64 B | 9.8 ns | memmem 26 ns |
| substring | 4 KB | 61 ns | memmem 80 ns |
| substring | 1 MB | 14.4 µs | memmem 21.8 µs |

`find_byte` on larger inputs is limited by SSE2 width; `memchr` uses AVX2 when the
CPU has it. An AVX2 path under runtime dispatch is the next step.

### text: hex and UTF-8

| Operation | Size | tpt-simd | Incumbent |
|---|---|---|---|
| hex encode | 4 KB | 241 ns | faster-hex 87 ns |
| hex decode | 4 KB | 839 ns | faster-hex 271 ns |
| UTF-8 validate | 4 KB | 2.56 µs | simdutf8 212 ns |

Hex encode is a 16-byte SSE2 kernel (nibble to ASCII by compare and add, then two
interleaving unpacks). Hex decode is a 16-byte SSE2 classify-and-pack kernel,
with a validation pass before any output is written, so an error leaves the
output untouched. Both are about 3× behind `faster-hex` on this machine, which
uses wider registers; an AVX2 path is the next step.

The UTF-8 "tpt" path is `core::str::from_utf8`, which is the standard library's
validator, not a SIMD validator. Matching `simdutf8` needs a SIMD validator
implementation.

### sort

| Size | tpt-simd `sort_unstable` | std `sort_unstable` | tpt-simd `argsort` | std `argsort` |
|---|---|---|---|---|
| 1 K | 6.7 µs | 5.0 µs | 8.6 µs | 7.2 µs |
| 100 K | 1.72 ms | 0.98 ms | 2.23 ms | 2.00 ms |
| 1 M | 21.3 ms | 12.0 ms | 33.9 ms | 28.6 ms |

The partition is now branch-free (two Lomuto passes, one for `< pivot` and one
for `== pivot`), which is what took the 1 M case from 93 ms to about 21 ms.
Remaining gap: the recursion and the insertion sort for small slices.

## Changes this session (from the earlier record)

- **search:** Two-Way substring search with a last-byte prefilter and bounded
  fallback; SSE2 `find_byte`; SSE2 `count_byte` using byte-lane counting.
- **hash:** `runtime-dispatch` feature; PCLMULQDQ and VPCLMULQDQ-256 CRC-32;
  SSE2 and AVX2 Adler-32.
- **text:** SSE2 hex encode and decode; table-driven validation.
- **sort:** branch-free partition.
- **matrix:** by-value transpose; branch-free 3×3 inverse scale.
- **umbrella:** `runtime-dispatch` on by default, and `tpt-simd-hash` re-exported.

## Open gaps

In rough order of size (ratios from the refresh run):

1. UTF-8 validation: about 1.05× at 4 KB and 256 KB, about 1.5× at 64 B (Round 4).
2. Hex decode, about 3× (needs an AVX2 path).
3. Hex encode, about 1.5–2.8×.
4. Adler-32, about 1.5×.
5. find_byte on 4 KB and 1 MB, about 1.4–1.8× (needs AVX2).
6. Sort, about 1.2–1.8×.
7. Matrix, about 1.1–2.3×.
8. gemm, about 1.0–1.5× (f32 at 64 is the largest).

## Round 2: close the gaps (same-run pairs only)

Each row below is a pair measured back to back in one session. Ratio = tpt / incumbent.
Two runs are shown where the first run was noisy. Status uses the goal from the plan:
within 10% of the incumbent, or faster.

| Area | Operation | Size | tpt | Incumbent | Ratio (runs) | Status |
|---|---|---|---|---|---|---|
| search | find_byte (AVX2) | 64 B | 2.9–3.0 ns | memchr 3.5 ns | 0.85 | ahead |
| search | find_byte (AVX2) | 4 KB | 27–33 ns | memchr 26–27 ns | 1.04, 1.23 | near parity |
| search | find_byte (AVX2) | 1 MB | 8.1–8.2 µs | memchr 8.5–8.8 µs | 0.93, 0.96 | ahead |
| search | count_byte (AVX2) | 1 MB | 9.1–9.6 µs | memchr_iter 21.7 µs | 0.42 | ahead |
| search | count_byte (AVX2) | 4 KB | 52 ns | memchr_iter 83 ns | 0.62 | ahead |
| text | hex encode (AVX2) | 4 KB | 101 ns, 339 ns | faster-hex 98 ns, 340 ns | 1.03, 1.00 | parity |
| text | hex encode (AVX2) | 256 KB | 8.1 µs, 22.3 µs | faster-hex 8.2 µs, 22.7 µs | 1.00, 0.98 | parity |
| text | hex decode (AVX2) | 4 KB | 438 ns, 1.38 µs | faster-hex 262 ns, 0.90 µs | 1.67, 1.53 | behind |
| text | hex decode (AVX2) | 256 KB | 29.6 µs, 87.9 µs | faster-hex 17.4 µs, 56.0 µs | 1.70, 1.57 | behind |
| text | UTF-8 validate (own AVX2) | 4 KB | 0.50 µs | simdutf8 0.21 µs | 2.4 | behind |
| text | UTF-8 validate (own AVX2) | 256 KB | 31.5 µs | simdutf8 13.3 µs | 2.4 | behind |
| hash | Adler-32 (pmaddubsw) | 4 KB | 235 ns, 243 ns | simd-adler32 235 ns, 240 ns | 1.00, 1.01 | parity |
| hash | Adler-32 (pmaddubsw) | 1 MB | 62.7 µs, 61.8 µs | simd-adler32 60.8 µs, 62.2 µs | 1.03, 0.99 | parity |
| hash | Adler-32 (pmaddubsw) | 64 B | 20.3 ns, 20.2 ns | simd-adler32 11.4 ns, 12.1 ns | 1.78, 1.67 | behind |
| matrix | mat4 mul | 4×4 | 3.7 ns | glam 3.35 ns | 1.10 | near parity |
| matrix | mat4 transpose, by value | 4×4 | 1.72 ns | glam 1.07 ns | 1.60 | behind |
| matrix | mat3 inverse | 3×3 | 22.3 ns | glam 14.3 ns | 1.56 | behind |

Notes on this round:

- **Machine state changed mid-session.** The Adler-32 runs show every implementation
  about 2.5× slower than the earlier run, including `simd-adler32`. Only the ratios
  above are meaningful.
- **Hex decode is limited by its two passes.** It validates the whole input before it
  writes anything, so an error leaves `dst` untouched. That costs a second pass over
  the input. Keeping the guarantee and matching `faster-hex` needs a different design,
  and the cost is measured, not estimated.
- **UTF-8 is limited by the vector check.** About 70 vector operations per 32-byte block,
  mostly range comparisons. `simdutf8` classifies bytes with table lookups, so the next
  step is a table-driven classification, checked by the exhaustive tests already in
  `tpt-simd-text/tests/differential.rs` (they fail on a deliberately wrong range, which
  was verified).
- **Matrix is limited by the array ABI.** `[[f32; N]; N]` is returned through memory,
  while `glam` returns register-sized values. `#[inline(always)]` made no difference.
  Matching `glam` needs a register-typed matrix API, which is a public API decision.
- **Still open from the plan:** the gemm small-size work (plan step 6), and the sort
  work (plan step 8, including the AVX2 sort gate). Neither was started in this round.

## Round 3: sort and gemm (same-run pairs)

| Area | Operation | Size | tpt | Incumbent | Ratio | Status |
|---|---|---|---|---|---|---|
| sort | `sort_unstable` (generic, pdqsort-style equal-key rule) | 1 K | 5.4 µs | std 4.9 µs | 1.11 | near parity |
| sort | `sort_unstable` (generic) | 100 K | 1.52 ms | std 0.98 ms | 1.55 | behind |
| sort | `sort_unstable` (generic) | 1 M | 17.5 ms | std 12.0 ms | 1.46 | behind |
| sort | `sort_u32` (LSD radix, new API) | 1 K | 3.9 µs | std 4.9 µs | 0.80 | ahead |
| sort | `sort_u32` (LSD radix, new API) | 100 K | 0.41 ms | std 0.98 ms | 0.41 | ahead |
| sort | `sort_u32` (LSD radix, new API) | 1 M | 4.7 ms | std 12.0 ms | 0.39 | ahead |
| gemm f32 (kc 512) | 64 | 5.8 µs | faer 4.1 µs | 1.42 | behind |
| gemm f32 (kc 512) | 256 | 259 µs | faer 238 µs | 1.09 | near parity |
| gemm f32 (kc 512) | 512 | 1.98 ms | faer 1.96 ms | 1.01 | parity |
| gemm f64 (mc 128, kc 512) | 64 | 11.2 µs | faer 8.7 µs | 1.28 | behind |
| gemm f64 (mc 128, kc 512) | 256 | 533–549 µs | faer 484–490 µs | 1.10–1.12 | near parity |
| gemm f64 (mc 128, kc 512) | 512 | 4.07–4.35 ms | faer 3.98–4.07 ms | 1.02–1.07 | parity |

Notes:

- The radix sort is the one-pass-per-digit LSD sort with a caller-supplied scratch buffer
  (the crate stays `no_std` with no allocation). It is a deviation from the plan, which
  named an AVX2 sort: radix gets a clear win on `u32` without a runtime feature.
  The generic `sort_unstable` still uses comparisons and is not covered by this win.
- The generic sort's equal-key rule means a second partition pass only runs when the
  next pivot is not greater than the previous one. That removed one of the two passes on
  random data (1 M went from 93 ms to about 17 ms).
- Larger gemm blocking (`kc = 512`, plus `mc = 128` for f64) was measured against `faer`
  in the same run, with the second run repeated for f64. Small sizes (64) are still
  behind by 1.3–1.4×; the cause is not profiled, and the packing and edge-tile work in
  plan step 6 is the next candidate.

## Round 4: full-mode re-run (2026-10-12) and the UTF-8 validator

Full criterion mode (default warm-up and measurement), one suite at a time. Ratio =
tpt-simd ÷ incumbent, from the same run. Times are criterion medians (noisy machine).

Status of this round:

- **hash, text, search:** run 1 complete (15:46–15:55). Run 2 was interrupted.
- **sort, matrix, gemm f32:** run 1 complete (15:55–16:04). Run 2 was not done.
- **gemm f64:** not recorded. The run was interrupted and overlapped with a rebuild,
  so its figures are unreliable and are left out.
- **UTF-8:** rebuilt and measured twice after the change (below).

Because the second runs are missing, treat the single-run rows as indicative. Rows
that are within about 20% of 1.0 may be noise.

### Hash, search, text (run 1)

| Area | Operation | Size | tpt-simd | Incumbent | Ratio | Status |
|---|---|---|---|---|---|---|
| hash | CRC-32 | 64 B | 6.4 ns | crc32fast 6.7 ns | 0.95 | ahead |
| hash | CRC-32 | 4 KB | 59.4 ns | crc32fast 59.4 ns | 1.00 | parity |
| hash | CRC-32 | 1 MB | 18.3 µs | crc32fast 15.2 µs | 1.21 | behind |
| hash | Adler-32 | 64 B | 7.3 ns | simd-adler32 5.0 ns | 1.47 | behind |
| hash | Adler-32 | 4 KB | 68.6 ns | simd-adler32 70.2 ns | 0.98 | parity |
| hash | Adler-32 | 1 MB | 19.9 µs | simd-adler32 17.8 µs | 1.12 | near parity |
| search | find_byte | 64 B | 2.9 ns | memchr 3.6 ns | 0.79 | ahead |
| search | find_byte | 4 KB | 26.5 ns | memchr 26.2 ns | 1.01 | parity |
| search | find_byte | 1 MB | 8.1 µs | memchr 9.1 µs | 0.89 | ahead |
| search | count_byte | 64 B / 4 KB / 1 MB | 3.0 / 51 ns / 11.0 µs | memchr_iter 6.4 / 83 ns / 21.2 µs | 0.47 / 0.62 / 0.52 | ahead |
| search | substring | 64 B / 4 KB / 1 MB | 11.3 / 45 ns / 13.4 µs | memmem 25.4 / 82 ns / 24.2 µs | 0.44 / 0.55 / 0.55 | ahead* |
| text | hex encode | 64 B | 4.4 ns | faster-hex 2.8 ns | 1.58 | behind |
| text | hex encode | 4 KB | 91.7 ns | faster-hex 90.5 ns | 1.01 | parity |
| text | hex encode | 256 KB | 9.1 µs | faster-hex 9.6 µs | 0.95 | ahead |
| text | hex decode | 4 KB | 469 ns | faster-hex 280 ns | 1.68 | behind |
| text | hex decode | 256 KB | 31.1 µs | faster-hex 18.2 µs | 1.71 | behind |

\* Same caveat as the summary: the substring pattern is favourable.

### Sort, select, matrix, gemm f32 (run 1)

| Area | Operation | Size | tpt-simd | Incumbent | Ratio | Status |
|---|---|---|---|---|---|---|
| sort | `sort_unstable` (generic) | 1 K | 10.9 µs | std 5.0 µs | 2.2 | behind |
| sort | `sort_unstable` (generic) | 100 K | 1.53 ms | std 1.00 ms | 1.54 | behind |
| sort | `sort_unstable` (generic) | 1 M | 17.4 ms | std 11.7 ms | 1.48 | behind |
| sort | `sort_u32` (LSD radix) | 1 K / 100 K / 1 M | 3.7 µs / 0.41 ms / 4.6 ms | std 5.0 µs / 1.00 ms / 11.7 ms | 0.75 / 0.41 / 0.39 | ahead |
| sort | `select_nth` | 1 K / 100 K / 1 M | 1.36 / 137 µs / 1.45 ms | std 1.10 / 101 µs / 1.18 ms | 1.24 / 1.36 / 1.23 | behind |
| sort | `argsort` | 1 K / 100 K / 1 M | 6.6 / 1.81 / 26.6 ms | std 7.2 / 1.97 / 28.5 ms | 0.91 / 0.92 / 0.93 | ahead |
| matrix | mat4 mul | 4×4 | 3.7 ns | glam 3.3 ns | 1.10 | near parity |
| matrix | mat4 transpose, by value | 4×4 | 1.7 ns | glam 1.1 ns | 1.5 | behind |
| matrix | mat4 transpose, in place | 4×4 | 2.5 ns | glam 1.1 ns | 2.3 | behind |
| matrix | mat3 inverse | 3×3 | 22.3 ns | glam 13.0 ns | 1.7 | behind |
| gemm f32 | `tpt_workspace` | 64 / 256 / 512 | 5.7 µs / 262 µs / 2.05 ms | faer 4.2 µs / 242 µs / 2.02 ms | 1.36 / 1.09 / 1.02 | behind / near / parity |

The generic `sort_unstable` at 1 K is 2.2× behind here, but the round 3 table recorded
1.11×. Those are different runs and the gap is not explained yet. The 1 M and 100 K
ratios agree with round 3 (1.46–1.55×).

### UTF-8 validation: lookup-table validator

The AVX2 validator in `tpt-simd-text` used about 70 vector range checks per 32-byte
block. It now uses three `vpshufb` nibble lookups and one continuation check per block
(the Keiser–Lemire approach, as in simdjson). The tail and the block-boundary rules are
unchanged. The differential suite (exhaustive two-byte pairs, structured three- and
four-byte sequences, every prefix and corruption, random bytes) passes with runtime
dispatch and with `target-cpu=native`.

Same-run timings against `simdutf8`, two runs each, after the change:

| Size | Before (run 1, old validator) | After run 1 | After run 2 | simdutf8 (run 1 / run 2) | Ratio after |
|---|---|---|---|---|---|
| 64 B | 11.2 ns | 7.2 ns | 7.4 ns | 4.9 / 4.7 ns | ~1.5 |
| 4 KB | 507 ns | 223 ns | 217 ns | 208 / 207 ns | ~1.05 |
| 256 KB | 40.1 µs | 13.9 µs | 14.3 µs | 13.9 / 13.4 µs | ~1.05 |

At 4 KB and 256 KB the validator is now within about 5% of `simdutf8`, so it meets the
10% parity rule. At 64 B it is still about 1.5× behind: that size is dominated by
per-call overhead, not the vector loop. Moving the lookup tables to constant loads
made no difference at 64 B.

Open items from this round:

- UTF-8 at 64 B (above).
- The harness: the todo item "fix the harness so inputs cannot be reused across
  iterations" was checked. The benches reuse input buffers on purpose, and the sort
  suites clone their input per iteration. No reuse bug was found, so the cause of the
  earlier invalid quick-mode numbers is still unknown. The item stays open until the
  full runs are repeated.
- Run 2 for sort, matrix and gemm, and gemm f64.
