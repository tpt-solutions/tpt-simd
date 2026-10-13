# Head-to-head benchmarks against incumbents

This page records how tpt-simd compares with established crates for the same
operations. The goal is parity: a kernel is at parity when its median time is
within about 10% of the fastest incumbent measured in the same run, or faster.

The `bench-vs` crate (unpublished) holds the benches. Per-crate kernel history
is in [benchmarks.md](benchmarks.md).

Latest full status run: **Round 10** (2026-10-13), at the end of this file.

## How to read the numbers

- **Compare ratios within one run, not absolute times across runs.** On this
  machine the same incumbent moved by up to 3–4× between sessions (for example
  `gemm` f32 at 64 was 13.0 µs in one run and 4.1 µs in another), so only the
  ratio of two rows measured together means anything.
- **Ratio** = tpt-simd time ÷ incumbent time. Below 1.0 is faster.
- Changes under about 20% between runs are noise.

## Method

- **Machine (Round 5 onward):** Intel Core i5-13500 (Raptor Lake, 20 threads),
  Windows 11. It has no AVX-512, so AVX-512 kernels are not used. Feature
  detection was not re-checked on this machine for Round 5.
- **Previous machine (Rounds 1–4):** Intel Core i5-1135G7 (Tiger Lake), Windows 11.
  It reported AVX2, FMA and VPCLMULQDQ to Rust's feature detection and did not
  report `avx512f`. Its numbers are not comparable with Round 5.
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

## Summary (Round 5: new machine, two same-run passes)

Intel Core i5-13500, Criterion full mode, two passes back to back. All eight suites
passed both times (16 runs). Ratio = tpt-simd ÷ incumbent, shown as run 1 / run 2,
each measured in its own run. For gemm the incumbent is the fastest one in that run.

Status: **ahead** = both runs below 0.95. **parity** = worst run at most 1.10.
**near** = worst run at most 1.15. **behind** = worst run above 1.15. **noisy** =
the two runs differ by more than 20%, so neither ratio is reliable on its own.

| Area | Operation | Size | Ratio (run 1 / run 2) | Status |
|---|---|---|---|---|
| blas | f32 gemm vs faer / gemm | 64 | 1.51 / 1.42 | behind |
| blas | f32 gemm | 256 | 1.13 / 1.08 | near |
| blas | f32 gemm | 512 | 1.05 / 1.03 | parity |
| blas | f64 gemm vs gemm / faer | 64 | 1.39 / 1.37 | behind |
| blas | f64 gemm | 256 | 1.11 / 1.12 | near |
| blas | f64 gemm | 512 | 1.06 / 1.03 | parity |
| hash | CRC-32 vs crc32fast | 64 B | 0.85 / 0.92 | ahead |
| hash | CRC-32 | 4 KB | 1.00 / 1.00 | parity |
| hash | CRC-32 | 1 MB | 1.02 / 1.00 | parity |
| hash | Adler-32 vs simd-adler32 | 64 B | 1.56 / 1.51 | behind |
| hash | Adler-32 | 4 KB | 1.04 / 1.03 | parity |
| hash | Adler-32 | 1 MB | 1.02 / 1.03 | parity |
| matrix | mat4 mul vs glam | 4×4 | 1.13 / 1.14 | near |
| matrix | mat4 transpose, by value vs glam | 4×4 | 1.54 / 1.68 | behind |
| matrix | mat4 transpose, in place vs glam | 4×4 | 2.06 / 2.26 | behind |
| matrix | mat3 inverse vs glam | 3×3 | 1.74 / 1.77 | behind |
| search | find_byte vs memchr | 64 B | 0.69 / 0.77 | ahead |
| search | find_byte | 4 KB | 1.21 / 0.98 | noisy |
| search | find_byte | 1 MB | 0.92 / 0.93 | ahead |
| search | count_byte vs memchr_iter | 64 B | 0.63 / 0.55 | ahead |
| search | count_byte | 4 KB | 0.59 / 0.55 | ahead |
| search | count_byte | 1 MB | 0.43 / 0.43 | ahead |
| search | substring vs memmem | 64 B | 0.37 / 0.37 | ahead\* |
| search | substring | 4 KB | 0.49 / 0.50 | ahead\* |
| search | substring | 1 MB | 0.63 / 0.60 | ahead\* |
| text | hex encode vs faster-hex | 64 B | 1.66 / 1.61 | behind |
| text | hex encode | 4 KB | 1.05 / 1.00 | parity |
| text | hex encode | 256 KB | 0.97 / 1.00 | parity |
| text | hex decode vs faster-hex | 64 B | 2.34 / 2.42 | behind |
| text | hex decode | 4 KB | 1.64 / 1.65 | behind |
| text | hex decode | 256 KB | 1.54 / 1.68 | behind |
| text | UTF-8 validate vs simdutf8 | 64 B | 1.53 / 1.54 | behind |
| text | UTF-8 validate | 4 KB | 1.07 / 1.05 | parity |
| text | UTF-8 validate | 256 KB | 1.09 / 1.04 | parity |
| sort | `sort_unstable` vs std | 1 K | 0.97 / 1.12 | near |
| sort | `sort_unstable` | 100 K | 1.61 / 1.54 | behind |
| sort | `sort_unstable` | 1 M | 1.19 / 1.47 | behind |
| sort | `sort_u32` radix vs std | 1 K | 1.00 / 0.76 | noisy |
| sort | `sort_u32` radix | 100 K | 0.42 / 0.41 | ahead |
| sort | `sort_u32` radix | 1 M | 0.41 / 0.40 | ahead |
| sort | `select_nth` vs std | 1 K | 1.24 / 1.24 | behind |
| sort | `select_nth` | 100 K | 1.39 / 1.42 | behind |
| sort | `select_nth` | 1 M | 1.53 / 1.21 | noisy |
| sort | `argsort` vs std | 1 K | 0.89 / 0.88 | ahead |
| sort | `argsort` | 100 K | 0.89 / 0.92 | ahead |
| sort | `argsort` | 1 M | 0.99 / 0.91 | parity |

\* Substring uses the same favourable pattern as before (last needle byte occurs
once), so these rows do not show performance on common last bytes.

## Summary (previous machine, Round 1 refresh, superseded)

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

## Open gaps (Round 5, new machine)

Rows behind in both runs, largest first:

1. hex decode: 1.5–2.4× at every size in Round 5. Round 6 (lookup-table validator) brings 4 KB and 256 KB to about 1.1–1.4×; 64 B stays about 2×.
2. Matrix: mat4 transpose 1.5–2.3×, mat3 inverse 1.7×. mat4 mul is 1.13–1.14×.
3. Generic `sort_unstable`: 1.5–1.6× at 100 K, 1.2–1.5× at 1 M.
4. `select_nth`: 1.2–1.4× at 1 K and 100 K (the 1 M row is noisy).
5. UTF-8 validate at 64 B: about 1.5×.
6. Hex encode at 64 B: about 1.6×.
7. Adler-32 at 64 B: about 1.5×.
8. gemm at 64: f32 1.4–1.5×, f64 1.4×. gemm at 256: f32 1.08–1.13×, f64 1.1×.

Already ahead or at parity: CRC-32 (64 B ahead, larger sizes at parity), find_byte at
64 B and 1 MB, all count_byte and substring rows, radix `sort_u32`, `argsort`.

## Open gaps (previous machine, superseded)

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

## Round 5: new machine (Intel Core i5-13500)

The whole set was re-measured on the new machine, as two same-run passes, with no code
changes. It supersedes the Round 4 open items:

- gemm f64 is now recorded (rows in the Round 5 summary).
- The second runs for hash, text, search, sort, matrix and gemm f32 are done, and
  all eight suites passed in both passes.
- Still open: UTF-8 at 64 B, Adler-32 at 64 B, and the cause of the earlier invalid
  quick-mode numbers. The todo item on input reuse is for the maintainer to close.
- Not yet compared with incumbents: `tpt-simd-fixed`, `tpt-simd-window`,
  `tpt-simd-aligned`, and `tpt-simd-blend` / `select` / `shift` (todo.md, benchmark
  candidates). The `kernels` suite gives absolute timings and scalar baselines only.
- Noisy rows (the two runs disagree by more than 20%): find_byte at 4 KB, radix
  `sort_u32` at 1 K, and `select_nth` at 1 M. Re-run these before quoting them.

## Round 6: hex decode, lookup-table validator

`hex_decode` keeps its two passes (validate everything, then write), because the
guarantee that an error leaves `dst` untouched needs them. The change is in both passes:

- Validation: a digit is classified with two `vpshufb` lookups, one on the high nibble
  and one on the low nibble, instead of range comparisons. Each byte's high-nibble class
  and low-nibble class must share a bit.
- Decode: digit values come from a mask-and-add, and pairs are combined with one
  `vpmaddubsw` instead of shifts and an `or`.
- Tests: `tests/differential.rs` gains an exhaustive check. Every byte value is put at
  every position of 32-, 64- and 96-byte inputs and compared with `faster-hex`, for both
  the result and the output bytes. The crate's tests pass with and without
  `runtime-dispatch`, and clippy is clean.

Same-session A/B (the same machine, the original code and the change run back to back,
times in ns, criterion medians):

| Size | tpt before | tpt after | tpt change | faster-hex before / after |
|---|---|---|---|---|
| 64 B | 12.3 | 11.1 | 0.90× | 5.1 / 5.4 |
| 4 KB | 505 | 393 | 0.78× | 303 / 358 |
| 256 KB | 33 062 | 27 465 | 0.83× | 30 246 / 19 663 |

The incumbent moved by up to 35% between the two halves of the session (256 KB), so the
ratios were checked against a second run with the change in place. The kernel's own
time is the more reliable number: about 20% faster at 4 KB and 256 KB.

Ratio, tpt-simd ÷ faster-hex, after the change: 4 KB 1.10–1.23, 256 KB 1.15–1.40,
64 B about 2.1 (unchanged; the cost at 64 B is call overhead, not the kernel).

Not changed in this round, and still open:

- Hex encode and UTF-8 validation at 64 B (about 1.5–1.6×). The cost is per call, in the
  dispatch and the setup around the vector loop, not in the loop itself.
- Generic `sort_unstable` (1.5× at 100 K). The crate is `#![deny(unsafe_code)]`, so the
  partition cannot use unchecked indexing, and a block partition is not practical without it.
  That policy needs a decision.
- Matrix transpose and 3×3 inverse (needs a register-typed API) and gemm at 64 (packing).

## Round 6b: profiling the 64 B hex decode path

Profiling used a standalone release harness (thin LTO, `codegen-units = 1`) that calls
`hex_decode` and `faster_hex::hex_decode` in the same process. Samples were taken over
input sizes from 0 to 1024 bytes, and each size's best of seven timed batches is reported.
A PE binary without per-function symbols, and with the kernels inlined, could not be
profiled by instruction, so the cost was split by size instead.

- **Fixed cost:** an empty call takes 3.3 ns against 1.4 ns for `faster-hex`. The
  dispatch path called the feature check twice and went through a separate validate
  wrapper and an empty SSE2 call.
- **Per byte:** at 64 B the remaining work is about 4.5 ns against about 3 ns, and the
  kernel cost is roughly linear above 128 B.

Change: validation and decoding of the head are now one AVX2 call, with the feature check
done once. The order is unchanged: the tail is checked first, then the head validates
itself before it writes, so an error still leaves `dst` untouched.

Same-session A/B, old (Round 6) against new, ns, two alternating builds each:

| Size | old | new | Ratio to faster-hex (old / new) |
|---|---|---|---|
| 0 B | 3.35–3.44 | 3.01–3.27 | 2.2–2.3 / 2.0–2.2 |
| 64 B | 8.33–8.75 | 7.44–7.53 | 1.7–1.9 / 1.65–1.75 |
| 4 KB | 169–171 | 169 | 1.22–1.23 / 1.22–1.24 |
| 256 KB | 11 216–11 678 | 11 148–11 489 | 1.24–1.33 / 1.22–1.24 |

The 64 B path is about 13–15% faster. Larger sizes are unchanged, which is expected:
they are limited by the two passes over the input, not by call overhead.

What remains at 64 B is the fixed call cost (about 1.5 ns over `faster-hex`) and the two
validation passes. Removing the second pass would change the error guarantee, so that is a
design decision and not part of this change.

## Round 6c: hex encode and UTF-8 at 64 B (investigated, no change kept)

A size sweep (0 to 4 KB) with the standalone harness found the same shape as hex decode:
per-byte cost matches `faster-hex`, and the gap is a fixed cost of about 1–2 ns per call.
Hex encode is about 1.8× at 64 B and UTF-8 about 1.5×; both reach parity by 4 KB.

Three suspects were tested:

- **Dispatch chain:** merging the remainder into one AVX2 entry point (as for decode) gave
  no measurable change. Encode: 64 B 4.96 / 5.08 ns before, 4.92 / 4.74 ns after. UTF-8:
  7.75 / 7.64 ns before, 7.82 / 7.58 ns after. The change was reverted.
- **Runtime feature check:** `is_x86_feature_detected!("avx2")` costs about 0.5 ns per call.
  That is part of the fixed cost, but not all of it.
- **Result type:** returning `Result<usize, TextError>` costs about 0.35 ns over returning
  `Option<usize>`.

The remaining 1–2 ns is not yet located. Finding it needs a copy of the crate in the
harness and a bisection, which was not done. The decode change in Round 6b, which did
reduce the 64 B cost, does not carry over, because encode and UTF-8 do not have the
decode's two-pass structure.

## Round 6d: whole-session A/B with `bench-vs/ab_compare.py`

`HEAD` (before this round's code changes) against the working tree, text suite, two
interleaved runs each, in one session. Each value is the median of the two runs (ns).
B / A is the ratio of the working tree to `HEAD`.

| Benchmark | A (HEAD) | B (working tree) | B / A |
|---|---|---|---|
| hex decode 64 B | 14.02 | 8.91 | 0.64 |
| hex decode 4 KB | 493.6 | 353.0 | 0.72 |
| hex decode 256 KB | 31 381 | 22 051 | 0.70 |
| hex encode 64 B | 5.26 | 4.63 | 0.88 |
| hex encode 4 KB | 107.9 | 105.1 | 0.97 |
| UTF-8 validate 64 B | 8.77 | 7.78 | 0.89 |
| UTF-8 validate 4 KB | 233.1 | 223.6 | 0.96 |

Hex encode and UTF-8 have no code changes from `HEAD`, so their ratios are a control for
the noise floor. They land between 0.88 and 0.97, which puts the noise at about ±10% on
this machine. Only the decode rows (0.64–0.72) are clearly above that.

Against `faster-hex` in the same session, hex decode at 64 B went from about 2.3× to about
1.8×, and at 4 KB and 256 KB from about 1.6× to about 1.2×.

## Round 7: uncovered crates (window, blend/shift, aligned, fixed)

Four suites added in `bench-vs`: `window_vs`, `blend_shift_vs`, `aligned_vs`, `fixed_vs`. Two
runs each, same session. Ratio = tpt-simd ÷ incumbent, run 1 / run 2.

| Area | Operation | Size | Ratio (run 1 / run 2) | Status |
|---|---|---|---|---|
| window | hamming vs apodize | 1024 | 0.082 / 0.087 | ahead, ~11× |
| window | hanning vs apodize | 1024 | 0.080 / 0.085 | ahead, ~12× |
| window | blackman vs apodize | 1024 | 0.169 / 0.145 | ahead, ~6× |
| blend | `blend_f32` vs wide | 4096 | 0.385 / 0.406 | ahead, ~2.5× |
| shift | `shift_left_i32` vs wide | 4096 | 1.02 / 1.10 | parity (noise) |
| fixed | `saturating_mul` 8-lane vs scalar `fixed` | 4096 | 0.168 / 0.158 | ahead, ~6× |
| aligned | zeroed vs aligned-vec | 64 | 1.03 / 1.07 | parity |
| aligned | zeroed vs aligned-vec | 4096 | 1.36 / 1.40 | behind |
| aligned | from_slice vs aligned-vec | 4096 | 1.72 / 1.60 | behind |
| aligned | zeroed vs aligned-vec | 256 KB | 0.94 / 0.92 | ahead |
| aligned | from_slice vs aligned-vec | 256 KB | 1.03 / 0.87 | parity |

Notes:

- **Window:** apodize computes in `f64` and yields an iterator, so the window speed includes
  the wider arithmetic. Both sides agree to 1e-5 for hamming; the bench asserts it before timing.
- **Blend:** both sides load the lanes from arrays and store the result. `wide`'s `blend` is
  deprecated in favour of `select`, which the bench uses.
- **Fixed:** `fixed` has no SIMD. This measures the value of the 8-lane vector type over
  scalar operations, which is the question the todo asks. It is not a same-kind comparison.
- **Aligned:** the 4 KB `from_slice` gap is the open item. It is not yet profiled.

## Round 8: full session A/B, `857e78b` to `a8e1ece`

`bench-vs/ab_compare.py --a 857e78b --b a8e1ece`, seven suites, two interleaved runs per side,
one session. A is the commit before this session's code changes. B is the current `master`,
which includes the hex decode validator, the encode `#[inline]` change, and the benchmark additions.
B / A below 1 is faster.

Changed code, same-session ratios:

| Benchmark | B / A |
|---|---|
| hex decode 64 B | 0.80 |
| hex decode 4 KB | 0.78 |
| hex decode 256 KB | 0.77 |
| hex encode 64 B (`#[inline]`) | 0.73 |
| hex encode 64 B, `faster-hex` control | 1.02 |

Unchanged code, which is the noise control: most rows land between 0.9 and 1.1. The
largest outlier is `crc32/64B` run 1 on side B (18.8 ns against 6.7 ns in run 2), which is
noise and is not reflected in the median. Two other values sit just outside the band:
hex encode 256 KB at 1.13–1.16, where the `faster-hex` control moved the same way, and
`gemm_f32_512` `tpt_workspace` at 1.09, where the same-run `gemm` moved 1.07.

<details>
<summary>Full table</summary>

| suite | benchmark | A median (ns) | B median (ns) | B / A | A runs | B runs |
|---|---|---|---|---|---|---|
| hash_vs | crc32/64B/tpt/64 | 6.88 | 12.75 | 1.85 | 6.7 / 7.0 | 18.8 / 6.7 |
| hash_vs | crc32/64B/crc32fast/64 | 7.35 | 7.34 | 1.00 | 7.8 / 6.9 | 7.5 / 7.2 |
| hash_vs | adler32/64B/tpt/64 | 8.21 | 9.09 | 1.11 | 7.9 / 8.5 | 9.9 / 8.3 |
| hash_vs | adler32/64B/simd-adler32/64 | 5.24 | 5.81 | 1.11 | 5.0 / 5.5 | 5.7 / 5.9 |
| hash_vs | crc32/4096B/tpt/4096 | 62.40 | 62.49 | 1.00 | 62.8 / 62.0 | 63.3 / 61.7 |
| hash_vs | crc32/4096B/crc32fast/4096 | 62.45 | 62.69 | 1.00 | 61.9 / 63.0 | 63.2 / 62.1 |
| hash_vs | adler32/4096B/tpt/4096 | 75.77 | 77.54 | 1.02 | 79.1 / 72.4 | 77.9 / 77.2 |
| hash_vs | adler32/4096B/simd-adler32/4096 | 74.56 | 75.08 | 1.01 | 80.1 / 69.1 | 77.8 / 72.3 |
| hash_vs | crc32/1048576B/tpt/1048576 | 15902.00 | 15162.50 | 0.95 | 16840.0 / 14964.0 | 15494.0 / 14831.0 |
| hash_vs | crc32/1048576B/crc32fast/1048576 | 16778.00 | 15361.00 | 0.92 | 18285.0 / 15271.0 | 15096.0 / 15626.0 |
| hash_vs | adler32/1048576B/tpt/1048576 | 21187.00 | 20358.00 | 0.96 | 22186.0 / 20188.0 | 21575.0 / 19141.0 |
| hash_vs | adler32/1048576B/simd-adler32/1048576 | 20152.00 | 18789.00 | 0.93 | 21222.0 / 19082.0 | 19140.0 / 18438.0 |
| text_vs | hex_encode/64B/tpt/64 | 4.76 | 3.49 | 0.73 | 4.4 / 5.1 | 3.6 / 3.4 |
| text_vs | hex_encode/64B/faster-hex/64 | 3.00 | 3.05 | 1.02 | 3.1 / 2.9 | 3.0 / 3.1 |
| text_vs | hex_decode/64B/tpt/64 | 13.89 | 11.11 | 0.80 | 13.1 / 14.7 | 10.5 / 11.7 |
| text_vs | hex_decode/64B/faster-hex/64 | 5.10 | 6.01 | 1.18 | 4.9 / 5.3 | 6.4 / 5.7 |
| text_vs | utf8_validate/64B/tpt-core/64 | 8.40 | 8.63 | 1.03 | 7.9 / 8.9 | 8.9 / 8.4 |
| text_vs | utf8_validate/64B/simdutf8/64 | 5.81 | 5.67 | 0.97 | 5.4 / 6.2 | 5.9 / 5.5 |
| text_vs | hex_encode/4096B/tpt/4096 | 121.09 | 111.75 | 0.92 | 108.3 / 133.9 | 127.8 / 95.7 |
| text_vs | hex_encode/4096B/faster-hex/4096 | 114.81 | 110.24 | 0.96 | 107.0 / 122.6 | 119.9 / 100.5 |
| text_vs | hex_decode/4096B/tpt/4096 | 476.85 | 370.00 | 0.78 | 462.0 / 491.7 | 369.4 / 370.6 |
| text_vs | hex_decode/4096B/faster-hex/4096 | 289.88 | 304.07 | 1.05 | 289.2 / 290.6 | 313.5 / 294.6 |
| text_vs | utf8_validate/4096B/tpt-core/4096 | 232.78 | 242.62 | 1.04 | 230.1 / 235.4 | 246.8 / 238.4 |
| text_vs | utf8_validate/4096B/simdutf8/4096 | 220.42 | 233.50 | 1.06 | 217.2 / 223.6 | 233.2 / 233.8 |
| text_vs | hex_encode/262144B/tpt/262144 | 9772.10 | 11381.50 | 1.16 | 9225.2 / 10319.0 | 12186.0 / 10577.0 |
| text_vs | hex_encode/262144B/faster-hex/262144 | 9793.00 | 11089.50 | 1.13 | 9238.0 / 10348.0 | 11573.0 / 10606.0 |
| text_vs | hex_decode/262144B/tpt/262144 | 31199.00 | 24158.00 | 0.77 | 30020.0 / 32378.0 | 24111.0 / 24205.0 |
| text_vs | hex_decode/262144B/faster-hex/262144 | 18444.00 | 18958.50 | 1.03 | 17877.0 / 19011.0 | 19582.0 / 18335.0 |
| text_vs | utf8_validate/262144B/tpt-core/262144 | 15638.00 | 15172.50 | 0.97 | 15777.0 / 15499.0 | 15465.0 / 14880.0 |
| text_vs | utf8_validate/262144B/simdutf8/262144 | 14511.00 | 14210.00 | 0.98 | 14319.0 / 14703.0 | 14464.0 / 13956.0 |
| search_vs | find_byte/64B/tpt/64 | 3.04 | 2.86 | 0.94 | 3.2 / 2.9 | 2.9 / 2.8 |
| search_vs | find_byte/64B/memchr/64 | 4.10 | 3.67 | 0.90 | 4.3 / 3.9 | 3.7 / 3.6 |
| search_vs | substring/64B/tpt/64 | 10.03 | 9.53 | 0.95 | 10.7 / 9.4 | 9.6 / 9.5 |
| search_vs | substring/64B/memmem/64 | 27.20 | 26.42 | 0.97 | 28.3 / 26.1 | 27.2 / 25.6 |
| search_vs | count_byte/64B/tpt/64 | 3.20 | 3.17 | 0.99 | 3.2 / 3.2 | 3.2 / 3.2 |
| search_vs | count_byte/64B/memchr_iter/64 | 6.79 | 6.22 | 0.92 | 6.8 / 6.8 | 6.7 / 5.7 |
| search_vs | find_byte/4096B/tpt/4096 | 33.69 | 30.78 | 0.91 | 34.0 / 33.4 | 33.6 / 27.9 |
| search_vs | find_byte/4096B/memchr/4096 | 28.34 | 28.41 | 1.00 | 28.8 / 27.8 | 29.2 / 27.6 |
| search_vs | substring/4096B/tpt/4096 | 46.92 | 46.23 | 0.99 | 47.1 / 46.7 | 46.4 / 46.1 |
| search_vs | substring/4096B/memmem/4096 | 96.31 | 97.27 | 1.01 | 96.5 / 96.1 | 100.8 / 93.8 |
| search_vs | count_byte/4096B/tpt/4096 | 47.53 | 48.34 | 1.02 | 47.1 / 47.9 | 48.4 / 48.3 |
| search_vs | count_byte/4096B/memchr_iter/4096 | 85.80 | 85.36 | 0.99 | 84.2 / 87.4 | 86.0 / 84.7 |
| search_vs | find_byte/1048576B/tpt/1048576 | 8165.60 | 8329.85 | 1.02 | 8212.5 / 8118.7 | 8309.8 / 8349.9 |
| search_vs | find_byte/1048576B/memchr/1048576 | 9006.75 | 9322.90 | 1.04 | 9101.3 / 8912.2 | 9656.4 / 8989.4 |
| search_vs | substring/1048576B/tpt/1048576 | 13695.50 | 13506.50 | 0.99 | 13656.0 / 13735.0 | 13673.0 / 13340.0 |
| search_vs | substring/1048576B/memmem/1048576 | 21900.50 | 22898.00 | 1.05 | 21893.0 / 21908.0 | 23354.0 / 22442.0 |
| search_vs | count_byte/1048576B/tpt/1048576 | 8909.30 | 9017.30 | 1.01 | 8831.9 / 8986.7 | 9047.4 / 8987.2 |
| search_vs | count_byte/1048576B/memchr_iter/1048576 | 20693.00 | 21139.50 | 1.02 | 20291.0 / 21095.0 | 21196.0 / 21083.0 |
| sort_vs | sort_unstable/1000/tpt/1000 | 5971.90 | 6071.60 | 1.02 | 5746.2 / 6197.6 | 6095.0 / 6048.2 |
| sort_vs | sort_unstable/1000/std/1000 | 5472.50 | 5392.15 | 0.99 | 5121.3 / 5823.7 | 5572.7 / 5211.6 |
| sort_vs | sort_u32_radix/1000/tpt_radix/1000 | 4048.30 | 4410.60 | 1.09 | 3809.6 / 4287.0 | 4449.5 / 4371.7 |
| sort_vs | sort_u32_radix/1000/std/1000 | 5340.20 | 5703.60 | 1.07 | 5094.1 / 5586.3 | 5861.2 / 5546.0 |
| sort_vs | select_nth/1000/tpt/1000 | 1414.60 | 1558.70 | 1.10 | 1406.1 / 1423.1 | 1486.4 / 1631.0 |
| sort_vs | select_nth/1000/std/1000 | 1179.85 | 1175.65 | 1.00 | 1112.8 / 1246.9 | 1155.3 / 1196.0 |
| sort_vs | argsort/1000/tpt/1000 | 7590.15 | 7147.65 | 0.94 | 7971.1 / 7209.2 | 6957.3 / 7338.0 |
| sort_vs | argsort/1000/std/1000 | 8520.55 | 8059.35 | 0.95 | 8237.7 / 8803.4 | 8249.6 / 7869.1 |
| sort_vs | sort_unstable/100000/tpt/100000 | 1669850.00 | 1687650.00 | 1.01 | 1688800.0 / 1650900.0 | 1689700.0 / 1685600.0 |
| sort_vs | sort_unstable/100000/std/100000 | 1101600.00 | 1063300.00 | 0.97 | 1093200.0 / 1110000.0 | 1085600.0 / 1041000.0 |
| sort_vs | sort_u32_radix/100000/tpt_radix/100000 | 464105.00 | 494970.00 | 1.07 | 454650.0 / 473560.0 | 461150.0 / 528790.0 |
| sort_vs | sort_u32_radix/100000/std/100000 | 1153600.00 | 1158650.00 | 1.00 | 1131000.0 / 1176200.0 | 1128100.0 / 1189200.0 |
| sort_vs | select_nth/100000/tpt/100000 | 165255.00 | 157195.00 | 0.95 | 153280.0 / 177230.0 | 153180.0 / 161210.0 |
| sort_vs | select_nth/100000/std/100000 | 113265.00 | 105825.00 | 0.93 | 115620.0 / 110910.0 | 107650.0 / 104000.0 |
| sort_vs | argsort/100000/tpt/100000 | 2069800.00 | 2011000.00 | 0.97 | 2068600.0 / 2071000.0 | 2060000.0 / 1962000.0 |
| sort_vs | argsort/100000/std/100000 | 2230000.00 | 2150400.00 | 0.96 | 2213800.0 / 2246200.0 | 2197100.0 / 2103700.0 |
| sort_vs | sort_unstable/1000000/tpt/1000000 | 19592500.00 | 18946000.00 | 0.97 | 19650000.0 / 19535000.0 | 19720000.0 / 18172000.0 |
| sort_vs | sort_unstable/1000000/std/1000000 | 13226000.00 | 13026500.00 | 0.98 | 12880000.0 / 13572000.0 | 13703000.0 / 12350000.0 |
| sort_vs | sort_u32_radix/1000000/tpt_radix/1000000 | 5463050.00 | 5285200.00 | 0.97 | 5175700.0 / 5750400.0 | 5640500.0 / 4929900.0 |
| sort_vs | sort_u32_radix/1000000/std/1000000 | 13991000.00 | 13334000.00 | 0.95 | 13845000.0 / 14137000.0 | 14113000.0 / 12555000.0 |
| sort_vs | select_nth/1000000/tpt/1000000 | 1870750.00 | 1714750.00 | 0.92 | 1821200.0 / 1920300.0 | 1775000.0 / 1654500.0 |
| sort_vs | select_nth/1000000/std/1000000 | 1450500.00 | 1356700.00 | 0.94 | 1355400.0 / 1545600.0 | 1421000.0 / 1292400.0 |
| sort_vs | argsort/1000000/tpt/1000000 | 31373000.00 | 29609500.00 | 0.94 | 31182000.0 / 31564000.0 | 31477000.0 / 27742000.0 |
| sort_vs | argsort/1000000/std/1000000 | 34194500.00 | 34639000.00 | 1.01 | 32875000.0 / 35514000.0 | 35723000.0 / 33555000.0 |
| matrix_vs_glam | mat4_mul/tpt | 4.18 | 3.99 | 0.95 | 4.0 / 4.3 | 4.0 / 3.9 |
| matrix_vs_glam | mat4_mul/glam | 3.66 | 3.51 | 0.96 | 3.6 / 3.7 | 3.5 / 3.5 |
| matrix_vs_glam | mat4_mul/ultraviolet | 10.35 | 10.25 | 0.99 | 10.1 / 10.6 | 10.5 / 10.0 |
| matrix_vs_glam | mat4_transpose/tpt | 2.66 | 2.55 | 0.96 | 2.5 / 2.8 | 2.6 / 2.5 |
| matrix_vs_glam | mat4_transpose/tpt_by_value | 2.01 | 1.88 | 0.93 | 1.9 / 2.1 | 1.9 / 1.9 |
| matrix_vs_glam | mat4_transpose/glam | 1.29 | 1.13 | 0.88 | 1.3 / 1.3 | 1.2 / 1.1 |
| matrix_vs_glam | mat4_transpose/ultraviolet | 7.30 | 7.10 | 0.97 | 7.5 / 7.1 | 7.2 / 7.0 |
| matrix_vs_glam | mat3_inverse/tpt | 23.69 | 22.98 | 0.97 | 23.9 / 23.5 | 23.1 / 22.8 |
| matrix_vs_glam | mat3_inverse/glam | 13.53 | 13.28 | 0.98 | 13.3 / 13.7 | 13.3 / 13.3 |
| matrix_vs_glam | mat3_inverse/ultraviolet | 14.07 | 14.49 | 1.03 | 14.0 / 14.1 | 15.1 / 13.9 |
| gemm_vs | gemm_f32_64/tpt | 6990.15 | 7298.35 | 1.04 | 6708.9 / 7271.4 | 7302.6 / 7294.1 |
| gemm_vs | gemm_f32_64/tpt_workspace | 6446.15 | 6809.25 | 1.06 | 6375.9 / 6516.4 | 6766.3 / 6852.2 |
| gemm_vs | gemm_f32_64/gemm | 4798.95 | 4711.55 | 0.98 | 4904.3 / 4693.6 | 4662.4 / 4760.7 |
| gemm_vs | gemm_f32_64/matrixmultiply | 5805.40 | 5931.75 | 1.02 | 6039.9 / 5570.9 | 5889.9 / 5973.6 |
| gemm_vs | gemm_f32_64/faer | 4439.85 | 4249.10 | 0.96 | 4459.5 / 4420.2 | 4258.6 / 4239.6 |
| gemm_vs | gemm_f32_256/tpt | 289605.00 | 291915.00 | 1.01 | 289750.0 / 289460.0 | 286270.0 / 297560.0 |
| gemm_vs | gemm_f32_256/tpt_workspace | 291720.00 | 295505.00 | 1.01 | 288560.0 / 294880.0 | 290510.0 / 300500.0 |
| gemm_vs | gemm_f32_256/gemm | 277835.00 | 287440.00 | 1.03 | 290920.0 / 264750.0 | 292100.0 / 282780.0 |
| gemm_vs | gemm_f32_256/matrixmultiply | 310900.00 | 306350.00 | 0.99 | 315580.0 / 306220.0 | 306630.0 / 306070.0 |
| gemm_vs | gemm_f32_256/faer | 275365.00 | 262180.00 | 0.95 | 273320.0 / 277410.0 | 260870.0 / 263490.0 |
| gemm_vs | gemm_f32_512/tpt | 2650150.00 | 2739950.00 | 1.03 | 2708500.0 / 2591800.0 | 2575300.0 / 2904600.0 |
| gemm_vs | gemm_f32_512/tpt_workspace | 2287100.00 | 2486600.00 | 1.09 | 2363000.0 / 2211200.0 | 2419300.0 / 2553900.0 |
| gemm_vs | gemm_f32_512/gemm | 2281950.00 | 2452000.00 | 1.07 | 2276800.0 / 2287100.0 | 2275900.0 / 2628100.0 |
| gemm_vs | gemm_f32_512/matrixmultiply | 2480500.00 | 2583500.00 | 1.04 | 2466400.0 / 2494600.0 | 2480900.0 / 2686100.0 |
| gemm_vs | gemm_f32_512/faer | 2284300.00 | 2207350.00 | 0.97 | 2301600.0 / 2267000.0 | 2132800.0 / 2281900.0 |
| gemm_f64_vs | gemm_f64_64/tpt_workspace | 12584.00 | 12718.50 | 1.01 | 13099.0 / 12069.0 | 12872.0 / 12565.0 |
| gemm_f64_vs | gemm_f64_64/gemm | 9094.80 | 10012.00 | 1.10 | 9334.8 / 8854.8 | 9756.0 / 10268.0 |
| gemm_f64_vs | gemm_f64_64/matrixmultiply | 11258.50 | 11204.00 | 1.00 | 11433.0 / 11084.0 | 11029.0 / 11379.0 |
| gemm_f64_vs | gemm_f64_64/faer | 9441.55 | 9349.85 | 0.99 | 9734.2 / 9148.9 | 9399.7 / 9300.0 |
| gemm_f64_vs | gemm_f64_256/tpt_workspace | 574890.00 | 584570.00 | 1.02 | 577830.0 / 571950.0 | 574920.0 / 594220.0 |
| gemm_f64_vs | gemm_f64_256/gemm | 552335.00 | 563985.00 | 1.02 | 554250.0 / 550420.0 | 553410.0 / 574560.0 |
| gemm_f64_vs | gemm_f64_256/matrixmultiply | 618900.00 | 623590.00 | 1.01 | 618440.0 / 619360.0 | 631170.0 / 616010.0 |
| gemm_f64_vs | gemm_f64_256/faer | 530420.00 | 531825.00 | 1.00 | 540150.0 / 520690.0 | 527970.0 / 535680.0 |
| gemm_f64_vs | gemm_f64_512/tpt_workspace | 4704450.00 | 4590700.00 | 0.98 | 4772500.0 / 4636400.0 | 4558800.0 / 4622600.0 |
| gemm_f64_vs | gemm_f64_512/gemm | 4422600.00 | 4382000.00 | 0.99 | 4545700.0 / 4299500.0 | 4446700.0 / 4317300.0 |
| gemm_f64_vs | gemm_f64_512/matrixmultiply | 5248200.00 | 5166550.00 | 0.98 | 5233000.0 / 5263400.0 | 5214300.0 / 5118800.0 |
| gemm_f64_vs | gemm_f64_512/faer | 4400850.00 | 4408850.00 | 1.00 | 4481700.0 / 4320000.0 | 4373500.0 / 4444200.0 |

</details>

## Round 9: closing gaps in sort, text, matrix and gemm

Goal: improve the four crates that were still behind (`tpt-simd-text`, `tpt-simd-sort`,
`tpt-simd-matrix`, `tpt-simd-blas`), keeping the ADR 0005 decisions unless the maintainer
changed them. Changes were measured with `bench-vs/ab_compare.py` (same session, two
interleaved runs per side, `HEAD` against the working tree). Ratios are B / A, where B is the
working tree. Each change is read against its control row (`std` or `faster-hex`, which was
not changed). The controls moved by up to about 8% in the final run (hex encode at 4 KB), so a
ratio is only counted as a change when it moves by more than about 10% and the control does not
move the same way.

### What changed

- **sort: `select_nth`.** The partition now runs only on the side that holds `k`. The
  equal-to-pivot pass runs only when `k` lands on the upper side, and the pivot comes from the
  same chooser as the sort (ninther for long slices). The control (`std` `select_nth_unstable`)
  stayed flat.
- **text: `hex_decode`.** `#[inline]` on the public entry point. The change is small, but the
  64 B row is call-bound, so inlining removes a real part of the fixed cost. The 4 KB and
  256 KB rows did not move.
- **matrix: register-typed API.** `Mat4` (`from_rows`, `to_rows`, `transpose`, `Mul`) and
  `Mat3` (`from_rows`, `to_rows`, `try_inverse`) in `tpt-simd-matrix::register`. Rows are held
  as `__m128` on x86_64 with SSE2, and as `Simd<f32, 4>` elsewhere. `Mat4` multiplication is
  bit-identical to `mat4x4_mul_scalar_f32`, and `Mat3::try_inverse` uses the array function's
  formula. Both are checked by property tests.

### Tried and not kept

- **Block partition for `sort_unstable`** (safe offset-buffer partition in the style of
  BlockQuicksort): about 2× slower than the Lomuto loop at 100 K, both before and after a
  fix that advanced finished blocks. Removed.
- **`#[inline(always)]` on the in-place transpose:** no change (2.97 → 2.92 ns).
- **`#[inline]` on the UTF-8 entry points:** no change beyond noise.

### Same-session A/B: sort and text (final)

| Suite | Benchmark | A median | B median | B / A | Control B / A |
|---|---|---|---|---|---|
| sort_vs | select_nth 1 K (tpt) | 1503 ns | 1390 ns | 0.92 | std 0.97 |
| sort_vs | select_nth 100 K (tpt) | 169 µs | 116 µs | 0.68 | std 0.99 |
| sort_vs | select_nth 1 M (tpt) | 1.71 ms | 1.16 ms | 0.67 | std 1.00 |
| sort_vs | sort_unstable 1 K / 100 K / 1 M (tpt) | 6.32 / 1.77 / 19.8 ms | 6.18 / 1.67 / 19.5 ms | 0.98 / 0.95 / 0.99 | std 1.00 / 0.98 / 0.98 |
| text_vs | hex_decode 64 B (tpt) | 9.53 ns | 7.84 ns | 0.82 | faster-hex 1.02 |
| text_vs | hex_decode 4 KB / 256 KB (tpt) | 365 ns / 23.5 µs | 367 ns / 23.8 µs | 1.01 / 1.01 | faster-hex 1.01 / 0.97 |
| text_vs | hex_encode 64 B / 4 KB / 256 KB | 3.49 / 100 ns / 10.7 µs | 3.47 / 107 ns / 9.9 µs | 1.00 / 1.07 / 0.93 | faster-hex 1.00 / 1.08 / 0.92 |
| text_vs | UTF-8 validate 64 B / 4 KB / 256 KB | 8.36 / 238 ns / 15.3 µs | 8.52 / 242 ns / 15.5 µs | 1.02 / 1.02 / 1.01 | simdutf8 0.98 / 1.01 / 0.99 |

Current ratios, tpt-simd ÷ incumbent in the same run (B side):

- `select_nth`: 1 K 1.19×, 100 K 1.05×, 1 M 0.84× (ahead). Previously 1.24×, 1.51×, 1.24×.
- `hex_decode`: 64 B 1.48× (was 1.84×). 4 KB and 256 KB still about 1.2×, as ADR 0005 item 1 records.
- `sort_unstable` (generic): still 1.10× at 1 K and about 1.48× at 100 K and 1 M. ADR 0005 item 2.
- `hex_encode` and UTF-8 validate at 64 B: unchanged, about 1.1× and 1.5×.

### Register-typed matrix API

Same process, `matrix_vs_glam`, two full runs (values agree within about 2%):

| Operation | array API (`tpt`) | register API (`tpt_reg`) | glam | register / glam |
|---|---|---|---|---|
| 4×4 multiply | 4.32 ns | 4.10 ns | 3.82 ns | 1.06–1.09 |
| 4×4 transpose, in place (array) / by value (array) | 2.68 / 2.01 ns | — / — | 1.23 ns | — |
| 4×4 transpose (register) | — | 1.94 ns | 1.23 ns | 1.56–1.59 |
| 3×3 inverse | 24.7 ns | 9.9 ns | 14.2 ns | 0.70 |

- The register transpose is about 1.4× faster than the in-place array transpose, and removes most
  of the store-and-reload cost. It is still about 1.6× behind `glam`, so ADR 0005 item 3 is only
  partly resolved for transpose.
- The register 3×3 inverse is about 2.5× faster than the array version and now ahead of
  `glam`. Both call the same scalar formula, so the gap comes from how the inputs and output
  move, not from different arithmetic. The mechanism is not isolated; it is measured, and it
  holds across both runs.

### gemm at 512

One filtered run, `gemm_vs` at 512 (f32, same process): `tpt_workspace` 2.28 ms, `gemm` 2.17 ms,
`faer` 2.18 ms, `matrixmultiply` 2.59 ms. The ratio to the fastest incumbent is 1.05, so
the 1.13 in Round 8 was noise and gemm at 512 is at parity. No gemm change was made. The 64 B
gap (about 1.4–1.6×) is ADR 0005 item 4, and it is unchanged.

### Open gaps after Round 9

1. `sort_unstable` generic at 100 K and above (about 1.48×): ADR 0005 item 2, needs unchecked
   indexing in the partition.
2. `hex_decode` at 4 KB and 256 KB (about 1.2×) and at 64 B (1.48×): ADR 0005 item 1.
3. `select_nth` at 1 K (1.19×): not profiled; the partition's per-call cost is the likely
   cause, and no decision is recorded for it.
4. UTF-8 validate and hex encode at 64 B (about 1.5× and 1.1×): per-call cost, not yet located.
5. Matrix 4×4 transpose by register (about 1.6×): the remaining cost is in the shuffle sequence
   and the 64-byte return, and is not resolved by the register type alone.
6. gemm at 64: ADR 0005 item 4, unchanged.

## Round 10: full status run (2026-10-13)

Every `bench-vs` suite with an incumbent, run in full Criterion mode, each suite run twice
back to back, following the recipe in [bench-vs/README.md](../bench-vs/README.md). Commit
`1c55498`, no code changes. This is a status snapshot, not a same-session A/B: ratios are
tpt-simd ÷ the fastest incumbent in the same run, and they are shown as run 1 / run 2.

Machine: Intel Core i5-13500 (same as Round 5 onward). Eleven suites: `matrix_vs_glam`,
`gemm_vs`, `gemm_f64_vs`, `hash_vs`, `text_vs`, `search_vs`, `sort_vs`, and the four Round 7
suites (`window_vs`, `blend_shift_vs`, `aligned_vs`, `fixed_vs`). All 22 runs exited 0.

Status uses the Round 5 rules: **ahead** = both runs below 0.95; **parity** = worst run at most
1.10; **near** = worst run at most 1.15; **behind** = both runs above 1.15; **noisy** = the runs
disagree by more than 20% across a boundary, so neither ratio is reliable on its own.

### Where we are behind now

| Area | Operation | Size | Ratio (run 1 / run 2) | Status | Decision or open |
|---|---|---|---|---|---|
| matrix | mat4 transpose, in place (array) | 4×4 | 2.09 / 2.15 | behind | ADR 0005 item 3 |
| matrix | mat4 transpose, by value (array) | 4×4 | 1.64 / 1.59 | behind | ADR 0005 item 3 |
| matrix | mat4 transpose (register) | 4×4 | 1.49 / 1.57 | behind | open (ADR 0005 item 3 partly) |
| matrix | mat3 inverse (array) | 3×3 | 1.73 / 1.90 | behind | ADR 0005 item 3 |
| matrix | mat4 mul (array) | 4×4 | 1.11 / 1.20 | behind (small) | open |
| sort | `sort_unstable` (generic) | 1 K | 1.12 / 1.22 | behind | open |
| sort | `sort_unstable` (generic) | 100 K | 1.51 / 1.68 | behind | ADR 0005 item 2 |
| sort | `sort_unstable` (generic) | 1 M | 1.19 / 1.39 | behind | ADR 0005 item 2 |
| sort | `select_nth` | 1 K | 1.20 / 1.41 | behind | open |
| sort | `select_nth` | 100 K | 1.28 / 1.09 | behind | open (Round 9 had 1.05) |
| text | hex decode | 64 B | 1.46 / 1.44 | behind | ADR 0005 item 1 |
| text | hex decode | 4 KB | 1.22 / 1.20 | behind | ADR 0005 item 1 |
| text | hex decode | 256 KB | 1.18 / 1.19 | behind | ADR 0005 item 1 |
| text | UTF-8 validate vs simdutf8 | 64 B | 1.48 / 1.70 | behind | open |
| hash | Adler-32 vs simd-adler32 | 64 B | 1.50 / 1.53 | behind | open |
| blas | f32 gemm, `tpt` (allocating) | 64 | 1.46 / 1.55 | behind | ADR 0005 item 4 |
| blas | f32 gemm, `tpt` (allocating) | 512 | 1.27 / 1.34 | behind | open |
| blas | f64 gemm, `tpt_workspace` | 64 | 1.37 / 1.32 | behind | ADR 0005 item 4 |
| blas | f64 gemm, `tpt_workspace` | 512 | 1.18 / 1.08 | near to behind | open |
| aligned | `from_slice` vs aligned-vec | 4 KB | 1.65 / 1.80 | behind | open (Round 7 also behind) |

Behind by a smaller margin, or within noise, so not listed above: hex encode 64 B (1.10 / 1.12,
near); f32 gemm 256 `tpt` (1.15 / 1.11, near); f64 gemm 256 `tpt_workspace` (1.10 / 1.10,
parity); register mat4 mul (1.10 / 1.07, near).

### Ahead or at parity

- **Ahead, both runs:** CRC-32 at 64 B (0.93 / 0.87); `find_byte` at 64 B and 1 MB; all
  `count_byte` and `substring` rows; `sort_u32` radix at 100 K and 1 M (0.39–0.50);
  register 3×3 inverse (0.69–0.71); fixed `saturating_mul` vs scalar `fixed` (0.15–0.17); all
  window rows vs `apodize` (ratios 0.03–0.36, very noisy in absolute terms but ahead in every run);
  blend `f32` 4 KB vs `wide` (0.67 / 0.41, noisy).
- **Parity:** CRC-32 at 4 KB and 1 MB; Adler-32 at 4 KB and 1 MB (1.03–1.08); hex encode at 4 KB
  and 256 KB; UTF-8 validate at 4 KB and 256 KB (1.04–1.08); `argsort`
  at 1 K (0.90 / 1.03); `shift_left` i32 (0.92–0.96); f32 gemm 512 workspace (1.03–1.08).
- **Noisy, not decided:** `select_nth` 1 M; `argsort` 100 K and 1 M (the two runs disagree by
  more than 2×); `aligned` `zeroed` at 64 B and 4 KB; `aligned` `from_slice` at 64 B and 256 KB;
  `find_byte` at 4 KB (0.97 / 1.20); radix `sort_u32` at 1 K (1.13 / 0.85); f32 gemm 64 and 256
  `tpt_workspace`. The tpt side moved by up to 2× between runs on several of these rows, so a
  new number needs a same-session A/B before it can be called a change.

### Compared with Round 9

Rows that Round 9 measured against the same incumbent agree within about 0.05 this run: register
mat4 mul (1.06–1.09 then, 1.07–1.10 now), register mat4 transpose (1.56–1.59 then, 1.49–1.57
now), register 3×3 inverse (0.70 then, 0.69–0.71 now), hex decode at 64 B (1.48 then, 1.44–1.46
now), hex decode at 4 KB and 256 KB (about 1.2 then, 1.18–1.22 now), `sort_unstable` at 1 K
(1.10 then, 1.12–1.22 now), `sort_unstable` at 100 K (about 1.48 then, 1.51–1.68 now), and f32
gemm 512 workspace (1.05 then, 1.03–1.08 now).

The exception is `select_nth` at 100 K. Round 9 recorded 1.05 and the change was counted as a
gain. This run has 1.28 and 1.09, so either the gain was noise or the change is not holding.
It has not been investigated. The matching `select_nth` 1 K row is also worse than Round 9
(1.20 / 1.41 against 1.19).

### Open gaps, largest first

1. **`aligned` `from_slice` at 4 KB** (1.65–1.80×, unchanged since Round 7). Not an ADR decision,
   not yet profiled.
2. **`select_nth` at 1 K and 100 K** (1.1–1.4×). Round 9 said the 100 K row was closed; this run
   says it is not, or the Round 9 measurement was noise. Needs a same-session A/B.
3. **UTF-8 validate at 64 B** (1.5–1.7×, widened from about 1.5 in Round 9). Per-call cost, not
   yet located.
4. **Matrix transpose** (1.5–2.2×, all variants). Partly resolved by the register API (ADR 0005
   item 3), not closed.
5. **Adler-32 at 64 B** (1.5×). Unprofiled since Round 5.
6. **f32 gemm at 512, `tpt` (allocating)** (1.27–1.34×). The workspace variant is at parity, so the
   gap is the per-call allocation on this path, not the kernel. Worth a note on the public API doc
   or a fix in the allocating wrapper.
7. Decided, not gaps to close: generic `sort_unstable` at 100 K and above (ADR 0005 item 2),
   hex decode validation cost (item 1), gemm at 64 (item 4), array-typed matrix API (item 3).

## Round 11: matrix transposes and SIMD 3x3 inverse

Changes, found by reading the generated assembly and by isolated probes:

- **Transposes.** `tpt-simd-matrix::register::Mat4::transpose` and `tpt-simd-permute::transpose_4x4_f32`
  (the array API) now use the DirectXMath ordering of eight `shufps`, as glam does, instead of four
  `unpck` plus four `movlh`/`movhl`. The disassembly showed both loops doing the same work; the
  shuffle choice was the difference.
- **3x3 inverse.** `mat3x3_inverse_f32` and `Mat3::try_inverse` now run a SSE2 path: three cross
  products, a transpose, and a multiply by `1/det`. The old formula is kept as the public reference
  `mat3x3_inverse_scalar_f32`. The SIMD path performs the same operations in the same order, and
  tests check bit identity on random, singular and non-finite inputs. The portable build keeps the
  scalar formula.

### Same-session A/B, `matrix_vs_glam` (`ab_compare.py`, HEAD against working tree, two runs per side)

| Benchmark | A median (ns) | B median (ns) | B / A | Note |
|---|---|---|---|---|
| mat4_transpose/tpt_reg | 2.14 | 1.52 | 0.71 | |
| mat4_transpose/tpt | 3.44 | 2.99 | 0.87 | array, in place |
| mat4_transpose/tpt_by_value | 2.82 | 2.51 | 0.89 | array |
| mat3_inverse/tpt_reg | 12.01 | 9.58 | 0.80 | by value |
| mat3_inverse/tpt | 29.15 | 20.99 | 0.72 | by value |
| mat4_mul/tpt | 5.71 | 4.72 | 0.83 | run-to-run spread on A is large |
| mat4_mul/tpt_reg | 5.19 | 5.32 | 1.03 | unchanged code |
| glam control, transpose | 1.46 | 1.89 | 1.30 | control moved by about 30%, so the transpose gains are indicative |

The criterion harness passes 36- and 64-byte inputs by value through `black_box`, which penalises
the array types. So the by-value rows above understate the kernel change. The by-reference
comparison below isolates it.

### Same-session, by reference, 3x3 inverse (temporary probe, not committed)

| Variant | Time (two matrices) |
|---|---|
| scalar reference (previous formula) | 9.2 / 9.5 ns |
| SIMD, array API | 6.4 / 6.3 ns |
| SIMD, `Mat3::try_inverse` | 4.9 / 5.2 ns |
| glam `Mat3::inverse` | 3.6 / 3.7 ns |

So the SIMD path is about 1.45x faster than the scalar formula on the array API and about 1.9x on
the register type. The register inverse is about 1.4x slower than glam, down from about 2.7x.

### Still open

- The bench still passes by value, so the criterion rows for the array types mix harness and kernel
  cost. A by-reference row is needed for every matrix operation.
- `mat4_mul` is 1.07-1.20x against glam. Not profiled in this round.
- ADR 0005 item 3 quotes Round 9 transpose and inverse numbers. The decision is unchanged, but those
  figures are now stale and the ADR should be refreshed from this round.
