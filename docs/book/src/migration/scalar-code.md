# From scalar code to tpt-simd

Scalar loops are easy to write but rarely vectorise. tpt-simd lets you write
the same loop logic on vectors with zero runtime overhead.

## Before: scalar loop

```rust
let mut out = vec![0.0; n];
for i in 0..n {
    out[i] = a[i] * b[i] + c[i];
}
```

## After: tpt-simd

```rust
use tpt_simd_core::Simd;

let mut out = vec![0.0; n];
let mut i = 0;
while i + 8 <= n {
    let a = Simd::<f32, 8>::from_array(unsafe { a_ptr.add(i).read_unaligned() });
    let b = Simd::<f32, 8>::from_array(unsafe { b_ptr.add(i).read_unaligned() });
    let c = Simd::<f32, 8>::from_array(unsafe { c_ptr.add(i).read_unaligned() });
    let o = a * b + c;
    o.to_array().into_iter().enumerate()
        .for_each(|(k, v)| unsafe { out_ptr.add(i + k).write_unaligned(v) });
    i += 8;
}
// tail handled separately (or let the backtrack bridge handle it)
```

## Better: use slice helpers where available

Most slice-based APIs avoid the manual per-lane loads. For example:

```rust
use tpt_simd_window::apply_window_f32;

let mut data = vec![1.0f32; 1024];
let window = tpt_simd_window::hanning_window_f32(1024);
apply_window_f32(&mut data, &window);
```

## Porting table

Common scalar loops and the tpt-simd function that replaces them. Each name is
a public function in the named crate (`_f32` / `_f64` suffixes are the
float variants).

| Scalar loop | tpt-simd function | Crate |
| --- | --- | --- |
| `sum += a[i] * b[i]` (f32) | `dot_product_f32`, `dot_f32` (BLAS-style) | `tpt-simd-dot`, `tpt-simd-blas` |
| `sum += a[i] * b[i]` (i16, i32 accumulate) | `dot_product_i16` | `tpt-simd-dot` |
| `sum += x[i]` with compensated error | `sum_compensated_f32`, `pairwise_sum_f32` | `tpt-simd-reduce` |
| `mean` / `variance` / `min` / `max` | `mean_f32`, `variance_f32`, `min_f32`, `max_f32` (`mean`/`min`/`max` return `Option` on empty input) | `tpt-simd-reduce` |
| `argmin` / `argmax` | `argmin_f32`, `argmax_f32` | `tpt-simd-reduce` |
| `y[i] += alpha * x[i]` | `axpy_f32` | `tpt-simd-blas` |
| FIR filter over `i16` samples | `fir_filter_i16` (`fir_filter_scalar_i16` is the reference) | `tpt-simd-convolve` |
| 1-D convolution over `f32` | `convolve_1d_f32` (`convolve_1d_scalar_f32` is the reference) | `tpt-simd-convolve` |
| Window multiply `x[i] *= w[i]` | `apply_window_f32`, `hanning_window_f32` | `tpt-simd-window` |
| `c[i] = mask[i] ? a[i] : b[i]` | `blend_f32` | `tpt-simd-blend` |
| 8x8 `i16` transpose | `transpose_8x8_i16` | `tpt-simd-permute` |
| 4x4 `f32` matrix multiply / transpose | `mat4x4_mul_f32`, `mat4x4_transpose_f32` | `tpt-simd-matrix` |
| 3x3 `f32` inverse | `mat3x3_inverse_f32` | `tpt-simd-matrix` |
| `count_ones` over a `u64` slice | `popcount_u64` | `tpt-simd-bits` |
| prefix sum over `u32` / `u64` | `prefix_sum_u32`, `prefix_sum_u64` | `tpt-simd-bits` |
| `rank` / `select` on a bitset | `RankSelect::rank1`, `RankSelect::select1` | `tpt-simd-bits` |
| CG solver update (two axpy + norm) | `cg_update` | `tpt-simd-sparse` |
| CSR `y = alpha * A * x + beta * y` | `spmv_csr` | `tpt-simd-sparse` |

Some kernels also have a scalar version (`fir_filter_scalar_i16`,
`convolve_1d_scalar_f32`). Keep the scalar loop in your own tests as the
oracle when you port.

## Migration steps

1. Identify the hot loop and the vector width that fits your target
   (`F32x8` on x86_64 baseline, `F32x16`/`F32x32` with `-C target-cpu=native`).
2. Replace element-wise operations with vector operators.
3. Replace the index loop with chunked loads (`from_array`) + a scalar tail.
4. Use the pre-built slice functions (`apply_window_f32`, `blend_f32`,
   `transpose_8x8_i16`, `dot_product_i16`) where they exist — these handle
   tail lengths and are tested against the scalar reference.
5. Keep a scalar reference and run proptest (`SIMD == scalar`).

## Performance notes

- The workspace uses **compile-time dispatch**; without `-C target-cpu=native`
  (or explicit `+avx2,+fma`) you get the portable path. The speedups in
  [benchmarks.md](../../benchmarks.md) assume native flags.
- For element-wise ops, tpt-simd is matched (not beaten) against an
  auto-vectorised scalar iterator loop — its value is portability, consistent
  semantics and testing, not beating LLVM.
- Real wins come where intrinsics beat LLVM: `i16` dot product, fixed-point
  multiply, and FMA-complex multiply.

## See also

- [From manual intrinsics](manual-intrinsics.md)
- [From other SIMD crates](other-crates.md)
