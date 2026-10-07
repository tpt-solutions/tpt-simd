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
