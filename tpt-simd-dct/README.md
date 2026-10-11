# tpt-simd-dct

Orthonormal DCT-II / DCT-III and MDCT / IMDCT for codec-sized blocks.

- `Dct`: orthonormal DCT-II (`forward`) and its inverse DCT-III (`inverse`)
- `Mdct`: MDCT (`forward`, 2N samples to N coefficients) and IMDCT (`inverse`),
  with time-domain aliasing cancellation when windowed overlap-add is used

These are table-based O(N²) transforms. They are meant for block sizes up to a
few thousand. Plain DCT for large sizes is better served by `rustdct`; an
FFT-based path is not implemented here.

`no_std` with `alloc`, MIT OR Apache-2.0.
