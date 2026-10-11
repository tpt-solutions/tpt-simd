# tpt-simd-pixel

Pixel kernels with no runtime dependencies:

- RGB8 to RGBA8 and back, with a fixed alpha value
- planar and interleaved 8-bit layouts (three planes)
- u8 to unit-range f32, and back with rounding
- premultiplied alpha for RGBA8
- RGB to HSV and back (f32)
- 256-bin histograms

Image resizing, YUV and blur are not included; those are separate concerns.
MIT OR Apache-2.0.
