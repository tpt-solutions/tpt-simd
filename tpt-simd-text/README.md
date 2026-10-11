# tpt-simd-text

Text kernels:

- `is_valid_utf8`: wraps `simdutf8` (MIT OR Apache-2.0)
- `hex_encode_lower` / `hex_decode`: wrap `faster-hex` (MIT), without allocation
- `utf8_to_utf16` / `utf16_to_utf8`: built here, with surrogate validation and
  exact size reporting

`no_std`, MIT OR Apache-2.0.
