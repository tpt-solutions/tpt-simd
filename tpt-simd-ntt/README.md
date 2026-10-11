# tpt-simd-ntt

Modular arithmetic and number-theoretic transforms for odd moduli below 2^31:

- `Montgomery`: reduction, multiplication, addition, subtraction, exponentiation
- `Barrett`: reduction of `u64` values modulo a `u64` divisor
- `NttPlan`: in-place forward and inverse NTT of a power-of-two length, and
  cyclic convolution

The NTT needs a prime modulus with a primitive root `g` where the length
divides `q - 1` (for example `q = 998244353`, `g = 3`). Primality is not
checked. `no_std`, no dependencies, MIT OR Apache-2.0.
