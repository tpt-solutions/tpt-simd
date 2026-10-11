//! Cross-architecture conformance: each kernel's output is hashed and compared
//! against a golden digest computed on x86_64.
//!
//! Only kernels whose results are specified bit-exact are included: integer
//! arithmetic, lane moves, IEEE add/sub/mul with a fixed order (no FMA
//! contraction), correctly rounded rounding functions, and the fused complex
//! multiply (fma is correctly rounded everywhere). Reductions that reassociate
//! (`dot_product_f32`, `sum` of f32) are documented as tolerance-only (ADR 0003)
//! and are not hashed here. Inputs are finite; NaN outputs are canonicalised
//! before hashing because the NaN sign and payload differ between x86 and arm.
//!
//! The same test runs on every target (`cargo test -p tpt-simd --test
//! cross_arch_golden`); under emulation (Docker, arm64 / riscv64) it checks the
//! other architectures against the same constants.

use tpt_simd::saturate::{
    pack_i16_to_i8, saturating_add_i16, saturating_mul_i16, saturating_sub_i16,
};
use tpt_simd::{ComplexSimd, F32x8, Simd};

/// Golden digests, computed on `x86_64-pc-windows-msvc` (AVX2 and SSE2 builds
/// agree). Regenerate with `GOLDEN_PRINT=1 cargo test -p tpt-simd --test
/// cross_arch_golden -- --nocapture` only after a deliberate semantic change.
const GOLDEN: &[(&str, u64)] = &[
    ("dot_product_i16", 0x5c10_8181_24f1_85f1),
    ("saturate_i16", 0xd212_93f2_1b86_2f68),
    ("transpose_8x8_i16", 0xc339_86fb_a8a6_8f12),
    ("butterfly_f32", 0xbeb5_f21c_b223_a87f),
    ("mul_int", 0x5583_81b6_3104_9e06),
    ("rounding", 0x89f2_f702_bb0b_4142),
    ("horizontal_select_shift", 0xadd6_4670_c946_1d16),
    ("matrix", 0xdc2a_32a8_7f7a_c5b0),
    ("fixed_q16_16", 0x7de5_45c2_0920_c6e7),
    ("complex_mul_f32", 0xe517_3dfa_c643_4fc3),
];

/// Deterministic generator (SplitMix64); no dependency on the rng crate.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn i16s(&mut self, n: usize) -> Vec<i16> {
        (0..n).map(|_| self.next() as i16).collect()
    }
    fn i32s(&mut self, n: usize, bound: i32) -> Vec<i32> {
        (0..n)
            .map(|_| (self.next() % bound as u64) as i32)
            .collect()
    }
    /// Finite floats in roughly [-1000, 1000) with a fractional part, so that
    /// rounding functions see ties and near-ties.
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % 2_000_001) as f32 - 1_000_000.0) / 512.0)
            .collect()
    }
}

/// FNV-1a over the little-endian bytes of each value.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    fn i16s(&mut self, v: &[i16]) {
        v.iter().for_each(|x| self.bytes(&x.to_le_bytes()));
    }
    fn i32s(&mut self, v: &[i32]) {
        v.iter().for_each(|x| self.bytes(&x.to_le_bytes()));
    }
    fn i8s(&mut self, v: &[i8]) {
        v.iter().for_each(|x| self.bytes(&x.to_le_bytes()));
    }
    fn f32s(&mut self, v: &[f32]) {
        for &x in v {
            // Canonical NaN: sign and payload are not specified across targets.
            let bits = if x.is_nan() { 0x7FC0_0000 } else { x.to_bits() };
            self.bytes(&bits.to_le_bytes());
        }
    }
}

fn lanes16(v: &[i16]) -> Simd<i16, 16> {
    Simd::from_slice(v)
}

fn lanes8i(v: &[i32]) -> Simd<i32, 8> {
    Simd::from_slice(v)
}

fn lanes8f(v: &[f32]) -> F32x8 {
    Simd::from_slice(v)
}

/// Computes every kernel's digest, in the order of [`GOLDEN`].
fn digests() -> Vec<(&'static str, u64)> {
    let mut g = Gen(0x5EED_0001);
    let mut out = Vec::new();
    let mut push = |name: &'static str, d: Digest| out.push((name, d.0));

    // Integer dot product, exact (wrapping i32 accumulation, order-independent).
    let (a, b) = (g.i16s(300), g.i16s(300));
    let mut d = Digest::new();
    d.i32s(&[tpt_simd::dot::dot_product_i16(&a, &b)]);
    push("dot_product_i16", d);

    // Saturating i16 arithmetic on 16-lane vectors.
    let mut d = Digest::new();
    for _ in 0..8 {
        let (x, y) = (g.i16s(16), g.i16s(16));
        d.i16s(&saturating_add_i16(lanes16(&x), lanes16(&y)).to_array());
        d.i16s(&saturating_sub_i16(lanes16(&x), lanes16(&y)).to_array());
        d.i16s(&saturating_mul_i16(lanes16(&x), lanes16(&y)).to_array());
        d.i8s(&pack_i16_to_i8(lanes16(&x)).to_array());
    }
    push("saturate_i16", d);

    // 8x8 transpose (pure moves).
    let mut m = [[0i16; 8]; 8];
    for row in m.iter_mut() {
        row.copy_from_slice(&g.i16s(8));
    }
    tpt_simd::permute::transpose_8x8_i16(&mut m);
    let mut d = Digest::new();
    m.iter().for_each(|r| d.i16s(r));
    push("transpose_8x8_i16", d);

    // FFT butterfly: IEEE add and sub.
    let mut d = Digest::new();
    for _ in 0..8 {
        let (x, y) = (g.f32s(8), g.f32s(8));
        let (mut a, mut b) = (lanes8f(&x), lanes8f(&y));
        tpt_simd::butterfly::butterfly_f32(&mut a, &mut b);
        d.f32s(&a.to_array());
        d.f32s(&b.to_array());
    }
    push("butterfly_f32", d);

    // Integer multiplies (exact by construction).
    let mut d = Digest::new();
    for _ in 0..8 {
        let (x, y) = (g.i16s(16), g.i16s(16));
        d.i16s(&tpt_simd::mul::mul_hi_i16(lanes16(&x), lanes16(&y)).to_array());
        let (p, q) = (g.i32s(8, i32::MAX), g.i32s(8, i32::MAX));
        d.i32s(&tpt_simd::mul::mul_lo_i32(lanes8i(&p), lanes8i(&q)).to_array());
        d.i32s(&tpt_simd::mul::mul_hi_i32(lanes8i(&p), lanes8i(&q)).to_array());
    }
    push("mul_int", d);

    // Rounding: exact by definition (round/floor are correctly rounded).
    let mut d = Digest::new();
    for _ in 0..8 {
        let x = g.f32s(8);
        d.f32s(&tpt_simd::rounding::round_f32(lanes8f(&x)).to_array());
        d.f32s(&tpt_simd::rounding::floor_f32(lanes8f(&x)).to_array());
        d.i32s(&tpt_simd::rounding::round_to_nearest_even_i32(lanes8f(&x)).to_array());
    }
    push("rounding", d);

    // Horizontal reductions on integers, and lane selection.
    let mut d = Digest::new();
    for _ in 0..8 {
        let (x, y) = (g.i32s(8, 1 << 30), g.i32s(8, 8));
        d.i32s(&[tpt_simd::horizontal::horizontal_sum_i32(lanes8i(&x))]);
        d.i32s(&tpt_simd::select::select_i32(lanes8i(&x), lanes8i(&y)).to_array());
        d.i32s(&tpt_simd::shift::shift_left_i32(lanes8i(&x), (y[0] as u32) * 4).to_array());
        d.i32s(&tpt_simd::shift::rotate_left_i32(lanes8i(&x), (y[1] as u32) * 3).to_array());
    }
    push("horizontal_select_shift", d);

    // Matrix products: f32 with a fixed order and no FMA contraction, i16 exact.
    let mut d = Digest::new();
    let mut ma = [[0f32; 4]; 4];
    let mut mb = [[0f32; 4]; 4];
    for r in 0..4 {
        ma[r].copy_from_slice(&g.f32s(4));
        mb[r].copy_from_slice(&g.f32s(4));
    }
    for row in tpt_simd::matrix::mat4x4_mul_f32(ma, mb) {
        d.f32s(&row);
    }
    let mut ia = [[0i16; 8]; 8];
    let mut ib = [[0i16; 8]; 8];
    for r in 0..8 {
        ia[r].copy_from_slice(&g.i16s(8));
        ib[r].copy_from_slice(&g.i16s(8));
    }
    for row in tpt_simd::matrix::mat8x8_mul_i16(ia, ib) {
        d.i16s(&row);
    }
    push("matrix", d);

    // Fixed-point: from_f32 rounds ties to even, multiply rounds ties away.
    let mut d = Digest::new();
    type Q = tpt_simd::fixed::Fixed<i32, 16, 16, 8>;
    for _ in 0..8 {
        let (x, y) = (g.f32s(8), g.f32s(8));
        let (qa, qb) = (Q::from_f32(lanes8f(&x)), Q::from_f32(lanes8f(&y)));
        d.i32s(&qa.raw().to_array());
        d.i32s(&(qa * qb).raw().to_array());
    }
    push("fixed_q16_16", d);

    // Complex multiply with fused semantics (correctly rounded fma).
    let mut d = Digest::new();
    for _ in 0..8 {
        let (ar, ai, br, bi) = (g.f32s(8), g.f32s(8), g.f32s(8), g.f32s(8));
        let p = tpt_simd::mul::complex_mul_f32(
            ComplexSimd::new(lanes8f(&ar), lanes8f(&ai)),
            ComplexSimd::new(lanes8f(&br), lanes8f(&bi)),
        );
        d.f32s(&p.real.to_array());
        d.f32s(&p.imag.to_array());
    }
    push("complex_mul_f32", d);

    out
}

#[test]
fn kernels_match_golden_digests() {
    let got = digests();
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        for (name, d) in &got {
            println!("    (\"{name}\", {d:#018x}),");
        }
    }
    assert_eq!(got.len(), GOLDEN.len(), "kernel count changed");
    for ((name, d), (gname, gd)) in got.iter().zip(GOLDEN) {
        assert_eq!(name, gname);
        assert_eq!(
            d, gd,
            "{name}: output differs from the x86_64 golden digest"
        );
    }
}
