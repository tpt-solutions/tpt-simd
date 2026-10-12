//! tpt-simd-matrix vs glam and ultraviolet (4x4 mul, 4x4 transpose, 3x3 inverse). Phase 12 overlap review.
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn m4(seed: f32) -> [[f32; 4]; 4] {
    let mut m = [[0.0; 4]; 4];
    for (i, r) in m.iter_mut().enumerate() {
        for (j, v) in r.iter_mut().enumerate() {
            *v = ((i * 4 + j) as f32 * 0.37 + seed).sin();
        }
    }
    m
}

fn uv4(m: &[[f32; 4]; 4]) -> ultraviolet::Mat4 {
    ultraviolet::Mat4::new(
        ultraviolet::Vec4::new(m[0][0], m[0][1], m[0][2], m[0][3]),
        ultraviolet::Vec4::new(m[1][0], m[1][1], m[1][2], m[1][3]),
        ultraviolet::Vec4::new(m[2][0], m[2][1], m[2][2], m[2][3]),
        ultraviolet::Vec4::new(m[3][0], m[3][1], m[3][2], m[3][3]),
    )
}

fn bench(c: &mut Criterion) {
    let (a, b) = (m4(1.0), m4(2.0));
    let (ga, gb) = (
        glam::Mat4::from_cols_array_2d(&a),
        glam::Mat4::from_cols_array_2d(&b),
    );
    let (ua, ub) = (uv4(&a), uv4(&b));

    let mut g = c.benchmark_group("mat4_mul");
    g.bench_function("tpt", |bn| {
        bn.iter(|| tpt_simd_matrix::mat4x4_mul_f32(black_box(a), black_box(b)))
    });
    g.bench_function("glam", |bn| bn.iter(|| black_box(ga) * black_box(gb)));
    g.bench_function("ultraviolet", |bn| {
        bn.iter(|| black_box(ua) * black_box(ub))
    });
    g.finish();

    let mut g = c.benchmark_group("mat4_transpose");
    g.bench_function("tpt", |bn| {
        let mut m = a;
        bn.iter(|| tpt_simd_matrix::mat4x4_transpose_f32(black_box(&mut m)))
    });
    g.bench_function("tpt_by_value", |bn| {
        bn.iter(|| tpt_simd_matrix::mat4x4_transposed_f32(black_box(a)))
    });
    g.bench_function("glam", |bn| bn.iter(|| black_box(ga).transpose()));
    g.bench_function("ultraviolet", |bn| bn.iter(|| black_box(ua).transposed()));
    g.finish();

    let m3 = [[2.0f32, 0.5, 0.1], [0.3, 3.0, 0.2], [0.1, 0.4, 4.0]];
    let g3 = glam::Mat3::from_cols_array_2d(&m3);
    let u3 = ultraviolet::Mat3::new(
        ultraviolet::Vec3::new(m3[0][0], m3[0][1], m3[0][2]),
        ultraviolet::Vec3::new(m3[1][0], m3[1][1], m3[1][2]),
        ultraviolet::Vec3::new(m3[2][0], m3[2][1], m3[2][2]),
    );
    let mut g = c.benchmark_group("mat3_inverse");
    g.bench_function("tpt", |bn| {
        bn.iter(|| tpt_simd_matrix::mat3x3_inverse_f32(black_box(m3)))
    });
    g.bench_function("glam", |bn| bn.iter(|| black_box(g3).inverse()));
    g.bench_function("ultraviolet", |bn| bn.iter(|| black_box(u3).inversed()));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
