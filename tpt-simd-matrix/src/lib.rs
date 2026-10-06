//! Small fixed-size matrix kernels (4x4 / 8x8 multiply and transpose, 3x3 inverse).
//!
//! Matrices are row-major arrays of rows. Transposes reuse
//! [`tpt_simd_permute`].
//!
//! # Numeric policy
//! * `mat4x4_mul_f32` uses fused multiply-add per lane, so results can differ
//!   from the unfused scalar reference by rounding error.
//! * `mat8x8_mul_i16` accumulates in `i64` (8 products of up to 2^30 can exceed `i32`) and
//!   **saturates** the result to `i16`.
//! * `mat3x3_inverse_f32` returns `None` when the determinant is not finite
//!   or `|det| <= SINGULAR_EPSILON * max|m|^3`.
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::Simd;

/// Relative threshold under which a 3x3 determinant is treated as singular.
pub const SINGULAR_EPSILON: f32 = 1e-6;

/// Scalar reference for [`mat4x4_mul_f32`] (`a * b`, unfused).
pub fn mat4x4_mul_scalar_f32(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc = 0.0f32;
            for k in 0..4 {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// 4x4 `f32` matrix product `a * b`.
///
/// # Examples
/// ```
/// let i = [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];
/// let m = [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]];
/// assert_eq!(tpt_simd_matrix::mat4x4_mul_f32(m, i), m);
/// ```
pub fn mat4x4_mul_f32(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let rows = b.map(Simd::<f32, 4>::from_array);
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        let mut acc = Simd::<f32, 4>::splat(0.0);
        for k in 0..4 {
            acc = Simd::splat(a[i][k]).mul_add(rows[k], acc);
        }
        out[i] = acc.to_array();
    }
    out
}

/// Scalar reference for [`mat8x8_mul_i16`].
pub fn mat8x8_mul_scalar_i16(a: [[i16; 8]; 8], b: [[i16; 8]; 8]) -> [[i16; 8]; 8] {
    let mut out = [[0i16; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = 0i64;
            for k in 0..8 {
                acc += a[i][k] as i64 * b[k][j] as i64;
            }
            out[i][j] = acc.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
    out
}

/// 8x8 `i16` matrix product, `i64` accumulation, saturated to `i16`.
///
/// # Examples
/// ```
/// let mut a = [[0i16; 8]; 8];
/// for i in 0..8 { a[i][i] = 1; }
/// let b = [[300i16; 8]; 8];
/// assert_eq!(tpt_simd_matrix::mat8x8_mul_i16(a, b), b);
/// // 8 * 300 * 300 saturates:
/// assert_eq!(tpt_simd_matrix::mat8x8_mul_i16(b, b)[0][0], i16::MAX);
/// ```
pub fn mat8x8_mul_i16(a: [[i16; 8]; 8], b: [[i16; 8]; 8]) -> [[i16; 8]; 8] {
    let rows = b.map(|r| Simd::<i64, 8>::from_array(r.map(i64::from)));
    let mut out = [[0i16; 8]; 8];
    for i in 0..8 {
        let mut acc = Simd::<i64, 8>::splat(0);
        for k in 0..8 {
            acc += Simd::splat(a[i][k] as i64) * rows[k];
        }
        out[i] = acc
            .to_array()
            .map(|v| v.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
    }
    out
}

/// In-place 4x4 `f32` transpose.
pub fn mat4x4_transpose_f32(m: &mut [[f32; 4]; 4]) {
    tpt_simd_permute::transpose_4x4_f32(m);
}

/// In-place 8x8 `i16` transpose.
pub fn mat8x8_transpose_i16(m: &mut [[i16; 8]; 8]) {
    tpt_simd_permute::transpose_8x8_i16(m);
}

/// Inverse of a 3x3 matrix via the adjugate, or `None` if (nearly) singular.
///
/// # Examples
/// ```
/// let m = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]];
/// let inv = tpt_simd_matrix::mat3x3_inverse_f32(m).unwrap();
/// assert_eq!(inv, [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.125]]);
/// assert!(tpt_simd_matrix::mat3x3_inverse_f32([[1.0; 3]; 3]).is_none());
/// ```
pub fn mat3x3_inverse_f32(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let mut scale = 0.0f32;
    for r in &m {
        for &v in r {
            let a = if v < 0.0 { -v } else { v };
            if a > scale {
                scale = a;
            }
        }
    }
    let abs_det = if det < 0.0 { -det } else { det };
    if !det.is_finite() || abs_det <= SINGULAR_EPSILON * scale * scale * scale {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            c00 * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            c01 * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            c02 * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mul4_matches_scalar(
            a in proptest::array::uniform4(proptest::array::uniform4(-100.0f32..100.0)),
            b in proptest::array::uniform4(proptest::array::uniform4(-100.0f32..100.0)),
        ) {
            let (x, y) = (mat4x4_mul_f32(a, b), mat4x4_mul_scalar_f32(a, b));
            for i in 0..4 { for j in 0..4 { prop_assert!((x[i][j] - y[i][j]).abs() <= 1e-2); } }
        }

        #[test]
        fn mul8_matches_scalar(
            a in proptest::array::uniform8(proptest::array::uniform8(any::<i16>())),
            b in proptest::array::uniform8(proptest::array::uniform8(any::<i16>())),
        ) {
            prop_assert_eq!(mat8x8_mul_i16(a, b), mat8x8_mul_scalar_i16(a, b));
        }

        #[test]
        fn transpose_twice_is_identity(
            a in proptest::array::uniform8(proptest::array::uniform8(any::<i16>())),
        ) {
            let mut m = a;
            mat8x8_transpose_i16(&mut m);
            mat8x8_transpose_i16(&mut m);
            prop_assert_eq!(m, a);
        }

        #[test]
        fn inverse_roundtrip(m in proptest::array::uniform3(proptest::array::uniform3(-4.0f32..4.0))) {
            if let Some(inv) = mat3x3_inverse_f32(m) {
                let n: f32 = inv.iter().flatten().map(|v| v.abs()).fold(0.0, f32::max);
                if n < 20.0 {
                    for i in 0..3 { for j in 0..3 {
                        let mut s = 0.0f32;
                        for k in 0..3 { s += m[i][k] * inv[k][j]; }
                        let e = if i == j { 1.0 } else { 0.0 };
                        prop_assert!((s - e).abs() < 1e-2);
                    } }
                }
            }
        }
    }

    #[test]
    fn transpose4() {
        let mut m = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = (i * 4 + j) as f32;
            }
        }
        mat4x4_transpose_f32(&mut m);
        assert_eq!(m[1][0], 1.0);
        assert_eq!(m[0][3], 12.0);
    }

    #[test]
    fn nan_inverse_none() {
        assert!(mat3x3_inverse_f32([[f32::NAN; 3]; 3]).is_none());
    }
}
