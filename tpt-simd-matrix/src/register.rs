//! Register-typed `f32` matrices: [`Mat4`] and [`Mat3`].
//!
//! The array functions in this crate take and return `[[f32; N]; N]` by value, which the ABI
//! passes through memory. These types hold each row as one 128-bit lane value (`__m128` on
//! x86_64 with SSE2, [`Simd`] elsewhere). Chained operations such as `a * b * c` or
//! `(a * b).transpose()` therefore stay in vector registers, and rows are only stored when the
//! caller asks for them with `to_rows`.
//!
//! Numeric policy matches the array API. [`Mat4`] multiplication is unfused and adds the
//! products in the same order as [`mat4x4_mul_scalar_f32`](crate::mat4x4_mul_scalar_f32), so the
//! two are bit-identical. [`Mat3::try_inverse`] uses the same formula and threshold as
//! [`mat3x3_inverse_f32`](crate::mat3x3_inverse_f32).
//!
//! # Examples
//!
//! ```
//! use tpt_simd_matrix::Mat4;
//!
//! let m = Mat4::from_rows([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
//!                          [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]);
//! // The product is transposed without leaving registers.
//! let t = (m * m).transpose();
//! assert_eq!(t.to_rows()[0][1], (m * m).to_rows()[1][0]);
//! ```

use core::fmt;
use core::ops::Mul;

/// One 128-bit row: `__m128` on x86_64 with SSE2, otherwise a portable lane array.
#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "sse2"
))]
mod lane {
    use core::arch::x86_64::*;

    pub type F4 = __m128;

    #[inline(always)]
    pub fn load(a: [f32; 4]) -> F4 {
        // SAFETY: `a` is 16 readable bytes; `_mm_loadu_ps` has no alignment requirement.
        unsafe { _mm_loadu_ps(a.as_ptr()) }
    }

    #[inline(always)]
    pub fn store(v: F4) -> [f32; 4] {
        let mut a = [0.0f32; 4];
        // SAFETY: `a` is 16 writable bytes; `_mm_storeu_ps` has no alignment requirement.
        unsafe { _mm_storeu_ps(a.as_mut_ptr(), v) };
        a
    }

    #[inline(always)]
    pub fn zero() -> F4 {
        // SAFETY: SSE2 is enabled at compile time for this module.
        unsafe { _mm_setzero_ps() }
    }

    #[inline(always)]
    pub fn add(a: F4, b: F4) -> F4 {
        // SAFETY: SSE2 is enabled at compile time for this module.
        unsafe { _mm_add_ps(a, b) }
    }

    #[inline(always)]
    pub fn mul(a: F4, b: F4) -> F4 {
        // SAFETY: SSE2 is enabled at compile time for this module.
        unsafe { _mm_mul_ps(a, b) }
    }

    /// Lane `k` copied to all four lanes.
    #[inline(always)]
    pub fn broadcast(v: F4, k: usize) -> F4 {
        // SAFETY: SSE2 is enabled at compile time for this module.
        unsafe {
            match k {
                0 => _mm_shuffle_ps::<0x00>(v, v),
                1 => _mm_shuffle_ps::<0x55>(v, v),
                2 => _mm_shuffle_ps::<0xAA>(v, v),
                _ => _mm_shuffle_ps::<0xFF>(v, v),
            }
        }
    }

    /// Transposes four rows held in registers (4 unpacks and 4 moves, no memory round trip).
    #[inline(always)]
    pub fn transpose4(r: [F4; 4]) -> [F4; 4] {
        // SAFETY: SSE2 is enabled at compile time for this module.
        unsafe {
            let t0 = _mm_unpacklo_ps(r[0], r[1]);
            let t1 = _mm_unpackhi_ps(r[0], r[1]);
            let t2 = _mm_unpacklo_ps(r[2], r[3]);
            let t3 = _mm_unpackhi_ps(r[2], r[3]);
            [
                _mm_movelh_ps(t0, t2),
                _mm_movehl_ps(t2, t0),
                _mm_movelh_ps(t1, t3),
                _mm_movehl_ps(t3, t1),
            ]
        }
    }
}

#[cfg(not(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "sse2"
)))]
mod lane {
    use tpt_simd_core::Simd;

    pub type F4 = Simd<f32, 4>;

    #[inline(always)]
    pub fn load(a: [f32; 4]) -> F4 {
        Simd::from_array(a)
    }

    #[inline(always)]
    pub fn store(v: F4) -> [f32; 4] {
        v.to_array()
    }

    #[inline(always)]
    pub fn zero() -> F4 {
        Simd::splat(0.0)
    }

    #[inline(always)]
    pub fn add(a: F4, b: F4) -> F4 {
        a + b
    }

    #[inline(always)]
    pub fn mul(a: F4, b: F4) -> F4 {
        a * b
    }

    #[inline(always)]
    pub fn broadcast(v: F4, k: usize) -> F4 {
        Simd::splat(v.to_array()[k])
    }

    #[inline(always)]
    pub fn transpose4(r: [F4; 4]) -> [F4; 4] {
        let m = [
            r[0].to_array(),
            r[1].to_array(),
            r[2].to_array(),
            r[3].to_array(),
        ];
        let mut out = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = m[j][i];
            }
        }
        out.map(Simd::from_array)
    }
}

use lane::F4;

/// A 4x4 `f32` matrix with each row held as a 128-bit lane value. Row-major, as in the array API.
#[derive(Clone, Copy)]
pub struct Mat4 {
    rows: [F4; 4],
}

impl Mat4 {
    /// Builds a matrix from its rows.
    #[inline]
    #[must_use]
    pub fn from_rows(rows: [[f32; 4]; 4]) -> Self {
        Self {
            rows: rows.map(lane::load),
        }
    }

    /// Returns the rows.
    #[inline]
    #[must_use]
    pub fn to_rows(self) -> [[f32; 4]; 4] {
        self.rows.map(lane::store)
    }

    /// Returns the transpose. The rows stay in registers.
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            rows: lane::transpose4(self.rows),
        }
    }
}

impl Mul for Mat4 {
    type Output = Mat4;

    /// Matrix product `self * rhs`, unfused, bit-identical to
    /// [`mat4x4_mul_scalar_f32`](crate::mat4x4_mul_scalar_f32).
    #[inline]
    fn mul(self, rhs: Mat4) -> Mat4 {
        let b = rhs.rows;
        let rows = self.rows.map(|a| {
            // acc = ((0 + a0*b0) + a1*b1) + ...: the same order as the scalar reference.
            let mut acc = lane::zero();
            for (k, row) in b.iter().enumerate() {
                acc = lane::add(acc, lane::mul(lane::broadcast(a, k), *row));
            }
            acc
        });
        Mat4 { rows }
    }
}

impl PartialEq for Mat4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_rows() == other.to_rows()
    }
}

impl fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Mat4").field(&self.to_rows()).finish()
    }
}

/// A 3x3 `f32` matrix. Each row occupies one lane value; the fourth lane is unused and zero.
#[derive(Clone, Copy)]
pub struct Mat3 {
    rows: [F4; 3],
}

impl Mat3 {
    /// Builds a matrix from its rows.
    #[inline]
    #[must_use]
    pub fn from_rows(rows: [[f32; 3]; 3]) -> Self {
        Self {
            rows: rows.map(|r| lane::load([r[0], r[1], r[2], 0.0])),
        }
    }

    /// Returns the rows.
    #[inline]
    #[must_use]
    pub fn to_rows(self) -> [[f32; 3]; 3] {
        self.rows.map(|r| {
            let a = lane::store(r);
            [a[0], a[1], a[2]]
        })
    }

    /// Inverse, or `None` if (nearly) singular. Same formula and threshold as
    /// [`mat3x3_inverse_f32`](crate::mat3x3_inverse_f32).
    #[inline]
    #[must_use]
    pub fn try_inverse(self) -> Option<Mat3> {
        crate::mat3x3_inverse_f32(self.to_rows()).map(Mat3::from_rows)
    }
}

impl PartialEq for Mat3 {
    fn eq(&self, other: &Self) -> bool {
        self.to_rows() == other.to_rows()
    }
}

impl fmt::Debug for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Mat3").field(&self.to_rows()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mat4_mul_bit_identical_to_array_api(
            a in proptest::array::uniform4(proptest::array::uniform4(-1e6f32..1e6)),
            b in proptest::array::uniform4(proptest::array::uniform4(-1e6f32..1e6)),
        ) {
            let got = (Mat4::from_rows(a) * Mat4::from_rows(b)).to_rows();
            let want = crate::mat4x4_mul_scalar_f32(a, b);
            for i in 0..4 { for j in 0..4 {
                prop_assert_eq!(got[i][j].to_bits(), want[i][j].to_bits());
            } }
        }

        #[test]
        fn mat4_transpose_matches_array_api(
            a in proptest::array::uniform4(proptest::array::uniform4(any::<f32>())),
        ) {
            let got = Mat4::from_rows(a).transpose().to_rows();
            let mut want = a;
            crate::mat4x4_transpose_f32(&mut want);
            for i in 0..4 { for j in 0..4 {
                prop_assert_eq!(got[i][j].to_bits(), want[i][j].to_bits());
            } }
        }

        #[test]
        fn mat3_inverse_matches_array_api(
            m in proptest::array::uniform3(proptest::array::uniform3(-4.0f32..4.0)),
        ) {
            let got = Mat3::from_rows(m).try_inverse().map(|x| x.to_rows());
            prop_assert_eq!(got, crate::mat3x3_inverse_f32(m));
        }
    }

    #[test]
    fn rows_round_trip() {
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        assert_eq!(Mat4::from_rows(a).to_rows(), a);
        assert_eq!(
            Mat4::from_rows(a).transpose().transpose(),
            Mat4::from_rows(a)
        );
    }

    #[test]
    fn singular_mat3_is_none() {
        assert!(Mat3::from_rows([[1.0; 3]; 3]).try_inverse().is_none());
    }
}
