//! CPU feature detection.
//!
//! [`Features::compile_time`] reports what the compiler was told it may
//! assume (`-C target-feature` / `target-cpu`). With the `std` feature,
//! [`Features::runtime`] queries the actual CPU; without `std` it falls back
//! to the compile-time answer. The crates in this workspace dispatch at
//! compile time; runtime detection is provided for applications that want to
//! choose between separately built code paths.

/// A set of SIMD capabilities relevant to tpt-simd.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// x86 SSE2.
    pub sse2: bool,
    /// x86 SSE4.1 (`roundps`, `blendvps`, ...).
    pub sse41: bool,
    /// x86 AVX.
    pub avx: bool,
    /// x86 AVX2.
    pub avx2: bool,
    /// x86 FMA3.
    pub fma: bool,
    /// x86 AVX-512 foundation.
    pub avx512f: bool,
    /// AArch64 NEON.
    pub neon: bool,
    /// AArch64 SVE.
    pub sve: bool,
    /// RISC-V vector extension.
    pub rvv: bool,
}

impl Features {
    /// Features the compiler may assume for this build.
    pub const fn compile_time() -> Self {
        Self {
            sse2: cfg!(target_feature = "sse2"),
            sse41: cfg!(target_feature = "sse4.1"),
            avx: cfg!(target_feature = "avx"),
            avx2: cfg!(target_feature = "avx2"),
            fma: cfg!(target_feature = "fma"),
            avx512f: cfg!(target_feature = "avx512f"),
            neon: cfg!(target_feature = "neon"),
            sve: cfg!(target_feature = "sve"),
            rvv: cfg!(target_feature = "v"),
        }
    }

    /// Features available on the running CPU (needs `std`; otherwise the
    /// compile-time set).
    #[cfg(feature = "std")]
    pub fn runtime() -> Self {
        #[allow(unused_mut)]
        let mut f = Self::compile_time();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            f.sse2 |= std::is_x86_feature_detected!("sse2");
            f.sse41 |= std::is_x86_feature_detected!("sse4.1");
            f.avx |= std::is_x86_feature_detected!("avx");
            f.avx2 |= std::is_x86_feature_detected!("avx2");
            f.fma |= std::is_x86_feature_detected!("fma");
            f.avx512f |= std::is_x86_feature_detected!("avx512f");
        }
        #[cfg(target_arch = "aarch64")]
        {
            f.neon |= std::arch::is_aarch64_feature_detected!("neon");
            f.sve |= std::arch::is_aarch64_feature_detected!("sve");
        }
        f
    }

    /// Without `std` there is no runtime query; returns the compile-time set.
    #[cfg(not(feature = "std"))]
    pub const fn runtime() -> Self {
        Self::compile_time()
    }
}
