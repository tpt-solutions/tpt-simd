//! aarch64 CRC-32 by carry-less folding, the same algorithm as the PCLMULQDQ path (Gopal et al.,
//! 2009) with the reflected IEEE constants. `vmull_p64` is PMULL, which is part of the optional
//! `aes` crypto extension, not the baseline. It is therefore used only where that extension is
//! detected at run time (`runtime-dispatch`) or enabled at compile time; otherwise the
//! slicing-by-8 code runs.

#![allow(unsafe_code)]
// The `aes` target feature is enabled per function on top of the `neon` feature. This module is
// compiled only when `neon` is enabled (see `mod neon` in lib.rs), so the softfloat-ABI case the
// lint guards against (rust-lang/rust#134375) cannot occur here.
#![allow(aarch64_softfloat_neon)]

use super::crc_update_scalar;
use core::arch::aarch64::vmull_p64;

const K1: u64 = 0x1_5444_2bd4;
const K2: u64 = 0x1_c6e4_1596;
const K3: u64 = 0x1_7519_97d0;
const K4: u64 = 0x0_ccaa_009e;
const K5: u64 = 0x1_63cd_6124;
const P_X: u64 = 0x1_DB71_0641;
const U_PRIME: u64 = 0x1_F701_1641;

/// `true` if the PMULL kernel may be called on this CPU.
pub(super) fn available() -> bool {
    #[cfg(feature = "runtime-dispatch")]
    {
        !cfg!(miri) && std::arch::is_aarch64_feature_detected!("aes")
    }
    #[cfg(not(feature = "runtime-dispatch"))]
    {
        cfg!(target_feature = "aes")
    }
}

/// Carry-less product of two 64-bit values.
///
/// # Safety
/// The CPU must support the `aes` feature (PMULL). `available()` checks this.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul(a: u64, b: u64) -> u128 {
    vmull_p64(a, b)
}

/// Multiplies the low halves of `a` and `keys`, and the high halves, and folds both into `b`.
///
/// # Safety
/// As for [`clmul`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn reduce128(a: u128, b: u128, keys: u128) -> u128 {
    // SAFETY: the `aes` feature is enabled for this function.
    unsafe {
        let t1 = clmul(a as u64, keys as u64);
        let t2 = clmul((a >> 64) as u64, (keys >> 64) as u64);
        b ^ t1 ^ t2
    }
}

/// Loads 16 bytes (little-endian, so the low 64 bits are the first eight bytes) and advances.
#[inline(always)]
fn get(a: &mut &[u8]) -> u128 {
    let mut b = [0u8; 16];
    b.copy_from_slice(&a[..16]);
    *a = &a[16..];
    u128::from_le_bytes(b)
}

/// Reduces the 128-bit folded state to a finalised 32-bit CRC.
///
/// # Safety
/// As for [`clmul`].
#[target_feature(enable = "aes")]
unsafe fn reduce_to_crc(x: u128) -> u32 {
    let low32 = u128::from(u32::MAX);
    // SAFETY: the `aes` feature is enabled for this function.
    unsafe {
        // Fold 64 bits down: the low half of x times K4, XORed with the high half.
        let x = clmul(x as u64, K4) ^ (x >> 64);
        // Fold 32 bits down with K5.
        let x = clmul((x & low32) as u64, K5) ^ (x >> 32);
        // Barrett reduction for the bit-reflected variant.
        let t1 = clmul((x & low32) as u64, U_PRIME);
        let t2 = clmul((t1 & low32) as u64, P_X);
        !(((x ^ t2) >> 32) as u32)
    }
}

/// Single-accumulator fold for 16..128 bytes.
///
/// # Safety
/// As for [`clmul`].
#[target_feature(enable = "aes")]
unsafe fn calculate_short(crc: u32, mut data: &[u8]) -> u32 {
    let k3k4 = u128::from(K3) | (u128::from(K4) << 64);
    let mut x = get(&mut data);
    x ^= u128::from(!crc);
    // SAFETY: the `aes` feature is enabled for this function.
    unsafe {
        while data.len() >= 16 {
            x = reduce128(x, get(&mut data), k3k4);
        }
        let crc = reduce_to_crc(x);
        !crc_update_scalar(!crc, data)
    }
}

/// Fold by 4 for 128 bytes and more, then a single-accumulator fold and the scalar tail.
///
/// # Safety
/// As for [`clmul`].
#[target_feature(enable = "aes")]
unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
    if data.len() < 128 {
        // SAFETY: the `aes` feature is enabled for this function, and the callee has the same
        // requirement.
        return unsafe { calculate_short(crc, data) };
    }
    let mut x3 = get(&mut data);
    let mut x2 = get(&mut data);
    let mut x1 = get(&mut data);
    let mut x0 = get(&mut data);
    x3 ^= u128::from(!crc);

    let k1k2 = u128::from(K1) | (u128::from(K2) << 64);
    let k3k4 = u128::from(K3) | (u128::from(K4) << 64);
    // SAFETY: the `aes` feature is enabled for this function.
    unsafe {
        while data.len() >= 64 {
            x3 = reduce128(x3, get(&mut data), k1k2);
            x2 = reduce128(x2, get(&mut data), k1k2);
            x1 = reduce128(x1, get(&mut data), k1k2);
            x0 = reduce128(x0, get(&mut data), k1k2);
        }
        let mut x = reduce128(x3, x2, k3k4);
        x = reduce128(x, x1, k3k4);
        x = reduce128(x, x0, k3k4);
        while data.len() >= 16 {
            x = reduce128(x, get(&mut data), k3k4);
        }
        let crc = reduce_to_crc(x);
        !crc_update_scalar(!crc, data)
    }
}

/// Updates the finalised CRC `crc` over `data` with PMULL folding. Requires `available()`.
pub(super) fn update(crc: u32, data: &[u8]) -> u32 {
    if data.len() < 16 {
        return !crc_update_scalar(!crc, data);
    }
    // SAFETY: the caller checked `available()`, so the `aes` feature is present.
    unsafe { calculate(crc, data) }
}
