//! Integer stream codecs: LEB128 varints and fixed-width bitpacking.
//!
//! These are scalar, `no_std` and allocation-free. Encoders write into a caller
//! buffer and return the number of bytes used; decoders return `None` on malformed
//! input or when the output slice is too small.

/// Bytes needed for `value` as an unsigned LEB128 varint.
#[must_use]
pub fn varint_len_u64(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    bits.max(1).div_ceil(7)
}

/// Bytes needed for all of `values` as unsigned LEB128 varints.
#[must_use]
pub fn varint_encoded_len_u32(values: &[u32]) -> usize {
    values.iter().map(|&v| varint_len_u64(u64::from(v))).sum()
}

/// Writes `values` as unsigned LEB128 varints to the start of `out`.
///
/// Returns the number of bytes written.
///
/// # Panics
///
/// Panics if `out` is shorter than [`varint_encoded_len_u32`].
pub fn varint_encode_u32(values: &[u32], out: &mut [u8]) -> usize {
    let needed = varint_encoded_len_u32(values);
    assert!(out.len() >= needed, "varint_encode_u32: output too short");
    let mut pos = 0;
    for &v in values {
        let mut x = v;
        loop {
            let byte = (x & 0x7f) as u8;
            x >>= 7;
            if x == 0 {
                out[pos] = byte;
                pos += 1;
                break;
            }
            out[pos] = byte | 0x80;
            pos += 1;
        }
    }
    pos
}

/// Decodes unsigned LEB128 varints from `input` into `out`.
///
/// Decoding stops when `out` is full or `input` is exhausted. Returns
/// `(values_decoded, bytes_consumed)`, or `None` if a varint is truncated, longer
/// than 5 bytes, or does not fit in `u32`.
#[must_use]
pub fn varint_decode_u32(input: &[u8], out: &mut [u32]) -> Option<(usize, usize)> {
    let mut pos = 0;
    let mut count = 0;
    while count < out.len() && pos < input.len() {
        let mut value: u64 = 0;
        let mut shift = 0;
        loop {
            let byte = *input.get(pos)?;
            pos += 1;
            if shift >= 35 {
                return None;
            }
            value |= u64::from(byte & 0x7f) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        out[count] = u32::try_from(value).ok()?;
        count += 1;
    }
    Some((count, pos))
}

/// Bytes used by `n` values of `bits` bits each, rounded up to whole bytes.
#[must_use]
pub fn bitpacked_len(n: usize, bits: u32) -> usize {
    (n * bits as usize).div_ceil(8)
}

/// Packs `values` at `bits` bits each (1 to 32), least significant bit first, into
/// the start of `out`. Returns the number of bytes written.
///
/// # Panics
///
/// Panics if `bits` is outside `1..=32`, if a value does not fit in `bits` bits,
/// or if `out` is shorter than [`bitpacked_len`].
pub fn bitpack_u32(values: &[u32], bits: u32, out: &mut [u8]) -> usize {
    assert!((1..=32).contains(&bits), "bitpack_u32: bits must be 1..=32");
    let needed = bitpacked_len(values.len(), bits);
    assert!(out.len() >= needed, "bitpack_u32: output too short");
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut acc: u64 = 0;
    let mut acc_bits = 0u32;
    let mut pos = 0;
    for &v in values {
        assert!(
            v & !mask == 0,
            "bitpack_u32: value {v} does not fit in {bits} bits"
        );
        acc |= u64::from(v) << acc_bits;
        acc_bits += bits;
        while acc_bits >= 8 {
            out[pos] = acc as u8;
            pos += 1;
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        out[pos] = acc as u8;
        pos += 1;
    }
    pos
}

/// Unpacks `out.len()` values of `bits` bits each from `input` (LSB first).
///
/// Returns the number of values written (`out.len()`), or `None` if `input` is too
/// short or `bits` is outside `1..=32`.
#[must_use]
pub fn bitunpack_u32(input: &[u8], bits: u32, out: &mut [u32]) -> Option<usize> {
    if !(1..=32).contains(&bits) {
        return None;
    }
    if input.len() < bitpacked_len(out.len(), bits) {
        return None;
    }
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut acc: u64 = 0;
    let mut acc_bits = 0u32;
    let mut pos = 0;
    for slot in out.iter_mut() {
        while acc_bits < bits {
            acc |= u64::from(input[pos]) << acc_bits;
            pos += 1;
            acc_bits += 8;
        }
        *slot = (acc as u32) & mask;
        acc >>= bits;
        acc_bits -= bits;
    }
    Some(out.len())
}
