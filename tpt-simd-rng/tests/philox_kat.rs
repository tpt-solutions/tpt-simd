//! Philox4x32-10 against the published Random123 known-answer vectors.

use tpt_simd_rng::philox4x32_10;

#[test]
fn random123_known_answers() {
    // Random123 `kat_vectors` for philox4x32 with 10 rounds.
    assert_eq!(
        philox4x32_10([0, 0, 0, 0], [0, 0]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        philox4x32_10([0xffff_ffff; 4], [0xffff_ffff; 2]),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
    );
    // The digits of pi as counter and key.
    assert_eq!(
        philox4x32_10(
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0]
        ),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
}

#[test]
fn counter_changes_every_output_word() {
    let base = philox4x32_10([1, 2, 3, 4], [5, 6]);
    for lane in 0..4 {
        let mut c = [1, 2, 3, 4];
        c[lane] ^= 1;
        let other = philox4x32_10(c, [5, 6]);
        assert_ne!(base, other, "flipping counter word {lane} changed nothing");
    }
}
