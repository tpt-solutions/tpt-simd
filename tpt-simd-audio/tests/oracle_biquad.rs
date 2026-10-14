//! Biquad output checked against the `biquad` crate (independent RBJ-cookbook implementation).
//!
//! Our filter runs in f32 on f32 samples, the oracle in f64, so the comparison allows for f32
//! rounding that accumulates through the recursion.
//!
//! `biquad` 0.5.0 maps the cutoff to `omega = pi * f0 / (2 fs)`, a quarter of the RBJ
//! `omega = 2 pi f0 / fs` that this crate uses. The oracle is therefore given `4 * f0`, which puts
//! both filters at the same cutoff (checked by hand: impulse responses agree to f32 precision).

use biquad::{Biquad as _, Coefficients, DirectForm1, ToHertz, Type};
use proptest::prelude::*;
use tpt_simd_audio::Biquad;

const FS: f64 = 48_000.0;

proptest! {
    #[test]
    fn lowpass_matches_oracle(
        x in prop::collection::vec(-1.0f32..1.0, 1..400),
        f0 in 200.0f64..5_000.0,
        q in 0.5f64..3.0,
    ) {
        check(true, &x, f0, q)?;
    }

    #[test]
    fn highpass_matches_oracle(
        x in prop::collection::vec(-1.0f32..1.0, 1..400),
        f0 in 200.0f64..5_000.0,
        q in 0.5f64..3.0,
    ) {
        check(false, &x, f0, q)?;
    }
}

fn check(lowpass: bool, x: &[f32], f0: f64, q: f64) -> Result<(), TestCaseError> {
    let ours_ctor: fn(f64, f64, f64) -> Biquad = if lowpass {
        Biquad::lowpass
    } else {
        Biquad::highpass
    };
    let oracle_kind: Type<f64> = if lowpass {
        Type::LowPass
    } else {
        Type::HighPass
    };
    // See the module docs: the oracle needs 4 * f0 to sit at the same cutoff as ours.
    let coeffs = Coefficients::<f64>::from_params(oracle_kind, FS.hz(), (4.0 * f0).hz(), q)
        .expect("valid parameters");
    let mut oracle = DirectForm1::<f64>::new(coeffs);
    let mut ours = ours_ctor(FS, f0, q);

    let mut out = x.to_vec();
    ours.process(&mut out);
    for (i, &s) in x.iter().enumerate() {
        let want = oracle.run(f64::from(s));
        let got = f64::from(out[i]);
        prop_assert!(
            (got - want).abs() <= 1e-3 * want.abs().max(1.0),
            "sample {}: ours {} oracle {}",
            i,
            got,
            want
        );
    }
    Ok(())
}
