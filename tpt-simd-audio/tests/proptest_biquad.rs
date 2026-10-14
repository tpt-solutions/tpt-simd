//! Property tests: biquad block processing matches sample-by-sample ticks, and reset restores state.

use proptest::prelude::*;
use tpt_simd_audio::Biquad;

proptest! {
    #[test]
    fn process_equals_repeated_tick(
        x in prop::collection::vec(-1.0f32..1.0, 0..400),
        f0 in 100.0f64..10_000.0,
        q in 0.3f64..4.0,
        lowpass in any::<bool>(),
    ) {
        let make = || if lowpass {
            Biquad::lowpass(48_000.0, f0, q)
        } else {
            Biquad::highpass(48_000.0, f0, q)
        };

        let mut block = make();
        let mut out = x.clone();
        block.process(&mut out);

        let mut per_sample = make();
        for (i, &s) in x.iter().enumerate() {
            let y = per_sample.tick(s);
            prop_assert_eq!(out[i].to_bits(), y.to_bits());
        }
    }

    #[test]
    fn reset_makes_the_filter_repeat_its_output(x in prop::collection::vec(-1.0f32..1.0, 1..300)) {
        let mut filter = Biquad::lowpass(48_000.0, 2_000.0, 0.707);
        let mut first = x.clone();
        filter.process(&mut first);
        filter.reset();
        let mut second = x.clone();
        filter.process(&mut second);
        prop_assert_eq!(first, second);
    }
}
