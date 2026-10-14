//! Resampler checked against `rubato`, an independent sinc resampler.
//!
//! The two kernels differ, so output samples are not compared directly. Instead, a
//! tone is resampled with each, and the amplitude of the tone is measured at the output
//! rate in the settled middle of the signal. Both must keep the tone's amplitude, and must
//! agree with each other, to a few per cent.

use audioadapter_buffers::owned::SequentialOwned;
use rubato::audioadapter::Adapter;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use tpt_simd_audio::resample_sinc;

const IN_RATE: f64 = 48_000.0;
const OUT_RATE: f64 = 24_000.0;
const TONE_HZ: f64 = 1_000.0;

/// Amplitude of a sine at `freq` in `signal` (`fs` sample rate), by a single-bin DFT
/// over the middle half of the signal, where both filters have settled.
fn tone_amplitude(signal: &[f32], freq: f64, fs: f64) -> f64 {
    let start = signal.len() / 4;
    let end = signal.len() * 3 / 4;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, &s) in signal[start..end].iter().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * freq * (n + start) as f64 / fs;
        re += f64::from(s) * phase.cos();
        im += f64::from(s) * phase.sin();
    }
    2.0 * (re * re + im * im).sqrt() / (end - start) as f64
}

fn tone(len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| (2.0 * std::f64::consts::PI * TONE_HZ * n as f64 / IN_RATE).sin() as f32)
        .collect()
}

fn resample_with_rubato(input: &[f32]) -> Vec<f32> {
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler =
        Async::<f32>::new_sinc(OUT_RATE / IN_RATE, 1.1, &params, 1024, 1, FixedAsync::Input)
            .expect("rubato construction");

    let out_len = resampler.process_all_needed_output_len(input.len());
    let input_buf = SequentialOwned::new_from(input.to_vec(), 1, input.len()).expect("input");
    let mut output_buf =
        SequentialOwned::new_from(vec![0f32; out_len], 1, out_len).expect("output");
    resampler
        .process_all_into_buffer(&input_buf, &mut output_buf, input.len(), None)
        .expect("rubato processing");

    (0..out_len)
        .map(|n| output_buf.read_sample(0, n).expect("in range"))
        .collect()
}

#[test]
fn downsampled_tone_keeps_amplitude_and_matches_rubato() {
    let input = tone(9_600);
    let out_len = (input.len() as f64 * OUT_RATE / IN_RATE) as usize;

    let mut ours = vec![0f32; out_len];
    let written = resample_sinc(&input, IN_RATE, OUT_RATE, 32, &mut ours);
    assert_eq!(written, out_len);

    let theirs = resample_with_rubato(&input);

    let a_ours = tone_amplitude(&ours, TONE_HZ, OUT_RATE);
    let a_theirs = tone_amplitude(&theirs, TONE_HZ, OUT_RATE);
    assert!((a_ours - 1.0).abs() < 0.02, "ours: amplitude {a_ours}");
    assert!(
        (a_theirs - 1.0).abs() < 0.02,
        "rubato: amplitude {a_theirs}"
    );
    assert!(
        (a_ours - a_theirs).abs() < 0.02,
        "ours {a_ours} vs rubato {a_theirs}"
    );
}
