use std::io::Cursor;

use super::*;

#[test]
fn test_odd_fft_uses_discrete_frequency_spacing() {
    // Upstream Slaney reference: FFT length 5 at 16 kHz has bins at 0, 3200 and 6400 Hz.
    let odd = mel_filters(16000, 5, 2);
    for (index, expected) in [(1, 0.000_265_809_77f32), (2, 0.000_088_603_26)] {
        let actual = odd[1].iter().find(|&&(bin, _)| bin == index).unwrap().1;
        assert!((actual - expected).abs() < expected * 1e-5);
    }
}

#[test]
fn test_prepare_wav_resamples_audio() {
    let bytes = sine_wav(8_000, 1.0);
    let source = fixture_preprocessor().prepare_wav(&bytes).unwrap();

    assert_eq!(source.num_mel_bins(), 128);
    assert_eq!(source.num_frames(), 100);
    assert_eq!(source.features().len(), 128 * 100);
    assert!(source.features().iter().all(|value| value.is_finite()));
}

#[test]
fn test_prepare_wav_matches_whisper_log_mel_reference() {
    let bytes = sine_wav(16_000, 1.0);
    let source = fixture_preprocessor().prepare_wav(&bytes).unwrap();
    let indexes = [0, 1, 50, 99, 1_037, 6_450, 12_799];
    let expected = [
        1.169_255, 0.821_978, 0.826_407, 0.821_445, 0.181_805, -0.694_741, -0.694_741,
    ];

    for (index, expected) in indexes.into_iter().zip(expected) {
        let actual = source.features()[index];
        assert!(
            (actual - expected).abs() < 1e-5,
            "feature[{index}]={actual}; expected {expected}"
        );
    }
}

#[test]
fn test_prepare_wav_rejects_audio_outside_checkpoint_duration() {
    let too_short = sine_wav(16_000, 0.005);
    assert!(fixture_preprocessor().prepare_wav(&too_short).is_err());

    let config = fixture_config();
    let too_long = sine_wav(16_000, (config.n_samples + 1) as f32 / 16_000.0);
    assert!(Qwen3ASRAudioPreprocessor::new(config).prepare_wav(&too_long).is_err());
}

fn fixture_preprocessor() -> Qwen3ASRAudioPreprocessor {
    Qwen3ASRAudioPreprocessor::new(fixture_config())
}

fn fixture_config() -> Qwen3ASRPreprocessorConfig {
    Qwen3ASRPreprocessorConfig {
        chunk_length_seconds: 30,
        feature_size: 128,
        hop_length: 160,
        n_fft: 400,
        n_samples: 480_000,
        max_frames: 3_000,
        dither: 0.0,
    }
}

fn sine_wav(sample_rate: u32, duration_seconds: f32) -> Vec<u8> {
    let mut bytes = Cursor::new(vec![]);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut bytes, spec).unwrap();
    let num_samples = (sample_rate as f32 * duration_seconds).round() as usize;
    for index in 0..num_samples {
        let sample = 0.25 * (2.0 * std::f32::consts::PI * 100.0 * index as f32 / sample_rate as f32).sin();
        writer.write_sample((sample * i16::MAX as f32) as i16).unwrap();
    }
    writer.finalize().unwrap();
    bytes.into_inner()
}
