use std::io::Cursor;
use std::sync::Arc;

use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioSource;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRPreprocessorConfig;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use rubato::Fft as FftResampler;
use rubato::FixedSync;
use rubato::Resampler;
use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rustfft::Fft;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

const TARGET_SAMPLE_RATE: u32 = 16_000;
const RESAMPLE_CHUNK_SIZE: usize = 1_024;

pub struct Qwen3ASRAudioPreprocessor {
    config: Qwen3ASRPreprocessorConfig,
    fft: Arc<dyn Fft<f32>>,
    mel_filters: Vec<Vec<(usize, f32)>>,
    window: Vec<f32>,
}

impl Qwen3ASRAudioPreprocessor {
    pub fn new(config: Qwen3ASRPreprocessorConfig) -> Self {
        assert!(
            config.chunk_length_seconds > 0,
            "ASR audio chunk length must be positive"
        );
        assert!(config.feature_size > 0, "ASR audio feature size must be positive");
        assert!(config.hop_length > 0, "ASR audio hop length must be positive");
        assert!(config.n_fft >= 2, "ASR audio FFT length must be at least two");
        assert!(
            config.n_samples >= config.hop_length,
            "ASR audio sample capacity is too small"
        );
        assert_eq!(
            config.max_frames,
            config.n_samples / config.hop_length,
            "ASR audio frame capacity must match its sample and hop capacities"
        );
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let mel_filters = mel_filters(TARGET_SAMPLE_RATE, config.n_fft, config.feature_size);
        let window = (0..config.n_fft)
            .map(|index| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / config.n_fft as f32).cos())
            .collect();
        Self {
            config,
            fft,
            mel_filters,
            window,
        }
    }

    pub fn prepare_wav(&self, bytes: &[u8]) -> Result<Qwen3ASRAudioSource> {
        let decoded = decode_wav(bytes)?;
        let mut samples = resample(
            decoded.samples,
            decoded.sample_rate,
            TARGET_SAMPLE_RATE,
            self.config.n_samples,
        )?;
        // Decoding already normalized the peak. Only resampling can introduce overshoot.
        if decoded.sample_rate != TARGET_SAMPLE_RATE {
            normalize_samples(&mut samples);
        }
        self.prepare_samples(&samples)
    }

    pub fn prepare_pcm(&self, samples: impl Iterator<Item = i16>) -> Result<Qwen3ASRAudioSource> {
        let mut samples: Vec<_> = samples.map(|sample| f32::from(sample) / 32768.0).collect();
        if samples.is_empty() {
            return Err(Error::invalid_argument("PCM range must not be empty"));
        }
        // Match the client's former 10-ms padding for short language ranges.
        samples.resize(samples.len().max(self.config.hop_length), 0.0);
        self.prepare_samples(&samples)
    }

    fn prepare_samples(&self, samples: &[f32]) -> Result<Qwen3ASRAudioSource> {
        if samples.len() < self.config.hop_length || samples.len() > self.config.n_samples {
            return Err(Error::invalid_argument(format!(
                "Qwen3-ASR audio must contain {}..={} canonical samples",
                self.config.hop_length, self.config.n_samples
            )));
        }

        let num_frames = samples.len() / self.config.hop_length;
        let features = self.log_mel(samples, num_frames);
        Ok(Qwen3ASRAudioSource::new(features, self.config.feature_size, num_frames)
            .expect("Qwen3-ASR preprocessing must produce a valid Audio Tower source"))
    }

    fn log_mel(&self, samples: &[f32], num_frames: usize) -> Vec<f32> {
        let num_frequency_bins = self.config.n_fft / 2 + 1;
        let mut frame = vec![Complex32::default(); self.config.n_fft];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; num_frequency_bins];
        let mut features = vec![0.0; self.config.feature_size * num_frames];
        let pad = self.config.n_fft / 2;

        for frame_index in 0..num_frames {
            for (index, value) in frame.iter_mut().enumerate() {
                let sample_index = frame_index * self.config.hop_length + index;
                let sample_index = sample_index as isize - pad as isize;
                *value = Complex32::new(reflect_sample(samples, sample_index) * self.window[index], 0.0);
            }
            self.fft.process_with_scratch(&mut frame, &mut scratch);
            for (value, frequency) in power.iter_mut().zip(&frame[..num_frequency_bins]) {
                *value = frequency.norm_sqr();
            }
            for (mel_index, filter) in self.mel_filters.iter().enumerate() {
                features[mel_index * num_frames + frame_index] = filter
                    .iter()
                    .map(|&(index, weight)| weight * power[index])
                    .sum::<f32>()
                    .max(1e-10)
                    .log10();
            }
        }

        let maximum = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let minimum = maximum - 8.0;
        for value in &mut features {
            *value = (value.max(minimum) + 4.0) / 4.0;
        }
        features
    }
}

struct DecodedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn decode_wav(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|error| Error::invalid_argument(format!("unable to parse WAV input: {error}")))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Err(Error::invalid_argument("WAV sample rate must be positive"));
    }
    if spec.channels == 0 {
        return Err(Error::invalid_argument("WAV channel count must be positive"));
    }
    if spec.bits_per_sample == 0 {
        return Err(Error::invalid_argument("WAV sample width must be positive"));
    }

    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|error| Error::invalid_argument(format!("unable to decode float WAV samples: {error}")))?
        },
        hound::SampleFormat::Int => {
            if spec.bits_per_sample > 32 {
                return Err(Error::invalid_argument(format!(
                    "unsupported WAV integer width {}",
                    spec.bits_per_sample
                )));
            }
            let scale = 2.0f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|value| value as f32 / scale).map_err(|error| {
                        Error::invalid_argument(format!("unable to decode integer WAV sample: {error}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        },
    };

    let num_channels = usize::from(spec.channels);
    if !interleaved.len().is_multiple_of(num_channels) {
        return Err(Error::invalid_argument(
            "WAV sample count must be a multiple of its channel count",
        ));
    }
    let mut samples = Vec::with_capacity(interleaved.len() / num_channels);
    for channels in interleaved.chunks_exact(num_channels) {
        let sample = channels.iter().map(|&value| f64::from(value)).sum::<f64>() / num_channels as f64;
        if !sample.is_finite() {
            return Err(Error::invalid_argument("WAV contains a non-finite sample"));
        }
        samples.push(sample as f32);
    }

    normalize_samples(&mut samples);
    Ok(DecodedAudio {
        samples,
        sample_rate: spec.sample_rate,
    })
}

fn normalize_samples(samples: &mut [f32]) {
    let peak = samples.iter().copied().map(f32::abs).fold(0.0, f32::max);
    if peak > 1.0 {
        for sample in samples {
            *sample /= peak;
        }
    }
}

fn resample(samples: Vec<f32>, source_rate: u32, target_rate: u32, max_output_samples: usize) -> Result<Vec<f32>> {
    let output_len = samples
        .len()
        .checked_mul(target_rate as usize)
        .ok_or_else(|| Error::invalid_argument("resampled WAV length exceeds usize"))?
        .div_ceil(source_rate as usize);
    if output_len > max_output_samples {
        return Err(Error::invalid_argument(format!(
            "Qwen3-ASR WAV has {output_len} samples after resampling; maximum is {max_output_samples}"
        )));
    }
    if source_rate == target_rate {
        return Ok(samples);
    }
    if samples.is_empty() {
        return Ok(samples);
    }

    let input_len = samples.len();
    let input = InterleavedOwned::new_from(samples, 1, input_len)
        .expect("mono WAV samples must form a valid interleaved audio buffer");
    let mut resampler = FftResampler::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        RESAMPLE_CHUNK_SIZE,
        1,
        FixedSync::Both,
    )
    .expect("positive WAV sample rates must initialize the resampler");
    let output = resampler
        .process_all(&input, input_len, None)
        .expect("the WAV resampler must process a valid mono audio buffer");
    Ok(output.take_data())
}

fn reflect_sample(samples: &[f32], index: isize) -> f32 {
    if samples.len() == 1 {
        return samples[0];
    }
    let period = 2 * (samples.len() - 1) as isize;
    let index = index.rem_euclid(period);
    let index = if index >= samples.len() as isize {
        period - index
    } else {
        index
    };
    samples[index as usize]
}

fn mel_filters(sample_rate: u32, n_fft: usize, num_mel_bins: usize) -> Vec<Vec<(usize, f32)>> {
    let num_frequency_bins = n_fft / 2 + 1;
    let max_frequency = sample_rate as f32 / 2.0;
    let max_mel = hz_to_slaney_mel(max_frequency);
    let mel_points = (0..num_mel_bins + 2)
        .map(|index| max_mel * index as f32 / (num_mel_bins + 1) as f32)
        .map(slaney_mel_to_hz)
        .collect::<Vec<_>>();
    let frequencies = (0..num_frequency_bins)
        .map(|index| sample_rate as f32 * index as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    (0..num_mel_bins)
        .map(|mel_index| {
            let left = mel_points[mel_index];
            let center = mel_points[mel_index + 1];
            let right = mel_points[mel_index + 2];
            let normalization = 2.0 / (right - left);
            frequencies
                .iter()
                .enumerate()
                .filter_map(|(index, &frequency)| {
                    let lower = (frequency - left) / (center - left);
                    let upper = (right - frequency) / (right - center);
                    let weight = 0.0f32.max(lower.min(upper)) * normalization;
                    (weight != 0.0).then_some((index, weight))
                })
                .collect()
        })
        .collect()
}

fn hz_to_slaney_mel(frequency: f32) -> f32 {
    const LINEAR_HZ_PER_MEL: f32 = 200.0 / 3.0;
    if frequency < 1_000.0 {
        frequency / LINEAR_HZ_PER_MEL
    } else {
        15.0 + (frequency / 1_000.0).ln() / (6.4f32.ln() / 27.0)
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const LINEAR_HZ_PER_MEL: f32 = 200.0 / 3.0;
    if mel < 15.0 {
        LINEAR_HZ_PER_MEL * mel
    } else {
        1_000.0 * ((6.4f32.ln() / 27.0) * (mel - 15.0)).exp()
    }
}

#[cfg(test)]
#[path = "audio_test.rs"]
mod tests;
