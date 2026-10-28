use super::*;

#[test]
fn test_vad_recurrence() {
    let mut vad = Vad::load().unwrap();
    let (mut h, mut c) = (vec![0.0f32; 128], vec![0.0f32; 128]);
    let mut context = [0.0; 64];
    for (index, count) in [512, 17, 512, 31, 511, 512].into_iter().enumerate() {
        let pcm: Vec<_> = (0..count)
            .map(|i| (((i + index * 73) as f32 * 0.037).sin() * 12000.0) as i16)
            .collect();
        let mut input = vec![0.0; 576];
        input[..64].copy_from_slice(&context);
        for (value, &sample) in input[64..].iter_mut().zip(&pcm) {
            *value = f32::from(sample) / 32768.0;
        }
        context.copy_from_slice(&input[512..]);
        let actual = vad.probability(&pcm).unwrap();
        // The graph is stateless. Independent input tensors carry the reference recurrence.
        let outputs = vad
            .session
            .run(ort::inputs![
                "input" => Tensor::from_array(([1,576],input)).unwrap(),
                "h" => Tensor::from_array(([1,1,128],h)).unwrap(),
                "c" => Tensor::from_array(([1,1,128],c)).unwrap(),
            ])
            .unwrap();
        let expected = outputs["speech_probs"].try_extract_tensor::<f32>().unwrap().1[0];
        assert_eq!(actual.to_bits(), expected.to_bits());
        h = outputs["hn"].try_extract_tensor::<f32>().unwrap().1.to_vec();
        c = outputs["cn"].try_extract_tensor::<f32>().unwrap().1.to_vec();
    }
}

#[test]
#[ignore = "requires native speech assets and KHALAMARI_ASR_ORACLE reference data"]
fn test_models_migration() {
    let config = LanguageConfig::default();
    let mut vad = Vad::load().unwrap();
    let mut model = AcousticModel::load(&config).unwrap();
    let path = std::env::var("KHALAMARI_ASR_ORACLE").expect("set KHALAMARI_ASR_ORACLE to the migration reference JSON");
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let bytes = std::fs::read(case["pcm"].as_str().unwrap()).unwrap();
        let pcm: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| i16::from_le_bytes(*v))
            .collect();
        let (features, frames) = model.mel.features(&pcm);
        let reference = std::fs::read(case["features"].as_str().unwrap()).unwrap();
        assert_eq!(frames, reference.len() / (80 * 4));
        assert_eq!(features.len() * 4, reference.len());
        let max_error = features
            .iter()
            .zip(reference.as_chunks::<4>().0.iter())
            .map(|(a, b)| (a - f32::from_le_bytes(*b)).abs())
            .fold(0.0, f32::max);
        println!("{} Mel max error: {max_error}", case["name"]);
        assert!(max_error < 2e-4);
        let (language, probability) = model.language(&pcm).unwrap();
        println!("{} LID: {language} {probability}", case["name"]);
        assert_eq!(language, case["language"]);
        println!(
            "LID probability delta: {}",
            (f64::from(probability) - case["score"].as_f64().unwrap()).abs()
        );
        assert!(probability.is_finite() && (0.0..=1.0).contains(&probability));
        for (index, expected) in case["vad"].as_array().unwrap().iter().enumerate() {
            let score = vad.probability(&pcm[index * 512..(index + 1) * 512]).unwrap();
            assert!((f64::from(score) - expected.as_f64().unwrap()).abs() < 1e-5);
        }
        let text = case["text"].as_str().unwrap();
        if let Some(units) = model.align(&pcm, text, &language).unwrap() {
            assert_eq!(units.iter().map(|unit| unit.text.as_str()).collect::<String>(), text);
            assert!(crate::overlap::timed_words(text, &units, 0).is_some());
        } else {
            panic!("reference text must align: {}", case["name"]);
        }
    }
}
