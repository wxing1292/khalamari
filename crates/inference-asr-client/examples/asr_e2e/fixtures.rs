use super::*;

pub async fn canonical(path: &Path) -> Result<Vec<u8>> {
    let result = tokio::time::timeout(
        Duration::from_secs(30),
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| Error::deadline_exceeded("fixture conversion timed out"))?
    .map_err(external)?;
    if !result.status.success() {
        return Err(Error::unavailable(format!(
            "fixture conversion: {}",
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    Ok(result.stdout)
}

async fn speak(directory: &Path, name: &str, voice: &str, text: &str) -> Result<Vec<u8>> {
    let path = directory.join(format!("{name}.aiff"));
    if !path.is_file() {
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            Command::new("say")
                .args(["-v", voice, "-r", "160", "-o"])
                .arg(&path)
                .arg(text)
                .kill_on_drop(true)
                .output(),
        )
        .await
        .map_err(|_| Error::deadline_exceeded("fixture synthesis timed out"))?
        .map_err(external)?;
        check(result.status.success(), "macOS say failed; install the selected voice")?;
    }
    canonical(&path).await
}

fn wav(path: &Path, pcm: &[u8], silence_samples: usize) -> Result<()> {
    check(pcm.len().is_multiple_of(2), "fixture has an incomplete i16 sample")?;
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .map_err(external)?;
    for sample in pcm.as_chunks::<2>().0 {
        writer.write_sample(i16::from_le_bytes(*sample)).map_err(external)?;
    }
    for _ in 0..silence_samples {
        writer.write_sample(0i16).map_err(external)?;
    }
    writer.finalize().map_err(external)
}

pub async fn synthetic(directory: &Path, paired_overlap: bool) -> Result<Vec<Case>> {
    let mut audio = BTreeMap::new();
    for (name, voice, text) in [
        (
            "english",
            "Samantha (English (US))",
            "Please open the project and check the streaming output.",
        ),
        ("chinese", "Tingting", "请打开这个项目，检查缓存机制是否正常工作。"),
        (
            "mixed-native",
            "Tingting",
            "Please open 这个项目 and check 缓存机制 with streaming output.",
        ),
    ] {
        audio.insert(name, speak(directory, name, voice, text).await?);
    }
    audio.insert(
        "english-chinese",
        [&audio["english"][..], &[0; 3200], &audio["chinese"]].concat(),
    );
    audio.insert(
        "chinese-english",
        [&audio["chinese"][..], &[0; 3200], &audio["english"]].concat(),
    );
    audio.insert(
        "long-mixed",
        [&audio["mixed-native"][..], &[0; 3200]].concat().repeat(4),
    );
    let context = "Software engineering discussion. Vocabulary: cache, KV cache, cached data, memory.";
    let mut cases = Vec::new();
    for (name, fragments) in [
        ("english", vec!["open the project", "streaming output"]),
        ("chinese", vec!["打开这个项目", "缓存机制"]),
        (
            "mixed-native",
            vec!["Please open", "这个项目", "缓存机制", "streaming output"],
        ),
        (
            "english-chinese",
            vec!["open the project", "streaming output", "缓存机制"],
        ),
        (
            "chinese-english",
            vec!["缓存机制", "open the project", "streaming output"],
        ),
        (
            "long-mixed",
            vec!["Please open", "这个项目", "缓存机制", "streaming output"],
        ),
    ] {
        let path = directory.join(format!("{name}.wav"));
        wav(&path, &audio[name], 16000)?;
        cases.push(serde_json::from_value(json!({"name":name,"file":path,"expected_fragments":fragments,"context":context,"window_seconds":if name=="long-mixed"{8}else{24},"expected_final_reasons":if name=="long-mixed"{vec!["limit"]}else{vec![]}})).unwrap());
    }
    if paired_overlap {
        for (name, voice, text, language, fragments, counts) in [
            (
                "repeat-en",
                "Samantha (English (US))",
                "No, no. Please keep the cache, keep the cache. Do not delete it.",
                Some("English"),
                vec!["no", "keep the cache", "do not delete"],
                json!({"no":2,"keep the cache":2}),
            ),
            (
                "repeat-zh",
                "Tingting",
                "不要，不要。保留缓存，保留缓存。不要删除这个项目。",
                Some("Chinese"),
                vec!["保留缓存", "不要删除"],
                json!({"不要":3,"保留缓存":2}),
            ),
            (
                "repeat-mix",
                "Tingting",
                "No, no, 保留缓存，保留缓存。Please keep the cache. 不要删除这个项目。",
                None,
                vec!["保留缓存", "keep the cache", "不要删除"],
                json!({"no":2,"保留缓存":2}),
            ),
        ] {
            let pcm = speak(directory, name, voice, text).await?;
            let path = directory.join(format!("{name}.wav"));
            wav(&path, &pcm, 32000)?;
            for overlap in [0, 1] {
                let mut case = json!({"name":format!("{name}-overlap-{overlap}"),"file":path,"reference_text":text,"expected_fragments":fragments,"expected_occurrences":counts,"window_seconds":3,"interval":1,"overlap_seconds":overlap,"context":context});
                if let Some(language) = language {
                    case["language"] = json!(language);
                }
                cases.push(serde_json::from_value(case).unwrap());
            }
        }
        for name in ["mixed-native", "long-mixed"] {
            for overlap in [0, 1] {
                cases.push(serde_json::from_value(json!({"name":format!("{name}-overlap-{overlap}"),"file":directory.join(format!("{name}.wav")),"reference_text":"Please open这个项目 and check缓存机制with streaming output.".repeat(if name=="long-mixed"{4}else{1}),"expected_fragments":["Please open","这个项目","缓存机制","streaming output"],"window_seconds":8,"overlap_seconds":overlap,"context":context})).unwrap());
            }
        }
    }
    Ok(cases)
}

pub async fn official(directory: &Path) -> Result<Vec<Case>> {
    let mut audio = BTreeMap::new();
    for language in ["en", "zh"] {
        let path = directory.join(format!("official-{language}.wav"));
        if !path.is_file() {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .map_err(external)?;
            let response = client
                .get(format!(
                    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_{language}.wav"
                ))
                .send()
                .await
                .and_then(reqwest::Response::error_for_status)
                .map_err(external)?;
            let mut stream = response.bytes_stream();
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(external)?;
                check(
                    chunk.len() <= 16 * 1024 * 1024 - bytes.len(),
                    "official fixture exceeds 16 MiB",
                )?;
                bytes.extend_from_slice(&chunk);
            }
            // Keep an incomplete download outside the final fixture path.
            let temporary = directory.join(format!("official-{language}.partial"));
            std::fs::write(&temporary, &bytes).map_err(external)?;
            std::fs::rename(temporary, &path).map_err(external)?;
        }
        audio.insert(language, canonical(&path).await?);
    }
    let mut result = Vec::new();
    for (first, second) in [("en", "zh"), ("zh", "en")] {
        let name = format!("human-{first}-{second}");
        let path = directory.join(format!("{name}.wav"));
        wav(&path, &[&audio[first][..], &[0; 3200], &audio[second]].concat(), 16000)?;
        result.push(serde_json::from_value(json!({"name":name,"file":path,"expected_fragments":["交易","writing for other people"],"window_seconds":8})).unwrap());
    }
    Ok(result)
}
