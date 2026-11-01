use super::*;

#[test]
fn test_feed_silence() {
    let (mut recording, mut session) = fixture_segmenter(&fixture_config());
    feed(&mut recording, &mut session, 10000, false);
    feed(&mut recording, &mut session, 32, true);
    let job = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(job.start > 5_000_000);
    assert_eq!(job.audio.len() as u64, job.end - job.start);
}

#[test]
fn test_feed_limit() {
    let (mut recording, mut session) = fixture_segmenter(&fixture_config());
    feed(&mut recording, &mut session, 63, true);
    let first = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!((first.start, first.end, first.reason), (0, 32000, "limit"));
    session.accept(&first, fixture_transcript("a")).unwrap();
    feed(&mut recording, &mut session, 20, true);
    session.stop();
    let second = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!(second.start, first.end);
    assert_eq!(second.end, 83 * 512);
}

#[test]
fn test_limit_cuts_at_the_quiet_interval_on_the_recording_clock() {
    let cfg = StreamConfig {
        window_seconds: 10.0,
        lookback_seconds: 2.0,
        ..fixture_config()
    };
    let (mut recording, mut session) = fixture_segmenter(&cfg);
    let mut audio = vec![1000i16; 160_000];
    audio[134_400..144_000].fill(0);
    for frame in audio.chunks(512) {
        recording
            .append(&frame.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        session.feed(session.end() + frame.len() as u64, true, None).unwrap();
    }
    let job = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!((job.reason, job.end), ("limit", 139_200));
}

#[test]
fn test_offline_cut_retains_suffix() {
    let cfg = fixture_config();
    let (mut recording, _path) = crate::recording::AudioRecordingWriter::create().unwrap();
    let mut session = AudioSegmenter::new(
        &cfg,
        recording.reader(),
        AnalysisPass::Offline,
        VecDeque::from([25_600]),
    );
    feed(&mut recording, &mut session, 63, true);
    let prefix = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!((prefix.start, prefix.end, prefix.reason), (0, 25_600, "limit"));
    session.stop();
    let suffix = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!((suffix.start, suffix.end), (prefix.end, 63 * 512));
}

#[test]
fn test_next_request_overlap() {
    let mut cfg = fixture_config();
    cfg.overlap_seconds = 1.0;
    cfg.window_seconds = 3.0;
    let (mut recording, mut session) = fixture_segmenter(&cfg);
    feed(&mut recording, &mut session, 32, true);
    feed(&mut recording, &mut session, 16, false);
    assert!(session.next_request(u64::MAX).unwrap().is_none());
    session.stop();
    let job = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(job.is_final());
    assert_eq!(
        (job.start, job.end, job.audio_start, job.audio.len()),
        (0, 32 * 512 + 3200, 0, 48 * 512)
    );
}

#[test]
fn test_final_reads_published_right_context() {
    let mut cfg = fixture_config();
    cfg.overlap_seconds = 0.5;
    let (mut recording, mut session) = fixture_segmenter(&cfg);
    feed(&mut recording, &mut session, 32, true);
    feed(&mut recording, &mut session, 16, false);
    let processed = session.end();
    let required = session.waiting_for_context().unwrap();
    recording.append(&vec![0; (required - processed) as usize * 2]).unwrap();
    let job = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(job.is_final());
    assert_eq!(job.audio_start + job.audio.len() as u64, required);
    assert_eq!(session.end(), processed);
}

#[test]
fn test_next_request_final_priority() {
    let (mut recording, mut session) = fixture_segmenter(&fixture_config());
    feed(&mut recording, &mut session, 32, true);
    feed(&mut recording, &mut session, 16, false);
    feed(&mut recording, &mut session, 32, true);
    let first = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(first.is_final());
    let second = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(!second.is_final());
    assert!(second.segment > first.segment);
    assert!(second.start >= first.end);
}

#[test]
fn test_accept_sentence() {
    let mut cfg = fixture_config();
    cfg.window_seconds = 10.0;
    let (mut recording, mut session) = fixture_segmenter(&cfg);
    feed(&mut recording, &mut session, 32, true);
    let first = session.next_request(u64::MAX).unwrap().unwrap();
    session.accept(&first, fixture_transcript("你好 hello.")).unwrap();
    feed(&mut recording, &mut session, 32, true);
    let second = session.next_request(u64::MAX).unwrap().unwrap();
    session.accept(&second, fixture_transcript("你好 hello.")).unwrap();
    feed(&mut recording, &mut session, 8, false);
    let final_job = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!(final_job.reason, "sentence");
    assert_eq!(final_job.end, 64 * 512 + 3200);
    feed(&mut recording, &mut session, 32, true);
    let next = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!(next.start, final_job.end);
}

#[test]
fn test_accept_closed_segment() {
    let (mut recording, mut session) = fixture_segmenter(&fixture_config());
    feed(&mut recording, &mut session, 32, true);
    let first = session.next_request(u64::MAX).unwrap().unwrap();
    feed(&mut recording, &mut session, 16, false);
    feed(&mut recording, &mut session, 32, true);
    session.accept(&first, fixture_transcript("Old sentence.")).unwrap();
    assert!(session.active.as_ref().unwrap().previous_sentence.is_empty());
}

#[test]
fn test_snapshot_ownership() {
    let (mut recording, mut session) = fixture_segmenter(&fixture_config());
    feed(&mut recording, &mut session, 32, true);
    let job = session.next_request(u64::MAX).unwrap().unwrap();
    recording.append(&2000i16.to_le_bytes().repeat(512)).unwrap();
    session.feed(session.end() + 512, true, None).unwrap();
    feed(&mut recording, &mut session, 16, false);
    let final_job = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(final_job.is_final());
    feed(&mut recording, &mut session, 1000, false);
    assert_eq!(&*job.audio, vec![1000; 32 * 512]);
    assert_eq!(job.owned(), &*job.audio);
}

#[test]
fn test_finish_full_queue() {
    let mut cfg = fixture_config();
    cfg.max_pending_finals = 1;
    let (mut recording, mut session) = fixture_segmenter(&cfg);
    feed(&mut recording, &mut session, 63, true);
    assert!(!session.can_feed());
    let first = session.next_request(u64::MAX).unwrap().unwrap();
    assert!(session.can_feed());
    session.stop();
    let second = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!(first.end, second.start);
    assert_eq!(second.end, 63 * 512);
    assert!(first.is_final() && second.is_final());
    assert!(session.next_request(u64::MAX).unwrap().is_none());
}

#[test]
fn test_accept_revision_order() {
    let mut cfg = fixture_config();
    cfg.window_seconds = 10.0;
    let (mut recording, mut session) = fixture_segmenter(&cfg);
    feed(&mut recording, &mut session, 32, true);
    let first = session.next_request(u64::MAX).unwrap().unwrap();
    feed(&mut recording, &mut session, 20, true);
    assert!(session.next_request(u64::MAX).unwrap().is_none());
    feed(&mut recording, &mut session, 12, true);
    let second = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!((first.revision, second.revision), (1, 2));
    let event = session
        .accept(&second, fixture_transcript("new 中文"))
        .unwrap()
        .unwrap();
    assert_eq!((event.kind, event.segment, event.revision), ("partial", 0, 2));
    assert!(
        session
            .accept(&first, Err(inference_error::Error::unavailable("late HTTP failure")))
            .unwrap()
            .is_none()
    );
    feed(&mut recording, &mut session, 16, false);
    let final_job = session.next_request(u64::MAX).unwrap().unwrap();
    assert_eq!(final_job.revision, 3);
    let event = session
        .accept(&final_job, fixture_transcript("final 中文"))
        .unwrap()
        .unwrap();
    assert_eq!((event.kind, event.revision), ("final", 3));
    assert_eq!(event.start_ms, 0.0);
    assert_eq!(event.end_ms, event.end as f64 / 16.0);
    assert!(
        session
            .accept(&second, fixture_transcript("late draft"))
            .unwrap()
            .is_none()
    );
    assert!(session.next_request(u64::MAX).unwrap().is_none());
}

fn fixture_config() -> StreamConfig {
    StreamConfig {
        sentence_silence_seconds: 0.25,
        window_seconds: 2.0,
        ..StreamConfig::default()
    }
}

fn fixture_segmenter(config: &StreamConfig) -> (crate::recording::AudioRecordingWriter, AudioSegmenter) {
    let (recording, _path) = crate::recording::AudioRecordingWriter::create().unwrap();
    let segmenter = AudioSegmenter::new(config, recording.reader(), AnalysisPass::Live, VecDeque::new());
    (recording, segmenter)
}

fn feed(
    recording: &mut crate::recording::AudioRecordingWriter,
    session: &mut AudioSegmenter,
    frames: usize,
    speech: bool,
) {
    for _ in 0..frames {
        recording.append(&1000i16.to_le_bytes().repeat(512)).unwrap();
        session.feed(session.end() + 512, speech, None).unwrap();
    }
}

fn fixture_transcript(text: &str) -> Result<AudioSegmentTranscribeResp> {
    Ok(AudioSegmentTranscribeResp {
        transcript: crate::transcript::AudioTranscript {
            sample_rate: SAMPLE_RATE,
            text: text.into(),
            language: "Chinese,English".into(),
            spans: vec![],
            turns: vec![],
        },
        metrics: Default::default(),
        transport: Default::default(),
        language_ranges: None,
    })
}
