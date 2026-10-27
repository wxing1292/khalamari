use super::*;

#[test]
fn test_snapshot_sample_limit() {
    let mut cfg = fixture_config();
    cfg.window_seconds = 3.0;
    cfg.overlap_seconds = 1.0;
    cfg.padding_seconds = 1.0;
    cfg.silence_seconds = 0.1;
    cfg.sentence_silence_seconds = 0.1;
    let mut session = AudioSegmenter::new(&cfg);
    let base = u64::MAX - 16000;
    session.base = base;
    session.end = base;
    feed(&mut session, 1, true);
    feed(&mut session, 4, false);
    session.finish().unwrap();
    let job = session.next_request().unwrap();
    assert_eq!(job.reason, "silence");
    assert_eq!((job.start_sample, job.end_sample), (base, base + 5 * 512));
    assert_eq!((job.audio_start_sample + job.audio.len() as u64), job.end_sample);
    assert_eq!(job.owned().len(), 5 * 512);
}

#[test]
fn test_feed_silence() {
    let mut session = AudioSegmenter::new(&fixture_config());
    feed(&mut session, 10000, false);
    assert!(session.pcm.len() <= 3200);
    feed(&mut session, 32, true);
    let job = session.next_request().unwrap();
    assert!(job.start_sample > 5_000_000);
    assert_eq!(job.audio.len() as u64, job.end_sample - job.start_sample);
}

#[test]
fn test_feed_limit() {
    let mut session = AudioSegmenter::new(&fixture_config());
    feed(&mut session, 63, true);
    let first = session.next_request().unwrap();
    assert_eq!(
        (first.start_sample, first.end_sample, first.reason),
        (0, 32000, "limit")
    );
    session.accept(&first, fixture_transcript("a")).unwrap();
    feed(&mut session, 20, true);
    session.finish().unwrap();
    let second = session.next_request().unwrap();
    assert_eq!(second.start_sample, first.end_sample);
    assert_eq!(second.end_sample, 83 * 512);
}

#[test]
fn test_next_request_overlap() {
    let mut cfg = fixture_config();
    cfg.overlap_seconds = 1.0;
    cfg.window_seconds = 3.0;
    let mut session = AudioSegmenter::new(&cfg);
    feed(&mut session, 32, true);
    feed(&mut session, 16, false);
    assert!(session.next_request().is_none());
    session.finish().unwrap();
    let job = session.next_request().unwrap();
    assert!(job.is_final());
    assert_eq!(
        (
            job.start_sample,
            job.end_sample,
            job.audio_start_sample,
            job.audio.len()
        ),
        (0, 32 * 512 + 3200, 0, 48 * 512)
    );
}

#[test]
fn test_next_request_final_priority() {
    let mut session = AudioSegmenter::new(&fixture_config());
    feed(&mut session, 32, true);
    feed(&mut session, 16, false);
    feed(&mut session, 32, true);
    let first = session.next_request().unwrap();
    assert!(first.is_final());
    let second = session.next_request().unwrap();
    assert!(!second.is_final());
    assert!(second.segment > first.segment);
    assert!(second.start_sample >= first.end_sample);
}

#[test]
fn test_feed_overload() {
    let mut cfg = fixture_config();
    cfg.max_pending_finals = 1;
    let mut session = AudioSegmenter::new(&cfg);
    feed(&mut session, 124, true);
    assert!(matches!(
        session.feed(&[1000; 512], true),
        Err(Error::ResourceExhausted(_))
    ));
}

#[test]
fn test_accept_sentence() {
    let mut cfg = fixture_config();
    cfg.window_seconds = 10.0;
    let mut session = AudioSegmenter::new(&cfg);
    feed(&mut session, 32, true);
    let first = session.next_request().unwrap();
    session.accept(&first, fixture_transcript("你好 hello.")).unwrap();
    feed(&mut session, 32, true);
    let second = session.next_request().unwrap();
    session.accept(&second, fixture_transcript("你好 hello.")).unwrap();
    feed(&mut session, 8, false);
    let final_job = session.next_request().unwrap();
    assert_eq!(final_job.reason, "sentence");
    assert_eq!(final_job.end_sample, 64 * 512 + 3200);
    feed(&mut session, 32, true);
    let next = session.next_request().unwrap();
    assert_eq!(next.start_sample, final_job.end_sample);
}

#[test]
fn test_accept_closed_segment() {
    let mut session = AudioSegmenter::new(&fixture_config());
    feed(&mut session, 32, true);
    let first = session.next_request().unwrap();
    feed(&mut session, 16, false);
    feed(&mut session, 32, true);
    session.accept(&first, fixture_transcript("Old sentence.")).unwrap();
    assert!(session.active.as_ref().unwrap().previous_sentence.is_empty());
}

#[test]
fn test_snapshot_ownership() {
    let mut session = AudioSegmenter::new(&fixture_config());
    feed(&mut session, 32, true);
    let job = session.next_request().unwrap();
    session.feed(&[2000; 512], true).unwrap();
    feed(&mut session, 16, false);
    let final_job = session.next_request().unwrap();
    assert!(final_job.is_final());
    feed(&mut session, 1000, false);
    assert!(session.base > job.start_sample);
    assert_eq!(&*job.audio, vec![1000; 32 * 512]);
    assert_eq!(job.owned(), &*job.audio);
}

#[test]
fn test_finish_full_queue() {
    let mut cfg = fixture_config();
    cfg.max_pending_finals = 1;
    let mut session = AudioSegmenter::new(&cfg);
    feed(&mut session, 70, true);
    assert_eq!(session.pending.len(), 1);
    session.finish().unwrap();
    let first = session.next_request().unwrap();
    let second = session.next_request().unwrap();
    assert_eq!(first.end_sample, second.start_sample);
    assert_eq!(second.end_sample, 70 * 512);
    assert!(first.is_final() && second.is_final());
    assert!(session.next_request().is_none());
}

#[test]
fn test_accept_revision_order() {
    let mut cfg = fixture_config();
    cfg.window_seconds = 10.0;
    let mut session = AudioSegmenter::new(&cfg);
    feed(&mut session, 32, true);
    let first = session.next_request().unwrap();
    feed(&mut session, 20, true);
    assert!(session.next_request().is_none());
    feed(&mut session, 12, true);
    let second = session.next_request().unwrap();
    assert_eq!((first.revision, second.revision), (1, 2));
    let events = session.accept(&second, fixture_transcript("new 中文")).unwrap();
    assert_eq!(
        (events[0].kind, events[0].segment, events[0].revision),
        ("partial", 0, 2)
    );
    assert!(
        session
            .accept(&first, Err(Error::unavailable("late HTTP failure")))
            .unwrap()
            .is_empty()
    );
    feed(&mut session, 16, false);
    let final_job = session.next_request().unwrap();
    assert_eq!(final_job.revision, 3);
    let events = session.accept(&final_job, fixture_transcript("final 中文")).unwrap();
    assert_eq!((events[0].kind, events[0].revision), ("final", 3));
    assert_eq!(events[0].start_ms, 0.0);
    assert_eq!(events[0].end_ms, events[0].end_sample as f64 / 16.0);
    assert!(
        session
            .accept(&second, fixture_transcript("late draft"))
            .unwrap()
            .is_empty()
    );
    assert!(session.next_request().is_none());
}

#[test]
fn test_accept_final_order() {
    let mut session = AudioSegmenter::new(&fixture_config());
    feed(&mut session, 63, true);
    let first = session.next_request().unwrap();
    feed(&mut session, 63, true);
    let second = session.next_request().unwrap();
    assert!(first.is_final() && second.is_final());
    assert!(
        session
            .accept(&second, fixture_transcript("second 中文"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(session.num_completed(), 1);
    let events = session.accept(&first, fixture_transcript("first English")).unwrap();
    assert_eq!(events.iter().map(|event| event.segment).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(events[0].end_sample, events[1].start_sample);
    assert_eq!(session.num_completed(), 0);
}

fn fixture_config() -> StreamConfig {
    StreamConfig {
        sentence_silence_seconds: 0.25,
        window_seconds: 2.0,
        ..StreamConfig::default()
    }
}

fn feed(session: &mut AudioSegmenter, frames: usize, speech: bool) {
    for _ in 0..frames {
        session.feed(&[1000; 512], speech).unwrap();
    }
}

fn fixture_transcript(text: &str) -> Result<AudioSegmentResp> {
    Ok(AudioSegmentResp {
        text: text.into(),
        language: "Chinese,English".into(),
        spans: vec![],
        metrics: Default::default(),
        transport: Default::default(),
        language_ranges: None,
    })
}
