use super::*;
use crate::diarization::SpeakerSpan;
use crate::recording::AudioRecordingWriter;

#[test]
fn test_replace_preserves_outside_activity_and_empty_replacement_clears_labels() {
    let mut timeline = AudioTimeline::new();
    timeline.apply(AudioAnalysisUpdate::Replace {
        range: 0..100,
        spans: vec![SpeakerSpan {
            start: 10,
            end: 90,
            speaker: 0,
        }],
    });
    timeline.apply(AudioAnalysisUpdate::Replace {
        range: 30..60,
        spans: vec![SpeakerSpan {
            start: 40,
            end: 60,
            speaker: 1,
        }],
    });
    timeline.apply(AudioAnalysisUpdate::Replace {
        range: 45..50,
        spans: vec![],
    });
    timeline.apply(AudioAnalysisUpdate::Seal { range: 0..60 });
    timeline.apply(AudioAnalysisUpdate::Replace {
        range: 80..100,
        spans: vec![SpeakerSpan {
            start: 85,
            end: 100,
            speaker: 2,
        }],
    });
    timeline.apply(AudioAnalysisUpdate::Seal { range: 60..100 });
    assert_eq!(timeline.sealed, 100);
    assert_eq!(
        timeline
            .spans
            .iter()
            .map(|s| (s.start, s.end, s.speaker))
            .collect::<Vec<_>>(),
        [(10, 30, 0), (40, 45, 1), (50, 60, 1), (60, 80, 0), (85, 100, 2)]
    );
}

#[tokio::test]
async fn test_reader_drains_coalesced_input_and_short_tail_before_stop() {
    let (mut writer, _spool) = AudioRecordingWriter::create().unwrap();
    let input = AudioAnalysisInput::new(writer.reader());
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (continue_tx, continue_rx) = std::sync::mpsc::channel();
    let model = AudioProcessor::load(move || Ok((input, Some((started_tx, continue_rx)))))
        .await
        .unwrap();
    writer.append(&vec![0; 1024]).unwrap();
    let mut task = AudioReaderTask::start(writer.reader(), model.clone(), |(input, barrier), range, stopped| {
        if let Some((started, resume)) = barrier.take() {
            started.send(()).unwrap();
            resume.recv().unwrap();
        }
        if !range.is_empty() {
            input.append(range.clone());
        }
        if stopped {
            input.stop();
        }
        Ok(vec![(range, stopped)])
    });
    started_rx.await.unwrap();
    writer.append(&vec![0; 2 * 40_000]).unwrap();
    writer.append(&[1, 0, 2, 0, 3, 0]).unwrap();
    writer.stop().unwrap();
    continue_tx.send(()).unwrap();
    let mut end = 0;
    let mut stops = 0;
    while let Some(updates) = task.next().await {
        for (range, stopped) in updates.unwrap() {
            assert_eq!(range.start, end);
            end = range.end;
            stops += usize::from(stopped);
        }
    }
    assert_eq!((end, stops), (40_515, 1));
    model
        .process(|(input, _)| {
            assert!(input.stopped);
            assert_eq!(input.recording.read(input.end - 3..input.end)?, [1, 2, 3]);
            Ok(())
        })
        .await
        .unwrap();
}
