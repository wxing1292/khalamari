use super::*;

#[test]
fn test_single_speaker_turn_does_not_require_word_timing() {
    let words = ["一", "hello", "中文"]
        .into_iter()
        .zip([(0, 0, 0), (0, 16_000, 4), (16_000, 32_000, 10)])
        .map(|(text, (start, end, byte_start))| {
            Word {
                text: text.into(),
                start,
                end,
                alignment_score: 1.0,
                timing_status: if start == end { "unresolved" } else { "estimated" },
                speakers: Vec::new(),
                byte_start,
            }
        })
        .collect::<Vec<_>>();
    let span = Span {
        start: 0,
        end: 32_000,
        language_hint: String::new(),
        language: String::new(),
        chinese_audio_retry: false,
        text: "一 hello 中文".into(),
        words,
        speaker_activity: Vec::new(),
        overlap: None,
    };
    let speakers = [SpeakerSpan {
        start: 4_000,
        end: 32_000,
        speaker: 0,
    }];
    let mut transcript = AudioTranscript::from_spans(vec![span.clone()]);
    assert!(transcript.attribute(&speakers));
    assert_eq!(transcript.turns.len(), 1);
    assert_eq!(transcript.turns[0].speakers, [0]);
    assert_eq!(transcript.turns[0].text, transcript.text);
    assert_eq!((transcript.turns[0].start, transcript.turns[0].end), (0, 32_000));
    // Turn-level attribution must not invent a timestamp or speaker for an unaligned word.
    let word = &transcript.spans[0].words[0];
    assert_eq!((word.start, word.end, word.timing_status), (0, 0, "unresolved"));
    assert!(word.speakers.is_empty());

    let mut overlapping = speakers.to_vec();
    overlapping.push(SpeakerSpan {
        start: 16_000,
        end: 32_000,
        speaker: 1,
    });
    assert!(transcript.attribute(&overlapping));
    assert_eq!(
        transcript
            .turns
            .iter()
            .map(|turn| turn.speakers.clone())
            .collect::<Vec<_>>(),
        [vec![0], vec![0, 1]]
    );
    assert_eq!(
        transcript
            .turns
            .iter()
            .map(|turn| turn.text.as_str())
            .collect::<String>(),
        transcript.text
    );

    let mut unaligned = span;
    unaligned.words.clear();
    let mut transcript = AudioTranscript::from_spans(vec![unaligned]);
    assert!(transcript.attribute(&speakers));
    assert_eq!(transcript.turns.len(), 1);
    assert_eq!(transcript.turns[0].speakers, [0]);
}

#[test]
fn test_word_attribution_uses_union_coverage_and_retains_ties() {
    let mut transcript = AudioTranscript::from_spans(vec![Span {
        start: 0,
        end: 16_000,
        language_hint: String::new(),
        language: String::new(),
        chinese_audio_retry: false,
        text: "hello".into(),
        words: vec![Word {
            text: "hello".into(),
            start: 0,
            end: 16_000,
            alignment_score: 1.0,
            timing_status: "estimated",
            speakers: Vec::new(),
            byte_start: 0,
        }],
        speaker_activity: Vec::new(),
        overlap: None,
    }]);
    let mut speakers = vec![
        SpeakerSpan {
            start: 0,
            end: 12_000,
            speaker: 0,
        },
        SpeakerSpan {
            start: 0,
            end: 16_000,
            speaker: 1,
        },
        SpeakerSpan {
            start: 8_000,
            end: 16_000,
            speaker: 0,
        },
    ];
    assert!(transcript.attribute(&speakers));
    assert_eq!(transcript.spans[0].words[0].speakers, [0, 1]);

    // A brief secondary activity at the midpoint must not outweigh a whole-word speaker.
    speakers[1].start = 7_000;
    speakers[1].end = 9_000;
    assert!(transcript.attribute(&speakers));
    assert_eq!(transcript.spans[0].words[0].speakers, [0]);
    assert_eq!(transcript.turns[0].speakers, [0]);
    assert_eq!(transcript.spans[0].speaker_activity, speakers);

    // An exclusive short turn remains attributable to its speaker.
    speakers = vec![SpeakerSpan {
        start: 7_000,
        end: 9_000,
        speaker: 1,
    }];
    assert!(transcript.attribute(&speakers));
    assert_eq!(transcript.spans[0].words[0].speakers, [1]);
}

#[test]
fn test_unaligned_text_does_not_bridge_different_speakers_or_conflicting_activity() {
    let mut span = Span {
        start: 0,
        end: 32_000,
        language_hint: String::new(),
        language: String::new(),
        chinese_audio_retry: false,
        text: "yes嗯no".into(),
        words: [
            ("yes", 0, 16_000, 0, vec![0]),
            ("嗯", 16_000, 16_000, 3, vec![]),
            ("no", 16_000, 32_000, 6, vec![1]),
        ]
        .into_iter()
        .map(|(text, start, end, byte_start, speakers)| {
            Word {
                text: text.into(),
                start,
                end,
                byte_start,
                speakers,
                alignment_score: 1.0,
                timing_status: if start == end { "unresolved" } else { "estimated" },
            }
        })
        .collect(),
        speaker_activity: vec![
            SpeakerSpan {
                start: 0,
                end: 16_000,
                speaker: 0,
            },
            SpeakerSpan {
                start: 16_000,
                end: 32_000,
                speaker: 1,
            },
        ],
        overlap: None,
    };
    let turns = SpeakerTurn::from_span(&span);
    assert_eq!(
        turns.iter().map(|turn| turn.speakers.clone()).collect::<Vec<_>>(),
        [vec![0], vec![], vec![1]]
    );

    // Neighboring words agree, but another voice is active inside the unaligned range.
    span.words[2].speakers = vec![0];
    span.speaker_activity = vec![
        SpeakerSpan {
            start: 0,
            end: 32_000,
            speaker: 0,
        },
        SpeakerSpan {
            start: 15_900,
            end: 16_100,
            speaker: 1,
        },
    ];
    let turns = SpeakerTurn::from_span(&span);
    assert_eq!(
        turns.iter().map(|turn| turn.speakers.clone()).collect::<Vec<_>>(),
        [vec![0], vec![], vec![0]]
    );

    span.speaker_activity.truncate(1);
    let turns = SpeakerTurn::from_span(&span);
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].text, span.text);
}

#[test]
fn test_speaker_replacement_revises_all_spans_and_ignores_unchanged_labels() {
    let span = |start, text: &str| {
        Span {
            start,
            end: start + 16_000,
            language_hint: String::new(),
            language: String::new(),
            chinese_audio_retry: false,
            text: text.into(),
            words: vec![Word {
                text: text.into(),
                start,
                end: start + 16_000,
                alignment_score: 1.0,
                timing_status: "estimated",
                speakers: Vec::new(),
                byte_start: 0,
            }],
            speaker_activity: Vec::new(),
            overlap: None,
        }
    };
    let mut transcript = AudioTranscript::from_spans(vec![span(0, "hello"), span(16_000, "中文")]);
    let speakers = [
        SpeakerSpan {
            start: 0,
            end: 16_000,
            speaker: 0,
        },
        SpeakerSpan {
            start: 16_000,
            end: 48_000,
            speaker: 1,
        },
    ];
    assert!(transcript.attribute(&speakers));
    assert_eq!(
        transcript
            .turns
            .iter()
            .map(|turn| turn.speakers.clone())
            .collect::<Vec<_>>(),
        [vec![0], vec![1]]
    );
    assert!(!transcript.attribute(&speakers));

    // The diarizer can revise future audio without changing this transcript's owned ranges.
    let mut future = speakers;
    future[1].end = 64_000;
    assert!(!transcript.attribute(&future));

    assert!(transcript.attribute(&[]));
    assert!(transcript.turns.iter().all(|turn| turn.speakers.is_empty()));
    assert!(
        transcript
            .spans
            .iter()
            .flat_map(|span| &span.words)
            .all(|word| word.speakers.is_empty())
    );
    assert_eq!(transcript.text, "hello 中文");
    assert!(!transcript.attribute(&[]));

    let mut same_speaker = future;
    same_speaker[1].speaker = 0;
    assert!(transcript.attribute(&same_speaker));
    assert_eq!(transcript.turns.len(), 1);
    assert_eq!(transcript.turns[0].text, "hello 中文");
}
