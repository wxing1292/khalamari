use super::*;

#[test]
fn test_speaker_revision_updates_final_turns_without_overwriting_a_manual_correction() {
    let line = |speaker| {
        Line {
            segment: 0,
            revision: 1,
            start: 0,
            end: 16_000,
            sample_rate: SAMPLE_RATE,
            text: "hello 中文".into(),
            turns: vec![SpeakerTurn {
                start: 0,
                end: 16_000,
                speakers: vec![speaker],
                text: "hello 中文".into(),
            }],
        }
    };
    let mut document = Document::default();
    document.apply(Update::Final(line(0))).unwrap();
    let update = serde_json::to_string(&Update::Speakers(line(1))).unwrap();
    document.apply(serde_json::from_str(&update).unwrap()).unwrap();
    assert_eq!(document.lines()[0].turns[0].speakers, [1]);
    assert!(!document.commit(Update::Speakers(line(1))));
    assert!(document.commit(Update::Speakers(line(0))));
    assert!(document.commit(Update::Speakers(line(1))));
    let correction = document.correction(0, "corrected 中文".into()).unwrap();
    document.apply(Update::Correction(correction)).unwrap();
    assert!(!document.commit(Update::Speakers(line(2))));
    assert_eq!(document.lines()[0].text, "corrected 中文");
    assert_eq!(document.lines()[0].turns[0].speakers, [1]);
    assert_eq!(document.lines()[0].revision, 2);
}

#[test]
fn test_speaker_update_matches_the_displayed_draft_revision() {
    let line = |revision, speaker| {
        Line {
            segment: 0,
            revision,
            start: 0,
            end: 16_000,
            sample_rate: SAMPLE_RATE,
            text: "hello 中文".into(),
            turns: vec![SpeakerTurn {
                start: 0,
                end: 16_000,
                speakers: vec![speaker],
                text: "hello 中文".into(),
            }],
        }
    };
    let mut document = Document::default();
    document.apply(Update::Partial(line(1, 0))).unwrap();
    document.apply(Update::Speakers(line(1, 1))).unwrap();
    assert_eq!(document.draft().unwrap().turns[0].speakers, [1]);
    document.apply(Update::Partial(line(2, 0))).unwrap();
    assert!(!document.commit(Update::Speakers(line(1, 1))));
    assert_eq!(document.draft().unwrap().turns[0].speakers, [0]);
    document.apply(Update::Final(line(3, 0))).unwrap();
    assert!(document.draft().is_none());
    assert!(!document.commit(Update::Speakers(line(2, 1))));
    assert_eq!(document.lines()[0].turns[0].speakers, [0]);
}

#[test]
fn test_replay_ignores_superseded_speaker_updates() {
    let line = |segment, revision, start, end| {
        Line {
            segment,
            revision,
            start,
            end,
            sample_rate: SAMPLE_RATE,
            text: "hello 中文".into(),
            turns: Vec::new(),
        }
    };
    let mut document = Document::default();
    document.apply(Update::Partial(line(0, 1, 0, 16_000))).unwrap();
    document.apply(Update::Partial(line(0, 2, 0, 24_000))).unwrap();
    document.apply(Update::Speakers(line(0, 1, 0, 16_000))).unwrap();
    assert_eq!(document.draft().unwrap().end, 24_000);
    document.apply(Update::Partial(line(1, 1, 32_000, 48_000))).unwrap();
    document.apply(Update::Speakers(line(0, 2, 0, 24_000))).unwrap();
    assert_eq!(document.draft().unwrap().segment, 1);
    document.apply(Update::Final(line(0, 3, 0, 24_000))).unwrap();
    assert!(document.apply(Update::Speakers(line(1, 2, 32_000, 48_000))).is_err());
    assert!(document.apply(Update::Speakers(line(1, 1, 32_000, 49_000))).is_err());
    document.apply(Update::Speakers(line(0, 2, 0, 16_000))).unwrap();
    document.apply(Update::Final(line(1, 2, 32_000, 48_000))).unwrap();
    assert_eq!(document.lines().len(), 2);
}
