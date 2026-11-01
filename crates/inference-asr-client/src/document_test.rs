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
