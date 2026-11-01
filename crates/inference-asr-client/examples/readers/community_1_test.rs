use super::*;

#[test]
fn test_short_shared_activity_keeps_distinct_identities_without_embeddings() {
    let mut speakers = vec![
        Speaker {
            anchor: Vec::new(),
            recent: Vec::new(),
        },
        Speaker {
            anchor: Vec::new(),
            recent: Vec::new(),
        },
    ];
    let previous = vec![
        Span {
            start: 100,
            end: 110,
            speaker: 0,
        },
        Span {
            start: 100,
            end: 110,
            speaker: 1,
        },
    ];
    let mut spans = previous.clone();
    identify(&mut spans, &[Vec::new(), Vec::new()], &previous, &mut speakers, 0);
    assert_eq!(speakers.len(), 2);
    assert_eq!(spans.iter().map(|span| span.speaker).collect::<Vec<_>>(), [0, 1]);
}

#[test]
fn test_returning_voice_matches_anchor_or_recent_reference() {
    let mut speakers = vec![Speaker {
        anchor: vec![1.0, 0.0],
        recent: vec![0.0, 1.0],
    }];
    let mut spans = vec![Span {
        start: 20,
        end: 30,
        speaker: 0,
    }];
    identify(&mut spans, &[vec![0.0, 1.0]], &[], &mut speakers, 20);
    identify(&mut spans, &[vec![1.0, 0.0]], &[], &mut speakers, 20);
    assert_eq!(speakers.len(), 1);
    assert_eq!(spans[0].speaker, 0);
    assert_eq!(speakers[0].anchor, [1.0, 0.0]);
}

#[test]
fn test_sealed_only_activity_does_not_enroll_or_refresh_references() {
    let mut speakers = vec![Speaker {
        anchor: vec![1.0, 0.0],
        recent: vec![0.0, 1.0],
    }];
    let previous = vec![Span {
        start: 0,
        end: 10,
        speaker: 0,
    }];
    let mut spans = vec![
        Span {
            start: 0,
            end: 10,
            speaker: 0,
        },
        Span {
            start: 1,
            end: 4,
            speaker: 1,
        },
    ];
    identify(&mut spans, &[vec![-1.0, 0.0], Vec::new()], &previous, &mut speakers, 10);
    assert_eq!(speakers.len(), 1);
    assert_eq!(spans.len(), 1);
    assert_eq!(speakers[0].anchor, [1.0, 0.0]);
    assert_eq!(speakers[0].recent, [0.0, 1.0]);
}
