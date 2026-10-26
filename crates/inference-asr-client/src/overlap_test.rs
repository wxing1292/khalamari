use super::*;

#[test]
fn test_stitch_ownership() {
    for base in [0, u64::MAX - 5000] {
        let history = vec![fixture_word("no", base + 1000, base + 2000, 0)];
        let words = vec![
            fixture_word("no", base + 1050, base + 2050, 0),
            fixture_word("no", base + 3000, base + 4000, 4),
        ];
        let (text, retained, skipped) = stitch("no, no.", &words, &history, base + 2500, base + 4500).unwrap();
        assert_eq!(text, "no.");
        assert_eq!(skipped, 1);
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].start_sample, base + 3000);
        let words = vec![fixture_word("a", base + 1000, base + 1320, 0)];
        assert_eq!(stitch("a", &words, &[], base, base + 3000).unwrap().0, "a");
    }
}

#[test]
fn test_stitch_unresolved() {
    for base in [0, u64::MAX - 500_000] {
        for (text, words, range) in [
            ("no, no", vec![("no", 1000, 2000, 0), ("no", 1000, 2000, 4)], 0..3000),
            ("缓存", vec![("缓存", 0, 0, 0)], 0..100),
            ("缓存", vec![("缓存", 10, 20, 0)], 0..100),
            (
                "Please open",
                vec![("Please", 245760, 246080, 0), ("open", 246080, 251520, 7)],
                128000..238720,
            ),
        ] {
            let words: Vec<_> = words
                .into_iter()
                .map(|(text, start, end, lo)| fixture_word(text, base + start, base + end, lo))
                .collect();
            assert!(
                stitch(text, &words, &[], base + range.start, base + range.end).is_none(),
                "{text}"
            );
        }
    }
}

#[test]
fn test_timed_words_utf8() {
    let units = vec![AlignmentUnit {
        text: "中文 cache".into(),
        start_sample: 0,
        end_sample: 1000,
        probability: 0.9,
    }];
    let words = timed_words("中文 cache", &units, 5000).unwrap();
    assert_eq!(
        words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
        ["中", "文", "cache"]
    );
    assert_eq!(stitch("中文 cache", &words, &[], 4000, 7000).unwrap().0, "中文 cache");
    assert!(timed_words("中文 cache extra", &units, 0).is_none());
}

fn fixture_word(text: &str, start: u64, end: u64, lo: usize) -> Word {
    Word {
        text: text.into(),
        start_sample: start,
        end_sample: end,
        char_start: lo,
        alignment_score: 0.9,
        timing_status: if start == end { "unresolved" } else { "estimated" },
    }
}
