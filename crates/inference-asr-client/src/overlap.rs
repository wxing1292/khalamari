use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

use crate::models::ALIGNMENT_STEP_SAMPLES;
use crate::models::AlignmentUnit;

static LEXEME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\x{3400}-\x{9fff}]|[A-Za-z0-9_]+(?:['’][A-Za-z0-9_]+)*|[^\W\d_]+").unwrap());

#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub alignment_score: f32,
    pub timing_status: &'static str,
    #[serde(skip)]
    char_start: usize,
}

pub fn timed_words(text: &str, units: &[AlignmentUnit], audio_start: u64) -> Option<Vec<Word>> {
    let mut cursor = 0;
    let mut previous_end = 0;
    let mut ranges = Vec::new();
    for unit in units {
        if !text[cursor..].starts_with(&unit.text) || unit.start_sample < previous_end {
            return None;
        }
        ranges.push((cursor..cursor + unit.text.len(), unit));
        cursor += unit.text.len();
        previous_end = unit.end_sample;
    }
    if cursor != text.len() {
        return None;
    }
    LEXEME
        .find_iter(text)
        .map(|word| {
            let relevant: Vec<_> = ranges
                .iter()
                .filter(|(range, _)| range.start < word.end() && range.end > word.start())
                .collect();
            let first = relevant.first()?.1;
            let last = relevant.last()?.1;
            Some(Word {
                text: word.as_str().to_owned(),
                char_start: word.start(),
                start_sample: audio_start + first.start_sample as u64,
                end_sample: audio_start + last.end_sample as u64,
                alignment_score: relevant.iter().map(|(_, unit)| unit.probability).fold(1.0, f32::min),
                timing_status: if first.start_sample < last.end_sample {
                    "estimated"
                } else {
                    "unresolved"
                },
            })
        })
        .collect()
}

fn same_interval(a: &Word, b: &Word) -> bool {
    a.start_sample < a.end_sample
        && b.start_sample < b.end_sample
        && (a.start_sample..a.end_sample).contains(&b.start_sample.midpoint(b.end_sample))
        && (b.start_sample..b.end_sample).contains(&a.start_sample.midpoint(a.end_sample))
}

fn same_word(a: &Word, b: &Word) -> bool {
    a.text
        .nfkc()
        .flat_map(char::to_lowercase)
        .eq(b.text.nfkc().flat_map(char::to_lowercase))
        && same_interval(a, b)
}

/// None is acoustic ambiguity. The caller must decode only the owned audio.
pub fn stitch(
    text: &str,
    words: &[Word],
    history: &[Word],
    start: u64,
    end: u64,
) -> Option<(String, Vec<Word>, usize)> {
    let first = words.first().map_or(start, |w| w.start_sample);
    let shared: Vec<_> = history.iter().filter(|w| w.end_sample > first).cloned().collect();
    if [&shared[..], words]
        .iter()
        .any(|sequence| sequence.windows(2).any(|pair| same_word(&pair[0], &pair[1])))
    {
        return None;
    }
    let skipped = (1..=shared.len().min(words.len()))
        .rfind(|&count| {
            shared[shared.len() - count..]
                .iter()
                .zip(&words[..count])
                .all(|(a, b)| same_word(a, b))
        })
        .unwrap_or(0);
    let remaining = &words[skipped..];
    // A boundary word confined to one DTW cell cannot establish ownership reliably.
    if remaining.iter().any(|w| {
        (w.start_sample < start && shared.iter().any(|old| same_interval(w, old)))
            || w.end_sample <= start
            || (w.end_sample - w.start_sample <= ALIGNMENT_STEP_SAMPLES
                && (w.start_sample.saturating_sub(start) <= ALIGNMENT_STEP_SAMPLES
                    || w.end_sample >= end.saturating_sub(ALIGNMENT_STEP_SAMPLES)))
    }) {
        return None;
    }
    let selected: Vec<_> = remaining.iter().take_while(|w| w.end_sample <= end).cloned().collect();
    if selected.is_empty() {
        return Some((String::new(), selected, skipped));
    }
    let lo = if skipped > 0 { selected[0].char_start } else { 0 };
    let hi = words
        .get(skipped + selected.len())
        .map_or(text.len(), |word| word.char_start);
    Some((text[lo..hi].trim().to_owned(), selected, skipped))
}

#[cfg(test)]
#[path = "overlap_test.rs"]
mod tests;
