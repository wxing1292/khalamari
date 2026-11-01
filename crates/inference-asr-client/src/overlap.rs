use unicode_normalization::UnicodeNormalization;

use crate::models::ALIGNMENT_STEP_SAMPLES;
use crate::transcript::Word;

fn same_interval(a: &Word, b: &Word) -> bool {
    a.start < a.end
        && b.start < b.end
        && (a.start..a.end).contains(&b.start.midpoint(b.end))
        && (b.start..b.end).contains(&a.start.midpoint(a.end))
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
    if words.is_empty() && !text.is_empty() {
        return None;
    }
    let first = words.first().map_or(start, |w| w.start);
    let shared: Vec<_> = history.iter().filter(|w| w.end > first).cloned().collect();
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
        (w.start < start && shared.iter().any(|old| same_interval(w, old)))
            || w.end <= start
            || (w.end - w.start <= ALIGNMENT_STEP_SAMPLES
                && (w.start.saturating_sub(start) <= ALIGNMENT_STEP_SAMPLES
                    || w.end >= end.saturating_sub(ALIGNMENT_STEP_SAMPLES)))
    }) {
        return None;
    }
    let mut selected: Vec<_> = remaining.iter().take_while(|w| w.end <= end).cloned().collect();
    if selected.is_empty() {
        return Some((String::new(), selected, skipped));
    }
    let lo = if skipped > 0 { selected[0].byte_start } else { 0 };
    let hi = words
        .get(skipped + selected.len())
        .map_or(text.len(), |word| word.byte_start);
    let raw = &text[lo..hi];
    let start = lo + raw.len() - raw.trim_start().len();
    for word in &mut selected {
        word.byte_start -= start;
    }
    Some((raw.trim().to_owned(), selected, skipped))
}

#[cfg(test)]
#[path = "overlap_test.rs"]
mod tests;
