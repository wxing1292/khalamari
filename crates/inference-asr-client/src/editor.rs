use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::IsTerminal;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use inference_error::Error;
use inference_error::Result;
use unicode_width::UnicodeWidthChar;

use crate::document::Document;
use crate::document::Update;
use crate::output::write_json;
use crate::segment::AudioEvent;
use crate::terminal::display_text;
use crate::terminal::prefix;
use crate::terminal::size;
use crate::terminal::visible_suffix;

/// Owns the editable terminal view, append-only events, and the current final-text snapshot.
pub struct Editor {
    document: Document,
    events: BufWriter<File>,
    transcript: Option<BufWriter<File>>,
    directory: PathBuf,
    status: String,
    screen: bool,
}

impl Editor {
    pub fn open(directory: &Path) -> Result<Self> {
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            return Err(Error::invalid_argument(
                "--editor requires an interactive terminal on stdin and stdout",
            ));
        }
        std::fs::create_dir(directory).map_err(editor_error)?;
        let mut editor = Self {
            document: Document::default(),
            events: BufWriter::new(File::create_new(directory.join("events.jsonl")).map_err(editor_error)?),
            transcript: Some(BufWriter::new(
                File::create_new(directory.join("transcript.txt")).map_err(editor_error)?,
            )),
            directory: directory.to_owned(),
            status: "Enter: stop/save | :replace ID TEXT | :sub OLD => NEW | Ctrl-C: cancel".into(),
            screen: false,
        };
        std::io::stdout().write_all(b"\x1b[?1049h").map_err(editor_error)?;
        editor.screen = true;
        editor.render("")?;
        Ok(editor)
    }

    pub fn emit(&mut self, event: &AudioEvent) -> Result<()> {
        write_json(&mut self.events, event)?;
        self.document.observe(event);
        Ok(())
    }

    pub fn replay(&mut self, path: &Path) -> Result<()> {
        for (index, line) in BufReader::new(File::open(path).map_err(editor_error)?)
            .lines()
            .enumerate()
        {
            let line = line.map_err(editor_error)?;
            let update = serde_json::from_str(&line)
                .map_err(|e| Error::invalid_argument(format!("replay line {}: {e}", index + 1)))?;
            self.document.apply(update)?;
            writeln!(self.events, "{line}").map_err(editor_error)?;
        }
        self.events.flush().map_err(editor_error)?;
        self.status = "Replay loaded. :replace ID TEXT | :sub OLD => NEW | Enter: save and exit".into();
        self.render("")
    }

    pub fn command(&mut self, command: &str) -> Result<()> {
        if let Some(arguments) = command.strip_prefix(":replace ") {
            let (segment, text) = arguments.split_once(' ').unwrap_or((arguments, ""));
            let segment = segment
                .parse()
                .map_err(|_| Error::invalid_argument("use :replace ID TEXT with a finalized segment ID"))?;
            self.replace(segment, text.to_owned())?;
            self.status = format!("Updated segment {segment}. Original ASR text remains in events.jsonl.");
        } else if let Some(arguments) = command.strip_prefix(":sub ") {
            let (old, new) = arguments.split_once(" => ").ok_or_else(|| {
                Error::invalid_argument("use :sub OLD => NEW for an exact, case-sensitive replacement")
            })?;
            if old.is_empty() {
                return Err(Error::invalid_argument("replacement source must not be empty"));
            }
            let mut count = 0;
            for segment in 0..self.document.lines().len() {
                let line = &self.document.lines()[segment];
                if line.text.contains(old) {
                    let text = line.text.replace(old, new);
                    if text != line.text {
                        self.replace(segment as u64, text)?;
                        count += 1;
                    }
                }
            }
            self.status = format!("Updated {count} finalized segments. Future ASR results are unchanged.");
        } else {
            return Err(Error::invalid_argument(
                "use :replace ID TEXT, :sub OLD => NEW, or Enter to stop/save",
            ));
        }
        Ok(())
    }

    pub fn input(&mut self, command: Option<&str>, draft: &str) -> Result<bool> {
        let Some(command) = command else {
            self.prompt(draft)?;
            return Ok(false);
        };
        let stop = command.is_empty();
        if stop {
            self.status("Stopping capture and flushing final recognition.");
        } else {
            match self.command(command) {
                Ok(()) => {},
                Err(error @ Error::InvalidArgument(_)) => self.status(error.to_string()),
                Err(error) => return Err(error),
            }
        }
        self.render("")?;
        Ok(stop)
    }

    fn replace(&mut self, segment: u64, text: String) -> Result<()> {
        let update = Update::Correction(self.document.correction(segment, text)?);
        write_json(&mut self.events, &update)?;
        self.document.commit(update);
        Ok(())
    }

    pub fn status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub fn prompt(&self, command: &str) -> Result<()> {
        let (columns, rows) = size();
        let mut output = std::io::stdout().lock();
        write!(
            output,
            "\x1b[{rows};1H\x1b[2K> {}",
            visible_suffix(&display_text(command), columns.saturating_sub(3))
        )
        .map_err(editor_error)?;
        output.flush().map_err(editor_error)
    }

    pub fn render(&self, command: &str) -> Result<()> {
        let (columns, rows) = size();
        let mut output = std::io::stdout().lock();
        write!(output, "\x1b[H\x1b[2J").map_err(editor_error)?;
        let visible = rows.saturating_sub(4);
        let mut rendered = Vec::with_capacity(visible);
        for (line, kind) in self
            .document
            .lines()
            .iter()
            .map(|line| (line, "final"))
            .chain(self.document.draft().into_iter().map(|line| (line, "partial")))
            .rev()
        {
            let prefix = prefix(kind, line.segment, line.revision, line.start_sample, line.end_sample);
            let text = display_text(&line.text);
            let available = columns.saturating_sub(prefix.len() + 1);
            let wrapped = if available >= 2 {
                wrap(&text, available)
            } else {
                vec![""]
            };
            for text in wrapped.into_iter().rev().take(visible - rendered.len()) {
                rendered.push(format!("{}{text}", visible_suffix(&prefix, columns.saturating_sub(1))));
            }
            if rendered.len() == visible {
                break;
            }
        }
        for line in rendered.iter().rev() {
            write!(output, "{line}\r\n").map_err(editor_error)?;
        }
        write!(
            output,
            "\x1b[{};1H{}\r\n{}\r\n> {}",
            rows.saturating_sub(2).max(1),
            visible_suffix(
                &display_text(&self.directory.display().to_string()),
                columns.saturating_sub(1)
            ),
            visible_suffix(&display_text(&self.status), columns.saturating_sub(1)),
            visible_suffix(&display_text(command), columns.saturating_sub(3)),
        )
        .map_err(editor_error)?;
        output.flush().map_err(editor_error)
    }

    fn save(&mut self) -> Result<()> {
        if let Some(mut transcript) = self.transcript.take() {
            for line in self.document.lines() {
                writeln!(transcript, "{}", line.text).map_err(editor_error)?;
            }
            transcript.flush().map_err(editor_error)?;
        }
        self.events.flush().map_err(editor_error)
    }

    pub fn finish(&mut self) -> Result<()> {
        self.save()?;
        self.restore()?;
        eprintln!("Transcript saved in {}", self.directory.display());
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        if self.screen {
            std::io::stdout().write_all(b"\x1b[?1049l").map_err(editor_error)?;
            std::io::stdout().flush().map_err(editor_error)?;
            self.screen = false;
        }
        Ok(())
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        let _ = self.save();
        let _ = self.restore();
    }
}

fn wrap(text: &str, columns: usize) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut width = 0;
    for (index, c) in text.char_indices() {
        let extent = c.width().unwrap_or(0);
        if width + extent > columns && index > start {
            lines.push(&text[start..index]);
            start = index;
            width = 0;
        }
        width += extent;
    }
    lines.push(&text[start..]);
    lines
}

fn editor_error(error: std::io::Error) -> Error {
    Error::unavailable(format!("transcript editor: {error}"))
}
