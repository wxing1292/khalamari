use std::fs::File;
use std::io::BufWriter;
use std::io::IsTerminal;
use std::io::Write;
use std::path::Path;

use inference_error::Error;
use inference_error::Result;
use serde::Serialize;
use unicode_width::UnicodeWidthStr;

use crate::config::OutputConfig;
use crate::document::Document;
use crate::document::Line;
use crate::editor::Editor;
use crate::segment::AnalysisPass;
use crate::segment::AudioEvent;
use crate::terminal::TerminalInput;
use crate::terminal::display_text;
use crate::terminal::prefix;
use crate::terminal::size;
use crate::terminal::visible_suffix;

/// The current pass owns one raw document for both rendering and saved text.
/// The editor separately owns explicit user corrections to the live document.
pub struct Output {
    view: View,
    document: Document,
    pass: AnalysisPass,
    transcript: Option<BufWriter<File>>,
}

enum View {
    Json,
    Text { tty: bool, draft_visible: bool },
    Editor(Box<Editor>),
}

impl Output {
    pub fn open(config: &OutputConfig) -> Result<Self> {
        let view = if let Some(path) = &config.editor {
            View::Editor(Box::new(Editor::open(path)?))
        } else if config.json {
            View::Json
        } else {
            View::Text {
                tty: std::io::stdout().is_terminal(),
                draft_visible: false,
            }
        };
        Ok(Self {
            view,
            document: Document::default(),
            pass: AnalysisPass::Live,
            transcript: None,
        })
    }

    pub fn start_pass(&mut self, pass: AnalysisPass, path: &Path) -> Result<()> {
        assert!(
            self.transcript.is_none(),
            "analysis passes must close before another starts"
        );
        self.pass = pass;
        self.document = Document::default();
        self.transcript = Some(BufWriter::new(File::create_new(path).map_err(output_error)?));
        Ok(())
    }

    pub fn stop_pass(&mut self) -> Result<()> {
        if let Some(mut transcript) = self.transcript.take() {
            self.document.write(&mut transcript)?;
            transcript.flush().map_err(output_error)?;
        }
        Ok(())
    }

    pub fn is_editor(&self) -> bool {
        matches!(self.view, View::Editor(_))
    }

    pub fn emit(&mut self, event: &AudioEvent, command: &str) -> Result<()> {
        let changed = self.document.observe(event);
        let (tty, draft_visible) = match &mut self.view {
            View::Json => return write_json(&mut std::io::stdout().lock(), event),
            View::Editor(editor) => {
                if event.pass == AnalysisPass::Live && editor.emit(event)? {
                    editor.render(command)?;
                }
                return Ok(());
            },
            View::Text { tty, draft_visible } => (tty, draft_visible),
        };
        if !changed {
            return Ok(());
        }
        let mut output = std::io::stdout().lock();
        if *tty {
            clear_draft(&mut output, draft_visible)?;
        }
        let draft = self.document.draft().filter(|draft| draft.segment == event.segment);
        let line = draft.unwrap_or_else(|| &self.document.lines()[event.segment as usize]);
        if !*tty || draft.is_none() {
            let prefix = line_prefix(event.kind, self.pass, line);
            let text = line.display(if draft.is_some() { " | " } else { "\n" });
            writeln!(output, "{prefix}{text}").map_err(output_error)?;
        }
        if *tty {
            render_draft(&mut output, draft_visible, &self.document, self.pass)?;
        }
        output.flush().map_err(output_error)
    }

    pub fn input(&mut self, command: Option<String>, draft: &str) -> Result<bool> {
        match &mut self.view {
            View::Editor(editor) => editor.input(command.as_deref(), draft),
            _ => Ok(true),
        }
    }

    pub fn status(&mut self, status: &str, command: &str) -> Result<()> {
        if let View::Editor(editor) = &mut self.view {
            editor.status(status);
            return editor.render(command);
        }
        let mut output = std::io::stdout().lock();
        if let View::Text {
            tty: true,
            draft_visible,
            ..
        } = &mut self.view
        {
            clear_draft(&mut output, draft_visible)?;
        }
        output.flush().map_err(output_error)?;
        writeln!(std::io::stderr().lock(), "{}", display_text(status)).map_err(output_error)?;
        if let View::Text {
            tty: true,
            draft_visible,
        } = &mut self.view
        {
            render_draft(&mut output, draft_visible, &self.document, self.pass)?;
        }
        output.flush().map_err(output_error)
    }

    pub fn replay(&mut self, path: &Path) -> Result<()> {
        let View::Editor(editor) = &mut self.view else {
            unreachable!("replay configuration requires an editor")
        };
        editor.replay(path)
    }

    pub async fn edit(&mut self, input: &mut TerminalInput) -> Result<()> {
        loop {
            let command = input.wait().await?;
            if self.input(command, &input.line())? {
                return Ok(());
            }
        }
    }

    pub fn finish(&mut self) -> Result<()> {
        let saved = self.stop_pass();
        if let View::Editor(editor) = &mut self.view {
            editor.finish()?;
        } else if let View::Text {
            tty: true,
            draft_visible,
            ..
        } = &mut self.view
        {
            let mut output = std::io::stdout().lock();
            clear_draft(&mut output, draft_visible)?;
            output.flush().map_err(output_error)?;
        }
        saved
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if matches!(
            self.view,
            View::Text {
                draft_visible: true,
                ..
            }
        ) {
            let _ = writeln!(std::io::stdout());
        }
    }
}

fn line_prefix(kind: &str, pass: AnalysisPass, line: &Line) -> String {
    let mut prefix = prefix(kind, line.segment, line.revision, line.start, line.end);
    if pass == AnalysisPass::Offline {
        prefix.insert_str(0, "[offline] ");
    }
    prefix
}

fn clear_draft(output: &mut impl Write, visible: &mut bool) -> Result<()> {
    if std::mem::take(visible) {
        write!(output, "\r\x1b[2K").map_err(output_error)?;
    }
    Ok(())
}

fn render_draft(output: &mut impl Write, visible: &mut bool, document: &Document, pass: AnalysisPass) -> Result<()> {
    if let Some(line) = document.draft() {
        // Keep the complete render inside one physical row, including its prefix and wide Unicode text.
        let available = size().0.saturating_sub(1);
        let mut prefix = line_prefix("partial", pass, line);
        // Prefer transcript text when metadata would leave no room for an ellipsis and one wide character.
        if prefix.width() + 3 > available {
            prefix.clear();
        }
        let text = visible_suffix(&line.display(" | "), available.saturating_sub(prefix.width()));
        write!(output, "\r\x1b[2K{prefix}{text}").map_err(output_error)?;
        *visible = true;
    }
    Ok(())
}

pub fn write_json(output: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *output, value).map_err(|e| Error::unavailable(format!("transcript output: {e}")))?;
    writeln!(output).map_err(output_error)?;
    output.flush().map_err(output_error)
}

fn output_error(error: std::io::Error) -> Error {
    Error::unavailable(format!("transcript output: {error}"))
}
