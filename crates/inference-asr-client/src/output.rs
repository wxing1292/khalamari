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

pub enum Output {
    Json,
    Text {
        tty: bool,
        draft_visible: bool,
        document: Document,
        pass: AnalysisPass,
    },
    Editor(Box<Editor>),
}

impl Output {
    pub fn open(config: &OutputConfig) -> Result<Self> {
        Ok(if let Some(path) = &config.editor {
            Self::Editor(Box::new(Editor::open(path)?))
        } else if config.json {
            Self::Json
        } else {
            Self::Text {
                tty: std::io::stdout().is_terminal(),
                draft_visible: false,
                document: Document::default(),
                pass: AnalysisPass::Live,
            }
        })
    }

    pub fn is_editor(&self) -> bool {
        matches!(self, Self::Editor(_))
    }

    pub fn emit(&mut self, event: &AudioEvent, command: &str) -> Result<()> {
        let (tty, draft_visible, document, pass) = match self {
            Self::Json => return write_json(&mut std::io::stdout().lock(), event),
            Self::Editor(editor) => {
                if editor.emit(event)? {
                    editor.render(command)?;
                }
                return Ok(());
            },
            Self::Text {
                tty,
                draft_visible,
                document,
                pass,
            } => {
                if *pass != event.pass {
                    *document = Document::default();
                    *pass = event.pass;
                }
                if !document.observe(event) {
                    return Ok(());
                }
                (tty, draft_visible, document, pass)
            },
        };
        let mut output = std::io::stdout().lock();
        if *tty {
            clear_draft(&mut output, draft_visible)?;
        }
        let draft = document.draft().filter(|draft| draft.segment == event.segment);
        let line = draft.unwrap_or_else(|| &document.lines()[event.segment as usize]);
        if !*tty || draft.is_none() {
            let prefix = line_prefix(event.kind, *pass, line);
            let text = line.display(if draft.is_some() { " | " } else { "\n" });
            writeln!(output, "{prefix}{text}").map_err(output_error)?;
        }
        if *tty {
            render_draft(&mut output, draft_visible, document, *pass)?;
        }
        output.flush().map_err(output_error)
    }

    pub fn input(&mut self, command: Option<String>, draft: &str) -> Result<bool> {
        match self {
            Self::Editor(editor) => editor.input(command.as_deref(), draft),
            _ => Ok(true),
        }
    }

    pub fn status(&mut self, status: &str, command: &str) -> Result<()> {
        if let Self::Editor(editor) = self {
            editor.status(status);
            return editor.render(command);
        }
        let mut output = std::io::stdout().lock();
        if let Self::Text {
            tty: true,
            draft_visible,
            ..
        } = self
        {
            clear_draft(&mut output, draft_visible)?;
        }
        output.flush().map_err(output_error)?;
        writeln!(std::io::stderr().lock(), "{}", display_text(status)).map_err(output_error)?;
        if let Self::Text {
            tty: true,
            draft_visible,
            document,
            pass,
        } = self
        {
            render_draft(&mut output, draft_visible, document, *pass)?;
        }
        output.flush().map_err(output_error)
    }

    pub fn replay(&mut self, path: &Path) -> Result<()> {
        let Self::Editor(editor) = self else {
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
        if let Self::Editor(editor) = self {
            editor.finish()?;
        } else if let Self::Text {
            tty: true,
            draft_visible,
            ..
        } = self
        {
            let mut output = std::io::stdout().lock();
            clear_draft(&mut output, draft_visible)?;
            output.flush().map_err(output_error)?;
        }
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if matches!(
            self,
            Self::Text {
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
