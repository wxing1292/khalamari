use std::io::IsTerminal;
use std::io::Write;
use std::path::Path;

use inference_error::Error;
use inference_error::Result;
use serde::Serialize;

use crate::config::OutputConfig;
use crate::editor::Editor;
use crate::segment::AudioEvent;
use crate::service::AudioTranscribeResp;
use crate::terminal::TerminalInput;
use crate::terminal::display_text;
use crate::terminal::prefix;
use crate::terminal::size;
use crate::terminal::visible_suffix;

pub enum Output {
    Json,
    Text { tty: bool, partial: Option<(u64, String)> },
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
                partial: None,
            }
        })
    }

    pub fn is_editor(&self) -> bool {
        matches!(self, Self::Editor(_))
    }

    pub fn collected(&self, recognition: &AudioTranscribeResp) -> Result<()> {
        let mut output = std::io::stdout().lock();
        match self {
            Self::Json => write_json(&mut output, recognition),
            Self::Text { .. } => {
                writeln!(output, "{}", recognition.text).map_err(output_error)?;
                output.flush().map_err(output_error)
            },
            Self::Editor(_) => unreachable!("editor configuration requires streaming"),
        }
    }

    pub fn emit(&mut self, event: &AudioEvent, command: &str) -> Result<()> {
        let (tty, partial) = match self {
            Self::Json => return write_json(&mut std::io::stdout().lock(), event),
            Self::Editor(editor) => {
                editor.emit(event)?;
                return editor.render(command);
            },
            Self::Text { tty, partial } => (tty, partial),
        };
        let mut output = std::io::stdout().lock();
        let prefix = prefix(
            event.kind,
            event.segment,
            event.revision,
            event.start_sample,
            event.end_sample,
        );
        let text = display_text(&event.transcript.text);
        if *tty {
            let draft = event.kind == "partial";
            let text = if draft {
                visible_suffix(&text, size().0.saturating_sub(prefix.len() + 2))
            } else {
                text
            };
            write!(output, "\r\x1b[2K{prefix}{text}{}", if draft { "" } else { "\n" }).map_err(output_error)?;
            if draft {
                *partial = Some((event.segment, format!("{prefix}{text}")));
            } else if let Some((segment, draft)) = partial.take()
                && segment > event.segment
            {
                write!(output, "{draft}").map_err(output_error)?;
                *partial = Some((segment, draft));
            }
        } else {
            writeln!(output, "{prefix}{text}").map_err(output_error)?;
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
            editor.render(command)?;
        }
        Ok(())
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
        }
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if matches!(self, Self::Text { partial: Some(_), .. }) {
            let _ = writeln!(std::io::stdout());
        }
    }
}

pub fn write_json(output: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *output, value).map_err(|e| Error::unavailable(format!("transcript output: {e}")))?;
    writeln!(output).map_err(output_error)?;
    output.flush().map_err(output_error)
}

fn output_error(error: std::io::Error) -> Error {
    Error::unavailable(format!("transcript output: {error}"))
}
