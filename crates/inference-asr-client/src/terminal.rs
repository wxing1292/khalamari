use std::ffi::CStr;
use std::ffi::OsStr;
use std::fs::File;
use std::io::IsTerminal;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;

use inference_error::Error;
use inference_error::Result;
use tokio::io::unix::AsyncFd;
use unicode_width::UnicodeWidthChar;

/// Owns an independent terminal input descriptor and restores terminal input settings.
pub struct TerminalInput {
    input: AsyncFd<File>,
    original_terminal: Option<libc::termios>,
    line: Vec<u8>,
    escape: u8,
}

impl TerminalInput {
    pub fn new(enabled: bool, editing: bool) -> Result<Option<Self>> {
        if !enabled || !std::io::stdin().is_terminal() {
            return Ok(None);
        }
        let mut path = [0; libc::PATH_MAX as usize];
        let error = unsafe { libc::ttyname_r(libc::STDIN_FILENO, path.as_mut_ptr(), path.len()) };
        if error != 0 {
            return Err(input_error(std::io::Error::from_raw_os_error(error)));
        }
        // ttyname_r returned a terminated path. Reopening the device gives independent file status flags.
        // A dup would share O_NONBLOCK with stdin and, in some terminals, stdout.
        let path = unsafe { CStr::from_ptr(path.as_ptr()) };
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY)
            .open(OsStr::from_bytes(path.to_bytes()))
            .map_err(input_error)?;
        let descriptor = file.as_raw_fd();
        // The registered File remains owned by AsyncFd and is never replaced or closed early.
        let mut input = Self {
            input: unsafe { AsyncFd::register(file) }
                .map_err(|e| Error::unavailable(format!("terminal registration: {e}")))?,
            original_terminal: None,
            line: Vec::new(),
            escape: 0,
        };
        if editing {
            let mut terminal = std::mem::MaybeUninit::<libc::termios>::uninit();
            if unsafe { libc::tcgetattr(descriptor, terminal.as_mut_ptr()) } != 0 {
                return Err(input_error(std::io::Error::last_os_error()));
            }
            let original = unsafe { terminal.assume_init() };
            let mut terminal = original;
            terminal.c_lflag &= !(libc::ICANON | libc::ECHO);
            terminal.c_cc[libc::VMIN] = 1;
            terminal.c_cc[libc::VTIME] = 0;
            if unsafe { libc::tcsetattr(descriptor, libc::TCSANOW, &terminal) } != 0 {
                return Err(input_error(std::io::Error::last_os_error()));
            }
            input.original_terminal = Some(original);
        }
        Ok(Some(input))
    }

    pub fn line(&self) -> String {
        String::from_utf8_lossy(&self.line).into_owned()
    }

    /// Returns None after an edit, or the submitted line after Enter/EOF.
    pub async fn wait(&mut self) -> Result<Option<String>> {
        loop {
            let mut ready = self.input.readable().await.map_err(input_error)?;
            let mut byte = [0u8];
            match ready.try_io(|input| input.get_ref().read(&mut byte)) {
                Ok(Ok(0)) => return Ok(Some(String::new())),
                Ok(Ok(_)) if byte[0] == b'\n' || byte[0] == b'\r' => {
                    let line = String::from_utf8(std::mem::take(&mut self.line))
                        .map_err(|e| Error::invalid_argument(format!("terminal input must be UTF-8: {e}")))?;
                    return Ok(Some(line));
                },
                Ok(Ok(_)) if self.original_terminal.is_some() => {
                    if self.escape != 0 {
                        self.escape = match (self.escape, byte[0]) {
                            (1, b'[' | b'O') => 2,
                            (_, 0x40..=0x7e) => 0,
                            _ => self.escape,
                        };
                        continue;
                    }
                    match byte[0] {
                        8 | 127 => while self.line.pop().is_some_and(|byte| byte & 0xc0 == 0x80) {},
                        21 => self.line.clear(), // Ctrl-U clears the command.
                        27 => {
                            self.escape = 1;
                            continue;
                        },
                        4 if self.line.is_empty() => return Ok(Some(String::new())),
                        0..=31 => {},
                        byte => self.line.push(byte),
                    }
                    if let Err(error) = std::str::from_utf8(&self.line) {
                        if error.error_len().is_none() {
                            continue; // Keep receiving a split UTF-8 character before updating the prompt.
                        }
                        return Err(Error::invalid_argument(format!(
                            "terminal input must be UTF-8: {error}"
                        )));
                    }
                    return Ok(None);
                },
                Ok(Ok(_)) | Err(_) => {},
                Ok(Err(error)) => return Err(input_error(error)),
            }
        }
    }
}

impl Drop for TerminalInput {
    fn drop(&mut self) {
        unsafe {
            if let Some(terminal) = &self.original_terminal {
                libc::tcsetattr(self.input.get_ref().as_raw_fd(), libc::TCSANOW, terminal);
            }
        }
    }
}

fn input_error(error: std::io::Error) -> Error {
    Error::unavailable(format!("terminal input: {error}"))
}

pub fn size() -> (usize, usize) {
    let mut size = std::mem::MaybeUninit::<libc::winsize>::zeroed();
    // ioctl initializes winsize on success. stdout is borrowed for this call only.
    if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, size.as_mut_ptr()) } == 0 {
        let size = unsafe { size.assume_init() };
        (usize::from(size.ws_col).max(1), usize::from(size.ws_row).max(1))
    } else {
        (80, 24)
    }
}

pub fn visible_suffix(text: &str, available: usize) -> String {
    if text.chars().map(|c| c.width().unwrap_or(0)).sum::<usize>() <= available {
        return text.to_owned();
    }
    if available == 0 {
        return String::new();
    }
    let mut width = 1;
    let mut suffix = Vec::new();
    for c in text.chars().rev() {
        let extent = c.width().unwrap_or(0);
        if width + extent > available {
            break;
        }
        suffix.push(c);
        width += extent;
    }
    while suffix.last().is_some_and(|c| c.width() == Some(0)) {
        suffix.pop();
    }
    format!("…{}", suffix.into_iter().rev().collect::<String>())
}

pub fn display_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for part in text.split_whitespace() {
        if !result.is_empty() {
            result.push(' ');
        }
        result.extend(part.chars().filter(|c| !c.is_control()));
    }
    result
}

pub fn prefix(kind: &str, segment: u64, revision: u64, start_sample: u64, end_sample: u64) -> String {
    format!(
        "[{kind} {segment} r{revision} {:.2}-{:.2}s] ",
        start_sample as f64 / crate::audio::SAMPLE_RATE as f64,
        end_sample as f64 / crate::audio::SAMPLE_RATE as f64
    )
}
