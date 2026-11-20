//! Text -> IPA phonemes using eSpeak NG (local, offline).
//!
//! The shared library is loaded at runtime (so the binary builds without it);
//! if it can't be found we fall back to the `espeak-ng` command-line tool.
//! Override paths with `VOX_ESPEAK_LIB` and `VOX_ESPEAK_DATA`.

use anyhow::{anyhow, bail, Context, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

type InitFn = unsafe extern "C" fn(c_int, c_int, *const c_char, c_int) -> c_int;
type SetVoiceFn = unsafe extern "C" fn(*const c_char) -> c_int;
type ToPhonemesFn = unsafe extern "C" fn(*mut *const c_void, c_int, c_int) -> *const c_char;

const AUDIO_OUTPUT_SYNCHRONOUS: c_int = 2;
const ESPEAK_CHARS_UTF8: c_int = 1;
const ESPEAK_PHONEMES_IPA: c_int = 0x02;

enum Inner {
    Lib {
        _lib: libloading::Library,
        set_voice: SetVoiceFn,
        to_phonemes: ToPhonemesFn,
        voice: String,
    },
    Cli(PathBuf),
}

pub struct Phonemizer {
    inner: Mutex<Inner>,
    pub source: String,
}

fn lib_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("VOX_ESPEAK_LIB") {
        v.push(PathBuf::from(p));
    }
    for p in [
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so.1",
        "/usr/lib64/libespeak-ng.so.1",
        "libespeak-ng.so.1",
        "libespeak-ng.dylib",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
        r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
        "libespeak-ng.dll",
    ] {
        v.push(PathBuf::from(p));
    }
    // bundled next to the executable / project
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            v.push(d.join("libespeak-ng.dylib"));
            v.push(d.join("libespeak-ng.so.1"));
            v.push(d.join("libespeak-ng.dll"));
        }
    }
    v.push(PathBuf::from("lib/espeak-ng/lib/libespeak-ng.dylib"));
    v.push(PathBuf::from("lib/espeak-ng/lib/libespeak-ng.so.1"));
    v
}

fn find_cli() -> Option<PathBuf> {
    for p in [
        "/opt/homebrew/bin/espeak-ng",
        "/usr/local/bin/espeak-ng",
        "/usr/bin/espeak-ng",
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    ] {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let ok = Command::new("espeak-ng").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    ok.then(|| PathBuf::from("espeak-ng"))
}

impl Phonemizer {
    pub fn new() -> Result<Self> {
        let data = std::env::var("VOX_ESPEAK_DATA").ok();
        let mut errors = Vec::new();
        for path in lib_candidates() {
            if path.is_absolute() && !path.exists() {
                continue;
            }
            match unsafe { Self::load_lib(&path, data.as_deref()) } {
                Ok(inner) => {
                    return Ok(Phonemizer { inner: Mutex::new(inner), source: format!("libespeak-ng ({})", path.display()) })
                }
                Err(e) if path.exists() => errors.push(format!("{}: {e}", path.display())),
                Err(_) => {}
            }
        }
        if let Some(cli) = find_cli() {
            return Ok(Phonemizer { source: format!("espeak-ng CLI ({})", cli.display()), inner: Mutex::new(Inner::Cli(cli)) });
        }
        bail!(
            "eSpeak NG not found (needed to turn text into phonemes, fully offline).\n  \
             macOS:  brew install espeak-ng\n  Debian/Ubuntu: sudo apt install espeak-ng\n  \
             Windows: install eSpeak NG from github.com/espeak-ng/espeak-ng/releases\n  \
             or set VOX_ESPEAK_LIB=/path/to/libespeak-ng.(dylib|so).\n  \
             Tip: <ipa>…</ipa> input works without eSpeak.{}",
            if errors.is_empty() { String::new() } else { format!("\n  tried: {}", errors.join("; ")) }
        )
    }

    unsafe fn load_lib(path: &PathBuf, data: Option<&str>) -> Result<Inner> {
        let lib = libloading::Library::new(path)?;
        let init: libloading::Symbol<InitFn> = lib.get(b"espeak_Initialize\0")?;
        let set_voice: SetVoiceFn = *lib.get::<SetVoiceFn>(b"espeak_SetVoiceByName\0")?;
        let to_phonemes: ToPhonemesFn = *lib.get::<ToPhonemesFn>(b"espeak_TextToPhonemes\0")?;
        let data_c = data.map(|d| CString::new(d).unwrap());
        let rate = init(
            AUDIO_OUTPUT_SYNCHRONOUS,
            0,
            data_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            0,
        );
        if rate <= 0 {
            bail!("espeak_Initialize failed (data directory missing? set VOX_ESPEAK_DATA)");
        }
        Ok(Inner::Lib { _lib: lib, set_voice, to_phonemes, voice: String::new() })
    }

    /// Phonemize `text` with the given eSpeak voice ("en-us", "fr-fr").
    /// Clause punctuation is preserved so the neural model gets prosody cues.
    pub fn phonemize(&self, text: &str, voice: &str) -> Result<String> {
        let mut parts = Vec::new();
        for (clause, punct) in split_clauses(text) {
            let ph = self.raw(&clause, voice)?;
            let ph = clean_phonemes(&ph);
            if ph.is_empty() {
                if let (Some(p), Some(last)) = (punct, parts.last_mut()) {
                    let last: &mut String = last;
                    last.push(p);
                }
                continue;
            }
            let mut s = ph;
            if let Some(p) = punct {
                s.push(p);
            }
            parts.push(s);
        }
        Ok(parts.join(" "))
    }

    fn raw(&self, text: &str, voice: &str) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }
        let mut inner = self.inner.lock().map_err(|_| anyhow!("phonemizer poisoned"))?;
        match &mut *inner {
            Inner::Lib { set_voice, to_phonemes, voice: cur, .. } => {
                if cur != voice {
                    let v = CString::new(voice)?;
                    let r = unsafe { set_voice(v.as_ptr()) };
                    if r != 0 {
                        bail!("eSpeak has no voice '{voice}'");
                    }
                    *cur = voice.to_string();
                }
                let c = CString::new(text.replace('\0', " "))?;
                let mut ptr: *const c_void = c.as_ptr() as *const c_void;
                let mut out = Vec::new();
                while !ptr.is_null() {
                    let res = unsafe { to_phonemes(&mut ptr, ESPEAK_CHARS_UTF8, ESPEAK_PHONEMES_IPA) };
                    if res.is_null() {
                        break;
                    }
                    let s = unsafe { CStr::from_ptr(res) }.to_string_lossy().trim().to_string();
                    if !s.is_empty() {
                        out.push(s);
                    }
                }
                Ok(out.join(" "))
            }
            Inner::Cli(bin) => {
                let o = Command::new(&*bin)
                    .args(["-q", "--ipa", "-v", voice, "--", text])
                    .output()
                    .with_context(|| format!("running {}", bin.display()))?;
                if !o.status.success() {
                    bail!("espeak-ng failed: {}", String::from_utf8_lossy(&o.stderr));
                }
                Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().collect::<Vec<_>>().join(" "))
            }
        }
    }
}

/// Split into clauses at punctuation, returning the (normalized) punctuation mark.
pub fn split_clauses(text: &str) -> Vec<(String, Option<char>)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
        let next_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
        let p = match c {
            '.' if prev_digit && next_digit => None, // 3.14
            ',' if prev_digit && next_digit => None, // 1,000 / 3,5
            '.' => {
                // collapse "..." into an ellipsis
                if i + 2 < chars.len() && chars[i + 1] == '.' && chars[i + 2] == '.' {
                    i += 2;
                    Some('…')
                } else {
                    Some('.')
                }
            }
            '!' | '?' | ',' | ';' | ':' | '…' => Some(c),
            '—' | '–' => Some(','),
            '¡' | '¿' => {
                i += 1;
                continue;
            }
            _ => None,
        };
        match p {
            Some(p) => {
                // "?!" -> keep the first mark only
                while i + 1 < chars.len() && "!?.".contains(chars[i + 1]) {
                    i += 1;
                }
                out.push((std::mem::take(&mut cur), Some(p)));
            }
            None => cur.push(c),
        }
        i += 1;
    }
    if !cur.trim().is_empty() {
        out.push((cur, None));
    }
    out
}

/// Remove eSpeak language-switch markers like "(en)" / "(fr)" and stray whitespace.
fn clean_phonemes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' {
            if let Some(j) = (i + 1..chars.len().min(i + 8)).find(|&j| chars[j] == ')') {
                if chars[i + 1..j].iter().all(|c| c.is_ascii_alphabetic() || *c == '-') {
                    i = j + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clauses() {
        let c = split_clauses("Hello, world... It costs 3.50 dollars!? Yes");
        let marks: Vec<_> = c.iter().map(|(_, p)| *p).collect();
        assert_eq!(marks, vec![Some(','), Some('…'), Some('!'), None]);
        assert!(c[2].0.contains("3.50"));
    }
    #[test]
    fn clean() {
        assert_eq!(clean_phonemes("lə (en)wˈiːkˈɛnd(fr) ɛ"), "lə wˈiːkˈɛnd ɛ");
    }
}
