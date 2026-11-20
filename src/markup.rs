//! Inline markup parser.
//!
//! Supported syntax (can be mixed freely):
//!
//! ```text
//! [happy] text            switch mood until the next mood tag   ([happy 0.5], [happy:50%])
//! [happy + whisper 0.4]   mix moods
//! [/]  or [neutral]       back to the persona's default mood
//! [pause 400ms] [pause 1s] [pause]
//! [breath]                audible inhale
//! [laughs] [sighs]        non-verbal sounds
//! [voice thomas]          switch persona (dialogue)     also: [persona x], [speaker x]
//! *word*                  emphasis
//! <mood name="sad" intensity="0.7"> ... </mood>   scoped mood   (or <sad> ... </sad>)
//! <emph> ... </emph>
//! <prosody rate="1.2" pitch="+2st" volume="+3dB"> ... </prosody>
//! <voice name="pierre"> ... </voice>
//! <break time="500ms"/>
//! <ipa>həlˈoʊ</ipa>       raw IPA phonemes
//! \[ \* \<                escapes
//! ```

use crate::util::{parse_db, parse_duration_ms, parse_rate, parse_semitones};
use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub text: String,
    pub ipa: bool,
    pub persona: Option<String>,
    /// Mood mix; empty = persona/default mood.
    pub moods: Vec<(String, f32)>,
    pub rate: f32,
    pub pitch_st: f32,
    pub volume_db: f32,
    pub emphasis: bool,
    /// Non-verbal sound ("laugh", "sigh"); `text` then holds an IPA fallback.
    pub nonverbal: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Speech(Span),
    Pause { ms: f32 },
    Breath,
}

#[derive(Clone, Debug)]
struct State {
    persona: Option<String>,
    moods: Vec<(String, f32)>,
    rate: f32,
    pitch_st: f32,
    volume_db: f32,
    emphasis: bool,
    ipa: bool,
    tag: String,
}

impl Default for State {
    fn default() -> Self {
        State {
            persona: None,
            moods: vec![],
            rate: 1.0,
            pitch_st: 0.0,
            volume_db: 0.0,
            emphasis: false,
            ipa: false,
            tag: String::new(),
        }
    }
}

pub struct Parser<'a> {
    is_mood: &'a dyn Fn(&str) -> bool,
}

impl<'a> Parser<'a> {
    pub fn new(is_mood: &'a dyn Fn(&str) -> bool) -> Self {
        Parser { is_mood }
    }

    pub fn parse(&self, input: &str) -> Result<Vec<Item>> {
        let chars: Vec<char> = input.chars().collect();
        let mut out = Vec::new();
        let mut stack = vec![State::default()];
        let mut buf = String::new();
        let mut i = 0;

        macro_rules! flush {
            () => {
                if !buf.trim().is_empty() {
                    let s = stack.last().unwrap();
                    out.push(Item::Speech(Span {
                        text: std::mem::take(&mut buf),
                        ipa: s.ipa,
                        persona: s.persona.clone(),
                        moods: s.moods.clone(),
                        rate: s.rate,
                        pitch_st: s.pitch_st,
                        volume_db: s.volume_db,
                        emphasis: s.emphasis,
                        nonverbal: None,
                    }));
                } else if !buf.is_empty() {
                    // keep whitespace between spans so words don't glue together
                    if let Some(Item::Speech(sp)) = out.last_mut() {
                        if !sp.text.ends_with(char::is_whitespace) {
                            sp.text.push(' ');
                        }
                    }
                    buf.clear();
                }
            };
        }

        while i < chars.len() {
            let c = chars[i];
            match c {
                '\\' if i + 1 < chars.len() => {
                    buf.push(chars[i + 1]);
                    i += 2;
                }
                '[' => {
                    let Some(end) = find(&chars, i + 1, ']') else {
                        buf.push(c);
                        i += 1;
                        continue;
                    };
                    let cmd: String = chars[i + 1..end].iter().collect();
                    flush!();
                    self.bracket(cmd.trim(), &mut stack, &mut out)?;
                    i = end + 1;
                }
                '<' => {
                    let Some(end) = find(&chars, i + 1, '>') else {
                        buf.push(c);
                        i += 1;
                        continue;
                    };
                    let tag: String = chars[i + 1..end].iter().collect();
                    flush!();
                    self.angle(tag.trim(), &mut stack, &mut out)?;
                    i = end + 1;
                }
                '*' if !stack.last().unwrap().ipa => {
                    flush!();
                    let top = stack.last_mut().unwrap();
                    top.emphasis = !top.emphasis;
                    i += 1;
                }
                _ => {
                    buf.push(c);
                    i += 1;
                }
            }
        }
        flush!();
        Ok(out)
    }

    fn parse_mood_mix(&self, cmd: &str) -> Result<Option<Vec<(String, f32)>>> {
        let mut mix = Vec::new();
        for part in cmd.split('+') {
            let part = part.trim().replace(':', " ").replace('=', " ");
            let mut it = part.split_whitespace();
            let Some(name) = it.next() else { continue };
            let name = name.to_ascii_lowercase();
            if !(self.is_mood)(&name) {
                return Ok(None);
            }
            let k = match it.next() {
                Some(v) => parse_intensity(v)?,
                None => 1.0,
            };
            mix.push((name, k));
        }
        Ok(if mix.is_empty() { None } else { Some(mix) })
    }

    fn bracket(&self, cmd: &str, stack: &mut Vec<State>, out: &mut Vec<Item>) -> Result<()> {
        let lower = cmd.to_ascii_lowercase();
        let mut words = lower.split_whitespace();
        let head = words.next().unwrap_or("");
        let rest: String = words.collect::<Vec<_>>().join(" ");
        let top = stack.last_mut().unwrap();
        match head {
            "" => {}
            "/" | "/mood" | "reset" | "default" => top.moods.clear(),
            "pause" | "break" | "silence" => {
                let ms = if rest.is_empty() {
                    400.0
                } else {
                    parse_duration_ms(&rest)
                        .ok_or_else(|| anyhow::anyhow!("bad pause duration '{rest}'"))?
                };
                out.push(Item::Pause { ms });
            }
            "breath" | "breathes" | "inhale" | "inhales" => out.push(Item::Breath),
            "laugh" | "laughs" | "laughing" | "chuckles" => {
                out.push(self.nonverbal(top, "laugh", "hˈɑ hɑ hɑ!", &[("excited", 0.8), ("happy", 0.5)]));
            }
            "sigh" | "sighs" => {
                out.push(self.nonverbal(top, "sigh", "hˈɑːːh.", &[("sad", 0.6), ("whisper", 0.75)]));
                out.push(Item::Pause { ms: 150.0 });
            }
            "voice" | "persona" | "speaker" => {
                if rest.is_empty() {
                    bail!("[{head}] needs a persona name, e.g. [voice ava]");
                }
                top.persona = Some(rest);
            }
            _ => {
                if let Some(rest) = head.strip_prefix("voice:").or_else(|| head.strip_prefix("persona:")) {
                    top.persona = Some(rest.to_string());
                    return Ok(());
                }
                match self.parse_mood_mix(&lower)? {
                    Some(mix) => {
                        if mix.len() == 1 && mix[0].0 == "neutral" && mix[0].1 == 1.0 {
                            top.moods = vec![("neutral".into(), 1.0)];
                        } else {
                            top.moods = mix;
                        }
                    }
                    None => bail!(
                        "unknown tag [{cmd}] — not a mood, pause, breath, laughs, sighs or voice. \
                         Escape literal brackets as \\["
                    ),
                }
            }
        }
        Ok(())
    }

    fn nonverbal(&self, top: &State, kind: &str, text: &str, moods: &[(&str, f32)]) -> Item {
        Item::Speech(Span {
            text: text.into(),
            ipa: true, // IPA so it works for every accent without a phonemizer
            persona: top.persona.clone(),
            moods: moods.iter().map(|(m, k)| (m.to_string(), *k)).collect(),
            rate: top.rate,
            pitch_st: top.pitch_st,
            volume_db: top.volume_db - 2.0,
            emphasis: false,
            nonverbal: Some(kind.to_string()),
        })
    }

    fn angle(&self, tag: &str, stack: &mut Vec<State>, out: &mut Vec<Item>) -> Result<()> {
        let self_closing = tag.ends_with('/');
        let tag = tag.trim_end_matches('/').trim();
        if let Some(name) = tag.strip_prefix('/') {
            let name = name.trim().to_ascii_lowercase();
            if stack.len() > 1 && stack.last().unwrap().tag == name {
                stack.pop();
                return Ok(());
            }
            if name == "speak" {
                return Ok(());
            }
            bail!("closing </{name}> without matching opening tag");
        }
        let (name, attrs) = split_tag(tag);
        let name = name.to_ascii_lowercase();
        let attr = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());

        if name == "break" || name == "pause" {
            let ms = attr("time")
                .and_then(parse_duration_ms)
                .or_else(|| match attr("strength") {
                    Some("weak") => Some(200.0),
                    Some("strong") => Some(800.0),
                    Some("x-strong") => Some(1200.0),
                    _ => None,
                })
                .unwrap_or(400.0);
            out.push(Item::Pause { ms });
            return Ok(());
        }
        if name == "speak" || name == "s" || name == "p" || name.starts_with('?') || name.starts_with('!') {
            return Ok(());
        }
        let mut st = stack.last().unwrap().clone();
        st.tag = name.clone();
        match name.as_str() {
            "mood" | "emotion" | "style" => {
                let mood_name = attr("name").or(attr("type")).unwrap_or("neutral").to_ascii_lowercase();
                let k = match attr("intensity").or(attr("level")).or(attr("degree")) {
                    Some(v) => parse_intensity(v)?,
                    None => 1.0,
                };
                match self.parse_mood_mix(&mood_name)? {
                    Some(mut mix) => {
                        for m in mix.iter_mut() {
                            m.1 *= k;
                        }
                        st.moods = mix;
                    }
                    None => bail!("unknown mood '{mood_name}'"),
                }
            }
            "emph" | "emphasis" | "em" | "strong" | "b" => st.emphasis = true,
            "prosody" => {
                if let Some(r) = attr("rate") {
                    st.rate *= parse_rate(r).ok_or_else(|| anyhow::anyhow!("bad rate '{r}'"))?;
                }
                if let Some(p) = attr("pitch") {
                    st.pitch_st += parse_semitones(p).ok_or_else(|| anyhow::anyhow!("bad pitch '{p}'"))?;
                }
                if let Some(v) = attr("volume") {
                    st.volume_db += parse_db(v).ok_or_else(|| anyhow::anyhow!("bad volume '{v}'"))?;
                }
            }
            "voice" | "persona" | "speaker" => {
                let n = attr("name").ok_or_else(|| anyhow::anyhow!("<voice> needs name=\"...\""))?;
                st.persona = Some(n.to_ascii_lowercase());
            }
            "ipa" | "phoneme" => {
                if let Some(ph) = attr("ph") {
                    // <phoneme ph="..."/> form
                    let top = stack.last().unwrap();
                    out.push(Item::Speech(Span {
                        text: ph.to_string(),
                        ipa: true,
                        persona: top.persona.clone(),
                        moods: top.moods.clone(),
                        rate: top.rate,
                        pitch_st: top.pitch_st,
                        volume_db: top.volume_db,
                        emphasis: top.emphasis,
                        nonverbal: None,
                    }));
                    return Ok(());
                }
                st.ipa = true
            }
            other if (self.is_mood)(other) => {
                let k = match attr("intensity") {
                    Some(v) => parse_intensity(v)?,
                    None => 1.0,
                };
                st.moods = vec![(other.to_string(), k)];
            }
            other => bail!("unknown tag <{other}>"),
        }
        if !self_closing {
            stack.push(st);
        }
        Ok(())
    }
}

fn parse_intensity(v: &str) -> Result<f32> {
    let v = v.trim();
    let k = if let Some(p) = v.strip_suffix('%') {
        p.trim().parse::<f32>().map(|x| x / 100.0)
    } else {
        v.parse::<f32>()
    };
    match k {
        Ok(k) if (0.0..=2.0).contains(&k) => Ok(k),
        _ => bail!("bad intensity '{v}' (use 0..1, e.g. 0.6 or 60%)"),
    }
}

fn find(chars: &[char], from: usize, c: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == c)
}

fn split_tag(tag: &str) -> (String, Vec<(String, String)>) {
    let mut it = tag.splitn(2, char::is_whitespace);
    let name = it.next().unwrap_or("").to_string();
    let rest = it.next().unwrap_or("");
    let mut attrs = Vec::new();
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        let ks = i;
        while i < chars.len() && chars[i] != '=' && !chars[i].is_whitespace() {
            i += 1;
        }
        let key: String = chars[ks..i].iter().collect::<String>().to_ascii_lowercase();
        if i < chars.len() && chars[i] == '=' {
            i += 1;
            let quote = if i < chars.len() && (chars[i] == '"' || chars[i] == '\'') {
                i += 1;
                Some(chars[i - 1])
            } else {
                None
            };
            let vs = i;
            while i < chars.len() && (if let Some(q) = quote { chars[i] != q } else { !chars[i].is_whitespace() }) {
                i += 1;
            }
            let val: String = chars[vs..i].iter().collect();
            if quote.is_some() {
                i += 1;
            }
            if !key.is_empty() {
                attrs.push((key, val));
            }
        } else if !key.is_empty() {
            attrs.push((key, String::new()));
        }
    }
    (name, attrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn moods(n: &str) -> bool {
        ["happy", "sad", "whisper", "neutral", "excited"].contains(&n)
    }

    #[test]
    fn mid_sentence_mood() {
        let f = moods;
        let p = Parser::new(&f);
        let items = p.parse("I was so [happy] glad, [sad 0.5] until you left.").unwrap();
        let spans: Vec<_> = items
            .iter()
            .filter_map(|i| if let Item::Speech(s) = i { Some(s) } else { None })
            .collect();
        assert_eq!(spans.len(), 3);
        assert!(spans[0].moods.is_empty());
        assert_eq!(spans[1].moods, vec![("happy".to_string(), 1.0)]);
        assert_eq!(spans[2].moods, vec![("sad".to_string(), 0.5)]);
    }

    #[test]
    fn tags_and_pauses() {
        let f = moods;
        let p = Parser::new(&f);
        let items = p
            .parse("<speak>Hi <mood name=\"happy\" intensity=\"60%\">there</mood><break time=\"1s\"/> *you* [voice pierre] salut</speak>")
            .unwrap();
        assert!(items.contains(&Item::Pause { ms: 1000.0 }));
        let last = items.iter().rev().find_map(|i| if let Item::Speech(s) = i { Some(s) } else { None }).unwrap();
        assert_eq!(last.persona.as_deref(), Some("pierre"));
        assert!(items.iter().any(|i| matches!(i, Item::Speech(s) if s.emphasis && s.text.contains("you"))));
    }

    #[test]
    fn unknown_tag_errors() {
        let f = moods;
        let p = Parser::new(&f);
        assert!(p.parse("hello [hapy] world").is_err());
        assert!(p.parse("a \\[literal] bracket").is_ok());
    }
}
