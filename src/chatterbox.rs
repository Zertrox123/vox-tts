//! Chatterbox Multilingual (Resemble AI, MIT) — emotional zero-shot TTS, 24 kHz.
//!
//! Pipeline (all ONNX, CPU, offline):
//!   reference wav --speech_encoder--> cond embeddings, prompt tokens, speaker embedding/features
//!   text --BPE--> ids --embed_tokens(+exaggeration)--> embeds
//!   0.5B Llama (KV cache, classifier-free guidance) samples speech tokens @25 Hz
//!   conditional_decoder (flow matching + vocoder) -> waveform
//!
//! `exaggeration` is the model's own emotion-intensity control.

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::{DynValue, Tensor, ValueType};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

pub const SAMPLE_RATE: u32 = 24_000;
const START_SPEECH: i64 = 6561;
const STOP_SPEECH: i64 = 6562;
const EXAGGERATION_TOK: i64 = 6563;
const START_TEXT: i64 = 255;
const STOP_TEXT: i64 = 0;

// ---------------------------------------------------------------- tokenizer

pub struct CbTokenizer {
    vocab: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    added: Vec<(Vec<char>, i64)>,
    unk: i64,
}

fn is_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || is_mark(c)
}

impl CbTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let t: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let m = &t["model"];
        let mut vocab = HashMap::new();
        for (k, v) in m["vocab"].as_object().ok_or_else(|| anyhow!("tokenizer: no vocab"))? {
            vocab.insert(k.clone(), v.as_i64().unwrap_or(1));
        }
        let mut ranks = HashMap::new();
        for (i, mg) in m["merges"].as_array().ok_or_else(|| anyhow!("tokenizer: no merges"))?.iter().enumerate() {
            let pair = match mg {
                serde_json::Value::String(s) => {
                    let mut it = s.splitn(2, ' ');
                    (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
                }
                serde_json::Value::Array(a) => (
                    a.first().and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    a.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                ),
                _ => continue,
            };
            ranks.insert(pair, i);
        }
        let mut added: Vec<(Vec<char>, i64)> = t["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| Some((x["content"].as_str()?.chars().collect(), x["id"].as_i64()?)))
                    .collect()
            })
            .unwrap_or_default();
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let unk = *vocab.get("[UNK]").unwrap_or(&1);
        Ok(CbTokenizer { vocab, ranks, added, unk })
    }

    pub fn has_token(&self, s: &str) -> bool {
        let c: Vec<char> = s.chars().collect();
        self.added.iter().any(|(a, _)| *a == c)
    }

    fn bpe(&self, word: &str, out: &mut Vec<i64>) {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        for s in syms {
            out.push(*self.vocab.get(&s).unwrap_or(&self.unk));
        }
    }

    /// Full Chatterbox-MTL preprocessing: lowercase, NFKD, language tag, [SPACE].
    pub fn encode(&self, text: &str, lang: &str) -> Vec<i64> {
        let pre: String = if std::env::var("VOX_CB_RAW").is_ok() { text.to_string() } else { text.to_lowercase().nfkd().collect() };
        let full = format!("[{lang}]{pre}").replace(' ', "[SPACE]");
        let chars: Vec<char> = full.chars().collect();
        let mut ids = Vec::new();
        let mut seg = String::new();
        let flush = |seg: &mut String, ids: &mut Vec<i64>| {
            // Whitespace pre-tokenizer: \w+ | [^\w\s]+
            let mut word = String::new();
            let mut cls: Option<bool> = None;
            for c in seg.chars() {
                if c.is_whitespace() {
                    if !word.is_empty() {
                        self.bpe(&word, ids);
                        word.clear();
                    }
                    cls = None;
                    continue;
                }
                let w = is_word(c);
                if cls.is_some_and(|k| k != w) && !word.is_empty() {
                    self.bpe(&word, ids);
                    word.clear();
                }
                cls = Some(w);
                word.push(c);
            }
            if !word.is_empty() {
                self.bpe(&word, ids);
            }
            seg.clear();
        };
        let mut i = 0;
        'outer: while i < chars.len() {
            if chars[i] == '[' {
                for (tok, id) in &self.added {
                    if chars[i..].starts_with(tok) {
                        flush(&mut seg, &mut ids);
                        ids.push(*id);
                        i += tok.len();
                        continue 'outer;
                    }
                }
            }
            seg.push(chars[i]);
            i += 1;
        }
        flush(&mut seg, &mut ids);
        ids
    }
}

/// Chatterbox's punctuation normalisation (punc_norm).
pub fn punc_norm(text: &str) -> String {
    let mut t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        return "You need to add some text for me to talk.".into();
    }
    for (a, b) in [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
        ("«", "\""),
        ("»", "\""),
    ] {
        t = t.replace(a, b);
    }
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut cs = t.chars();
    let mut t: String = match cs.next() {
        Some(f) => f.to_uppercase().chain(cs).collect(),
        None => t,
    };
    if !t.ends_with(['.', '!', '?', '-', ',', '、', '，', '。', '？', '！']) {
        t.push('.');
    }
    t
}

// ---------------------------------------------------------------- model

#[derive(Clone, Copy, Debug)]
pub struct CbParams {
    /// Emotion intensity, 0.25 (flat) .. 2.0 (very dramatic). 0.5 = default.
    pub exaggeration: f32,
    /// Classifier-free guidance weight (0 disables). Higher = closer to the text, clearer.
    pub cfg_weight: f32,
    pub temperature: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for CbParams {
    fn default() -> Self {
        CbParams {
            exaggeration: 0.5,
            cfg_weight: 0.5,
            temperature: 0.8,
            min_p: 0.05,
            top_p: 1.0,
            repetition_penalty: 2.0,
            seed: 7,
        }
    }
}

/// Encoded reference voice (speaker identity + prosody prompt).
#[derive(Clone)]
pub struct VoiceCond {
    cond_emb: Vec<f32>,
    cond_len: usize,
    prompt_tokens: Vec<i64>,
    spk_emb: Vec<f32>,
    spk_feat: Vec<f32>,
    spk_feat_len: usize,
}

pub struct Chatterbox {
    pub tokenizer: CbTokenizer,
    encoder: Session,
    embed: Session,
    lm: Session,
    decoder: Session,
    kv_inputs: Vec<String>,
    kv_heads: i64,
    head_dim: i64,
    hidden: usize,
    /// Whether the last `synth` ended with the model's own stop token.
    pub last_stopped: bool,
}

fn session(path: &Path) -> Result<Session> {
    if !path.exists() {
        bail!("Chatterbox model missing: {} (run scripts/download_models.sh)", path.display());
    }
    let n = std::env::var("VOX_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8));
    Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(n)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow!("loading {}: {e}", path.display()))
}

fn f32s(v: &DynValue) -> Result<(Vec<i64>, Vec<f32>)> {
    let (s, d) = v.try_extract_tensor::<f32>()?;
    Ok((s.iter().copied().collect(), d.to_vec()))
}

impl Chatterbox {
    pub fn available(dir: &Path) -> bool {
        dir.join("onnx/conditional_decoder.onnx").exists() && dir.join("tokenizer.json").exists()
    }

    /// `quality`: "high" = fp32 language model, otherwise the 4-bit one (≈3x faster).
    pub fn load(dir: &Path, quality: &str) -> Result<Self> {
        let lm_file = if quality == "high" && dir.join("onnx/language_model.onnx").exists() {
            "onnx/language_model.onnx"
        } else {
            "onnx/language_model_q4.onnx"
        };
        let lm = session(&dir.join(lm_file))?;
        let mut kv_inputs = Vec::new();
        let (mut kv_heads, mut head_dim) = (16i64, 64i64);
        for i in &lm.inputs {
            if i.name.starts_with("past_key_values") {
                if let ValueType::Tensor { shape, .. } = &i.input_type {
                    if shape.len() == 4 {
                        if shape[1] > 0 {
                            kv_heads = shape[1];
                        }
                        if shape[3] > 0 {
                            head_dim = shape[3];
                        }
                    }
                }
                kv_inputs.push(i.name.clone());
            }
        }
        Ok(Chatterbox {
            tokenizer: CbTokenizer::load(&dir.join("tokenizer.json")).context("chatterbox tokenizer")?,
            encoder: session(&dir.join("onnx/speech_encoder.onnx"))?,
            embed: session(&dir.join("onnx/embed_tokens.onnx"))?,
            decoder: session(&dir.join("onnx/conditional_decoder.onnx"))?,
            lm,
            kv_inputs,
            kv_heads,
            head_dim,
            hidden: 1024,
            last_stopped: false,
        })
    }

    /// Encode a 24 kHz mono reference clip (≈5–10 s of clean speech is ideal).
    pub fn encode_voice(&mut self, wav24k: &[f32]) -> Result<VoiceCond> {
        let max = SAMPLE_RATE as usize * 10;
        let a: Vec<f32> = wav24k[..wav24k.len().min(max)].to_vec();
        let n = a.len();
        let inputs: Vec<(Cow<str>, SessionInputValue)> =
            vec![(Cow::from("audio_values"), Tensor::<f32>::from_array((vec![1i64, n as i64], a))?.into())];
        let out = self.encoder.run(inputs)?;
        let (cs, cond_emb) = f32s(&out["audio_features"])?;
        let (_, toks) = out["audio_tokens"].try_extract_tensor::<i64>()?;
        let (_, spk_emb) = f32s(&out["speaker_embeddings"])?;
        let (fs, spk_feat) = f32s(&out["speaker_features"])?;
        Ok(VoiceCond {
            cond_len: cs[1] as usize,
            cond_emb,
            prompt_tokens: toks.to_vec(),
            spk_emb,
            spk_feat_len: fs[1] as usize,
            spk_feat,
        })
    }

    fn embed_ids(&mut self, ids: &[Vec<i64>], pos: &[Vec<i64>], exag: f32) -> Result<(usize, Vec<f32>)> {
        let b = ids.len();
        let n = ids[0].len();
        let flat_ids: Vec<i64> = ids.concat();
        let flat_pos: Vec<i64> = pos.concat();
        let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (Cow::from("input_ids"), Tensor::<i64>::from_array((vec![b as i64, n as i64], flat_ids))?.into()),
            (Cow::from("position_ids"), Tensor::<i64>::from_array((vec![b as i64, n as i64], flat_pos))?.into()),
            (Cow::from("exaggeration"), Tensor::<f32>::from_array((vec![b as i64], vec![exag; b]))?.into()),
        ];
        let out = self.embed.run(inputs)?;
        let (_, e) = f32s(&out["inputs_embeds"])?;
        Ok((n, e))
    }

    /// Generate speech for `text` in `lang` ("en", "fr", ...).
    pub fn synth(&mut self, text: &str, lang: &str, voice: &VoiceCond, p: &CbParams) -> Result<Vec<f32>> {
        let text = punc_norm(text);
        let mut ids = vec![EXAGGERATION_TOK, START_TEXT];
        ids.extend(self.tokenizer.encode(&text, lang));
        ids.extend([STOP_TEXT, START_SPEECH, START_SPEECH]);
        let n_text = ids.len();
        let pos: Vec<i64> = ids.iter().enumerate().map(|(i, &t)| if t >= START_SPEECH { 0 } else { i as i64 - 1 }).collect();

        let cfg = p.cfg_weight > 0.0;
        let b = if cfg { 2 } else { 1 };
        let (_, emb) = self.embed_ids(&vec![ids.clone(); b], &vec![pos; b], p.exaggeration)?;
        let h = self.hidden;
        // [cond_emb | text embeds] per batch row; CFG row 2 has text embeddings zeroed
        let seq = voice.cond_len + n_text;
        let mut x = Vec::with_capacity(b * seq * h);
        for row in 0..b {
            x.extend_from_slice(&voice.cond_emb);
            let e = &emb[row * n_text * h..(row + 1) * n_text * h];
            if row == 1 {
                for t in 0..n_text {
                    let keep = t == 0 || t >= n_text - 2; // exaggeration token & speech-start stay
                    if keep {
                        x.extend_from_slice(&e[t * h..(t + 1) * h]);
                    } else {
                        x.extend(std::iter::repeat_n(0.0, h));
                    }
                }
            } else {
                x.extend_from_slice(e);
            }
        }

        let max_new = ((text.chars().count() as f32 * 1.8) as usize + 80).min(1500);
        let mut rng = crate::util::Rng::new(p.seed);
        let mut generated: Vec<i64> = vec![START_SPEECH];
        let mut past: Vec<DynValue> = Vec::new();
        let alloc = ort::memory::Allocator::default();
        for _ in &self.kv_inputs {
            // empty cache: zero-length sequence axis (from_array rejects 0-sized dims)
            past.push(Tensor::<f32>::new(&alloc, [b, self.kv_heads as usize, 0, self.head_dim as usize])?.into_dyn());
        }
        let mut cur_embeds = Tensor::<f32>::from_array((vec![b as i64, seq as i64, h as i64], x))?.into_dyn();
        let mut total_len = seq;
        let mut run_len = 0usize;
        self.last_stopped = false;
        for step in 0..max_new {
            let mask = vec![1i64; b * total_len];
            let mut inputs: Vec<(Cow<str>, SessionInputValue)> = Vec::with_capacity(2 + past.len());
            inputs.push((Cow::from("inputs_embeds"), cur_embeds.into()));
            inputs.push((
                Cow::from("attention_mask"),
                Tensor::<i64>::from_array((vec![b as i64, total_len as i64], mask))?.into(),
            ));
            for (name, v) in self.kv_inputs.iter().zip(past.drain(..)) {
                inputs.push((Cow::from(name.clone()), v.into()));
            }
            let mut out = self.lm.run(inputs)?;
            let logits_v = out.remove("logits").ok_or_else(|| anyhow!("lm: no logits"))?;
            for name in &self.kv_inputs {
                let pn = name.replace("past_key_values", "present");
                past.push(out.remove(&pn).ok_or_else(|| anyhow!("lm: missing {pn}"))?);
            }
            drop(out);
            let (shape, logits) = logits_v.try_extract_tensor::<f32>()?;
            let (sl, vsz) = (shape[1] as usize, shape[2] as usize);
            let last = |row: usize| &logits[(row * sl + sl - 1) * vsz..(row * sl + sl) * vsz];
            let mut l: Vec<f32> = last(0).to_vec();
            if cfg {
                let u = last(1);
                for (a, bu) in l.iter_mut().zip(u) {
                    *a += p.cfg_weight * (*a - bu);
                }
            }
            let tok = sample(&mut l, &generated, p, &mut rng);
            generated.push(tok);
            if tok == STOP_SPEECH {
                self.last_stopped = true;
                break;
            }
            // runaway guard: the same token over and over = hallucinated silence/drone
            run_len = if generated.len() > 2 && generated[generated.len() - 2] == tok { run_len + 1 } else { 0 };
            if run_len > 50 {
                break;
            }
            let (_, e) = self.embed_ids(&vec![vec![tok]; b], &vec![vec![step as i64 + 1]; b], p.exaggeration)?;
            cur_embeds = Tensor::<f32>::from_array((vec![b as i64, 1, h as i64], e))?.into_dyn();
            total_len += 1;
        }

        let mut tokens = voice.prompt_tokens.clone();
        tokens.extend(generated.iter().skip(1).filter(|&&t| t < START_SPEECH));
        if tokens.len() == voice.prompt_tokens.len() {
            return Ok(vec![]);
        }
        let n = tokens.len();
        let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (Cow::from("speech_tokens"), Tensor::<i64>::from_array((vec![1i64, n as i64], tokens))?.into()),
            (
                Cow::from("speaker_embeddings"),
                Tensor::<f32>::from_array((vec![1i64, voice.spk_emb.len() as i64], voice.spk_emb.clone()))?.into(),
            ),
            (
                Cow::from("speaker_features"),
                Tensor::<f32>::from_array((vec![1i64, voice.spk_feat_len as i64, 80], voice.spk_feat.clone()))?.into(),
            ),
        ];
        let out = self.decoder.run(inputs)?;
        let (_, wav) = out["waveform"].try_extract_tensor::<f32>()?;
        Ok(wav.to_vec())
    }
}

/// Repetition penalty -> temperature -> min-p -> top-p -> multinomial.
fn sample(logits: &mut [f32], history: &[i64], p: &CbParams, rng: &mut crate::util::Rng) -> i64 {
    let mut seen = std::collections::HashSet::new();
    for &t in history {
        if seen.insert(t) {
            let v = &mut logits[t as usize];
            *v = if *v < 0.0 { *v * p.repetition_penalty } else { *v / p.repetition_penalty };
        }
    }
    let temp = p.temperature.max(1e-3);
    let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, ((v - maxl) / temp).exp())).collect();
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    probs.iter_mut().for_each(|x| x.1 /= sum);
    let pmax = probs.iter().map(|x| x.1).fold(0.0, f32::max);
    probs.retain(|x| x.1 >= p.min_p * pmax);
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    if p.top_p < 1.0 {
        let mut acc = 0.0;
        let mut k = 0;
        for (i, x) in probs.iter().enumerate() {
            acc += x.1;
            k = i + 1;
            if acc >= p.top_p {
                break;
            }
        }
        probs.truncate(k.max(1));
    }
    let total: f32 = probs.iter().map(|x| x.1).sum();
    let mut r = rng.uniform() * total;
    for (i, pr) in &probs {
        r -= pr;
        if r <= 0.0 {
            return *i as i64;
        }
    }
    probs[0].0 as i64
}

/// Read a WAV file as mono f32 at 24 kHz.
pub fn read_wav_24k(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<std::result::Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    Ok(crate::dsp::effects::resample(&mono, spec.sample_rate, SAMPLE_RATE))
}

pub fn default_dir(models: &Path) -> PathBuf {
    models.join("chatterbox")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokenizer_roundtrip() {
        let p = Path::new("models/chatterbox/tokenizer.json");
        if !p.exists() {
            return;
        }
        let t = CbTokenizer::load(p).unwrap();
        let ids = t.encode("Bonjour, ça va?", "fr");
        assert_eq!(ids[0], 634); // [fr]
        assert!(ids.contains(&2)); // [SPACE]
        assert!(!ids.contains(&1), "unexpected [UNK]: {ids:?}");
        let en = t.encode("The weather is nice today.", "en");
        assert!(!en.contains(&1));
    }
    #[test]
    fn punc() {
        assert_eq!(punc_norm("hello there"), "Hello there.");
        assert_eq!(punc_norm("wait… what"), "Wait, what.");
    }
}
