//! Neural acoustic backends (ONNX Runtime, CPU, offline):
//! - Kokoro-82M (StyleTTS2-based, 24 kHz): best quality, voice blending.
//! - Piper (VITS, 22.05 kHz): many voices/speakers, controllable sampling noise.

use crate::persona::VoiceWeight;
use anyhow::{anyhow, bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::Tensor;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Synthesis output. `char_starts[i]` = sample where phoneme char i starts, when the
/// model exposes durations (enables exact word-level mood alignment).
pub struct SynthOut {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub char_starts: Option<Vec<usize>>,
}

/// Model-level controls derived from the mood.
#[derive(Clone, Copy, Debug)]
pub struct ModelControls {
    pub speed: f32,
    pub expressiveness: f32,
    pub rhythm: f32,
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)
}

fn open_session(path: &Path) -> Result<Session> {
    if !path.exists() {
        bail!("model not found: {} (run scripts/download_models.sh)", path.display());
    }
    let s = Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(threads())
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
    Ok(s)
}

pub fn describe_model(path: &Path) -> Result<String> {
    let s = open_session(path)?;
    let mut out = String::new();
    for i in &s.inputs {
        out += &format!("input  {:<16} {:?}\n", i.name, i.input_type);
    }
    for o in &s.outputs {
        out += &format!("output {:<16} {:?}\n", o.name, o.output_type);
    }
    Ok(out)
}

fn find_name(names: &[String], wanted: &[&str]) -> Option<String> {
    for w in wanted {
        if let Some(n) = names.iter().find(|n| n.to_ascii_lowercase().contains(w)) {
            return Some(n.clone());
        }
    }
    None
}

// ---------------------------------------------------------------- Kokoro

pub struct Kokoro {
    session: Session,
    vocab: HashMap<char, i64>,
    voices_dir: PathBuf,
    styles: HashMap<String, Vec<f32>>,
    in_ids: String,
    in_style: String,
    in_speed: String,
    out_wave: String,
    out_dur: Option<String>,
}

const STYLE_DIM: usize = 256;
const KOKORO_MAX_TOKENS: usize = 510;

impl Kokoro {
    pub fn load(dir: &Path) -> Result<Self> {
        let session = open_session(&dir.join("model.onnx"))?;
        let tok: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("tokenizer.json")).context("reading kokoro tokenizer.json")?,
        )?;
        let mut vocab = HashMap::new();
        for (k, v) in tok["model"]["vocab"].as_object().ok_or_else(|| anyhow!("bad tokenizer.json"))? {
            let mut cs = k.chars();
            if let (Some(c), None) = (cs.next(), cs.next()) {
                vocab.insert(c, v.as_i64().unwrap_or(0));
            }
        }
        let ins: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        let outs: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
        Ok(Kokoro {
            in_ids: find_name(&ins, &["input_ids", "tokens", "input"]).ok_or_else(|| anyhow!("kokoro: no token input"))?,
            in_style: find_name(&ins, &["style", "ref_s"]).ok_or_else(|| anyhow!("kokoro: no style input"))?,
            in_speed: find_name(&ins, &["speed"]).ok_or_else(|| anyhow!("kokoro: no speed input"))?,
            out_wave: find_name(&outs, &["waveform", "audio", "wav"]).unwrap_or_else(|| outs[0].clone()),
            out_dur: find_name(&outs, &["dur"]),
            session,
            vocab,
            voices_dir: dir.join("voices"),
            styles: HashMap::new(),
        })
    }

    pub fn has_voice(dir: &Path, id: &str) -> bool {
        dir.join("voices").join(format!("{id}.bin")).exists()
    }

    fn style(&mut self, blend: &[VoiceWeight]) -> Result<&Vec<f32>> {
        let key = blend.iter().map(|v| format!("{}:{}", v.id, v.weight)).collect::<Vec<_>>().join(",");
        if !self.styles.contains_key(&key) {
            let total: f32 = blend.iter().map(|v| v.weight.max(0.0)).sum();
            if total <= 0.0 {
                bail!("voice blend weights must be positive");
            }
            let mut acc: Vec<f32> = Vec::new();
            for v in blend {
                let p = self.voices_dir.join(format!("{}.bin", v.id));
                let bytes = std::fs::read(&p)
                    .with_context(|| format!("kokoro voice '{}' not found at {}", v.id, p.display()))?;
                let data: Vec<f32> =
                    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                if data.len() % STYLE_DIM != 0 {
                    bail!("bad voice file {}", p.display());
                }
                if acc.is_empty() {
                    acc = vec![0.0; data.len()];
                }
                let w = v.weight.max(0.0) / total;
                for (a, d) in acc.iter_mut().zip(&data) {
                    *a += d * w;
                }
            }
            self.styles.insert(key.clone(), acc);
        }
        Ok(&self.styles[&key])
    }

    /// Phoneme normalization (eSpeak IPA -> Kokoro/misaki symbol set).
    fn normalize(ph: &str, english: bool) -> String {
        let mut s = ph.replace('ʲ', "j").replace('ɬ', "l");
        if english {
            s = s.replace('r', "ɹ").replace('x', "k");
            s = s.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ");
        }
        s
    }

    pub fn synth(&mut self, phonemes: &str, blend: &[VoiceWeight], speed: f32, english: bool) -> Result<SynthOut> {
        let norm = Self::normalize(phonemes, english);
        // map original char index -> token index (normalization is 1:1 char wise except
        // the rare multi-char replacement; we keep positions by char count)
        let chars: Vec<char> = norm.chars().collect();
        let mut tokens = vec![0i64];
        let mut char_tok = Vec::with_capacity(chars.len());
        for c in &chars {
            char_tok.push(tokens.len());
            if let Some(&id) = self.vocab.get(c) {
                if tokens.len() <= KOKORO_MAX_TOKENS {
                    tokens.push(id);
                }
            }
        }
        if tokens.len() > KOKORO_MAX_TOKENS {
            tokens.truncate(KOKORO_MAX_TOKENS + 1);
        }
        let n_ph = tokens.len() - 1;
        tokens.push(0);
        let style_all = self.style(blend)?.clone();
        let rows = style_all.len() / STYLE_DIM;
        let idx = n_ph.min(rows - 1);
        let style = style_all[idx * STYLE_DIM..(idx + 1) * STYLE_DIM].to_vec();

        let n = tokens.len();
        let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (Cow::from(self.in_ids.clone()), Tensor::<i64>::from_array((vec![1i64, n as i64], tokens))?.into()),
            (Cow::from(self.in_style.clone()), Tensor::<f32>::from_array((vec![1i64, STYLE_DIM as i64], style))?.into()),
            (Cow::from(self.in_speed.clone()), Tensor::<f32>::from_array((vec![1i64], vec![speed.clamp(0.4, 2.5)]))?.into()),
        ];
        let out_wave = self.out_wave.clone();
        let out_dur = self.out_dur.clone();
        let outputs = self.session.run(inputs)?;
        let (_, wave) = outputs[out_wave.as_str()].try_extract_tensor::<f32>()?;
        let audio = wave.to_vec();

        let mut char_starts = None;
        if let Some(dn) = out_dur {
            let dur: Option<Vec<f64>> = outputs[dn.as_str()]
                .try_extract_tensor::<i64>()
                .map(|(_, d)| d.iter().map(|&x| x as f64).collect())
                .or_else(|_| outputs[dn.as_str()].try_extract_tensor::<f32>().map(|(_, d)| d.iter().map(|&x| x as f64).collect()))
                .ok();
            if let Some(dur) = dur {
                let total: f64 = dur.iter().sum();
                if total > 0.0 && dur.len() == n {
                    let spf = audio.len() as f64 / total;
                    let mut tok_start = Vec::with_capacity(n + 1);
                    let mut acc = 0.0;
                    for d in &dur {
                        tok_start.push((acc * spf) as usize);
                        acc += d;
                    }
                    tok_start.push(audio.len());
                    char_starts = Some(char_tok.iter().map(|&t| tok_start[t.min(n)]).collect());
                }
            }
        }
        Ok(SynthOut { audio, sample_rate: 24_000, char_starts })
    }
}

// ---------------------------------------------------------------- Piper

pub struct Piper {
    session: Session,
    id_map: HashMap<char, Vec<i64>>,
    pub sample_rate: u32,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speakers: HashMap<String, i64>,
    has_sid: bool,
    has_scales: bool,
}

impl Piper {
    pub fn load(model: &Path) -> Result<Self> {
        let cfg_path = PathBuf::from(format!("{}.json", model.display()));
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?,
        )?;
        let session = open_session(model)?;
        let mut id_map = HashMap::new();
        for (k, v) in cfg["phoneme_id_map"].as_object().ok_or_else(|| anyhow!("no phoneme_id_map"))? {
            let mut cs = k.chars();
            if let (Some(c), None) = (cs.next(), cs.next()) {
                let ids: Vec<i64> = v.as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                id_map.insert(c, ids);
            }
        }
        let inf = &cfg["inference"];
        let mut speakers = HashMap::new();
        if let Some(m) = cfg["speaker_id_map"].as_object() {
            for (k, v) in m {
                speakers.insert(k.to_ascii_lowercase(), v.as_i64().unwrap_or(0));
            }
        }
        let ins: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        Ok(Piper {
            has_sid: ins.iter().any(|n| n == "sid"),
            has_scales: ins.iter().any(|n| n == "scales"),
            sample_rate: cfg["audio"]["sample_rate"].as_u64().unwrap_or(22050) as u32,
            noise_scale: inf["noise_scale"].as_f64().unwrap_or(0.667) as f32,
            length_scale: inf["length_scale"].as_f64().unwrap_or(1.0) as f32,
            noise_w: inf["noise_w"].as_f64().unwrap_or(0.8) as f32,
            session,
            id_map,
            speakers,
        })
    }

    pub fn speaker_names(&self) -> Vec<String> {
        self.speakers.keys().cloned().collect()
    }

    pub fn synth(&mut self, phonemes: &str, speaker: Option<&str>, ctl: &ModelControls) -> Result<SynthOut> {
        let get = |c: char| self.id_map.get(&c).cloned().unwrap_or_default();
        let pad = get('_');
        let mut ids = get('^');
        ids.extend(&pad);
        for c in phonemes.chars() {
            if let Some(v) = self.id_map.get(&c) {
                ids.extend(v);
                ids.extend(&pad);
            }
        }
        ids.extend(get('$'));
        let n = ids.len();
        let sid = match speaker {
            Some(s) => *self.speakers.get(&s.to_ascii_lowercase()).ok_or_else(|| {
                anyhow!("speaker '{s}' not in model (have: {})", self.speaker_names().join(", "))
            })?,
            None => 0,
        };
        let scales = vec![
            (self.noise_scale * ctl.expressiveness).clamp(0.1, 1.5),
            (self.length_scale / ctl.speed.clamp(0.4, 2.5)).clamp(0.3, 3.0),
            (self.noise_w * ctl.rhythm).clamp(0.1, 1.6),
        ];
        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (Cow::from("input"), Tensor::<i64>::from_array((vec![1i64, n as i64], ids))?.into()),
            (Cow::from("input_lengths"), Tensor::<i64>::from_array((vec![1i64], vec![n as i64]))?.into()),
        ];
        if self.has_scales {
            inputs.push((Cow::from("scales"), Tensor::<f32>::from_array((vec![3i64], scales))?.into()));
        }
        if self.has_sid {
            inputs.push((Cow::from("sid"), Tensor::<i64>::from_array((vec![1i64], vec![sid]))?.into()));
        }
        let outputs = self.session.run(inputs)?;
        let (_, wave) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(SynthOut { audio: wave.to_vec(), sample_rate: self.sample_rate, char_starts: None })
    }
}
