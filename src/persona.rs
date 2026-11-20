//! Personas: a named voice (neural model + speaker / style blend), its accent,
//! gender, and default delivery (speed, pitch, mood).

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Female,
    Male,
    Neutral,
}

impl fmt::Display for Gender {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Gender::Female => "female",
            Gender::Male => "male",
            Gender::Neutral => "neutral",
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Accent {
    #[serde(rename = "en-us", alias = "us", alias = "en_US")]
    EnUs,
    #[serde(rename = "fr-fr", alias = "fr", alias = "fr_FR")]
    FrFr,
}

impl Accent {
    pub fn espeak_voice(&self) -> &'static str {
        match self {
            Accent::EnUs => "en-us",
            Accent::FrFr => "fr-fr",
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Accent::EnUs => "US English",
            Accent::FrFr => "French (France)",
        }
    }
    pub fn is_french(&self) -> bool {
        matches!(self, Accent::FrFr)
    }
}

impl fmt::Display for Accent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.espeak_voice())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceWeight {
    pub id: String,
    #[serde(default = "one")]
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "engine", rename_all = "lowercase")]
pub enum VoiceRef {
    /// Kokoro-82M. Several style vectors can be blended to design new voices.
    Kokoro { voices: Vec<VoiceWeight> },
    /// Chatterbox Multilingual: emotional zero-shot model conditioned on a reference voice.
    /// `reference`: "default" (bundled voice), "persona:<name>" (rendered once from a
    /// -lite persona and cached), or a path to any WAV file (voice cloning).
    Chatterbox { reference: String },
    /// Piper / VITS model file stem in models/piper, optional speaker for multi-speaker models.
    Piper {
        model: String,
        #[serde(default)]
        speaker: Option<String>,
    },
}

impl VoiceRef {
    pub fn engine(&self) -> &'static str {
        match self {
            VoiceRef::Kokoro { .. } => "kokoro",
            VoiceRef::Chatterbox { .. } => "chatterbox",
            VoiceRef::Piper { .. } => "piper",
        }
    }
    pub fn describe(&self) -> String {
        match self {
            VoiceRef::Kokoro { voices } => voices
                .iter()
                .map(|v| {
                    if voices.len() == 1 {
                        v.id.clone()
                    } else {
                        format!("{}×{:.2}", v.id, v.weight)
                    }
                })
                .collect::<Vec<_>>()
                .join(" + "),
            VoiceRef::Chatterbox { reference } => format!("ref={reference}"),
            VoiceRef::Piper { model, speaker } => match speaker {
                Some(s) => format!("{model}#{s}"),
                None => model.clone(),
            },
        }
    }
}

fn one() -> f32 {
    1.0
}
fn neutral() -> String {
    "neutral".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Persona {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub gender: Gender,
    pub accent: Accent,
    pub voice: VoiceRef,
    /// Base speaking rate multiplier.
    #[serde(default = "one")]
    pub speed: f32,
    /// Base pitch offset in semitones (formant-preserving).
    #[serde(default)]
    pub pitch_st: f32,
    /// Default mood when none is given.
    #[serde(default = "neutral")]
    pub mood: String,
}

fn kokoro(name: &str, g: Gender, a: Accent, desc: &str, voices: &[(&str, f32)]) -> Persona {
    Persona {
        name: name.into(),
        description: desc.into(),
        gender: g,
        accent: a,
        voice: VoiceRef::Kokoro {
            voices: voices
                .iter()
                .map(|(id, w)| VoiceWeight { id: (*id).into(), weight: *w })
                .collect(),
        },
        speed: 1.0,
        pitch_st: 0.0,
        mood: neutral(),
    }
}

fn piper(name: &str, g: Gender, a: Accent, desc: &str, model: &str, spk: Option<&str>) -> Persona {
    Persona {
        name: name.into(),
        description: desc.into(),
        gender: g,
        accent: a,
        voice: VoiceRef::Piper { model: model.into(), speaker: spk.map(Into::into) },
        speed: 1.0,
        pitch_st: 0.0,
        mood: neutral(),
    }
}

fn cb(name: &str, g: Gender, a: Accent, desc: &str, reference: &str) -> Persona {
    Persona {
        name: name.into(),
        description: desc.into(),
        gender: g,
        accent: a,
        voice: VoiceRef::Chatterbox { reference: reference.into() },
        speed: 1.0,
        pitch_st: 0.0,
        mood: neutral(),
    }
}

pub fn builtin_personas() -> Vec<Persona> {
    use Accent::*;
    use Gender::*;
    let mut v = vec![
        // --- Expressive (Chatterbox, real emotion control) ---
        cb("emma", Female, EnUs, "Expressive American female (Chatterbox's own voice)", "default"),
        cb("ava", Female, EnUs, "Expressive, warm American female", "persona:ava-lite"),
        cb("bella", Female, EnUs, "Expressive, bright American female", "persona:bella-lite"),
        cb("michael", Male, EnUs, "Expressive, friendly American male", "persona:michael-lite"),
        cb("ryan", Male, EnUs, "Expressive, clear American male", "persona:ryan-lite"),
        cb("fenrir", Male, EnUs, "Expressive, deep American male", "persona:fenrir-lite"),
        cb("sam", Neutral, EnUs, "Expressive androgynous voice", "persona:sam-lite"),
        cb("camille", Female, FrFr, "Voix féminine expressive", "persona:camille-lite"),
        cb("jessica", Female, FrFr, "Voix féminine expressive, posée", "persona:jessica-lite"),
        cb("thomas", Male, FrFr, "Voix masculine expressive", "persona:thomas-lite"),
        cb("pierre", Male, FrFr, "Voix masculine expressive, grave", "persona:pierre-lite"),
        cb("alex", Neutral, FrFr, "Voix neutre / androgyne expressive", "persona:alex-lite"),
        // --- Lite (fast, Kokoro / Piper; moods applied by DSP only) ---
        kokoro("ava-lite", Female, EnUs, "Fast: warm narrator (Kokoro)", &[("af_heart", 1.0)]),
        kokoro("bella-lite", Female, EnUs, "Fast: bright (Kokoro)", &[("af_bella", 1.0)]),
        kokoro("nicole-lite", Female, EnUs, "Fast: soft, close-mic (Kokoro)", &[("af_nicole", 1.0)]),
        kokoro("nova-lite", Female, EnUs, "Fast: blend heart + bella (Kokoro)", &[("af_heart", 0.6), ("af_bella", 0.4)]),
        kokoro("michael-lite", Male, EnUs, "Fast: friendly (Kokoro)", &[("am_michael", 1.0)]),
        kokoro("fenrir-lite", Male, EnUs, "Fast: deep (Kokoro)", &[("am_fenrir", 1.0)]),
        kokoro("puck-lite", Male, EnUs, "Fast: playful (Kokoro)", &[("am_puck", 1.0)]),
        kokoro("sam-lite", Neutral, EnUs, "Fast: androgynous blend (Kokoro)", &[("af_sky", 0.5), ("am_michael", 0.5)]),
        piper("lessac-lite", Female, EnUs, "Fast: studio female (Piper)", "en_US-lessac-high", None),
        piper("ryan-lite", Male, EnUs, "Fast: studio male (Piper)", "en_US-ryan-high", None),
        kokoro("camille-lite", Female, FrFr, "Rapide : féminine (Kokoro)", &[("ff_siwis", 1.0)]),
        piper("siwis-lite", Female, FrFr, "Rapide : féminine claire (Piper)", "fr_FR-siwis-medium", None),
        piper("jessica-lite", Female, FrFr, "Rapide : féminine posée (Piper)", "fr_FR-upmc-medium", Some("jessica")),
        piper("thomas-lite", Male, FrFr, "Rapide : masculine (Piper)", "fr_FR-tom-medium", None),
        piper("pierre-lite", Male, FrFr, "Rapide : masculine grave (Piper)", "fr_FR-upmc-medium", Some("pierre")),
    ];
    let mut alex = piper("alex-lite", Neutral, FrFr, "Rapide : neutre / androgyne (Piper)", "fr_FR-tom-medium", None);
    alex.pitch_st = 3.5;
    v.push(alex);
    v
}
