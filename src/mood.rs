//! Moods: named bundles of prosody / voice-quality controls.
//!
//! A mood is expressed relative to neutral. Several moods can be mixed with
//! individual intensities (`[happy 0.6 + whisper 0.3]`): multiplicative controls
//! combine as weighted products, additive ones as weighted sums and bounded
//! amounts (0..1) as "probabilistic OR", so mixing never explodes.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Mood {
    pub name: String,
    pub description: String,

    // --- neural model controls ---
    /// Emotion intensity added to Chatterbox's exaggeration (base 0.5; range 0.25..2).
    pub exaggeration: f32,
    /// Speaking-rate multiplier (1 = persona default).
    pub speed: f32,
    /// Multiplier on the model's sampling noise (Piper noise_scale): >1 more varied/expressive.
    pub expressiveness: f32,
    /// Multiplier on duration noise (Piper noise_w): >1 more irregular rhythm.
    pub rhythm: f32,

    // --- intonation (applied by PSOLA, formant-preserving) ---
    /// Global pitch offset in semitones.
    pub pitch_st: f32,
    /// Intonation range: 1 = as spoken, >1 more melodic, <1 flatter/monotone.
    pub pitch_range: f32,
    /// Pitch drift over the span in semitones (negative = falling, e.g. sadness).
    pub pitch_slope_st: f32,
    /// Random micro pitch variation (cents), e.g. tension or a shaky voice.
    pub jitter_cents: f32,
    pub vibrato_hz: f32,
    pub vibrato_cents: f32,

    // --- voice quality ---
    pub gain_db: f32,
    /// Spectral tilt: + brighter / tenser, - darker / softer.
    pub brightness_db: f32,
    /// 0..1 aspiration noise mixed into the voice.
    pub breathiness: f32,
    /// 0..1 replace voicing with noise excitation (LPC whisper).
    pub whisper: f32,
    pub tremolo_hz: f32,
    pub tremolo_depth: f32,
    /// 0..1 wet amount of a small room reverb.
    pub reverb: f32,

    // --- timing ---
    /// Multiplier on pauses between sentences.
    pub pause_scale: f32,

    // --- emotional voice prompts (expressive engine) ---
    /// A line performed in this mood to build the emotional voice prompt (English).
    pub prompt_en: String,
    /// Same, in French.
    pub prompt_fr: String,
}

impl Default for Mood {
    fn default() -> Self {
        Mood {
            name: "neutral".into(),
            description: String::new(),
            exaggeration: 0.0,
            speed: 1.0,
            expressiveness: 1.0,
            rhythm: 1.0,
            pitch_st: 0.0,
            pitch_range: 1.0,
            pitch_slope_st: 0.0,
            jitter_cents: 0.0,
            vibrato_hz: 5.5,
            vibrato_cents: 0.0,
            gain_db: 0.0,
            brightness_db: 0.0,
            breathiness: 0.0,
            whisper: 0.0,
            tremolo_hz: 6.0,
            tremolo_depth: 0.0,
            reverb: 0.0,
            pause_scale: 1.0,
            prompt_en: String::new(),
            prompt_fr: String::new(),
        }
    }
}

impl Mood {
    /// Mix moods with intensities. `parts` empty -> neutral.
    pub fn mix(parts: &[(&Mood, f32)]) -> Mood {
        let mut m = Mood::default();
        if parts.is_empty() {
            return m;
        }
        let mut names = Vec::new();
        let (mut vib_w, mut trem_w) = (0.0f32, 0.0f32);
        let (mut vib_hz, mut trem_hz) = (0.0f32, 0.0f32);
        let mut keep = [1.0f32; 4]; // 1 - amount products for bounded params
        for (mood, k) in parts {
            let k = k.clamp(0.0, 2.0);
            names.push(if (k - 1.0).abs() < 1e-3 {
                mood.name.clone()
            } else {
                format!("{} {:.0}%", mood.name, k * 100.0)
            });
            m.speed *= mood.speed.powf(k);
            m.expressiveness *= mood.expressiveness.powf(k);
            m.rhythm *= mood.rhythm.powf(k);
            m.pitch_range *= mood.pitch_range.powf(k);
            m.pause_scale *= mood.pause_scale.powf(k);
            m.pitch_st += mood.pitch_st * k;
            m.exaggeration += mood.exaggeration * k;
            m.pitch_slope_st += mood.pitch_slope_st * k;
            m.jitter_cents += mood.jitter_cents * k;
            m.vibrato_cents += mood.vibrato_cents * k;
            m.gain_db += mood.gain_db * k;
            m.brightness_db += mood.brightness_db * k;
            m.tremolo_depth += mood.tremolo_depth * k;
            keep[0] *= 1.0 - (mood.breathiness * k).clamp(0.0, 1.0);
            keep[1] *= 1.0 - (mood.whisper * k).clamp(0.0, 1.0);
            keep[2] *= 1.0 - (mood.reverb * k).clamp(0.0, 1.0);
            let vw = mood.vibrato_cents * k;
            if vw > 0.0 {
                vib_hz += mood.vibrato_hz * vw;
                vib_w += vw;
            }
            let tw = mood.tremolo_depth * k;
            if tw > 0.0 {
                trem_hz += mood.tremolo_hz * tw;
                trem_w += tw;
            }
        }
        m.breathiness = 1.0 - keep[0];
        m.whisper = 1.0 - keep[1];
        m.reverb = 1.0 - keep[2];
        m.tremolo_depth = m.tremolo_depth.clamp(0.0, 0.6);
        if vib_w > 0.0 {
            m.vibrato_hz = vib_hz / vib_w;
        }
        if trem_w > 0.0 {
            m.tremolo_hz = trem_hz / trem_w;
        }
        m.name = names.join("+");
        m
    }

    /// True when this mood needs the pitch/time (PSOLA) stage.
    pub fn needs_psola(&self) -> bool {
        self.pitch_st.abs() > 0.05
            || (self.pitch_range - 1.0).abs() > 0.02
            || self.pitch_slope_st.abs() > 0.05
            || self.jitter_cents > 0.5
            || self.vibrato_cents > 0.5
    }
}

macro_rules! mood {
    ($name:expr, $desc:expr, { $($field:ident : $val:expr),* $(,)? }) => {{
        #[allow(unused_mut)]
        let mut m = Mood { name: $name.into(), description: $desc.into(), ..Mood::default() };
        $( m.$field = $val; )*
        m
    }};
}

const PROMPTS: &[(&str, &str, &str)] = &[
    ("happy",
     "Oh, this is wonderful news! Really, it made my whole day, I'm smiling just thinking about it.",
     "Oh, c'est une merveilleuse nouvelle ! Vraiment, ça illumine ma journée, j'ai le sourire rien que d'y penser."),
    ("excited",
     "Oh my god, oh my god! We actually did it! This is incredible, I can't believe it!",
     "Oh là là, c'est pas possible ! On a réussi ! C'est incroyable, j'en reviens pas !"),
    ("sad",
     "I just... I really miss her. Everything feels so empty now, and I don't know what to do anymore.",
     "Elle me manque tellement... Tout me semble si vide maintenant, et je ne sais plus quoi faire."),
    ("angry",
     "No! This is completely unacceptable! I told you a hundred times, and you never listen to me!",
     "Non ! C'est complètement inacceptable ! Je te l'ai dit cent fois, et tu ne m'écoutes jamais !"),
    ("calm",
     "Take a slow, deep breath. Everything is fine. There's no rush, we have all the time we need.",
     "Respire lentement, profondément. Tout va bien. Rien ne presse, on a tout le temps qu'il faut."),
    ("fearful",
     "Did you hear that? There's something out there... please, please, let's just get out of here.",
     "Tu as entendu ça ? Il y a quelque chose dehors... s'il te plaît, partons d'ici, vite."),
    ("whisper",
     "Shh, keep your voice down. I'll tell you a secret, but nobody else can ever know.",
     "Chut, parle tout bas. Je vais te dire un secret, mais personne ne doit jamais savoir."),
    ("serious",
     "Listen to me carefully. What I'm about to say is important, and you need to take it seriously.",
     "Écoute-moi attentivement. Ce que je vais te dire est important, et tu dois le prendre au sérieux."),
    ("sarcastic",
     "Oh, wow. What a brilliant idea. I'm sure that will work out perfectly, just like last time.",
     "Oh, génial. Quelle idée brillante. Ça va sûrement marcher parfaitement, comme la dernière fois."),
    ("tired",
     "What a day... I barely slept last night. Can we just... do all of this tomorrow?",
     "Quelle journée... J'ai à peine dormi cette nuit. On peut... faire tout ça demain ?"),
    ("shouting",
     "Hey! Hey, over here! Can you hear me? Come back, right now!",
     "Hé ! Hé, par ici ! Tu m'entends ? Reviens, tout de suite !"),
    ("dramatic",
     "And on that night, everything changed. Nothing... would ever be the same again.",
     "Et cette nuit-là, tout a basculé. Plus rien... ne serait jamais comme avant."),
    ("cheerful",
     "Hi there! Thanks so much for calling, how can I help you today?",
     "Bonjour ! Merci beaucoup pour votre appel, comment puis-je vous aider aujourd'hui ?"),
    ("mysterious",
     "Some doors are better left closed... but you've already found the key, haven't you?",
     "Certaines portes devraient rester fermées... mais tu as déjà trouvé la clé, n'est-ce pas ?"),
];

/// Built-in mood presets.
pub fn builtin_moods() -> Vec<Mood> {
    let mut v = builtin_moods_raw();
    for m in v.iter_mut() {
        if let Some((_, en, fr)) = PROMPTS.iter().find(|(n, _, _)| *n == m.name) {
            m.prompt_en = en.to_string();
            m.prompt_fr = fr.to_string();
        }
    }
    v
}

fn builtin_moods_raw() -> Vec<Mood> {
    vec![
        mood!("neutral", "Plain, as the voice naturally reads", {}),
        mood!("happy", "Warm, smiling, lifted intonation", { exaggeration: 0.35,
            speed: 1.05, expressiveness: 1.12, pitch_st: 1.5, pitch_range: 1.3,
            brightness_db: 2.0, gain_db: 1.0, pause_scale: 0.9 }),
        mood!("excited", "High energy, fast, wide melodic range", { exaggeration: 0.75,
            speed: 1.14, expressiveness: 1.25, rhythm: 1.1, pitch_st: 3.0, pitch_range: 1.55,
            pitch_slope_st: 0.8, brightness_db: 3.5, gain_db: 2.5, pause_scale: 0.75 }),
        mood!("sad", "Slow, low, falling and darker", { exaggeration: 0.3,
            speed: 0.84, expressiveness: 0.8, pitch_st: -2.0, pitch_range: 0.6,
            pitch_slope_st: -1.5, brightness_db: -4.0, gain_db: -3.0, breathiness: 0.15,
            pause_scale: 1.45 }),
        mood!("angry", "Tense, loud, bright and clipped", { exaggeration: 0.85,
            speed: 1.07, expressiveness: 1.3, rhythm: 1.15, pitch_st: 0.8, pitch_range: 1.35,
            jitter_cents: 10.0, brightness_db: 5.5, gain_db: 3.5, pause_scale: 0.8 }),
        mood!("calm", "Relaxed, soft and steady", { exaggeration: -0.2,
            speed: 0.92, expressiveness: 0.85, rhythm: 0.9, pitch_st: -0.5, pitch_range: 0.8,
            brightness_db: -2.0, gain_db: -1.0, pause_scale: 1.25 }),
        mood!("fearful", "Shaky, breathy, higher pitched", { exaggeration: 0.55,
            speed: 1.1, expressiveness: 1.2, pitch_st: 2.5, pitch_range: 1.2, jitter_cents: 14.0,
            vibrato_hz: 6.5, vibrato_cents: 22.0, tremolo_hz: 7.0, tremolo_depth: 0.12,
            breathiness: 0.22, gain_db: -1.0 }),
        mood!("whisper", "Unvoiced, intimate whisper", { exaggeration: -0.1,
            speed: 0.95, whisper: 1.0, gain_db: -3.0, brightness_db: 1.0, pause_scale: 1.1 }),
        mood!("serious", "Authoritative, lower and measured", { exaggeration: -0.1,
            speed: 0.95, expressiveness: 0.9, pitch_st: -1.5, pitch_range: 0.85,
            brightness_db: -0.5, gain_db: 1.0, pause_scale: 1.15 }),
        mood!("sarcastic", "Exaggerated, drawn-out intonation", { exaggeration: 0.4,
            speed: 0.93, expressiveness: 1.15, pitch_range: 1.65, pitch_slope_st: 1.0,
            brightness_db: 1.0 }),
        mood!("tired", "Low energy, flat, breathy", { exaggeration: -0.2,
            speed: 0.86, expressiveness: 0.75, pitch_st: -1.5, pitch_range: 0.5,
            pitch_slope_st: -0.8, breathiness: 0.25, brightness_db: -3.0, gain_db: -2.0,
            pause_scale: 1.3 }),
        mood!("shouting", "Projected, loud and bright", { exaggeration: 1.1,
            speed: 1.0, expressiveness: 1.2, pitch_st: 3.5, pitch_range: 1.25,
            brightness_db: 6.5, gain_db: 5.0 }),
        mood!("dramatic", "Narration with weight, space and long pauses", { exaggeration: 0.45,
            speed: 0.9, expressiveness: 1.1, pitch_st: -0.5, pitch_range: 1.4,
            brightness_db: -0.5, reverb: 0.18, pause_scale: 1.6 }),
        mood!("cheerful", "Light, friendly customer-service tone", { exaggeration: 0.25,
            speed: 1.03, expressiveness: 1.05, pitch_st: 1.0, pitch_range: 1.18,
            brightness_db: 1.5, gain_db: 0.5 }),
        mood!("mysterious", "Low, slow, breathy with a hint of room", { exaggeration: 0.0,
            speed: 0.88, pitch_st: -1.0, pitch_range: 0.9, breathiness: 0.3,
            brightness_db: -2.5, reverb: 0.22, pause_scale: 1.4 }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mix_is_bounded() {
        let w = builtin_moods().into_iter().find(|m| m.name == "whisper").unwrap();
        let m = Mood::mix(&[(&w, 1.0), (&w, 1.0)]);
        assert!(m.whisper <= 1.0);
        let n = Mood::mix(&[]);
        assert!(!n.needs_psola());
    }
}
