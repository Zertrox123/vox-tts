use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
use vox::engine::{parse_mood_arg, Engine, RenderOptions, UserConfig};
use vox::export::{export, write_subtitles, ExportOptions, Format};

#[derive(Parser)]
#[command(name = "vox", version, about = "Offline expressive neural text-to-speech", long_about = None)]
struct Cli {
    /// Models directory (default: $VOX_MODELS or ./models)
    #[arg(long, global = true)]
    models: Option<PathBuf>,
    /// Custom personas / moods file (default: ./vox.toml if present)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Synthesize speech from text or a script file
    Say(SayArgs),
    /// List personas (voices)
    Personas,
    /// List mood presets
    Moods,
    /// Render a sample for every persona and mood
    Demo {
        #[arg(short, long, default_value = "out/demo")]
        out_dir: PathBuf,
        #[arg(long, default_value = "mp3")]
        format: String,
    },
    /// Show text -> phoneme conversion
    Phonemize {
        text: String,
        #[arg(short, long, default_value = "en-us")]
        accent: String,
    },
    /// Show an ONNX model's inputs/outputs
    Inspect { model: PathBuf },
}

#[derive(Args)]
struct SayArgs {
    /// Text with optional inline tags, e.g. "Hi! [sad] I have to go."
    text: Option<String>,
    /// Read text from a file ('-' = stdin)
    #[arg(short, long)]
    file: Option<PathBuf>,
    #[arg(short, long, default_value = "emma")]
    persona: String,
    /// Emotion intensity for expressive personas, 0.25 (flat) .. 2.0 (very dramatic); overrides moods
    #[arg(long)]
    emotion: Option<f32>,
    /// Clone a voice from a WAV file (5-10 s of clean speech) for expressive personas
    #[arg(long)]
    clone: Option<PathBuf>,
    /// Print progress
    #[arg(short, long)]
    verbose: bool,
    /// Default mood mix, e.g. "happy", "sad:0.5", "calm+whisper:0.3"
    #[arg(short, long)]
    mood: Option<String>,
    /// Speaking rate multiplier
    #[arg(long, default_value_t = 1.0)]
    rate: f32,
    /// Pitch offset in semitones
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pitch: f32,
    /// Volume offset in dB
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    volume: f32,
    /// Output file(s); format from extension. Repeat for several formats.
    #[arg(short, long)]
    out: Vec<PathBuf>,
    /// Output format(s) when no extension / no --out: wav, flac, mp3, ogg, opus, aiff, raw, pcm
    #[arg(long, value_delimiter = ',')]
    format: Vec<String>,
    #[arg(long, default_value_t = 24000)]
    sample_rate: u32,
    /// 16, 24 or 32 (float) for WAV/FLAC
    #[arg(long, default_value_t = 16)]
    bit_depth: u16,
    #[arg(long, default_value_t = 192)]
    mp3_kbps: u32,
    #[arg(long, default_value_t = 0.6)]
    ogg_quality: f32,
    #[arg(long, default_value_t = 64)]
    opus_kbps: u32,
    /// Subtitles: .srt, .vtt, or .json (sentence level)
    #[arg(long)]
    subtitles: Option<PathBuf>,
    /// JSON timeline of every styled run (persona, mood, start/end)
    #[arg(long)]
    timeline: Option<PathBuf>,
    /// Input is IPA phonemes (no eSpeak needed)
    #[arg(long)]
    ipa: bool,
    /// Pause between sentences, ms
    #[arg(long, default_value_t = 380.0)]
    pause: f32,
    /// Target loudness (gated RMS dBFS)
    #[arg(long, default_value_t = -18.0, allow_hyphen_values = true)]
    loudness: f32,
    #[arg(long)]
    no_normalize: bool,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

fn models_dir(cli: &Option<PathBuf>) -> PathBuf {
    if let Some(p) = cli {
        return p.clone();
    }
    if let Ok(p) = std::env::var("VOX_MODELS") {
        return PathBuf::from(p);
    }
    let local = PathBuf::from("models");
    if local.exists() {
        return local;
    }
    if let Ok(exe) = std::env::current_exe() {
        for a in exe.ancestors().skip(1).take(4) {
            if a.join("models").exists() {
                return a.join("models");
            }
        }
    }
    local
}

fn engine(cli: &Cli) -> Result<Engine> {
    let cfg_path = cli.config.clone().or_else(|| Some(PathBuf::from("vox.toml")).filter(|p| p.exists()));
    let user = match cfg_path {
        Some(p) => Some(UserConfig::load(&p)?),
        None => None,
    };
    Ok(Engine::new(models_dir(&cli.models), user))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Personas => {
            let e = engine(&cli)?;
            println!("{:<13} {:<8} {:<16} {:<10} {:<34} {}", "NAME", "GENDER", "ACCENT", "ENGINE", "VOICE", "DESCRIPTION");
            for p in e.personas() {
                let mark = if e.available(p) { "" } else { "  [model missing]" };
                println!(
                    "{:<13} {:<8} {:<16} {:<10} {:<34} {}{}",
                    p.name,
                    p.gender.to_string(),
                    p.accent.label(),
                    p.voice.engine(),
                    p.voice.describe(),
                    p.description,
                    mark
                );
            }
        }
        Cmd::Moods => {
            let e = engine(&cli)?;
            for m in e.moods() {
                println!("{:<11} {}", m.name, m.description);
            }
            println!("\nMix with '+', scale with an intensity: --mood \"happy:0.6+whisper:0.3\" or inline [sad 40%]");
        }
        Cmd::Phonemize { text, accent } => {
            let p = vox::phonemize::Phonemizer::new()?;
            eprintln!("using {}", p.source);
            println!("{}", p.phonemize(text, accent)?);
        }
        Cmd::Inspect { model } => print!("{}", vox::backend::describe_model(model)?),
        Cmd::Say(a) => say(&cli, a)?,
        Cmd::Demo { out_dir, format } => demo(&cli, out_dir, format)?,
    }
    Ok(())
}

fn say(cli: &Cli, a: &SayArgs) -> Result<()> {
    let text = match (&a.text, &a.file) {
        (_, Some(f)) if f.as_os_str() == "-" => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
        (_, Some(f)) => std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
        (Some(t), None) => t.clone(),
        (None, None) => bail!("give text or --file"),
    };
    let mut e = engine(cli)?;
    let opt = RenderOptions {
        persona: a.persona.clone(),
        moods: match &a.mood {
            Some(m) => parse_mood_arg(m)?,
            None => vec![],
        },
        rate: a.rate,
        pitch_st: a.pitch,
        volume_db: a.volume,
        sample_rate: a.sample_rate,
        sentence_pause_ms: a.pause,
        normalize: !a.no_normalize,
        target_db: a.loudness,
        seed: a.seed,
        ipa: a.ipa,
        emotion: a.emotion,
        clone: a.clone.clone(),
    };
    e.verbose = a.verbose;
    let t0 = Instant::now();
    let r = e.render(&text, &opt)?;
    let secs = r.samples.len() as f64 / r.sample_rate as f64;
    let elapsed = t0.elapsed().as_secs_f64();

    // outputs: explicit paths, plus --format variants
    let formats: Vec<Format> = a.format.iter().map(|f| Format::parse(f)).collect::<Result<_>>()?;
    let mut outs: Vec<(PathBuf, Format)> = Vec::new();
    for p in &a.out {
        match Format::from_path(p) {
            Some(f) => outs.push((p.clone(), f)),
            None => {
                let fs = if formats.is_empty() { vec![Format::Wav] } else { formats.clone() };
                for f in fs {
                    outs.push((p.with_extension(f.ext()), f));
                }
            }
        }
    }
    if a.out.is_empty() {
        let fs = if formats.is_empty() { vec![Format::Wav] } else { formats.clone() };
        for f in fs {
            outs.push((PathBuf::from(format!("out.{}", f.ext())), f));
        }
    }
    let eo = ExportOptions {
        bit_depth: a.bit_depth,
        mp3_kbps: a.mp3_kbps,
        ogg_quality: a.ogg_quality,
        opus_kbps: a.opus_kbps,
        title: Some(text.chars().take(60).collect()),
    };
    for (p, f) in &outs {
        export(p, &r.samples, r.sample_rate, *f, &eo)?;
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("wrote {} ({}, {:.1} KB)", p.display(), f.ext(), size as f64 / 1024.0);
    }
    if let Some(s) = &a.subtitles {
        write_subtitles(s, &r.sentences)?;
        println!("wrote {}", s.display());
    }
    if let Some(t) = &a.timeline {
        std::fs::write(t, serde_json::to_string_pretty(&r.runs)?)?;
        println!("wrote {}", t.display());
    }
    eprintln!("{:.2}s of audio in {:.2}s ({:.1}x real-time)", secs, elapsed, secs / elapsed.max(1e-6));
    Ok(())
}

fn demo(cli: &Cli, dir: &Path, format: &str) -> Result<()> {
    let f = Format::parse(format)?;
    let mut e = engine(cli)?;
    e.verbose = true;
    let personas: Vec<_> = e
        .personas()
        .iter()
        .filter(|p| e.available(p) && p.voice.engine() == "chatterbox")
        .cloned()
        .collect();
    let moods: Vec<String> = e.moods().iter().map(|m| m.name.clone()).collect();
    std::fs::create_dir_all(dir)?;
    let eo = ExportOptions::default();
    for p in &personas {
        let text = if p.accent.is_french() {
            format!("Bonjour, je m'appelle {}. [excited] Quelle belle journée ! [sad] Mais demain, il va pleuvoir.", p.name)
        } else {
            format!("Hi, I'm {}. [excited] What a beautiful day! [sad] But tomorrow, it's going to rain.", p.name)
        };
        let opt = RenderOptions { persona: p.name.clone(), ..Default::default() };
        let r = e.render(&text, &opt)?;
        let path = dir.join(format!("persona_{}.{}", p.name, f.ext()));
        export(&path, &r.samples, r.sample_rate, f, &eo)?;
        println!("wrote {}", path.display());
    }
    let narrator = personas.first().map(|p| p.name.clone()).unwrap_or_else(|| "emma".into());
    for m in &moods {
        let text = "I can't believe it's finally here. We waited so long for this moment.";
        let opt = RenderOptions { persona: narrator.clone(), moods: vec![(m.clone(), 1.0)], ..Default::default() };
        let r = e.render(text, &opt)?;
        let path = dir.join(format!("mood_{m}.{}", f.ext()));
        export(&path, &r.samples, r.sample_rate, f, &eo)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
