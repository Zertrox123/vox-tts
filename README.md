# vox: offline expressive neural TTS in Rust

Everything runs locally on CPU. There is no cloud API, and nothing touches the network after the one-time model download.

- **Expressive engine (default)**: Chatterbox Multilingual (Resemble AI, MIT), a 0.5B neural model with real emotion control. Its makers benchmark it against ElevenLabs.
  It reads plain French and English text directly, so no phonemizer is needed.
- **Lite engines**: Kokoro-82M and Piper (VITS). These are fast and low-latency, and apply moods as audio effects (DSP) only.
- **Personas**: 12 expressive voices and 18 lite voices: female, male and neutral, US English and French. You can **clone any voice** with `--clone voice.wav`.
- **Moods**: 15 presets, which you can mix with intensities (`happy:0.6+whisper:0.3`). On expressive voices, each mood combines:
  1. the model's emotion intensity (`exaggeration`),
  2. an **emotional voice prompt**: the persona first performs a line in that mood (e.g. "Oh my god, we actually did it!"). That clip, cached on disk, conditions the generation, so the emotion carries into your text.
  3. light prosody shaping of that prompt (tempo, intonation, whisper, breath).
- **Changing mood mid-sentence**: `Salut ! [excited] On a gagné ! [sad] Mais Julien s'est blessé.`
- **Export**: WAV (16/24/32-float), FLAC, MP3, OGG Vorbis, Opus, AIFF and raw PCM, at any sample rate, with SRT/VTT/JSON subtitles and a timeline.

## Setup

```bash
scripts/download_models.sh     # ~4.5 GB one time: Chatterbox, Kokoro, Piper, ONNX Runtime
cargo build --release
```

Optional extras:
- `brew install espeak-ng`: needed only for plain-text input with *lite* personas.
- `brew install opus`: needed for Opus export.

## Usage

```bash
vox personas
vox moods

vox say "Hello! [excited] We won the final! [sad] But Julien got hurt." -p emma -o out.mp3
vox say "Salut ! [excited] On a gagné la finale ! [sad] Mais Julien s'est blessé." -p thomas -o fr.wav -o fr.flac
vox say -f script.txt -p camille -m "calm:0.7" --format mp3,ogg --sample-rate 48000 --subtitles script.srt
vox say "This is my voice." --clone me.wav -o clone.wav     # voice cloning
vox say "..." --emotion 1.2                                 # global emotion intensity (0.25..2)
```

The first use of a persona+mood builds its emotional prompt (a few seconds of compute). It's cached in `models/chatterbox/voices/`, so later runs skip this step.

### Inline markup

| Syntax | Effect |
|---|---|
| `[happy]` `[sad 0.5]` `[angry:60%]` | switch mood for the following text |
| `[happy + whisper 0.3]` | mix moods |
| `[/]` | back to the default mood |
| `<sad intensity="0.7">…</sad>`, `<mood name="calm">…</mood>` | scoped mood |
| `*word*`, `<emph>…</emph>` | emphasis |
| `<prosody rate="1.2" pitch="+2st" volume="-3dB">…</prosody>` | fine control |
| `[pause 500ms]`, `<break time="1s"/>` | silence |
| `[breath]` `[laughs]` `[sighs]` | non-verbal sounds |
| `[voice pierre]`, `<voice name="camille">…</voice>` | change speaker (dialogue) |
| `<ipa>bɔ̃ʒuʁ</ipa>` | raw phonemes (lite personas) |

### Personas

| Expressive (Chatterbox) | Gender | Accent |
|---|---|---|
| emma, ava, bella | female | US |
| michael, ryan, fenrir | male | US |
| sam | neutral | US |
| camille, jessica | female | FR |
| thomas, pierre | male | FR |
| alex | neutral | FR |

Every persona also has a fast `-lite` version (Kokoro/Piper), e.g. `ava-lite` or `thomas-lite`, plus `nicole-lite`, `nova-lite`, `puck-lite`, `lessac-lite` and `siwis-lite`.
You can add your own personas, including your own reference WAV, in `vox.toml`; see `vox.example.toml`.

## How it works

```
markup → spans (persona, mood mix, prosody) → sentences → styled runs
expressive run:
  persona reference voice ──► emotional prompt (persona performs a mood line, cached)
                          ──► prosody/voice-quality shaping ──► speech encoder (cached)
  text ──► BPE ──► embeddings (+ exaggeration) ──► 0.5B Llama, KV cache, greedy + repetition penalty
       ──► speech tokens @25 Hz ──► flow-matching decoder + vocoder ──► 24 kHz audio
  quality gate: must stop on its own, plausible speaking rate, else retry with calmer settings
lite sentence: eSpeak IPA ──► Kokoro/Piper ──► TD-PSOLA intonation + LPC whisper/breath + EQ
both ──► explicit pitch/rate, level, room ──► stitch with mood-scaled pauses ──► loudness + limiter ──► encode
```

Code map: `chatterbox.rs` (expressive engine), `backend.rs` (Kokoro/Piper), `engine.rs` (pipeline),
`markup.rs`, `mood.rs`, `persona.rs`, `phonemize.rs`, `dsp/pitch.rs` (YIN + PSOLA), `dsp/effects.rs`, `export.rs`.

## Performance

On CPU, the expressive engine runs at roughly 0.3–0.5× real time on an Apple-silicon laptop. Model loading takes ~20 s per process, so render whole scripts in one `vox say -f` call.
Lite personas run at 5–13× real time.

## License

The code is MIT; see [LICENSE](LICENSE).

The models aren't in this repository. `scripts/download_models.sh` downloads them, and each keeps its own license:
- Chatterbox (Resemble AI): MIT
- Kokoro-82M: Apache-2.0
- ONNX Runtime: MIT
- eSpeak NG (optional): GPL-3.0, loaded at runtime and not bundled
- Piper voices: license depends on the voice, so check each voice's model card on Hugging Face before commercial use

Voice cloning: only clone voices you have the right to use.

## Building in a sandbox

If `~/.cargo` isn't writable, use `CARGO_HOME=$PWD/.cargo-home cargo build --release`.
`.cargo/config.toml` points `ORT_LIB_LOCATION` to `lib/onnxruntime`, which the setup script fills in.
