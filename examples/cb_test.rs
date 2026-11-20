use std::time::Instant;
use vox::chatterbox::*;
fn main() {
    let dir = std::path::Path::new("models/chatterbox");
    let q = std::env::args().nth(1).unwrap_or("fast".into());
    let t0 = Instant::now();
    let mut cb = Chatterbox::load(dir, &q).unwrap();
    println!("load {:.1}s", t0.elapsed().as_secs_f32());
    let refw = read_wav_24k(&dir.join("default_voice.wav")).unwrap();
    let t0 = Instant::now();
    let v = cb.encode_voice(&refw).unwrap();
    println!("encode voice {:.2}s", t0.elapsed().as_secs_f32());
    for (lang, text, ex) in [("fr", "Bonjour, comment ça va ? Je suis vraiment contente de te voir aujourd'hui !", 0.5),
                             ("en", "Oh my god, I can't believe we actually won! This is the best day of my life!", 1.0)] {
        let t0 = Instant::now();
        let g = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let p = CbParams { exaggeration: ex, cfg_weight: g("CFG", 0.5), repetition_penalty: g("REP", 2.0), temperature: g("TEMP", 0.8), ..Default::default() };
        let w = cb.synth(text, lang, &v, &p).unwrap();
        let secs = w.len() as f32 / 24000.0;
        println!("{lang}: {:.2}s audio in {:.2}s", secs, t0.elapsed().as_secs_f32());
        let spec = hound::WavSpec { channels: 1, sample_rate: 24000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut wr = hound::WavWriter::create(format!("out/cb_{lang}_{q}{}.wav", std::env::var("TAG").unwrap_or_default()), spec).unwrap();
        for s in w { wr.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap(); }
        wr.finalize().unwrap();
    }
}
