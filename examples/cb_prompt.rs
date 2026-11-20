use vox::chatterbox::*;
fn main() {
    let dir = std::path::Path::new("models/chatterbox");
    let mut cb = Chatterbox::load(dir, "high").unwrap();
    let r = read_wav_24k(&dir.join("default_voice.wav")).unwrap();
    let v = cb.encode_voice(&r[..240000.min(r.len())]).unwrap();
    let text = std::env::args().nth(1).unwrap();
    let ex: f32 = std::env::args().nth(2).unwrap().parse().unwrap();
    let p = CbParams { exaggeration: ex, cfg_weight: 0.0, temperature: 0.0, repetition_penalty: 1.2, ..Default::default() };
    let w = cb.synth(&text, "en", &v, &p).unwrap();
    println!("{:.2}s", w.len() as f32 / 24000.0);
    let spec = hound::WavSpec { channels: 1, sample_rate: 24000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut wr = hound::WavWriter::create("out/prompt_dbg.wav", spec).unwrap();
    for s in w { wr.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap(); }
    wr.finalize().unwrap();
}
