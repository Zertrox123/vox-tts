//! Dev tool: per-run pitch / loudness report for a render.
//! cargo run --release --example analyze -- out.wav out.json
use vox::dsp::pitch::yin;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut r = hound::WavReader::open(&a[1]).unwrap();
    let sr = r.spec().sample_rate;
    let x: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let cues: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    println!("{:<12} {:>7} {:>9} {:>8} {:>8}  text", "mood", "f0(Hz)", "f0sd(st)", "rms(dB)", "dur(s)");
    for c in cues {
        let s = (c["start"].as_f64().unwrap() * sr as f64) as usize;
        let e = ((c["end"].as_f64().unwrap() * sr as f64) as usize).min(x.len());
        let seg = &x[s..e];
        let f = yin(seg, sr);
        let v: Vec<f32> = f.f0.iter().cloned().filter(|v| *v > 0.0).collect();
        let f0 = if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
        let lm: Vec<f32> = v.iter().map(|x| 12.0 * (x / f0.max(1.0)).log2()).collect();
        let sd = if lm.is_empty() { 0.0 } else { (lm.iter().map(|x| x * x).sum::<f32>() / lm.len() as f32).sqrt() };
        println!("{:<12} {:>7.1} {:>9.2} {:>8.1} {:>8.2}  {}", c["mood"].as_str().unwrap(), f0, sd,
            vox::util::lin_to_db(vox::util::rms(seg)), (e - s) as f32 / sr as f32, c["text"].as_str().unwrap());
    }
}
