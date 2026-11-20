use vox::dsp::pitch::yin;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut r = hound::WavReader::open(&a[1]).unwrap();
    let sr = r.spec().sample_rate;
    let x: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let f = yin(&x, sr);
    let per = 20; // 100ms
    for (i, c) in f.f0.chunks(per).enumerate() {
        let v: Vec<f32> = c.iter().cloned().filter(|v| *v > 0.0).collect();
        let m = if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
        print!("{:.1}:{:.0} ", i as f32 * 0.1, m);
    }
    println!();
}
