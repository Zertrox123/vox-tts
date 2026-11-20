fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut r = hound::WavReader::open(&a[1]).unwrap();
    let sr = r.spec().sample_rate as usize;
    let x: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let f = vox::dsp::pitch::yin(&x, sr as u32);
    for (i, c) in x.chunks(sr / 4).enumerate() {
        let fr: Vec<f32> = f.f0[i * 50..((i + 1) * 50).min(f.f0.len())].iter().cloned().filter(|v| *v > 0.0).collect();
        let f0 = if fr.is_empty() { 0.0 } else { fr.iter().sum::<f32>() / fr.len() as f32 };
        print!("{:.2}s:{:.0}dB/{:.0}Hz  ", i as f32 / 4.0, vox::util::lin_to_db(vox::util::rms(c)), f0);
    }
    println!();
}
