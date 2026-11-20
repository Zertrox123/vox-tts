//! Voice-quality effects and mastering. All "track" arguments are per-sample
//! control curves so effects can change smoothly mid-sentence.

use crate::util::{db_to_lin, Rng};
use std::f32::consts::PI;

/// LPC analysis/resynthesis with a noise excitation: keeps the vocal-tract
/// envelope (what is being said) but removes voicing -> whisper / aspiration.
pub fn lpc_noise(x: &[f32], sr: u32, seed: u64) -> Vec<f32> {
    let hop = ((sr as f32 * 0.0125) as usize).max(64);
    let n = hop * 2;
    let order = (sr / 1000 + 4) as usize;
    let win: Vec<f32> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos()).collect();
    let mut out = vec![0.0f32; x.len() + n];
    let mut rng = Rng::new(seed);
    // pre-emphasis helps LPC model the upper formants that carry whisper intelligibility
    let pre: Vec<f32> = (0..x.len()).map(|i| x[i] - if i > 0 { 0.92 * x[i - 1] } else { 0.0 }).collect();
    let mut frame = vec![0.0f32; n];
    let mut syn = vec![0.0f32; n];
    let mut start = 0;
    while start < x.len() {
        let mut energy = 0.0f32;
        for i in 0..n {
            let v = if start + i < x.len() { x[start + i] } else { 0.0 };
            energy += (v * win[i]).powi(2);
            frame[i] = if start + i < pre.len() { pre[start + i] } else { 0.0 } * win[i];
        }
        if energy > 1e-9 {
            let a = levinson(&autocorr(&frame, order), order);
            for i in 0..n {
                let mut y = rng.gauss();
                for k in 1..=order {
                    if i >= k {
                        y -= a[k] * syn[i - k];
                    }
                }
                syn[i] = y;
            }
            let mut se = 0.0f32;
            for i in 0..n {
                se += (syn[i] * win[i]).powi(2);
            }
            let g = if se > 1e-12 { (energy / se).sqrt() } else { 0.0 };
            for i in 0..n {
                out[start + i] += syn[i] * win[i] * g;
            }
        }
        start += hop;
    }
    out.truncate(x.len());
    // remove low rumble: whispers have no energy below ~300 Hz
    highpass(&mut out, sr, 300.0);
    out
}

fn autocorr(x: &[f32], order: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; order + 1];
    for (lag, rv) in r.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in lag..x.len() {
            s += x[i] * x[i - lag];
        }
        *rv = s;
    }
    r[0] *= 1.0 + 1e-4; // white-noise correction for stability
    r
}

/// Returns a[0..=order] with a[0] = 1 for A(z) = 1 + sum a_k z^-k.
fn levinson(r: &[f32], order: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; order + 1];
    a[0] = 1.0;
    let mut e = r[0];
    if e <= 0.0 {
        return a;
    }
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / e;
        let prev = a.clone();
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
        if e <= 1e-12 {
            break;
        }
    }
    a
}

pub fn highpass(x: &mut [f32], sr: u32, fc: f32) {
    let rc = 1.0 / (2.0 * PI * fc);
    let dt = 1.0 / sr as f32;
    let alpha = rc / (rc + dt);
    let (mut px, mut py) = (0.0f32, 0.0f32);
    for v in x.iter_mut() {
        let y = alpha * (py + *v - px);
        px = *v;
        py = y;
        *v = y;
    }
}

/// Spectral tilt EQ around ~1.2 kHz: +dB brighter, -dB darker.
pub fn tilt(x: &mut [f32], sr: u32, db: &dyn Fn(usize) -> f32) {
    let a = (-2.0 * PI * 1200.0 / sr as f32).exp();
    let mut lp = 0.0f32;
    for (i, v) in x.iter_mut().enumerate() {
        let d = db(i);
        if d.abs() < 0.05 {
            lp = (1.0 - a) * *v + a * lp;
            continue;
        }
        lp = (1.0 - a) * *v + a * lp;
        let hi = *v - lp;
        let g = db_to_lin(d / 2.0);
        *v = lp / g + hi * g;
    }
}

/// Small Schroeder/Freeverb-style mono room.
pub fn reverb(x: &[f32], sr: u32) -> Vec<f32> {
    let scale = sr as f32 / 44100.0;
    let combs = [1116, 1188, 1277, 1356];
    let allp = [556, 441];
    let mut y = vec![0.0f32; x.len() + (sr as usize) / 2];
    for &c in &combs {
        let d = ((c as f32) * scale) as usize;
        let mut buf = vec![0.0f32; d];
        let (mut idx, mut filt) = (0usize, 0.0f32);
        for i in 0..y.len() {
            let inp = if i < x.len() { x[i] } else { 0.0 };
            let out = buf[idx];
            filt = out * 0.6 + filt * 0.4; // damping
            buf[idx] = inp + filt * 0.78;
            idx = (idx + 1) % d;
            y[i] += out * 0.25;
        }
    }
    for &a in &allp {
        let d = ((a as f32) * scale) as usize;
        let mut buf = vec![0.0f32; d];
        let mut idx = 0usize;
        for v in y.iter_mut() {
            let b = buf[idx];
            let out = -*v + b;
            buf[idx] = *v + b * 0.5;
            idx = (idx + 1) % d;
            *v = out;
        }
    }
    y
}

/// Procedural inhale: band-limited noise with a breathy envelope.
pub fn breath(sr: u32, ms: f32, seed: u64) -> Vec<f32> {
    let n = (sr as f32 * ms / 1000.0) as usize;
    let mut rng = Rng::new(seed);
    let mut y: Vec<f32> = (0..n).map(|_| rng.gauss()).collect();
    // band-pass ~ 600..5000 Hz with two one-poles
    highpass(&mut y, sr, 600.0);
    let a = (-2.0 * PI * 5000.0 / sr as f32).exp();
    let mut lp = 0.0;
    for v in y.iter_mut() {
        lp = (1.0 - a) * *v + a * lp;
        *v = lp;
    }
    for (i, v) in y.iter_mut().enumerate() {
        let t = i as f32 / n as f32;
        let env = (PI * t).sin().powf(1.5) * (1.0 - 0.4 * t);
        *v *= env * 0.035;
    }
    y
}

/// Trim leading/trailing silence, keeping `margin_ms`. Returns (start, end) kept.
pub fn trim_bounds(x: &[f32], sr: u32, margin_ms: f32) -> (usize, usize) {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak <= 0.0 {
        return (0, 0);
    }
    let th = peak * db_to_lin(-42.0);
    let win = (sr / 100) as usize;
    let loud = |i: usize| x[i..(i + win).min(x.len())].iter().any(|v| v.abs() > th);
    let mut s = 0;
    while s < x.len() && !loud(s) {
        s += win;
    }
    let mut e = x.len();
    while e > s && !loud(e.saturating_sub(win)) {
        e = e.saturating_sub(win);
    }
    let m = (sr as f32 * margin_ms / 1000.0) as usize;
    (s.saturating_sub(m), (e + m).min(x.len()))
}

pub fn fade(x: &mut [f32], sr: u32, in_ms: f32, out_ms: f32) {
    let ni = ((sr as f32 * in_ms / 1000.0) as usize).min(x.len());
    for i in 0..ni {
        x[i] *= i as f32 / ni as f32;
    }
    let no = ((sr as f32 * out_ms / 1000.0) as usize).min(x.len());
    let l = x.len();
    for i in 0..no {
        x[l - 1 - i] *= i as f32 / no as f32;
    }
}

/// Gated RMS loudness normalisation to `target_db` dBFS, then a look-ahead limiter.
pub fn master(x: &mut [f32], sr: u32, target_db: f32, ceiling_db: f32) {
    if x.is_empty() {
        return;
    }
    // DC removal
    highpass(x, sr, 40.0);
    let win = (sr / 20) as usize;
    let frames: Vec<f32> = x.chunks(win).map(crate::util::rms).collect();
    let loudest = frames.iter().cloned().fold(0.0, f32::max);
    let gate = loudest * db_to_lin(-30.0);
    let voiced: Vec<f32> = frames.into_iter().filter(|r| *r > gate).collect();
    if !voiced.is_empty() {
        let cur = (voiced.iter().map(|r| r * r).sum::<f32>() / voiced.len() as f32).sqrt();
        let g = db_to_lin(target_db) / cur.max(1e-6);
        x.iter_mut().for_each(|v| *v *= g);
    }
    limiter(x, sr, ceiling_db);
}

pub fn limiter(x: &mut [f32], sr: u32, ceiling_db: f32) {
    let ceil = db_to_lin(ceiling_db);
    let look = (sr as f32 * 0.005) as usize;
    let rel = (-1.0 / (sr as f32 * 0.08)).exp();
    let n = x.len();
    // required gain per sample, min over look-ahead window
    let need: Vec<f32> = x.iter().map(|v| if v.abs() > ceil { ceil / v.abs() } else { 1.0 }).collect();
    let mut env = 1.0f32;
    let mut out = vec![0.0f32; n];
    let mut dq: std::collections::VecDeque<usize> = Default::default();
    for i in 0..n + look {
        if i < n {
            while let Some(&b) = dq.back() {
                if need[b] >= need[i] {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);
        }
        if i >= look {
            let j = i - look;
            while let Some(&f) = dq.front() {
                if f + look < j {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            let target = dq.front().map(|&f| need[f]).unwrap_or(1.0);
            env = if target < env { target } else { target + (env - target) * rel };
            out[j] = x[j] * env;
        }
    }
    for (v, o) in x.iter_mut().zip(out) {
        *v = o.clamp(-ceil, ceil);
    }
}

/// Windowed-sinc resampler (offline, high quality).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0) * 0.94;
    let half = (24.0 / cutoff).ceil() as isize;
    let out_len = (x.len() as f64 * ratio).ceil() as usize;
    let mut y = vec![0.0f32; out_len];
    for (i, yv) in y.iter_mut().enumerate() {
        let t = i as f64 / ratio;
        let c = t.floor() as isize;
        let mut acc = 0.0f64;
        for k in (c - half + 1)..=(c + half) {
            if k < 0 || k as usize >= x.len() {
                continue;
            }
            let d = t - k as f64;
            let u = d / half as f64;
            if u.abs() >= 1.0 {
                continue;
            }
            let w = 0.42 + 0.5 * (std::f64::consts::PI * u).cos() + 0.08 * (2.0 * std::f64::consts::PI * u).cos();
            let arg = std::f64::consts::PI * cutoff * d;
            let s = if arg.abs() < 1e-9 { 1.0 } else { arg.sin() / arg };
            acc += x[k as usize] as f64 * s * w * cutoff;
        }
        *yv = acc as f32;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resample_len_and_tone() {
        let sr = 24000;
        let x: Vec<f32> = (0..sr).map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin()).collect();
        let y = resample(&x, sr, 48000);
        assert_eq!(y.len(), 48000);
        let r = crate::util::rms(&y[1000..47000]);
        assert!((r - 0.707).abs() < 0.02, "rms {r}");
    }
    #[test]
    fn limiter_caps() {
        let mut x = vec![0.0f32; 1000];
        x[500] = 3.0;
        limiter(&mut x, 24000, -1.0);
        assert!(x.iter().all(|v| v.abs() <= db_to_lin(-1.0) + 1e-6));
    }
    #[test]
    fn whisper_keeps_energy() {
        let sr = 22050;
        let x: Vec<f32> = (0..sr).map(|i| (2.0 * PI * 140.0 * i as f32 / sr as f32).sin() * 0.3).collect();
        let w = lpc_noise(&x, sr, 1);
        let (a, b) = (crate::util::rms(&x), crate::util::rms(&w));
        assert!(b > a * 0.2 && b < a * 2.0, "{a} {b}");
    }
}
