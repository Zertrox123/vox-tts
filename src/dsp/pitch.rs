//! F0 tracking (YIN) and TD-PSOLA pitch / time modification.
//!
//! PSOLA moves pitch periods instead of resampling, so formants (the timbre of the
//! voice) are preserved: a +3 semitone "excited" lift doesn't sound like a chipmunk.

/// F0 track sampled every `hop` samples; 0.0 = unvoiced.
pub struct F0Track {
    pub hop: usize,
    pub f0: Vec<f32>,
}

impl F0Track {
    pub fn at(&self, sample: usize) -> f32 {
        if self.f0.is_empty() {
            return 0.0;
        }
        self.f0[(sample / self.hop).min(self.f0.len() - 1)]
    }

    /// Mean of log-F0 over voiced frames (None if fully unvoiced).
    pub fn log_mean(&self) -> Option<f32> {
        let v: Vec<f32> = self.f0.iter().filter(|f| **f > 0.0).map(|f| f.ln()).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }
}

pub fn yin(x: &[f32], sr: u32) -> F0Track {
    let hop = (sr / 200) as usize; // 5 ms
    let fmin = 60.0f32;
    let fmax = 500.0f32;
    let tau_min = (sr as f32 / fmax) as usize;
    let tau_max = (sr as f32 / fmin) as usize;
    let w = tau_max;
    let n_frames = x.len() / hop + 1;
    let peak_rms = {
        let mut m = 0.0f32;
        let mut i = 0;
        while i + w <= x.len() {
            m = m.max(crate::util::rms(&x[i..i + w]));
            i += hop * 4;
        }
        m.max(1e-6)
    };
    let mut f0 = vec![0.0f32; n_frames];
    let mut d = vec![0.0f32; tau_max + 2];
    for (fi, out) in f0.iter_mut().enumerate() {
        let center = fi * hop;
        let start = center.saturating_sub(w / 2);
        if start + w + tau_max + 1 >= x.len() {
            continue;
        }
        let frame = &x[start..start + w];
        if crate::util::rms(frame) < peak_rms * 0.03 {
            continue;
        }
        d[0] = 1.0;
        let mut running = 0.0f32;
        for tau in 1..=tau_max {
            let mut s = 0.0f32;
            let a = &x[start..start + w];
            let b = &x[start + tau..start + tau + w];
            for j in 0..w {
                let e = a[j] - b[j];
                s += e * e;
            }
            running += s;
            d[tau] = if running > 0.0 { s * tau as f32 / running } else { 1.0 };
        }
        let mut best = None;
        let mut tau = tau_min;
        while tau < tau_max {
            if d[tau] < 0.15 {
                while tau + 1 < tau_max && d[tau + 1] < d[tau] {
                    tau += 1;
                }
                best = Some(tau);
                break;
            }
            tau += 1;
        }
        if let Some(t) = best {
            // parabolic interpolation
            let (a, b, c) = (d[t - 1], d[t], d[t + 1]);
            let den = a - 2.0 * b + c;
            let shift = if den.abs() > 1e-9 { 0.5 * (a - c) / den } else { 0.0 };
            let tf = t as f32 + shift.clamp(-1.0, 1.0);
            *out = sr as f32 / tf;
        }
    }
    // 5-point median smoothing of voiced runs, drop very short voiced islands
    let raw = f0.clone();
    for i in 0..f0.len() {
        if raw[i] <= 0.0 {
            continue;
        }
        let mut win: Vec<f32> = (i.saturating_sub(2)..(i + 3).min(raw.len())).map(|j| raw[j]).filter(|v| *v > 0.0).collect();
        win.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f0[i] = win[win.len() / 2];
    }
    let mut i = 0;
    while i < f0.len() {
        if f0[i] > 0.0 {
            let s = i;
            while i < f0.len() && f0[i] > 0.0 {
                i += 1;
            }
            if i - s < 4 {
                f0[s..i].iter_mut().for_each(|v| *v = 0.0);
            }
        } else {
            i += 1;
        }
    }
    F0Track { hop, f0 }
}

#[derive(Clone, Copy, Debug)]
pub struct Mark {
    pub pos: usize,
    pub period: usize,
    pub voiced: bool,
}

/// Place analysis pitch marks on signal peaks, one per period in voiced regions.
pub fn pitch_marks(x: &[f32], f0: &F0Track, sr: u32) -> Vec<Mark> {
    let unv = (sr / 200) as usize; // 5 ms grains in unvoiced regions
    let mut marks = Vec::new();
    let mut t = 0usize;
    let mut prev_voiced = false;
    while t < x.len() {
        let f = f0.at(t);
        if f > 0.0 {
            let p = (sr as f32 / f).round().max(8.0) as usize;
            let (lo, hi) = if prev_voiced {
                (t.saturating_sub(p / 4), (t + p / 4).min(x.len() - 1))
            } else {
                (t, (t + p).min(x.len() - 1))
            };
            let mut c = lo;
            for j in lo..=hi {
                if x[j] > x[c] {
                    c = j;
                }
            }
            if let Some(last) = marks.last() {
                let last: &Mark = last;
                if c <= last.pos + p / 2 {
                    c = last.pos + p;
                }
            }
            if c >= x.len() {
                break;
            }
            marks.push(Mark { pos: c, period: p, voiced: true });
            t = c + p;
            prev_voiced = true;
        } else {
            marks.push(Mark { pos: t, period: unv, voiced: false });
            t += unv;
            prev_voiced = false;
        }
    }
    // Use the true spacing to the next mark as the period, so synthesis marks land
    // exactly on analysis marks when no modification is requested (transparent PSOLA).
    for i in 0..marks.len().saturating_sub(1) {
        if marks[i].voiced && marks[i + 1].voiced {
            let d = marks[i + 1].pos - marks[i].pos;
            let p = marks[i].period;
            if d * 10 > p * 6 && d * 10 < p * 16 {
                marks[i].period = d;
            }
        }
    }
    marks
}

/// TD-PSOLA. `pitch(in_pos, f0)` returns the pitch factor (2.0 = one octave up),
/// `stretch(in_pos)` returns the local duration factor (1.2 = 20% longer).
/// Returns the output and anchors (out_pos, in_pos) for mapping timelines.
pub fn psola(
    x: &[f32],
    marks: &[Mark],
    sr: u32,
    mut pitch: impl FnMut(usize, f32) -> f32,
    stretch: impl Fn(usize) -> f32,
) -> (Vec<f32>, Vec<(f64, f64)>) {
    if marks.is_empty() || x.len() < 64 {
        return (x.to_vec(), vec![(0.0, 0.0), (x.len() as f64, x.len() as f64)]);
    }
    let est: f64 = (0..x.len()).step_by(64).map(|i| stretch(i) as f64 * 64.0).sum();
    let cap = (est * 1.1) as usize + 4 * sr as usize / 50 + x.len() / 10 + 1024;
    let mut out = vec![0.0f32; cap];
    let mut wsum = vec![0.0f32; cap];
    let mut anchors = Vec::new();
    let mut s = marks[0].pos as f64; // synthesis time
    let mut tau = marks[0].pos as f64; // input time
    anchors.push((0.0, 0.0));
    let mut mi = 0usize;
    let mut prev_voiced = false;
    // Hann window cache by length
    let mut win_cache: std::collections::HashMap<usize, Vec<f32>> = Default::default();
    loop {
        while mi + 1 < marks.len() && (marks[mi + 1].pos as f64) <= tau {
            mi += 1;
        }
        let k = if mi + 1 < marks.len() && (marks[mi + 1].pos as f64 - tau) < (tau - marks[mi].pos as f64) {
            mi + 1
        } else {
            mi
        };
        let m = marks[k];
        let p = m.period;
        // entering a voiced region: re-align both clocks on the first pitch mark so
        // voiced grains line up with the unvoiced material around them
        if m.voiced && !prev_voiced {
            let delta = m.pos as f64 - tau;
            s += delta * stretch(tau.max(0.0) as usize).clamp(0.25, 4.0) as f64;
            tau = m.pos as f64;
        }
        prev_voiced = m.voiced;
        // voiced: grain centred on the pitch mark; unvoiced: centred exactly at tau
        // (avoids periodic repetition buzz when stretching noise)
        let center = if m.voiced { m.pos as isize } else { tau.round() as isize };
        let factor = if m.voiced { pitch(m.pos, sr as f32 / p as f32).clamp(0.5, 2.0) } else { 1.0 };
        let win = win_cache.entry(2 * p + 1).or_insert_with(|| {
            let n = 2 * p + 1;
            (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()).collect()
        });
        let so = s.round() as isize;
        for (j, w) in win.iter().enumerate() {
            let xi = center - p as isize + j as isize;
            let oi = so - p as isize + j as isize;
            if xi < 0 || xi as usize >= x.len() || oi < 0 || oi as usize >= cap {
                continue;
            }
            out[oi as usize] += x[xi as usize] * w;
            wsum[oi as usize] += w;
        }
        anchors.push((s, tau));
        let step_out = p as f64 / factor as f64;
        s += step_out;
        tau += step_out / stretch(tau as usize).clamp(0.25, 4.0) as f64;
        if tau >= x.len() as f64 || s as usize + p + 1 >= cap {
            break;
        }
    }
    let end = (s as usize).min(cap);
    for i in 0..end {
        out[i] /= wsum[i].max(0.3);
    }
    out.truncate(end);
    anchors.push((end as f64, x.len() as f64));
    (out, anchors)
}

/// Interpolate a monotonic anchor list: map `a` -> `b`.
pub fn map_time(anchors: &[(f64, f64)], t: f64, inverse: bool) -> f64 {
    let key = |p: &(f64, f64)| if inverse { p.1 } else { p.0 };
    let val = |p: &(f64, f64)| if inverse { p.0 } else { p.1 };
    let idx = anchors.partition_point(|p| key(p) <= t);
    if idx == 0 {
        return val(&anchors[0]);
    }
    if idx >= anchors.len() {
        return val(anchors.last().unwrap());
    }
    let (a, b) = (&anchors[idx - 1], &anchors[idx]);
    let span = key(b) - key(a);
    if span <= 0.0 {
        return val(a);
    }
    val(a) + (t - key(a)) / span * (val(b) - val(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tone(f: f32, sr: u32, secs: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                // glottal-ish pulse train: a few harmonics
                (1..6).map(|h| (2.0 * std::f32::consts::PI * f * h as f32 * t).sin() / h as f32).sum::<f32>() * 0.3
            })
            .collect()
    }

    #[test]
    fn yin_detects_pitch() {
        let x = tone(150.0, 22050, 0.5);
        let tr = yin(&x, 22050);
        let mid = tr.f0[tr.f0.len() / 2];
        assert!((mid - 150.0).abs() < 3.0, "f0 {mid}");
    }

    #[test]
    fn psola_shifts_pitch_keeps_duration() {
        let sr = 22050;
        let x = tone(150.0, sr, 0.6);
        let tr = yin(&x, sr);
        let marks = pitch_marks(&x, &tr, sr);
        let factor = 2f32.powf(4.0 / 12.0);
        let (y, _) = psola(&x, &marks, sr, |_, _| factor, |_| 1.0);
        assert!((y.len() as f32 / x.len() as f32 - 1.0).abs() < 0.05);
        let ty = yin(&y, sr);
        let mid = ty.f0[ty.f0.len() / 2];
        assert!((mid - 150.0 * factor).abs() < 6.0, "shifted f0 {mid}");
    }

    #[test]
    fn psola_stretches() {
        let sr = 22050;
        let x = tone(120.0, sr, 0.5);
        let tr = yin(&x, sr);
        let marks = pitch_marks(&x, &tr, sr);
        let (y, _) = psola(&x, &marks, sr, |_, _| 1.0, |_| 1.5);
        let r = y.len() as f32 / x.len() as f32;
        assert!((r - 1.5).abs() < 0.08, "ratio {r}");
    }
}
