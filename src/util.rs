//! Small numeric helpers shared across the crate.

/// Fast deterministic PRNG (xorshift64*), so renders are reproducible for a given seed.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[inline]
pub fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[inline]
pub fn lin_to_db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Parse durations like "500ms", "1.5s", "2 s", "300" (ms).
pub fn parse_duration_ms(s: &str) -> Option<f32> {
    let s = s.trim().to_ascii_lowercase();
    if let Some(v) = s.strip_suffix("ms") {
        return v.trim().parse().ok();
    }
    if let Some(v) = s.strip_suffix('s') {
        return v.trim().parse::<f32>().ok().map(|x| x * 1000.0);
    }
    s.parse().ok()
}

/// Parse "+2st", "-3", "+1.5 semitones" into semitones.
pub fn parse_semitones(s: &str) -> Option<f32> {
    let s = s.trim().to_ascii_lowercase();
    let s = s
        .trim_end_matches("semitones")
        .trim_end_matches("st")
        .trim();
    s.trim_start_matches('+').parse().ok()
}

/// Parse "+3dB", "-2", "loud"/"soft" into decibels.
pub fn parse_db(s: &str) -> Option<f32> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "x-soft" => return Some(-9.0),
        "soft" => return Some(-4.0),
        "medium" | "default" => return Some(0.0),
        "loud" => return Some(4.0),
        "x-loud" => return Some(8.0),
        _ => {}
    }
    s.trim_end_matches("db").trim().trim_start_matches('+').parse().ok()
}

/// Parse "1.2", "120%", "fast", "slow" into a rate multiplier.
pub fn parse_rate(s: &str) -> Option<f32> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "x-slow" => return Some(0.6),
        "slow" => return Some(0.8),
        "medium" | "default" => return Some(1.0),
        "fast" => return Some(1.2),
        "x-fast" => return Some(1.45),
        _ => {}
    }
    if let Some(p) = s.strip_suffix('%') {
        return p.trim().parse::<f32>().ok().map(|v| v / 100.0);
    }
    s.parse().ok()
}

/// Levenshtein distance, for "did you mean" suggestions.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c);
        }
        prev = cur;
    }
    prev[b.len()]
}

pub fn suggest<'a>(name: &str, options: impl Iterator<Item = &'a str>) -> String {
    let best = options
        .map(|o| (edit_distance(name, o), o))
        .min_by_key(|(d, _)| *d);
    match best {
        Some((d, o)) if d <= 3 => format!(" (did you mean '{o}'?)"),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parsing() {
        assert_eq!(parse_duration_ms("500ms"), Some(500.0));
        assert_eq!(parse_duration_ms("1.5s"), Some(1500.0));
        assert_eq!(parse_semitones("+2st"), Some(2.0));
        assert_eq!(parse_db("-3dB"), Some(-3.0));
        assert_eq!(parse_rate("120%"), Some(1.2));
        assert_eq!(edit_distance("hapy", "happy"), 1);
    }
}
