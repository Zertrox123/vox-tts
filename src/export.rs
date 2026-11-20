//! Audio export: WAV, FLAC, MP3, OGG Vorbis, Opus, AIFF, raw PCM.
//! Plus subtitle / timeline sidecars (SRT, WebVTT, JSON).

use crate::dsp::effects::resample;
use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wav,
    Flac,
    Mp3,
    Ogg,
    Opus,
    Aiff,
    /// 32-bit float little-endian, headerless
    RawF32,
    /// 16-bit signed little-endian, headerless
    Pcm16,
}

impl Format {
    pub const ALL: &'static [(&'static str, Format)] = &[
        ("wav", Format::Wav),
        ("flac", Format::Flac),
        ("mp3", Format::Mp3),
        ("ogg", Format::Ogg),
        ("opus", Format::Opus),
        ("aiff", Format::Aiff),
        ("raw", Format::RawF32),
        ("pcm", Format::Pcm16),
    ];

    pub fn parse(s: &str) -> Result<Format> {
        let s = s.trim().trim_start_matches('.').to_ascii_lowercase();
        let s = match s.as_str() {
            "wave" => "wav",
            "aif" => "aiff",
            "oga" | "vorbis" => "ogg",
            "f32" => "raw",
            "s16" | "s16le" => "pcm",
            x => x,
        };
        Format::ALL
            .iter()
            .find(|(n, _)| *n == s)
            .map(|(_, f)| *f)
            .ok_or_else(|| anyhow!("unknown format '{s}' (supported: {})", Self::names()))
    }

    pub fn from_path(p: &Path) -> Option<Format> {
        p.extension().and_then(|e| e.to_str()).and_then(|e| Format::parse(e).ok())
    }

    pub fn names() -> String {
        Self::ALL.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
    }

    pub fn ext(&self) -> &'static str {
        Self::ALL.iter().find(|(_, f)| f == self).unwrap().0
    }
}

#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// 16, 24 or 32 (32 = float for WAV)
    pub bit_depth: u16,
    pub mp3_kbps: u32,
    /// Vorbis quality -0.1 .. 1.0
    pub ogg_quality: f32,
    pub opus_kbps: u32,
    pub title: Option<String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions { bit_depth: 16, mp3_kbps: 192, ogg_quality: 0.6, opus_kbps: 64, title: None }
    }
}

fn to_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

fn to_i24(x: f32) -> i32 {
    (x.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32
}

pub fn export(path: &Path, samples: &[f32], sr: u32, fmt: Format, opt: &ExportOptions) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let bytes = encode(samples, sr, fmt, opt)?;
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

pub fn encode(samples: &[f32], sr: u32, fmt: Format, opt: &ExportOptions) -> Result<Vec<u8>> {
    match fmt {
        Format::Wav => wav(samples, sr, opt.bit_depth),
        Format::Flac => flac(samples, sr, opt.bit_depth),
        Format::Mp3 => mp3(samples, sr, opt.mp3_kbps),
        Format::Ogg => vorbis(samples, sr, opt.ogg_quality, opt.title.as_deref()),
        Format::Opus => opus(samples, sr, opt.opus_kbps, opt.title.as_deref()),
        Format::Aiff => Ok(aiff(samples, sr)),
        Format::RawF32 => Ok(samples.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Format::Pcm16 => Ok(samples.iter().flat_map(|v| to_i16(*v).to_le_bytes()).collect()),
    }
}

fn wav(samples: &[f32], sr: u32, bits: u16) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: match bits {
            24 => 24,
            32 => 32,
            _ => 16,
        },
        sample_format: if bits == 32 { hound::SampleFormat::Float } else { hound::SampleFormat::Int },
    };
    let mut cur = std::io::Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(&mut cur, spec)?;
        for &s in samples {
            match spec.bits_per_sample {
                32 => w.write_sample(s.clamp(-1.0, 1.0))?,
                24 => w.write_sample(to_i24(s))?,
                _ => w.write_sample(to_i16(s))?,
            }
        }
        w.finalize()?;
    }
    Ok(cur.into_inner())
}

fn flac(samples: &[f32], sr: u32, bits: u16) -> Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    let bits = if bits >= 24 { 24 } else { 16 };
    let ints: Vec<i32> = samples.iter().map(|&s| if bits == 24 { to_i24(s) } else { to_i16(s) as i32 }).collect();
    let config = flacenc::config::Encoder::default().into_verified().map_err(|e| anyhow!("flac config: {e:?}"))?;
    let source = flacenc::source::MemSource::from_samples(&ints, 1, bits, sr as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow!("flac encode: {e:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).map_err(|e| anyhow!("flac write: {e:?}"))?;
    Ok(sink.as_slice().to_vec())
}

fn mp3(samples: &[f32], sr: u32, kbps: u32) -> Result<Vec<u8>> {
    use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality};
    // MPEG-1/2 layer III supports these rates; resample otherwise
    let ok = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000];
    let (pcm, rate) = if ok.contains(&sr) { (samples.to_vec(), sr) } else { (resample(samples, sr, 44100), 44100) };
    let br = match kbps {
        0..=40 => Bitrate::Kbps32,
        41..=56 => Bitrate::Kbps48,
        57..=72 => Bitrate::Kbps64,
        73..=88 => Bitrate::Kbps80,
        89..=104 => Bitrate::Kbps96,
        105..=120 => Bitrate::Kbps112,
        121..=144 => Bitrate::Kbps128,
        145..=176 => Bitrate::Kbps160,
        177..=208 => Bitrate::Kbps192,
        209..=240 => Bitrate::Kbps224,
        241..=288 => Bitrate::Kbps256,
        _ => Bitrate::Kbps320,
    };
    let mut b = Builder::new().ok_or_else(|| anyhow!("lame init failed"))?;
    b.set_num_channels(1).map_err(|e| anyhow!("mp3: {e:?}"))?;
    b.set_sample_rate(rate).map_err(|e| anyhow!("mp3: {e:?}"))?;
    b.set_brate(br).map_err(|e| anyhow!("mp3: {e:?}"))?;
    b.set_quality(Quality::Best).map_err(|e| anyhow!("mp3: {e:?}"))?;
    let mut enc = b.build().map_err(|e| anyhow!("mp3 build: {e:?}"))?;
    let ints: Vec<i16> = pcm.iter().map(|&s| to_i16(s)).collect();
    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(ints.len()));
    for chunk in ints.chunks(8192) {
        out.reserve(mp3lame_encoder::max_required_buffer_size(chunk.len()));
        enc.encode_to_vec(MonoPcm(chunk), &mut out).map_err(|e| anyhow!("mp3 encode: {e:?}"))?;
    }
    out.reserve(7200);
    enc.flush_to_vec::<FlushNoGap>(&mut out).map_err(|e| anyhow!("mp3 flush: {e:?}"))?;
    Ok(out)
}

fn vorbis(samples: &[f32], sr: u32, q: f32, title: Option<&str>) -> Result<Vec<u8>> {
    use std::num::{NonZeroU32, NonZeroU8};
    use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoderBuilder};
    let mut buf = Vec::new();
    {
        let mut b = VorbisEncoderBuilder::new(NonZeroU32::new(sr).unwrap(), NonZeroU8::new(1).unwrap(), &mut buf)?;
        b.bitrate_management_strategy(VorbisBitrateManagementStrategy::QualityVbr {
            target_quality: q.clamp(-0.1, 1.0),
        });
        if let Some(t) = title {
            b.comment_tag("TITLE", t)?;
        }
        b.comment_tag("ENCODER", "vox")?;
        let mut enc = b.build()?;
        for chunk in samples.chunks(4096) {
            enc.encode_audio_block([chunk])?;
        }
        enc.finish()?;
    }
    Ok(buf)
}

fn aiff(samples: &[f32], sr: u32) -> Vec<u8> {
    let n = samples.len() as u32;
    let data_len = n * 2;
    let mut v = Vec::with_capacity(54 + data_len as usize);
    v.extend(b"FORM");
    v.extend((4 + 26 + 16 + data_len).to_be_bytes());
    v.extend(b"AIFF");
    v.extend(b"COMM");
    v.extend(18u32.to_be_bytes());
    v.extend(1i16.to_be_bytes());
    v.extend(n.to_be_bytes());
    v.extend(16i16.to_be_bytes());
    v.extend(f64_to_extended(sr as f64));
    v.extend(b"SSND");
    v.extend((8 + data_len).to_be_bytes());
    v.extend(0u32.to_be_bytes());
    v.extend(0u32.to_be_bytes());
    for &s in samples {
        v.extend(to_i16(s).to_be_bytes());
    }
    v
}

/// IEEE 754 80-bit extended (AIFF sample rate field).
fn f64_to_extended(x: f64) -> [u8; 10] {
    let mut out = [0u8; 10];
    if x <= 0.0 {
        return out;
    }
    let exp = x.log2().floor() as i32;
    let mant = (x / 2f64.powi(exp) * (1u64 << 63) as f64) as u64;
    let e = (exp + 16383) as u16;
    out[0..2].copy_from_slice(&e.to_be_bytes());
    out[2..10].copy_from_slice(&mant.to_be_bytes());
    out
}

// ---------------------------------------------------------------- Opus (libopus, loaded at runtime)

type OpusCreate = unsafe extern "C" fn(i32, i32, i32, *mut i32) -> *mut std::ffi::c_void;
type OpusEncode = unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, i32, *mut u8, i32) -> i32;
type OpusCtl = unsafe extern "C" fn(*mut std::ffi::c_void, i32, ...) -> i32;
type OpusDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);

fn load_opus() -> Result<libloading::Library> {
    let mut cands: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("VOX_OPUS_LIB") {
        cands.push(p);
    }
    cands.extend(
        [
            "/opt/homebrew/lib/libopus.dylib",
            "/usr/local/lib/libopus.dylib",
            "/usr/lib/x86_64-linux-gnu/libopus.so.0",
            "/usr/lib/aarch64-linux-gnu/libopus.so.0",
            "libopus.so.0",
            "libopus.dylib",
        ]
        .map(String::from),
    );
    for c in &cands {
        if let Ok(l) = unsafe { libloading::Library::new(c) } {
            return Ok(l);
        }
    }
    bail!("Opus export needs libopus (brew install opus / apt install libopus0), or set VOX_OPUS_LIB")
}

fn opus(samples: &[f32], sr: u32, kbps: u32, title: Option<&str>) -> Result<Vec<u8>> {
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};
    let lib = load_opus()?;
    let pcm = resample(samples, sr, 48000);
    let frame = 960usize; // 20 ms
    let mut buf = Vec::new();
    unsafe {
        let create: OpusCreate = *lib.get(b"opus_encoder_create\0")?;
        let encode: OpusEncode = *lib.get(b"opus_encode_float\0")?;
        let ctl: OpusCtl = *lib.get(b"opus_encoder_ctl\0")?;
        let destroy: OpusDestroy = *lib.get(b"opus_encoder_destroy\0")?;
        let mut err = 0i32;
        let enc = create(48000, 1, 2049 /* OPUS_APPLICATION_AUDIO */, &mut err);
        if enc.is_null() || err != 0 {
            bail!("opus_encoder_create failed ({err})");
        }
        ctl(enc, 4002 /* SET_BITRATE */, (kbps.clamp(6, 510) * 1000) as i32);
        ctl(enc, 4010 /* SET_COMPLEXITY */, 10i32);
        let mut lookahead = 312i32;
        ctl(enc, 4027 /* GET_LOOKAHEAD */, &mut lookahead as *mut i32);
        let pre_skip = lookahead.max(0) as u16;

        let serial = 0x766f_7821u32;
        let mut w = PacketWriter::new(&mut buf);
        let mut head = Vec::from(&b"OpusHead"[..]);
        head.push(1);
        head.push(1);
        head.extend(pre_skip.to_le_bytes());
        head.extend(sr.to_le_bytes());
        head.extend(0i16.to_le_bytes());
        head.push(0);
        w.write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)?;
        let mut tags = Vec::from(&b"OpusTags"[..]);
        let vendor = b"vox (libopus)";
        tags.extend((vendor.len() as u32).to_le_bytes());
        tags.extend(vendor);
        let comments: Vec<String> = title.map(|t| vec![format!("TITLE={t}")]).unwrap_or_default();
        tags.extend((comments.len() as u32).to_le_bytes());
        for c in &comments {
            tags.extend((c.len() as u32).to_le_bytes());
            tags.extend(c.as_bytes());
        }
        w.write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)?;

        let total = pcm.len() + pre_skip as usize;
        let mut padded = pcm.clone();
        padded.resize(pcm.len().div_ceil(frame) * frame + frame, 0.0);
        let n_frames = padded.len() / frame;
        let mut packet = vec![0u8; 4000];
        for f in 0..n_frames {
            let len = encode(enc, padded[f * frame..].as_ptr(), frame as i32, packet.as_mut_ptr(), packet.len() as i32);
            if len < 0 {
                destroy(enc);
                bail!("opus_encode_float failed ({len})");
            }
            let last = f + 1 == n_frames;
            let gp = (((f + 1) * frame + pre_skip as usize).min(total)) as u64;
            let info = if last {
                PacketWriteEndInfo::EndStream
            } else if (f + 1) % 50 == 0 {
                PacketWriteEndInfo::EndPage
            } else {
                PacketWriteEndInfo::NormalPacket
            };
            w.write_packet(packet[..len as usize].to_vec(), serial, info, gp)?;
        }
        destroy(enc);
    }
    Ok(buf)
}

// ---------------------------------------------------------------- sidecars

#[derive(Clone, Debug, Serialize)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub persona: String,
    pub mood: String,
}

fn ts(t: f64, sep: char) -> String {
    let ms = (t * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
}

pub fn write_subtitles(path: &Path, cues: &[Cue]) -> Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("srt").to_ascii_lowercase();
    let mut f = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    match ext.as_str() {
        "json" => f.write_all(serde_json::to_string_pretty(cues)?.as_bytes())?,
        "vtt" => {
            writeln!(f, "WEBVTT\n")?;
            for c in cues {
                writeln!(f, "{} --> {}\n<v {}>{}\n", ts(c.start, '.'), ts(c.end, '.'), c.persona, c.text)?;
            }
        }
        _ => {
            for (i, c) in cues.iter().enumerate() {
                writeln!(f, "{}\n{} --> {}\n{}\n", i + 1, ts(c.start, ','), ts(c.end, ','), c.text)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sine() -> Vec<f32> {
        (0..24000).map(|i| (i as f32 * 0.05).sin() * 0.5).collect()
    }
    #[test]
    fn all_formats_encode() {
        let x = sine();
        let o = ExportOptions::default();
        for (name, f) in Format::ALL {
            if *f == Format::Opus && load_opus().is_err() {
                continue;
            }
            let b = encode(&x, 24000, *f, &o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(b.len() > 100, "{name} too small");
        }
    }
    #[test]
    fn aiff_rate() {
        let e = f64_to_extended(44100.0);
        assert_eq!(e, [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]);
    }
}
