//! Offline WAV transition renderer and differential verification harness for `mix-transitions`.
//!
//! Renders either two stereo 44.1 kHz WAV files (16-bit PCM or 32-bit float) or synthetic
//! beat-grid test tracks through [`librespot::playback::mix::TransitionProcessor`] and writes
//! a 32-bit IEEE float stereo WAV alongside loudness and boundary-continuity telemetry.

use std::{
    env,
    f64::consts::PI,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    process::exit,
};

use librespot::{
    connect::mix::{
        MixConfig,
        presets::{BuiltinPreset, curves_for_builtin_preset},
        resolver::{
            TrackMixMetadata, TrackTransitionContext, decode_transition_recipe,
            transition_proto_to_spec,
        },
    },
    playback::{
        NUM_CHANNELS, SAMPLE_RATE,
        mix::{EqTopology, TempoPlan, TimeStretchMode, TransitionProcessor, TransitionSpec},
    },
};

const CHANNELS: usize = NUM_CHANNELS as usize;

fn usage(bin: &str) -> ! {
    eprintln!(
        "Usage: {bin} [OPTIONS] --out OUTPUT.wav\n\
         \n\
         Options:\n\
           --track-a FILE         44.1 kHz stereo WAV (PCM16 or Float32); omit for synthetic track A\n\
           --track-b FILE         44.1 kHz stereo WAV (PCM16 or Float32); omit for synthetic track B\n\
           --preset NAME          fade | eq-swap | bass-swap | hp-filter | lp-filter | cut (default: bass-swap)\n\
           --recipe BASE64        Explicit Spotify base64 transition_recipe protobuf (overrides preset/overlap)\n\
           --bpm-a FLOAT          Outgoing BPM (default: 145.0)\n\
           --bpm-b FLOAT          Incoming BPM (default: 150.0)\n\
           --bars INT             Overlap bars when beatmatched (default: 8)\n\
           --start-a-ms INT       Outgoing overlap start in ms (default: 2000 for synthetic / 0 for WAV)\n\
           --start-b-ms INT       Incoming overlap start in ms (default: 0)\n\
           --stretch MODE         pitch-preserving | varispeed | off (default: pitch-preserving)\n\
           --out FILE             Output 32-bit float stereo WAV path"
    );
    exit(2);
}

fn read_wav_stereo_f64(path: &PathBuf) -> Result<Vec<f64>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF WAVE file", path.display()));
    }

    let mut pos = 12usize;
    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let payload_start = pos + 8;
        let payload_end = (payload_start + size).min(bytes.len());
        let payload = &bytes[payload_start..payload_end];

        if id == b"fmt " && payload.len() >= 16 {
            audio_format = u16::from_le_bytes(payload[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(payload[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(payload[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(payload[14..16].try_into().unwrap());
        } else if id == b"data" {
            data_slice = Some(payload);
        }

        pos = payload_start + size + (size & 1);
    }

    if sample_rate != SAMPLE_RATE {
        return Err(format!(
            "{}: expected {SAMPLE_RATE} Hz sample rate, got {sample_rate} Hz",
            path.display()
        ));
    }
    if channels != 1 && channels != 2 {
        return Err(format!(
            "{}: expected mono or stereo, got {channels} channels",
            path.display()
        ));
    }
    let data = data_slice.ok_or_else(|| format!("{}: missing data chunk", path.display()))?;
    let mut samples = Vec::new();
    match (audio_format, bits_per_sample) {
        (1, 16) => {
            for chunk in data.chunks_exact(2) {
                let s = f64::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 32768.0;
                samples.push(s);
                if channels == 1 {
                    samples.push(s);
                }
            }
        }
        (3, 32) => {
            for chunk in data.chunks_exact(4) {
                let s = f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                samples.push(s);
                if channels == 1 {
                    samples.push(s);
                }
            }
        }
        _ => {
            return Err(format!(
                "{}: unsupported WAV format={audio_format} bits={bits_per_sample} (expected PCM16 or Float32)",
                path.display()
            ));
        }
    }
    Ok(samples)
}

fn write_wav_stereo_f32(path: &PathBuf, interleaved: &[f64]) -> Result<(), String> {
    let data_bytes = (interleaved.len() * 4) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let mut file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;

    file.write_all(b"RIFF").map_err(|e| e.to_string())?;
    file.write_all(&riff_size.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(b"WAVEfmt ").map_err(|e| e.to_string())?;
    file.write_all(&16u32.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&3u16.to_le_bytes()) // IEEE float
        .map_err(|e| e.to_string())?;
    file.write_all(&(CHANNELS as u16).to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())
        .map_err(|e| e.to_string())?;
    let byte_rate = SAMPLE_RATE * (CHANNELS as u32) * 4;
    file.write_all(&byte_rate.to_le_bytes())
        .map_err(|e| e.to_string())?;
    let block_align = (CHANNELS as u16) * 4;
    file.write_all(&block_align.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&32u16.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(b"data").map_err(|e| e.to_string())?;
    file.write_all(&data_bytes.to_le_bytes())
        .map_err(|e| e.to_string())?;

    let mut buf = Vec::with_capacity(interleaved.len() * 4);
    for &s in interleaved {
        buf.extend_from_slice(&(s as f32).to_le_bytes());
    }
    file.write_all(&buf).map_err(|e| e.to_string())
}

fn generate_synthetic_track(bpm: f64, bass_hz: f64, pad_hz: f64, duration_sec: f64) -> Vec<f64> {
    let fs = f64::from(SAMPLE_RATE);
    let total_frames = (duration_sec * fs).ceil() as usize;
    let beat_frames = fs * 60.0 / bpm;
    let mut pcm = vec![0.0_f64; total_frames * CHANNELS];

    for i in 0..total_frames {
        let t = i as f64 / fs;
        let beat_phase = (i as f64 % beat_frames) / fs;
        // Punchy pitch-swept kick transient + sub-bass + gentle stereo pad
        let kick_env = (-beat_phase / 0.045).exp();
        let kick_freq = 48.0 + 110.0 * (-beat_phase / 0.015).exp();
        let kick = 0.45 * kick_env * (2.0 * PI * kick_freq * beat_phase).sin();
        let bass = 0.22 * (2.0 * PI * bass_hz * t).sin();
        let pad_l = 0.12 * (2.0 * PI * pad_hz * t).sin();
        let pad_r = 0.12 * (2.0 * PI * (pad_hz * 1.003) * t).sin();

        pcm[2 * i] = kick + bass + pad_l;
        pcm[2 * i + 1] = kick + bass + pad_r;
    }
    pcm
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bin = args[0].as_str();

    let mut track_a_path: Option<PathBuf> = None;
    let mut track_b_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut preset = BuiltinPreset::BassSwap;
    let mut recipe_b64: Option<String> = None;
    let mut bpm_a = 145.0_f64;
    let mut bpm_b = 150.0_f64;
    let mut bars = 8u32;
    let mut start_a_ms: Option<u64> = None;
    let mut start_b_ms = 0u64;
    let mut stretch_mode = TimeStretchMode::PitchPreserving;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--track-a" if i + 1 < args.len() => {
                track_a_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--track-b" if i + 1 < args.len() => {
                track_b_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--preset" if i + 1 < args.len() => {
                preset = BuiltinPreset::from_name(&args[i + 1]).unwrap_or_else(|| {
                    eprintln!("Unknown preset '{}'", args[i + 1]);
                    exit(2);
                });
                i += 2;
            }
            "--recipe" if i + 1 < args.len() => {
                recipe_b64 = Some(args[i + 1].clone());
                i += 2;
            }
            "--bpm-a" if i + 1 < args.len() => {
                bpm_a = args[i + 1].parse().unwrap_or_else(|_| usage(bin));
                i += 2;
            }
            "--bpm-b" if i + 1 < args.len() => {
                bpm_b = args[i + 1].parse().unwrap_or_else(|_| usage(bin));
                i += 2;
            }
            "--bars" if i + 1 < args.len() => {
                bars = args[i + 1].parse().unwrap_or_else(|_| usage(bin));
                i += 2;
            }
            "--start-a-ms" if i + 1 < args.len() => {
                start_a_ms = Some(args[i + 1].parse().unwrap_or_else(|_| usage(bin)));
                i += 2;
            }
            "--start-b-ms" if i + 1 < args.len() => {
                start_b_ms = args[i + 1].parse().unwrap_or_else(|_| usage(bin));
                i += 2;
            }
            "--stretch" if i + 1 < args.len() => {
                stretch_mode = match args[i + 1].as_str() {
                    "pitch-preserving" => TimeStretchMode::PitchPreserving,
                    "varispeed" => TimeStretchMode::Varispeed,
                    "off" => TimeStretchMode::Off,
                    _ => usage(bin),
                };
                i += 2;
            }
            "-h" | "--help" => usage(bin),
            other => {
                eprintln!("Unknown argument: {other}");
                usage(bin);
            }
        }
    }

    let out_path = out_path.unwrap_or_else(|| usage(bin));

    let pcm_a = match track_a_path {
        Some(ref p) => read_wav_stereo_f64(p).unwrap_or_else(|e| {
            eprintln!("Error loading track A: {e}");
            exit(1);
        }),
        None => generate_synthetic_track(bpm_a, 55.0, 220.0, 24.0),
    };
    let pcm_b = match track_b_path {
        Some(ref p) => read_wav_stereo_f64(p).unwrap_or_else(|e| {
            eprintln!("Error loading track B: {e}");
            exit(1);
        }),
        None => generate_synthetic_track(bpm_b, 65.41, 261.63, 24.0),
    };

    let start_a_ms = start_a_ms.unwrap_or(if track_a_path.is_none() { 2000 } else { 0 });

    let spec = if let Some(ref b64) = recipe_b64 {
        let proto = decode_transition_recipe(b64).unwrap_or_else(|| {
            eprintln!("Failed to decode base64 transition_recipe");
            exit(1);
        });
        let cfg = MixConfig {
            stretch_mode,
            ..MixConfig::default()
        };
        let meta = TrackMixMetadata::default();
        let ctx = TrackTransitionContext {
            track_uri: "",
            row_uid: "",
            meta: &meta,
        };
        transition_proto_to_spec(&proto, ctx, ctx, &cfg).unwrap_or_else(|| {
            eprintln!("Transition recipe did not produce a valid TransitionSpec");
            exit(1);
        })
    } else {
        let (out_curves, in_curves) = curves_for_builtin_preset(preset);
        let tempo = (stretch_mode != TimeStretchMode::Off).then_some(TempoPlan {
            bpm_a,
            bpm_b,
            beatmatched: true,
            max_bpm_ratio: 1.15,
            beats_a: None,
            beats_b: None,
            stretch_mode,
        });
        let dur_beats = f64::from(bars) * 4.0;
        let avg_bpm = 0.5 * (bpm_a + bpm_b);
        let dur_frames = (dur_beats * 60.0 / avg_bpm * f64::from(SAMPLE_RATE)).round() as u64;
        TransitionSpec {
            start_a_frame: start_a_ms * u64::from(SAMPLE_RATE) / 1000,
            start_b_frame: start_b_ms * u64::from(SAMPLE_RATE) / 1000,
            duration_frames: dur_frames,
            duration_beats: tempo.as_ref().map(|_| dur_beats),
            handoff_fraction: preset.handoff_fraction(),
            tempo,
            out_curves,
            in_curves,
            eq_topology: EqTopology::default(),
        }
    };

    let start_a_idx = ((spec.start_a_frame as usize) * CHANNELS).min(pcm_a.len());
    let start_b_idx = ((spec.start_b_frame as usize) * CHANNELS).min(pcm_b.len());

    // Emit up to 2 s of dry Track A pre-roll so boundary entry can be heard/inspected.
    let preroll_samples = (SAMPLE_RATE as usize * 2 * CHANNELS).min(start_a_idx);
    let mut rendered = Vec::new();
    rendered.extend_from_slice(&pcm_a[start_a_idx - preroll_samples..start_a_idx]);

    let mut proc = TransitionProcessor::new(spec);
    let mut pos_a = start_a_idx;
    let mut pos_b = start_b_idx;
    let feed = 2048 * CHANNELS;

    while !proc.is_complete() {
        while proc.needs_deck_a_input() {
            if pos_a < pcm_a.len() {
                let end = (pos_a + feed).min(pcm_a.len());
                proc.push_deck_a(&pcm_a[pos_a..end]);
                pos_a = end;
            } else {
                proc.set_deck_a_eof();
                break;
            }
        }
        while proc.needs_deck_b_input() {
            if pos_b < pcm_b.len() {
                let end = (pos_b + feed).min(pcm_b.len());
                proc.push_deck_b(&pcm_b[pos_b..end]);
                pos_b = end;
            } else {
                proc.set_deck_b_eof();
                break;
            }
        }
        let wrote = proc.render(512, &mut rendered);
        if wrote == 0 && pos_a >= pcm_a.len() && pos_b >= pcm_b.len() {
            break;
        }
    }

    // Append exact zero-drop continuation + up to 2 s dry Track B post-roll.
    let (_next_frame, cont) = proc.drain_deck_b_continuation();
    rendered.extend_from_slice(&cont);
    let postroll = (SAMPLE_RATE as usize * 2 * CHANNELS).min(pcm_b.len().saturating_sub(pos_b));
    rendered.extend_from_slice(&pcm_b[pos_b..pos_b + postroll]);

    write_wav_stereo_f32(&out_path, &rendered).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {e}", out_path.display());
        exit(1);
    });

    let peak = rendered.iter().fold(0.0_f64, |acc, &s| acc.max(s.abs()));
    let peak_db = 20.0 * peak.max(1e-12).log10();
    let total_sec = (rendered.len() / CHANNELS) as f64 / f64::from(SAMPLE_RATE);
    println!(
        "Wrote {} ({:.2} s, {} frames, peak {:.2} dBFS)",
        out_path.display(),
        total_sec,
        rendered.len() / CHANNELS,
        peak_db
    );
}
