//! Transition metadata resolution, Spotify `transition_recipe` decoding,
//! and client-side Auto DJ transition synthesis.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use data_encoding::{BASE64, BASE64_NOPAD, BASE64URL, BASE64URL_NOPAD};
use protobuf::Message;
use tokio::sync::Mutex;

use crate::{
    core::{Session, SpotifyId, SpotifyUri},
    mix::{
        MixConfig, MixMode,
        presets::{
            BuiltinPreset, bipolar_eq_semantics, bipolar_resonance_semantics,
            curves_for_builtin_preset, curves_from_proto_preset,
        },
    },
    playback::{
        SAMPLE_RATE,
        mix::{
            AutomationCurve, AutomationCurves, BeatGrid, BeatMarker, CurveInterpolation,
            CurvePoint, DeckCurves, TempoPlan, TimeStretchMode, TransitionSpec,
            beat_clock::fold_bpm_octave,
        },
    },
    protocol::{
        audio_attributes_v2::AudioAttributes,
        automix_mode::{AutomixMode, AutomixStyle},
        cuepoints::{Cuepoint, Cuepoints, Origin},
        extension_kind::ExtensionKind,
        mix_beats::Beats,
        mix_state::MixState,
        mix_vocal_activity::VocalActivity,
        mixability::Mixability,
        player::ProvidedTrack,
        playlist4_external::SelectedListContent,
        transition::{BeatmatchPreference, Transition},
    },
};

const MAX_CACHED_TRACKS: usize = 128;
const MAX_CACHED_CONTEXTS: usize = 32;
const DEFAULT_FALLBACK_FADE_MS: u64 = 5_000;
const MAX_OVERLAP_SEC: f64 = 32.0;

/// Metadata keys checked on [`ProvidedTrack::metadata`] for user-made or backend transition recipes.
const RECIPE_METADATA_KEYS: &[&str] = &[
    "transition_recipe",
    "automix.transition_recipe",
    "automix.auto_transition_recipe",
    "mix.transition_recipe",
];

/// Cached Spotify Mix analysis metadata for one track.
#[derive(Debug, Clone, Default)]
pub struct TrackMixMetadata {
    /// Track duration in milliseconds (from `ConnectState` or audio analysis fallback).
    pub duration_ms: u32,
    /// Estimated tempo in beats per minute.
    pub bpm: Option<f64>,
    /// Camelot musical key notation (e.g. `"2B"`, `"11A"`).
    pub camelot_key: Option<String>,
    /// Per-beat timing grid from `ExtensionKind::BEATS`.
    pub beats: Option<BeatGrid>,
    /// Beat-grid hash used to validate cached transitions.
    pub beats_hash: Option<String>,
    /// Fade-in / fade-out cuepoint candidates from `ExtensionKind::CUEPOINTS`.
    pub cuepoints: Option<Cuepoints>,
    /// Windowed vocal presence probabilities from `ExtensionKind::VOCAL_ACTIVITY`.
    pub vocal_activity: Option<VocalActivity>,
    /// Whether Spotify considers this track suitable for mixing.
    pub mixable: Option<bool>,
    /// Genre-derived beatmatchability score in `[0.0, 1.0]`.
    pub beatmatchability: Option<f64>,
    /// Legacy/hint automix style from `ExtensionKind::AUTOMIX_MODE`.
    pub automix_style: Option<AutomixStyle>,
}

/// Request snapshot captured from `ConnectState` to resolve the upcoming `(current -> next)` transition.
#[derive(Debug, Clone)]
pub struct TransitionResolveRequest {
    /// Monotonic generation ID used to drop stale async resolutions.
    pub generation: u64,
    /// Active context URI (e.g. `spotify:playlist:...`).
    pub context_uri: String,
    /// Whether `ConnectState` modes or context metadata already signal Mix ON.
    pub context_signals_mixed: bool,
    /// Outgoing track URI.
    pub current_track_id: SpotifyUri,
    /// Outgoing track playlist row UID (when available).
    pub current_row_uid: String,
    /// Outgoing track duration in milliseconds (0 if unknown yet).
    pub current_duration_ms: u32,
    /// Outgoing track metadata map from `ProvidedTrack`.
    pub current_metadata: HashMap<String, String>,
    /// Incoming track URI.
    pub next_track_id: SpotifyUri,
    /// Incoming track playlist row UID (when available).
    pub next_row_uid: String,
    /// Incoming track duration in milliseconds (0 if unknown yet).
    pub next_duration_ms: u32,
    /// Incoming track metadata map from `ProvidedTrack`.
    pub next_metadata: HashMap<String, String>,
}

/// Resolved result sent back from the background resolver to `SpircTask`.
#[derive(Debug, Clone)]
pub struct TransitionResolveResult {
    /// Generation ID echoed from [`TransitionResolveRequest::generation`].
    pub generation: u64,
    /// Outgoing track URI this transition was resolved for.
    pub current_track_id: SpotifyUri,
    /// Incoming track URI this transition was resolved for.
    pub next_track_id: SpotifyUri,
    /// Resolved transition specification, or `None` for normal gapless playback.
    pub spec: Option<TransitionSpec>,
}

/// Check whether Connect state `modes` or context metadata explicitly enable Mix mode.
pub fn context_signals_mixed(
    modes: Option<&HashMap<String, String>>,
    context_metadata: Option<&HashMap<String, String>>,
    current_track: Option<&ProvidedTrack>,
    next_track: Option<&ProvidedTrack>,
) -> bool {
    let map_is_mixed = |m: &HashMap<String, String>| -> bool {
        m.get("automix.mode").is_some_and(|v| {
            let l = v.to_ascii_lowercase();
            l == "mixed" || l == "on" || l == "true" || l == "auto"
        }) || m.get("mix").is_some_and(|v| {
            let l = v.to_ascii_lowercase();
            l == "on" || l == "true" || l == "mixed"
        }) || m
            .get("is_mixed")
            .is_some_and(|v| v.eq_ignore_ascii_case("true") || v == "1")
            || m.get("mixer_enabled")
                .is_some_and(|v| v.eq_ignore_ascii_case("true") || v == "1")
    };

    if modes.is_some_and(map_is_mixed) || context_metadata.is_some_and(map_is_mixed) {
        return true;
    }

    let has_track_mix_data = |t: &ProvidedTrack| {
        RECIPE_METADATA_KEYS
            .iter()
            .any(|k| t.metadata.get(*k).is_some_and(|v| !v.is_empty()))
            || t.metadata
                .get("audio.fade_out_start_time")
                .is_some_and(|v| !v.is_empty())
            || t.metadata
                .get("audio.fade_in_start_time")
                .is_some_and(|v| !v.is_empty())
    };
    current_track.is_some_and(has_track_mix_data) || next_track.is_some_and(has_track_mix_data)
}

/// Decode a base64-encoded `spotify.automix.proto.Transition` (`transition_recipe`).
pub fn decode_transition_recipe(encoded: &str) -> Option<Transition> {
    let trimmed = encoded.trim();
    if trimmed.is_empty() {
        return None;
    }
    for enc in [BASE64, BASE64_NOPAD, BASE64URL, BASE64URL_NOPAD] {
        if let Ok(bytes) = enc.decode(trimmed.as_bytes()) {
            if let Ok(transition) = Transition::parse_from_bytes(&bytes) {
                if transition.overlap.is_some() {
                    return Some(transition);
                }
            }
        }
    }
    None
}

/// Parse a Camelot key string such as `"2B"` or `"11A"` into `(hour 1..=12, is_major)`.
pub fn parse_camelot_key(key: &str) -> Option<(u8, bool)> {
    let s = key.trim().to_ascii_uppercase();
    if s.len() < 2 || s.len() > 3 {
        return None;
    }
    let mode_char = s.chars().last()?;
    let is_major = match mode_char {
        'A' => false,
        'B' => true,
        _ => return None,
    };
    let num: u8 = s[..s.len() - 1].parse().ok()?;
    if (1..=12).contains(&num) {
        Some((num, is_major))
    } else {
        None
    }
}

/// Harmonic compatibility tier on the Camelot wheel (`0` = identical, `1` = +/- 1 or relative A/B,
/// `2` = energy boost +/- 2, `3` = distant/clashing).
pub fn camelot_compatibility_tier(key_a: Option<&str>, key_b: Option<&str>) -> u8 {
    let (Some((na, ma)), Some((nb, mb))) = (
        key_a.and_then(parse_camelot_key),
        key_b.and_then(parse_camelot_key),
    ) else {
        return 1; // Neutral when key metadata is absent.
    };
    let diff = na.abs_diff(nb);
    let wheel_dist = diff.min(12 - diff);
    match (wheel_dist, ma == mb) {
        (0, true) => 0,
        (0, false) | (1, true) => 1,
        (2, true) | (1, false) => 2,
        _ => 3,
    }
}

#[inline]
fn ms_to_frames(ms: u64) -> u64 {
    (ms.saturating_mul(u64::from(SAMPLE_RATE))) / 1000
}

#[inline]
fn sec_to_frames(sec: f64) -> u64 {
    (sec.max(0.0) * f64::from(SAMPLE_RATE)).round() as u64
}

/// Track identity and cached analysis metadata passed to [`transition_proto_to_spec`].
#[derive(Debug, Clone, Copy)]
pub struct TrackTransitionContext<'a> {
    /// Canonical Spotify track URI (empty to skip URI validation).
    pub track_uri: &'a str,
    /// Playlist row UID (empty to skip row validation).
    pub row_uid: &'a str,
    /// Cached track analysis metadata.
    pub meta: &'a TrackMixMetadata,
}

/// Translate an explicit Spotify [`Transition`] protobuf message (from a user-made
/// `transition_recipe` or backend payload) into a [`TransitionSpec`].
///
/// Validates track URIs and playlist row UIDs when present so reordered playlists never apply
/// a stale transition recipe to the wrong adjacent track pair.
pub fn transition_proto_to_spec(
    proto: &Transition,
    track_a: TrackTransitionContext<'_>,
    track_b: TrackTransitionContext<'_>,
    config: &MixConfig,
) -> Option<TransitionSpec> {
    let overlap = proto.overlap.as_ref()?;
    let track_a_uri = track_a.track_uri;
    let row_a_uid = track_a.row_uid;
    let meta_a = track_a.meta;
    let track_b_uri = track_b.track_uri;
    let row_b_uid = track_b.row_uid;
    let meta_b = track_b.meta;

    if !overlap.track_a_uri.is_empty() && overlap.track_a_uri != track_a_uri {
        debug!(
            "Skipping transition recipe: track_a_uri mismatch ({} != {track_a_uri})",
            overlap.track_a_uri
        );
        return None;
    }
    if !overlap.track_b_uri.is_empty() && overlap.track_b_uri != track_b_uri {
        debug!(
            "Skipping transition recipe: track_b_uri mismatch ({} != {track_b_uri})",
            overlap.track_b_uri
        );
        return None;
    }
    if !overlap.track_a_row_id.is_empty()
        && !row_a_uid.is_empty()
        && overlap.track_a_row_id != row_a_uid
    {
        debug!(
            "Skipping transition recipe: track_a_row_id mismatch ({} != {row_a_uid})",
            overlap.track_a_row_id
        );
        return None;
    }
    if !overlap.track_b_row_id.is_empty()
        && !row_b_uid.is_empty()
        && overlap.track_b_row_id != row_b_uid
    {
        debug!(
            "Skipping transition recipe: track_b_row_id mismatch ({} != {row_b_uid})",
            overlap.track_b_row_id
        );
        return None;
    }

    let start_a_ms = overlap.start_a_ms.max(0) as u64;
    let start_b_ms = overlap.start_b_ms.max(0) as u64;
    let duration_ms = overlap.duration_ms.max(0) as u64;
    if duration_ms < 250 {
        return None;
    }

    let (out_curves, in_curves, handoff_fraction) = proto
        .preset
        .as_ref()
        .map(curves_from_proto_preset)
        .unwrap_or_else(|| {
            let (o, i) = curves_for_builtin_preset(BuiltinPreset::BassSwap);
            (o, i, BuiltinPreset::BassSwap.handoff_fraction())
        });

    let want_beatmatch = match proto.beatmatch_preference.enum_value_or_default() {
        BeatmatchPreference::BEATMATCH_PREFERENCE_OFF => false,
        BeatmatchPreference::BEATMATCH_PREFERENCE_ON => true,
        BeatmatchPreference::BEATMATCH_PREFERENCE_AUTO => overlap.is_beatmatched,
    };

    let scale_bpm = |raw: f64, speed_opt: Option<f32>, item_speed: f64| -> f64 {
        let mut s = 1.0;
        if let Some(sp) = speed_opt {
            let sp = f64::from(sp);
            if sp.is_finite() && (0.5..=2.0).contains(&sp) {
                s *= sp;
            }
        }
        if item_speed.is_finite() && (0.5..=2.0).contains(&item_speed) {
            s *= item_speed;
        }
        raw * s
    };

    let bpm_a = if overlap.bpm_a > 20.0 {
        Some(scale_bpm(
            f64::from(overlap.bpm_a),
            overlap.speed_a,
            overlap.item_speed_a,
        ))
    } else {
        meta_a.bpm
    };
    let bpm_b = if overlap.bpm_b > 20.0 {
        Some(scale_bpm(
            f64::from(overlap.bpm_b),
            overlap.speed_b,
            overlap.item_speed_b,
        ))
    } else {
        meta_b.bpm
    };

    let beats_per_bar = meta_a
        .beats
        .as_ref()
        .map(|g| g.beats_per_bar)
        .filter(|&b| b > 0)
        .unwrap_or(4);
    let duration_beats = overlap
        .duration_bars
        .filter(|&bars| bars > 0)
        .map(|bars| f64::from(bars) * f64::from(beats_per_bar));

    let tempo = if want_beatmatch && config.stretch_mode != TimeStretchMode::Off {
        if let (Some(ba), Some(bb)) = (bpm_a, bpm_b) {
            Some(TempoPlan {
                bpm_a: ba,
                bpm_b: bb,
                beatmatched: true,
                max_bpm_ratio: config.max_bpm_ratio,
                beats_a: meta_a.beats.clone(),
                beats_b: meta_b.beats.clone(),
                stretch_mode: config.stretch_mode,
            })
        } else {
            None
        }
    } else {
        None
    };

    Some(TransitionSpec {
        start_a_frame: ms_to_frames(start_a_ms),
        start_b_frame: ms_to_frames(start_b_ms),
        duration_frames: ms_to_frames(duration_ms),
        duration_beats,
        handoff_fraction,
        tempo,
        out_curves,
        in_curves,
        eq_topology: config.eq_topology,
    })
}

/// Parse Spotify's `audio.fade_*` JSON curve array into protocol-agnostic [`AutomationCurves`].
///
/// Expected schema (as transferred on `ProvidedTrack.metadata`):
/// `[{"start_point": 0.0, "end_point": 0.5, "fade_curve": [{"x": 0.0, "y": 1.0}, ...]}, ...]`.
/// Two-point segments are piecewise linear; 4-point (and `3k + 1`-point) segments are cubic Bézier.
pub fn parse_fade_json_curves(raw_json: &str) -> Option<AutomationCurves> {
    let val: serde_json::Value = serde_json::from_str(raw_json).ok()?;
    let arr = val.as_array()?;
    if arr.is_empty() {
        return None;
    }

    let mut curves = Vec::with_capacity(arr.len());
    for seg in arr {
        let start = seg.get("start_point")?.as_f64()?;
        let end = seg.get("end_point")?.as_f64()?;
        if !start.is_finite() || !end.is_finite() || end < start {
            return None;
        }
        let pts_arr = seg.get("fade_curve")?.as_array()?;
        if pts_arr.is_empty() {
            continue;
        }
        let mut points = Vec::with_capacity(pts_arr.len());
        for pt in pts_arr {
            let x = pt.get("x")?.as_f64()?;
            let y = pt.get("y")?.as_f64()?;
            if !x.is_finite() || !y.is_finite() {
                return None;
            }
            points.push(CurvePoint { x, y });
        }
        let interpolation = if points.len() >= 4 && (points.len() - 1) % 3 == 0 {
            CurveInterpolation::CubicBezier
        } else {
            CurveInterpolation::Linear
        };
        curves.push(AutomationCurve {
            points,
            start,
            end,
            interpolation,
        });
    }

    if curves.is_empty() {
        return None;
    }

    Some(AutomationCurves {
        curves,
        minimum: 0.0,
        maximum: 1.0,
    })
}

fn parse_ms_meta(map: &HashMap<String, String>, key: &str) -> Option<u64> {
    let raw = map.get(key)?.trim();
    if raw.is_empty() {
        return None;
    }
    raw.parse::<u64>().ok().or_else(|| {
        raw.parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(|v| v.round() as u64)
    })
}

fn parse_deck_fade_curves(map: &HashMap<String, String>, dir: &str) -> DeckCurves {
    let key = |suffix: &str| format!("audio.fade_{dir}_{suffix}");
    let mut deck = DeckCurves {
        volume: map
            .get(&key("curves"))
            .and_then(|s| parse_fade_json_curves(s)),
        eq_low: map
            .get(&key("eq_low_gain_curves"))
            .and_then(|s| parse_fade_json_curves(s)),
        eq_mid: map
            .get(&key("eq_mid_gain_curves"))
            .and_then(|s| parse_fade_json_curves(s)),
        eq_high: map
            .get(&key("eq_high_gain_curves"))
            .and_then(|s| parse_fade_json_curves(s)),
        filter_cutoff: map
            .get(&key("filter_cutoff_curves"))
            .and_then(|s| parse_fade_json_curves(s)),
        filter_resonance: map
            .get(&key("filter_resonance_curves"))
            .and_then(|s| parse_fade_json_curves(s)),
        ..Default::default()
    };

    if deck.eq_low.is_some() || deck.eq_mid.is_some() || deck.eq_high.is_some() {
        deck.eq_semantics = Some(bipolar_eq_semantics());
    }
    if deck.filter_resonance.is_some() {
        deck.filter_resonance_semantics = Some(bipolar_resonance_semantics());
    }
    deck
}

/// Build a [`TransitionSpec`] from Spotify's pre-computed `audio.fade_*` track metadata
/// (sent on `ProvidedTrack.metadata` during Connect playback transfer).
///
/// Enforces pair consistency via `audio.fade_out_uri` (when present) and `audio.fade_overlap`
/// matching `audio.fade_out_duration` within 500 ms so manual queue insertions or reorders
/// cleanly fall back to Auto DJ synthesis instead of applying stale curves to the wrong track.
pub fn transition_from_fade_metadata(
    track_a: TrackTransitionContext<'_>,
    meta_a_map: &HashMap<String, String>,
    track_b: TrackTransitionContext<'_>,
    meta_b_map: &HashMap<String, String>,
    config: &MixConfig,
) -> Option<TransitionSpec> {
    let start_a_ms = parse_ms_meta(meta_a_map, "audio.fade_out_start_time")?;
    let fade_out_dur_ms = parse_ms_meta(meta_a_map, "audio.fade_out_duration")?;
    if !(250..=45_000).contains(&fade_out_dur_ms) {
        return None;
    }

    if let Some(target_uri) = meta_a_map.get("audio.fade_out_uri") {
        if !target_uri.is_empty()
            && !track_b.track_uri.is_empty()
            && target_uri != track_b.track_uri
        {
            debug!(
                "Skipping transferred fade metadata: audio.fade_out_uri mismatch ({target_uri} != {})",
                track_b.track_uri
            );
            return None;
        }
    }

    let fade_overlap_ms = parse_ms_meta(meta_b_map, "audio.fade_overlap");
    let fade_in_dur_ms = parse_ms_meta(meta_b_map, "audio.fade_in_duration");
    if fade_overlap_ms.is_none() && fade_in_dur_ms.is_none() {
        return None;
    }

    if let Some(overlap_ms) = fade_overlap_ms {
        if overlap_ms.abs_diff(fade_out_dur_ms) > 500 {
            debug!(
                "Skipping transferred fade metadata: fade_overlap ({overlap_ms} ms) != fade_out_duration ({fade_out_dur_ms} ms)"
            );
            return None;
        }
    }

    let start_b_ms = parse_ms_meta(meta_b_map, "audio.fade_in_start_time").unwrap_or(0);
    let fade_in_dur_ms =
        fade_in_dur_ms.unwrap_or_else(|| fade_overlap_ms.unwrap_or(fade_out_dur_ms));
    if !(250..=45_000).contains(&fade_in_dur_ms) {
        return None;
    }

    let out_curves = parse_deck_fade_curves(meta_a_map, "out");
    let in_curves = parse_deck_fade_curves(meta_b_map, "in");

    let bpm_a = effective_bpm(track_a.meta);
    let bpm_b = effective_bpm(track_b.meta);
    let beatmatchable_genre = track_a.meta.beatmatchability.unwrap_or(1.0) >= 0.25
        && track_b.meta.beatmatchability.unwrap_or(1.0) >= 0.25;

    let (tempo, duration_beats, duration_ms) = match (bpm_a, bpm_b) {
        (Some(ba), Some(bb)) if config.stretch_mode != TimeStretchMode::Off => {
            let folded_b = fold_bpm_octave(ba, bb);
            let bpm_ratio = (folded_b / ba).max(ba / folded_b);
            let dur_ratio = fade_out_dur_ms as f64 / fade_in_dur_ms as f64;
            let expected_dur_ratio = folded_b / ba;
            let durations_match_bpm = fade_out_dur_ms != fade_in_dur_ms
                && (dur_ratio / expected_dur_ratio - 1.0).abs() <= 0.06;
            let same_bpm_beatmatched =
                (ba - folded_b).abs() <= 0.75 && beatmatchable_genre && fade_out_dur_ms >= 4_000;

            if bpm_ratio <= config.max_bpm_ratio.max(1.0)
                && (durations_match_bpm || same_bpm_beatmatched || beatmatchable_genre)
            {
                let beats_per_bar = track_a
                    .meta
                    .beats
                    .as_ref()
                    .map(|g| g.beats_per_bar)
                    .filter(|&b| b == 3 || b == 4)
                    .unwrap_or(4);
                let raw_beats = (fade_out_dur_ms as f64 / 1000.0) * ba / 60.0;
                let nearest_bar_beats =
                    (raw_beats / f64::from(beats_per_bar)).round() * f64::from(beats_per_bar);
                let snapped_beats = if nearest_bar_beats >= f64::from(beats_per_bar)
                    && (raw_beats - nearest_bar_beats).abs() <= 0.75
                {
                    Some(nearest_bar_beats)
                } else if raw_beats >= 4.0 {
                    Some(raw_beats.round())
                } else {
                    None
                };

                let avg_dur_ms = (fade_out_dur_ms + fade_in_dur_ms) / 2;
                (
                    Some(TempoPlan {
                        bpm_a: ba,
                        bpm_b: bb,
                        beatmatched: true,
                        max_bpm_ratio: config.max_bpm_ratio,
                        beats_a: track_a.meta.beats.clone(),
                        beats_b: track_b.meta.beats.clone(),
                        stretch_mode: config.stretch_mode,
                    }),
                    snapped_beats,
                    avg_dur_ms,
                )
            } else {
                (None, None, fade_overlap_ms.unwrap_or(fade_out_dur_ms))
            }
        }
        _ => (None, None, fade_overlap_ms.unwrap_or(fade_out_dur_ms)),
    };

    Some(TransitionSpec {
        start_a_frame: ms_to_frames(start_a_ms),
        start_b_frame: ms_to_frames(start_b_ms),
        duration_frames: ms_to_frames(duration_ms),
        duration_beats,
        handoff_fraction: 0.5,
        tempo,
        out_curves,
        in_curves,
        eq_topology: config.eq_topology,
    })
}

/// Estimate effective BPM from `AudioAttributes`, `Cuepoints`, or median `BeatGrid` spacing.
fn effective_bpm(meta: &TrackMixMetadata) -> Option<f64> {
    if let Some(bpm) = meta.bpm.filter(|&b| (30.0..=260.0).contains(&b)) {
        return Some(bpm);
    }
    if let Some(ref cues) = meta.cuepoints {
        for c in cues
            .best_fade_out_cuepoint
            .as_ref()
            .into_iter()
            .chain(cues.best_fade_in_cuepoint.as_ref())
        {
            let b = f64::from(c.tempo_bpm);
            if (30.0..=260.0).contains(&b) {
                return Some(b);
            }
        }
    }
    if let Some(ref grid) = meta.beats {
        if grid.beats.len() >= 4 {
            let mut durs: Vec<f64> = grid
                .beats
                .windows(2)
                .map(|w| w[1].time_sec - w[0].time_sec)
                .filter(|&d| (0.2..=2.0).contains(&d))
                .collect();
            if !durs.is_empty() {
                durs.sort_by(f64::total_cmp);
                let med = durs[durs.len() / 2];
                return Some(60.0 / med);
            }
        }
    }
    None
}

/// Snap a cue timestamp (seconds) to the nearest high-confidence downbeat (`beat_in_bar == 1`)
/// within `max_snap_sec`, falling back to nearest beat if no downbeat is nearby.
fn snap_to_downbeat(grid: &BeatGrid, target_sec: f64, max_snap_sec: f64) -> f64 {
    if grid.beats.is_empty() {
        return target_sec;
    }
    let best_downbeat = grid
        .beats
        .iter()
        .filter(|b| b.beat_in_bar == 1 && (b.time_sec - target_sec).abs() <= max_snap_sec)
        .min_by(|a, b| {
            let sa = (a.time_sec - target_sec).abs() - f64::from(a.downbeat_confidence) * 0.25;
            let sb = (b.time_sec - target_sec).abs() - f64::from(b.downbeat_confidence) * 0.25;
            sa.total_cmp(&sb)
        });
    if let Some(db) = best_downbeat {
        return db.time_sec;
    }
    grid.beats
        .iter()
        .filter(|b| (b.time_sec - target_sec).abs() <= max_snap_sec)
        .min_by(|a, b| {
            (a.time_sec - target_sec)
                .abs()
                .total_cmp(&(b.time_sec - target_sec).abs())
        })
        .map(|b| b.time_sec)
        .unwrap_or(target_sec)
}

/// Evaluate vocal probability in `[0.0, 1.0]` at timestamp `sec`.
fn vocal_prob_at(va: &VocalActivity, sec: f64) -> f64 {
    if va.vocal_activity_probabilities.is_empty()
        || va.source_sample_rate_hz <= 0.0
        || va.samples_between_windows <= 0
    {
        return 0.0;
    }
    let sample = sec * f64::from(va.source_sample_rate_hz);
    let idx = ((sample - f64::from(va.first_window_sample_start))
        / f64::from(va.samples_between_windows))
    .round() as isize;
    if idx < 0 || idx as usize >= va.vocal_activity_probabilities.len() {
        return 0.0;
    }
    let max_val = va
        .vocal_activity_probabilities
        .iter()
        .copied()
        .max()
        .unwrap_or(100);
    let scale = if max_val > 100 { 1000.0 } else { 100.0 };
    (f64::from(va.vocal_activity_probabilities[idx as usize]) / scale).clamp(0.0, 1.0)
}

/// Estimate vocal clash energy `mean_t(v_a(t) * v_b(t))` over an overlap window.
fn vocal_clash_score(
    va_a: Option<&VocalActivity>,
    va_b: Option<&VocalActivity>,
    start_a_sec: f64,
    start_b_sec: f64,
    dur_sec: f64,
) -> f64 {
    let (Some(a), Some(b)) = (va_a, va_b) else {
        return 0.0;
    };
    const STEPS: usize = 16;
    let mut sum = 0.0;
    for i in 0..STEPS {
        let dt = (i as f64 + 0.5) / STEPS as f64 * dur_sec;
        sum += vocal_prob_at(a, start_a_sec + dt) * vocal_prob_at(b, start_b_sec + dt);
    }
    sum / STEPS as f64
}

fn collect_out_cues(meta: &TrackMixMetadata, dur_a_sec: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if let Some(ref cues) = meta.cuepoints {
        if let Some(best) = cues.best_fade_out_cuepoint.as_ref() {
            out.push((
                best.position_ms.max(0) as f64 / 1000.0,
                cue_bonus(best, true),
            ));
        }
        for c in &cues.fade_out_cuepoints {
            let sec = c.position_ms.max(0) as f64 / 1000.0;
            if out.iter().all(|&(t, _)| (t - sec).abs() > 0.5) {
                out.push((sec, cue_bonus(c, false)));
            }
        }
    }
    if out.is_empty() && dur_a_sec > 20.0 {
        // Fallback fade-out END cue near track tail (`start_a = end_a - dur_sec`).
        out.push(((dur_a_sec - 4.0).max(12.0), 0.3));
    }
    out
}

fn collect_in_cues(meta: &TrackMixMetadata) -> Vec<(f64, f64)> {
    let mut inc = Vec::new();
    if let Some(ref cues) = meta.cuepoints {
        if let Some(best) = cues.best_fade_in_cuepoint.as_ref() {
            inc.push((
                best.position_ms.max(0) as f64 / 1000.0,
                cue_bonus(best, true),
            ));
        }
        for c in &cues.fade_in_cuepoints {
            let sec = c.position_ms.max(0) as f64 / 1000.0;
            if inc.iter().all(|&(t, _)| (t - sec).abs() > 0.5) {
                inc.push((sec, cue_bonus(c, false)));
            }
        }
    }
    if inc.is_empty() {
        inc.push((0.0, 0.3));
    }
    inc
}

fn cue_bonus(c: &Cuepoint, is_best: bool) -> f64 {
    let mut s = c.confidence.clamp(0.0, 1.0) * 0.5;
    if is_best {
        s += 0.35;
    }
    if c.origin.enum_value_or_default() == Origin::HUMAN {
        s += 0.25;
    }
    s
}

/// Synthesize an Auto DJ transition from track analysis metadata (`BEATS`, `CUEPOINTS`,
/// `VOCAL_ACTIVITY`, `AUDIO_ATTRIBUTES_V2`, `MIXABILITY`), mirroring Spotify's `core-automix` rules.
pub fn synthesize_auto_transition(
    meta_a: &TrackMixMetadata,
    meta_b: &TrackMixMetadata,
    config: &MixConfig,
) -> Option<TransitionSpec> {
    if meta_a.mixable == Some(false) || meta_b.mixable == Some(false) {
        return None;
    }

    let dur_a_sec = f64::from(meta_a.duration_ms) / 1000.0;
    let dur_b_sec = f64::from(meta_b.duration_ms) / 1000.0;
    if dur_a_sec > 0.0 && dur_a_sec < 12.0 {
        return None;
    }

    let bpm_a = effective_bpm(meta_a);
    let bpm_b = effective_bpm(meta_b);
    let beatmatchable_genre = meta_a.beatmatchability.unwrap_or(1.0) >= 0.25
        && meta_b.beatmatchability.unwrap_or(1.0) >= 0.25;

    let beatmatch_bpm = match (bpm_a, bpm_b) {
        (Some(ba), Some(bb))
            if beatmatchable_genre && config.stretch_mode != TimeStretchMode::Off =>
        {
            let folded_b = fold_bpm_octave(ba, bb);
            let ratio = (folded_b / ba).max(ba / folded_b);
            if ratio <= config.max_bpm_ratio.max(1.0) {
                Some((ba, bb, folded_b))
            } else {
                None
            }
        }
        _ => None,
    };

    let key_tier =
        camelot_compatibility_tier(meta_a.camelot_key.as_deref(), meta_b.camelot_key.as_deref());

    let out_cues = collect_out_cues(meta_a, dur_a_sec);
    let in_cues = collect_in_cues(meta_b);
    if out_cues.is_empty() || in_cues.is_empty() {
        return None;
    }

    if let Some((ba, bb, folded_b)) = beatmatch_bpm {
        let beats_per_bar = meta_a
            .beats
            .as_ref()
            .map(|g| g.beats_per_bar)
            .filter(|&b| b == 3 || b == 4)
            .unwrap_or(4);
        let avg_bpm = 0.5 * (ba + folded_b);
        let sec_per_bar = f64::from(beats_per_bar) * 60.0 / avg_bpm;

        // Harmonic clash prefers tighter overlap (4 bars); harmonic matches prefer 8-bar DJ blends.
        let bar_candidates: &[u32] = if key_tier >= 3 { &[4, 8] } else { &[8, 4, 16] };

        let mut best: Option<(f64, f64, f64, u32)> = None;
        for &(raw_end_a, score_a) in &out_cues {
            let end_a = if dur_a_sec > 0.0 {
                raw_end_a.min(dur_a_sec)
            } else {
                raw_end_a
            };
            for &(raw_b, score_b) in &in_cues {
                let start_b = meta_b
                    .beats
                    .as_ref()
                    .map(|g| snap_to_downbeat(g, raw_b, 1.5))
                    .unwrap_or(raw_b);
                for &bars in bar_candidates {
                    let dur_sec = f64::from(bars) * sec_per_bar;
                    if dur_sec > MAX_OVERLAP_SEC {
                        continue;
                    }
                    // Spotify `fade_out_cuepoint` marks the END of Track A's fade-out;
                    // back-compute `start_a` in Track A's native tempo clock before snapping.
                    let dur_a_native = f64::from(bars * beats_per_bar) * 60.0 / ba;
                    let raw_start_a = end_a - dur_a_native;
                    if raw_start_a < 4.0 {
                        continue;
                    }
                    let start_a = meta_a
                        .beats
                        .as_ref()
                        .map(|g| snap_to_downbeat(g, raw_start_a, 1.5))
                        .unwrap_or(raw_start_a);
                    if start_a < 4.0 {
                        continue;
                    }
                    if dur_a_sec > 0.0 && start_a + dur_a_native > dur_a_sec + 0.5 {
                        continue;
                    }
                    if dur_b_sec > 0.0 && start_b + dur_sec > dur_b_sec * 0.6 {
                        continue;
                    }

                    let clash = vocal_clash_score(
                        meta_a.vocal_activity.as_ref(),
                        meta_b.vocal_activity.as_ref(),
                        start_a,
                        start_b,
                        dur_sec,
                    );
                    let bar_pref = match (bars, key_tier >= 3) {
                        (8, false) => 0.30,
                        (4, true) => 0.30,
                        (4, false) => 0.15,
                        (16, false) => 0.10,
                        _ => 0.0,
                    };
                    let total_score = score_a + score_b + bar_pref - clash * 1.4;
                    if best.is_none_or(|(bs, _, _, _)| total_score > bs) {
                        best = Some((total_score, start_a, start_b, bars));
                    }
                }
            }
        }

        if let Some((_, start_a, start_b, bars)) = best {
            let preset = if key_tier >= 3 {
                BuiltinPreset::BassSwapHpFilterIn
            } else {
                BuiltinPreset::BassSwap
            };
            let (out_curves, in_curves) = curves_for_builtin_preset(preset);
            let dur_sec = f64::from(bars) * sec_per_bar;
            return Some(TransitionSpec {
                start_a_frame: sec_to_frames(start_a),
                start_b_frame: sec_to_frames(start_b),
                duration_frames: sec_to_frames(dur_sec),
                duration_beats: Some(f64::from(bars) * f64::from(beats_per_bar)),
                handoff_fraction: preset.handoff_fraction(),
                tempo: Some(TempoPlan {
                    bpm_a: ba,
                    bpm_b: bb,
                    beatmatched: true,
                    max_bpm_ratio: config.max_bpm_ratio,
                    beats_a: meta_a.beats.clone(),
                    beats_b: meta_b.beats.clone(),
                    stretch_mode: config.stretch_mode,
                }),
                out_curves,
                in_curves,
                eq_topology: config.eq_topology,
            });
        }
    }

    // Non-beatmatched fallback (e.g. BPM ratio > max_bpm_ratio): smooth equal-power fade
    // ending at Track A's `fade_out_cuepoint`.
    let raw_end_a = out_cues[0].0;
    let end_a = if dur_a_sec > 0.0 {
        raw_end_a.min(dur_a_sec)
    } else {
        raw_end_a
    };
    let fade_ms = DEFAULT_FALLBACK_FADE_MS.clamp(2_000, 6_000);
    let fade_sec = fade_ms as f64 / 1000.0;
    if end_a <= fade_sec + 1.0 {
        return None;
    }
    let start_a = end_a - fade_sec;
    let start_b = in_cues[0].0;

    let preset = BuiltinPreset::SimpleFade;
    let (out_curves, in_curves) = curves_for_builtin_preset(preset);
    Some(TransitionSpec {
        start_a_frame: sec_to_frames(start_a),
        start_b_frame: sec_to_frames(start_b),
        duration_frames: ms_to_frames(fade_ms),
        duration_beats: None,
        handoff_fraction: preset.handoff_fraction(),
        tempo: None,
        out_curves,
        in_curves,
        eq_topology: config.eq_topology,
    })
}

#[derive(Default)]
struct ResolverCache {
    tracks: HashMap<String, TrackMixMetadata>,
    track_order: VecDeque<String>,
    /// Cached `MIX_STATE.is_mixed` per playlist URI.
    playlist_mixed: HashMap<String, bool>,
    playlist_order: VecDeque<String>,
    /// Cached per-playlist user-made recipes keyed by `(playlist_uri, track_a_uri, track_b_uri)`.
    playlist_recipes: HashMap<String, Vec<Transition>>,
}

impl ResolverCache {
    fn insert_track(&mut self, uri: String, meta: TrackMixMetadata) {
        if !self.tracks.contains_key(&uri) {
            if self.track_order.len() >= MAX_CACHED_TRACKS {
                if let Some(evict) = self.track_order.pop_front() {
                    self.tracks.remove(&evict);
                }
            }
            self.track_order.push_back(uri.clone());
        }
        self.tracks.insert(uri, meta);
    }

    fn insert_playlist(&mut self, uri: String, is_mixed: bool, recipes: Vec<Transition>) {
        if !self.playlist_mixed.contains_key(&uri) {
            if self.playlist_order.len() >= MAX_CACHED_CONTEXTS {
                if let Some(evict) = self.playlist_order.pop_front() {
                    self.playlist_mixed.remove(&evict);
                    self.playlist_recipes.remove(&evict);
                }
            }
            self.playlist_order.push_back(uri.clone());
        }
        self.playlist_mixed.insert(uri.clone(), is_mixed);
        self.playlist_recipes.insert(uri, recipes);
    }

    fn invalidate_playlist(&mut self, uri: &str) {
        self.playlist_mixed.remove(uri);
        self.playlist_recipes.remove(uri);
    }
}

/// Async transition resolver owned by `SpircTask` with bounded metadata caching.
#[derive(Clone)]
pub struct TransitionResolver {
    config: MixConfig,
    cache: Arc<Mutex<ResolverCache>>,
}

impl TransitionResolver {
    /// Construct a new resolver with the given [`MixConfig`].
    pub fn new(config: MixConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(ResolverCache::default())),
        }
    }

    /// Return the active [`MixConfig`].
    pub fn config(&self) -> &MixConfig {
        &self.config
    }

    /// Drop cached playlist mix state and user-made recipes after a live playlist modification.
    pub async fn invalidate_playlist(&self, playlist_uri: &str) {
        self.cache.lock().await.invalidate_playlist(playlist_uri);
    }

    /// Resolve a [`TransitionSpec`] for `(current -> next)` in the background.
    pub async fn resolve(
        &self,
        session: &Session,
        req: TransitionResolveRequest,
    ) -> TransitionResolveResult {
        let mut result = TransitionResolveResult {
            generation: req.generation,
            current_track_id: req.current_track_id.clone(),
            next_track_id: req.next_track_id.clone(),
            spec: None,
        };

        if self.config.mode == MixMode::Off {
            return result;
        }

        let uri_a = req.current_track_id.to_uri();
        let uri_b = req.next_track_id.to_uri();

        // 1. Determine whether Mix is active for this context (and fetch playlist recipes if playlist).
        let (is_mixed, playlist_recipes) = self.check_context_mixed(session, &req).await;
        if !is_mixed {
            return result;
        }

        // 2. Fetch / read cached per-track mix analysis metadata for Track A and Track B.
        let (mut meta_a, mut meta_b) = self
            .fetch_track_pair_metadata(session, &uri_a, &uri_b)
            .await;
        if req.current_duration_ms > 0 {
            meta_a.duration_ms = req.current_duration_ms;
        }
        if req.next_duration_ms > 0 {
            meta_b.duration_ms = req.next_duration_ms;
        }

        // 3. Priority 1: explicit user-made `transition_recipe` from track metadata or playlist4.
        if let Some(proto) = find_track_metadata_recipe(&req.current_metadata, &req.next_metadata)
            .or_else(|| {
                find_matching_playlist_recipe(
                    &playlist_recipes,
                    &uri_a,
                    &req.current_row_uid,
                    &uri_b,
                    &req.next_row_uid,
                )
            })
        {
            if let Some(spec) = transition_proto_to_spec(
                &proto,
                TrackTransitionContext {
                    track_uri: &uri_a,
                    row_uid: &req.current_row_uid,
                    meta: &meta_a,
                },
                TrackTransitionContext {
                    track_uri: &uri_b,
                    row_uid: &req.next_row_uid,
                    meta: &meta_b,
                },
                &self.config,
            ) {
                debug!("Resolved user-made transition recipe for <{uri_a}> -> <{uri_b}>");
                result.spec = Some(spec);
                return result;
            }
        }

        // 4. Priority 2: pre-computed `audio.fade_*` JSON curves & timing transferred from Connect.
        if let Some(spec) = transition_from_fade_metadata(
            TrackTransitionContext {
                track_uri: &uri_a,
                row_uid: &req.current_row_uid,
                meta: &meta_a,
            },
            &req.current_metadata,
            TrackTransitionContext {
                track_uri: &uri_b,
                row_uid: &req.next_row_uid,
                meta: &meta_b,
            },
            &req.next_metadata,
            &self.config,
        ) {
            debug!(
                "Resolved transferred audio.fade_* transition for <{uri_a}> -> <{uri_b}> (beatmatched={}, start_a_frame={}, start_b_frame={}, dur_frames={})",
                spec.tempo.as_ref().is_some_and(|t| t.beatmatched),
                spec.start_a_frame,
                spec.start_b_frame,
                spec.duration_frames
            );
            result.spec = Some(spec);
            return result;
        }

        // 5. Priority 3: client-side Auto DJ synthesizer from Spotify track analysis extensions.
        if let Some(spec) = synthesize_auto_transition(&meta_a, &meta_b, &self.config) {
            debug!(
                "Synthesized Auto transition for <{uri_a}> -> <{uri_b}> (beatmatched={}, start_a_frame={}, dur_frames={})",
                spec.tempo.as_ref().is_some_and(|t| t.beatmatched),
                spec.start_a_frame,
                spec.duration_frames
            );
            result.spec = Some(spec);
        }

        result
    }

    async fn check_context_mixed(
        &self,
        session: &Session,
        req: &TransitionResolveRequest,
    ) -> (bool, Vec<Transition>) {
        let is_playlist = req.context_uri.starts_with("spotify:playlist:");
        if !is_playlist {
            let active = self.config.mode == MixMode::ForceAuto || req.context_signals_mixed;
            return (active, Vec::new());
        }

        {
            let guard = self.cache.lock().await;
            if let Some(&cached_mixed) = guard.playlist_mixed.get(&req.context_uri) {
                let active = self.config.mode == MixMode::ForceAuto
                    || req.context_signals_mixed
                    || cached_mixed;
                let recipes = guard
                    .playlist_recipes
                    .get(&req.context_uri)
                    .cloned()
                    .unwrap_or_default();
                return (active, recipes);
            }
        }

        // Query MIX_STATE on the playlist URI via extended-metadata.
        let mut is_mixed = req.context_signals_mixed;
        if let Ok(entries) = session
            .spclient()
            .get_extended_metadata_entries(vec![(
                req.context_uri.clone(),
                vec![ExtensionKind::MIX_STATE],
            )])
            .await
        {
            for entry in entries {
                if entry.kind == ExtensionKind::MIX_STATE {
                    if let Ok(ms) = MixState::parse_from_bytes(&entry.data) {
                        is_mixed |= ms.is_mixed;
                    }
                }
            }
        }

        // If mixed (or ForceAuto), inspect playlist4 attributes for any user-made transition recipes.
        let mut recipes = Vec::new();
        if is_mixed || self.config.mode == MixMode::ForceAuto {
            if let Ok(pl_uri) = SpotifyUri::from_uri(&req.context_uri) {
                if let Ok(pl_id) = SpotifyId::try_from(&pl_uri) {
                    if let Ok(raw) = session.spclient().get_playlist(&pl_id).await {
                        if let Ok(pl) = SelectedListContent::parse_from_bytes(&raw) {
                            extract_playlist_recipes(&pl, &mut is_mixed, &mut recipes);
                        }
                    }
                }
            }
        }

        self.cache
            .lock()
            .await
            .insert_playlist(req.context_uri.clone(), is_mixed, recipes.clone());

        let active = self.config.mode == MixMode::ForceAuto || is_mixed;
        (active, recipes)
    }

    async fn fetch_track_pair_metadata(
        &self,
        session: &Session,
        uri_a: &str,
        uri_b: &str,
    ) -> (TrackMixMetadata, TrackMixMetadata) {
        let (cached_a, cached_b) = {
            let guard = self.cache.lock().await;
            (
                guard.tracks.get(uri_a).cloned(),
                guard.tracks.get(uri_b).cloned(),
            )
        };

        const TRACK_KINDS: &[ExtensionKind] = &[
            ExtensionKind::AUDIO_ATTRIBUTES_V2,
            ExtensionKind::BEATS,
            ExtensionKind::CUEPOINTS,
            ExtensionKind::VOCAL_ACTIVITY,
            ExtensionKind::MIXABILITY,
            ExtensionKind::AUTOMIX_MODE,
        ];

        let mut to_fetch: Vec<(String, Vec<ExtensionKind>)> = Vec::new();
        let mut requested_uris: Vec<String> = Vec::new();
        if cached_a.is_none() {
            to_fetch.push((uri_a.to_string(), TRACK_KINDS.to_vec()));
            requested_uris.push(uri_a.to_string());
        }
        if cached_b.is_none() && uri_b != uri_a {
            to_fetch.push((uri_b.to_string(), TRACK_KINDS.to_vec()));
            requested_uris.push(uri_b.to_string());
        }

        if !to_fetch.is_empty() {
            match session
                .spclient()
                .get_extended_metadata_entries(to_fetch)
                .await
            {
                Ok(entries) => {
                    let mut parsed: HashMap<String, TrackMixMetadata> = HashMap::new();
                    for u in requested_uris {
                        parsed.insert(u, TrackMixMetadata::default());
                    }
                    for entry in entries {
                        let meta = parsed.entry(entry.entity_uri.clone()).or_default();
                        apply_extension_to_track_meta(meta, entry.kind, &entry.data);
                    }
                    let mut guard = self.cache.lock().await;
                    for (u, m) in parsed {
                        guard.insert_track(u, m);
                    }
                }
                Err(e) => {
                    debug!("Failed to fetch extended metadata for mix transition: {e}");
                }
            }
        }

        let guard = self.cache.lock().await;
        (
            guard.tracks.get(uri_a).cloned().unwrap_or_default(),
            guard.tracks.get(uri_b).cloned().unwrap_or_default(),
        )
    }
}

fn apply_extension_to_track_meta(meta: &mut TrackMixMetadata, kind: ExtensionKind, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    match kind {
        ExtensionKind::AUDIO_ATTRIBUTES_V2 => {
            if let Ok(aa) = AudioAttributes::parse_from_bytes(bytes) {
                if aa.bpm > 20.0 {
                    meta.bpm = Some(aa.bpm);
                }
                if let Some(k) = aa.key.as_ref() {
                    if let Some(c) = k.camelot_key.as_ref() {
                        if !c.value.is_empty() {
                            meta.camelot_key = Some(c.value.clone());
                        }
                    }
                }
            }
        }
        ExtensionKind::BEATS => {
            if let Ok(b) = Beats::parse_from_bytes(bytes) {
                let beats: Vec<BeatMarker> = b
                    .beats
                    .iter()
                    .map(|m| BeatMarker {
                        time_sec: f64::from(m.time),
                        duration_sec: f64::from(m.duration),
                        beat_in_bar: m.value,
                        beat_confidence: m.beat_confidence,
                        downbeat_confidence: m.downbeat_confidence,
                    })
                    .collect();
                if !beats.is_empty() {
                    meta.beats = Some(BeatGrid {
                        beats_per_bar: b.beats_per_bar.max(1),
                        beats,
                    });
                }
                if !b.beats_hash.is_empty() {
                    meta.beats_hash = Some(b.beats_hash);
                }
            }
        }
        ExtensionKind::CUEPOINTS => {
            if let Ok(c) = Cuepoints::parse_from_bytes(bytes) {
                meta.cuepoints = Some(c);
            }
        }
        ExtensionKind::VOCAL_ACTIVITY => {
            if let Ok(v) = VocalActivity::parse_from_bytes(bytes) {
                meta.vocal_activity = Some(v);
            }
        }
        ExtensionKind::MIXABILITY => {
            if let Ok(m) = Mixability::parse_from_bytes(bytes) {
                meta.mixable = Some(m.mixable);
                meta.beatmatchability = Some(m.genre_based_beatmatchability);
            }
        }
        ExtensionKind::AUTOMIX_MODE => {
            if let Ok(m) = AutomixMode::parse_from_bytes(bytes) {
                meta.automix_style = Some(m.style.enum_value_or_default());
            }
        }
        _ => {}
    }
}

fn find_track_metadata_recipe(
    meta_a: &HashMap<String, String>,
    meta_b: &HashMap<String, String>,
) -> Option<Transition> {
    for map in [meta_a, meta_b] {
        for &key in RECIPE_METADATA_KEYS {
            if let Some(val) = map.get(key) {
                if let Some(tr) = decode_transition_recipe(val) {
                    return Some(tr);
                }
            }
        }
    }
    None
}

fn find_matching_playlist_recipe(
    recipes: &[Transition],
    uri_a: &str,
    row_a_uid: &str,
    uri_b: &str,
    row_b_uid: &str,
) -> Option<Transition> {
    recipes
        .iter()
        .find(|tr| {
            let Some(ov) = tr.overlap.as_ref() else {
                return false;
            };
            let uri_match = (ov.track_a_uri.is_empty() || ov.track_a_uri == uri_a)
                && (ov.track_b_uri.is_empty() || ov.track_b_uri == uri_b);
            let row_match = (ov.track_a_row_id.is_empty()
                || row_a_uid.is_empty()
                || ov.track_a_row_id == row_a_uid)
                && (ov.track_b_row_id.is_empty()
                    || row_b_uid.is_empty()
                    || ov.track_b_row_id == row_b_uid);
            uri_match && row_match && (!ov.track_a_uri.is_empty() || !ov.track_a_row_id.is_empty())
        })
        .cloned()
}

fn extract_playlist_recipes(
    pl: &SelectedListContent,
    is_mixed: &mut bool,
    recipes: &mut Vec<Transition>,
) {
    if let Some(attrs) = pl.attributes.as_ref() {
        for fa in &attrs.format_attributes {
            let key = fa.key.as_deref().unwrap_or_default();
            let val = fa.value.as_deref().unwrap_or_default();
            if (key.contains("mix") || key.contains("automix"))
                && (val.eq_ignore_ascii_case("true")
                    || val.eq_ignore_ascii_case("mixed")
                    || val == "1")
            {
                *is_mixed = true;
            }
            if key.contains("transition") {
                if let Some(tr) = decode_transition_recipe(val) {
                    *is_mixed = true;
                    recipes.push(tr);
                }
            }
        }
    }

    if let Some(contents) = pl.contents.as_ref() {
        for item in &contents.items {
            if let Some(attrs) = item.attributes.as_ref() {
                for fa in &attrs.format_attributes {
                    let key = fa.key.as_deref().unwrap_or_default();
                    let val = fa.value.as_deref().unwrap_or_default();
                    if key.contains("transition") {
                        if let Some(tr) = decode_transition_recipe(val) {
                            *is_mixed = true;
                            recipes.push(tr);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::transition::{Overlap, Preset};
    use protobuf::MessageField;

    fn regular_beat_grid(bpm: f64, duration_sec: f64) -> BeatGrid {
        let dt = 60.0 / bpm;
        let count = (duration_sec / dt) as usize;
        let beats = (0..count)
            .map(|i| BeatMarker {
                time_sec: i as f64 * dt,
                duration_sec: dt,
                beat_in_bar: (i as u32 % 4) + 1,
                beat_confidence: 0.95,
                downbeat_confidence: if i % 4 == 0 { 0.95 } else { 0.1 },
            })
            .collect();
        BeatGrid {
            beats_per_bar: 4,
            beats,
        }
    }

    #[test]
    fn camelot_wheel_handles_exact_adjacent_relative_and_wraparound() {
        assert_eq!(camelot_compatibility_tier(Some("2B"), Some("2B")), 0);
        assert_eq!(camelot_compatibility_tier(Some("2B"), Some("3B")), 1);
        assert_eq!(camelot_compatibility_tier(Some("2B"), Some("2A")), 1);
        assert_eq!(camelot_compatibility_tier(Some("12A"), Some("1A")), 1);
        assert_eq!(camelot_compatibility_tier(Some("3A"), Some("9B")), 3);
    }

    #[test]
    fn user_transition_recipe_decodes_and_rejects_mismatched_playlist_rows() {
        let proto = Transition {
            overlap: MessageField::some(Overlap {
                track_a_uri: "spotify:track:aaa".to_string(),
                track_b_uri: "spotify:track:bbb".to_string(),
                track_a_row_id: "row-1".to_string(),
                track_b_row_id: "row-2".to_string(),
                start_a_ms: 170_000,
                start_b_ms: 8_000,
                duration_ms: 13_000,
                duration_bars: Some(8),
                is_beatmatched: true,
                bpm_a: 145.0,
                bpm_b: 150.0,
                ..Default::default()
            }),
            preset: MessageField::some(Preset {
                id: 3,
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = BASE64.encode(&proto.write_to_bytes().unwrap());
        let decoded = decode_transition_recipe(&encoded).expect("valid recipe must decode");

        let cfg = MixConfig::default();
        let meta = TrackMixMetadata::default();

        // Matching pair & rows succeeds.
        let spec = transition_proto_to_spec(
            &decoded,
            TrackTransitionContext {
                track_uri: "spotify:track:aaa",
                row_uid: "row-1",
                meta: &meta,
            },
            TrackTransitionContext {
                track_uri: "spotify:track:bbb",
                row_uid: "row-2",
                meta: &meta,
            },
            &cfg,
        )
        .expect("matching pair must produce spec");
        assert_eq!(spec.start_a_frame, ms_to_frames(170_000));
        assert_eq!(spec.start_b_frame, ms_to_frames(8_000));
        assert_eq!(spec.duration_beats, Some(32.0));
        assert!(spec.tempo.as_ref().unwrap().beatmatched);

        // Reordered playlist (row-2 moved elsewhere) must reject stale recipe.
        assert!(
            transition_proto_to_spec(
                &decoded,
                TrackTransitionContext {
                    track_uri: "spotify:track:aaa",
                    row_uid: "row-1",
                    meta: &meta,
                },
                TrackTransitionContext {
                    track_uri: "spotify:track:bbb",
                    row_uid: "row-99",
                    meta: &meta,
                },
                &cfg,
            )
            .is_none()
        );
    }

    #[test]
    fn auto_synthesizer_handles_electronic_rising_archetypes() {
        let cfg = MixConfig::default();

        // 1. 145 bpm 2B -> 150 bpm 2B (same key, within +/-8%): 8-bar beatmatched BassSwap snapped to downbeat.
        let meta_145 = TrackMixMetadata {
            duration_ms: 201_000,
            bpm: Some(145.0),
            camelot_key: Some("2B".to_string()),
            beats: Some(regular_beat_grid(145.0, 201.0)),
            cuepoints: Some(Cuepoints {
                best_fade_out_cuepoint: MessageField::some(Cuepoint {
                    position_ms: 175_000,
                    tempo_bpm: 145.0,
                    confidence: 0.9,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            mixable: Some(true),
            beatmatchability: Some(0.95),
            ..Default::default()
        };
        let meta_150 = TrackMixMetadata {
            duration_ms: 256_000,
            bpm: Some(150.0),
            camelot_key: Some("2B".to_string()),
            beats: Some(regular_beat_grid(150.0, 256.0)),
            cuepoints: Some(Cuepoints {
                best_fade_in_cuepoint: MessageField::some(Cuepoint {
                    position_ms: 16_000,
                    tempo_bpm: 150.0,
                    confidence: 0.9,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            mixable: Some(true),
            beatmatchability: Some(0.95),
            ..Default::default()
        };
        let spec_match = synthesize_auto_transition(&meta_145, &meta_150, &cfg).unwrap();
        assert!(spec_match.tempo.as_ref().unwrap().beatmatched);
        assert_eq!(spec_match.duration_beats, Some(32.0));
        assert!(spec_match.out_curves.eq_low.is_some());
        assert!(spec_match.in_curves.filter_cutoff.is_none());

        // 2. 155 bpm 3A -> 153 bpm 9B (clashing tritone key): tighter 4-bar beatmatched filter transition.
        let mut meta_155 = meta_145.clone();
        meta_155.bpm = Some(155.0);
        meta_155.camelot_key = Some("3A".to_string());
        meta_155.beats = Some(regular_beat_grid(155.0, 201.0));
        let mut meta_153 = meta_150.clone();
        meta_153.bpm = Some(153.0);
        meta_153.camelot_key = Some("9B".to_string());
        meta_153.beats = Some(regular_beat_grid(153.0, 256.0));
        let spec_clash = synthesize_auto_transition(&meta_155, &meta_153, &cfg).unwrap();
        assert!(spec_clash.tempo.as_ref().unwrap().beatmatched);
        assert_eq!(spec_clash.duration_beats, Some(16.0));
        assert!(spec_clash.in_curves.filter_cutoff.is_some());

        // 3. 153 bpm 9B -> 125 bpm 1A (BPM ratio 1.22 > 1.08): non-beatmatched equal-power fade.
        let mut meta_125 = meta_150.clone();
        meta_125.bpm = Some(125.0);
        meta_125.camelot_key = Some("1A".to_string());
        meta_125.beats = Some(regular_beat_grid(125.0, 256.0));
        let spec_non_bm = synthesize_auto_transition(&meta_153, &meta_125, &cfg).unwrap();
        assert!(spec_non_bm.tempo.is_none());
        assert!(spec_non_bm.duration_beats.is_none());
    }

    #[test]
    fn fade_out_cuepoint_near_track_end_aligns_as_overlap_end() {
        let cfg = MixConfig::default();
        // Matches real Spotify `CUEPOINTS` where `best_fade_out_cuepoint` sits ~1.6 s before EOF.
        let meta_a = TrackMixMetadata {
            duration_ms: 201_000,
            bpm: Some(145.0),
            camelot_key: Some("2B".to_string()),
            beats: Some(regular_beat_grid(145.0, 201.0)),
            cuepoints: Some(Cuepoints {
                best_fade_out_cuepoint: MessageField::some(Cuepoint {
                    position_ms: 199_421,
                    tempo_bpm: 145.0,
                    confidence: 0.9,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            mixable: Some(true),
            beatmatchability: Some(0.95),
            ..Default::default()
        };
        let meta_b = TrackMixMetadata {
            duration_ms: 256_000,
            bpm: Some(150.0),
            camelot_key: Some("2B".to_string()),
            beats: Some(regular_beat_grid(150.0, 256.0)),
            cuepoints: Some(Cuepoints {
                best_fade_in_cuepoint: MessageField::some(Cuepoint {
                    position_ms: 12_800,
                    tempo_bpm: 150.0,
                    confidence: 0.9,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            mixable: Some(true),
            beatmatchability: Some(0.95),
            ..Default::default()
        };

        let spec = synthesize_auto_transition(&meta_a, &meta_b, &cfg)
            .expect("end-aligned fade_out_cuepoint near EOF must still synthesize");
        let start_a_sec = spec.start_a_frame as f64 / f64::from(SAMPLE_RATE);
        let dur_sec = spec.duration_frames as f64 / f64::from(SAMPLE_RATE);
        // 8 bars at 145 BPM is ~13.24 s, so start_a must land ~186 s (ending near 199.4 s).
        assert!(
            start_a_sec > 180.0 && start_a_sec < 192.0,
            "unexpected start_a_sec={start_a_sec}"
        );
        assert!(
            start_a_sec + dur_sec <= 201.5,
            "transition overflows track A EOF: end={}",
            start_a_sec + dur_sec
        );

        // Non-beatmatched fallback must also back-compute `start_a = end_a - fade_sec`.
        let mut meta_slow = meta_b;
        meta_slow.bpm = Some(110.0);
        meta_slow.beats = Some(regular_beat_grid(110.0, 256.0));
        let spec_fb = synthesize_auto_transition(&meta_a, &meta_slow, &cfg)
            .expect("non-beatmatched fallback must align end at best_fade_out_cuepoint");
        let fb_start_sec = spec_fb.start_a_frame as f64 / f64::from(SAMPLE_RATE);
        let fb_dur_sec = spec_fb.duration_frames as f64 / f64::from(SAMPLE_RATE);
        assert!((fb_start_sec + fb_dur_sec - 199.421).abs() < 0.05);
    }

    #[test]
    fn transferred_audio_fade_json_parses_cubic_bezier_and_validates_pair() {
        use crate::playback::mix::SpecialRole;
        use crate::playback::mix::curve::{apply_semantics, eval_curves};

        let cfg = MixConfig::default();
        let meta_a = TrackMixMetadata {
            duration_ms: 201_000,
            bpm: Some(145.0),
            beats: Some(regular_beat_grid(145.0, 201.0)),
            beatmatchability: Some(0.95),
            ..Default::default()
        };
        let meta_b = TrackMixMetadata {
            duration_ms: 256_000,
            bpm: Some(150.0),
            beats: Some(regular_beat_grid(150.0, 256.0)),
            beatmatchability: Some(0.95),
            ..Default::default()
        };

        let mut map_a = HashMap::new();
        map_a.insert(
            "audio.fade_out_start_time".to_string(),
            "170444".to_string(),
        );
        map_a.insert("audio.fade_out_duration".to_string(), "26483".to_string());
        map_a.insert(
            "audio.fade_out_curves".to_string(),
            r#"[{"start_point":0,"end_point":0.5,"fade_curve":[{"x":0,"y":1},{"x":1,"y":1}]},{"start_point":0.5,"end_point":1,"fade_curve":[{"x":0,"y":1},{"x":1,"y":0}]}]"#.to_string(),
        );
        map_a.insert(
            "audio.fade_out_eq_low_gain_curves".to_string(),
            r#"[{"start_point":0,"end_point":0.49609375,"fade_curve":[{"x":0,"y":0.5},{"x":1,"y":0.5}]},{"start_point":0.49609375,"end_point":0.5,"fade_curve":[{"x":0,"y":0.5},{"x":0.5,"y":0.5},{"x":0.5,"y":0},{"x":1,"y":0}]},{"start_point":0.5,"end_point":1,"fade_curve":[{"x":0,"y":0},{"x":1,"y":0}]}]"#.to_string(),
        );
        map_a.insert(
            "audio.fade_out_filter_cutoff_curves".to_string(),
            r#"[{"start_point":0,"end_point":0.5,"fade_curve":[{"x":0,"y":0.5},{"x":1,"y":0.5}]},{"start_point":0.5,"end_point":1,"fade_curve":[{"x":0,"y":0.5},{"x":0,"y":0.5},{"x":1,"y":1},{"x":1,"y":1}]}]"#.to_string(),
        );
        map_a.insert(
            "audio.fade_out_filter_resonance_curves".to_string(),
            r#"[{"start_point":0,"end_point":1,"fade_curve":[{"x":0,"y":0.5},{"x":1,"y":0.5}]}]"#
                .to_string(),
        );

        let mut map_b = HashMap::new();
        map_b.insert("audio.fade_in_start_time".to_string(), "12800".to_string());
        map_b.insert("audio.fade_in_duration".to_string(), "25600".to_string());
        map_b.insert("audio.fade_overlap".to_string(), "26483".to_string());
        map_b.insert(
            "audio.fade_in_curves".to_string(),
            r#"[{"start_point":0,"end_point":0.5,"fade_curve":[{"x":0,"y":0},{"x":1,"y":1}]},{"start_point":0.5,"end_point":1,"fade_curve":[{"x":0,"y":1},{"x":1,"y":1}]}]"#.to_string(),
        );
        map_b.insert(
            "audio.fade_in_eq_low_gain_curves".to_string(),
            r#"[{"start_point":0,"end_point":0.49609375,"fade_curve":[{"x":0,"y":0},{"x":1,"y":0}]},{"start_point":0.49609375,"end_point":0.5,"fade_curve":[{"x":0,"y":0},{"x":0.5,"y":0},{"x":0.5,"y":0.5},{"x":1,"y":0.5}]},{"start_point":0.5,"end_point":1,"fade_curve":[{"x":0,"y":0.5},{"x":1,"y":0.5}]}]"#.to_string(),
        );

        let spec = transition_from_fade_metadata(
            TrackTransitionContext {
                track_uri: "spotify:track:fixture_a",
                row_uid: "row_a",
                meta: &meta_a,
            },
            &map_a,
            TrackTransitionContext {
                track_uri: "spotify:track:fixture_b",
                row_uid: "row_b",
                meta: &meta_b,
            },
            &map_b,
            &cfg,
        )
        .expect("valid transferred fade metadata must build TransitionSpec");

        assert_eq!(spec.start_a_frame, ms_to_frames(170_444));
        assert_eq!(spec.start_b_frame, ms_to_frames(12_800));
        assert_eq!(spec.duration_beats, Some(64.0));
        assert!(spec.tempo.as_ref().unwrap().beatmatched);

        // Bipolar EQ semantics: 0.5 -> 1.0 linear unity, 0.0 -> Kill.
        let eq_sem = spec.out_curves.eq_semantics.as_ref().unwrap();
        let eq_start = apply_semantics(
            eval_curves(spec.out_curves.eq_low.as_ref().unwrap(), 0.2).unwrap(),
            eq_sem,
        );
        assert!((eq_start.as_linear_gain().unwrap() - 1.0).abs() < 1e-6);
        let eq_end = apply_semantics(
            eval_curves(spec.out_curves.eq_low.as_ref().unwrap(), 0.8).unwrap(),
            eq_sem,
        );
        assert_eq!(eq_end.role, Some(SpecialRole::Kill));

        // Resonance semantics: 0.5 -> neutral Butterworth Q (~0.7071).
        let res_sem = spec.out_curves.filter_resonance_semantics.as_ref().unwrap();
        let res_val = apply_semantics(
            eval_curves(spec.out_curves.filter_resonance.as_ref().unwrap(), 0.5).unwrap(),
            res_sem,
        );
        assert!((res_val.numeric.unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);

        // Pair consistency guard: mismatched `audio.fade_overlap` (e.g. queued track) must reject.
        map_b.insert("audio.fade_overlap".to_string(), "14656".to_string());
        assert!(
            transition_from_fade_metadata(
                TrackTransitionContext {
                    track_uri: "spotify:track:fixture_a",
                    row_uid: "row_a",
                    meta: &meta_a,
                },
                &map_a,
                TrackTransitionContext {
                    track_uri: "spotify:track:fixture_b",
                    row_uid: "row_b",
                    meta: &meta_b,
                },
                &map_b,
                &cfg,
            )
            .is_none()
        );
    }

    #[test]
    fn custom_recipe_handles_halftime_dnb_cut_and_preset_type_fallback() {
        use crate::protocol::transition::{FilterFxStyle, PresetType, VolumeStyle};

        let cfg = MixConfig::default();
        let meta = TrackMixMetadata::default();

        // 1. Half-time non-beatmatched custom cut (174 BPM -> 86.87 BPM, 5000 ms, volume cut + filter 6).
        let mut preset_cut = Preset {
            id: 1,
            volume_style_override: MessageField::some(VolumeStyle {
                id: 1,
                ..Default::default()
            }),
            filter_fx_style_override: MessageField::some(FilterFxStyle {
                id: 6,
                ..Default::default()
            }),
            ..Default::default()
        };
        preset_cut.type_ = protobuf::EnumOrUnknown::new(PresetType::PRESET_TYPE_FADE);
        let proto_cut = Transition {
            overlap: MessageField::some(Overlap {
                track_a_uri: "spotify:track:dnb_a".to_string(),
                track_b_uri: "spotify:track:dnb_b".to_string(),
                track_a_row_id: "r1".to_string(),
                track_b_row_id: "r2".to_string(),
                start_a_ms: 157_043,
                start_b_ms: 62_814,
                duration_ms: 5_000,
                speed_a: Some(1.0),
                speed_b: Some(1.0),
                is_beatmatched: false,
                bpm_a: 174.0,
                bpm_b: 86.87,
                item_speed_a: 1.0,
                item_speed_b: 1.0,
                ..Default::default()
            }),
            preset: MessageField::some(preset_cut),
            is_overlap_override: true,
            ..Default::default()
        };
        let spec_cut = transition_proto_to_spec(
            &proto_cut,
            TrackTransitionContext {
                track_uri: "spotify:track:dnb_a",
                row_uid: "r1",
                meta: &meta,
            },
            TrackTransitionContext {
                track_uri: "spotify:track:dnb_b",
                row_uid: "r2",
                meta: &meta,
            },
            &cfg,
        )
        .expect("custom cut recipe must produce spec");
        assert!(spec_cut.tempo.is_none());
        assert!(spec_cut.out_curves.filter_cutoff.is_some());

        // 2. Newer preset ID (8) with `PRESET_TYPE_BEATMATCHED_FADE` falls back cleanly to BassSwap.
        let mut preset_8 = Preset {
            id: 8,
            ..Default::default()
        };
        preset_8.type_ = protobuf::EnumOrUnknown::new(PresetType::PRESET_TYPE_BEATMATCHED_FADE);
        let proto_bm = Transition {
            overlap: MessageField::some(Overlap {
                track_a_uri: "spotify:track:dnb_b".to_string(),
                track_b_uri: "spotify:track:dnb_c".to_string(),
                start_a_ms: 215_169,
                start_b_ms: 74_481,
                duration_ms: 11_035,
                duration_bars: Some(8),
                is_beatmatched: true,
                bpm_a: 174.0,
                bpm_b: 174.0,
                ..Default::default()
            }),
            preset: MessageField::some(preset_8),
            is_overlap_override: true,
            ..Default::default()
        };
        let spec_bm = transition_proto_to_spec(
            &proto_bm,
            TrackTransitionContext {
                track_uri: "spotify:track:dnb_b",
                row_uid: "",
                meta: &meta,
            },
            TrackTransitionContext {
                track_uri: "spotify:track:dnb_c",
                row_uid: "",
                meta: &meta,
            },
            &cfg,
        )
        .expect("beatmatched preset 8 recipe must produce spec");
        assert!(spec_bm.tempo.as_ref().unwrap().beatmatched);
        assert_eq!(spec_bm.duration_beats, Some(32.0));
        assert!(spec_bm.out_curves.eq_low.is_some());
    }
}
