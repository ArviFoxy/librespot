//! Clean-room MIT automation curve presets and Spotify `Transition.preset` translation.
//!
//! Maps Spotify Mix numeric preset IDs, combined volume/EQ/filter style overrides, and
//! user-edited piecewise automation curve overrides onto [`crate::playback::mix::DeckCurves`].

use std::f64::consts::FRAC_PI_2;

use crate::{
    playback::mix::{
        AutomationCurve, AutomationCurves, CurvePoint, DeckCurves, FilterMode,
        InterpolationSegment, SegmentEasing, SegmentInterpolation, SpecialCase, SpecialResult,
        SpecialRole, ValueMapping, ValueSemantics, ValueUnit,
    },
    protocol::transition::{
        AutomationCurves as ProtoCurves, EqCurveOverrides, FilterCurveOverrides, Preset, PresetType,
    },
};

/// Well-known Spotify Mix preset identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPreset {
    /// Preset 1 (`preset-simple-fade`): smooth equal-power volume crossfade.
    SimpleFade,
    /// Preset 2 (`preset-eq-swap`): full overlap with progressive 3-band EQ swap.
    EqSwap,
    /// Preset 3 (`preset-bass-swap`): full overlap with midpoint sub-bass swap.
    BassSwap,
    /// Preset 4 (`preset-bass-swap-filter-in`): bass swap + high-pass filter sweep in.
    BassSwapHpFilterIn,
    /// Preset 11 (`preset-simple-cut`): crisp phrase-aligned cut at midpoint.
    SimpleCut,
    /// Preset 17 (`preset-bass-swap-low-pass-filter-in`): bass swap + low-pass filter sweep in.
    BassSwapLpFilterIn,
}

impl BuiltinPreset {
    /// Map a numeric `spotify.automix.proto.Preset.id` to a built-in preset.
    pub fn from_id(id: i32) -> Option<Self> {
        match id {
            1 => Some(Self::SimpleFade),
            2 => Some(Self::EqSwap),
            3 => Some(Self::BassSwap),
            4 => Some(Self::BassSwapHpFilterIn),
            11 => Some(Self::SimpleCut),
            17 => Some(Self::BassSwapLpFilterIn),
            _ => None,
        }
    }

    /// Parse a human-readable preset name (for CLI/offline tooling).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "fade" | "simple-fade" | "preset-simple-fade" => Some(Self::SimpleFade),
            "eq-swap" | "rise" | "preset-eq-swap" => Some(Self::EqSwap),
            "bass-swap" | "blend" | "preset-bass-swap" => Some(Self::BassSwap),
            "hp-filter" | "filter" | "bass-swap-filter-in" | "preset-bass-swap-filter-in" => {
                Some(Self::BassSwapHpFilterIn)
            }
            "cut" | "simple-cut" | "preset-simple-cut" => Some(Self::SimpleCut),
            "lp-filter"
            | "bass-swap-low-pass-filter-in"
            | "preset-bass-swap-low-pass-filter-in" => Some(Self::BassSwapLpFilterIn),
            _ => None,
        }
    }

    /// Recommended Spotify Connect track handoff fraction in `[0.0, 1.0]` for this preset.
    pub fn handoff_fraction(self) -> f64 {
        match self {
            Self::SimpleCut => 0.5,
            Self::SimpleFade
            | Self::EqSwap
            | Self::BassSwap
            | Self::BassSwapHpFilterIn
            | Self::BassSwapLpFilterIn => 0.5,
        }
    }
}

/// Build `(outgoing_curves, incoming_curves)` for a clean-room built-in preset.
pub fn curves_for_builtin_preset(preset: BuiltinPreset) -> (DeckCurves, DeckCurves) {
    match preset {
        BuiltinPreset::SimpleFade => {
            let (vol_out, vol_in) = volume_curves_for_style(2);
            (
                DeckCurves {
                    volume: Some(vol_out),
                    ..Default::default()
                },
                DeckCurves {
                    volume: Some(vol_in),
                    ..Default::default()
                },
            )
        }
        BuiltinPreset::EqSwap => {
            let (vol_out, vol_in) = volume_curves_for_style(7);
            let mut out = DeckCurves {
                volume: Some(vol_out),
                ..Default::default()
            };
            let mut inc = DeckCurves {
                volume: Some(vol_in),
                ..Default::default()
            };
            apply_eq_style(1, &mut out, &mut inc);
            (out, inc)
        }
        BuiltinPreset::BassSwap => {
            let (vol_out, vol_in) = volume_curves_for_style(7);
            let mut out = DeckCurves {
                volume: Some(vol_out),
                ..Default::default()
            };
            let mut inc = DeckCurves {
                volume: Some(vol_in),
                ..Default::default()
            };
            apply_eq_style(4, &mut out, &mut inc);
            (out, inc)
        }
        BuiltinPreset::BassSwapHpFilterIn => {
            let (vol_out, vol_in) = volume_curves_for_style(5);
            let mut out = DeckCurves {
                volume: Some(vol_out),
                ..Default::default()
            };
            let mut inc = DeckCurves {
                volume: Some(vol_in),
                ..Default::default()
            };
            apply_eq_style(4, &mut out, &mut inc);
            apply_filter_style(5, &mut out, &mut inc);
            (out, inc)
        }
        BuiltinPreset::SimpleCut => {
            let (vol_out, vol_in) = volume_curves_for_style(1);
            (
                DeckCurves {
                    volume: Some(vol_out),
                    ..Default::default()
                },
                DeckCurves {
                    volume: Some(vol_in),
                    ..Default::default()
                },
            )
        }
        BuiltinPreset::BassSwapLpFilterIn => {
            let (vol_out, vol_in) = volume_curves_for_style(7);
            let mut out = DeckCurves {
                volume: Some(vol_out),
                ..Default::default()
            };
            let mut inc = DeckCurves {
                volume: Some(vol_in),
                ..Default::default()
            };
            apply_eq_style(4, &mut out, &mut inc);
            apply_filter_style(9, &mut out, &mut inc);
            (out, inc)
        }
    }
}

/// Resolve a Spotify [`Preset`] (base preset ID + optional style overrides + optional custom
/// user-drawn curves) into `(outgoing_curves, incoming_curves, handoff_fraction)`.
pub fn curves_from_proto_preset(preset: &Preset) -> (DeckCurves, DeckCurves, f64) {
    let base = BuiltinPreset::from_id(preset.id).unwrap_or_else(|| {
        match preset.type_.enum_value_or_default() {
            PresetType::PRESET_TYPE_CUT | PresetType::PRESET_TYPE_BEATMATCHED_CUT => {
                BuiltinPreset::SimpleCut
            }
            PresetType::PRESET_TYPE_BEATMATCHED_FADE => BuiltinPreset::BassSwap,
            _ => BuiltinPreset::SimpleFade,
        }
    });

    let (mut out, mut inc) = curves_for_builtin_preset(base);

    if let Some(vol_style) = preset.volume_style_override.as_ref() {
        let (v_out, v_in) = volume_curves_for_style(vol_style.id);
        out.volume = Some(v_out);
        inc.volume = Some(v_in);
    }

    if let Some(eq_style) = preset.eq_style_override.as_ref() {
        clear_eq(&mut out);
        clear_eq(&mut inc);
        apply_eq_style(eq_style.id, &mut out, &mut inc);
    }

    if let Some(filter_style) = preset.filter_fx_style_override.as_ref() {
        clear_filter(&mut out);
        clear_filter(&mut inc);
        apply_filter_style(filter_style.id, &mut out, &mut inc);
    }

    // Apply user-edited curve overrides last so hand-drawn points always win.
    if let Some(v_out) = preset.volume_out_curve_override.as_ref() {
        out.volume = Some(convert_proto_curves(v_out));
        out.volume_semantics = Some(infer_gain_semantics(v_out));
    }
    if let Some(v_in) = preset.volume_in_curve_override.as_ref() {
        inc.volume = Some(convert_proto_curves(v_in));
        inc.volume_semantics = Some(infer_gain_semantics(v_in));
    }

    if let Some(eq_out) = preset.eq_out_curve_overrides.as_ref() {
        apply_eq_override(eq_out, &mut out);
    }
    if let Some(eq_in) = preset.eq_in_curve_overrides.as_ref() {
        apply_eq_override(eq_in, &mut inc);
    }

    if let Some(filt_out) = preset.filter_out_curve_overrides.as_ref() {
        apply_filter_override(filt_out, &mut out);
    }
    if let Some(filt_in) = preset.filter_in_curve_overrides.as_ref() {
        apply_filter_override(filt_in, &mut inc);
    }

    (out, inc, base.handoff_fraction())
}

fn clear_eq(deck: &mut DeckCurves) {
    deck.eq_low = None;
    deck.eq_mid = None;
    deck.eq_high = None;
    deck.eq_semantics = None;
}

fn clear_filter(deck: &mut DeckCurves) {
    deck.filter_cutoff = None;
    deck.filter_cutoff_semantics = None;
    deck.filter_resonance = None;
    deck.filter_resonance_semantics = None;
}

fn curve_from_points(pts: &[(f64, f64)], min: f64, max: f64) -> AutomationCurves {
    AutomationCurves {
        curves: vec![AutomationCurve {
            points: pts.iter().map(|&(x, y)| CurvePoint { x, y }).collect(),
            start: 0.0,
            end: 1.0,
        }],
        minimum: min,
        maximum: max,
    }
}

fn sampled_curve(steps: usize, min: f64, max: f64, f: impl Fn(f64) -> f64) -> AutomationCurves {
    let points: Vec<CurvePoint> = (0..=steps)
        .map(|i| {
            let x = i as f64 / steps as f64;
            CurvePoint {
                x,
                y: f(x).clamp(min, max),
            }
        })
        .collect();
    AutomationCurves {
        curves: vec![AutomationCurve {
            points,
            start: 0.0,
            end: 1.0,
        }],
        minimum: min,
        maximum: max,
    }
}

/// Clean-room volume curve pairs for Spotify's combined volume style IDs.
fn volume_curves_for_style(style_id: i32) -> (AutomationCurves, AutomationCurves) {
    match style_id {
        // 1: cut (sharp cosine switch at midpoint)
        1 => (
            curve_from_points(
                &[(0.0, 1.0), (0.485, 1.0), (0.515, 0.0), (1.0, 0.0)],
                0.0,
                1.0,
            ),
            curve_from_points(
                &[(0.0, 0.0), (0.485, 0.0), (0.515, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ),
        ),
        // 5: slow-in-slow-out (incoming reaches unity at midpoint, outgoing holds unity to midpoint)
        5 => (
            sampled_curve(32, 0.0, 1.0, |u| {
                if u <= 0.5 {
                    1.0
                } else {
                    (((u - 0.5) * 2.0) * FRAC_PI_2).cos()
                }
            }),
            sampled_curve(32, 0.0, 1.0, |u| {
                if u <= 0.5 {
                    ((u * 2.0) * FRAC_PI_2).sin()
                } else {
                    1.0
                }
            }),
        ),
        // 6: cross-shape (gentle S-curve crossfade)
        6 => (
            sampled_curve(32, 0.0, 1.0, |u| {
                let t = ((u - 0.15) / 0.70).clamp(0.0, 1.0);
                (t * FRAC_PI_2).cos()
            }),
            sampled_curve(32, 0.0, 1.0, |u| {
                let t = ((u - 0.15) / 0.70).clamp(0.0, 1.0);
                (t * FRAC_PI_2).sin()
            }),
        ),
        // 7: full-overlap (incoming ramps up early, both stay full in middle, outgoing fades late)
        7 => (
            sampled_curve(32, 0.0, 1.0, |u| {
                if u <= 0.65 {
                    1.0
                } else {
                    (((u - 0.65) / 0.35) * FRAC_PI_2).cos()
                }
            }),
            sampled_curve(32, 0.0, 1.0, |u| {
                if u >= 0.35 {
                    1.0
                } else {
                    ((u / 0.35) * FRAC_PI_2).sin()
                }
            }),
        ),
        // 9: slow-in-fast-out-at-end
        9 => (
            sampled_curve(32, 0.0, 1.0, |u| {
                if u <= 0.88 {
                    1.0
                } else {
                    (((u - 0.88) / 0.12) * FRAC_PI_2).cos()
                }
            }),
            sampled_curve(32, 0.0, 1.0, |u| (u * FRAC_PI_2).sin()),
        ),
        // 10: fast-in-at-start-slow-out
        10 => (
            sampled_curve(32, 0.0, 1.0, |u| (u * FRAC_PI_2).cos()),
            sampled_curve(32, 0.0, 1.0, |u| {
                if u >= 0.12 {
                    1.0
                } else {
                    ((u / 0.12) * FRAC_PI_2).sin()
                }
            }),
        ),
        // 11: slow-in-semi-fast-out-at-end
        11 => (
            sampled_curve(32, 0.0, 1.0, |u| {
                if u <= 0.75 {
                    1.0
                } else {
                    (((u - 0.75) / 0.25) * FRAC_PI_2).cos()
                }
            }),
            sampled_curve(32, 0.0, 1.0, |u| {
                if u >= 0.5 {
                    1.0
                } else {
                    ((u / 0.5) * FRAC_PI_2).sin()
                }
            }),
        ),
        // 2 (crossfade) and fallback: constant-power crossfade
        _ => (
            sampled_curve(32, 0.0, 1.0, |u| (1.0 - u).sqrt()),
            sampled_curve(32, 0.0, 1.0, f64::sqrt),
        ),
    }
}

/// Clean-room 3-band EQ isolator curve pairs for Spotify's combined EQ style IDs.
fn apply_eq_style(style_id: i32, out: &mut DeckCurves, inc: &mut DeckCurves) {
    match style_id {
        // 1: eq-swap (progressive highs -> mids -> lows swap across the overlap)
        1 => {
            out.eq_low = Some(curve_from_points(
                &[(0.0, 1.0), (0.45, 1.0), (0.55, 0.0), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            out.eq_mid = Some(curve_from_points(
                &[(0.0, 1.0), (0.50, 1.0), (0.75, 0.2), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            out.eq_high = Some(curve_from_points(
                &[(0.0, 1.0), (0.35, 1.0), (0.70, 0.2), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            inc.eq_low = Some(curve_from_points(
                &[(0.0, 0.0), (0.45, 0.0), (0.55, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
            inc.eq_mid = Some(curve_from_points(
                &[(0.0, 0.2), (0.30, 0.4), (0.60, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
            inc.eq_high = Some(curve_from_points(
                &[(0.0, 0.3), (0.40, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
        }
        // 4: bass-swap (clean sub-bass handoff at midpoint)
        4 => {
            out.eq_low = Some(curve_from_points(
                &[(0.0, 1.0), (0.46, 1.0), (0.54, 0.0), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            inc.eq_low = Some(curve_from_points(
                &[(0.0, 0.0), (0.46, 0.0), (0.54, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
        }
        // 10: bass-swap-at-end
        10 => {
            out.eq_low = Some(curve_from_points(
                &[(0.0, 1.0), (0.80, 1.0), (0.90, 0.0), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            inc.eq_low = Some(curve_from_points(
                &[(0.0, 0.0), (0.80, 0.0), (0.90, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
        }
        // 12: bass-cuts (outgoing bass drops early, incoming bass enters late for tension)
        12 => {
            out.eq_low = Some(curve_from_points(
                &[(0.0, 1.0), (0.35, 1.0), (0.42, 0.0), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            inc.eq_low = Some(curve_from_points(
                &[(0.0, 0.0), (0.65, 0.0), (0.72, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
        }
        // 13: no-bass (incoming enters without low band until right at end)
        13 => {
            out.eq_low = Some(curve_from_points(
                &[(0.0, 1.0), (0.10, 0.0), (1.0, 0.0)],
                0.0,
                1.0,
            ));
            inc.eq_low = Some(curve_from_points(
                &[(0.0, 0.0), (0.85, 0.0), (0.92, 1.0), (1.0, 1.0)],
                0.0,
                1.0,
            ));
        }
        // 0 or unknown: flat EQ
        _ => {}
    }
}

fn direct_hz_filter_semantics(mode: FilterMode) -> ValueSemantics {
    let special_cases = match mode {
        FilterMode::LowPass => vec![SpecialCase {
            raw_min: 19_500.0,
            raw_max: 25_000.0,
            result: SpecialResult::Role(SpecialRole::Bypass),
        }],
        FilterMode::HighPass => vec![SpecialCase {
            raw_min: -1.0,
            raw_max: 25.0,
            result: SpecialResult::Role(SpecialRole::Bypass),
        }],
        FilterMode::Bypass => vec![],
    };
    ValueSemantics {
        output_unit: ValueUnit::Hertz,
        special_cases,
        mapping: ValueMapping::Piecewise(vec![InterpolationSegment {
            raw_start: 15.0,
            raw_end: 22_050.0,
            output_start: 15.0,
            output_end: 22_050.0,
            output_mode: Some(mode),
            interpolation: SegmentInterpolation::Linear,
            easing: SegmentEasing::None,
        }]),
    }
}

fn set_hp_in(deck: &mut DeckCurves) {
    // High-pass sweeps down from 5500 Hz -> 20 Hz (bypass) over the first 60% of overlap.
    deck.filter_cutoff = Some(sampled_curve(32, 0.0, 9000.0, |u| {
        if u >= 0.65 {
            0.0
        } else {
            let t = u / 0.65;
            5500.0 * (20.0 / 5500.0_f64).powf(t)
        }
    }));
    deck.filter_cutoff_semantics = Some(direct_hz_filter_semantics(FilterMode::HighPass));
    deck.filter_resonance = Some(curve_from_points(
        &[(0.0, 1.8), (0.55, 1.4), (0.65, 0.707), (1.0, 0.707)],
        0.5,
        4.0,
    ));
}

fn set_hp_out(deck: &mut DeckCurves) {
    // High-pass sweeps up from bypass (0 Hz) -> 6000 Hz over the second half of overlap.
    deck.filter_cutoff = Some(sampled_curve(32, 0.0, 9000.0, |u| {
        if u <= 0.40 {
            0.0
        } else {
            let t = (u - 0.40) / 0.60;
            25.0 * (6000.0 / 25.0_f64).powf(t)
        }
    }));
    deck.filter_cutoff_semantics = Some(direct_hz_filter_semantics(FilterMode::HighPass));
    deck.filter_resonance = Some(curve_from_points(
        &[(0.0, 0.707), (0.40, 0.707), (0.85, 1.8), (1.0, 1.5)],
        0.5,
        4.0,
    ));
}

fn set_lp_in(deck: &mut DeckCurves) {
    // Low-pass opens from 220 Hz -> 22050 Hz (bypass) over the first 65% of overlap.
    deck.filter_cutoff = Some(sampled_curve(32, 120.0, 22_050.0, |u| {
        if u >= 0.65 {
            22_050.0
        } else {
            let t = u / 0.65;
            220.0 * (20_000.0 / 220.0_f64).powf(t)
        }
    }));
    deck.filter_cutoff_semantics = Some(direct_hz_filter_semantics(FilterMode::LowPass));
    deck.filter_resonance = Some(curve_from_points(
        &[(0.0, 1.8), (0.55, 1.4), (0.65, 0.707), (1.0, 0.707)],
        0.5,
        4.0,
    ));
}

fn set_lp_out(deck: &mut DeckCurves) {
    // Low-pass closes from 22050 Hz (bypass) -> 180 Hz over the second half of overlap.
    deck.filter_cutoff = Some(sampled_curve(32, 120.0, 22_050.0, |u| {
        if u <= 0.40 {
            22_050.0
        } else {
            let t = (u - 0.40) / 0.60;
            20_000.0 * (180.0 / 20_000.0_f64).powf(t)
        }
    }));
    deck.filter_cutoff_semantics = Some(direct_hz_filter_semantics(FilterMode::LowPass));
    deck.filter_resonance = Some(curve_from_points(
        &[(0.0, 0.707), (0.40, 0.707), (0.85, 1.8), (1.0, 1.5)],
        0.5,
        4.0,
    ));
}

/// Clean-room modulated filter sweeps for Spotify's combined Filter FX style IDs.
fn apply_filter_style(style_id: i32, out: &mut DeckCurves, inc: &mut DeckCurves) {
    match style_id {
        5 => set_hp_in(inc),
        6 => set_hp_out(out),
        7 => {
            set_hp_out(out);
            set_hp_in(inc);
        }
        9 => set_lp_in(inc),
        10 => set_lp_out(out),
        11 => {
            set_lp_out(out);
            set_lp_in(inc);
        }
        12 => {
            set_lp_out(out);
            set_hp_in(inc);
        }
        13 => {
            set_hp_out(out);
            set_lp_in(inc);
        }
        _ => {}
    }
}

/// Convert a protobuf [`ProtoCurves`] message into protocol-agnostic [`AutomationCurves`].
pub fn convert_proto_curves(proto: &ProtoCurves) -> AutomationCurves {
    AutomationCurves {
        curves: proto
            .curves
            .iter()
            .map(|c| AutomationCurve {
                points: c
                    .points
                    .iter()
                    .map(|p| CurvePoint { x: p.x, y: p.y })
                    .collect(),
                start: c.start,
                end: c.end,
            })
            .collect(),
        minimum: proto.minimum,
        maximum: proto.maximum,
    }
}

fn infer_gain_semantics(proto: &ProtoCurves) -> ValueSemantics {
    let has_negative = proto.minimum < 0.0
        || proto
            .curves
            .iter()
            .flat_map(|c| c.points.iter())
            .any(|p| p.y < 0.0);
    if has_negative {
        ValueSemantics {
            output_unit: ValueUnit::Decibels,
            mapping: ValueMapping::Identity,
            special_cases: vec![SpecialCase {
                raw_min: -200.0,
                raw_max: -36.0,
                result: SpecialResult::Role(SpecialRole::Kill),
            }],
        }
    } else {
        ValueSemantics::default()
    }
}

fn apply_eq_override(ov: &EqCurveOverrides, deck: &mut DeckCurves) {
    let mut sem = None;
    if let Some(low) = ov.low.as_ref() {
        sem.get_or_insert_with(|| infer_gain_semantics(low));
        deck.eq_low = Some(convert_proto_curves(low));
    }
    if let Some(mid) = ov.mid.as_ref() {
        sem.get_or_insert_with(|| infer_gain_semantics(mid));
        deck.eq_mid = Some(convert_proto_curves(mid));
    }
    if let Some(high) = ov.high.as_ref() {
        sem.get_or_insert_with(|| infer_gain_semantics(high));
        deck.eq_high = Some(convert_proto_curves(high));
    }
    if let Some(s) = sem {
        deck.eq_semantics = Some(s);
    }
}

fn apply_filter_override(ov: &FilterCurveOverrides, deck: &mut DeckCurves) {
    if let Some(cutoff) = ov.cutoff.as_ref() {
        let converted = convert_proto_curves(cutoff);
        let max_abs_y = converted
            .curves
            .iter()
            .flat_map(|c| c.points.iter())
            .fold(0.0_f64, |acc, p| acc.max(p.y.abs()));
        let has_neg_y = converted
            .curves
            .iter()
            .flat_map(|c| c.points.iter())
            .any(|p| p.y < 0.0);

        // Distinguish direct Hz (> 10), bipolar [-1, 1], or normalized [0, 1] knob convention.
        if max_abs_y > 10.0 {
            let mode = if has_neg_y {
                FilterMode::HighPass
            } else {
                FilterMode::LowPass
            };
            deck.filter_cutoff_semantics = Some(direct_hz_filter_semantics(mode));
        } else if has_neg_y {
            // Bipolar [-1, 1]: [-1, -0.02] = LP 20..20kHz, ~0 = bypass, [0.02, 1] = HP 20..20kHz.
            deck.filter_cutoff_semantics = Some(ValueSemantics {
                output_unit: ValueUnit::Hertz,
                special_cases: vec![SpecialCase {
                    raw_min: -0.02,
                    raw_max: 0.02,
                    result: SpecialResult::Role(SpecialRole::Bypass),
                }],
                mapping: ValueMapping::Piecewise(vec![
                    InterpolationSegment {
                        raw_start: -1.0,
                        raw_end: -0.02,
                        output_start: 20.0,
                        output_end: 20_000.0,
                        output_mode: Some(FilterMode::LowPass),
                        interpolation: SegmentInterpolation::SignedGeometric,
                        easing: SegmentEasing::None,
                    },
                    InterpolationSegment {
                        raw_start: 0.02,
                        raw_end: 1.0,
                        output_start: 20.0,
                        output_end: 20_000.0,
                        output_mode: Some(FilterMode::HighPass),
                        interpolation: SegmentInterpolation::SignedGeometric,
                        easing: SegmentEasing::None,
                    },
                ]),
            });
        } else {
            // Normalized [0, 1] defaults to TransitionProcessor's built-in bipolar mapping.
            deck.filter_cutoff_semantics = None;
        }
        deck.filter_cutoff = Some(converted);
    }
    if let Some(res) = ov.resonance.as_ref() {
        deck.filter_resonance = Some(convert_proto_curves(res));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        playback::mix::curve::{apply_semantics, eval_curves},
        protocol::transition::{AutomationCurve as ProtoCurve, Point as ProtoPoint, VolumeStyle},
    };
    use protobuf::MessageField;

    #[test]
    fn all_builtin_presets_start_and_end_at_clean_volume_boundaries() {
        for preset in [
            BuiltinPreset::SimpleFade,
            BuiltinPreset::EqSwap,
            BuiltinPreset::BassSwap,
            BuiltinPreset::BassSwapHpFilterIn,
            BuiltinPreset::SimpleCut,
            BuiltinPreset::BassSwapLpFilterIn,
        ] {
            let (out, inc) = curves_for_builtin_preset(preset);
            let vo = out.volume.as_ref().unwrap();
            let vi = inc.volume.as_ref().unwrap();
            assert!(
                (eval_curves(vo, 0.0).unwrap() - 1.0).abs() < 1e-6,
                "{preset:?}"
            );
            assert!(eval_curves(vo, 1.0).unwrap().abs() < 1e-6, "{preset:?}");
            assert!(eval_curves(vi, 0.0).unwrap().abs() < 1e-6, "{preset:?}");
            assert!(
                (eval_curves(vi, 1.0).unwrap() - 1.0).abs() < 1e-6,
                "{preset:?}"
            );
        }
    }

    #[test]
    fn filter_presets_sweep_and_bypass_cleanly_at_exit() {
        let (_, inc_hp) = curves_for_builtin_preset(BuiltinPreset::BassSwapHpFilterIn);
        let hp_sem = inc_hp.filter_cutoff_semantics.as_ref().unwrap();
        let hp_start = apply_semantics(
            eval_curves(inc_hp.filter_cutoff.as_ref().unwrap(), 0.0).unwrap(),
            hp_sem,
        );
        let hp_end = apply_semantics(
            eval_curves(inc_hp.filter_cutoff.as_ref().unwrap(), 1.0).unwrap(),
            hp_sem,
        );
        assert_eq!(hp_start.mode, Some(FilterMode::HighPass));
        assert!(hp_start.numeric.unwrap() > 1000.0);
        assert_eq!(hp_end.role, Some(SpecialRole::Bypass));

        let (_, inc_lp) = curves_for_builtin_preset(BuiltinPreset::BassSwapLpFilterIn);
        let lp_sem = inc_lp.filter_cutoff_semantics.as_ref().unwrap();
        let lp_start = apply_semantics(
            eval_curves(inc_lp.filter_cutoff.as_ref().unwrap(), 0.0).unwrap(),
            lp_sem,
        );
        let lp_end = apply_semantics(
            eval_curves(inc_lp.filter_cutoff.as_ref().unwrap(), 1.0).unwrap(),
            lp_sem,
        );
        assert_eq!(lp_start.mode, Some(FilterMode::LowPass));
        assert!(lp_start.numeric.unwrap() < 500.0);
        assert_eq!(lp_end.role, Some(SpecialRole::Bypass));
    }

    #[test]
    fn user_db_eq_override_and_style_override_win_over_base_preset() {
        let mut preset = Preset {
            id: 1, // SimpleFade base (no EQ)
            volume_style_override: MessageField::some(VolumeStyle {
                id: 1, // Cut
                ..Default::default()
            }),
            eq_out_curve_overrides: MessageField::some(EqCurveOverrides {
                low: MessageField::some(ProtoCurves {
                    curves: vec![ProtoCurve {
                        points: vec![
                            ProtoPoint {
                                x: 0.0,
                                y: 0.0,
                                ..Default::default()
                            },
                            ProtoPoint {
                                x: 1.0,
                                y: -20.0,
                                ..Default::default()
                            },
                        ],
                        start: 0.0,
                        end: 1.0,
                        ..Default::default()
                    }],
                    minimum: -20.0,
                    maximum: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        preset.type_ = protobuf::EnumOrUnknown::new(PresetType::PRESET_TYPE_BEATMATCHED_FADE);

        let (out, _inc, _handoff) = curves_from_proto_preset(&preset);
        // Volume style 1 (cut) holds 1.0 at u = 0.4 and drops to 0.0 at u = 0.6.
        assert!((eval_curves(out.volume.as_ref().unwrap(), 0.4).unwrap() - 1.0).abs() < 1e-6);
        assert!(
            eval_curves(out.volume.as_ref().unwrap(), 0.6)
                .unwrap()
                .abs()
                < 1e-6
        );

        // User dB EQ override inferred ValueUnit::Decibels and maps -20 dB -> 0.1 linear gain.
        let sem = out.eq_semantics.as_ref().unwrap();
        assert_eq!(sem.output_unit, ValueUnit::Decibels);
        let raw_end = eval_curves(out.eq_low.as_ref().unwrap(), 1.0).unwrap();
        let gain_end = apply_semantics(raw_end, sem).as_linear_gain().unwrap();
        assert!((gain_end - 0.1).abs() < 1e-4);
    }
}
