//! Real-time dual-deck DJ transition processor combining beat-clock synchronization,
//! time-stretching, 3-band EQ, modulated SVF filtering, automation curves, and
//! anti-click boundary/skip fades.

use std::{collections::VecDeque, f64::consts::FRAC_PI_2};

use super::{
    beat_clock::BeatClock,
    curve::{apply_semantics, equal_power_in, equal_power_out, eval_curves},
    filters::{DeckEq, DeckFilter},
    spec::{DeckCurves, FilterMode, SpecialRole, TimeStretchMode, TransitionSpec},
    stretch::DeckStretcher,
};
use crate::{NUM_CHANNELS, SAMPLE_RATE};

const CHANNELS: usize = NUM_CHANNELS as usize;
/// Control update block size in stereo frames (~5.8 ms at 44.1 kHz).
const CONTROL_BLOCK_FRAMES: usize = 256;
/// Cosine dry-to-wet entry/exit blend duration (~20 ms at 44.1 kHz).
const BOUNDARY_BLEND_FRAMES: usize = 882;
/// Anti-click cosine micro-fade when aborting a transition on manual skip (~15 ms at 44.1 kHz).
const ABORT_FADE_FRAMES: usize = 661;
/// Raw Deck B frames kept behind the stretcher playback cursor for exact wet-to-dry exit handoff.
const RAW_B_LOOKBEHIND_FRAMES: u64 = 4096;

struct DeckChain {
    stretcher: DeckStretcher,
    eq: DeckEq,
    filter: DeckFilter,
    curves: DeckCurves,
    is_incoming: bool,
    scratch: Vec<f64>,
    /// Buffered rendered stereo frames from `stretcher`, guaranteeing zero sample drop when
    /// Deck A and Deck B return different chunk sizes.
    out_buf: VecDeque<f64>,
    /// Last ratio applied to `stretcher`, used to map `out_buf` depth back to source frames.
    last_ratio: f64,
    /// Tracks whether EQ/filter need processing (bypassed when curves are neutral/absent).
    has_eq: bool,
    has_filter: bool,
    /// Per-sample 2 ms volume smoother preventing clicks on discontinuous user volume steps.
    smoothed_vol: f64,
    vol_smooth_coeff: f64,
}

impl DeckChain {
    fn new(
        stretch_mode: TimeStretchMode,
        origin_frame: u64,
        initial_ratio: f64,
        curves: DeckCurves,
        eq_topology: super::spec::EqTopology,
        is_incoming: bool,
    ) -> Self {
        let has_eq = curves.eq_low.is_some() || curves.eq_mid.is_some() || curves.eq_high.is_some();
        let has_filter = curves.filter_cutoff.is_some();
        let vol_smooth_coeff = (-1.0 / (0.002 * f64::from(SAMPLE_RATE))).exp();
        let mut chain = Self {
            stretcher: DeckStretcher::new(stretch_mode, origin_frame, initial_ratio),
            eq: DeckEq::new(eq_topology),
            filter: DeckFilter::new(),
            curves,
            is_incoming,
            scratch: Vec::with_capacity(CONTROL_BLOCK_FRAMES * CHANNELS),
            out_buf: VecDeque::with_capacity(CONTROL_BLOCK_FRAMES * 2 * CHANNELS),
            last_ratio: initial_ratio,
            has_eq,
            has_filter,
            smoothed_vol: if is_incoming { 0.0 } else { 1.0 },
            vol_smooth_coeff,
        };
        chain.smoothed_vol = chain.evaluate_volume(0.0);
        chain
    }

    #[inline]
    fn smoothed_volume(&mut self, u: f64) -> f64 {
        let target = self.evaluate_volume(u);
        self.smoothed_vol =
            self.vol_smooth_coeff * self.smoothed_vol + (1.0 - self.vol_smooth_coeff) * target;
        self.smoothed_vol
    }

    #[inline]
    fn out_frames(&self) -> usize {
        self.out_buf.len() / CHANNELS
    }

    fn set_ratio(&mut self, ratio: f64) {
        let clamped = ratio.clamp(0.5, 2.0);
        self.last_ratio = clamped;
        self.stretcher.set_ratio(clamped);
    }

    /// Source frame index of the next frame to be popped from `out_buf`.
    #[inline]
    fn playback_source_frame(&self) -> f64 {
        self.stretcher.current_source_frame() - self.out_frames() as f64 * self.last_ratio
    }

    fn fill_out_buf(&mut self, target_frames: usize) {
        let have = self.out_frames();
        if have >= target_frames || self.stretcher.is_finished() {
            return;
        }
        let want = target_frames - have;
        self.scratch.clear();
        let got = self.stretcher.render(want, &mut self.scratch);
        if got > 0 {
            self.out_buf.extend(self.scratch.drain(..));
        }
    }

    #[inline]
    fn pop_frame(&mut self) -> [f64; 2] {
        let l = self.out_buf.pop_front().unwrap_or(0.0);
        let r = self.out_buf.pop_front().unwrap_or(0.0);
        [l, r]
    }

    fn is_drained(&self) -> bool {
        self.stretcher.is_finished() && self.out_frames() == 0
    }

    fn evaluate_volume(&self, u: f64) -> f64 {
        let Some(ref curves) = self.curves.volume else {
            return if self.is_incoming {
                equal_power_in(u)
            } else {
                equal_power_out(u)
            };
        };
        let Some(raw) = eval_curves(curves, u) else {
            return if self.is_incoming {
                equal_power_in(u)
            } else {
                equal_power_out(u)
            };
        };
        if let Some(ref sem) = self.curves.volume_semantics {
            let ev = apply_semantics(raw, sem);
            if matches!(
                ev.role,
                Some(SpecialRole::Mute | SpecialRole::Kill | SpecialRole::Off)
            ) {
                return 0.0;
            }
            if let Some(v) = ev.as_linear_gain() {
                return v.clamp(0.0, 4.0);
            }
        }
        raw.clamp(0.0, 4.0)
    }

    fn evaluate_eq_gains(&self, u: f64) -> [f64; 3] {
        let eval_band = |band: &Option<super::spec::AutomationCurves>| -> f64 {
            let Some(curves) = band else {
                return 1.0;
            };
            let Some(raw) = eval_curves(curves, u) else {
                return 1.0;
            };
            if let Some(ref sem) = self.curves.eq_semantics {
                let ev = apply_semantics(raw, sem);
                return match ev.role {
                    Some(SpecialRole::Kill | SpecialRole::Mute) => 0.0,
                    Some(SpecialRole::Bypass | SpecialRole::Off) => 1.0,
                    None => ev.as_linear_gain().unwrap_or(1.0).clamp(0.0, 4.0),
                };
            }
            raw.clamp(0.0, 4.0)
        };
        [
            eval_band(&self.curves.eq_low),
            eval_band(&self.curves.eq_mid),
            eval_band(&self.curves.eq_high),
        ]
    }

    fn evaluate_filter(&self, u: f64) -> (FilterMode, f64, f64) {
        let Some(ref cutoff_curves) = self.curves.filter_cutoff else {
            return (FilterMode::Bypass, 1000.0, std::f64::consts::FRAC_1_SQRT_2);
        };
        let Some(raw_cutoff) = eval_curves(cutoff_curves, u) else {
            return (FilterMode::Bypass, 1000.0, std::f64::consts::FRAC_1_SQRT_2);
        };

        let q = self
            .curves
            .filter_resonance
            .as_ref()
            .and_then(|c| eval_curves(c, u))
            .map(|raw_q| {
                if let Some(ref sem) = self.curves.filter_resonance_semantics {
                    apply_semantics(raw_q, sem)
                        .numeric
                        .unwrap_or(std::f64::consts::FRAC_1_SQRT_2)
                } else {
                    raw_q
                }
            })
            .unwrap_or(std::f64::consts::FRAC_1_SQRT_2)
            .clamp(0.5, 10.0);

        if let Some(ref sem) = self.curves.filter_cutoff_semantics {
            let ev = apply_semantics(raw_cutoff, sem);
            if matches!(ev.role, Some(SpecialRole::Bypass | SpecialRole::Off)) {
                return (FilterMode::Bypass, 1000.0, q);
            }
            let mode = ev.mode.unwrap_or(FilterMode::LowPass);
            let hz = ev.numeric.unwrap_or(1000.0);
            return (mode, hz, q);
        }

        // Default bipolar convention: [0.0, 0.49] = low-pass 20..20kHz, ~0.5 = bypass, [0.51, 1.0] = high-pass 20..20kHz.
        if (raw_cutoff - 0.5).abs() < 0.01 {
            (FilterMode::Bypass, 1000.0, q)
        } else if raw_cutoff < 0.5 {
            let t = (raw_cutoff / 0.49).clamp(0.0, 1.0);
            let hz = 20.0 * (1000.0_f64).powf(t);
            (FilterMode::LowPass, hz, q)
        } else {
            let t = ((raw_cutoff - 0.51) / 0.49).clamp(0.0, 1.0);
            let hz = 20.0 * (1000.0_f64).powf(t);
            (FilterMode::HighPass, hz, q)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExitBlend {
    anchor_frame: u64,
    pos: usize,
    total: usize,
}

/// Real-time dual-deck transition engine.
///
/// Consumes normalized interleaved stereo `f64` PCM from outgoing Deck A and incoming
/// Deck B, performs beat-locked time-stretching, 3-band EQ, modulated SVF filtering,
/// and volume automation, and emits summed interleaved stereo `f64` samples ready for
/// [`crate::player::MasterLimiter`].
pub struct TransitionProcessor {
    spec: TransitionSpec,
    clock: BeatClock,
    stretch_mode: TimeStretchMode,
    deck_a: DeckChain,
    deck_b: DeckChain,
    handoff_emitted: bool,
    frames_rendered: u64,
    abort_fade_remaining: Option<usize>,
    /// Dry copies of Deck A's first `BOUNDARY_BLEND_FRAMES` samples for click-free entry.
    entry_dry_a: Vec<f64>,
    entry_blend_pos: usize,
    /// Sliding ring buffer of raw Deck B samples starting at `raw_b_start_frame` for
    /// exact zero-seek, click-free wet-to-dry exit handoff into single-deck playback.
    raw_b_buf: VecDeque<f64>,
    raw_b_start_frame: u64,
    exit_blend: Option<ExitBlend>,
}

impl TransitionProcessor {
    pub fn new(spec: TransitionSpec) -> Self {
        let clock = BeatClock::new(
            spec.start_a_frame,
            spec.start_b_frame,
            spec.duration_frames,
            spec.duration_beats,
            spec.tempo.as_ref(),
        );
        let stretch_mode = if clock.needs_time_stretch() {
            spec.tempo
                .as_ref()
                .map(|t| t.stretch_mode)
                .unwrap_or(TimeStretchMode::PitchPreserving)
        } else {
            TimeStretchMode::Off
        };
        let (r_a0, r_b0) = if stretch_mode != TimeStretchMode::Off {
            clock.deck_ratios()
        } else {
            (1.0, 1.0)
        };

        let deck_a = DeckChain::new(
            stretch_mode,
            spec.start_a_frame,
            r_a0,
            spec.out_curves.clone(),
            spec.eq_topology,
            false,
        );
        let deck_b = DeckChain::new(
            stretch_mode,
            spec.start_b_frame,
            r_b0,
            spec.in_curves.clone(),
            spec.eq_topology,
            true,
        );

        let raw_b_start_frame = spec.start_b_frame;

        Self {
            spec,
            clock,
            stretch_mode,
            deck_a,
            deck_b,
            handoff_emitted: false,
            frames_rendered: 0,
            abort_fade_remaining: None,
            entry_dry_a: Vec::with_capacity(BOUNDARY_BLEND_FRAMES * CHANNELS),
            entry_blend_pos: 0,
            raw_b_buf: VecDeque::with_capacity(16_384 * CHANNELS),
            raw_b_start_frame,
            exit_blend: None,
        }
    }

    pub fn spec(&self) -> &TransitionSpec {
        &self.spec
    }

    pub fn progress(&self) -> f64 {
        self.clock.progress()
    }

    pub fn is_complete(&self) -> bool {
        if let Some(rem) = self.abort_fade_remaining {
            return rem == 0;
        }
        if let Some(exit) = self.exit_blend {
            return exit.pos >= exit.total;
        }
        self.deck_a.is_drained() && self.deck_b.is_drained()
    }

    /// Returns `true` once when progress crosses `spec.handoff_fraction` (the point where
    /// Deck B should become the reported active track in Spotify Connect).
    pub fn take_handoff_due(&mut self) -> bool {
        if !self.handoff_emitted
            && self.clock.progress() >= self.spec.handoff_fraction.clamp(0.0, 1.0)
        {
            self.handoff_emitted = true;
            true
        } else {
            false
        }
    }

    /// Trigger a click-free ~15 ms cosine fade-out (for manual skip / abort mid-overlap).
    pub fn begin_abort_fade(&mut self) {
        if self.abort_fade_remaining.is_none() {
            self.abort_fade_remaining = Some(ABORT_FADE_FRAMES);
        }
    }

    pub fn push_deck_a(&mut self, samples: &[f64]) {
        if self.entry_dry_a.len() < BOUNDARY_BLEND_FRAMES * CHANNELS {
            let need = BOUNDARY_BLEND_FRAMES * CHANNELS - self.entry_dry_a.len();
            let take = need.min(samples.len());
            self.entry_dry_a.extend_from_slice(&samples[..take]);
        }
        self.deck_a.stretcher.push_samples(samples);
    }

    pub fn push_deck_b(&mut self, samples: &[f64]) {
        self.raw_b_buf
            .extend(samples.iter().map(|&s| if s.is_finite() { s } else { 0.0 }));
        self.deck_b.stretcher.push_samples(samples);
    }

    pub fn set_deck_a_eof(&mut self) {
        self.deck_a.stretcher.set_eof();
    }

    pub fn set_deck_b_eof(&mut self) {
        self.deck_b.stretcher.set_eof();
    }

    pub fn needs_deck_a_input(&self) -> bool {
        self.exit_blend.is_none()
            && !self.deck_a.stretcher.is_finished()
            && self.deck_a.stretcher.buffered_input_frames()
                < self.deck_a.stretcher.recommended_input_frames()
    }

    pub fn needs_deck_b_input(&self) -> bool {
        !self.deck_b.stretcher.is_finished()
            && self.deck_b.stretcher.buffered_input_frames()
                < self.deck_b.stretcher.recommended_input_frames()
    }

    /// Current exact source frame position in Deck A at the playback cursor.
    pub fn deck_a_source_frame(&self) -> f64 {
        self.deck_a.playback_source_frame()
    }

    /// Current exact source frame position in Deck B at the playback cursor.
    pub fn deck_b_source_frame(&self) -> f64 {
        self.deck_b.playback_source_frame()
    }

    /// Drain remaining dry Deck B samples starting at the exact frame following the
    /// transition exit, returning `(next_source_frame, interleaved_samples)` so single-deck
    /// playback can resume with zero dropped frames, zero seeks, and zero clicks.
    pub fn drain_deck_b_continuation(&mut self) -> (u64, Vec<f64>) {
        let next_frame = self
            .exit_blend
            .map(|e| e.anchor_frame + e.pos as u64)
            .unwrap_or_else(|| self.deck_b_source_frame().round().max(0.0) as u64);

        if next_frame > self.raw_b_start_frame {
            let drop_frames = (next_frame - self.raw_b_start_frame) as usize;
            let drop_frames = drop_frames.min(self.raw_b_buf.len() / CHANNELS);
            self.raw_b_buf.drain(..drop_frames * CHANNELS);
            self.raw_b_start_frame += drop_frames as u64;
        }

        let remaining: Vec<f64> = self.raw_b_buf.drain(..).collect();
        (next_frame, remaining)
    }

    fn trim_raw_b_behind_cursor(&mut self) {
        if self.exit_blend.is_some() {
            return;
        }
        let cursor = self.deck_b.playback_source_frame().floor().max(0.0) as u64;
        let keep_from = cursor.saturating_sub(RAW_B_LOOKBEHIND_FRAMES);
        if keep_from > self.raw_b_start_frame {
            let drop_frames = (keep_from - self.raw_b_start_frame) as usize;
            let drop_frames = drop_frames.min(self.raw_b_buf.len() / CHANNELS);
            self.raw_b_buf.drain(..drop_frames * CHANNELS);
            self.raw_b_start_frame += drop_frames as u64;
        }
    }

    fn read_raw_b_frame(&self, frame_idx: u64) -> Option<[f64; 2]> {
        let offset = frame_idx.checked_sub(self.raw_b_start_frame)? as usize;
        let sample_idx = offset * CHANNELS;
        if sample_idx + 1 < self.raw_b_buf.len() {
            Some([self.raw_b_buf[sample_idx], self.raw_b_buf[sample_idx + 1]])
        } else {
            None
        }
    }

    /// Render up to `max_frames` mixed interleaved stereo `f64` frames into `out`.
    pub fn render(&mut self, max_frames: usize, out: &mut Vec<f64>) -> usize {
        let mut total_written = 0usize;

        while total_written < max_frames && !self.is_complete() {
            let chunk = (max_frames - total_written).min(CONTROL_BLOCK_FRAMES);

            // Transition into wet-to-dry exit blend once the beat clock finishes.
            if self.clock.is_complete()
                && self.exit_blend.is_none()
                && self.abort_fade_remaining.is_none()
            {
                let anchor_frame = self
                    .deck_b
                    .playback_source_frame()
                    .round()
                    .max(self.raw_b_start_frame as f64) as u64;
                let needs_exit_blend = self.stretch_mode != TimeStretchMode::Off
                    || self.deck_b.has_eq
                    || self.deck_b.has_filter;
                if needs_exit_blend && self.read_raw_b_frame(anchor_frame).is_some() {
                    self.deck_b.set_ratio(1.0);
                    self.exit_blend = Some(ExitBlend {
                        anchor_frame,
                        pos: 0,
                        total: BOUNDARY_BLEND_FRAMES,
                    });
                } else {
                    // No wet-to-dry blend required: mark completed at `pos = 0` so
                    // `drain_deck_b_continuation()` resumes at exact `anchor_frame` with zero skip.
                    self.exit_blend = Some(ExitBlend {
                        anchor_frame,
                        pos: 0,
                        total: 0,
                    });
                    break;
                }
            }

            // Wet-to-dry exit blend phase: Deck A is silent, Deck B blends from wet to exact dry PCM.
            if let Some(mut exit) = self.exit_blend {
                if exit.pos >= exit.total {
                    // If an abort fade was requested right before exit, keep draining dry Deck B
                    // frames through the abort cosine fade until silence is reached.
                    if let Some(mut rem) = self.abort_fade_remaining {
                        if rem == 0 {
                            break;
                        }
                        let want = chunk.min(rem);
                        for _ in 0..want {
                            let frame_idx = exit.anchor_frame + exit.pos as u64;
                            let dry = self.read_raw_b_frame(frame_idx).unwrap_or([0.0, 0.0]);
                            let g = (rem as f64 / ABORT_FADE_FRAMES as f64 * FRAC_PI_2).sin();
                            let l = dry[0] * g;
                            let r = dry[1] * g;
                            out.push(if l.is_finite() { l } else { 0.0 });
                            out.push(if r.is_finite() { r } else { 0.0 });
                            rem -= 1;
                            exit.pos += 1;
                            exit.total = exit.pos;
                            total_written += 1;
                            self.frames_rendered += 1;
                        }
                        self.abort_fade_remaining = Some(rem);
                        self.exit_blend = Some(exit);
                        continue;
                    }
                    break;
                }
                let want = chunk.min(exit.total - exit.pos);
                self.deck_b.fill_out_buf(want);
                let avail_b = self.deck_b.out_frames();
                if avail_b == 0 {
                    if self.deck_b.is_drained() {
                        exit.total = exit.pos;
                        self.exit_blend = Some(exit);
                    }
                    break;
                }
                let equal_power_exit = self.stretch_mode == TimeStretchMode::PitchPreserving
                    || (self.deck_b.has_eq
                        && matches!(
                            self.spec.eq_topology,
                            super::spec::EqTopology::LinkwitzRileyIsolator { .. }
                        ));
                let n = want.min(avail_b);
                for _ in 0..n {
                    let frame_idx = exit.anchor_frame + exit.pos as u64;
                    let Some(dry) = self.read_raw_b_frame(frame_idx) else {
                        if self.deck_b.is_drained() {
                            exit.total = exit.pos;
                        }
                        self.exit_blend = Some(exit);
                        break;
                    };

                    let mut wet_b = self.deck_b.pop_frame();
                    if self.deck_b.has_eq {
                        let eq_gains = self.deck_b.evaluate_eq_gains(1.0);
                        wet_b = self.deck_b.eq.process(wet_b, eq_gains);
                    }
                    if self.deck_b.has_filter {
                        let (mode, hz, q) = self.deck_b.evaluate_filter(1.0);
                        wet_b = self.deck_b.filter.process(wet_b, mode, hz, q);
                    }
                    let vol_b = self.deck_b.smoothed_volume(1.0);
                    wet_b[0] *= vol_b;
                    wet_b[1] *= vol_b;

                    let theta = (exit.pos + 1) as f64 / exit.total as f64 * FRAC_PI_2;
                    let (w_wet, w_dry) = if equal_power_exit {
                        (theta.cos(), theta.sin())
                    } else {
                        let alpha = theta.sin().powi(2);
                        (1.0 - alpha, alpha)
                    };
                    let mut mixed = [
                        wet_b[0] * w_wet + dry[0] * w_dry,
                        wet_b[1] * w_wet + dry[1] * w_dry,
                    ];

                    if let Some(ref mut rem) = self.abort_fade_remaining {
                        if *rem == 0 {
                            break;
                        }
                        let g = (*rem as f64 / ABORT_FADE_FRAMES as f64 * FRAC_PI_2).sin();
                        mixed[0] *= g;
                        mixed[1] *= g;
                        *rem -= 1;
                    }

                    out.push(if mixed[0].is_finite() { mixed[0] } else { 0.0 });
                    out.push(if mixed[1].is_finite() { mixed[1] } else { 0.0 });
                    exit.pos += 1;
                    total_written += 1;
                    self.frames_rendered += 1;
                }
                self.exit_blend = Some(exit);
                continue;
            }

            // Closed-loop feedforward + proportional beat-phase servo: cancels hop-quantization
            // drift so Deck A and Deck B stay phase-locked across arbitrary transition lengths.
            if self.clock.beatmatched && self.stretch_mode != TimeStretchMode::Off {
                let (ff_a, ff_b) = self.clock.deck_ratios();
                let (target_a, target_b) = self.clock.expected_source_frames();
                let err_a = target_a - self.deck_a.playback_source_frame();
                let err_b = target_b - self.deck_b.playback_source_frame();
                let horizon = 0.25 * f64::from(SAMPLE_RATE); // 250 ms gentle proportional horizon
                let corr_a = (1.0 + err_a / horizon).clamp(0.985, 1.015);
                let corr_b = (1.0 + err_b / horizon).clamp(0.985, 1.015);
                self.deck_a.set_ratio(ff_a * corr_a);
                self.deck_b.set_ratio(ff_b * corr_b);
            }

            self.deck_a.fill_out_buf(chunk);
            self.deck_b.fill_out_buf(chunk);

            let avail_a = self.deck_a.out_frames();
            let avail_b = self.deck_b.out_frames();

            let n = match (self.deck_a.is_drained(), self.deck_b.is_drained()) {
                (false, false) => avail_a.min(avail_b).min(chunk),
                (true, false) => avail_b.min(chunk),
                (false, true) => avail_a.min(chunk),
                (true, true) => 0,
            };
            if n == 0 {
                if self.abort_fade_remaining.is_some()
                    && self.deck_a.is_drained()
                    && self.deck_b.is_drained()
                {
                    self.abort_fade_remaining = Some(0);
                }
                break;
            }

            let u_start = self.clock.progress();
            self.clock.advance_frames(n);
            let u_end = self.clock.progress();

            let equal_power_entry = self.stretch_mode == TimeStretchMode::PitchPreserving
                || (self.deck_a.has_eq
                    && matches!(
                        self.spec.eq_topology,
                        super::spec::EqTopology::LinkwitzRileyIsolator { .. }
                    ));

            for i in 0..n {
                let frac = i as f64 / n as f64;
                let u = u_start + frac * (u_end - u_start);

                let mut a = if i < avail_a {
                    self.deck_a.pop_frame()
                } else {
                    [0.0, 0.0]
                };

                // Click-free blend from dry Deck A into stretched/filtered Deck A over the first ~20 ms.
                let mut wet_a = a;
                if self.deck_a.has_eq {
                    let eq_gains = self.deck_a.evaluate_eq_gains(u);
                    wet_a = self.deck_a.eq.process(wet_a, eq_gains);
                }
                if self.deck_a.has_filter {
                    let (mode, hz, q) = self.deck_a.evaluate_filter(u);
                    wet_a = self.deck_a.filter.process(wet_a, mode, hz, q);
                }
                if self.entry_blend_pos < BOUNDARY_BLEND_FRAMES
                    && (self.entry_blend_pos + 1) * CHANNELS <= self.entry_dry_a.len()
                {
                    let dry = [
                        self.entry_dry_a[self.entry_blend_pos * CHANNELS],
                        self.entry_dry_a[self.entry_blend_pos * CHANNELS + 1],
                    ];
                    let theta =
                        self.entry_blend_pos as f64 / BOUNDARY_BLEND_FRAMES as f64 * FRAC_PI_2;
                    let (w_dry, w_wet) = if equal_power_entry {
                        (theta.cos(), theta.sin())
                    } else {
                        let alpha = theta.sin().powi(2);
                        (1.0 - alpha, alpha)
                    };
                    a = [
                        dry[0] * w_dry + wet_a[0] * w_wet,
                        dry[1] * w_dry + wet_a[1] * w_wet,
                    ];
                    self.entry_blend_pos += 1;
                } else {
                    a = wet_a;
                }

                let vol_a = self.deck_a.smoothed_volume(u);
                a[0] *= vol_a;
                a[1] *= vol_a;

                let mut b = if i < avail_b {
                    self.deck_b.pop_frame()
                } else {
                    [0.0, 0.0]
                };
                if self.deck_b.has_eq {
                    let eq_gains = self.deck_b.evaluate_eq_gains(u);
                    b = self.deck_b.eq.process(b, eq_gains);
                }
                if self.deck_b.has_filter {
                    let (mode, hz, q) = self.deck_b.evaluate_filter(u);
                    b = self.deck_b.filter.process(b, mode, hz, q);
                }
                let vol_b = self.deck_b.smoothed_volume(u);
                b[0] *= vol_b;
                b[1] *= vol_b;

                let mut sum_l = a[0] + b[0];
                let mut sum_r = a[1] + b[1];

                if let Some(ref mut rem) = self.abort_fade_remaining {
                    if *rem == 0 {
                        break;
                    }
                    let g = (*rem as f64 / ABORT_FADE_FRAMES as f64 * FRAC_PI_2).sin();
                    sum_l *= g;
                    sum_r *= g;
                    *rem -= 1;
                }

                // Final NaN/Inf safety barrier before master limiter and audio sink.
                out.push(if sum_l.is_finite() { sum_l } else { 0.0 });
                out.push(if sum_r.is_finite() { sum_r } else { 0.0 });
                total_written += 1;
                self.frames_rendered += 1;
            }

            self.trim_raw_b_behind_cursor();
        }

        total_written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{NormalisationMethod, PlayerConfig},
        mix::spec::{AutomationCurve, AutomationCurves, CurvePoint, TempoPlan},
        player::{MasterLimiter, apply_track_normalisation, ratio_to_db},
    };
    use rand::{Rng, SeedableRng, rngs::SmallRng};
    use std::f64::consts::PI;

    /// Generate an interleaved stereo impulse click track at exact `bpm` (10 ms windowed 120 Hz kick burst per beat).
    fn generate_kick_track(bpm: f64, num_beats: usize, pan_left: f64, pan_right: f64) -> Vec<f64> {
        let fs = f64::from(SAMPLE_RATE);
        let beat_period_frames = fs * 60.0 / bpm;
        let total_frames = ((num_beats as f64 + 4.0) * beat_period_frames).ceil() as usize;
        let mut pcm = vec![0.0_f64; total_frames * CHANNELS];
        let burst_len = (0.015 * fs) as usize; // 15 ms percussive kick transient
        for beat in 0..num_beats {
            let start = (beat as f64 * beat_period_frames).round() as usize;
            for k in 0..burst_len {
                if start + k >= total_frames {
                    break;
                }
                let env = (-(k as f64) / (0.004 * fs)).exp();
                let s = 0.8 * env * (2.0 * PI * 100.0 * k as f64 / fs).sin();
                pcm[(start + k) * 2] += s * pan_left;
                pcm[(start + k) * 2 + 1] += s * pan_right;
            }
        }
        pcm
    }

    fn find_beat_peaks(mono: &[f64], min_spacing_frames: usize) -> Vec<usize> {
        let mut peaks = Vec::new();
        let mut i = 0;
        while i < mono.len() {
            if mono[i].abs() > 0.15 {
                let end = (i + min_spacing_frames / 2).min(mono.len());
                let mut best_idx = i;
                let mut best_val = 0.0;
                for (j, &s) in mono[i..end].iter().enumerate() {
                    if s.abs() > best_val {
                        best_val = s.abs();
                        best_idx = i + j;
                    }
                }
                peaks.push(best_idx);
                i = best_idx + min_spacing_frames;
            } else {
                i += 1;
            }
        }
        peaks
    }

    fn run_offline(spec: TransitionSpec, pcm_a: &[f64], pcm_b: &[f64]) -> Vec<f64> {
        let mut proc = TransitionProcessor::new(spec);
        let mut pos_a = 0usize;
        let mut pos_b = 0usize;
        let mut out = Vec::new();
        let feed_chunk = 2048 * CHANNELS;

        while !proc.is_complete() {
            if proc.needs_deck_a_input() {
                if pos_a < pcm_a.len() {
                    let end = (pos_a + feed_chunk).min(pcm_a.len());
                    proc.push_deck_a(&pcm_a[pos_a..end]);
                    pos_a = end;
                } else {
                    proc.set_deck_a_eof();
                }
            }
            if proc.needs_deck_b_input() {
                if pos_b < pcm_b.len() {
                    let end = (pos_b + feed_chunk).min(pcm_b.len());
                    proc.push_deck_b(&pcm_b[pos_b..end]);
                    pos_b = end;
                } else {
                    proc.set_deck_b_eof();
                }
            }
            let wrote = proc.render(512, &mut out);
            if wrote == 0 && pos_a >= pcm_a.len() && pos_b >= pcm_b.len() {
                break;
            }
        }
        out
    }

    #[test]
    fn beat_lock_aligns_145_to_150_bpm_kicks_within_bounds_across_4_bars() {
        // Put Deck A kicks in Left channel only and Deck B kicks in Right channel only so we
        // can measure exact per-beat lag between Deck A and Deck B across 16 beats (4 bars).
        let pcm_a = generate_kick_track(145.0, 22, 1.0, 0.0);
        let pcm_b = generate_kick_track(150.0, 22, 0.0, 1.0);

        let flat_full = AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![CurvePoint { x: 0.0, y: 1.0 }, CurvePoint { x: 1.0, y: 1.0 }],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        };

        let mut spec = TransitionSpec {
            start_a_frame: 0,
            start_b_frame: 0,
            duration_frames: 0,
            duration_beats: Some(16.0),
            handoff_fraction: 0.5,
            tempo: Some(TempoPlan::beatmatched(145.0, 150.0)),
            out_curves: DeckCurves {
                volume: Some(flat_full.clone()),
                ..Default::default()
            },
            in_curves: DeckCurves {
                volume: Some(flat_full),
                ..Default::default()
            },
            eq_topology: super::super::spec::EqTopology::default(),
        };

        for mode in [TimeStretchMode::Varispeed, TimeStretchMode::PitchPreserving] {
            spec.tempo.as_mut().unwrap().stretch_mode = mode;
            let mut proc = TransitionProcessor::new(spec.clone());
            if mode == TimeStretchMode::PitchPreserving {
                assert!(
                    proc.deck_a.stretcher.is_pitch_preserving()
                        && proc.deck_b.stretcher.is_pitch_preserving(),
                    "PitchPreserving stretcher fell back unexpectedly"
                );
            }
            let mut pos_a = 0usize;
            let mut pos_b = 0usize;
            let mut out = Vec::new();
            let feed_chunk = 2048 * CHANNELS;
            while !proc.is_complete() {
                while proc.needs_deck_a_input() {
                    if pos_a < pcm_a.len() {
                        let end = (pos_a + feed_chunk).min(pcm_a.len());
                        proc.push_deck_a(&pcm_a[pos_a..end]);
                        pos_a = end;
                    } else {
                        proc.set_deck_a_eof();
                        break;
                    }
                }
                while proc.needs_deck_b_input() {
                    if pos_b < pcm_b.len() {
                        let end = (pos_b + feed_chunk).min(pcm_b.len());
                        proc.push_deck_b(&pcm_b[pos_b..end]);
                        pos_b = end;
                    } else {
                        proc.set_deck_b_eof();
                        break;
                    }
                }
                let wrote = proc.render(512, &mut out);
                if wrote == 0 && pos_a >= pcm_a.len() && pos_b >= pcm_b.len() {
                    break;
                }
            }
            let left: Vec<f64> = out.iter().step_by(2).copied().collect();
            let right: Vec<f64> = out.iter().skip(1).step_by(2).copied().collect();

            let peaks_a = find_beat_peaks(&left, 10_000);
            let peaks_b = find_beat_peaks(&right, 10_000);
            assert!(
                peaks_a.len() >= 14 && peaks_b.len() >= 14,
                "{mode:?}: expected >= 14 beats, got A={} B={}",
                peaks_a.len(),
                peaks_b.len()
            );

            let paired = peaks_a.len().min(peaks_b.len());
            let mut max_err_ms = 0.0_f64;
            for idx in 1..(paired - 1) {
                let diff_frames = peaks_a[idx].abs_diff(peaks_b[idx]);
                let err_ms = diff_frames as f64 * 1000.0 / f64::from(SAMPLE_RATE);
                max_err_ms = max_err_ms.max(err_ms);
            }
            // Pitch-preserving STFT can shift a percussive peak by a fraction of a ms due to phase
            // dispersion, while varispeed is exact; check tight bounds and zero cumulative drift.
            let limit_ms = match mode {
                TimeStretchMode::Varispeed => 0.5,
                TimeStretchMode::PitchPreserving => 1.5,
                TimeStretchMode::Off => 0.5,
            };
            assert!(
                max_err_ms < limit_ms,
                "{mode:?}: max beat lag {max_err_ms:.2} ms exceeded {limit_ms} ms"
            );
        }
    }

    #[test]
    fn pitch_preserving_stretcher_preserves_mid_side_stereo_coherence() {
        // Stretching a correlated mono-in-stereo signal (100% Mid, 0% Side) must keep
        // Side leakage minimal rather than decorrelating Left and Right phases.
        let fs = f64::from(SAMPLE_RATE);
        let n_frames = SAMPLE_RATE as usize * 3;
        let mut dry = vec![0.0_f64; n_frames * CHANNELS];
        for i in 0..n_frames {
            let t = i as f64 / fs;
            let s = 0.4 * (2.0 * PI * 120.0 * t).sin()
                + 0.25 * (2.0 * PI * 440.0 * t).sin()
                + 0.15 * (2.0 * PI * 2200.0 * t).sin();
            dry[2 * i] = s;
            dry[2 * i + 1] = s;
        }

        let mut stretcher = DeckStretcher::new(TimeStretchMode::PitchPreserving, 0, 1.05);
        assert!(
            stretcher.is_pitch_preserving(),
            "PitchPreserving stretcher fell back unexpectedly"
        );
        stretcher.push_samples(&dry);
        stretcher.set_eof();
        let mut out = Vec::new();
        while !stretcher.is_finished() {
            if stretcher.render(1024, &mut out) == 0 {
                break;
            }
        }
        assert!(out.len() > SAMPLE_RATE as usize * 2);

        let mut mid_energy = 0.0_f64;
        let mut side_energy = 0.0_f64;
        for frame in out.chunks_exact(2) {
            let m = 0.5 * (frame[0] + frame[1]);
            let s = 0.5 * (frame[0] - frame[1]);
            mid_energy += m * m;
            side_energy += s * s;
        }
        let mid_fraction = mid_energy / (mid_energy + side_energy).max(1e-12);
        assert!(
            mid_fraction > 0.999,
            "stretcher broke stereo M/S phase coherence: mid_fraction={mid_fraction:.5}"
        );
    }

    #[test]
    fn equal_power_crossfade_holds_pink_noise_rms_constant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let n_frames = SAMPLE_RATE as usize * 4;
        let mut a = vec![0.0_f64; n_frames * 2];
        let mut b = vec![0.0_f64; n_frames * 2];
        for x in &mut a {
            *x = rng.random_range(-0.4..0.4);
        }
        for x in &mut b {
            *x = rng.random_range(-0.4..0.4);
        }

        let spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64 * 3);
        let out = run_offline(spec, &a, &b);

        let window = SAMPLE_RATE as usize / 2; // 0.5 s windows
        let mut rms_dbs = Vec::new();
        for chunk in out.chunks_exact(window * 2) {
            let mean_sq = chunk.iter().map(|s| s * s).sum::<f64>() / chunk.len() as f64;
            rms_dbs.push(10.0 * mean_sq.max(1e-12).log10());
        }
        let min_db = rms_dbs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_db = rms_dbs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_db - min_db < 0.4,
            "equal-power RMS varied by {:.2} dB across crossfade: {rms_dbs:?}",
            max_db - min_db
        );
    }

    #[test]
    fn boundary_entry_and_skip_abort_produce_no_step_clicks() {
        let fs = f64::from(SAMPLE_RATE);
        let freq = 220.0;
        let n_frames = SAMPLE_RATE as usize * 2;
        let mut sine = vec![0.0_f64; n_frames * 2];
        for i in 0..n_frames {
            let s = 0.5 * (2.0 * PI * freq * i as f64 / fs).sin();
            sine[2 * i] = s;
            sine[2 * i + 1] = s;
        }
        let silence = vec![0.0_f64; n_frames * 2];

        let mut spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64);
        spec.out_curves.eq_low = Some(AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![CurvePoint { x: 0.0, y: 1.0 }, CurvePoint { x: 1.0, y: 1.0 }],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        });

        let mut proc = TransitionProcessor::new(spec);
        proc.push_deck_a(&sine);
        proc.push_deck_b(&silence);

        let mut out = Vec::new();
        proc.render(4096, &mut out);
        proc.begin_abort_fade();
        while !proc.is_complete() {
            if proc.render(256, &mut out) == 0 {
                break;
            }
        }

        // Max per-sample step of a 0.5-amplitude 220 Hz sine is 0.5 * 2*pi*220/44100 ~= 0.0157;
        // allow generous margin for phase/fade slope while catching any click (> 0.08).
        let mut max_step = 0.0_f64;
        for w in out.chunks_exact(2).collect::<Vec<_>>().windows(2) {
            max_step = max_step.max((w[1][0] - w[0][0]).abs());
        }
        assert!(
            max_step < 0.08,
            "click detected at entry or abort fade: max_step={max_step:.4}"
        );
        // Final sample after abort fade must settle near silence.
        let last_l = out[out.len() - 2].abs();
        assert!(
            last_l < 0.01,
            "abort fade did not settle to silence: {last_l}"
        );
    }

    #[test]
    fn extreme_overlap_with_resonant_filter_and_master_limiter_stays_finite_and_safe() {
        let mut rng = SmallRng::seed_from_u64(1337);
        let n_frames = SAMPLE_RATE as usize * 2;
        let mut a: Vec<f64> = (0..n_frames * 2)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut b: Vec<f64> = (0..n_frames * 2)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let config = PlayerConfig {
            normalisation: true,
            normalisation_method: NormalisationMethod::Dynamic,
            normalisation_threshold_dbfs: -2.0,
            ..Default::default()
        };
        apply_track_normalisation(&mut a, &config, 1.0);
        apply_track_normalisation(&mut b, &config, 1.0);

        let mut spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64);
        spec.out_curves.filter_cutoff = Some(AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![
                    CurvePoint { x: 0.0, y: 0.45 },
                    CurvePoint { x: 1.0, y: 0.05 },
                ],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        });
        spec.out_curves.filter_resonance = Some(AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![CurvePoint { x: 0.0, y: 5.0 }, CurvePoint { x: 1.0, y: 5.0 }],
                0.0,
                1.0,
            )],
            minimum: 0.5,
            maximum: 8.0,
        });

        let mut mixed = run_offline(spec, &a, &b);
        let mut limiter = MasterLimiter::new(&config);
        limiter.process_post_mix(&mut mixed, &config, 1.0);

        assert!(mixed.iter().all(|s| s.is_finite()));
        let settle = (SAMPLE_RATE as usize / 10) * 2;
        let tail = &mixed[settle..];
        let peak = tail.iter().fold(0.0_f64, |acc, &s| acc.max(s.abs()));
        let peak_db = ratio_to_db(peak);
        let rms = (tail.iter().map(|s| s * s).sum::<f64>() / tail.len() as f64).sqrt();
        let rms_db = ratio_to_db(rms);
        // Feedforward limiter has ~5 ms attack so isolated white-noise resonant spikes can
        // momentarily approach 0 dBFS, while staying non-clipping and holding RMS well below threshold.
        assert!(
            peak_db <= 0.5 && rms_db <= -2.0,
            "post-limiter resonant safety failed: peak={peak_db:.2} dBFS, rms={rms_db:.2} dBFS"
        );
    }

    #[test]
    fn bass_swap_transfers_low_band_smoothly_without_overload() {
        let fs = f64::from(SAMPLE_RATE);
        let n_frames = SAMPLE_RATE as usize * 3;
        let mut a = vec![0.0_f64; n_frames * 2];
        let mut b = vec![0.0_f64; n_frames * 2];
        for i in 0..n_frames {
            let t = i as f64 / fs;
            // Distinct sub-bass frequencies (55 Hz vs 73 Hz) so energy sums incoherently.
            let sa = 0.5 * (2.0 * PI * 55.0 * t).sin();
            let sb = 0.5 * (2.0 * PI * 73.0 * t).sin();
            a[2 * i] = sa;
            a[2 * i + 1] = sa;
            b[2 * i] = sb;
            b[2 * i + 1] = sb;
        }

        let flat_vol = AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![CurvePoint { x: 0.0, y: 1.0 }, CurvePoint { x: 1.0, y: 1.0 }],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        };
        // Smooth bass swap centered at u = 0.5 (Deck A kills low band while Deck B brings low band in).
        let low_out = AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![
                    CurvePoint { x: 0.0, y: 1.0 },
                    CurvePoint { x: 0.35, y: 1.0 },
                    CurvePoint { x: 0.65, y: 0.0 },
                    CurvePoint { x: 1.0, y: 0.0 },
                ],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        };
        let low_in = AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![
                    CurvePoint { x: 0.0, y: 0.0 },
                    CurvePoint { x: 0.35, y: 0.0 },
                    CurvePoint { x: 0.65, y: 1.0 },
                    CurvePoint { x: 1.0, y: 1.0 },
                ],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        };

        let mut spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64 * 2);
        spec.out_curves.volume = Some(flat_vol.clone());
        spec.in_curves.volume = Some(flat_vol);
        spec.out_curves.eq_low = Some(low_out);
        spec.in_curves.eq_low = Some(low_in);

        let out = run_offline(spec, &a, &b);
        let nominal_rms = 0.5 / std::f64::consts::SQRT_2;
        let nominal_db = 20.0 * nominal_rms.log10();

        let window = SAMPLE_RATE as usize / 4; // 250 ms windows
        for (idx, chunk) in out.chunks_exact(window * 2).enumerate() {
            let rms = (chunk.iter().map(|s| s * s).sum::<f64>() / chunk.len() as f64).sqrt();
            let db = 20.0 * rms.max(1e-12).log10();
            // Linear bass swap dips at most ~3 dB at midpoint and never exceeds +1.5 dB overload.
            assert!(
                db <= nominal_db + 1.5 && db >= nominal_db - 4.0,
                "bass swap window {idx} drifted outside expected band: {db:.2} dB vs nominal {nominal_db:.2} dB"
            );
        }
    }

    #[test]
    fn exit_handoff_continues_deck_b_with_zero_frame_gap_and_no_click() {
        let fs = f64::from(SAMPLE_RATE);
        let n_frames = SAMPLE_RATE as usize * 5;
        let mut a = vec![0.0_f64; n_frames * 2];
        let mut b = vec![0.0_f64; n_frames * 2];
        for i in 0..n_frames {
            let t = i as f64 / fs;
            let sa = 0.35 * (2.0 * PI * 330.0 * t).sin();
            let sb = 0.45 * (2.0 * PI * 220.0 * t).sin();
            a[2 * i] = sa;
            a[2 * i + 1] = sa;
            b[2 * i] = sb;
            b[2 * i + 1] = sb;
        }

        let spec = TransitionSpec {
            start_a_frame: 0,
            start_b_frame: 0,
            duration_frames: 0,
            duration_beats: Some(8.0), // ~3.2 s at ~147.5 BPM
            handoff_fraction: 0.5,
            tempo: Some(TempoPlan::beatmatched(145.0, 150.0)),
            out_curves: DeckCurves::default(),
            in_curves: DeckCurves::default(),
            eq_topology: super::super::spec::EqTopology::default(),
        };

        let mut proc = TransitionProcessor::new(spec);
        assert!(proc.deck_b.stretcher.is_pitch_preserving());
        let mut pos_a = 0usize;
        let mut pos_b = 0usize;
        let mut out = Vec::new();
        let feed_chunk = 2048 * CHANNELS;

        while !proc.is_complete() {
            while proc.needs_deck_a_input() && pos_a < a.len() {
                let end = (pos_a + feed_chunk).min(a.len());
                proc.push_deck_a(&a[pos_a..end]);
                pos_a = end;
            }
            while proc.needs_deck_b_input() && pos_b < b.len() {
                let end = (pos_b + feed_chunk).min(b.len());
                proc.push_deck_b(&b[pos_b..end]);
                pos_b = end;
            }
            let wrote = proc.render(512, &mut out);
            if wrote == 0 && pos_a >= a.len() && pos_b >= b.len() {
                break;
            }
        }
        assert!(proc.is_complete());

        let pushed_b_frames = (pos_b / CHANNELS) as u64;
        let (next_frame, continuation) = proc.drain_deck_b_continuation();
        let cont_frames = (continuation.len() / CHANNELS) as u64;
        assert_eq!(
            next_frame + cont_frames,
            pushed_b_frames,
            "continuation frame accounting mismatch: next_frame={next_frame} + cont_frames={cont_frames} != pushed_b_frames={pushed_b_frames}"
        );
        assert!(
            !continuation.is_empty(),
            "expected buffered lookahead continuation frames from PitchPreserving stretcher"
        );

        // Verify the splice between last sample of transition output and first dry continuation
        // frame has zero step click (220 Hz sine max per-sample step is ~0.014).
        let last_l = out[out.len() - 2];
        let next_l = continuation[0];
        let splice_step = (next_l - last_l).abs();
        assert!(
            splice_step < 0.03,
            "click at exit handoff splice: last={last_l:.5} next={next_l:.5} step={splice_step:.5}"
        );
    }

    #[test]
    fn identity_bpm_beatmatched_mix_bypasses_stft_and_preserves_crest_factor() {
        // Brickwall-limited +/-0.5 square-ish pulse (crest factor ~0 dB). Passing through an
        // unnecessary 1.0x STFT stretcher would smear phase and inflate peak amplitude; bypassing
        // the stretcher when bpm_a == bpm_b keeps peaks bit-clean at 0.5.
        let n_frames = SAMPLE_RATE as usize * 3;
        let mut a = vec![0.0_f64; n_frames * 2];
        let mut b = vec![0.0_f64; n_frames * 2];
        for i in 0..n_frames {
            let sq = if (i / 100) % 2 == 0 { 0.5 } else { -0.5 };
            a[2 * i] = sq;
            a[2 * i + 1] = sq;
            b[2 * i] = 0.0;
            b[2 * i + 1] = 0.0;
        }
        let flat_one = AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![CurvePoint { x: 0.0, y: 1.0 }, CurvePoint { x: 1.0, y: 1.0 }],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        };
        let mut spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64 * 2);
        spec.duration_beats = Some(8.0);
        spec.tempo = Some(TempoPlan::beatmatched(174.0, 174.0));
        spec.out_curves.volume = Some(flat_one);
        spec.in_curves.volume = Some(AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 0.0 }],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        });

        let proc = TransitionProcessor::new(spec.clone());
        assert_eq!(proc.stretch_mode, TimeStretchMode::Off);
        let out = run_offline(spec, &a, &b);
        let peak = out.iter().fold(0.0_f64, |acc, &s| acc.max(s.abs()));
        assert!(
            (peak - 0.5).abs() < 1e-6,
            "identity-BPM passthrough inflated peak: {peak}"
        );
    }
}
