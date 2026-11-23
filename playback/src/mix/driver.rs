//! Dual-deck transition lifecycle driver used by [`crate::player`].
//!
//! Decouples the real-time deck pumping, sample-accurate entry splitting, pause/resume
//! lockstep, abort micro-fade draining, pre/post-commit seek target selection, and
//! error/EOF recovery from Spotify networking so every state-machine invariant can be
//! tested deterministically with synthetic decoders.

use crate::{
    NUM_CHANNELS, SAMPLE_RATE,
    config::PlayerConfig,
    decoder::{AudioDecoder, AudioPacket},
    mix::{TransitionProcessor, TransitionSpec},
    player::{MasterLimiter, apply_track_normalisation},
};

const CHANNELS: usize = NUM_CHANNELS as usize;

/// Maximum tolerance (in PCM frames, ~20 ms at 44.1 kHz) past `start_a_frame` within which
/// a newly ready incoming track can still enter a beatmatched transition without audible drift.
pub const ENTRY_GRACE_FRAMES: u64 = 882;

/// Decision for how a decoded outgoing Deck A packet relates to `spec.start_a_frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryPacketDecision {
    /// Entire packet ends at or before `start_a_frame`; play completely dry.
    BeforeEntry,
    /// Packet straddles `start_a_frame`; play `dry_prefix_samples` dry and feed the suffix
    /// directly into the transition processor as frame 0 of Deck A.
    CrossesEntry { dry_prefix_samples: usize },
    /// Packet starts slightly after `start_a_frame` within [`ENTRY_GRACE_FRAMES`]; enter immediately.
    WithinGraceWindow,
    /// Outgoing track already passed `start_a_frame + ENTRY_GRACE_FRAMES`; abandon the transition
    /// so playback falls through cleanly to EOF without off-beat entry.
    PastGraceWindow,
}

/// Classify an outgoing packet spanning `[pkt_start_frame, pkt_start_frame + pkt_frames)` against
/// the scheduled transition entry `start_a_frame`.
pub fn split_entry_packet(
    samples_len: usize,
    pkt_start_frame: u64,
    start_a_frame: u64,
) -> EntryPacketDecision {
    let pkt_frames = (samples_len / CHANNELS) as u64;
    let pkt_end_frame = pkt_start_frame.saturating_add(pkt_frames);

    if pkt_end_frame <= start_a_frame {
        EntryPacketDecision::BeforeEntry
    } else if pkt_start_frame <= start_a_frame {
        let dry_frames = (start_a_frame - pkt_start_frame) as usize;
        EntryPacketDecision::CrossesEntry {
            dry_prefix_samples: (dry_frames * CHANNELS).min(samples_len),
        }
    } else if pkt_start_frame <= start_a_frame.saturating_add(ENTRY_GRACE_FRAMES) {
        EntryPacketDecision::WithinGraceWindow
    } else {
        EntryPacketDecision::PastGraceWindow
    }
}

/// One deck (outgoing Track A or incoming Track B) owned by [`ActiveTransition`].
pub struct TransitionDeck<M> {
    pub decoder: Box<dyn AudioDecoder + Send>,
    pub normalisation_factor: f64,
    pub meta: M,
}

/// Output produced by one non-blocking [`ActiveTransition::step`] iteration.
#[derive(Debug)]
pub struct TransitionStep {
    /// Interleaved stereo `f64` samples already processed through the post-mix master limiter
    /// and volume stage, ready for [`crate::audio_backend::Sink::write`].
    pub samples: Vec<f64>,
    /// Emitted once when transition progress crosses `spec.handoff_fraction`, carrying the
    /// incoming track's current position in milliseconds.
    pub handoff_due_ms: Option<u32>,
    /// `true` once the transition has finished (including wet-to-dry exit blend).
    pub completed: bool,
}

/// Active dual-deck transition state machine owning both decoders and [`TransitionProcessor`].
pub struct ActiveTransition<M> {
    outgoing: TransitionDeck<M>,
    incoming: TransitionDeck<M>,
    processor: TransitionProcessor,
    committed: bool,
    paused: bool,
    deck_a_eof: bool,
    deck_b_eof: bool,
}

impl<M> ActiveTransition<M> {
    /// Start an active transition from outgoing Deck A to incoming Deck B.
    ///
    /// `initial_deck_a_normalized_pcm` must already have Deck A's per-track normalisation applied
    /// (typically the post-split suffix of the packet crossing `spec.start_a_frame`).
    pub fn new(
        outgoing: TransitionDeck<M>,
        incoming: TransitionDeck<M>,
        spec: TransitionSpec,
        initial_deck_a_normalized_pcm: &[f64],
    ) -> Self {
        let mut processor = TransitionProcessor::new(spec);
        if !initial_deck_a_normalized_pcm.is_empty() {
            processor.push_deck_a(initial_deck_a_normalized_pcm);
        }
        Self {
            outgoing,
            incoming,
            processor,
            committed: false,
            paused: false,
            deck_a_eof: false,
            deck_b_eof: false,
        }
    }

    pub fn outgoing_meta(&self) -> &M {
        &self.outgoing.meta
    }

    pub fn incoming_meta(&self) -> &M {
        &self.incoming.meta
    }

    /// Metadata of whichever track is currently reported to Spotify Connect (outgoing pre-commit,
    /// incoming post-commit).
    pub fn reported_meta(&self) -> &M {
        if self.committed {
            &self.incoming.meta
        } else {
            &self.outgoing.meta
        }
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    #[cfg(test)]
    pub fn progress(&self) -> f64 {
        self.processor.progress()
    }

    /// Position in milliseconds of the currently reported track (Deck A pre-commit, Deck B post-commit).
    pub fn reported_position_ms(&self) -> u32 {
        let frame = if self.committed {
            self.processor.deck_b_source_frame()
        } else {
            self.processor.deck_a_source_frame()
        };
        Self::frame_to_ms(frame)
    }

    /// Current incoming Deck B playback position in milliseconds.
    pub fn incoming_position_ms(&self) -> u32 {
        Self::frame_to_ms(self.processor.deck_b_source_frame())
    }

    #[inline]
    fn frame_to_ms(frame: f64) -> u32 {
        ((frame.max(0.0) * 1000.0) / f64::from(SAMPLE_RATE)).round() as u32
    }

    fn pump_deck_a(&mut self, config: &PlayerConfig) {
        // Bound packets per step so a corrupt/empty decoder can never spin forever.
        for _ in 0..32 {
            if self.deck_a_eof || !self.processor.needs_deck_a_input() {
                break;
            }
            match self.outgoing.decoder.next_packet() {
                Ok(Some((_, AudioPacket::Samples(mut samples)))) => {
                    if !samples.is_empty() {
                        apply_track_normalisation(
                            &mut samples,
                            config,
                            self.outgoing.normalisation_factor,
                        );
                        self.processor.push_deck_a(&samples);
                    }
                }
                Ok(Some((_, AudioPacket::Raw(_)))) => {
                    warn!("Mix transition encountered raw passthrough packet on outgoing deck");
                    self.deck_a_eof = true;
                    self.processor.set_deck_a_eof();
                    break;
                }
                Ok(None) => {
                    self.deck_a_eof = true;
                    self.processor.set_deck_a_eof();
                    break;
                }
                Err(e) => {
                    warn!(
                        "Outgoing deck decode error during transition ({e}); ending outgoing deck"
                    );
                    self.deck_a_eof = true;
                    self.processor.set_deck_a_eof();
                    break;
                }
            }
        }
    }

    fn pump_deck_b(&mut self, config: &PlayerConfig) {
        for _ in 0..32 {
            if self.deck_b_eof || !self.processor.needs_deck_b_input() {
                break;
            }
            match self.incoming.decoder.next_packet() {
                Ok(Some((_, AudioPacket::Samples(mut samples)))) => {
                    if !samples.is_empty() {
                        apply_track_normalisation(
                            &mut samples,
                            config,
                            self.incoming.normalisation_factor,
                        );
                        self.processor.push_deck_b(&samples);
                    }
                }
                Ok(Some((_, AudioPacket::Raw(_)))) => {
                    warn!("Mix transition encountered raw passthrough packet on incoming deck");
                    self.deck_b_eof = true;
                    self.processor.set_deck_b_eof();
                    break;
                }
                Ok(None) => {
                    self.deck_b_eof = true;
                    self.processor.set_deck_b_eof();
                    break;
                }
                Err(e) => {
                    warn!(
                        "Incoming deck decode error during transition ({e}); ending incoming deck"
                    );
                    self.deck_b_eof = true;
                    self.processor.set_deck_b_eof();
                    break;
                }
            }
        }
    }

    /// Advance the active transition by up to `max_frames` stereo frames.
    pub fn step(
        &mut self,
        config: &PlayerConfig,
        limiter: &mut MasterLimiter,
        volume: f64,
        max_frames: usize,
    ) -> TransitionStep {
        if self.paused {
            return TransitionStep {
                samples: Vec::new(),
                handoff_due_ms: None,
                completed: false,
            };
        }

        self.pump_deck_a(config);
        self.pump_deck_b(config);

        let mut mixed = Vec::with_capacity(max_frames * CHANNELS);
        let written = self.processor.render(max_frames, &mut mixed);

        // If priming needed more packets than the first pump batch, pump once more.
        if written == 0 && !self.processor.is_complete() {
            self.pump_deck_a(config);
            self.pump_deck_b(config);
            let _ = self.processor.render(max_frames, &mut mixed);
        }

        if !mixed.is_empty() {
            limiter.process_post_mix(&mut mixed, config, volume);
        }

        let mut handoff_due_ms = None;
        if !self.committed && self.processor.take_handoff_due() {
            self.committed = true;
            handoff_due_ms = Some(self.incoming_position_ms());
        }

        let completed = self.processor.is_complete()
            || (mixed.is_empty() && self.deck_a_eof && self.deck_b_eof);

        if completed && !self.committed {
            self.committed = true;
            handoff_due_ms = Some(self.incoming_position_ms());
        }

        TransitionStep {
            samples: mixed,
            handoff_due_ms,
            completed,
        }
    }

    /// Trigger a click-free ~15 ms cosine fade-out and drain its post-limiter PCM so a manual
    /// skip, load, stop, or seek never produces an audible step discontinuity.
    pub fn abort_and_drain(
        &mut self,
        config: &PlayerConfig,
        limiter: &mut MasterLimiter,
        volume: f64,
    ) -> Vec<f64> {
        if self.paused {
            return Vec::new();
        }
        self.processor.begin_abort_fade();
        let mut out = Vec::with_capacity(1024 * CHANNELS);
        for _ in 0..8 {
            if self.processor.is_complete() {
                break;
            }
            self.pump_deck_a(config);
            self.pump_deck_b(config);
            let wrote = self.processor.render(256, &mut out);
            if wrote == 0 {
                break;
            }
        }
        if !out.is_empty() {
            limiter.process_post_mix(&mut out, config, volume);
        }
        out
    }

    /// Consume the completed transition and return `(incoming_deck, next_dry_frame, continuation_pcm)`.
    ///
    /// `continuation_pcm` already has Deck B's per-track normalisation applied and must be fed
    /// directly through [`MasterLimiter::process_post_mix`] before resuming `incoming_deck.decoder`.
    pub fn into_incoming_continuation(mut self) -> (TransitionDeck<M>, u64, Vec<f64>) {
        let (next_frame, continuation) = self.processor.drain_deck_b_continuation();
        (self.incoming, next_frame, continuation)
    }

    /// Consume an aborted transition for a user seek, returning whichever deck the user sees as
    /// current (outgoing Deck A pre-commit, incoming Deck B post-commit).
    ///
    /// When aborted pre-commit, also returns `(incoming_deck, spec)` so the player can keep the
    /// transition armed (and the incoming deck preloaded) if the user scrubs back before `start_a`.
    pub fn into_abort_for_seek(
        self,
    ) -> (
        TransitionDeck<M>,
        Option<(TransitionDeck<M>, TransitionSpec)>,
    ) {
        if self.committed {
            (self.incoming, None)
        } else {
            let spec = self.processor.spec().clone();
            (self.outgoing, Some((self.incoming, spec)))
        }
    }

    /// Consume an aborted transition and return whichever deck's metadata satisfies `predicate`
    /// (checking incoming first, then outgoing), allowing `Load` commands during a transition
    /// (e.g. Skip Next to Track B) to reuse an already-loaded deck without re-buffering.
    pub fn into_matching_deck<F>(self, predicate: F) -> Option<TransitionDeck<M>>
    where
        F: Fn(&M) -> bool,
    {
        if predicate(&self.incoming.meta) {
            Some(self.incoming)
        } else if predicate(&self.outgoing.meta) {
            Some(self.outgoing)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::NormalisationMethod,
        decoder::{AudioPacketPosition, DecoderError, DecoderResult},
        mix::spec::TempoPlan,
    };
    use std::f64::consts::PI;

    struct SyntheticDecoder {
        samples: Vec<f64>,
        cursor_frames: usize,
        packet_frames: usize,
        error_at_frame: Option<usize>,
    }

    impl SyntheticDecoder {
        fn sine(freq_hz: f64, amplitude: f64, total_frames: usize, packet_frames: usize) -> Self {
            let fs = f64::from(SAMPLE_RATE);
            let mut samples = Vec::with_capacity(total_frames * CHANNELS);
            for i in 0..total_frames {
                let s = amplitude * (2.0 * PI * freq_hz * i as f64 / fs).sin();
                samples.push(s);
                samples.push(s);
            }
            Self {
                samples,
                cursor_frames: 0,
                packet_frames,
                error_at_frame: None,
            }
        }

        fn with_error_at_frame(mut self, frame: usize) -> Self {
            self.error_at_frame = Some(frame);
            self
        }
    }

    impl AudioDecoder for SyntheticDecoder {
        fn seek(&mut self, position_ms: u32) -> Result<u32, DecoderError> {
            let target = (u64::from(position_ms) * u64::from(SAMPLE_RATE) / 1000) as usize;
            self.cursor_frames = target.min(self.samples.len() / CHANNELS);
            Ok(position_ms)
        }

        fn seek_exact(&mut self, target_frame: u64) -> Result<u64, DecoderError> {
            let max_frames = self.samples.len() / CHANNELS;
            self.cursor_frames = (target_frame as usize).min(max_frames);
            Ok(self.cursor_frames as u64)
        }

        fn position_frames(&self) -> Option<u64> {
            Some(self.cursor_frames as u64)
        }

        fn next_packet(&mut self) -> DecoderResult<Option<(AudioPacketPosition, AudioPacket)>> {
            if let Some(err_frame) = self.error_at_frame {
                if self.cursor_frames >= err_frame {
                    return Err(DecoderError::SymphoniaDecoder(
                        "injected decode error".into(),
                    ));
                }
            }
            let total_frames = self.samples.len() / CHANNELS;
            if self.cursor_frames >= total_frames {
                return Ok(None);
            }
            let start = self.cursor_frames;
            let end = (start + self.packet_frames).min(total_frames);
            self.cursor_frames = end;
            let pos = AudioPacketPosition {
                position_ms: (start as u64 * 1000 / u64::from(SAMPLE_RATE)) as u32,
                skipped: false,
            };
            Ok(Some((
                pos,
                AudioPacket::Samples(self.samples[start * CHANNELS..end * CHANNELS].to_vec()),
            )))
        }
    }

    fn test_config() -> PlayerConfig {
        PlayerConfig {
            normalisation: true,
            normalisation_method: NormalisationMethod::Dynamic,
            normalisation_threshold_dbfs: -2.0,
            ..Default::default()
        }
    }

    #[test]
    fn split_entry_packet_handles_before_straddle_grace_and_late_timeout() {
        let pkt_samples = 1024 * CHANNELS;
        assert_eq!(
            split_entry_packet(pkt_samples, 0, 2000),
            EntryPacketDecision::BeforeEntry
        );
        assert_eq!(
            split_entry_packet(pkt_samples, 1024, 1500),
            EntryPacketDecision::CrossesEntry {
                dry_prefix_samples: (1500 - 1024) * CHANNELS
            }
        );
        assert_eq!(
            split_entry_packet(pkt_samples, 1500 + 400, 1500),
            EntryPacketDecision::WithinGraceWindow
        );
        // Failure injection: preload arrived too late (> 20 ms past start_a_frame) -> cancel transition.
        assert_eq!(
            split_entry_packet(pkt_samples, 1500 + ENTRY_GRACE_FRAMES + 1, 1500),
            EntryPacketDecision::PastGraceWindow
        );
    }

    #[test]
    fn full_lifecycle_emits_single_handoff_and_continues_deck_b_seamlessly() {
        let cfg = test_config();
        let mut limiter = MasterLimiter::new(&cfg);

        let dec_a = SyntheticDecoder::sine(330.0, 0.35, SAMPLE_RATE as usize * 5, 1024);
        let dec_b = SyntheticDecoder::sine(220.0, 0.45, SAMPLE_RATE as usize * 6, 1024);

        let spec = TransitionSpec {
            start_a_frame: 0,
            start_b_frame: 0,
            duration_frames: 0,
            duration_beats: Some(8.0),
            handoff_fraction: 0.5,
            tempo: Some(TempoPlan::beatmatched(145.0, 150.0)),
            out_curves: Default::default(),
            in_curves: Default::default(),
            eq_topology: Default::default(),
        };

        let mut active = ActiveTransition::new(
            TransitionDeck {
                decoder: Box::new(dec_a),
                normalisation_factor: 1.0,
                meta: "track_a",
            },
            TransitionDeck {
                decoder: Box::new(dec_b),
                normalisation_factor: 1.0,
                meta: "track_b",
            },
            spec,
            &[],
        );

        let mut handoffs = Vec::new();
        let mut out = Vec::new();
        let mut completed = false;

        assert_eq!(*active.reported_meta(), "track_a");
        for _ in 0..2000 {
            let step = active.step(&cfg, &mut limiter, 1.0, 512);
            out.extend_from_slice(&step.samples);
            if let Some(pos_ms) = step.handoff_due_ms {
                handoffs.push(pos_ms);
                assert_eq!(*active.reported_meta(), "track_b");
            }
            if step.completed {
                completed = true;
                break;
            }
        }
        assert!(completed, "transition did not complete");
        assert_eq!(handoffs.len(), 1, "handoff must fire exactly once");
        assert!(
            handoffs[0] > 500,
            "handoff occurred unexpectedly early: {handoffs:?}"
        );

        let (incoming, next_frame, mut continuation) = active.into_incoming_continuation();
        assert_eq!(incoming.meta, "track_b");
        assert!(!continuation.is_empty());

        // Playing continuation through post-mix limiter and then next decoded packet must match exact frame index.
        let decoder_pos = incoming.decoder.position_frames().unwrap();
        assert_eq!(
            next_frame + (continuation.len() / CHANNELS) as u64,
            decoder_pos,
            "continuation + decoder cursor drifted"
        );
        limiter.process_post_mix(&mut continuation, &cfg, 1.0);
        let splice_step = (continuation[0] - out[out.len() - 2]).abs();
        assert!(
            splice_step < 0.03,
            "step click at transition exit splice: {splice_step:.5}"
        );
    }

    #[test]
    fn pause_freezes_both_decks_and_resumes_without_click() {
        let cfg = test_config();
        let mut limiter = MasterLimiter::new(&cfg);

        let dec_a = SyntheticDecoder::sine(220.0, 0.4, SAMPLE_RATE as usize * 3, 512);
        let dec_b = SyntheticDecoder::sine(440.0, 0.4, SAMPLE_RATE as usize * 3, 512);
        let spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64 * 2);

        let mut active = ActiveTransition::new(
            TransitionDeck {
                decoder: Box::new(dec_a),
                normalisation_factor: 1.0,
                meta: "a",
            },
            TransitionDeck {
                decoder: Box::new(dec_b),
                normalisation_factor: 1.0,
                meta: "b",
            },
            spec,
            &[],
        );

        let before = active.step(&cfg, &mut limiter, 1.0, 512);
        assert!(!before.samples.is_empty());
        let pos_before_pause = active.reported_position_ms();

        active.pause();
        let during_pause = active.step(&cfg, &mut limiter, 1.0, 512);
        assert!(during_pause.samples.is_empty());
        assert!(!during_pause.completed);
        assert_eq!(active.reported_position_ms(), pos_before_pause);

        active.resume();
        let after = active.step(&cfg, &mut limiter, 1.0, 512);
        assert!(!after.samples.is_empty());

        let step = (after.samples[0] - before.samples[before.samples.len() - 2]).abs();
        assert!(
            step < 0.04,
            "discontinuity across pause/resume boundary: {step:.5}"
        );
    }

    #[test]
    fn abort_micro_fade_and_seek_target_depend_on_commit_state() {
        let cfg = test_config();
        let mut limiter = MasterLimiter::new(&cfg);

        for want_committed in [false, true] {
            let dec_a = SyntheticDecoder::sine(220.0, 0.5, SAMPLE_RATE as usize * 4, 512);
            let dec_b = SyntheticDecoder::sine(330.0, 0.5, SAMPLE_RATE as usize * 4, 512);
            let spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64);

            let mut active = ActiveTransition::new(
                TransitionDeck {
                    decoder: Box::new(dec_a),
                    normalisation_factor: 1.0,
                    meta: "deck_a",
                },
                TransitionDeck {
                    decoder: Box::new(dec_b),
                    normalisation_factor: 1.0,
                    meta: "deck_b",
                },
                spec,
                &[],
            );

            let mut last_sample = 0.0;
            loop {
                let s = active.step(&cfg, &mut limiter, 1.0, 512);
                if let Some(&l) = s.samples.get(s.samples.len().saturating_sub(2)) {
                    last_sample = l;
                }
                if !want_committed && active.progress() > 0.15 {
                    assert!(!active.is_committed());
                    break;
                }
                if want_committed && active.is_committed() {
                    break;
                }
            }

            let fade = active.abort_and_drain(&cfg, &mut limiter, 1.0);
            assert!(
                fade.len() >= 600 * CHANNELS,
                "expected ~15 ms abort fade, got {} samples",
                fade.len()
            );
            assert!(
                (fade[0] - last_sample).abs() < 0.06,
                "click at start of abort fade"
            );
            assert!(
                fade[fade.len() - 2].abs() < 0.015,
                "abort fade did not settle near silence: {}",
                fade[fade.len() - 2]
            );

            let (target, restored) = active.into_abort_for_seek();
            assert_eq!(
                target.meta,
                if want_committed { "deck_b" } else { "deck_a" }
            );
            if want_committed {
                assert!(restored.is_none());
            } else {
                let (incoming, restored_spec) =
                    restored.expect("pre-commit seek must preserve incoming deck and spec");
                assert_eq!(incoming.meta, "deck_b");
                assert_eq!(restored_spec.duration_frames, SAMPLE_RATE as u64);
            }
        }
    }

    #[test]
    fn into_matching_deck_reuses_incoming_or_outgoing_deck_on_manual_load() {
        let dec_a = SyntheticDecoder::sine(220.0, 0.5, SAMPLE_RATE as usize * 2, 512);
        let dec_b = SyntheticDecoder::sine(330.0, 0.5, SAMPLE_RATE as usize * 2, 512);
        let spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64);
        let make = |a, b, s| {
            ActiveTransition::new(
                TransitionDeck {
                    decoder: Box::new(a),
                    normalisation_factor: 1.0,
                    meta: "track_a",
                },
                TransitionDeck {
                    decoder: Box::new(b),
                    normalisation_factor: 1.0,
                    meta: "track_b",
                },
                s,
                &[],
            )
        };

        let matched_b = make(dec_a, dec_b, spec.clone()).into_matching_deck(|m| *m == "track_b");
        assert_eq!(matched_b.map(|d| d.meta), Some("track_b"));

        let matched_a = make(
            SyntheticDecoder::sine(220.0, 0.5, SAMPLE_RATE as usize * 2, 512),
            SyntheticDecoder::sine(330.0, 0.5, SAMPLE_RATE as usize * 2, 512),
            spec.clone(),
        )
        .into_matching_deck(|m| *m == "track_a");
        assert_eq!(matched_a.map(|d| d.meta), Some("track_a"));

        let matched_none = make(
            SyntheticDecoder::sine(220.0, 0.5, SAMPLE_RATE as usize * 2, 512),
            SyntheticDecoder::sine(330.0, 0.5, SAMPLE_RATE as usize * 2, 512),
            spec,
        )
        .into_matching_deck(|m| *m == "track_c");
        assert!(matched_none.is_none());
    }

    #[test]
    fn early_outgoing_decode_error_still_completes_and_commits_incoming_deck() {
        let cfg = test_config();
        let mut limiter = MasterLimiter::new(&cfg);

        // Outgoing Track A throws a decode error 0.25 s into a 1.0 s overlap.
        let dec_a = SyntheticDecoder::sine(220.0, 0.4, SAMPLE_RATE as usize * 3, 512)
            .with_error_at_frame(SAMPLE_RATE as usize / 4);
        let dec_b = SyntheticDecoder::sine(330.0, 0.4, SAMPLE_RATE as usize * 3, 512);
        let spec = TransitionSpec::equal_power_crossfade(0, 0, SAMPLE_RATE as u64);

        let mut active = ActiveTransition::new(
            TransitionDeck {
                decoder: Box::new(dec_a),
                normalisation_factor: 1.0,
                meta: "a",
            },
            TransitionDeck {
                decoder: Box::new(dec_b),
                normalisation_factor: 1.0,
                meta: "b",
            },
            spec,
            &[],
        );

        let mut committed = false;
        let mut completed = false;
        for _ in 0..1000 {
            let s = active.step(&cfg, &mut limiter, 1.0, 512);
            assert!(s.samples.iter().all(|x| x.is_finite()));
            if s.handoff_due_ms.is_some() {
                committed = true;
            }
            if s.completed {
                completed = true;
                break;
            }
        }
        assert!(completed && committed);
    }
}
