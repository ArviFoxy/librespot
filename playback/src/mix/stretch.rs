//! Real-time per-deck time-stretching (`PitchPreserving`, `Varispeed`, `Off`)
//! with exact source-frame position tracking.

use std::collections::VecDeque;

use timestretch::{
    Backend, Parameters, PreparedSession, ProcessStatus, RatePolicy, RatioConstraint,
    SessionConfig, StreamSession, TempoRate,
};

use super::spec::TimeStretchMode;
use crate::{NUM_CHANNELS, SAMPLE_RATE};

const CHANNELS: usize = NUM_CHANNELS as usize;

/// Real-time stereo time-stretcher for one deck, operating at [`SAMPLE_RATE`].
pub struct DeckStretcher {
    mode: StretcherImpl,
    source_origin_frame: f64,
    current_source_frame: f64,
    output_frames_emitted: u64,
}

enum StretcherImpl {
    Bypass {
        input: VecDeque<f64>,
        eof: bool,
    },
    Varispeed {
        /// Interleaved stereo samples buffered starting at `buf_start_frame`.
        buf: VecDeque<f64>,
        buf_start_frame: u64,
        read_pos_frames: f64,
        ratio: f64,
        eof: bool,
    },
    PitchPreserving {
        session: Box<StreamSession>,
        input: VecDeque<f32>,
        eof: bool,
        eof_notified: bool,
        next_cmd_id: u64,
        last_scheduled_ratio: f64,
        scratch_in: Vec<f32>,
        scratch_out: Vec<f32>,
    },
}

impl DeckStretcher {
    /// Construct a stretcher starting at `source_origin_frame` with initial speed `initial_ratio`.
    pub fn new(mode: TimeStretchMode, source_origin_frame: u64, initial_ratio: f64) -> Self {
        let ratio = initial_ratio.clamp(0.5, 2.0);
        let impl_state = match mode {
            TimeStretchMode::Off => StretcherImpl::Bypass {
                input: VecDeque::with_capacity(8192 * CHANNELS),
                eof: false,
            },
            TimeStretchMode::Varispeed => StretcherImpl::Varispeed {
                buf: VecDeque::with_capacity(8192 * CHANNELS),
                buf_start_frame: 0,
                read_pos_frames: 0.0,
                ratio,
                eof: false,
            },
            TimeStretchMode::PitchPreserving => {
                let tempo = TempoRate::new(ratio).unwrap_or_else(|_| TempoRate::new(1.0).unwrap());
                let mut parameters = Parameters::keylock(tempo);
                parameters.constraint = RatioConstraint::PreservePitch;
                let cfg = SessionConfig {
                    source_sample_rate: SAMPLE_RATE,
                    output_sample_rate: SAMPLE_RATE,
                    channels: CHANNELS,
                    rate_policy: RatePolicy::NativeOnly,
                    backend: Backend::Portable,
                    parameters,
                    max_commands: 1024,
                    max_inflight_frames: 65_536,
                    ..SessionConfig::default()
                };
                match PreparedSession::new(cfg) {
                    Ok(prepared) => StretcherImpl::PitchPreserving {
                        session: Box::new(prepared.activate()),
                        input: VecDeque::with_capacity(16_384 * CHANNELS),
                        eof: false,
                        eof_notified: false,
                        next_cmd_id: 1,
                        last_scheduled_ratio: ratio,
                        scratch_in: Vec::with_capacity(2048 * CHANNELS),
                        scratch_out: Vec::with_capacity(2048 * CHANNELS),
                    },
                    Err(e) => {
                        warn!(
                            "Failed to initialize pitch-preserving time-stretcher ({e:?}); falling back to varispeed"
                        );
                        StretcherImpl::Varispeed {
                            buf: VecDeque::with_capacity(8192 * CHANNELS),
                            buf_start_frame: 0,
                            read_pos_frames: 0.0,
                            ratio,
                            eof: false,
                        }
                    }
                }
            }
        };

        Self {
            mode: impl_state,
            source_origin_frame: source_origin_frame as f64,
            current_source_frame: source_origin_frame as f64,
            output_frames_emitted: 0,
        }
    }

    /// Whether this stretcher is actively running the pitch-preserving backend.
    pub fn is_pitch_preserving(&self) -> bool {
        matches!(self.mode, StretcherImpl::PitchPreserving { .. })
    }

    /// Push interleaved stereo `f64` samples into the stretcher input queue.
    pub fn push_samples(&mut self, interleaved: &[f64]) {
        match &mut self.mode {
            StretcherImpl::Bypass { input, .. } => {
                input.extend(
                    interleaved
                        .iter()
                        .map(|&s| if s.is_finite() { s } else { 0.0 }),
                );
            }
            StretcherImpl::Varispeed { buf, .. } => {
                buf.extend(
                    interleaved
                        .iter()
                        .map(|&s| if s.is_finite() { s } else { 0.0 }),
                );
            }
            StretcherImpl::PitchPreserving { input, .. } => {
                input.extend(interleaved.iter().map(
                    |&s| {
                        if s.is_finite() { s as f32 } else { 0.0 }
                    },
                ));
            }
        }
    }

    /// Signal that no more input samples will arrive for this deck.
    pub fn set_eof(&mut self) {
        match &mut self.mode {
            StretcherImpl::Bypass { eof, .. }
            | StretcherImpl::Varispeed { eof, .. }
            | StretcherImpl::PitchPreserving { eof, .. } => *eof = true,
        }
    }

    /// Number of unconsumed stereo input frames currently buffered ahead of processing.
    pub fn buffered_input_frames(&self) -> usize {
        match &self.mode {
            StretcherImpl::Bypass { input, .. } => input.len() / CHANNELS,
            StretcherImpl::Varispeed {
                buf,
                buf_start_frame,
                read_pos_frames,
                ..
            } => {
                let total = *buf_start_frame as f64 + (buf.len() / CHANNELS) as f64;
                (total - *read_pos_frames).max(0.0) as usize
            }
            StretcherImpl::PitchPreserving { input, .. } => input.len() / CHANNELS,
        }
    }

    /// Minimum input frames we prefer to keep buffered so stretching never starves.
    pub fn recommended_input_frames(&self) -> usize {
        match &self.mode {
            StretcherImpl::Bypass { .. } => 1024,
            StretcherImpl::Varispeed { .. } => 2048,
            StretcherImpl::PitchPreserving { session, .. } => {
                session.timing().startup_source_frames.max(4096) + 2048
            }
        }
    }

    /// Update the instantaneous tempo ratio `source_speed / output_speed`.
    pub fn set_ratio(&mut self, ratio: f64) {
        let ratio = ratio.clamp(0.5, 2.0);
        match &mut self.mode {
            StretcherImpl::Bypass { .. } => {}
            StretcherImpl::Varispeed { ratio: r, .. } => *r = ratio,
            StretcherImpl::PitchPreserving {
                session,
                next_cmd_id,
                last_scheduled_ratio,
                ..
            } => {
                if (ratio - *last_scheduled_ratio).abs() < 1e-4 {
                    return;
                }
                if let Ok(tempo) = TempoRate::new(ratio) {
                    let mut params = Parameters::keylock(tempo);
                    params.constraint = RatioConstraint::PreservePitch;
                    let at_frame = session.commit_horizon();
                    if session.schedule(*next_cmd_id, at_frame, params).is_ok() {
                        *next_cmd_id += 1;
                        *last_scheduled_ratio = ratio;
                    }
                }
            }
        }
    }

    /// Exact fractional source frame index corresponding to the last output frame emitted.
    #[inline]
    pub fn current_source_frame(&self) -> f64 {
        self.current_source_frame
    }

    /// Total output frames emitted by this stretcher so far.
    #[inline]
    pub fn output_frames_emitted(&self) -> u64 {
        self.output_frames_emitted
    }

    /// Whether the deck has reached EOF and drained all output frames.
    pub fn is_finished(&self) -> bool {
        match &self.mode {
            StretcherImpl::Bypass { input, eof } => *eof && input.len() < CHANNELS,
            StretcherImpl::Varispeed {
                buf,
                buf_start_frame,
                read_pos_frames,
                eof,
                ..
            } => {
                let end = *buf_start_frame as f64 + (buf.len() / CHANNELS) as f64;
                *eof && *read_pos_frames + 1.0 >= end
            }
            StretcherImpl::PitchPreserving { session, eof, .. } => *eof && session.finished(),
        }
    }

    /// Render up to `max_frames` stereo frames into `out` (appended as interleaved `f64`).
    /// Returns the number of stereo frames written.
    pub fn render(&mut self, max_frames: usize, out: &mut Vec<f64>) -> usize {
        if max_frames == 0 {
            return 0;
        }
        match &mut self.mode {
            StretcherImpl::Bypass { input, .. } => {
                let avail = input.len() / CHANNELS;
                let n = avail.min(max_frames);
                for _ in 0..(n * CHANNELS) {
                    if let Some(s) = input.pop_front() {
                        out.push(s);
                    }
                }
                self.current_source_frame += n as f64;
                self.output_frames_emitted += n as u64;
                n
            }
            StretcherImpl::Varispeed {
                buf,
                buf_start_frame,
                read_pos_frames,
                ratio,
                eof,
            } => {
                let mut written = 0usize;
                for _ in 0..max_frames {
                    let local_pos = *read_pos_frames - *buf_start_frame as f64;
                    let i1 = local_pos.floor() as isize;
                    let frac = local_pos - i1 as f64;
                    let avail_frames = (buf.len() / CHANNELS) as isize;

                    // Need 2 samples ahead for 4-point Catmull-Rom interpolation unless at EOF.
                    if i1 + 2 >= avail_frames && !*eof {
                        break;
                    }
                    if i1 >= avail_frames {
                        break;
                    }

                    let sample_at = |idx: isize, ch: usize| -> f64 {
                        let clamped = idx.clamp(0, avail_frames - 1) as usize;
                        buf[clamped * CHANNELS + ch]
                    };

                    for ch in 0..CHANNELS {
                        let y0 = sample_at(i1 - 1, ch);
                        let y1 = sample_at(i1, ch);
                        let y2 = sample_at(i1 + 1, ch);
                        let y3 = sample_at(i1 + 2, ch);
                        out.push(catmull_rom(y0, y1, y2, y3, frac));
                    }

                    *read_pos_frames += *ratio;
                    written += 1;

                    // Drop consumed frames while keeping 2 frames of lookbehind for interpolation.
                    let drop_frames = ((*read_pos_frames).floor() as u64)
                        .saturating_sub(*buf_start_frame)
                        .saturating_sub(2);
                    if drop_frames > 0 {
                        let drop = (drop_frames as usize).min(buf.len() / CHANNELS);
                        buf.drain(..drop * CHANNELS);
                        *buf_start_frame += drop as u64;
                    }
                }
                self.current_source_frame = self.source_origin_frame + *read_pos_frames;
                self.output_frames_emitted += written as u64;
                written
            }
            StretcherImpl::PitchPreserving {
                session,
                input,
                eof,
                eof_notified,
                scratch_in,
                scratch_out,
                ..
            } => {
                // Feed pending source requests and pump until we have `max_frames` ready or need more input.
                for _ in 0..64 {
                    while let Some(req) = session.source_request() {
                        let avail = input.len() / CHANNELS;
                        if avail == 0 {
                            if *eof && !*eof_notified {
                                let _ = session.finish_input();
                                *eof_notified = true;
                            }
                            break;
                        }
                        let take = avail.min(req.frames);
                        scratch_in.clear();
                        scratch_in.reserve(take * CHANNELS);
                        for _ in 0..(take * CHANNELS) {
                            scratch_in.push(input.pop_front().unwrap_or(0.0));
                        }
                        if session.supply(req, scratch_in).is_err() {
                            break;
                        }
                    }

                    if session.output_ready() >= max_frames {
                        break;
                    }

                    let report = session.advance(16);
                    // Drain any applied command events so the event ring never fills up.
                    while session.next_applied().is_some() {}

                    match report.status {
                        ProcessStatus::NeedSource => {
                            if input.len() < CHANNELS {
                                if *eof && !*eof_notified {
                                    let _ = session.finish_input();
                                    *eof_notified = true;
                                    continue;
                                }
                                break;
                            }
                        }
                        ProcessStatus::Finished
                        | ProcessStatus::Rendered
                        | ProcessStatus::Fault
                        | ProcessStatus::Paused
                        | ProcessStatus::NeedOutput => break,
                        ProcessStatus::Progress
                        | ProcessStatus::Priming
                        | ProcessStatus::NeedAcknowledgement => {}
                    }
                }

                let avail = session.output_ready();
                let to_read = avail.min(max_frames);
                if to_read == 0 {
                    return 0;
                }
                scratch_out.resize(to_read * CHANNELS, 0.0);
                let Ok(stamp) = session.read_output(&mut scratch_out[..to_read * CHANNELS]) else {
                    return 0;
                };
                if let Ok(receipt) = stamp.consumed(stamp.frames) {
                    let _ = session.acknowledge(receipt);
                }
                for &s in &scratch_out[..stamp.frames * CHANNELS] {
                    out.push(f64::from(s));
                }
                self.current_source_frame = self.source_origin_frame + session.position();
                self.output_frames_emitted += stamp.frames as u64;
                stamp.frames
            }
        }
    }

    /// Remaining unconsumed raw input samples in exact source-frame order after the current
    /// stretcher position (only exact when exiting a transition at unity ratio `1.0`).
    pub fn drain_unused_input_frames(&mut self) -> Vec<f64> {
        match &mut self.mode {
            StretcherImpl::Bypass { input, .. } => input.drain(..).collect(),
            StretcherImpl::Varispeed {
                buf,
                buf_start_frame,
                read_pos_frames,
                ..
            } => {
                let offset = ((*read_pos_frames).round() as u64).saturating_sub(*buf_start_frame);
                let drop = (offset as usize * CHANNELS).min(buf.len());
                buf.drain(..drop);
                buf.drain(..).collect()
            }
            StretcherImpl::PitchPreserving { input, .. } => {
                input.drain(..).map(f64::from).collect()
            }
        }
    }
}

#[inline]
fn catmull_rom(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;
    ((a0 * t + a1) * t + a2) * t + a3
}
