//! Beat-coordinate timing, octave-folded BPM matching, and beat-grid mapping.

use super::spec::{BeatGrid, TempoPlan};
use crate::SAMPLE_RATE;

/// Fold `bpm_b` by powers of 2 (`0.5x`, `1x`, `2x`) to minimize relative tempo
/// difference against `bpm_a` (handles half-time / double-time BPM estimates).
pub fn fold_bpm_octave(bpm_a: f64, bpm_b: f64) -> f64 {
    if bpm_a <= 0.0 || bpm_b <= 0.0 {
        return bpm_b;
    }
    let candidates = [bpm_b * 0.5, bpm_b, bpm_b * 2.0];
    candidates
        .into_iter()
        .min_by(|&x, &y| {
            let rx = (x / bpm_a).ln().abs();
            let ry = (y / bpm_a).ln().abs();
            rx.total_cmp(&ry)
        })
        .unwrap_or(bpm_b)
}

/// Shared beat-coordinate clock spanning `b in [0, total_beats]`.
///
/// Advances both decks along the same instantaneous tempo trajectory `bpm(b)`
/// ramping smoothly from `bpm_a` at `b = 0` to `effective_bpm_b` at `b = total_beats`,
/// so deck A starts at unity speed (`r_a(0) = 1.0`) and deck B exits at unity speed
/// (`r_b(B) = 1.0`) with zero cumulative phase drift between kicks.
#[derive(Debug, Clone)]
pub struct BeatClock {
    pub bpm_a: f64,
    pub bpm_b_effective: f64,
    pub total_beats: f64,
    pub beat_pos: f64,
    pub beatmatched: bool,
    pub start_a_sec: f64,
    pub start_b_sec: f64,
    pub beats_a: Option<BeatGrid>,
    pub beats_b: Option<BeatGrid>,
    start_a_beat_idx: f64,
    start_b_beat_idx: f64,
}

impl BeatClock {
    pub fn new(
        start_a_frame: u64,
        start_b_frame: u64,
        duration_frames: u64,
        duration_beats: Option<f64>,
        tempo: Option<&TempoPlan>,
    ) -> Self {
        let fs = f64::from(SAMPLE_RATE);
        let start_a_sec = start_a_frame as f64 / fs;
        let start_b_sec = start_b_frame as f64 / fs;

        let Some(plan) = tempo else {
            let dur_sec = (duration_frames as f64 / fs).max(1e-6);
            return Self {
                bpm_a: 60.0,
                bpm_b_effective: 60.0,
                total_beats: dur_sec,
                beat_pos: 0.0,
                beatmatched: false,
                start_a_sec,
                start_b_sec,
                beats_a: None,
                beats_b: None,
                start_a_beat_idx: start_a_sec,
                start_b_beat_idx: start_b_sec,
            };
        };

        let bpm_a = if plan.bpm_a > 20.0 { plan.bpm_a } else { 120.0 };
        let folded_b = fold_bpm_octave(bpm_a, if plan.bpm_b > 20.0 { plan.bpm_b } else { bpm_a });
        let ratio = (folded_b / bpm_a).max(bpm_a / folded_b);
        let can_beatmatch = plan.beatmatched && ratio <= plan.max_bpm_ratio.max(1.0);

        let effective_b = if can_beatmatch { folded_b } else { bpm_a };

        let total_beats = duration_beats.filter(|&b| b > 0.0).unwrap_or_else(|| {
            let dur_sec = (duration_frames as f64 / fs).max(1e-6);
            let avg_bpm = 0.5 * (bpm_a + effective_b);
            dur_sec * avg_bpm / 60.0
        });

        let start_a_beat_idx = plan
            .beats_a
            .as_ref()
            .and_then(|g| time_to_beat_index(g, start_a_sec))
            .unwrap_or(0.0);
        let start_b_beat_idx = plan
            .beats_b
            .as_ref()
            .and_then(|g| time_to_beat_index(g, start_b_sec))
            .unwrap_or(0.0);

        Self {
            bpm_a,
            bpm_b_effective: effective_b,
            total_beats: total_beats.max(1e-6),
            beat_pos: 0.0,
            beatmatched: can_beatmatch,
            start_a_sec,
            start_b_sec,
            beats_a: plan.beats_a.clone(),
            beats_b: plan.beats_b.clone(),
            start_a_beat_idx,
            start_b_beat_idx,
        }
    }

    /// Normalized transition progress `u in [0.0, 1.0]`.
    #[inline]
    pub fn progress(&self) -> f64 {
        (self.beat_pos / self.total_beats).clamp(0.0, 1.0)
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        self.beat_pos >= self.total_beats
    }

    /// Instantaneous target tempo (BPM) at normalized progress `u`.
    #[inline]
    pub fn target_bpm(&self, u: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        self.bpm_a + u * (self.bpm_b_effective - self.bpm_a)
    }

    /// Instantaneous local tempo (BPM) of track A at current beat offset `beat_pos`.
    pub fn local_bpm_a(&self) -> f64 {
        if let Some(ref grid) = self.beats_a {
            if let Some(bpm) = local_grid_bpm(grid, self.start_a_beat_idx + self.beat_pos) {
                return bpm;
            }
        }
        self.bpm_a
    }

    /// Instantaneous local tempo (BPM) of track B at current beat offset `beat_pos`,
    /// folded to match track A's octave.
    pub fn local_bpm_b(&self) -> f64 {
        if let Some(ref grid) = self.beats_b {
            if let Some(bpm) = local_grid_bpm(grid, self.start_b_beat_idx + self.beat_pos) {
                return fold_bpm_octave(self.bpm_a, bpm);
            }
        }
        self.bpm_b_effective
    }

    /// Instantaneous playback tempo ratio `source_speed / output_speed` for Deck A and Deck B.
    ///
    /// A ratio `> 1.0` means the source is played faster than nominal; `< 1.0` means slower.
    #[inline]
    pub fn deck_ratios(&self) -> (f64, f64) {
        if !self.beatmatched {
            return (1.0, 1.0);
        }
        let target = self.target_bpm(self.progress());
        let r_a = (target / self.local_bpm_a()).clamp(0.5, 2.0);
        let r_b = (target / self.local_bpm_b()).clamp(0.5, 2.0);
        (r_a, r_b)
    }

    /// Expected exact source frame index in Track A and Track B at the current beat position.
    pub fn expected_source_frames(&self) -> (f64, f64) {
        let fs = f64::from(SAMPLE_RATE);
        let t_a = self
            .beats_a
            .as_ref()
            .and_then(|g| beat_index_to_time(g, self.start_a_beat_idx + self.beat_pos))
            .unwrap_or_else(|| self.start_a_sec + self.beat_pos * 60.0 / self.bpm_a);
        let t_b = self
            .beats_b
            .as_ref()
            .and_then(|g| beat_index_to_time(g, self.start_b_beat_idx + self.beat_pos))
            .unwrap_or_else(|| self.start_b_sec + self.beat_pos * 60.0 / self.bpm_b_effective);
        (t_a * fs, t_b * fs)
    }

    /// Advance the shared beat clock by `frames` output samples using midpoint integration.
    #[inline]
    pub fn advance_frames(&mut self, frames: usize) {
        let dt_sec = frames as f64 / f64::from(SAMPLE_RATE);
        let bpm_0 = self.target_bpm(self.progress());
        let db_half = 0.5 * dt_sec * bpm_0 / 60.0;
        let u_mid = ((self.beat_pos + db_half) / self.total_beats).clamp(0.0, 1.0);
        let bpm_mid = self.target_bpm(u_mid);
        self.beat_pos += dt_sec * bpm_mid / 60.0;
    }

    /// Analytical wall-clock duration in output frames for a linear BPM ramp over `total_beats`:
    /// `D = integral_0^B (60 / (bpm_a + (bpm_b - bpm_a) * b / B)) db`.
    pub fn expected_output_frames(&self) -> u64 {
        let fs = f64::from(SAMPLE_RATE);
        let dur_sec = if (self.bpm_b_effective - self.bpm_a).abs() < 1e-6 {
            self.total_beats * 60.0 / self.bpm_a
        } else {
            60.0 * self.total_beats * (self.bpm_b_effective / self.bpm_a).ln()
                / (self.bpm_b_effective - self.bpm_a)
        };
        (dur_sec * fs).round().max(1.0) as u64
    }
}

fn time_to_beat_index(grid: &BeatGrid, time_sec: f64) -> Option<f64> {
    if grid.beats.len() < 2 {
        return None;
    }
    let idx = grid.beats.partition_point(|b| b.time_sec < time_sec);
    if idx == 0 {
        let b0 = grid.beats[0];
        let dt = b0.duration_sec.max(1e-4);
        return Some((time_sec - b0.time_sec) / dt);
    }
    if idx >= grid.beats.len() {
        let last = grid.beats.len() - 1;
        let bl = grid.beats[last];
        let dt = bl.duration_sec.max(1e-4);
        return Some(last as f64 + (time_sec - bl.time_sec) / dt);
    }
    let b0 = grid.beats[idx - 1];
    let b1 = grid.beats[idx];
    let span = (b1.time_sec - b0.time_sec).max(1e-6);
    Some((idx - 1) as f64 + (time_sec - b0.time_sec) / span)
}

fn beat_index_to_time(grid: &BeatGrid, beat_idx: f64) -> Option<f64> {
    if grid.beats.len() < 2 {
        return None;
    }
    if beat_idx <= 0.0 {
        let b0 = grid.beats[0];
        return Some(b0.time_sec + beat_idx * b0.duration_sec.max(1e-4));
    }
    let last = grid.beats.len() - 1;
    if beat_idx >= last as f64 {
        let bl = grid.beats[last];
        return Some(bl.time_sec + (beat_idx - last as f64) * bl.duration_sec.max(1e-4));
    }
    let i0 = beat_idx.floor() as usize;
    let frac = beat_idx - i0 as f64;
    let t0 = grid.beats[i0].time_sec;
    let t1 = grid.beats[i0 + 1].time_sec;
    Some(t0 + frac * (t1 - t0))
}

fn local_grid_bpm(grid: &BeatGrid, beat_idx: f64) -> Option<f64> {
    if grid.beats.len() < 2 {
        return None;
    }
    let i0 = (beat_idx.floor() as isize).clamp(0, (grid.beats.len() - 2) as isize) as usize;
    let dt = grid.beats[i0 + 1].time_sec - grid.beats[i0].time_sec;
    if dt > 0.1 && dt < 3.0 {
        Some(60.0 / dt)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octave_folding_handles_half_and_double_time() {
        assert!((fold_bpm_octave(145.0, 74.0) - 148.0).abs() < 1e-9);
        assert!((fold_bpm_octave(75.0, 152.0) - 76.0).abs() < 1e-9);
        assert!((fold_bpm_octave(145.0, 150.0) - 150.0).abs() < 1e-9);
    }

    #[test]
    fn beat_clock_boundary_ratios_and_analytical_duration() {
        let plan = TempoPlan::beatmatched(145.0, 150.0);
        let mut clock = BeatClock::new(0, 0, 0, Some(32.0), Some(&plan));

        let (ra0, rb0) = clock.deck_ratios();
        assert!((ra0 - 1.0).abs() < 1e-9);
        assert!((rb0 - 145.0 / 150.0).abs() < 1e-9);

        let expected_frames = clock.expected_output_frames();
        let mut total_frames = 0u64;
        while !clock.is_complete() {
            clock.advance_frames(64);
            total_frames += 64;
        }
        // Within one block of analytical integral duration
        assert!(total_frames.abs_diff(expected_frames) <= 128);

        let (ra1, rb1) = clock.deck_ratios();
        assert!((ra1 - 150.0 / 145.0).abs() < 1e-6);
        assert!((rb1 - 1.0).abs() < 1e-6);
    }
}
