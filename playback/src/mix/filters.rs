//! Unconditionally stable `f64` stereo filters for DJ transitions.
//!
//! References:
//! - Andrew Simper (Cytomic, 2013), *Linear Trapezoidal Integrated State Variable Filter*.
//! - Vadim Zavalishin (2018), *The Art of VA Filter Design*.
//! - Siegfried Linkwitz (1976), *Active Crossover Networks for Noncoincident Drivers*, JAES 24(1).

use std::f64::consts::{PI, SQRT_2};

use super::spec::{EqTopology, FilterMode};
use crate::SAMPLE_RATE;

/// 2-pole stereo Cytomic Topology-Preserving Transform (TPT) State Variable Filter.
///
/// Safe under per-sample cutoff and resonance modulation without coefficient-jump pops.
#[derive(Debug, Clone)]
pub struct TptSvf2 {
    ic1: [f64; 2],
    ic2: [f64; 2],
    g: f64,
    k: f64,
    a1: f64,
    a2: f64,
    a3: f64,
}

impl TptSvf2 {
    pub fn new(cutoff_hz: f64, q: f64) -> Self {
        let mut s = Self {
            ic1: [0.0; 2],
            ic2: [0.0; 2],
            g: 0.0,
            k: SQRT_2,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
        };
        s.set_params(cutoff_hz, q);
        s
    }

    pub fn reset(&mut self) {
        self.ic1 = [0.0; 2];
        self.ic2 = [0.0; 2];
    }

    #[inline]
    pub fn set_params(&mut self, cutoff_hz: f64, q: f64) {
        let fs = f64::from(SAMPLE_RATE);
        let fc = if cutoff_hz.is_finite() {
            cutoff_hz.clamp(15.0, 0.45 * fs)
        } else {
            1000.0
        };
        let q = if q.is_finite() {
            q.clamp(0.5, 12.0)
        } else {
            std::f64::consts::FRAC_1_SQRT_2
        };
        self.g = (PI * fc / fs).tan();
        self.k = 1.0 / q;
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        self.a2 = self.g * self.a1;
        self.a3 = self.g * self.a2;
    }

    #[inline]
    fn tick_raw(&mut self, ch: usize, v0: f64) -> (f64, f64, f64) {
        let v0 = if v0.is_finite() { v0 } else { 0.0 };
        let v3 = v0 - self.ic2[ch];
        let v1 = self.a1 * self.ic1[ch] + self.a2 * v3;
        let v2 = self.ic2[ch] + self.a2 * self.ic1[ch] + self.a3 * v3;
        self.ic1[ch] = 2.0 * v1 - self.ic1[ch];
        self.ic2[ch] = 2.0 * v2 - self.ic2[ch];
        (v0, v1, v2)
    }

    #[inline]
    pub fn process_lowpass(&mut self, frame: [f64; 2]) -> [f64; 2] {
        let (_, _, l) = self.tick_raw(0, frame[0]);
        let (_, _, r) = self.tick_raw(1, frame[1]);
        [l, r]
    }

    #[inline]
    pub fn process_highpass(&mut self, frame: [f64; 2]) -> [f64; 2] {
        let (v0_l, v1_l, v2_l) = self.tick_raw(0, frame[0]);
        let (v0_r, v1_r, v2_r) = self.tick_raw(1, frame[1]);
        [v0_l - self.k * v1_l - v2_l, v0_r - self.k * v1_r - v2_r]
    }

    /// 2nd-order all-pass `H_AP(s) = (s^2 - k s + 1) / (s^2 + k s + 1)`.
    #[inline]
    pub fn process_allpass(&mut self, frame: [f64; 2]) -> [f64; 2] {
        let (v0_l, v1_l, _) = self.tick_raw(0, frame[0]);
        let (v0_r, v1_r, _) = self.tick_raw(1, frame[1]);
        [v0_l - 2.0 * self.k * v1_l, v0_r - 2.0 * self.k * v1_r]
    }
}

/// Phase-matched 4th-order Linkwitz-Riley (`LR4`) 3-band stereo DJ isolator.
///
/// Splits input into Low (`< low_hz`), Mid (`low_hz..high_hz`), and High (`> high_hz`).
/// Because `LP4(f) + HP4(f) = AP2(f)`, passing the low band through a matching 2nd-order
/// Butterworth all-pass at `high_hz` yields an exact all-pass unity magnitude sum
/// (`|Low + Mid + High| = 1.0`) across DC..Nyquist when all band gains equal `1.0`,
/// while supporting full band kill (`gain = 0.0`) with zero coefficient modulation.
#[derive(Debug, Clone)]
pub struct LinkwitzRiley3Band {
    low_lp1: TptSvf2,
    low_lp2: TptSvf2,
    low_ap_high: TptSvf2,
    mh_hp1: TptSvf2,
    mh_hp2: TptSvf2,
    mid_lp1: TptSvf2,
    mid_lp2: TptSvf2,
    high_hp1: TptSvf2,
    high_hp2: TptSvf2,
    gains: [f64; 3],
    smooth_coeff: f64,
}

impl LinkwitzRiley3Band {
    pub fn new(low_hz: f64, high_hz: f64) -> Self {
        let butterworth_q = std::f64::consts::FRAC_1_SQRT_2;
        // 3 ms 1-pole gain smoother eliminates zipper noise on abrupt curves.
        let smooth_coeff = (-1.0 / (0.003 * f64::from(SAMPLE_RATE))).exp();
        Self {
            low_lp1: TptSvf2::new(low_hz, butterworth_q),
            low_lp2: TptSvf2::new(low_hz, butterworth_q),
            low_ap_high: TptSvf2::new(high_hz, butterworth_q),
            mh_hp1: TptSvf2::new(low_hz, butterworth_q),
            mh_hp2: TptSvf2::new(low_hz, butterworth_q),
            mid_lp1: TptSvf2::new(high_hz, butterworth_q),
            mid_lp2: TptSvf2::new(high_hz, butterworth_q),
            high_hp1: TptSvf2::new(high_hz, butterworth_q),
            high_hp2: TptSvf2::new(high_hz, butterworth_q),
            gains: [1.0, 1.0, 1.0],
            smooth_coeff,
        }
    }

    pub fn reset(&mut self) {
        self.low_lp1.reset();
        self.low_lp2.reset();
        self.low_ap_high.reset();
        self.mh_hp1.reset();
        self.mh_hp2.reset();
        self.mid_lp1.reset();
        self.mid_lp2.reset();
        self.high_hp1.reset();
        self.high_hp2.reset();
        self.gains = [1.0, 1.0, 1.0];
    }

    #[inline]
    pub fn process(&mut self, frame: [f64; 2], target_gains: [f64; 3]) -> [f64; 2] {
        let frame = [
            if frame[0].is_finite() { frame[0] } else { 0.0 },
            if frame[1].is_finite() { frame[1] } else { 0.0 },
        ];
        for (g, &target) in self.gains.iter_mut().zip(target_gains.iter()) {
            let clamped = if target.is_finite() {
                target.clamp(0.0, 4.0)
            } else {
                1.0
            };
            *g = self.smooth_coeff * *g + (1.0 - self.smooth_coeff) * clamped;
        }

        let low = self.low_ap_high.process_allpass(
            self.low_lp2
                .process_lowpass(self.low_lp1.process_lowpass(frame)),
        );
        let midhigh = self
            .mh_hp2
            .process_highpass(self.mh_hp1.process_highpass(frame));
        let mid = self
            .mid_lp2
            .process_lowpass(self.mid_lp1.process_lowpass(midhigh));
        let high = self
            .high_hp2
            .process_highpass(self.high_hp1.process_highpass(midhigh));

        [
            low[0] * self.gains[0] + mid[0] * self.gains[1] + high[0] * self.gains[2],
            low[1] * self.gains[0] + mid[1] * self.gains[1] + high[1] * self.gains[2],
        ]
    }
}

/// Cytomic 2-pole stereo TPT shelf/bell biquad. Reduces to exact `H(z) = 1` at 0 dB.
#[derive(Debug, Clone)]
struct TptShelfBell {
    ic1: [f64; 2],
    ic2: [f64; 2],
    base_g: f64,
    q: f64,
}

impl TptShelfBell {
    fn new(fc: f64, q: f64) -> Self {
        let fs = f64::from(SAMPLE_RATE);
        let fc = if fc.is_finite() {
            fc.clamp(15.0, 0.45 * fs)
        } else {
            1000.0
        };
        let q = if q.is_finite() {
            q.clamp(0.3, 12.0)
        } else {
            std::f64::consts::FRAC_1_SQRT_2
        };
        Self {
            ic1: [0.0; 2],
            ic2: [0.0; 2],
            base_g: (PI * fc / fs).tan(),
            q,
        }
    }

    fn reset(&mut self) {
        self.ic1 = [0.0; 2];
        self.ic2 = [0.0; 2];
    }

    #[inline]
    fn tick(&mut self, ch: usize, v0: f64, g: f64, k: f64, m: [f64; 3]) -> f64 {
        let v0 = if v0.is_finite() { v0 } else { 0.0 };
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = v0 - self.ic2[ch];
        let v1 = a1 * self.ic1[ch] + a2 * v3;
        let v2 = self.ic2[ch] + a2 * self.ic1[ch] + a3 * v3;
        self.ic1[ch] = 2.0 * v1 - self.ic1[ch];
        self.ic2[ch] = 2.0 * v2 - self.ic2[ch];
        m[0] * v0 + m[1] * v1 + m[2] * v2
    }

    #[inline]
    fn process_low_shelf(&mut self, frame: [f64; 2], gain_db: f64) -> [f64; 2] {
        let db = if gain_db.is_finite() {
            gain_db.clamp(-48.0, 12.0)
        } else {
            0.0
        };
        let a = 10.0_f64.powf(db / 40.0);
        let g = self.base_g / a.sqrt();
        let k = 1.0 / self.q;
        let m = [1.0, k * (a - 1.0), a * a - 1.0];
        [
            self.tick(0, frame[0], g, k, m),
            self.tick(1, frame[1], g, k, m),
        ]
    }

    #[inline]
    fn process_bell(&mut self, frame: [f64; 2], gain_db: f64) -> [f64; 2] {
        let db = if gain_db.is_finite() {
            gain_db.clamp(-48.0, 12.0)
        } else {
            0.0
        };
        let a = 10.0_f64.powf(db / 40.0);
        let g = self.base_g;
        let k = 1.0 / (self.q * a);
        let m = [1.0, k * (a * a - 1.0), 0.0];
        [
            self.tick(0, frame[0], g, k, m),
            self.tick(1, frame[1], g, k, m),
        ]
    }

    #[inline]
    fn process_high_shelf(&mut self, frame: [f64; 2], gain_db: f64) -> [f64; 2] {
        let db = if gain_db.is_finite() {
            gain_db.clamp(-48.0, 12.0)
        } else {
            0.0
        };
        let a = 10.0_f64.powf(db / 40.0);
        let g = self.base_g * a.sqrt();
        let k = 1.0 / self.q;
        let m = [a * a, k * (1.0 - a) * a, 1.0 - a * a];
        [
            self.tick(0, frame[0], g, k, m),
            self.tick(1, frame[1], g, k, m),
        ]
    }
}

/// 3-band stereo TPT shelving/bell EQ (gains specified in linear ratio, converted to dB).
#[derive(Debug, Clone)]
pub struct TptShelving3Band {
    low: TptShelfBell,
    mid: TptShelfBell,
    high: TptShelfBell,
    gains_db: [f64; 3],
    smooth_coeff: f64,
}

impl TptShelving3Band {
    pub fn new(low_hz: f64, mid_hz: f64, high_hz: f64) -> Self {
        let q = std::f64::consts::FRAC_1_SQRT_2;
        let smooth_coeff = (-1.0 / (0.003 * f64::from(SAMPLE_RATE))).exp();
        Self {
            low: TptShelfBell::new(low_hz, q),
            mid: TptShelfBell::new(mid_hz, q),
            high: TptShelfBell::new(high_hz, q),
            gains_db: [0.0; 3],
            smooth_coeff,
        }
    }

    pub fn reset(&mut self) {
        self.low.reset();
        self.mid.reset();
        self.high.reset();
        self.gains_db = [0.0; 3];
    }

    #[inline]
    pub fn process_linear_gains(&mut self, frame: [f64; 2], target_linear: [f64; 3]) -> [f64; 2] {
        let frame = [
            if frame[0].is_finite() { frame[0] } else { 0.0 },
            if frame[1].is_finite() { frame[1] } else { 0.0 },
        ];
        for (g_db, &lin) in self.gains_db.iter_mut().zip(target_linear.iter()) {
            let target_db = if !lin.is_finite() {
                0.0
            } else if lin <= 1e-4 {
                -48.0
            } else {
                (20.0 * lin.log10()).clamp(-48.0, 12.0)
            };
            *g_db = self.smooth_coeff * *g_db + (1.0 - self.smooth_coeff) * target_db;
        }
        let f = self.low.process_low_shelf(frame, self.gains_db[0]);
        let f = self.mid.process_bell(f, self.gains_db[1]);
        self.high.process_high_shelf(f, self.gains_db[2])
    }
}

/// Unified 3-band deck equalizer supporting either `LR4` isolator or `TPT` shelving.
#[derive(Debug, Clone)]
pub enum DeckEq {
    Isolator(Box<LinkwitzRiley3Band>),
    Shelving(Box<TptShelving3Band>),
}

impl DeckEq {
    pub fn new(topology: EqTopology) -> Self {
        match topology {
            EqTopology::LinkwitzRileyIsolator { low_hz, high_hz } => {
                Self::Isolator(Box::new(LinkwitzRiley3Band::new(low_hz, high_hz)))
            }
            EqTopology::TptShelving {
                low_hz,
                mid_hz,
                high_hz,
            } => Self::Shelving(Box::new(TptShelving3Band::new(low_hz, mid_hz, high_hz))),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Isolator(eq) => eq.reset(),
            Self::Shelving(eq) => eq.reset(),
        }
    }

    #[inline]
    pub fn process(&mut self, frame: [f64; 2], target_linear_gains: [f64; 3]) -> [f64; 2] {
        match self {
            Self::Isolator(eq) => eq.process(frame, target_linear_gains),
            Self::Shelving(eq) => eq.process_linear_gains(frame, target_linear_gains),
        }
    }
}

/// Modulated stereo low-pass / high-pass filter with per-sample log-cutoff smoothing
/// and click-free bypass crossfading.
#[derive(Debug, Clone)]
pub struct DeckFilter {
    svf: TptSvf2,
    mode: FilterMode,
    log_cutoff: f64,
    q: f64,
    wet: f64,
    cutoff_smooth: f64,
    wet_smooth: f64,
}

impl DeckFilter {
    pub fn new() -> Self {
        let fs = f64::from(SAMPLE_RATE);
        Self {
            svf: TptSvf2::new(1000.0, std::f64::consts::FRAC_1_SQRT_2),
            mode: FilterMode::Bypass,
            log_cutoff: 1000.0_f64.ln(),
            q: std::f64::consts::FRAC_1_SQRT_2,
            wet: 0.0,
            cutoff_smooth: (-1.0 / (0.005 * fs)).exp(),
            wet_smooth: (-1.0 / (0.010 * fs)).exp(),
        }
    }

    pub fn reset(&mut self) {
        self.svf.reset();
        self.mode = FilterMode::Bypass;
        self.wet = 0.0;
    }

    #[inline]
    pub fn process(
        &mut self,
        frame: [f64; 2],
        target_mode: FilterMode,
        target_cutoff_hz: f64,
        target_q: f64,
    ) -> [f64; 2] {
        let frame = [
            if frame[0].is_finite() { frame[0] } else { 0.0 },
            if frame[1].is_finite() { frame[1] } else { 0.0 },
        ];
        let fs = f64::from(SAMPLE_RATE);
        let fc = if target_cutoff_hz.is_finite() {
            target_cutoff_hz.clamp(15.0, 0.45 * fs)
        } else {
            self.log_cutoff.exp()
        };
        let q = if target_q.is_finite() {
            target_q.clamp(0.5, 10.0)
        } else {
            self.q
        };

        // Never flip instantaneously between LowPass and HighPass while wet; fade through
        // dry (`wet -> 0`), reset the SVF integrators, snap the new mode/cutoff, then fade in.
        let switching_mode = target_mode != FilterMode::Bypass
            && self.mode != FilterMode::Bypass
            && target_mode != self.mode;

        let target_wet = if target_mode == FilterMode::Bypass || (switching_mode && self.wet > 1e-3)
        {
            0.0
        } else {
            if self.mode == FilterMode::Bypass || self.wet <= 1e-3 {
                if self.mode != target_mode {
                    self.svf.reset();
                }
                self.mode = target_mode;
                self.log_cutoff = fc.ln();
                self.q = q;
            }
            1.0
        };

        self.wet = self.wet_smooth * self.wet + (1.0 - self.wet_smooth) * target_wet;
        if self.wet < 1e-6 && target_mode == FilterMode::Bypass {
            self.wet = 0.0;
            self.mode = FilterMode::Bypass;
            self.svf.reset();
            return frame;
        }

        // Freeze cutoff and resonance while fading out to Bypass or across a mode switch so
        // dummy bypass cutoffs never sweep back through the audible midband while still wet.
        if target_wet > 0.0 {
            self.log_cutoff =
                self.cutoff_smooth * self.log_cutoff + (1.0 - self.cutoff_smooth) * fc.ln();
            self.q = self.cutoff_smooth * self.q + (1.0 - self.cutoff_smooth) * q;
        }
        self.svf.set_params(self.log_cutoff.exp(), self.q);

        let filtered = match self.mode {
            FilterMode::LowPass => self.svf.process_lowpass(frame),
            FilterMode::HighPass => self.svf.process_highpass(frame),
            FilterMode::Bypass => frame,
        };

        let w = self.wet;
        let dry = 1.0 - w;
        [
            dry * frame[0] + w * filtered[0],
            dry * frame[1] + w * filtered[1],
        ]
    }
}

impl Default for DeckFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute magnitude response `|H(f)|` via DFT of an impulse response.
    fn dft_mag_db(ir: &[f64], freq_hz: f64) -> f64 {
        let w = 2.0 * PI * freq_hz / f64::from(SAMPLE_RATE);
        let mut re = 0.0;
        let mut im = 0.0;
        for (n, &x) in ir.iter().enumerate() {
            let angle = w * n as f64;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        let mag = (re * re + im * im).sqrt().max(1e-12);
        20.0 * mag.log10()
    }

    #[test]
    fn lr4_3band_isolator_has_flat_unity_sum_and_full_bass_kill() {
        let n = 8192;
        let mut eq = LinkwitzRiley3Band::new(250.0, 2500.0);
        let mut ir_unity = vec![0.0; n];
        for (i, slot) in ir_unity.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            *slot = eq.process([x, x], [1.0, 1.0, 1.0])[0];
        }

        // All-pass unity sum must hold within +/- 0.05 dB across DC..Nyquist, including crossovers.
        for &f in &[30.0, 100.0, 250.0, 700.0, 2500.0, 8000.0, 16000.0] {
            let db = dft_mag_db(&ir_unity, f);
            assert!(
                db.abs() < 0.05,
                "LR4 unity sum deviated at {f} Hz: {db:.4} dB"
            );
        }

        // With low band killed (0.0), settle smoother then measure impulse response.
        let mut eq_kill = LinkwitzRiley3Band::new(250.0, 2500.0);
        for _ in 0..4096 {
            eq_kill.process([0.0, 0.0], [0.0, 1.0, 1.0]);
        }
        let mut ir_kill = vec![0.0; n];
        for (i, slot) in ir_kill.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            *slot = eq_kill.process([x, x], [0.0, 1.0, 1.0])[0];
        }
        let sub_db = dft_mag_db(&ir_kill, 40.0);
        let high_db = dft_mag_db(&ir_kill, 8000.0);
        assert!(
            sub_db < -30.0,
            "expected deep bass kill at 40 Hz, got {sub_db:.2} dB"
        );
        assert!(
            high_db.abs() < 0.2,
            "high band disturbed by bass kill: {high_db:.2} dB"
        );
    }

    #[test]
    fn tpt_shelving_is_exact_identity_at_unity() {
        let mut eq = TptShelving3Band::new(200.0, 1000.0, 4000.0);
        for i in 0..2048 {
            let t = i as f64 / f64::from(SAMPLE_RATE);
            let s = (2.0 * PI * 440.0 * t).sin();
            let out = eq.process_linear_gains([s, -s], [1.0, 1.0, 1.0]);
            assert!((out[0] - s).abs() < 1e-12);
            assert!((out[1] + s).abs() < 1e-12);
        }
    }

    #[test]
    fn deck_filter_remains_finite_and_stable_under_rapid_sweep() {
        let mut filt = DeckFilter::new();
        let n = SAMPLE_RATE as usize; // 1 second sweep 20 Hz -> 18 kHz -> 20 Hz at high Q
        for i in 0..n {
            let u = i as f64 / n as f64;
            let fc = if u < 0.5 {
                20.0 * (900.0_f64).powf(u * 2.0)
            } else {
                18_000.0 * (1.0 / 900.0_f64).powf((u - 0.5) * 2.0)
            };
            let mode = if u < 0.5 {
                FilterMode::LowPass
            } else {
                FilterMode::HighPass
            };
            let x = (2.0 * PI * 220.0 * (i as f64 / f64::from(SAMPLE_RATE))).sin();
            let y = filt.process([x, x], mode, fc, 6.0);
            assert!(y[0].is_finite() && y[1].is_finite());
            assert!(y[0].abs() < 20.0 && y[1].abs() < 20.0);
        }
    }

    /// Measure steady-state gain in dB of an arbitrary stereo filter at `freq_hz` after settling.
    fn steady_state_gain_db<F>(freq_hz: f64, mut process_sample: F) -> f64
    where
        F: FnMut([f64; 2]) -> [f64; 2],
    {
        let fs = f64::from(SAMPLE_RATE);
        let settle = 4096usize;
        let measure = 4096usize;
        let mut sum_in2 = 0.0_f64;
        let mut sum_out2 = 0.0_f64;
        for i in 0..(settle + measure) {
            let x = (2.0 * PI * freq_hz * i as f64 / fs).sin();
            let y = process_sample([x, x]);
            if i >= settle {
                sum_in2 += x * x;
                sum_out2 += y[0] * y[0];
            }
        }
        let ratio = (sum_out2 / sum_in2.max(1e-18)).sqrt().max(1e-12);
        20.0 * ratio.log10()
    }

    #[test]
    fn lr4_3band_isolator_analytical_crossovers_slopes_and_per_band_accuracy() {
        let n = 16_384;
        // 1. All-pass unity sum across 15 log-spaced frequencies (25 Hz..18 kHz).
        let mut eq = LinkwitzRiley3Band::new(250.0, 2500.0);
        let mut ir_unity = vec![0.0; n];
        for (i, slot) in ir_unity.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            *slot = eq.process([x, x], [1.0, 1.0, 1.0])[0];
        }
        for &f in &[
            25.0, 50.0, 100.0, 180.0, 250.0, 400.0, 700.0, 1200.0, 2000.0, 2500.0, 4000.0, 7000.0,
            10_000.0, 14_000.0, 18_000.0,
        ] {
            let db = dft_mag_db(&ir_unity, f);
            assert!(db.abs() < 0.05, "LR4 unity deviated at {f} Hz: {db:.4} dB");
        }

        // 2. Solo each band and verify exact -6.02 dB crossover points and 24 dB/oct rolloff.
        let solo_ir = |gains: [f64; 3]| -> Vec<f64> {
            let mut e = LinkwitzRiley3Band::new(250.0, 2500.0);
            for _ in 0..4096 {
                e.process([0.0, 0.0], gains);
            }
            let mut ir = vec![0.0; n];
            for (i, slot) in ir.iter_mut().enumerate() {
                let x = if i == 0 { 1.0 } else { 0.0 };
                *slot = e.process([x, x], gains)[0];
            }
            ir
        };
        let ir_low = solo_ir([1.0, 0.0, 0.0]);
        let ir_mid = solo_ir([0.0, 1.0, 0.0]);
        let ir_high = solo_ir([0.0, 0.0, 1.0]);

        // LR4 crossover points are exactly -6.02 dB.
        assert!((dft_mag_db(&ir_low, 250.0) - (-6.02)).abs() < 0.2);
        assert!((dft_mag_db(&ir_mid, 250.0) - (-6.02)).abs() < 0.2);
        assert!((dft_mag_db(&ir_mid, 2500.0) - (-6.02)).abs() < 0.2);
        assert!((dft_mag_db(&ir_high, 2500.0) - (-6.02)).abs() < 0.2);

        // 24 dB/oct stopband slopes (2 octaves past crossover <= -44 dB).
        assert!(dft_mag_db(&ir_low, 1000.0) < -44.0);
        assert!(dft_mag_db(&ir_mid, 62.5) < -44.0);
        assert!(dft_mag_db(&ir_mid, 10_000.0) < -44.0);
        assert!(dft_mag_db(&ir_high, 625.0) < -44.0);

        // 3. Per-band boost/cut accuracy against the exact analytical LR4 sum
        //    |g_low * H_lp(f) + g_mid * H_hp(f)*H_lp2(f) + g_high * H_hp(f)*H_hp2(f)|.
        for &(band_idx, center_hz, target_db) in &[
            (0usize, 60.0_f64, 6.0_f64),
            (0, 60.0, -12.0),
            (1, 800.0, 6.0),
            (1, 800.0, -12.0),
            (2, 10_000.0, 6.0),
            (2, 10_000.0, -12.0),
        ] {
            let mut gains = [1.0, 1.0, 1.0];
            gains[band_idx] = 10.0_f64.powf(target_db / 20.0);
            let h_lp1 = 1.0 / (1.0 + (center_hz / 250.0).powi(4));
            let h_hp1 = 1.0 - h_lp1;
            let h_lp2 = 1.0 / (1.0 + (center_hz / 2500.0).powi(4));
            let h_hp2 = 1.0 - h_lp2;
            let expected_mag =
                gains[0] * h_lp1 + gains[1] * h_hp1 * h_lp2 + gains[2] * h_hp1 * h_hp2;
            let expected_db = 20.0 * expected_mag.log10();

            let mut e = LinkwitzRiley3Band::new(250.0, 2500.0);
            let measured = steady_state_gain_db(center_hz, |f| e.process(f, gains));
            assert!(
                (measured - expected_db).abs() < 0.2,
                "LR4 band {band_idx} at {center_hz} Hz expected {expected_db:.2} dB got {measured:.2} dB"
            );
        }
    }

    #[test]
    fn tpt_shelving_analytical_gains_and_boost_cut_complementarity() {
        // 1. Exact steady-state gain accuracy for low shelf (30 Hz), mid bell (1 kHz), high shelf (14 kHz).
        for &(band_idx, probe_hz, target_db) in &[
            (0, 30.0, 6.0_f64),
            (0, 30.0, -12.0),
            (0, 30.0, -24.0),
            (1, 1000.0, 6.0),
            (1, 1000.0, -12.0),
            (2, 14_000.0, 6.0),
            (2, 14_000.0, -12.0),
        ] {
            let mut gains = [1.0, 1.0, 1.0];
            gains[band_idx] = 10.0_f64.powf(target_db / 20.0);
            let mut eq = TptShelving3Band::new(200.0, 1000.0, 4000.0);
            let measured = steady_state_gain_db(probe_hz, |f| eq.process_linear_gains(f, gains));
            assert!(
                (measured - target_db).abs() < 0.6,
                "TPT shelf band {band_idx} at {probe_hz} Hz target {target_db} dB got {measured:.2} dB"
            );
        }

        // 2. Complementarity law: cascading +G dB followed by -G dB restores 0 dB across the spectrum.
        let boost = [2.0, 2.0, 2.0]; // +6.02 dB
        let cut = [0.5, 0.5, 0.5]; // -6.02 dB
        for &f in &[40.0, 150.0, 400.0, 1000.0, 3000.0, 8000.0, 14_000.0] {
            let mut eq_a = TptShelving3Band::new(200.0, 1000.0, 4000.0);
            let mut eq_b = TptShelving3Band::new(200.0, 1000.0, 4000.0);
            let db = steady_state_gain_db(f, |x| {
                let y = eq_a.process_linear_gains(x, boost);
                eq_b.process_linear_gains(y, cut)
            });
            assert!(
                db.abs() < 0.15,
                "TPT shelving +6/-6 complementarity failed at {f} Hz: {db:.4} dB"
            );
        }
    }

    #[test]
    fn deck_filter_analytical_lp_hp_resonance_bypass_freeze_and_mode_flip() {
        let butterworth_q = std::f64::consts::FRAC_1_SQRT_2;

        // 1. Settled 2-pole Butterworth LowPass at 1 kHz: -3.01 dB at fc, 12 dB/oct stopband.
        let mut lp = DeckFilter::new();
        let lp_pass = steady_state_gain_db(100.0, |x| {
            lp.process(x, FilterMode::LowPass, 1000.0, butterworth_q)
        });
        let lp_fc = steady_state_gain_db(1000.0, |x| {
            lp.process(x, FilterMode::LowPass, 1000.0, butterworth_q)
        });
        let lp_2x = steady_state_gain_db(2000.0, |x| {
            lp.process(x, FilterMode::LowPass, 1000.0, butterworth_q)
        });
        let lp_4x = steady_state_gain_db(4000.0, |x| {
            lp.process(x, FilterMode::LowPass, 1000.0, butterworth_q)
        });
        assert!(lp_pass.abs() < 0.15, "LP passband={lp_pass:.2} dB");
        assert!((lp_fc - (-3.01)).abs() < 0.2, "LP fc={lp_fc:.2} dB");
        assert!((lp_2x - (-12.3)).abs() < 0.5, "LP 2*fc={lp_2x:.2} dB");
        assert!((lp_4x - (-24.1)).abs() < 0.8, "LP 4*fc={lp_4x:.2} dB");

        // 2. Settled 2-pole Butterworth HighPass at 1 kHz: symmetric response.
        let mut hp = DeckFilter::new();
        let hp_pass = steady_state_gain_db(8000.0, |x| {
            hp.process(x, FilterMode::HighPass, 1000.0, butterworth_q)
        });
        let hp_fc = steady_state_gain_db(1000.0, |x| {
            hp.process(x, FilterMode::HighPass, 1000.0, butterworth_q)
        });
        let hp_half = steady_state_gain_db(500.0, |x| {
            hp.process(x, FilterMode::HighPass, 1000.0, butterworth_q)
        });
        assert!(hp_pass.abs() < 0.15, "HP passband={hp_pass:.2} dB");
        assert!((hp_fc - (-3.01)).abs() < 0.2, "HP fc={hp_fc:.2} dB");
        assert!((hp_half - (-12.3)).abs() < 0.5, "HP fc/2={hp_half:.2} dB");

        // 3. Exact resonant peak law: at Q = 4.0, |H_LP(fc)| ~= +12.04 dB.
        let mut res = DeckFilter::new();
        let peak_db =
            steady_state_gain_db(1000.0, |x| res.process(x, FilterMode::LowPass, 1000.0, 4.0));
        assert!(
            (peak_db - 20.0 * 4.0_f64.log10()).abs() < 0.4,
            "resonant peak at Q=4 got {peak_db:.2} dB"
        );

        // 4. Bypass cutoff-freeze regression: engaging LP wide open at 18 kHz and then switching
        // to Bypass with dummy cutoff 1000 Hz must NOT pull 6 kHz down during the 10 ms fade-out.
        let fs = f64::from(SAMPLE_RATE);
        let mut bf = DeckFilter::new();
        for i in 0..4096 {
            let s = (2.0 * PI * 6000.0 * i as f64 / fs).sin();
            bf.process([s, s], FilterMode::LowPass, 18_000.0, butterworth_q);
        }
        let mut min_env = 1.0_f64;
        let mut win_sum = 0.0_f64;
        for i in 0..2048 {
            let s = (2.0 * PI * 6000.0 * (4096 + i) as f64 / fs).sin();
            let y = bf.process([s, s], FilterMode::Bypass, 1000.0, butterworth_q);
            win_sum += y[0] * y[0];
            if (i + 1) % 128 == 0 {
                let rms = (win_sum / 128.0).sqrt();
                min_env = min_env.min(rms / std::f64::consts::FRAC_1_SQRT_2);
                win_sum = 0.0;
            }
        }
        assert!(
            min_env > 0.85,
            "bypass fade pulled 18 kHz LP down toward dummy 1 kHz cutoff (min_env={min_env:.3})"
        );

        // 5. Abrupt LowPass -> HighPass mode flip must crossfade through dry without a step click.
        let mut mf = DeckFilter::new();
        let mut prev = 0.0_f64;
        let mut max_step = 0.0_f64;
        for i in 0..6000 {
            let t = i as f64 / fs;
            let s = 0.5 * (2.0 * PI * 120.0 * t).sin() + 0.3 * (2.0 * PI * 2000.0 * t).sin();
            let mode = if i < 2000 {
                FilterMode::LowPass
            } else {
                FilterMode::HighPass
            };
            let y = mf.process([s, s], mode, 500.0, butterworth_q);
            if i > 100 {
                max_step = max_step.max((y[0] - prev).abs());
            }
            prev = y[0];
        }
        assert!(
            max_step < 0.12,
            "abrupt LP->HP mode flip produced step click: max_step={max_step:.4}"
        );
    }

    #[test]
    fn all_filters_stereo_independence_and_adversarial_edge_cases() {
        let fs = f64::from(SAMPLE_RATE);

        // 1. Stereo channel independence: signal on Left only must leak exact 0.0 to Right.
        let mut lr4 = LinkwitzRiley3Band::new(250.0, 2500.0);
        let mut tpt = TptShelving3Band::new(200.0, 1000.0, 4000.0);
        let mut flt = DeckFilter::new();
        for i in 0..1024 {
            let s = (2.0 * PI * 330.0 * i as f64 / fs).sin();
            let y1 = lr4.process([s, 0.0], [0.2, 1.5, 0.7]);
            let y2 = tpt.process_linear_gains([s, 0.0], [0.2, 1.5, 0.7]);
            let y3 = flt.process([s, 0.0], FilterMode::LowPass, 800.0, 2.5);
            assert_eq!(y1[1], 0.0, "LR4 leaked L->R");
            assert_eq!(y2[1], 0.0, "TPT shelving leaked L->R");
            assert_eq!(y3[1], 0.0, "DeckFilter leaked L->R");
        }

        // 2. Adversarial NaN/Inf/subnormal/extreme parameters must never poison filter state.
        let mut lr4 = LinkwitzRiley3Band::new(250.0, 2500.0);
        let mut tpt = TptShelving3Band::new(-50.0, 1e9, f64::NAN);
        let mut flt = DeckFilter::new();
        let nasty_frames = [
            [f64::NAN, f64::INFINITY],
            [f64::NEG_INFINITY, 1e-300],
            [-1e-300, 10.0],
            [0.0, -10.0],
        ];
        for &frame in &nasty_frames {
            let y1 = lr4.process(frame, [f64::NAN, f64::INFINITY, -5.0]);
            let y2 = tpt.process_linear_gains(frame, [f64::NAN, 0.0, f64::INFINITY]);
            let y3 = flt.process(frame, FilterMode::HighPass, f64::NAN, f64::INFINITY);
            assert!(y1[0].is_finite() && y1[1].is_finite());
            assert!(y2[0].is_finite() && y2[1].is_finite());
            assert!(y3[0].is_finite() && y3[1].is_finite());
        }
        // Verify clean recovery on normal sine wave afterwards.
        for i in 0..512 {
            let s = 0.5 * (2.0 * PI * 440.0 * i as f64 / fs).sin();
            let y1 = lr4.process([s, s], [1.0, 1.0, 1.0]);
            let y2 = tpt.process_linear_gains([s, s], [1.0, 1.0, 1.0]);
            let y3 = flt.process([s, s], FilterMode::LowPass, 2000.0, 0.707);
            assert!(y1[0].is_finite() && y1[0].abs() < 5.0);
            assert!(y2[0].is_finite() && y2[0].abs() < 5.0);
            assert!(y3[0].is_finite() && y3[0].abs() < 5.0);
        }
    }
}
