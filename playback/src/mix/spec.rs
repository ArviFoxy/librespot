//! Protocol-agnostic data types describing a transition between two tracks.

/// A single control point `(x, y)` on a normalized automation curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvePoint {
    pub x: f64,
    pub y: f64,
}

/// A piecewise-linear automation segment spanning `[start, end]` in normalized
/// transition progress `u in [0.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationCurve {
    pub points: Vec<CurvePoint>,
    pub start: f64,
    pub end: f64,
}

/// A set of automation curves bounded by `[minimum, maximum]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationCurves {
    pub curves: Vec<AutomationCurve>,
    pub minimum: f64,
    pub maximum: f64,
}

/// Special discrete roles encoded in Spotify's `AutomationValueSemantics`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialRole {
    Bypass,
    Off,
    Mute,
    Kill,
}

/// Filter operating mode for modulated filter cutoff automation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Bypass,
    LowPass,
    HighPass,
}

/// Interpolation law within a piecewise value-semantics segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentInterpolation {
    Linear,
    SignedGeometric,
}

/// Easing function applied to normalized progress within a value-semantics segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentEasing {
    None,
    PowerIn(f64),
    PowerOut(f64),
}

/// Single piecewise segment mapping raw curve values `[raw_start, raw_end]`
/// to physical values `[output_start, output_end]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpolationSegment {
    pub raw_start: f64,
    pub raw_end: f64,
    pub output_start: f64,
    pub output_end: f64,
    pub output_mode: Option<FilterMode>,
    pub interpolation: SegmentInterpolation,
    pub easing: SegmentEasing,
}

/// Discrete special-case override on a raw value or raw interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecialCase {
    pub raw_min: f64,
    pub raw_max: f64,
    pub result: SpecialResult,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialResult {
    Role(SpecialRole),
    Numeric(f64),
}

/// Mapping law from raw automation curve values to physical DSP units.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueMapping {
    /// Pass raw values through directly.
    Identity,
    /// `output = output_multiplier * log10(raw / raw_reference)`.
    Base10LogRatio {
        raw_reference: f64,
        output_multiplier: f64,
    },
    /// Piecewise mapping across ordered segments.
    Piecewise(Vec<InterpolationSegment>),
}

/// Physical unit produced by [`ValueSemantics`] after mapping raw curve values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueUnit {
    #[default]
    Linear,
    Decibels,
    Hertz,
    Percent,
}

/// Complete mapping semantics for one automation parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSemantics {
    pub output_unit: ValueUnit,
    pub mapping: ValueMapping,
    pub special_cases: Vec<SpecialCase>,
}

impl Default for ValueSemantics {
    fn default() -> Self {
        Self {
            output_unit: ValueUnit::Linear,
            mapping: ValueMapping::Identity,
            special_cases: Vec::new(),
        }
    }
}

/// A single beat marker from track beat-grid analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatMarker {
    pub time_sec: f64,
    pub duration_sec: f64,
    pub beat_in_bar: u32,
    pub beat_confidence: f32,
    pub downbeat_confidence: f32,
}

/// Track beat grid used for sample-accurate beat-clock mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatGrid {
    pub beats_per_bar: u32,
    pub beats: Vec<BeatMarker>,
}

/// Automation curves and value semantics for one deck (outgoing or incoming).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeckCurves {
    pub volume: Option<AutomationCurves>,
    pub volume_semantics: Option<ValueSemantics>,
    pub eq_low: Option<AutomationCurves>,
    pub eq_mid: Option<AutomationCurves>,
    pub eq_high: Option<AutomationCurves>,
    pub eq_semantics: Option<ValueSemantics>,
    pub filter_cutoff: Option<AutomationCurves>,
    pub filter_cutoff_semantics: Option<ValueSemantics>,
    pub filter_resonance: Option<AutomationCurves>,
    pub filter_resonance_semantics: Option<ValueSemantics>,
}

/// Time-stretching engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeStretchMode {
    /// Pitch-preserving spectral time-stretching (keylock).
    #[default]
    PitchPreserving,
    /// Linked pitch/tempo resampling (turntable varispeed).
    Varispeed,
    /// Disable tempo matching (play both decks at 1.0x speed).
    Off,
}

/// Tempo-matching parameters for a transition.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoPlan {
    pub bpm_a: f64,
    pub bpm_b: f64,
    pub beatmatched: bool,
    /// Maximum allowed tempo ratio (e.g. `1.08` for +/- 8%) after octave folding.
    pub max_bpm_ratio: f64,
    pub beats_a: Option<BeatGrid>,
    pub beats_b: Option<BeatGrid>,
    pub stretch_mode: TimeStretchMode,
}

impl TempoPlan {
    pub fn beatmatched(bpm_a: f64, bpm_b: f64) -> Self {
        Self {
            bpm_a,
            bpm_b,
            beatmatched: true,
            max_bpm_ratio: 1.08,
            beats_a: None,
            beats_b: None,
            stretch_mode: TimeStretchMode::PitchPreserving,
        }
    }
}

/// 3-band equalizer topology for deck processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EqTopology {
    /// Phase-matched 4th-order Linkwitz-Riley 3-band DJ isolator.
    /// Supports linear gain down to `0.0` (full band kill) with flat all-pass sum at unity.
    LinkwitzRileyIsolator { low_hz: f64, high_hz: f64 },
    /// 3-band Topology-Preserving Transform low-shelf / bell / high-shelf EQ (gains in dB).
    /// Exact unity identity (`H(z) = 1`) at 0 dB.
    TptShelving {
        low_hz: f64,
        mid_hz: f64,
        high_hz: f64,
    },
}

impl Default for EqTopology {
    fn default() -> Self {
        Self::LinkwitzRileyIsolator {
            low_hz: 250.0,
            high_hz: 2500.0,
        }
    }
}

/// Complete protocol-agnostic transition specification executed by [`crate::mix::TransitionProcessor`].
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionSpec {
    /// Exact PCM frame index in outgoing track A where overlap starts.
    pub start_a_frame: u64,
    /// Exact PCM frame index in incoming track B where overlap starts.
    pub start_b_frame: u64,
    /// Nominal overlap duration in output PCM frames at [`crate::SAMPLE_RATE`].
    pub duration_frames: u64,
    /// Optional overlap length in beats (drives the shared beat clock when beatmatched).
    pub duration_beats: Option<f64>,
    /// Fraction in `[0.0, 1.0]` of overlap progress at which track B becomes the reported current track.
    pub handoff_fraction: f64,
    /// Optional beatmatching / tempo-ramp plan.
    pub tempo: Option<TempoPlan>,
    pub out_curves: DeckCurves,
    pub in_curves: DeckCurves,
    pub eq_topology: EqTopology,
}

impl TransitionSpec {
    /// Plain equal-power crossfade without tempo matching, EQ, or filter sweeps.
    pub fn equal_power_crossfade(
        start_a_frame: u64,
        start_b_frame: u64,
        duration_frames: u64,
    ) -> Self {
        Self {
            start_a_frame,
            start_b_frame,
            duration_frames,
            duration_beats: None,
            handoff_fraction: 0.5,
            tempo: None,
            out_curves: DeckCurves::default(),
            in_curves: DeckCurves::default(),
            eq_topology: EqTopology::default(),
        }
    }
}
