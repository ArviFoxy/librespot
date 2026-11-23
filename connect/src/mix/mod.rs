//! Spotify Mix transition resolution, clean-room preset curves, and Connect integration.

pub mod presets;
pub mod resolver;

use std::str::FromStr;

use crate::playback::mix::{EqTopology, TimeStretchMode};

/// Activation policy for Spotify Mix dual-deck transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MixMode {
    /// Disable dual-deck transitions and do not advertise `supports_playlist_mixing`.
    Off,
    /// Advertise `supports_playlist_mixing` and engage transitions whenever Spotify signals
    /// that the current playlist/context is mixed (or carries explicit transition recipes).
    #[default]
    Auto,
    /// Synthesize beatmatched Auto transitions across any context with a next track,
    /// even when Spotify does not mark the playlist as mixed.
    ForceAuto,
}

impl FromStr for MixMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "disabled" | "none" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "force-auto" | "force_auto" | "force" => Ok(Self::ForceAuto),
            other => Err(format!(
                "invalid mix mode '{other}' (expected 'off', 'auto', or 'force-auto')"
            )),
        }
    }
}

/// Configuration for Spotify Mix transitions in [`crate::ConnectConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct MixConfig {
    /// Activation policy (`Off`, `Auto`, or `ForceAuto`).
    pub mode: MixMode,
    /// Maximum allowed tempo stretch ratio after octave BPM folding (default: `1.08`, i.e., +/- 8%).
    pub max_bpm_ratio: f64,
    /// Time-stretching engine used for beatmatched transitions.
    pub stretch_mode: TimeStretchMode,
    /// 3-band EQ topology used during transitions.
    pub eq_topology: EqTopology,
}

impl Default for MixConfig {
    fn default() -> Self {
        Self {
            mode: MixMode::Auto,
            max_bpm_ratio: 1.08,
            stretch_mode: TimeStretchMode::PitchPreserving,
            eq_topology: EqTopology::default(),
        }
    }
}
