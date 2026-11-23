//! Real-time dual-deck DJ transition processing for Spotify Mix.
//!
//! This module is protocol-agnostic: Spotify protobuf definitions are translated
//! into [`TransitionSpec`] outside `librespot-playback`, allowing deterministic
//! offline verification of beat alignment, filtering, stereo coherence, and
//! loudness safety without network access.

pub mod beat_clock;
pub mod curve;
pub(crate) mod driver;
pub mod engine;
pub mod filters;
pub mod spec;
pub mod stretch;

pub(crate) use driver::{
    ActiveTransition, ENTRY_GRACE_FRAMES, EntryPacketDecision, TransitionDeck, split_entry_packet,
};
pub use engine::TransitionProcessor;
pub use spec::{
    AutomationCurve, AutomationCurves, BeatGrid, BeatMarker, CurvePoint, DeckCurves, EqTopology,
    FilterMode, InterpolationSegment, SegmentEasing, SegmentInterpolation, SpecialCase,
    SpecialResult, SpecialRole, TempoPlan, TimeStretchMode, TransitionSpec, ValueMapping,
    ValueSemantics, ValueUnit,
};
