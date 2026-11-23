#[cfg(feature = "mix-transitions")]
use std::collections::VecDeque;
use std::{
    collections::HashMap,
    fmt, fs,
    fs::File,
    future::Future,
    io::{self, Read, Seek, SeekFrom},
    mem,
    pin::Pin,
    process::exit,
    sync::Mutex,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    thread,
    time::{Duration, Instant},
};

#[cfg(feature = "passthrough-decoder")]
use crate::decoder::PassthroughDecoder;
#[cfg(feature = "mix-transitions")]
use crate::mix::{
    ActiveTransition, EntryPacketDecision, TransitionDeck, TransitionSpec, split_entry_packet,
};
#[cfg(feature = "mix-transitions")]
use crate::{NUM_CHANNELS, SAMPLE_RATE};
use crate::{
    audio::{AudioDecrypt, AudioFetchParams, AudioFile, StreamLoaderController},
    audio_backend::Sink,
    config::{Bitrate, NormalisationMethod, NormalisationType, PlayerConfig},
    convert::Converter,
    core::{Error, Session, SpotifyId, SpotifyUri, util::SeqGenerator},
    decoder::{AudioDecoder, AudioPacket, AudioPacketPosition, SymphoniaDecoder},
    local_file::{LocalFileLookup, create_local_file_lookup},
    metadata::audio::{AudioFileFormat, AudioFiles, AudioItem},
    mixer::VolumeGetter,
};
use futures_util::{
    StreamExt, TryFutureExt, future, future::FusedFuture,
    stream::futures_unordered::FuturesUnordered,
};
use librespot_metadata::{audio::UniqueFields, track::Tracks};

use symphonia::core::io::MediaSource;
use symphonia::core::probe::Hint;
use tokio::sync::{mpsc, oneshot};

use crate::SAMPLES_PER_SECOND;

const PRELOAD_NEXT_TRACK_BEFORE_END_DURATION_MS: u32 = 30000;
pub const DB_VOLTAGE_RATIO: f64 = 20.0;
pub const PCM_AT_0DBFS: f64 = 1.0;

// Spotify inserts a custom Ogg packet at the start with custom metadata values, that you would
// otherwise expect in Vorbis comments. This packet isn't well-formed and players may balk at it.
const SPOTIFY_OGG_HEADER_END: u64 = 0xa7;

const LOAD_HANDLES_POISON_MSG: &str = "load handles mutex should not be poisoned";

pub type PlayerResult = Result<(), Error>;

pub struct Player {
    commands: Option<mpsc::UnboundedSender<PlayerCommand>>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum SinkStatus {
    Running,
    Closed,
    TemporarilyClosed,
}

pub type SinkEventCallback = Box<dyn Fn(SinkStatus) + Send>;

/// Per-track loudness normalisation applied before any dual-deck summing.
#[inline]
pub(crate) fn apply_track_normalisation(
    data: &mut [f64],
    config: &PlayerConfig,
    normalisation_factor: f64,
) {
    if !config.normalisation {
        return;
    }

    match config.normalisation_method {
        NormalisationMethod::Dynamic => {
            for sample in data.iter_mut() {
                *sample *= normalisation_factor;
            }
        }
        NormalisationMethod::Basic => {
            if normalisation_factor < 1.0 {
                for sample in data.iter_mut() {
                    *sample *= normalisation_factor;
                }
            }
        }
    }
}

/// Stereo-coupled feedforward dynamic soft-knee limiter (Giannoulis et al., 2012)
/// and master volume stage applied after track decoding / mixing.
pub(crate) struct MasterLimiter {
    integrators: [f64; 2],
    peaks: [f64; 2],
    channel: usize,
    knee_factor: f64,
}

impl MasterLimiter {
    pub(crate) fn new(config: &PlayerConfig) -> Self {
        Self {
            integrators: [0.0; 2],
            peaks: [0.0; 2],
            channel: 0,
            knee_factor: 1.0 / (8.0 * config.normalisation_knee_db),
        }
    }

    pub(crate) fn process_post_mix(
        &mut self,
        data: &mut [f64],
        config: &PlayerConfig,
        volume: f64,
    ) {
        match (config.normalisation, config.normalisation_method) {
            (true, NormalisationMethod::Dynamic) => {
                self.run_feedforward_limiter(
                    data,
                    config.normalisation_threshold_dbfs,
                    config.normalisation_knee_db,
                    self.knee_factor,
                    config.normalisation_attack_cf,
                    config.normalisation_release_cf,
                    volume,
                );
            }
            (false, _) | (true, NormalisationMethod::Basic) => {
                if volume < 1.0 {
                    for sample in data.iter_mut() {
                        *sample *= volume;
                    }
                }
            }
        }
    }

    /// Post-mix master stage for dual-deck transitions: uses the configured dynamic limiter when
    /// enabled, and engages a transparent `-0.5 dBFS` safety ceiling limiter when dynamic
    /// normalisation is disabled so full-overlap DJ transitions never hard-clip.
    #[cfg(feature = "mix-transitions")]
    pub(crate) fn process_transition_post_mix(
        &mut self,
        data: &mut [f64],
        config: &PlayerConfig,
        volume: f64,
    ) {
        match (config.normalisation, config.normalisation_method) {
            (true, NormalisationMethod::Dynamic) => {
                self.process_post_mix(data, config, volume);
            }
            (false, _) | (true, NormalisationMethod::Basic) => {
                const SAFETY_THRESHOLD_DB: f64 = -0.5;
                const SAFETY_KNEE_DB: f64 = 1.0;
                const SAFETY_KNEE_FACTOR: f64 = 1.0 / (8.0 * SAFETY_KNEE_DB);
                let attack_cf = (-1.0 / (0.0005 * f64::from(SAMPLE_RATE))).exp();
                let release_cf = (-1.0 / (0.080 * f64::from(SAMPLE_RATE))).exp();
                self.run_feedforward_limiter(
                    data,
                    SAFETY_THRESHOLD_DB,
                    SAFETY_KNEE_DB,
                    SAFETY_KNEE_FACTOR,
                    attack_cf,
                    release_cf,
                    volume,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_feedforward_limiter(
        &mut self,
        data: &mut [f64],
        threshold_db: f64,
        knee_db: f64,
        knee_factor: f64,
        attack_cf: f64,
        release_cf: f64,
        volume: f64,
    ) {
        for sample in data.iter_mut() {
            // Feedforward limiter in the log domain
            // After: Giannoulis, D., Massberg, M., & Reiss, J.D. (2012).
            // Digital Dynamic Range Compressor Design—A Tutorial and
            // Analysis. Journal of The Audio Engineering Society, 60,
            // 399-408.
            //
            // Assumes interleaved stereo audio.

            // step 1-4: half-wave rectification and conversion into dB, and
            // gain computer with soft knee and subtractor
            let limiter_db = {
                // Add slight DC offset. Some samples are silence, which is
                // -inf dB and gets the limiter stuck. Adding a small
                // positive offset prevents this.
                *sample += f64::MIN_POSITIVE;

                let bias_db = ratio_to_db(sample.abs()) - threshold_db;
                let knee_boundary_db = bias_db * 2.0;
                if knee_boundary_db < -knee_db {
                    0.0
                } else if knee_boundary_db.abs() <= knee_db {
                    let term = knee_boundary_db + knee_db;
                    term * term * knee_factor
                } else {
                    bias_db
                }
            };

            // track left/right channel
            let channel = self.channel;
            self.channel ^= 1;

            // step 5: smooth, decoupled peak detector for each channel
            let integrator = &mut self.integrators[channel];
            let peak = &mut self.peaks[channel];

            *integrator = f64::max(
                limiter_db,
                release_cf * *integrator + (1.0 - release_cf) * limiter_db,
            );
            *peak = attack_cf * *peak + (1.0 - attack_cf) * *integrator;

            // steps 6-8: conversion into level and multiplication into gain
            // stage. Find maximum peak across both channels to couple the
            // gain and maintain stereo imaging.
            let max_peak = f64::max(self.peaks[0], self.peaks[1]);
            *sample *= db_to_ratio(-max_peak) * volume;
        }
    }
}

struct PlayerInternal {
    session: Session,
    config: PlayerConfig,
    commands: mpsc::UnboundedReceiver<PlayerCommand>,
    load_handles: Arc<Mutex<HashMap<thread::ThreadId, thread::JoinHandle<()>>>>,

    state: PlayerState,
    preload: PlayerPreload,
    sink: Box<dyn Sink>,
    sink_status: SinkStatus,
    sink_event_callback: Option<SinkEventCallback>,
    volume_getter: Box<dyn VolumeGetter + Send>,
    event_senders: Vec<mpsc::UnboundedSender<PlayerEvent>>,
    converter: Converter,

    master_limiter: MasterLimiter,

    #[cfg(feature = "mix-transitions")]
    scheduled_transition: Option<ScheduledTransition>,

    auto_normalise_as_album: bool,

    player_id: usize,
    play_request_id_generator: SeqGenerator<u64>,
    last_progress_update: Instant,

    local_file_lookup: Arc<LocalFileLookup>,
}

static PLAYER_COUNTER: AtomicUsize = AtomicUsize::new(0);

enum PlayerCommand {
    Load {
        track_id: SpotifyUri,
        play: bool,
        position_ms: u32,
    },
    Preload {
        track_id: SpotifyUri,
    },
    Play,
    Pause,
    Stop,
    Seek(u32),
    SetSession(Session),
    AddEventSender(mpsc::UnboundedSender<PlayerEvent>),
    SetSinkEventCallback(Option<SinkEventCallback>),
    EmitVolumeChangedEvent(u16),
    SetAutoNormaliseAsAlbum(bool),
    EmitSessionDisconnectedEvent {
        connection_id: String,
        user_name: String,
    },
    EmitSessionConnectedEvent {
        connection_id: String,
        user_name: String,
    },
    EmitSessionClientChangedEvent {
        client_id: String,
        client_name: String,
        client_brand_name: String,
        client_model_name: String,
    },
    EmitFilterExplicitContentChangedEvent(bool),
    EmitShuffleChangedEvent(bool),
    EmitRepeatChangedEvent {
        context: bool,
        track: bool,
    },
    EmitAutoPlayChangedEvent(bool),
    EmitSetQueueEvent {
        context_uri: String,
        current_track: Option<QueueTrack>,
        next_tracks: Vec<QueueTrack>,
        prev_tracks: Vec<QueueTrack>,
    },
    #[cfg(feature = "mix-transitions")]
    ScheduleTransition {
        outgoing_track_id: SpotifyUri,
        next_track_id: SpotifyUri,
        spec: Box<TransitionSpec>,
    },
    #[cfg(feature = "mix-transitions")]
    CancelTransition,
}

/// Represents a track in the queue with its URI and provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueTrack {
    pub uri: String,
    pub provider: String,
}

#[derive(Debug, Clone)]
pub enum PlayerEvent {
    // Play request id changed
    PlayRequestIdChanged {
        play_request_id: u64,
    },
    // Fired when the player is stopped (e.g. by issuing a "stop" command to the player).
    Stopped {
        play_request_id: u64,
        track_id: SpotifyUri,
    },
    // The player is delayed by loading a track.
    Loading {
        play_request_id: u64,
        track_id: SpotifyUri,
        position_ms: u32,
    },
    // The player is preloading a track.
    Preloading {
        track_id: SpotifyUri,
    },
    // The player is playing a track.
    // This event is issued at the start of playback of whenever the position must be communicated
    // because it is out of sync. This includes:
    // start of a track
    // un-pausing
    // after a seek
    // after a buffer-underrun
    Playing {
        play_request_id: u64,
        track_id: SpotifyUri,
        position_ms: u32,
    },
    // The player entered a paused state.
    Paused {
        play_request_id: u64,
        track_id: SpotifyUri,
        position_ms: u32,
    },
    // The player thinks it's a good idea to issue a preload command for the next track now.
    // This event is intended for use within spirc.
    TimeToPreloadNextTrack {
        play_request_id: u64,
        track_id: SpotifyUri,
    },
    // The player reached the end of a track.
    // This event is intended for use within spirc. Spirc will respond by issuing another command.
    EndOfTrack {
        play_request_id: u64,
        track_id: SpotifyUri,
    },
    // The player was unable to load the requested track.
    Unavailable {
        play_request_id: u64,
        track_id: SpotifyUri,
    },
    // The mixer volume was set to a new level.
    VolumeChanged {
        volume: u16,
    },
    PositionCorrection {
        play_request_id: u64,
        track_id: SpotifyUri,
        position_ms: u32,
    },
    /// Requires `PlayerConfig::position_update_interval` to be set to Some.
    /// Once set this event will be sent periodically while playing the track to inform about the
    /// current playback position
    PositionChanged {
        play_request_id: u64,
        track_id: SpotifyUri,
        position_ms: u32,
    },
    Seeked {
        play_request_id: u64,
        track_id: SpotifyUri,
        position_ms: u32,
    },
    TrackChanged {
        audio_item: Box<AudioItem>,
    },
    SessionConnected {
        connection_id: String,
        user_name: String,
    },
    SessionDisconnected {
        connection_id: String,
        user_name: String,
    },
    SessionClientChanged {
        client_id: String,
        client_name: String,
        client_brand_name: String,
        client_model_name: String,
    },
    ShuffleChanged {
        shuffle: bool,
    },
    RepeatChanged {
        context: bool,
        track: bool,
    },
    AutoPlayChanged {
        auto_play: bool,
    },
    FilterExplicitContentChanged {
        filter: bool,
    },
    /// Fired when the queue is set or context is loaded with its track list.
    SetQueue {
        context_uri: String,
        current_track: Option<QueueTrack>,
        next_tracks: Vec<QueueTrack>,
        prev_tracks: Vec<QueueTrack>,
    },
    /// Fired when a scheduled dual-deck DJ transition begins overlapping outgoing and incoming tracks.
    #[cfg(feature = "mix-transitions")]
    TransitionStarted {
        outgoing_play_request_id: u64,
        outgoing_track_id: SpotifyUri,
        incoming_track_id: SpotifyUri,
    },
    /// Fired once when the active transition crosses `handoff_fraction` (or finishes), committing
    /// the incoming track as active without issuing a redundant `Player::load`.
    #[cfg(feature = "mix-transitions")]
    TransitionCommit {
        outgoing_play_request_id: u64,
        incoming_track_id: SpotifyUri,
        incoming_play_request_id: u64,
        incoming_position_ms: u32,
    },
}

impl PlayerEvent {
    pub fn get_play_request_id(&self) -> Option<u64> {
        use PlayerEvent::*;
        match self {
            Loading {
                play_request_id, ..
            }
            | Unavailable {
                play_request_id, ..
            }
            | Playing {
                play_request_id, ..
            }
            | TimeToPreloadNextTrack {
                play_request_id, ..
            }
            | EndOfTrack {
                play_request_id, ..
            }
            | Paused {
                play_request_id, ..
            }
            | Stopped {
                play_request_id, ..
            }
            | PositionCorrection {
                play_request_id, ..
            }
            | Seeked {
                play_request_id, ..
            } => Some(*play_request_id),
            #[cfg(feature = "mix-transitions")]
            TransitionStarted {
                outgoing_play_request_id,
                ..
            }
            | TransitionCommit {
                outgoing_play_request_id,
                ..
            } => Some(*outgoing_play_request_id),
            _ => None,
        }
    }
}

pub type PlayerEventChannel = mpsc::UnboundedReceiver<PlayerEvent>;

#[inline]
pub fn db_to_ratio(db: f64) -> f64 {
    f64::powf(10.0, db / DB_VOLTAGE_RATIO)
}

#[inline]
pub fn ratio_to_db(ratio: f64) -> f64 {
    ratio.log10() * DB_VOLTAGE_RATIO
}

pub fn duration_to_coefficient(duration: Duration) -> f64 {
    f64::exp(-1.0 / (duration.as_secs_f64() * SAMPLES_PER_SECOND as f64))
}

pub fn coefficient_to_duration(coefficient: f64) -> Duration {
    Duration::from_secs_f64(-1.0 / f64::ln(coefficient) / SAMPLES_PER_SECOND as f64)
}

#[derive(Clone, Copy, Debug)]
pub struct NormalisationData {
    // Spotify provides these as `f32`, but audio metadata can contain up to `f64`.
    // Also, this negates the need for casting during sample processing.
    pub track_gain_db: f64,
    pub track_peak: f64,
    pub album_gain_db: f64,
    pub album_peak: f64,
}

impl Default for NormalisationData {
    fn default() -> Self {
        Self {
            track_gain_db: 0.0,
            track_peak: 1.0,
            album_gain_db: 0.0,
            album_peak: 1.0,
        }
    }
}

impl NormalisationData {
    fn parse_from_ogg<T: Read + Seek>(mut file: T) -> io::Result<NormalisationData> {
        const SPOTIFY_NORMALIZATION_HEADER_START_OFFSET: u64 = 144;
        const NORMALISATION_DATA_SIZE: usize = 16;

        let newpos = file.seek(SeekFrom::Start(SPOTIFY_NORMALIZATION_HEADER_START_OFFSET))?;
        if newpos != SPOTIFY_NORMALIZATION_HEADER_START_OFFSET {
            error!(
                "NormalisationData::parse_from_file seeking to {SPOTIFY_NORMALIZATION_HEADER_START_OFFSET} but position is now {newpos}"
            );

            error!("Falling back to default (non-track and non-album) normalisation data.");

            return Ok(NormalisationData::default());
        }

        let mut buf = [0u8; NORMALISATION_DATA_SIZE];

        file.read_exact(&mut buf)?;

        let track_gain_db = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f64;
        let track_peak = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as f64;
        let album_gain_db = f32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as f64;
        let album_peak = f32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as f64;

        Ok(Self {
            track_gain_db,
            track_peak,
            album_gain_db,
            album_peak,
        })
    }

    fn get_factor(config: &PlayerConfig, data: NormalisationData) -> f64 {
        if !config.normalisation {
            return 1.0;
        }

        let (gain_db, gain_peak) = if config.normalisation_type == NormalisationType::Album {
            (data.album_gain_db, data.album_peak)
        } else {
            (data.track_gain_db, data.track_peak)
        };

        // As per the ReplayGain 1.0 & 2.0 (proposed) spec:
        // https://wiki.hydrogenaud.io/index.php?title=ReplayGain_1.0_specification#Clipping_prevention
        // https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#Clipping_prevention
        let normalisation_factor = if config.normalisation_method == NormalisationMethod::Basic {
            // For Basic Normalisation, factor = min(ratio of (ReplayGain + PreGain), 1.0 / peak level).
            // https://wiki.hydrogenaud.io/index.php?title=ReplayGain_1.0_specification#Peak_amplitude
            // https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#Peak_amplitude
            // We then limit that to 1.0 as not to exceed dBFS (0.0 dB).
            let factor = f64::min(
                db_to_ratio(gain_db + config.normalisation_pregain_db),
                PCM_AT_0DBFS / gain_peak,
            );

            if factor > PCM_AT_0DBFS {
                info!(
                    "Lowering gain by {:.2} dB for the duration of this track to avoid potentially exceeding dBFS.",
                    ratio_to_db(factor)
                );

                PCM_AT_0DBFS
            } else {
                factor
            }
        } else {
            // For Dynamic Normalisation it's up to the player to decide,
            // factor = ratio of (ReplayGain + PreGain).
            // We then let the dynamic limiter handle gain reduction.
            let factor = db_to_ratio(gain_db + config.normalisation_pregain_db);
            let threshold_ratio = db_to_ratio(config.normalisation_threshold_dbfs);

            if factor > PCM_AT_0DBFS {
                let factor_db = gain_db + config.normalisation_pregain_db;
                let limiting_db = factor_db + config.normalisation_threshold_dbfs.abs();

                warn!(
                    "This track may exceed dBFS by {factor_db:.2} dB and be subject to {limiting_db:.2} dB of dynamic limiting at its peak."
                );
            } else if factor > threshold_ratio {
                let limiting_db = gain_db
                    + config.normalisation_pregain_db
                    + config.normalisation_threshold_dbfs.abs();

                info!(
                    "This track may be subject to {limiting_db:.2} dB of dynamic limiting at its peak."
                );
            }

            factor
        };

        debug!("Normalisation Data: {data:?}");
        debug!(
            "Calculated Normalisation Factor for {:?}: {:.2}%",
            config.normalisation_type,
            normalisation_factor * 100.0
        );

        normalisation_factor
    }
}

impl Player {
    pub fn new<F>(
        config: PlayerConfig,
        session: Session,
        volume_getter: Box<dyn VolumeGetter + Send>,
        sink_builder: F,
    ) -> Arc<Self>
    where
        F: FnOnce() -> Box<dyn Sink> + Send + 'static,
    {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

        if config.normalisation {
            debug!("Normalisation Type: {:?}", config.normalisation_type);
            debug!(
                "Normalisation Pregain: {:.1} dB",
                config.normalisation_pregain_db
            );
            debug!(
                "Normalisation Threshold: {:.1} dBFS",
                config.normalisation_threshold_dbfs
            );
            debug!("Normalisation Method: {:?}", config.normalisation_method);

            if config.normalisation_method == NormalisationMethod::Dynamic {
                // as_millis() has rounding errors (truncates)
                debug!(
                    "Normalisation Attack: {:.0} ms",
                    coefficient_to_duration(config.normalisation_attack_cf).as_secs_f64() * 1000.
                );
                debug!(
                    "Normalisation Release: {:.0} ms",
                    coefficient_to_duration(config.normalisation_release_cf).as_secs_f64() * 1000.
                );
                debug!("Normalisation Knee: {} dB", config.normalisation_knee_db);
            }
        }

        let handle = thread::spawn(move || {
            let player_id = PLAYER_COUNTER.fetch_add(1, Ordering::AcqRel);
            debug!("new Player [{player_id}]");

            let converter = Converter::new(config.ditherer);
            let master_limiter = MasterLimiter::new(&config);

            // TODO: it would be neat if we could watch for added or modified files in the
            // specified directories, and dynamically update the lookup. Currently, a new player
            // must be created for any new local files to be playable.
            let local_file_lookup =
                create_local_file_lookup(config.local_file_directories.as_slice());

            let internal = PlayerInternal {
                session,
                config,
                commands: cmd_rx,
                load_handles: Arc::new(Mutex::new(HashMap::new())),

                state: PlayerState::Stopped,
                preload: PlayerPreload::None,
                sink: sink_builder(),
                sink_status: SinkStatus::Closed,
                sink_event_callback: None,
                volume_getter,
                event_senders: vec![],
                converter,

                master_limiter,

                #[cfg(feature = "mix-transitions")]
                scheduled_transition: None,

                auto_normalise_as_album: false,

                player_id,
                play_request_id_generator: SeqGenerator::new(0),
                last_progress_update: Instant::now(),

                local_file_lookup: Arc::new(local_file_lookup),
            };

            // While PlayerInternal is written as a future, it still contains blocking code.
            // It must be run by using block_on() in a dedicated thread.
            let runtime = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
            runtime.block_on(internal);

            debug!("PlayerInternal thread finished.");
        });

        Arc::new(Self {
            commands: Some(cmd_tx),
            thread_handle: Some(handle),
        })
    }

    pub fn is_invalid(&self) -> bool {
        if let Some(handle) = self.thread_handle.as_ref() {
            return handle.is_finished();
        }
        true
    }

    fn command(&self, cmd: PlayerCommand) {
        if let Some(commands) = self.commands.as_ref() {
            if let Err(e) = commands.send(cmd) {
                error!("Player Commands Error: {e}");
            }
        }
    }

    pub fn load(&self, track_id: SpotifyUri, start_playing: bool, position_ms: u32) {
        self.command(PlayerCommand::Load {
            track_id,
            play: start_playing,
            position_ms,
        });
    }

    pub fn preload(&self, track_id: SpotifyUri) {
        self.command(PlayerCommand::Preload { track_id });
    }

    #[cfg(feature = "mix-transitions")]
    pub fn schedule_transition(
        &self,
        outgoing_track_id: SpotifyUri,
        next_track_id: SpotifyUri,
        spec: TransitionSpec,
    ) {
        self.command(PlayerCommand::ScheduleTransition {
            outgoing_track_id,
            next_track_id,
            spec: Box::new(spec),
        });
    }

    #[cfg(feature = "mix-transitions")]
    pub fn cancel_transition(&self) {
        self.command(PlayerCommand::CancelTransition);
    }

    pub fn play(&self) {
        self.command(PlayerCommand::Play)
    }

    pub fn pause(&self) {
        self.command(PlayerCommand::Pause)
    }

    pub fn stop(&self) {
        self.command(PlayerCommand::Stop)
    }

    pub fn seek(&self, position_ms: u32) {
        self.command(PlayerCommand::Seek(position_ms));
    }

    pub fn set_session(&self, session: Session) {
        self.command(PlayerCommand::SetSession(session));
    }

    pub fn get_player_event_channel(&self) -> PlayerEventChannel {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        self.command(PlayerCommand::AddEventSender(event_sender));
        event_receiver
    }

    pub async fn await_end_of_track(&self) {
        let mut channel = self.get_player_event_channel();
        while let Some(event) = channel.recv().await {
            if matches!(
                event,
                PlayerEvent::EndOfTrack { .. } | PlayerEvent::Stopped { .. }
            ) {
                return;
            }
        }
    }

    pub fn set_sink_event_callback(&self, callback: Option<SinkEventCallback>) {
        self.command(PlayerCommand::SetSinkEventCallback(callback));
    }

    pub fn emit_volume_changed_event(&self, volume: u16) {
        self.command(PlayerCommand::EmitVolumeChangedEvent(volume));
    }

    pub fn set_auto_normalise_as_album(&self, setting: bool) {
        self.command(PlayerCommand::SetAutoNormaliseAsAlbum(setting));
    }

    pub fn emit_filter_explicit_content_changed_event(&self, filter: bool) {
        self.command(PlayerCommand::EmitFilterExplicitContentChangedEvent(filter));
    }

    pub fn emit_session_connected_event(&self, connection_id: String, user_name: String) {
        self.command(PlayerCommand::EmitSessionConnectedEvent {
            connection_id,
            user_name,
        });
    }

    pub fn emit_session_disconnected_event(&self, connection_id: String, user_name: String) {
        self.command(PlayerCommand::EmitSessionDisconnectedEvent {
            connection_id,
            user_name,
        });
    }

    pub fn emit_session_client_changed_event(
        &self,
        client_id: String,
        client_name: String,
        client_brand_name: String,
        client_model_name: String,
    ) {
        self.command(PlayerCommand::EmitSessionClientChangedEvent {
            client_id,
            client_name,
            client_brand_name,
            client_model_name,
        });
    }

    pub fn emit_shuffle_changed_event(&self, shuffle: bool) {
        self.command(PlayerCommand::EmitShuffleChangedEvent(shuffle));
    }

    pub fn emit_repeat_changed_event(&self, context: bool, track: bool) {
        self.command(PlayerCommand::EmitRepeatChangedEvent { context, track });
    }

    pub fn emit_auto_play_changed_event(&self, auto_play: bool) {
        self.command(PlayerCommand::EmitAutoPlayChangedEvent(auto_play));
    }

    pub fn emit_set_queue_event(
        &self,
        context_uri: String,
        current_track: Option<QueueTrack>,
        next_tracks: Vec<QueueTrack>,
        prev_tracks: Vec<QueueTrack>,
    ) {
        self.command(PlayerCommand::EmitSetQueueEvent {
            context_uri,
            current_track,
            next_tracks,
            prev_tracks,
        });
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        debug!("Shutting down player thread ...");
        self.commands = None;
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("Player thread Error: {e:?}");
            }
        }
    }
}

struct PlayerLoadedTrackData {
    decoder: Decoder,
    normalisation_data: NormalisationData,
    stream_loader_controller: StreamLoaderController,
    audio_item: AudioItem,
    bytes_per_second: usize,
    duration_ms: u32,
    stream_position_ms: u32,
    is_explicit: bool,
}

#[cfg(feature = "mix-transitions")]
struct ScheduledTransition {
    outgoing_track_id: SpotifyUri,
    next_track_id: SpotifyUri,
    spec: TransitionSpec,
    /// Avoid repeating the missed-entry warning while playback stays past `start_a_frame`
    /// (keeping the transition armed in case the user scrubs back before `start_a_frame`).
    missed_entry_logged: bool,
}

#[cfg(feature = "mix-transitions")]
struct TransitionDeckMeta {
    track_id: SpotifyUri,
    play_request_id: u64,
    audio_item: AudioItem,
    normalisation_data: NormalisationData,
    stream_loader_controller: StreamLoaderController,
    bytes_per_second: usize,
    duration_ms: u32,
    is_explicit: bool,
}

enum PlayerPreload {
    None,
    Loading {
        track_id: SpotifyUri,
        #[cfg(feature = "mix-transitions")]
        start_frame: Option<u64>,
        loader: Pin<Box<dyn FusedFuture<Output = Result<PlayerLoadedTrackData, ()>> + Send>>,
    },
    Ready {
        track_id: SpotifyUri,
        #[cfg(feature = "mix-transitions")]
        start_frame: Option<u64>,
        loaded_track: Box<PlayerLoadedTrackData>,
    },
}

type Decoder = Box<dyn AudioDecoder + Send>;

enum PlayerState {
    Stopped,
    Loading {
        track_id: SpotifyUri,
        play_request_id: u64,
        start_playback: bool,
        loader: Pin<Box<dyn FusedFuture<Output = Result<PlayerLoadedTrackData, ()>> + Send>>,
    },
    Paused {
        track_id: SpotifyUri,
        play_request_id: u64,
        decoder: Decoder,
        audio_item: AudioItem,
        normalisation_data: NormalisationData,
        normalisation_factor: f64,
        stream_loader_controller: StreamLoaderController,
        bytes_per_second: usize,
        duration_ms: u32,
        stream_position_ms: u32,
        suggested_to_preload_next_track: bool,
        is_explicit: bool,
        #[cfg(feature = "mix-transitions")]
        pending_normalized_pcm: VecDeque<f64>,
    },
    Playing {
        track_id: SpotifyUri,
        play_request_id: u64,
        decoder: Decoder,
        normalisation_data: NormalisationData,
        audio_item: AudioItem,
        normalisation_factor: f64,
        stream_loader_controller: StreamLoaderController,
        bytes_per_second: usize,
        duration_ms: u32,
        stream_position_ms: u32,
        reported_nominal_start_time: Option<Instant>,
        suggested_to_preload_next_track: bool,
        is_explicit: bool,
        #[cfg(feature = "mix-transitions")]
        pending_normalized_pcm: VecDeque<f64>,
    },
    #[cfg(feature = "mix-transitions")]
    Transitioning(Box<ActiveTransition<TransitionDeckMeta>>),
    EndOfTrack {
        track_id: SpotifyUri,
        play_request_id: u64,
        loaded_track: PlayerLoadedTrackData,
    },
    Invalid,
}

impl PlayerState {
    fn is_playing(&self) -> bool {
        use self::PlayerState::*;
        match *self {
            Stopped | EndOfTrack { .. } | Paused { .. } | Loading { .. } => false,
            Playing { .. } => true,
            #[cfg(feature = "mix-transitions")]
            Transitioning(ref active) => !active.is_paused(),
            Invalid => {
                error!("PlayerState::is_playing in invalid state");
                exit(1);
            }
        }
    }

    #[allow(dead_code)]
    fn is_stopped(&self) -> bool {
        use self::PlayerState::*;
        matches!(self, Stopped)
    }

    #[allow(dead_code)]
    fn is_loading(&self) -> bool {
        use self::PlayerState::*;
        matches!(self, Loading { .. })
    }

    fn decoder(&mut self) -> Option<&mut Decoder> {
        use self::PlayerState::*;
        match *self {
            Stopped | EndOfTrack { .. } | Loading { .. } => None,
            #[cfg(feature = "mix-transitions")]
            Transitioning(..) => None,
            Paused {
                ref mut decoder, ..
            }
            | Playing {
                ref mut decoder, ..
            } => Some(decoder),
            Invalid => {
                error!("PlayerState::decoder in invalid state");
                exit(1);
            }
        }
    }

    fn playing_to_end_of_track(&mut self) {
        use self::PlayerState::*;
        let new_state = mem::replace(self, Invalid);
        match new_state {
            Playing {
                track_id,
                play_request_id,
                decoder,
                duration_ms,
                bytes_per_second,
                normalisation_data,
                stream_loader_controller,
                stream_position_ms,
                is_explicit,
                audio_item,
                ..
            } => {
                *self = EndOfTrack {
                    track_id,
                    play_request_id,
                    loaded_track: PlayerLoadedTrackData {
                        decoder,
                        normalisation_data,
                        stream_loader_controller,
                        audio_item,
                        bytes_per_second,
                        duration_ms,
                        stream_position_ms,
                        is_explicit,
                    },
                };
            }
            _ => {
                error!("Called playing_to_end_of_track in non-playing state: {new_state:?}");
                exit(1);
            }
        }
    }

    fn paused_to_playing(&mut self) {
        use self::PlayerState::*;
        let new_state = mem::replace(self, Invalid);
        match new_state {
            Paused {
                track_id,
                play_request_id,
                decoder,
                audio_item,
                normalisation_data,
                normalisation_factor,
                stream_loader_controller,
                duration_ms,
                bytes_per_second,
                stream_position_ms,
                suggested_to_preload_next_track,
                is_explicit,
                #[cfg(feature = "mix-transitions")]
                pending_normalized_pcm,
            } => {
                *self = Playing {
                    track_id,
                    play_request_id,
                    decoder,
                    audio_item,
                    normalisation_data,
                    normalisation_factor,
                    stream_loader_controller,
                    duration_ms,
                    bytes_per_second,
                    stream_position_ms,
                    reported_nominal_start_time: Instant::now()
                        .checked_sub(Duration::from_millis(stream_position_ms as u64)),
                    suggested_to_preload_next_track,
                    is_explicit,
                    #[cfg(feature = "mix-transitions")]
                    pending_normalized_pcm,
                };
            }
            _ => {
                error!("PlayerState::paused_to_playing in invalid state: {new_state:?}");
                exit(1);
            }
        }
    }

    fn playing_to_paused(&mut self) {
        use self::PlayerState::*;
        let new_state = mem::replace(self, Invalid);
        match new_state {
            Playing {
                track_id,
                play_request_id,
                decoder,
                audio_item,
                normalisation_data,
                normalisation_factor,
                stream_loader_controller,
                duration_ms,
                bytes_per_second,
                stream_position_ms,
                suggested_to_preload_next_track,
                is_explicit,
                #[cfg(feature = "mix-transitions")]
                pending_normalized_pcm,
                ..
            } => {
                *self = Paused {
                    track_id,
                    play_request_id,
                    decoder,
                    audio_item,
                    normalisation_data,
                    normalisation_factor,
                    stream_loader_controller,
                    duration_ms,
                    bytes_per_second,
                    stream_position_ms,
                    suggested_to_preload_next_track,
                    is_explicit,
                    #[cfg(feature = "mix-transitions")]
                    pending_normalized_pcm,
                };
            }
            _ => {
                error!("PlayerState::playing_to_paused in invalid state: {new_state:?}");
                exit(1);
            }
        }
    }
}

struct PlayerTrackLoader {
    session: Session,
    config: PlayerConfig,
    local_file_lookup: Arc<LocalFileLookup>,
}

impl PlayerTrackLoader {
    async fn find_available_alternative(&self, audio_item: AudioItem) -> Option<AudioItem> {
        if let Err(e) = audio_item.availability {
            error!("Track is unavailable: {e}");
            None
        } else if !audio_item.files.is_empty() {
            Some(audio_item)
        } else if let Some(alternatives) = audio_item.alternatives {
            let Tracks(alternatives_vec) = alternatives; // required to make `into_iter` able to move

            let alternatives: FuturesUnordered<_> = alternatives_vec
                .into_iter()
                .map(|alt_id| AudioItem::get_file(&self.session, alt_id))
                .collect();

            alternatives
                .filter_map(|x| future::ready(x.ok()))
                .filter(|x| future::ready(x.availability.is_ok()))
                .next()
                .await
        } else {
            error!("Track should be available, but no alternatives found.");
            None
        }
    }

    fn stream_data_rate(&self, format: AudioFileFormat) -> Option<usize> {
        let kbps = match format {
            AudioFileFormat::OGG_VORBIS_96 => 12.,
            AudioFileFormat::OGG_VORBIS_160 => 20.,
            AudioFileFormat::OGG_VORBIS_320 => 40.,
            AudioFileFormat::MP3_256 => 32.,
            AudioFileFormat::MP3_320 => 40.,
            AudioFileFormat::MP3_160 => 20.,
            AudioFileFormat::MP3_96 => 12.,
            AudioFileFormat::MP3_160_ENC => 20.,
            AudioFileFormat::AAC_24 => 3.,
            AudioFileFormat::AAC_48 => 6.,
            AudioFileFormat::AAC_160 => 20.,
            AudioFileFormat::AAC_320 => 40.,
            AudioFileFormat::MP4_128 => 16.,
            AudioFileFormat::OTHER5 => 40.,
            AudioFileFormat::FLAC_FLAC => 112., // assume 900 kbit/s on average
            AudioFileFormat::XHE_AAC_12 => 1.5,
            AudioFileFormat::XHE_AAC_16 => 2.,
            AudioFileFormat::XHE_AAC_24 => 3.,
            AudioFileFormat::FLAC_FLAC_24BIT => 3.,
        };
        let data_rate: f32 = kbps * 1024.;
        Some(data_rate.ceil() as usize)
    }

    async fn load_track(
        &self,
        track_uri: SpotifyUri,
        position_ms: u32,
        #[cfg(feature = "mix-transitions")] exact_frame: Option<u64>,
    ) -> Option<PlayerLoadedTrackData> {
        match track_uri {
            SpotifyUri::Track { .. } | SpotifyUri::Episode { .. } => {
                self.load_remote_track(
                    track_uri,
                    position_ms,
                    #[cfg(feature = "mix-transitions")]
                    exact_frame,
                )
                .await
            }
            SpotifyUri::Local { .. } => {
                self.load_local_track(
                    track_uri,
                    position_ms,
                    #[cfg(feature = "mix-transitions")]
                    exact_frame,
                )
                .await
            }
            _ => {
                error!("Cannot handle load of track with URI: <{track_uri}>",);
                None
            }
        }
    }

    async fn load_remote_track(
        &self,
        track_uri: SpotifyUri,
        position_ms: u32,
        #[cfg(feature = "mix-transitions")] exact_frame: Option<u64>,
    ) -> Option<PlayerLoadedTrackData> {
        let track_id: SpotifyId = match (&track_uri).try_into() {
            Ok(id) => id,
            Err(_) => {
                warn!("<{track_uri}> could not be converted to a base62 ID");
                return None;
            }
        };

        let audio_item = match AudioItem::get_file(&self.session, track_uri).await {
            Ok(audio) => match self.find_available_alternative(audio).await {
                Some(audio) => audio,
                None => {
                    warn!("spotify:track:<{}> is not available", track_id.to_base62());
                    return None;
                }
            },
            Err(e) => {
                error!("Unable to load audio item: {e:?}");
                return None;
            }
        };

        info!(
            "Loading <{}> from <{}> with Spotify URI <{}>",
            audio_item.name,
            audio_item.unique_fields.artists_display(),
            audio_item.uri
        );

        // (Most) podcasts seem to support only 96 kbps Ogg Vorbis, so fall back to it
        let formats = match self.config.bitrate {
            Bitrate::Bitrate96 => [
                AudioFileFormat::OGG_VORBIS_96,
                AudioFileFormat::MP3_96,
                AudioFileFormat::OGG_VORBIS_160,
                AudioFileFormat::MP3_160,
                AudioFileFormat::MP3_256,
                AudioFileFormat::OGG_VORBIS_320,
                AudioFileFormat::MP3_320,
            ],
            Bitrate::Bitrate160 => [
                AudioFileFormat::OGG_VORBIS_160,
                AudioFileFormat::MP3_160,
                AudioFileFormat::OGG_VORBIS_96,
                AudioFileFormat::MP3_96,
                AudioFileFormat::MP3_256,
                AudioFileFormat::OGG_VORBIS_320,
                AudioFileFormat::MP3_320,
            ],
            Bitrate::Bitrate320 => [
                AudioFileFormat::OGG_VORBIS_320,
                AudioFileFormat::MP3_320,
                AudioFileFormat::MP3_256,
                AudioFileFormat::OGG_VORBIS_160,
                AudioFileFormat::MP3_160,
                AudioFileFormat::OGG_VORBIS_96,
                AudioFileFormat::MP3_96,
            ],
        };

        let (format, file_id) =
            match formats
                .iter()
                .find_map(|format| match audio_item.files.get(format) {
                    Some(&file_id) => Some((*format, file_id)),
                    _ => None,
                }) {
                Some(t) => t,
                None => {
                    warn!(
                        "<{}> is not available in any supported format",
                        audio_item.name
                    );
                    return None;
                }
            };

        let bytes_per_second = self.stream_data_rate(format)?;

        // This is only a loop to be able to reload the file if an error occurred
        // while opening a cached file.
        loop {
            let encrypted_file = AudioFile::open(&self.session, file_id, bytes_per_second);

            let encrypted_file = match encrypted_file.await {
                Ok(encrypted_file) => encrypted_file,
                Err(e) => {
                    error!("Unable to load encrypted file: {e:?}");
                    return None;
                }
            };

            let is_cached = encrypted_file.is_cached();

            let stream_loader_controller = encrypted_file.get_stream_loader_controller().ok()?;

            // Not all audio files are encrypted. If we can't get a key, try loading the track
            // without decryption. If the file was encrypted after all, the decoder will fail
            // parsing and bail out, so we should be safe from outputting ear-piercing noise.
            let key = match self.session.audio_key().request(track_id, file_id).await {
                Ok(key) => Some(key),
                Err(e) => {
                    warn!("Unable to load key, continuing without decryption: {e}");
                    None
                }
            };

            let mut decrypted_file = AudioDecrypt::new(key, encrypted_file);

            let is_ogg_vorbis = AudioFiles::is_ogg_vorbis(format);
            let (offset, mut normalisation_data) = if is_ogg_vorbis {
                // Spotify stores normalisation data in a custom Ogg packet instead of Vorbis comments.
                let normalisation_data =
                    NormalisationData::parse_from_ogg(&mut decrypted_file).ok();
                (SPOTIFY_OGG_HEADER_END, normalisation_data)
            } else {
                (0, None)
            };

            let audio_file = match Subfile::new(
                decrypted_file,
                offset,
                stream_loader_controller.len() as u64,
            ) {
                Ok(audio_file) => audio_file,
                Err(e) => {
                    error!("PlayerTrackLoader::load_track error opening subfile: {e}");
                    return None;
                }
            };

            let mut symphonia_decoder = |audio_file, format| {
                SymphoniaDecoder::new(audio_file, format).map(|mut decoder| {
                    // For formats other that Vorbis, we'll try getting normalisation data from
                    // ReplayGain metadata fields, if present.
                    if normalisation_data.is_none() {
                        normalisation_data = decoder.normalisation_data();
                    }
                    Box::new(decoder) as Decoder
                })
            };

            let mut hint = Hint::new();
            if let Some(mime_type) = AudioFiles::mime_type(format) {
                hint.mime_type(mime_type);
            }

            #[cfg(feature = "passthrough-decoder")]
            let decoder_type = if self.config.passthrough {
                PassthroughDecoder::new(audio_file, format).map(|x| Box::new(x) as Decoder)
            } else {
                symphonia_decoder(audio_file, hint)
            };

            #[cfg(not(feature = "passthrough-decoder"))]
            let decoder_type = { symphonia_decoder(audio_file, hint) };

            let normalisation_data = normalisation_data.unwrap_or_else(|| {
                warn!("Unable to get normalisation data, continuing with defaults.");
                NormalisationData::default()
            });

            let mut decoder = match decoder_type {
                Ok(decoder) => decoder,
                Err(e) if is_cached => {
                    warn!("Unable to read cached audio file: {e}. Trying to download it.");

                    match self.session.cache() {
                        Some(cache) => {
                            if cache.remove_file(file_id).is_err() {
                                error!("Error removing file from cache");
                                return None;
                            }
                        }
                        None => {
                            error!("If the audio file is cached, a cache should exist");
                            return None;
                        }
                    }

                    // Just try it again
                    continue;
                }
                Err(e) => {
                    error!("Unable to read audio file: {e}");
                    return None;
                }
            };

            let duration_ms = audio_item.duration_ms;
            // Don't try to seek past the track's duration.
            // If the position is invalid just start from
            // the beginning of the track.
            let position_ms = if position_ms > duration_ms {
                warn!(
                    "Invalid start position of {position_ms} ms exceeds track's duration of {duration_ms} ms, starting track from the beginning"
                );
                0
            } else {
                position_ms
            };

            // Ensure the starting position. Even when we want to play from the beginning,
            // the cursor may have been moved by parsing normalisation data. This may not
            // matter for playback (but won't hurt either), but may be useful for the
            // passthrough decoder.
            #[cfg(feature = "mix-transitions")]
            let stream_position_ms = if let Some(target_frame) = exact_frame {
                let max_frames = (u64::from(duration_ms) * u64::from(SAMPLE_RATE)) / 1000;
                let clamped = if max_frames > 0 {
                    target_frame.min(max_frames)
                } else {
                    target_frame
                };
                match decoder.seek_exact(clamped) {
                    Ok(actual_frame) => ((actual_frame * 1000) / u64::from(SAMPLE_RATE)) as u32,
                    Err(e) => {
                        error!(
                            "PlayerTrackLoader::load_track error exact-seeking to frame {clamped}: {e}"
                        );
                        return None;
                    }
                }
            } else {
                match decoder.seek(position_ms) {
                    Ok(new_position_ms) => new_position_ms,
                    Err(e) => {
                        error!(
                            "PlayerTrackLoader::load_track error seeking to starting position {position_ms}: {e}"
                        );
                        return None;
                    }
                }
            };
            #[cfg(not(feature = "mix-transitions"))]
            let stream_position_ms = match decoder.seek(position_ms) {
                Ok(new_position_ms) => new_position_ms,
                Err(e) => {
                    error!(
                        "PlayerTrackLoader::load_track error seeking to starting position {position_ms}: {e}"
                    );
                    return None;
                }
            };

            // Ensure streaming mode now that we are ready to play from the requested position.
            stream_loader_controller.set_stream_mode();

            let is_explicit = audio_item.is_explicit;

            info!(
                "<{}> from <{}> ({} ms) loaded",
                audio_item.name,
                audio_item.unique_fields.artists_display(),
                duration_ms
            );

            return Some(PlayerLoadedTrackData {
                decoder,
                normalisation_data,
                stream_loader_controller,
                audio_item,
                bytes_per_second,
                duration_ms,
                stream_position_ms,
                is_explicit,
            });
        }
    }

    async fn load_local_track(
        &self,
        track_uri: SpotifyUri,
        position_ms: u32,
        #[cfg(feature = "mix-transitions")] exact_frame: Option<u64>,
    ) -> Option<PlayerLoadedTrackData> {
        info!("Loading local file with Spotify URI <{}>", track_uri);

        let SpotifyUri::Local { duration, .. } = track_uri else {
            error!("Unable to determine track duration for local file: not a local file URI");
            return None;
        };

        let entry = self.local_file_lookup.get(&track_uri);

        let Some(path) = entry else {
            error!("Unable to find file path for local file <{track_uri}>");
            return None;
        };

        let src = match File::open(path) {
            Ok(src) => src,
            Err(e) => {
                error!("Failed to open local file: {e}");
                return None;
            }
        };

        let mut hint = Hint::new();
        if let Some(file_extension) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(file_extension);
        }

        let decoder = match SymphoniaDecoder::new(src, hint) {
            Ok(decoder) => decoder,
            Err(e) => {
                error!("Error decoding local file: {e}");
                return None;
            }
        };

        let mut decoder = Box::new(decoder);
        let normalisation_data = decoder.normalisation_data().unwrap_or_else(|| {
            warn!("Unable to get normalisation data, continuing with defaults.");
            NormalisationData::default()
        });

        let local_file_metadata = decoder.local_file_metadata().unwrap_or_default();

        #[cfg(feature = "mix-transitions")]
        let stream_position_ms = if let Some(target_frame) = exact_frame {
            match decoder.seek_exact(target_frame) {
                Ok(actual_frame) => ((actual_frame * 1000) / u64::from(SAMPLE_RATE)) as u32,
                Err(e) => {
                    error!(
                        "PlayerTrackLoader::load_local_track error exact-seeking to frame {target_frame}: {e}"
                    );
                    return None;
                }
            }
        } else {
            match decoder.seek(position_ms) {
                Ok(new_position_ms) => new_position_ms,
                Err(e) => {
                    error!(
                        "PlayerTrackLoader::load_local_track error seeking to starting position {position_ms}: {e}"
                    );
                    return None;
                }
            }
        };
        #[cfg(not(feature = "mix-transitions"))]
        let stream_position_ms = match decoder.seek(position_ms) {
            Ok(new_position_ms) => new_position_ms,
            Err(e) => {
                error!(
                    "PlayerTrackLoader::load_local_track error seeking to starting position {position_ms}: {e}"
                );
                return None;
            }
        };

        let file_size = fs::metadata(path).ok()?.len();
        let bytes_per_second = (file_size / duration.as_secs()) as usize;

        let stream_loader_controller = StreamLoaderController::from_local_file(file_size);

        let name = local_file_metadata.name.unwrap_or_default();

        info!("Loaded <{name}> from path <{}>", path.display());

        Some(PlayerLoadedTrackData {
            decoder,
            normalisation_data,
            stream_loader_controller,
            bytes_per_second,
            duration_ms: duration.as_millis() as u32,
            stream_position_ms,
            is_explicit: false,
            audio_item: AudioItem {
                duration_ms: duration.as_millis() as u32,
                uri: track_uri.to_uri(),
                track_id: track_uri,
                files: Default::default(),
                name,
                // We can't get a CoverImage.URL for the track image, applications will have to parse the file metadata themselves using unique_fields.path
                covers: vec![],
                language: local_file_metadata
                    .language
                    .map(|val| vec![val])
                    .unwrap_or_default(),
                is_explicit: false,
                availability: Ok(()),
                alternatives: None,
                unique_fields: UniqueFields::Local {
                    artists: local_file_metadata.artists,
                    album: local_file_metadata.album,
                    album_artists: local_file_metadata.album_artists,
                    number: local_file_metadata.number,
                    disc_number: local_file_metadata.disc_number,
                    path: path.to_path_buf(),
                },
            },
        })
    }
}

impl Future for PlayerInternal {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // While this is written as a future, it still contains blocking code.
        // It must be run on its own thread.
        let passthrough = self.config.passthrough;

        loop {
            let mut all_futures_completed_or_not_ready = true;

            // process commands that were sent to us
            let cmd = match self.commands.poll_recv(cx) {
                Poll::Ready(None) => return Poll::Ready(()), // client has disconnected - shut down.
                Poll::Ready(Some(cmd)) => {
                    all_futures_completed_or_not_ready = false;
                    Some(cmd)
                }
                _ => None,
            };

            if let Some(cmd) = cmd {
                if let Err(e) = self.handle_command(cmd) {
                    error!("Error handling command: {e}");
                }
            }

            // Handle loading of a new track to play
            if let PlayerState::Loading {
                ref mut loader,
                ref track_id,
                start_playback,
                play_request_id,
            } = self.state
            {
                // The loader may be terminated if we are trying to load the same track
                // as before, and that track failed to open before.
                let track_id = track_id.clone();

                if !loader.as_mut().is_terminated() {
                    match loader.as_mut().poll(cx) {
                        Poll::Ready(Ok(loaded_track)) => {
                            self.start_playback(
                                track_id,
                                play_request_id,
                                loaded_track,
                                start_playback,
                            );
                            if let PlayerState::Loading { .. } = self.state {
                                error!("The state wasn't changed by start_playback()");
                                exit(1);
                            }
                        }
                        Poll::Ready(Err(e)) => {
                            error!(
                                "Skipping to next track, unable to load track <{track_id:?}>: {e:?}"
                            );
                            self.send_event(PlayerEvent::Unavailable {
                                track_id,
                                play_request_id,
                            })
                        }
                        Poll::Pending => (),
                    }
                }
            }

            // handle pending preload requests.
            if let PlayerPreload::Loading {
                ref mut loader,
                ref track_id,
                #[cfg(feature = "mix-transitions")]
                start_frame,
            } = self.preload
            {
                let track_id = track_id.clone();
                match loader.as_mut().poll(cx) {
                    Poll::Ready(Ok(loaded_track)) => {
                        self.send_event(PlayerEvent::Preloading {
                            track_id: track_id.clone(),
                        });
                        self.preload = PlayerPreload::Ready {
                            track_id,
                            #[cfg(feature = "mix-transitions")]
                            start_frame,
                            loaded_track: Box::new(loaded_track),
                        };
                    }
                    Poll::Ready(Err(_)) => {
                        debug!("Unable to preload {track_id:?}");
                        self.preload = PlayerPreload::None;
                        #[cfg(feature = "mix-transitions")]
                        if self
                            .scheduled_transition
                            .as_ref()
                            .is_some_and(|s| s.next_track_id == track_id)
                        {
                            self.scheduled_transition = None;
                        }
                        // Let Spirc know that the track was unavailable.
                        if let PlayerState::Playing {
                            play_request_id, ..
                        }
                        | PlayerState::Paused {
                            play_request_id, ..
                        } = self.state
                        {
                            self.send_event(PlayerEvent::Unavailable {
                                track_id,
                                play_request_id,
                            });
                        }
                    }
                    Poll::Pending => (),
                }
            }

            if self.state.is_playing() {
                self.ensure_sink_running();

                #[cfg(feature = "mix-transitions")]
                if matches!(self.state, PlayerState::Transitioning(_)) {
                    self.step_active_transition();
                    continue;
                }

                #[cfg(feature = "mix-transitions")]
                if self.drain_pending_normalized_pcm() {
                    continue;
                }

                if let PlayerState::Playing {
                    ref track_id,
                    play_request_id,
                    ref mut decoder,
                    normalisation_factor,
                    ref mut stream_position_ms,
                    ref mut reported_nominal_start_time,
                    ..
                } = self.state
                {
                    let track_id = track_id.clone();
                    match decoder.next_packet() {
                        Ok(result) => {
                            #[cfg(feature = "mix-transitions")]
                            let decoder_end_frame = decoder.position_frames();

                            if let Some((ref packet_position, ref packet)) = result {
                                let new_stream_position_ms = packet_position.position_ms;
                                let expected_position_ms = std::mem::replace(
                                    &mut *stream_position_ms,
                                    new_stream_position_ms,
                                );

                                if !passthrough {
                                    match packet.samples() {
                                        Ok(_) => {
                                            let new_stream_position = Duration::from_millis(
                                                new_stream_position_ms as u64,
                                            );

                                            let now = Instant::now();

                                            // Only notify if we're skipped some packets *or* we are behind.
                                            // If we're ahead it's probably due to a buffer of the backend
                                            // and we're actually in time.
                                            let notify_about_position =
                                                match *reported_nominal_start_time {
                                                    None => true,
                                                    Some(reported_nominal_start_time) => {
                                                        let mut notify = false;

                                                        if packet_position.skipped {
                                                            if let Some(ahead) = new_stream_position
                                                                .checked_sub(Duration::from_millis(
                                                                    expected_position_ms as u64,
                                                                ))
                                                            {
                                                                notify |=
                                                                    ahead >= Duration::from_secs(1)
                                                            }
                                                        }

                                                        if let Some(lag) = now
                                                            .checked_duration_since(
                                                                reported_nominal_start_time,
                                                            )
                                                        {
                                                            if let Some(lag) =
                                                                lag.checked_sub(new_stream_position)
                                                            {
                                                                notify |=
                                                                    lag >= Duration::from_secs(1)
                                                            }
                                                        }

                                                        notify
                                                    }
                                                };

                                            if notify_about_position {
                                                *reported_nominal_start_time =
                                                    now.checked_sub(new_stream_position);
                                                self.send_event(PlayerEvent::PositionCorrection {
                                                    play_request_id,
                                                    track_id: track_id.clone(),
                                                    position_ms: new_stream_position_ms,
                                                });
                                            }

                                            if let Some(interval) =
                                                self.config.position_update_interval
                                            {
                                                let last_progress_update_since_ms =
                                                    now.duration_since(self.last_progress_update);

                                                if last_progress_update_since_ms > interval {
                                                    self.last_progress_update = now;
                                                    self.send_event(PlayerEvent::PositionChanged {
                                                        play_request_id,
                                                        track_id,
                                                        position_ms: new_stream_position_ms,
                                                    });
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            error!(
                                                "Skipping to next track, unable to decode samples for track <{track_id:?}>: {e:?}"
                                            );
                                            self.send_event(PlayerEvent::EndOfTrack {
                                                track_id,
                                                play_request_id,
                                            })
                                        }
                                    }
                                }
                            }

                            #[cfg(feature = "mix-transitions")]
                            if !passthrough
                                && self.scheduled_transition.is_some()
                                && self.maybe_enter_scheduled_transition(
                                    &result,
                                    decoder_end_frame,
                                    normalisation_factor,
                                )
                            {
                                continue;
                            }

                            self.handle_packet(result, normalisation_factor);
                        }
                        Err(e) => {
                            error!(
                                "Skipping to next track, unable to get next packet for track <{track_id:?}>: {e:?}"
                            );
                            self.send_event(PlayerEvent::EndOfTrack {
                                track_id,
                                play_request_id,
                            })
                        }
                    }
                } else {
                    error!("PlayerInternal poll: Invalid PlayerState");
                    exit(1);
                };
            }

            #[cfg(feature = "mix-transitions")]
            self.maybe_preload_scheduled_transition();

            if let PlayerState::Playing {
                ref track_id,
                play_request_id,
                duration_ms,
                stream_position_ms,
                ref mut stream_loader_controller,
                ref mut suggested_to_preload_next_track,
                ..
            }
            | PlayerState::Paused {
                ref track_id,
                play_request_id,
                duration_ms,
                stream_position_ms,
                ref mut stream_loader_controller,
                ref mut suggested_to_preload_next_track,
                ..
            } = self.state
            {
                let track_id = track_id.clone();

                if (!*suggested_to_preload_next_track)
                    && ((duration_ms as i64 - stream_position_ms as i64)
                        < PRELOAD_NEXT_TRACK_BEFORE_END_DURATION_MS as i64)
                    && stream_loader_controller.range_to_end_available()
                {
                    *suggested_to_preload_next_track = true;
                    self.send_event(PlayerEvent::TimeToPreloadNextTrack {
                        track_id,
                        play_request_id,
                    });
                }
            }

            if (!self.state.is_playing()) && all_futures_completed_or_not_ready {
                return Poll::Pending;
            }
        }
    }
}

impl PlayerInternal {
    fn ensure_sink_running(&mut self) {
        if self.sink_status != SinkStatus::Running {
            trace!("== Starting sink ==");
            if let Some(callback) = &mut self.sink_event_callback {
                callback(SinkStatus::Running);
            }
            match self.sink.start() {
                Ok(()) => self.sink_status = SinkStatus::Running,
                Err(e) => {
                    error!("{e}");
                    self.handle_pause();
                }
            }
        }
    }

    fn ensure_sink_stopped(&mut self, temporarily: bool) {
        match self.sink_status {
            SinkStatus::Running => {
                trace!("== Stopping sink ==");
                match self.sink.stop() {
                    Ok(()) => {
                        self.sink_status = if temporarily {
                            SinkStatus::TemporarilyClosed
                        } else {
                            SinkStatus::Closed
                        };
                        if let Some(callback) = &mut self.sink_event_callback {
                            callback(self.sink_status);
                        }
                    }
                    Err(e) => {
                        error!("{e}");
                        exit(1);
                    }
                }
            }
            SinkStatus::TemporarilyClosed => {
                if !temporarily {
                    self.sink_status = SinkStatus::Closed;
                    if let Some(callback) = &mut self.sink_event_callback {
                        callback(SinkStatus::Closed);
                    }
                }
            }
            SinkStatus::Closed => (),
        }
    }

    fn handle_player_stop(&mut self) {
        #[cfg(feature = "mix-transitions")]
        {
            self.scheduled_transition = None;
            if matches!(self.state, PlayerState::Transitioning(_)) {
                self.drain_active_transition_abort_fade();
                if let PlayerState::Transitioning(ref active) = self.state {
                    let meta = active.reported_meta();
                    let track_id = meta.track_id.clone();
                    let play_request_id = meta.play_request_id;
                    self.ensure_sink_stopped(false);
                    self.send_event(PlayerEvent::Stopped {
                        track_id,
                        play_request_id,
                    });
                    self.state = PlayerState::Stopped;
                    return;
                }
            }
        }

        match self.state {
            PlayerState::Playing {
                ref track_id,
                play_request_id,
                ..
            }
            | PlayerState::Paused {
                ref track_id,
                play_request_id,
                ..
            }
            | PlayerState::EndOfTrack {
                ref track_id,
                play_request_id,
                ..
            }
            | PlayerState::Loading {
                ref track_id,
                play_request_id,
                ..
            } => {
                let track_id = track_id.clone();

                self.ensure_sink_stopped(false);
                self.send_event(PlayerEvent::Stopped {
                    track_id,
                    play_request_id,
                });
                self.state = PlayerState::Stopped;
            }
            PlayerState::Stopped => (),
            #[cfg(feature = "mix-transitions")]
            PlayerState::Transitioning(_) => (),
            PlayerState::Invalid => {
                error!("PlayerInternal::handle_player_stop in invalid state");
                exit(1);
            }
        }
    }

    fn handle_play(&mut self) {
        match self.state {
            PlayerState::Paused {
                ref track_id,
                play_request_id,
                stream_position_ms,
                ..
            } => {
                let track_id = track_id.clone();

                self.state.paused_to_playing();
                self.send_event(PlayerEvent::Playing {
                    track_id,
                    play_request_id,
                    position_ms: stream_position_ms,
                });
                self.ensure_sink_running();
            }
            #[cfg(feature = "mix-transitions")]
            PlayerState::Transitioning(ref mut active) if active.is_paused() => {
                active.resume();
                let meta = active.reported_meta();
                let track_id = meta.track_id.clone();
                let play_request_id = meta.play_request_id;
                let position_ms = active.reported_position_ms();
                self.send_event(PlayerEvent::Playing {
                    track_id,
                    play_request_id,
                    position_ms,
                });
                self.ensure_sink_running();
            }
            PlayerState::Loading {
                ref mut start_playback,
                ..
            } => {
                *start_playback = true;
            }
            _ => error!("Player::play called from invalid state: {:?}", self.state),
        }
    }

    fn handle_pause(&mut self) {
        match self.state {
            PlayerState::Paused { .. } => self.ensure_sink_stopped(false),
            PlayerState::Playing {
                ref track_id,
                play_request_id,
                stream_position_ms,
                ..
            } => {
                let track_id = track_id.clone();

                self.state.playing_to_paused();

                self.ensure_sink_stopped(false);
                self.send_event(PlayerEvent::Paused {
                    track_id,
                    play_request_id,
                    position_ms: stream_position_ms,
                });
            }
            #[cfg(feature = "mix-transitions")]
            PlayerState::Transitioning(ref mut active) => {
                if active.is_paused() {
                    self.ensure_sink_stopped(false);
                } else {
                    active.pause();
                    let meta = active.reported_meta();
                    let track_id = meta.track_id.clone();
                    let play_request_id = meta.play_request_id;
                    let position_ms = active.reported_position_ms();
                    self.ensure_sink_stopped(false);
                    self.send_event(PlayerEvent::Paused {
                        track_id,
                        play_request_id,
                        position_ms,
                    });
                }
            }
            PlayerState::Loading {
                ref mut start_playback,
                ..
            } => {
                *start_playback = false;
            }
            _ => error!("Player::pause called from invalid state: {:?}", self.state),
        }
    }

    fn handle_packet(
        &mut self,
        packet: Option<(AudioPacketPosition, AudioPacket)>,
        normalisation_factor: f64,
    ) {
        match packet {
            Some((_, mut packet)) => {
                if !packet.is_empty() {
                    if let AudioPacket::Samples(ref mut data) = packet {
                        // Get the volume for the packet. In the case of hardware volume control
                        // this will always be 1.0 (no change).
                        let volume = self.volume_getter.attenuation_factor();

                        apply_track_normalisation(data, &self.config, normalisation_factor);
                        self.master_limiter
                            .process_post_mix(data, &self.config, volume);
                    }

                    if let Err(e) = self.sink.write(packet, &mut self.converter) {
                        error!("{e}");
                        self.handle_pause();
                    }
                }
            }

            None => {
                self.state.playing_to_end_of_track();
                if let PlayerState::EndOfTrack {
                    ref track_id,
                    play_request_id,
                    ..
                } = self.state
                {
                    self.send_event(PlayerEvent::EndOfTrack {
                        track_id: track_id.clone(),
                        play_request_id,
                    })
                } else {
                    error!("PlayerInternal handle_packet: Invalid PlayerState");
                    exit(1);
                }
            }
        }
    }

    fn compute_normalisation_factor(&self, data: NormalisationData) -> f64 {
        let mut config = self.config.clone();
        if config.normalisation_type == NormalisationType::Auto {
            if self.auto_normalise_as_album {
                config.normalisation_type = NormalisationType::Album;
            } else {
                config.normalisation_type = NormalisationType::Track;
            }
        }
        NormalisationData::get_factor(&config, data)
    }

    fn start_playback(
        &mut self,
        track_id: SpotifyUri,
        play_request_id: u64,
        loaded_track: PlayerLoadedTrackData,
        start_playback: bool,
    ) {
        let audio_item = Box::new(loaded_track.audio_item.clone());

        self.send_event(PlayerEvent::TrackChanged { audio_item });

        let position_ms = loaded_track.stream_position_ms;
        let normalisation_factor =
            self.compute_normalisation_factor(loaded_track.normalisation_data);

        if start_playback {
            self.ensure_sink_running();
            self.send_event(PlayerEvent::Playing {
                track_id: track_id.clone(),
                play_request_id,
                position_ms,
            });

            self.state = PlayerState::Playing {
                track_id,
                play_request_id,
                decoder: loaded_track.decoder,
                audio_item: loaded_track.audio_item,
                normalisation_data: loaded_track.normalisation_data,
                normalisation_factor,
                stream_loader_controller: loaded_track.stream_loader_controller,
                duration_ms: loaded_track.duration_ms,
                bytes_per_second: loaded_track.bytes_per_second,
                stream_position_ms: loaded_track.stream_position_ms,
                reported_nominal_start_time: Instant::now()
                    .checked_sub(Duration::from_millis(position_ms as u64)),
                suggested_to_preload_next_track: false,
                is_explicit: loaded_track.is_explicit,
                #[cfg(feature = "mix-transitions")]
                pending_normalized_pcm: VecDeque::new(),
            };
        } else {
            self.ensure_sink_stopped(false);

            self.state = PlayerState::Paused {
                track_id: track_id.clone(),
                play_request_id,
                decoder: loaded_track.decoder,
                audio_item: loaded_track.audio_item,
                normalisation_data: loaded_track.normalisation_data,
                normalisation_factor,
                stream_loader_controller: loaded_track.stream_loader_controller,
                duration_ms: loaded_track.duration_ms,
                bytes_per_second: loaded_track.bytes_per_second,
                stream_position_ms: loaded_track.stream_position_ms,
                suggested_to_preload_next_track: false,
                is_explicit: loaded_track.is_explicit,
                #[cfg(feature = "mix-transitions")]
                pending_normalized_pcm: VecDeque::new(),
            };

            self.send_event(PlayerEvent::Paused {
                track_id,
                play_request_id,
                position_ms,
            });
        }
    }

    fn handle_command_load(
        &mut self,
        track_id: SpotifyUri,
        play_request_id_option: Option<u64>,
        play: bool,
        position_ms: u32,
    ) -> PlayerResult {
        #[cfg(feature = "mix-transitions")]
        let aborted_matching_deck = {
            self.scheduled_transition = None;
            if matches!(self.state, PlayerState::Transitioning(_)) {
                self.drain_active_transition_abort_fade();
                if let PlayerState::Transitioning(active) =
                    mem::replace(&mut self.state, PlayerState::Stopped)
                {
                    active.into_matching_deck(|m| m.track_id == track_id)
                } else {
                    None
                }
            } else {
                None
            }
        };

        let play_request_id =
            play_request_id_option.unwrap_or(self.play_request_id_generator.get());

        self.send_event(PlayerEvent::PlayRequestIdChanged { play_request_id });

        if !self.config.gapless {
            self.ensure_sink_stopped(play);
        }

        #[cfg(feature = "mix-transitions")]
        if let Some(mut deck) = aborted_matching_deck {
            let stream_position_ms = match deck.decoder.seek(position_ms) {
                Ok(pos) => pos,
                Err(e) => {
                    self.send_event(PlayerEvent::Unavailable {
                        track_id,
                        play_request_id,
                    });
                    return Err(e.into());
                }
            };
            let meta = deck.meta;
            let loaded_track = PlayerLoadedTrackData {
                decoder: deck.decoder,
                normalisation_data: meta.normalisation_data,
                stream_loader_controller: meta.stream_loader_controller,
                audio_item: meta.audio_item,
                bytes_per_second: meta.bytes_per_second,
                duration_ms: meta.duration_ms,
                stream_position_ms,
                is_explicit: meta.is_explicit,
            };
            self.preload = PlayerPreload::None;
            self.start_playback(track_id, play_request_id, loaded_track, play);
            if let PlayerState::Invalid = self.state {
                return Err(Error::internal(format!(
                    "PlayerInternal::handle_command_load reusing transition deck: start_playback() did not transition to valid player state: {:?}",
                    self.state
                )));
            }
            return Ok(());
        }

        if matches!(self.state, PlayerState::Invalid) {
            return Err(Error::internal(format!(
                "Player::handle_command_load called from invalid state: {:?}",
                self.state
            )));
        }

        // Now we check at different positions whether we already have a pre-loaded version
        // of this track somewhere. If so, use it and return.

        // Check if there's a matching loaded track in the EndOfTrack player state.
        // This is the case if we're repeating the same track again.
        if let PlayerState::EndOfTrack {
            track_id: previous_track_id,
            ..
        } = &self.state
        {
            if *previous_track_id == track_id {
                let mut loaded_track = match mem::replace(&mut self.state, PlayerState::Invalid) {
                    PlayerState::EndOfTrack { loaded_track, .. } => loaded_track,
                    _ => {
                        return Err(Error::internal(format!(
                            "PlayerInternal::handle_command_load repeating the same track: invalid state: {:?}",
                            self.state
                        )));
                    }
                };

                if position_ms != loaded_track.stream_position_ms {
                    // This may be blocking.
                    loaded_track.stream_position_ms = loaded_track.decoder.seek(position_ms)?;
                }
                self.preload = PlayerPreload::None;
                self.start_playback(track_id, play_request_id, loaded_track, play);
                if let PlayerState::Invalid = self.state {
                    return Err(Error::internal(format!(
                        "PlayerInternal::handle_command_load repeating the same track: start_playback() did not transition to valid player state: {:?}",
                        self.state
                    )));
                }
                return Ok(());
            }
        }

        // Check if we are already playing the track. If so, just do a seek and update our info.
        if let PlayerState::Playing {
            track_id: ref current_track_id,
            ref mut stream_position_ms,
            ref mut decoder,
            ..
        }
        | PlayerState::Paused {
            track_id: ref current_track_id,
            ref mut stream_position_ms,
            ref mut decoder,
            ..
        } = self.state
        {
            if *current_track_id == track_id {
                // we can use the current decoder. Ensure it's at the correct position.
                if position_ms != *stream_position_ms {
                    // This may be blocking.
                    *stream_position_ms = decoder.seek(position_ms)?;
                }

                // Move the info from the current state into a PlayerLoadedTrackData so we can use
                // the usual code path to start playback.
                let old_state = mem::replace(&mut self.state, PlayerState::Invalid);

                if let PlayerState::Playing {
                    stream_position_ms,
                    decoder,
                    audio_item,
                    stream_loader_controller,
                    bytes_per_second,
                    duration_ms,
                    normalisation_data,
                    is_explicit,
                    ..
                }
                | PlayerState::Paused {
                    stream_position_ms,
                    decoder,
                    audio_item,
                    stream_loader_controller,
                    bytes_per_second,
                    duration_ms,
                    normalisation_data,
                    is_explicit,
                    ..
                } = old_state
                {
                    let loaded_track = PlayerLoadedTrackData {
                        decoder,
                        normalisation_data,
                        stream_loader_controller,
                        audio_item,
                        bytes_per_second,
                        duration_ms,
                        stream_position_ms,
                        is_explicit,
                    };

                    self.preload = PlayerPreload::None;
                    self.start_playback(track_id, play_request_id, loaded_track, play);

                    if let PlayerState::Invalid = self.state {
                        return Err(Error::internal(format!(
                            "PlayerInternal::handle_command_load already playing this track: start_playback() did not transition to valid player state: {:?}",
                            self.state
                        )));
                    }

                    return Ok(());
                } else {
                    return Err(Error::internal(format!(
                        "PlayerInternal::handle_command_load already playing this track: invalid state: {:?}",
                        self.state
                    )));
                }
            }
        }

        // Check if the requested track has been preloaded already. If so use the preloaded data.
        if let PlayerPreload::Ready {
            track_id: loaded_track_id,
            ..
        } = &self.preload
        {
            if track_id == *loaded_track_id {
                let preload = std::mem::replace(&mut self.preload, PlayerPreload::None);
                if let PlayerPreload::Ready {
                    track_id,
                    #[cfg(feature = "mix-transitions")]
                    start_frame,
                    mut loaded_track,
                } = preload
                {
                    #[cfg(feature = "mix-transitions")]
                    let needs_reseek =
                        position_ms != loaded_track.stream_position_ms || start_frame.is_some();
                    #[cfg(not(feature = "mix-transitions"))]
                    let needs_reseek = position_ms != loaded_track.stream_position_ms;

                    if needs_reseek {
                        // This may be blocking
                        loaded_track.stream_position_ms = loaded_track.decoder.seek(position_ms)?;
                    }
                    self.start_playback(track_id, play_request_id, *loaded_track, play);
                    return Ok(());
                } else {
                    return Err(Error::internal(format!(
                        "PlayerInternal::handle_command_loading preloaded track: invalid state: {:?}",
                        self.state
                    )));
                }
            }
        }

        self.send_event(PlayerEvent::Loading {
            track_id: track_id.clone(),
            play_request_id,
            position_ms,
        });

        // Try to extract a pending loader from the preloading mechanism
        let loader = if let PlayerPreload::Loading {
            track_id: loaded_track_id,
            #[cfg(feature = "mix-transitions")]
            start_frame,
            ..
        } = &self.preload
        {
            #[cfg(feature = "mix-transitions")]
            let starts_at_zero = start_frame.is_none() || *start_frame == Some(0);
            #[cfg(not(feature = "mix-transitions"))]
            let starts_at_zero = true;

            if (track_id == *loaded_track_id) && (position_ms == 0) && starts_at_zero {
                let mut preload = PlayerPreload::None;
                std::mem::swap(&mut preload, &mut self.preload);
                if let PlayerPreload::Loading { loader, .. } = preload {
                    Some(loader)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        self.preload = PlayerPreload::None;

        // If we don't have a loader yet, create one from scratch.
        let loader = loader.unwrap_or_else(|| {
            Box::pin(self.load_track(
                track_id.clone(),
                position_ms,
                #[cfg(feature = "mix-transitions")]
                None,
            ))
        });

        // Set ourselves to a loading state.
        self.state = PlayerState::Loading {
            track_id,
            play_request_id,
            start_playback: play,
            loader,
        };

        Ok(())
    }

    fn handle_command_preload(&mut self, track_id: SpotifyUri) {
        debug!("Preloading track");
        let mut preload_track = true;
        // check whether the track is already loaded somewhere or being loaded.
        if let PlayerPreload::Loading {
            track_id: currently_loading,
            ..
        }
        | PlayerPreload::Ready {
            track_id: currently_loading,
            ..
        } = &self.preload
        {
            if *currently_loading == track_id {
                // we're already preloading the requested track.
                preload_track = false;
            } else {
                // we're preloading something else - cancel it.
                self.preload = PlayerPreload::None;
            }
        }

        if let PlayerState::Playing {
            track_id: current_track_id,
            ..
        }
        | PlayerState::Paused {
            track_id: current_track_id,
            ..
        }
        | PlayerState::EndOfTrack {
            track_id: current_track_id,
            ..
        } = &self.state
        {
            if *current_track_id == track_id {
                // we already have the requested track loaded.
                preload_track = false;
            }
        }

        // schedule the preload of the current track if desired.
        if preload_track {
            let loader = self.load_track(
                track_id.clone(),
                0,
                #[cfg(feature = "mix-transitions")]
                None,
            );
            self.preload = PlayerPreload::Loading {
                track_id,
                #[cfg(feature = "mix-transitions")]
                start_frame: None,
                loader: Box::pin(loader),
            }
        }
    }

    fn handle_command_seek(&mut self, position_ms: u32) -> PlayerResult {
        #[cfg(feature = "mix-transitions")]
        if matches!(self.state, PlayerState::Transitioning(_)) {
            self.drain_active_transition_abort_fade();
            if let PlayerState::Transitioning(active) =
                mem::replace(&mut self.state, PlayerState::Invalid)
            {
                let was_paused = active.is_paused();
                let (deck, restored) = active.into_abort_for_seek();
                if let Some((mut incoming_deck, spec)) = restored {
                    let outgoing_track_id = deck.meta.track_id.clone();
                    let next_track_id = incoming_deck.meta.track_id.clone();
                    let target_b_frame = spec.start_b_frame;
                    self.scheduled_transition = Some(ScheduledTransition {
                        outgoing_track_id,
                        next_track_id: next_track_id.clone(),
                        spec,
                        missed_entry_logged: false,
                    });
                    if let Ok(actual_frame) = incoming_deck.decoder.seek_exact(target_b_frame) {
                        if actual_frame == target_b_frame {
                            let imeta = incoming_deck.meta;
                            let stream_pos_ms =
                                ((actual_frame * 1000) / u64::from(SAMPLE_RATE)) as u32;
                            self.preload = PlayerPreload::Ready {
                                track_id: next_track_id,
                                start_frame: Some(target_b_frame),
                                loaded_track: Box::new(PlayerLoadedTrackData {
                                    decoder: incoming_deck.decoder,
                                    normalisation_data: imeta.normalisation_data,
                                    stream_loader_controller: imeta.stream_loader_controller,
                                    audio_item: imeta.audio_item,
                                    bytes_per_second: imeta.bytes_per_second,
                                    duration_ms: imeta.duration_ms,
                                    stream_position_ms: stream_pos_ms,
                                    is_explicit: imeta.is_explicit,
                                }),
                            };
                        }
                    }
                }
                let meta = deck.meta;
                self.state = if was_paused {
                    PlayerState::Paused {
                        track_id: meta.track_id,
                        play_request_id: meta.play_request_id,
                        decoder: deck.decoder,
                        audio_item: meta.audio_item,
                        normalisation_data: meta.normalisation_data,
                        normalisation_factor: deck.normalisation_factor,
                        stream_loader_controller: meta.stream_loader_controller,
                        bytes_per_second: meta.bytes_per_second,
                        duration_ms: meta.duration_ms,
                        stream_position_ms: position_ms,
                        suggested_to_preload_next_track: false,
                        is_explicit: meta.is_explicit,
                        pending_normalized_pcm: VecDeque::new(),
                    }
                } else {
                    PlayerState::Playing {
                        track_id: meta.track_id,
                        play_request_id: meta.play_request_id,
                        decoder: deck.decoder,
                        audio_item: meta.audio_item,
                        normalisation_data: meta.normalisation_data,
                        normalisation_factor: deck.normalisation_factor,
                        stream_loader_controller: meta.stream_loader_controller,
                        bytes_per_second: meta.bytes_per_second,
                        duration_ms: meta.duration_ms,
                        stream_position_ms: position_ms,
                        reported_nominal_start_time: Instant::now()
                            .checked_sub(Duration::from_millis(position_ms as u64)),
                        suggested_to_preload_next_track: false,
                        is_explicit: meta.is_explicit,
                        pending_normalized_pcm: VecDeque::new(),
                    }
                };
            }
        }

        // When we are still loading, the user may immediately ask to
        // seek to another position yet the decoder won't be ready for
        // that. In this case just restart the loading process but
        // with the requested position.
        if let PlayerState::Loading {
            ref track_id,
            play_request_id,
            start_playback,
            ..
        } = self.state
        {
            return self.handle_command_load(
                track_id.clone(),
                Some(play_request_id),
                start_playback,
                position_ms,
            );
        }

        if let Some(decoder) = self.state.decoder() {
            match decoder.seek(position_ms) {
                Ok(new_position_ms) => {
                    if let PlayerState::Playing {
                        ref mut stream_position_ms,
                        ref track_id,
                        play_request_id,
                        #[cfg(feature = "mix-transitions")]
                        ref mut pending_normalized_pcm,
                        ..
                    }
                    | PlayerState::Paused {
                        ref mut stream_position_ms,
                        ref track_id,
                        play_request_id,
                        #[cfg(feature = "mix-transitions")]
                        ref mut pending_normalized_pcm,
                        ..
                    } = self.state
                    {
                        *stream_position_ms = new_position_ms;
                        #[cfg(feature = "mix-transitions")]
                        pending_normalized_pcm.clear();

                        self.send_event(PlayerEvent::Seeked {
                            play_request_id,
                            track_id: track_id.clone(),
                            position_ms: new_position_ms,
                        });
                    }
                    #[cfg(feature = "mix-transitions")]
                    {
                        if let Some(ref mut sched) = self.scheduled_transition {
                            let seek_frame =
                                (u64::from(new_position_ms) * u64::from(SAMPLE_RATE)) / 1000;
                            if seek_frame
                                <= sched
                                    .spec
                                    .start_a_frame
                                    .saturating_add(crate::mix::ENTRY_GRACE_FRAMES)
                            {
                                // Keep transition armed across overshoot seeks so dragging past
                                // `start_a` and back before `start_a` still triggers the mix.
                                sched.missed_entry_logged = false;
                            }
                        }
                        self.maybe_preload_scheduled_transition();
                    }
                }
                Err(e) => error!("PlayerInternal::handle_command_seek error: {e}"),
            }
        } else {
            error!("Player::seek called from invalid state: {:?}", self.state);
        }

        // ensure we have a bit of a buffer of downloaded data
        self.preload_data_before_playback()?;

        if let PlayerState::Playing {
            ref mut reported_nominal_start_time,
            ..
        } = self.state
        {
            *reported_nominal_start_time =
                Instant::now().checked_sub(Duration::from_millis(position_ms as u64));
        }

        Ok(())
    }

    fn handle_command(&mut self, cmd: PlayerCommand) -> PlayerResult {
        debug!("command={cmd:?}");
        match cmd {
            PlayerCommand::Load {
                track_id,
                play,
                position_ms,
            } => self.handle_command_load(track_id, None, play, position_ms)?,

            PlayerCommand::Preload { track_id } => self.handle_command_preload(track_id),

            #[cfg(feature = "mix-transitions")]
            PlayerCommand::ScheduleTransition {
                outgoing_track_id,
                next_track_id,
                spec,
            } => self.handle_command_schedule_transition(outgoing_track_id, next_track_id, *spec),

            #[cfg(feature = "mix-transitions")]
            PlayerCommand::CancelTransition => {
                self.handle_command_cancel_transition();
            }

            PlayerCommand::Seek(position_ms) => self.handle_command_seek(position_ms)?,

            PlayerCommand::Play => self.handle_play(),

            PlayerCommand::Pause => self.handle_pause(),

            PlayerCommand::Stop => self.handle_player_stop(),

            PlayerCommand::SetSession(session) => self.session = session,

            PlayerCommand::AddEventSender(sender) => self.event_senders.push(sender),

            PlayerCommand::SetSinkEventCallback(callback) => self.sink_event_callback = callback,

            PlayerCommand::EmitVolumeChangedEvent(volume) => {
                self.send_event(PlayerEvent::VolumeChanged { volume })
            }

            PlayerCommand::EmitRepeatChangedEvent { context, track } => {
                self.send_event(PlayerEvent::RepeatChanged { context, track })
            }

            PlayerCommand::EmitShuffleChangedEvent(shuffle) => {
                self.send_event(PlayerEvent::ShuffleChanged { shuffle })
            }

            PlayerCommand::EmitAutoPlayChangedEvent(auto_play) => {
                self.send_event(PlayerEvent::AutoPlayChanged { auto_play })
            }

            PlayerCommand::EmitSessionClientChangedEvent {
                client_id,
                client_name,
                client_brand_name,
                client_model_name,
            } => self.send_event(PlayerEvent::SessionClientChanged {
                client_id,
                client_name,
                client_brand_name,
                client_model_name,
            }),

            PlayerCommand::EmitSessionConnectedEvent {
                connection_id,
                user_name,
            } => self.send_event(PlayerEvent::SessionConnected {
                connection_id,
                user_name,
            }),

            PlayerCommand::EmitSessionDisconnectedEvent {
                connection_id,
                user_name,
            } => self.send_event(PlayerEvent::SessionDisconnected {
                connection_id,
                user_name,
            }),

            PlayerCommand::SetAutoNormaliseAsAlbum(setting) => {
                self.auto_normalise_as_album = setting
            }

            PlayerCommand::EmitSetQueueEvent {
                context_uri,
                current_track,
                next_tracks,
                prev_tracks,
            } => self.send_event(PlayerEvent::SetQueue {
                context_uri,
                current_track,
                next_tracks,
                prev_tracks,
            }),

            PlayerCommand::EmitFilterExplicitContentChangedEvent(filter) => {
                self.send_event(PlayerEvent::FilterExplicitContentChanged { filter });

                if filter {
                    if let PlayerState::Playing {
                        ref track_id,
                        play_request_id,
                        is_explicit,
                        ..
                    }
                    | PlayerState::Paused {
                        ref track_id,
                        play_request_id,
                        is_explicit,
                        ..
                    } = self.state
                    {
                        let track_id = track_id.clone();

                        if is_explicit {
                            warn!(
                                "Currently loaded track is explicit, which client setting forbids -- skipping to next track."
                            );
                            self.send_event(PlayerEvent::EndOfTrack {
                                track_id,
                                play_request_id,
                            })
                        }
                    }
                }
            }
        };

        Ok(())
    }

    fn send_event(&mut self, event: PlayerEvent) {
        self.event_senders
            .retain(|sender| sender.send(event.clone()).is_ok());
    }

    fn load_track(
        &mut self,
        spotify_uri: SpotifyUri,
        position_ms: u32,
        #[cfg(feature = "mix-transitions")] exact_frame: Option<u64>,
    ) -> impl FusedFuture<Output = Result<PlayerLoadedTrackData, ()>> + Send + 'static {
        // This method creates a future that returns the loaded stream and associated info.
        // Ideally all work should be done using asynchronous code. However, seek() on the
        // audio stream is implemented in a blocking fashion. Thus, we can't turn it into future
        // easily. Instead we spawn a thread to do the work and return a one-shot channel as the
        // future to work with.

        let loader = PlayerTrackLoader {
            session: self.session.clone(),
            config: self.config.clone(),
            local_file_lookup: self.local_file_lookup.clone(),
        };

        let (result_tx, result_rx) = oneshot::channel();

        let load_handles_clone = self.load_handles.clone();
        let handle = tokio::runtime::Handle::current();

        let load_handle = thread::spawn(move || {
            let data = handle.block_on(loader.load_track(
                spotify_uri,
                position_ms,
                #[cfg(feature = "mix-transitions")]
                exact_frame,
            ));
            if let Some(data) = data {
                let _ = result_tx.send(data);
            }

            let mut load_handles = load_handles_clone.lock().expect(LOAD_HANDLES_POISON_MSG);
            load_handles.remove(&thread::current().id());
        });

        let mut load_handles = self.load_handles.lock().expect(LOAD_HANDLES_POISON_MSG);
        load_handles.insert(load_handle.thread().id(), load_handle);

        result_rx.map_err(|_| ())
    }

    #[cfg(feature = "mix-transitions")]
    fn current_active_track_id(&self) -> Option<&SpotifyUri> {
        match &self.state {
            PlayerState::Loading { track_id, .. }
            | PlayerState::Playing { track_id, .. }
            | PlayerState::Paused { track_id, .. } => Some(track_id),
            PlayerState::Transitioning(active) => Some(&active.reported_meta().track_id),
            _ => None,
        }
    }

    #[cfg(feature = "mix-transitions")]
    fn handle_command_cancel_transition(&mut self) {
        self.scheduled_transition = None;
        // If a pre-commit transition is currently running (ConnectState still displays Deck A),
        // abort back to outgoing Deck A with an anti-click fade so TransitionCommit(B) cannot
        // fire against a mutated queue/option state.
        let abort_pre_commit = matches!(
            &self.state,
            PlayerState::Transitioning(active) if !active.is_committed()
        );
        if abort_pre_commit {
            self.drain_active_transition_abort_fade();
            if let PlayerState::Transitioning(active) =
                mem::replace(&mut self.state, PlayerState::Invalid)
            {
                let was_paused = active.is_paused();
                let pos_ms = active.reported_position_ms();
                let (deck, _) = active.into_abort_for_seek();
                let meta = deck.meta;
                let track_id = meta.track_id.clone();
                let play_request_id = meta.play_request_id;
                self.state = if was_paused {
                    PlayerState::Paused {
                        track_id: meta.track_id,
                        play_request_id: meta.play_request_id,
                        decoder: deck.decoder,
                        audio_item: meta.audio_item,
                        normalisation_data: meta.normalisation_data,
                        normalisation_factor: deck.normalisation_factor,
                        stream_loader_controller: meta.stream_loader_controller,
                        bytes_per_second: meta.bytes_per_second,
                        duration_ms: meta.duration_ms,
                        stream_position_ms: pos_ms,
                        suggested_to_preload_next_track: false,
                        is_explicit: meta.is_explicit,
                        pending_normalized_pcm: VecDeque::new(),
                    }
                } else {
                    PlayerState::Playing {
                        track_id: meta.track_id,
                        play_request_id: meta.play_request_id,
                        decoder: deck.decoder,
                        audio_item: meta.audio_item,
                        normalisation_data: meta.normalisation_data,
                        normalisation_factor: deck.normalisation_factor,
                        stream_loader_controller: meta.stream_loader_controller,
                        bytes_per_second: meta.bytes_per_second,
                        duration_ms: meta.duration_ms,
                        stream_position_ms: pos_ms,
                        reported_nominal_start_time: Instant::now()
                            .checked_sub(Duration::from_millis(u64::from(pos_ms))),
                        suggested_to_preload_next_track: false,
                        is_explicit: meta.is_explicit,
                        pending_normalized_pcm: VecDeque::new(),
                    }
                };
                self.send_event(PlayerEvent::PositionCorrection {
                    play_request_id,
                    track_id,
                    position_ms: pos_ms,
                });
            }
        }
    }

    #[cfg(feature = "mix-transitions")]
    fn handle_command_schedule_transition(
        &mut self,
        outgoing_track_id: SpotifyUri,
        next_track_id: SpotifyUri,
        spec: TransitionSpec,
    ) {
        if self.config.passthrough {
            debug!("Ignoring scheduled transition because passthrough mode is enabled");
            return;
        }
        if self.current_active_track_id() != Some(&outgoing_track_id) {
            debug!(
                "Ignoring stale scheduled transition <{outgoing_track_id}> -> <{next_track_id}> (active track mismatch)"
            );
            return;
        }
        debug!(
            "Scheduled mix transition to <{next_track_id}> at frame {} -> {}",
            spec.start_a_frame, spec.start_b_frame
        );
        self.scheduled_transition = Some(ScheduledTransition {
            outgoing_track_id,
            next_track_id,
            spec,
            missed_entry_logged: false,
        });
        self.maybe_preload_scheduled_transition();
    }

    #[cfg(feature = "mix-transitions")]
    fn maybe_preload_scheduled_transition(&mut self) {
        let Some(ref sched) = self.scheduled_transition else {
            return;
        };
        let (stream_position_ms, duration_ms) = match &self.state {
            PlayerState::Playing {
                stream_position_ms,
                duration_ms,
                ..
            }
            | PlayerState::Paused {
                stream_position_ms,
                duration_ms,
                ..
            } => (*stream_position_ms, *duration_ms),
            _ => return,
        };

        let start_a_ms = ((sched.spec.start_a_frame * 1000) / u64::from(SAMPLE_RATE))
            .min(u64::from(duration_ms)) as i64;
        let grace_ms = ((crate::mix::ENTRY_GRACE_FRAMES * 1000) / u64::from(SAMPLE_RATE)) as i64;
        let delta_ms = start_a_ms - i64::from(stream_position_ms);
        if delta_ms >= i64::from(PRELOAD_NEXT_TRACK_BEFORE_END_DURATION_MS) || delta_ms < -grace_ms
        {
            return;
        }

        let target_id = sched.next_track_id.clone();
        let target_frame = sched.spec.start_b_frame;

        let already_matching = match &self.preload {
            PlayerPreload::Loading {
                track_id,
                start_frame,
                ..
            }
            | PlayerPreload::Ready {
                track_id,
                start_frame,
                ..
            } => {
                *track_id == target_id
                    && (*start_frame == Some(target_frame)
                        || (target_frame == 0 && start_frame.is_none()))
            }
            PlayerPreload::None => false,
        };
        if already_matching {
            return;
        }

        // Reuse an already-Ready preload for the same track by exact-seeking in place instead of
        // re-downloading the track over the network.
        if let PlayerPreload::Ready {
            track_id,
            start_frame,
            loaded_track,
        } = &mut self.preload
        {
            if *track_id == target_id {
                if let Ok(actual) = loaded_track.decoder.seek_exact(target_frame) {
                    if actual == target_frame {
                        *start_frame = Some(target_frame);
                        loaded_track.stream_position_ms =
                            ((actual * 1000) / u64::from(SAMPLE_RATE)) as u32;
                        return;
                    }
                }
            }
        }

        let loader = self.load_track(target_id.clone(), 0, Some(target_frame));
        self.preload = PlayerPreload::Loading {
            track_id: target_id,
            start_frame: Some(target_frame),
            loader: Box::pin(loader),
        };
    }

    #[cfg(feature = "mix-transitions")]
    fn drain_pending_normalized_pcm(&mut self) -> bool {
        let channels = usize::from(NUM_CHANNELS);
        let max_samples = 1024 * channels;

        let chunk = if let PlayerState::Playing {
            ref mut pending_normalized_pcm,
            ref mut stream_position_ms,
            ..
        } = self.state
        {
            if pending_normalized_pcm.is_empty() {
                return false;
            }
            let take = pending_normalized_pcm.len().min(max_samples);
            let chunk: Vec<f64> = pending_normalized_pcm.drain(..take).collect();
            let advance_ms = ((take / channels) as u64 * 1000 / u64::from(SAMPLE_RATE)) as u32;
            *stream_position_ms = stream_position_ms.saturating_add(advance_ms);
            chunk
        } else {
            return false;
        };

        let mut data = chunk;
        let volume = self.volume_getter.attenuation_factor();
        self.master_limiter
            .process_post_mix(&mut data, &self.config, volume);
        if let Err(e) = self
            .sink
            .write(AudioPacket::Samples(data), &mut self.converter)
        {
            error!("{e}");
            self.handle_pause();
        }
        true
    }

    #[cfg(feature = "mix-transitions")]
    fn maybe_enter_scheduled_transition(
        &mut self,
        result: &Option<(AudioPacketPosition, AudioPacket)>,
        decoder_end_frame: Option<u64>,
        normalisation_factor: f64,
    ) -> bool {
        let Some((packet_position, AudioPacket::Samples(samples))) = result else {
            return false;
        };
        let Some(ref mut sched) = self.scheduled_transition else {
            return false;
        };
        let channels = usize::from(NUM_CHANNELS);
        let pkt_frames = (samples.len() / channels) as u64;
        let pkt_end_frame = decoder_end_frame.unwrap_or_else(|| {
            ((u64::from(packet_position.position_ms) * u64::from(SAMPLE_RATE)) / 1000)
                .saturating_add(pkt_frames)
        });
        let pkt_start_frame = pkt_end_frame.saturating_sub(pkt_frames);

        let dry_prefix_samples = match split_entry_packet(
            samples.len(),
            pkt_start_frame,
            sched.spec.start_a_frame,
        ) {
            EntryPacketDecision::BeforeEntry => return false,
            EntryPacketDecision::PastGraceWindow => {
                if !sched.missed_entry_logged {
                    warn!(
                        "Scheduled transition to <{}> missed entry window at frame {}; falling back to gapless EOF unless scrubbed back",
                        sched.next_track_id, sched.spec.start_a_frame
                    );
                    sched.missed_entry_logged = true;
                }
                return false;
            }
            EntryPacketDecision::CrossesEntry { dry_prefix_samples } => dry_prefix_samples,
            EntryPacketDecision::WithinGraceWindow => 0,
        };

        let preload_ready = matches!(
            &self.preload,
            PlayerPreload::Ready { track_id, .. } if *track_id == sched.next_track_id
        );
        if !preload_ready {
            // Keep playing Deck A dry while preload finishes (still inside grace window).
            return false;
        }

        let sched = self.scheduled_transition.take().unwrap();
        let PlayerPreload::Ready {
            track_id: incoming_track_id,
            start_frame,
            mut loaded_track,
        } = mem::replace(&mut self.preload, PlayerPreload::None)
        else {
            return false;
        };

        let already_at_target = start_frame == Some(sched.spec.start_b_frame)
            || (sched.spec.start_b_frame == 0 && start_frame.is_none());
        if !already_at_target {
            if let Err(e) = loaded_track.decoder.seek_exact(sched.spec.start_b_frame) {
                warn!(
                    "Failed to exact-seek incoming track <{incoming_track_id}> to frame {} ({e}); falling back to gapless EOF",
                    sched.spec.start_b_frame
                );
                self.preload = PlayerPreload::Ready {
                    track_id: incoming_track_id,
                    start_frame,
                    loaded_track,
                };
                return false;
            }
        }

        // Play the dry prefix of Deck A up to exact frame `start_a_frame`.
        if dry_prefix_samples > 0 {
            let mut prefix = samples[..dry_prefix_samples].to_vec();
            let volume = self.volume_getter.attenuation_factor();
            apply_track_normalisation(&mut prefix, &self.config, normalisation_factor);
            self.master_limiter
                .process_post_mix(&mut prefix, &self.config, volume);
            if let Err(e) = self
                .sink
                .write(AudioPacket::Samples(prefix), &mut self.converter)
            {
                error!("{e}");
                let target_b_frame = sched.spec.start_b_frame;
                self.handle_pause();
                self.scheduled_transition = Some(sched);
                self.preload = PlayerPreload::Ready {
                    track_id: incoming_track_id,
                    start_frame: Some(target_b_frame),
                    loaded_track,
                };
                return false;
            }
        }

        let mut suffix = samples[dry_prefix_samples..].to_vec();
        apply_track_normalisation(&mut suffix, &self.config, normalisation_factor);

        let PlayerState::Playing {
            track_id: outgoing_track_id,
            play_request_id: outgoing_play_request_id,
            decoder: outgoing_decoder,
            audio_item: outgoing_audio_item,
            normalisation_data: outgoing_norm_data,
            normalisation_factor: outgoing_norm_factor,
            stream_loader_controller: outgoing_slc,
            bytes_per_second: outgoing_bps,
            duration_ms: outgoing_duration_ms,
            is_explicit: outgoing_is_explicit,
            ..
        } = mem::replace(&mut self.state, PlayerState::Invalid)
        else {
            error!("PlayerInternal::maybe_enter_scheduled_transition in non-playing state");
            exit(1);
        };

        let incoming_play_request_id = self.play_request_id_generator.get();
        let incoming_norm_factor =
            self.compute_normalisation_factor(loaded_track.normalisation_data);

        let outgoing_deck = TransitionDeck {
            decoder: outgoing_decoder,
            normalisation_factor: outgoing_norm_factor,
            meta: TransitionDeckMeta {
                track_id: outgoing_track_id.clone(),
                play_request_id: outgoing_play_request_id,
                audio_item: outgoing_audio_item,
                normalisation_data: outgoing_norm_data,
                stream_loader_controller: outgoing_slc,
                bytes_per_second: outgoing_bps,
                duration_ms: outgoing_duration_ms,
                is_explicit: outgoing_is_explicit,
            },
        };
        let incoming_deck = TransitionDeck {
            decoder: loaded_track.decoder,
            normalisation_factor: incoming_norm_factor,
            meta: TransitionDeckMeta {
                track_id: incoming_track_id.clone(),
                play_request_id: incoming_play_request_id,
                audio_item: loaded_track.audio_item,
                normalisation_data: loaded_track.normalisation_data,
                stream_loader_controller: loaded_track.stream_loader_controller,
                bytes_per_second: loaded_track.bytes_per_second,
                duration_ms: loaded_track.duration_ms,
                is_explicit: loaded_track.is_explicit,
            },
        };

        let active = ActiveTransition::new(outgoing_deck, incoming_deck, sched.spec, &suffix);
        self.state = PlayerState::Transitioning(Box::new(active));
        self.send_event(PlayerEvent::TransitionStarted {
            outgoing_play_request_id,
            outgoing_track_id,
            incoming_track_id,
        });
        true
    }

    #[cfg(feature = "mix-transitions")]
    fn emit_transition_commit(&mut self, incoming_position_ms: u32) {
        let PlayerState::Transitioning(ref active) = self.state else {
            return;
        };
        let outgoing_play_request_id = active.outgoing_meta().play_request_id;
        let incoming = active.incoming_meta();
        let incoming_track_id = incoming.track_id.clone();
        let incoming_play_request_id = incoming.play_request_id;
        let incoming_audio_item = Box::new(incoming.audio_item.clone());

        self.send_event(PlayerEvent::TransitionCommit {
            outgoing_play_request_id,
            incoming_track_id: incoming_track_id.clone(),
            incoming_play_request_id,
            incoming_position_ms,
        });
        self.send_event(PlayerEvent::PlayRequestIdChanged {
            play_request_id: incoming_play_request_id,
        });
        self.send_event(PlayerEvent::TrackChanged {
            audio_item: incoming_audio_item,
        });
        self.send_event(PlayerEvent::Playing {
            play_request_id: incoming_play_request_id,
            track_id: incoming_track_id,
            position_ms: incoming_position_ms,
        });
    }

    #[cfg(feature = "mix-transitions")]
    fn step_active_transition(&mut self) {
        let volume = self.volume_getter.attenuation_factor();
        let (step, reported_pos_ms, reported_track_id, reported_play_request_id) = {
            let PlayerState::Transitioning(ref mut active) = self.state else {
                return;
            };
            let step = active.step(&self.config, &mut self.master_limiter, volume, 512);
            let meta = active.reported_meta();
            (
                step,
                active.reported_position_ms(),
                meta.track_id.clone(),
                meta.play_request_id,
            )
        };

        if !step.samples.is_empty() {
            if let Err(e) = self
                .sink
                .write(AudioPacket::Samples(step.samples), &mut self.converter)
            {
                error!("{e}");
                self.handle_pause();
                return;
            }
        }

        if let Some(incoming_pos_ms) = step.handoff_due_ms {
            self.emit_transition_commit(incoming_pos_ms);
        }

        if let Some(interval) = self.config.position_update_interval {
            let now = Instant::now();
            if now.duration_since(self.last_progress_update) > interval {
                self.last_progress_update = now;
                self.send_event(PlayerEvent::PositionChanged {
                    play_request_id: reported_play_request_id,
                    track_id: reported_track_id,
                    position_ms: reported_pos_ms,
                });
            }
        }

        if step.completed {
            let PlayerState::Transitioning(active) =
                mem::replace(&mut self.state, PlayerState::Invalid)
            else {
                return;
            };
            let (incoming_deck, next_frame, continuation) = active.into_incoming_continuation();
            let meta = incoming_deck.meta;
            let pos_ms = ((next_frame * 1000) / u64::from(SAMPLE_RATE)) as u32;
            let exit_track_id = meta.track_id.clone();
            let exit_play_request_id = meta.play_request_id;
            self.state = PlayerState::Playing {
                track_id: meta.track_id,
                play_request_id: meta.play_request_id,
                decoder: incoming_deck.decoder,
                audio_item: meta.audio_item,
                normalisation_data: meta.normalisation_data,
                normalisation_factor: incoming_deck.normalisation_factor,
                stream_loader_controller: meta.stream_loader_controller,
                bytes_per_second: meta.bytes_per_second,
                duration_ms: meta.duration_ms,
                stream_position_ms: pos_ms,
                reported_nominal_start_time: Instant::now()
                    .checked_sub(Duration::from_millis(u64::from(pos_ms))),
                suggested_to_preload_next_track: false,
                is_explicit: meta.is_explicit,
                pending_normalized_pcm: VecDeque::from(continuation),
            };
            // Drop any stale scheduled transition that belonged to an earlier outgoing track.
            if self
                .scheduled_transition
                .as_ref()
                .is_some_and(|s| s.outgoing_track_id != exit_track_id)
            {
                self.scheduled_transition = None;
            }
            // Re-sync Spirc's wall-clock position estimate if tempo stretching drifted >100 ms
            // during the post-commit half of the transition.
            self.send_event(PlayerEvent::PositionCorrection {
                play_request_id: exit_play_request_id,
                track_id: exit_track_id,
                position_ms: pos_ms,
            });
        }
    }

    #[cfg(feature = "mix-transitions")]
    fn drain_active_transition_abort_fade(&mut self) {
        let volume = self.volume_getter.attenuation_factor();
        let fade = {
            let PlayerState::Transitioning(ref mut active) = self.state else {
                return;
            };
            active.abort_and_drain(&self.config, &mut self.master_limiter, volume)
        };
        if !fade.is_empty() && self.sink_status == SinkStatus::Running {
            if let Err(e) = self
                .sink
                .write(AudioPacket::Samples(fade), &mut self.converter)
            {
                warn!("Sink write error during transition abort fade: {e}");
            }
        }
    }

    fn preload_data_before_playback(&mut self) -> PlayerResult {
        if let PlayerState::Playing {
            bytes_per_second,
            ref mut stream_loader_controller,
            ..
        } = self.state
        {
            let read_ahead_during_playback = AudioFetchParams::get().read_ahead_during_playback;
            // Request our read ahead range
            let request_data_length =
                (read_ahead_during_playback.as_secs_f32() * bytes_per_second as f32) as usize;

            // Request the part we want to wait for blocking. This effectively means we wait for the previous request to partially complete.
            let wait_for_data_length =
                (read_ahead_during_playback.as_secs_f32() * bytes_per_second as f32) as usize;

            stream_loader_controller.fetch_next_and_wait(request_data_length, wait_for_data_length)
        } else {
            Ok(())
        }
    }
}

impl Drop for PlayerInternal {
    fn drop(&mut self) {
        debug!("drop PlayerInternal[{}]", self.player_id);

        let handles: Vec<thread::JoinHandle<()>> = {
            // waiting for the thread while holding the mutex would result in a deadlock
            let mut load_handles = self.load_handles.lock().expect(LOAD_HANDLES_POISON_MSG);

            load_handles
                .drain()
                .map(|(_thread_id, handle)| handle)
                .collect()
        };

        for handle in handles {
            let _ = handle.join();
        }
    }
}

impl fmt::Debug for PlayerCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlayerCommand::Load {
                track_id,
                play,
                position_ms,
                ..
            } => f
                .debug_tuple("Load")
                .field(&track_id)
                .field(&play)
                .field(&position_ms)
                .finish(),
            PlayerCommand::Preload { track_id } => {
                f.debug_tuple("Preload").field(&track_id).finish()
            }
            PlayerCommand::Play => f.debug_tuple("Play").finish(),
            PlayerCommand::Pause => f.debug_tuple("Pause").finish(),
            PlayerCommand::Stop => f.debug_tuple("Stop").finish(),
            PlayerCommand::Seek(position) => f.debug_tuple("Seek").field(&position).finish(),
            PlayerCommand::SetSession(_) => f.debug_tuple("SetSession").finish(),
            PlayerCommand::AddEventSender(_) => f.debug_tuple("AddEventSender").finish(),
            PlayerCommand::SetSinkEventCallback(_) => {
                f.debug_tuple("SetSinkEventCallback").finish()
            }
            PlayerCommand::EmitVolumeChangedEvent(volume) => f
                .debug_tuple("EmitVolumeChangedEvent")
                .field(&volume)
                .finish(),
            PlayerCommand::SetAutoNormaliseAsAlbum(setting) => f
                .debug_tuple("SetAutoNormaliseAsAlbum")
                .field(&setting)
                .finish(),
            PlayerCommand::EmitFilterExplicitContentChangedEvent(filter) => f
                .debug_tuple("EmitFilterExplicitContentChangedEvent")
                .field(&filter)
                .finish(),
            PlayerCommand::EmitSessionConnectedEvent {
                connection_id,
                user_name,
            } => f
                .debug_tuple("EmitSessionConnectedEvent")
                .field(&connection_id)
                .field(&user_name)
                .finish(),
            PlayerCommand::EmitSessionDisconnectedEvent {
                connection_id,
                user_name,
            } => f
                .debug_tuple("EmitSessionDisconnectedEvent")
                .field(&connection_id)
                .field(&user_name)
                .finish(),
            PlayerCommand::EmitSessionClientChangedEvent {
                client_id,
                client_name,
                client_brand_name,
                client_model_name,
            } => f
                .debug_tuple("EmitSessionClientChangedEvent")
                .field(&client_id)
                .field(&client_name)
                .field(&client_brand_name)
                .field(&client_model_name)
                .finish(),
            PlayerCommand::EmitShuffleChangedEvent(shuffle) => f
                .debug_tuple("EmitShuffleChangedEvent")
                .field(&shuffle)
                .finish(),
            PlayerCommand::EmitRepeatChangedEvent { context, track } => f
                .debug_tuple("EmitRepeatChangedEvent")
                .field(&context)
                .field(&track)
                .finish(),
            PlayerCommand::EmitAutoPlayChangedEvent(auto_play) => f
                .debug_tuple("EmitAutoPlayChangedEvent")
                .field(&auto_play)
                .finish(),
            PlayerCommand::EmitSetQueueEvent {
                context_uri,
                next_tracks,
                prev_tracks,
                ..
            } => f
                .debug_tuple("EmitSetQueueEvent")
                .field(&context_uri)
                .field(&next_tracks.len())
                .field(&prev_tracks.len())
                .finish(),
            #[cfg(feature = "mix-transitions")]
            PlayerCommand::ScheduleTransition { next_track_id, .. } => f
                .debug_tuple("ScheduleTransition")
                .field(&next_track_id)
                .finish(),
            #[cfg(feature = "mix-transitions")]
            PlayerCommand::CancelTransition => f.debug_tuple("CancelTransition").finish(),
        }
    }
}

impl fmt::Debug for PlayerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PlayerState::*;
        match self {
            Stopped => f.debug_struct("Stopped").finish(),
            Loading {
                track_id,
                play_request_id,
                ..
            } => f
                .debug_struct("Loading")
                .field("track_id", &track_id)
                .field("play_request_id", &play_request_id)
                .finish(),
            Paused {
                track_id,
                play_request_id,
                ..
            } => f
                .debug_struct("Paused")
                .field("track_id", &track_id)
                .field("play_request_id", &play_request_id)
                .finish(),
            Playing {
                track_id,
                play_request_id,
                ..
            } => f
                .debug_struct("Playing")
                .field("track_id", &track_id)
                .field("play_request_id", &play_request_id)
                .finish(),
            #[cfg(feature = "mix-transitions")]
            Transitioning(active) => f
                .debug_struct("Transitioning")
                .field("outgoing", &active.outgoing_meta().track_id)
                .field("incoming", &active.incoming_meta().track_id)
                .field("committed", &active.is_committed())
                .field("paused", &active.is_paused())
                .finish(),
            EndOfTrack {
                track_id,
                play_request_id,
                ..
            } => f
                .debug_struct("EndOfTrack")
                .field("track_id", &track_id)
                .field("play_request_id", &play_request_id)
                .finish(),
            Invalid => f.debug_struct("Invalid").finish(),
        }
    }
}

struct Subfile<T: Read + Seek> {
    stream: T,
    offset: u64,
    length: u64,
}

impl<T: Read + Seek> Subfile<T> {
    pub fn new(mut stream: T, offset: u64, length: u64) -> Result<Subfile<T>, io::Error> {
        let target = SeekFrom::Start(offset);
        stream.seek(target)?;

        Ok(Subfile {
            stream,
            offset,
            length,
        })
    }
}

impl<T: Read + Seek> Read for Subfile<T> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stream.read(buf)
    }
}

impl<T: Read + Seek> Seek for Subfile<T> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let pos = match pos {
            SeekFrom::Start(offset) => SeekFrom::Start(offset + self.offset),
            SeekFrom::End(offset) => {
                if (self.length as i64 - offset) < self.offset as i64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "newpos would be < self.offset",
                    ));
                }
                pos
            }
            _ => pos,
        };

        let newpos = self.stream.seek(pos)?;
        Ok(newpos - self.offset)
    }
}

impl<R> MediaSource for Subfile<R>
where
    R: Read + Seek + Send + Sync,
{
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SAMPLE_RATE;

    #[test]
    fn split_dynamic_limiter_matches_fused_reference_bit_for_bit() {
        let config = PlayerConfig {
            normalisation: true,
            normalisation_method: NormalisationMethod::Dynamic,
            ..Default::default()
        };
        let factor = db_to_ratio(-3.5);
        let volume = 0.75;

        let input: Vec<f64> = (0..4096)
            .map(|i| {
                let t = (i / 2) as f64 / f64::from(SAMPLE_RATE);
                let freq = if i % 2 == 0 { 440.0 } else { 660.0 };
                1.4 * (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect();

        // Reference fused loop matching the pre-split implementation
        let mut reference = input.clone();
        let knee_factor = 1.0 / (8.0 * config.normalisation_knee_db);
        let mut integrators = [0.0_f64; 2];
        let mut peaks = [0.0_f64; 2];
        let mut channel = 0usize;
        for sample in &mut reference {
            *sample *= factor;
            let limiter_db = {
                *sample += f64::MIN_POSITIVE;
                let bias_db = ratio_to_db(sample.abs()) - config.normalisation_threshold_dbfs;
                let knee_boundary_db = bias_db * 2.0;
                if knee_boundary_db < -config.normalisation_knee_db {
                    0.0
                } else if knee_boundary_db.abs() <= config.normalisation_knee_db {
                    let term = knee_boundary_db + config.normalisation_knee_db;
                    term * term * knee_factor
                } else {
                    bias_db
                }
            };
            let ch = channel;
            channel ^= 1;
            integrators[ch] = f64::max(
                limiter_db,
                config.normalisation_release_cf * integrators[ch]
                    + (1.0 - config.normalisation_release_cf) * limiter_db,
            );
            peaks[ch] = config.normalisation_attack_cf * peaks[ch]
                + (1.0 - config.normalisation_attack_cf) * integrators[ch];
            let max_peak = f64::max(peaks[0], peaks[1]);
            *sample *= db_to_ratio(-max_peak) * volume;
        }

        let mut split = input;
        apply_track_normalisation(&mut split, &config, factor);
        let mut limiter = MasterLimiter::new(&config);
        limiter.process_post_mix(&mut split, &config, volume);

        assert_eq!(split.len(), reference.len());
        for (idx, (a, b)) in split.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "sample {idx} diverged: split={a} ref={b}"
            );
        }
    }

    #[test]
    fn post_mix_limiter_bounds_overlapping_full_scale_tracks() {
        let config = PlayerConfig {
            normalisation: true,
            normalisation_method: NormalisationMethod::Dynamic,
            normalisation_threshold_dbfs: -2.0,
            ..Default::default()
        };

        // Two overlapping tracks, each hitting 0 dBFS, summed into a +6 dBFS signal.
        let n_frames = SAMPLE_RATE as usize; // 1 second
        let mut deck_a = vec![0.0_f64; n_frames * 2];
        let mut deck_b = vec![0.0_f64; n_frames * 2];
        for i in 0..n_frames {
            let t = i as f64 / f64::from(SAMPLE_RATE);
            let a = (2.0 * std::f64::consts::PI * 100.0 * t).sin();
            let b = (2.0 * std::f64::consts::PI * 100.0 * t).sin();
            deck_a[2 * i] = a;
            deck_a[2 * i + 1] = a;
            deck_b[2 * i] = b;
            deck_b[2 * i + 1] = b;
        }

        apply_track_normalisation(&mut deck_a, &config, 1.0);
        apply_track_normalisation(&mut deck_b, &config, 1.0);

        let mut mixed: Vec<f64> = deck_a
            .iter()
            .zip(deck_b.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut limiter = MasterLimiter::new(&config);
        limiter.process_post_mix(&mut mixed, &config, 1.0);

        assert!(mixed.iter().all(|s| s.is_finite()));
        // After the 5 ms attack settles, steady-state peaks must sit near or below -2 dBFS.
        let settle_samples = (SAMPLE_RATE as usize / 20) * 2; // 50 ms
        let steady_peak = mixed[settle_samples..]
            .iter()
            .fold(0.0_f64, |acc, &s| acc.max(s.abs()));
        let steady_peak_db = ratio_to_db(steady_peak);
        assert!(
            steady_peak_db <= config.normalisation_threshold_dbfs + 0.5,
            "post-mix steady peak {steady_peak_db:.2} dBFS exceeded limiter ceiling"
        );
    }
}
