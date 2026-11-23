use std::ops::Deref;

use thiserror::Error;

#[cfg(feature = "passthrough-decoder")]
mod passthrough_decoder;
#[cfg(feature = "passthrough-decoder")]
pub use passthrough_decoder::PassthroughDecoder;

mod symphonia_decoder;
pub use symphonia_decoder::SymphoniaDecoder;

#[derive(Error, Debug)]
pub enum DecoderError {
    #[error("Passthrough Decoder Error: {0}")]
    PassthroughDecoder(String),
    #[error("Symphonia Decoder Error: {0}")]
    SymphoniaDecoder(String),
}

pub type DecoderResult<T> = Result<T, DecoderError>;

#[derive(Error, Debug)]
pub enum AudioPacketError {
    #[error("Decoder Raw Error: Can't return Raw on Samples")]
    Raw,
    #[error("Decoder Samples Error: Can't return Samples on Raw")]
    Samples,
}

pub type AudioPacketResult<T> = Result<T, AudioPacketError>;

pub enum AudioPacket {
    Samples(Vec<f64>),
    Raw(Vec<u8>),
}

impl AudioPacket {
    #[inline]
    pub fn samples(&self) -> AudioPacketResult<&[f64]> {
        match self {
            AudioPacket::Samples(s) => Ok(s),
            AudioPacket::Raw(_) => Err(AudioPacketError::Raw),
        }
    }

    #[inline]
    pub fn raw(&self) -> AudioPacketResult<&[u8]> {
        match self {
            AudioPacket::Raw(d) => Ok(d),
            AudioPacket::Samples(_) => Err(AudioPacketError::Samples),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            AudioPacket::Samples(s) => s.is_empty(),
            AudioPacket::Raw(d) => d.is_empty(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioPacketPosition {
    pub position_ms: u32,
    pub skipped: bool,
}

impl Deref for AudioPacketPosition {
    type Target = u32;
    fn deref(&self) -> &Self::Target {
        &self.position_ms
    }
}

pub trait AudioDecoder {
    fn seek(&mut self, position_ms: u32) -> Result<u32, DecoderError>;
    fn next_packet(&mut self) -> DecoderResult<Option<(AudioPacketPosition, AudioPacket)>>;

    /// Seek to an exact PCM frame index at [`crate::SAMPLE_RATE`].
    ///
    /// Returns the actual frame index positioned at. Decoders that do not
    /// support sample-accurate seeking (such as passthrough decoders) fall
    /// back to millisecond seeking.
    fn seek_exact(&mut self, target_frame: u64) -> Result<u64, DecoderError> {
        let position_ms = ((target_frame as f64) * crate::MS_PER_PAGE).round() as u32;
        let actual_ms = self.seek(position_ms)?;
        Ok(((actual_ms as f64) * crate::PAGES_PER_MS).round() as u64)
    }

    /// Current stream position in PCM frames at [`crate::SAMPLE_RATE`] (the
    /// frame index of the next sample to be returned by [`Self::next_packet`]).
    fn position_frames(&self) -> Option<u64> {
        None
    }
}

impl From<DecoderError> for librespot_core::error::Error {
    fn from(err: DecoderError) -> Self {
        librespot_core::error::Error::aborted(err)
    }
}

impl From<symphonia::core::errors::Error> for DecoderError {
    fn from(err: symphonia::core::errors::Error) -> Self {
        Self::SymphoniaDecoder(err.to_string())
    }
}
