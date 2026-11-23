use std::{io, time::Duration};

use symphonia::core::{
    audio::SampleBuffer,
    codecs::{Decoder, DecoderOptions},
    errors::Error,
    formats::{FormatOptions, SeekMode, SeekTo},
    io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions},
    meta::{MetadataOptions, StandardTagKey, Value},
    probe::{Hint, ProbeResult},
};

use super::{AudioDecoder, AudioPacket, AudioPacketPosition, DecoderError, DecoderResult};

use crate::{MS_PER_PAGE, NUM_CHANNELS, SAMPLE_RATE, player::NormalisationData, symphonia_util};

pub struct SymphoniaDecoder {
    probe_result: ProbeResult,
    decoder: Box<dyn Decoder>,
    sample_buffer: Option<SampleBuffer<f64>>,
    track_id: u32,
    current_frame: u64,
    buffered_packet: Option<(AudioPacketPosition, AudioPacket)>,
}

#[derive(Default)]
pub(crate) struct LocalFileMetadata {
    pub name: Option<String>,
    pub language: Option<String>,
    pub album: Option<String>,
    pub artists: Option<String>,
    pub album_artists: Option<String>,
    pub number: Option<u32>,
    pub disc_number: Option<u32>,
}

impl SymphoniaDecoder {
    pub fn new<R>(input: R, hint: Hint) -> DecoderResult<Self>
    where
        R: MediaSource + 'static,
    {
        let mss_opts = MediaSourceStreamOptions {
            buffer_len: librespot_audio::AudioFetchParams::get().minimum_download_size,
        };
        let mss = MediaSourceStream::new(Box::new(input), mss_opts);

        let format_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let metadata_opts: MetadataOptions = Default::default();

        let probe_result =
            symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        let format = &probe_result.format;

        let track = format.default_track().ok_or_else(|| {
            DecoderError::SymphoniaDecoder("Could not retrieve default track".into())
        })?;
        let track_id = track.id;

        let decoder_opts: DecoderOptions = Default::default();

        let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

        let rate = decoder.codec_params().sample_rate.ok_or_else(|| {
            DecoderError::SymphoniaDecoder("Could not retrieve sample rate".into())
        })?;

        // TODO: The official client supports local files with sample rates other than 44,100 kHz.
        // To play these accurately, we need to either resample the input audio, or introduce a way
        // to change the player's current sample rate (likely by closing and re-opening the sink
        // with new parameters).
        if rate != SAMPLE_RATE {
            return Err(DecoderError::SymphoniaDecoder(format!(
                "Unsupported sample rate: {rate}"
            )));
        }

        let channels = decoder.codec_params().channels.ok_or_else(|| {
            DecoderError::SymphoniaDecoder("Could not retrieve channel configuration".into())
        })?;
        if channels.count() != NUM_CHANNELS as usize {
            return Err(DecoderError::SymphoniaDecoder(format!(
                "Unsupported number of channels: {channels}"
            )));
        }

        Ok(Self {
            probe_result,
            decoder,
            // We set the sample buffer when decoding the first full packet,
            // whose duration is also the ideal sample buffer size.
            sample_buffer: None,
            track_id,
            current_frame: 0,
            buffered_packet: None,
        })
    }

    pub fn normalisation_data(&mut self) -> Option<NormalisationData> {
        let metadata = symphonia_util::get_latest_metadata(&mut self.probe_result)?;
        let tags = metadata.current()?.tags();

        if tags.is_empty() {
            None
        } else {
            let mut data = NormalisationData::default();

            for tag in tags {
                if let Value::Float(value) = tag.value {
                    match tag.std_key {
                        Some(StandardTagKey::ReplayGainAlbumGain) => data.album_gain_db = value,
                        Some(StandardTagKey::ReplayGainAlbumPeak) => data.album_peak = value,
                        Some(StandardTagKey::ReplayGainTrackGain) => data.track_gain_db = value,
                        Some(StandardTagKey::ReplayGainTrackPeak) => data.track_peak = value,
                        _ => (),
                    }
                }
            }

            Some(data)
        }
    }

    pub(crate) fn local_file_metadata(&mut self) -> Option<LocalFileMetadata> {
        let metadata = symphonia_util::get_latest_metadata(&mut self.probe_result)?;
        let tags = metadata.current()?.tags();
        let mut metadata = LocalFileMetadata::default();

        for tag in tags {
            if let Value::String(value) = &tag.value {
                match tag.std_key {
                    // We could possibly use mem::take here to avoid cloning, but that risks leaving
                    // the audio item metadata in a bad state.
                    Some(StandardTagKey::TrackTitle) => metadata.name = Some(value.clone()),
                    Some(StandardTagKey::Language) => metadata.language = Some(value.clone()),
                    Some(StandardTagKey::Artist) => metadata.artists = Some(value.clone()),
                    Some(StandardTagKey::AlbumArtist) => {
                        metadata.album_artists = Some(value.clone())
                    }
                    Some(StandardTagKey::Album) => metadata.album = Some(value.clone()),
                    Some(StandardTagKey::TrackNumber) => {
                        metadata.number = match value.parse::<u32>() {
                            Ok(value) => Some(value),
                            Err(e) => {
                                warn!(
                                    "Failed to parse local file's track number tag '{value}': {e}"
                                );
                                None
                            }
                        }
                    }
                    Some(StandardTagKey::DiscNumber) => {
                        metadata.disc_number = match value.parse::<u32>() {
                            Ok(value) => Some(value),
                            Err(e) => {
                                warn!(
                                    "Failed to parse local file's disc number tag '{value}': {e}"
                                );
                                None
                            }
                        }
                    }
                    _ => (),
                }
            } else if let Value::UnsignedInt(value) = &tag.value {
                match tag.std_key {
                    Some(StandardTagKey::TrackNumber) => metadata.number = Some(*value as u32),
                    Some(StandardTagKey::DiscNumber) => metadata.disc_number = Some(*value as u32),
                    _ => (),
                }
            } else if let Value::SignedInt(value) = &tag.value {
                match tag.std_key {
                    Some(StandardTagKey::TrackNumber) => metadata.number = Some(*value as u32),
                    Some(StandardTagKey::DiscNumber) => metadata.disc_number = Some(*value as u32),
                    _ => (),
                }
            }
        }

        Some(metadata)
    }

    #[inline]
    fn ts_to_ms(&self, ts: u64) -> u32 {
        match self.decoder.codec_params().time_base {
            Some(time_base) => {
                let time = Duration::from(time_base.calc_time(ts));
                time.as_millis() as u32
            }
            // Fallback in the unexpected case that the format has no base time set.
            None => (ts as f64 * MS_PER_PAGE) as u32,
        }
    }

    #[inline]
    fn ts_to_frames(&self, ts: u64) -> u64 {
        match self.decoder.codec_params().time_base {
            Some(time_base) if time_base.numer == 1 && time_base.denom == SAMPLE_RATE => ts,
            Some(time_base) => {
                let time = time_base.calc_time(ts);
                let whole = time.seconds.saturating_mul(u64::from(SAMPLE_RATE));
                let frac = (time.frac * f64::from(SAMPLE_RATE)).round() as u64;
                whole.saturating_add(frac)
            }
            None => ts,
        }
    }

    #[inline]
    fn frames_to_ms(frames: u64) -> u32 {
        ((u128::from(frames) * 1000) / u128::from(SAMPLE_RATE)) as u32
    }

    fn decode_next_raw(&mut self) -> DecoderResult<Option<(u64, bool, Vec<f64>)>> {
        let mut skipped = false;

        loop {
            let packet = match self.probe_result.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(err)) => {
                    if err.kind() == io::ErrorKind::UnexpectedEof {
                        return Ok(None);
                    } else {
                        return Err(DecoderError::SymphoniaDecoder(err.to_string()));
                    }
                }
                Err(err) => {
                    return Err(err.into());
                }
            };

            let start_frame = self.ts_to_frames(packet.ts());

            match self.decoder.decode(&packet) {
                Ok(decoded) => {
                    let sample_buffer = match self.sample_buffer.as_mut() {
                        Some(buffer) => buffer,
                        None => {
                            let spec = *decoded.spec();
                            let duration = decoded.capacity() as u64;
                            self.sample_buffer.insert(SampleBuffer::new(duration, spec))
                        }
                    };

                    sample_buffer.copy_interleaved_ref(decoded);
                    return Ok(Some((
                        start_frame,
                        skipped,
                        sample_buffer.samples().to_vec(),
                    )));
                }
                Err(Error::DecodeError(_)) => {
                    // The packet failed to decode due to corrupted or invalid data, get a new
                    // packet and try again.
                    warn!(
                        "Skipping malformed audio packet at {} ms",
                        Self::frames_to_ms(start_frame)
                    );
                    skipped = true;
                    continue;
                }
                Err(err) => return Err(err.into()),
            }
        }
    }
}

impl AudioDecoder for SymphoniaDecoder {
    fn seek(&mut self, position_ms: u32) -> Result<u32, DecoderError> {
        self.buffered_packet = None;

        // "Saturate" the position_ms to the duration of the track if it exceeds it.
        let mut target = Duration::from_millis(position_ms.into());
        let codec_params = self.decoder.codec_params();
        if let (Some(time_base), Some(n_frames)) = (codec_params.time_base, codec_params.n_frames) {
            let duration = Duration::from(time_base.calc_time(n_frames));
            if target > duration {
                target = duration;
            }
        }

        // `track_id: None` implies the default track ID (of the container, not of Spotify).
        let seeked_to_ts = self.probe_result.format.seek(
            SeekMode::Accurate,
            SeekTo::Time {
                time: target.into(),
                track_id: None,
            },
        )?;

        // Seeking is a `FormatReader` operation, so the decoder cannot reliably
        // know when a seek took place. Reset it to avoid audio glitches.
        self.decoder.reset();

        self.current_frame = self.ts_to_frames(seeked_to_ts.actual_ts);
        Ok(self.ts_to_ms(seeked_to_ts.actual_ts))
    }

    fn seek_exact(&mut self, mut target_frame: u64) -> Result<u64, DecoderError> {
        self.buffered_packet = None;

        let codec_params = self.decoder.codec_params();
        if let Some(n_frames) = codec_params.n_frames {
            let total_frames = self.ts_to_frames(n_frames);
            if target_frame > total_frames {
                target_frame = total_frames;
            }
        }

        let seek_to = match codec_params.time_base {
            Some(tb) if tb.numer == 1 && tb.denom == SAMPLE_RATE => SeekTo::TimeStamp {
                ts: target_frame,
                track_id: self.track_id,
            },
            _ => {
                let secs = target_frame / u64::from(SAMPLE_RATE);
                let rem = target_frame % u64::from(SAMPLE_RATE);
                let time =
                    symphonia::core::units::Time::new(secs, rem as f64 / f64::from(SAMPLE_RATE));
                SeekTo::Time {
                    time,
                    track_id: Some(self.track_id),
                }
            }
        };

        let seeked = self.probe_result.format.seek(SeekMode::Accurate, seek_to)?;
        self.decoder.reset();

        let mut last_frame = self.ts_to_frames(seeked.actual_ts);
        let channels = usize::from(NUM_CHANNELS);

        while let Some((pkt_start_frame, skipped, mut samples)) = self.decode_next_raw()? {
            let pkt_frames = (samples.len() / channels) as u64;
            if pkt_frames == 0 {
                continue;
            }
            let pkt_end_frame = pkt_start_frame.saturating_add(pkt_frames);
            last_frame = pkt_end_frame;

            if pkt_end_frame <= target_frame {
                continue;
            }

            let actual_frame = if pkt_start_frame < target_frame {
                let trim_frames = (target_frame - pkt_start_frame) as usize;
                samples.drain(..trim_frames * channels);
                target_frame
            } else {
                pkt_start_frame
            };

            let pos = AudioPacketPosition {
                position_ms: Self::frames_to_ms(actual_frame),
                skipped,
            };
            self.buffered_packet = Some((pos, AudioPacket::Samples(samples)));
            self.current_frame = actual_frame;
            return Ok(actual_frame);
        }

        self.current_frame = last_frame;
        Ok(last_frame)
    }

    fn position_frames(&self) -> Option<u64> {
        Some(self.current_frame)
    }

    fn next_packet(&mut self) -> DecoderResult<Option<(AudioPacketPosition, AudioPacket)>> {
        let channels = usize::from(NUM_CHANNELS);

        if let Some((pos, pkt)) = self.buffered_packet.take() {
            if let AudioPacket::Samples(ref s) = pkt {
                self.current_frame = self
                    .current_frame
                    .saturating_add((s.len() / channels) as u64);
            }
            return Ok(Some((pos, pkt)));
        }

        match self.decode_next_raw()? {
            Some((start_frame, skipped, samples)) => {
                let pkt_frames = (samples.len() / channels) as u64;
                self.current_frame = start_frame.saturating_add(pkt_frames);
                let packet_position = AudioPacketPosition {
                    position_ms: Self::frames_to_ms(start_frame),
                    skipped,
                };
                Ok(Some((packet_position, AudioPacket::Samples(samples))))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn flac_crc8(bytes: &[u8]) -> u8 {
        let mut crc: u8 = 0;
        for &b in bytes {
            crc ^= b;
            for _ in 0..8 {
                crc = if (crc & 0x80) != 0 {
                    (crc << 1) ^ 0x07
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    fn flac_crc16(bytes: &[u8]) -> u16 {
        let mut crc: u16 = 0;
        for &b in bytes {
            crc ^= u16::from(b) << 8;
            for _ in 0..8 {
                crc = if (crc & 0x8000) != 0 {
                    (crc << 1) ^ 0x8005
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    /// Build a valid 44.1 kHz, 16-bit stereo fixed-blocksize (4096) VERBATIM FLAC stream
    /// encoding `frames` (pairs of `(left, right)` i16 samples).
    fn encode_verbatim_flac(frames: &[(i16, i16)]) -> Vec<u8> {
        const BLOCK_SIZE: usize = 4096;
        assert_eq!(frames.len() % BLOCK_SIZE, 0);

        let mut out = Vec::new();
        out.extend_from_slice(b"fLaC");

        // STREAMINFO (last metadata block = 0x80, length = 34)
        out.push(0x80);
        out.extend_from_slice(&[0x00, 0x00, 34]);
        out.extend_from_slice(&(BLOCK_SIZE as u16).to_be_bytes());
        out.extend_from_slice(&(BLOCK_SIZE as u16).to_be_bytes());
        out.extend_from_slice(&[0u8; 6]); // min/max frame size unknown
        let total = frames.len() as u64;
        let packed: u64 = (u64::from(SAMPLE_RATE) << 44)
            | (1u64 << 41) // channels - 1 = 1 (stereo)
            | (15u64 << 36) // bits_per_sample - 1 = 15 (16-bit)
            | (total & ((1u64 << 36) - 1));
        out.extend_from_slice(&packed.to_be_bytes());
        out.extend_from_slice(&[0u8; 16]); // MD5 unset

        for (block_idx, chunk) in frames.chunks_exact(BLOCK_SIZE).enumerate() {
            let mut frame = Vec::with_capacity(BLOCK_SIZE * 4 + 16);
            frame.extend_from_slice(&[0xFF, 0xF8, 0xC9, 0x18, block_idx as u8]);
            let hdr_crc = flac_crc8(&frame);
            frame.push(hdr_crc);

            // Left channel VERBATIM subframe
            frame.push(0x02);
            for &(l, _) in chunk {
                frame.extend_from_slice(&l.to_be_bytes());
            }
            // Right channel VERBATIM subframe
            frame.push(0x02);
            for &(_, r) in chunk {
                frame.extend_from_slice(&r.to_be_bytes());
            }

            let footer_crc = flac_crc16(&frame);
            frame.extend_from_slice(&footer_crc.to_be_bytes());
            out.extend_from_slice(&frame);
        }

        out
    }

    #[test]
    fn seek_exact_lands_on_single_sample_inside_block() {
        // 4 blocks of 4096 frames = 16,384 frames; place a unique impulse at frame 10,000
        // (mid-block inside block 2 at offset 1808).
        const TOTAL_FRAMES: usize = 4 * 4096;
        const IMPULSE_FRAME: usize = 10_000;
        let mut pcm = vec![(0i16, 0i16); TOTAL_FRAMES];
        pcm[IMPULSE_FRAME] = (20_000, -15_000);
        // Also tag frame 4095 and 4096 across a block boundary.
        pcm[4095] = (10_000, 10_000);
        pcm[4096] = (-10_000, -10_000);

        let flac_bytes = encode_verbatim_flac(&pcm);
        let mut hint = Hint::new();
        hint.with_extension("flac");
        let mut decoder =
            SymphoniaDecoder::new(Cursor::new(flac_bytes), hint).expect("open verbatim flac");

        // Seek 5 frames before the impulse at 10,000 -> impulse must appear at exact index 5.
        let target = (IMPULSE_FRAME - 5) as u64;
        let actual = decoder.seek_exact(target).expect("seek_exact mid-block");
        assert_eq!(actual, target);
        assert_eq!(decoder.position_frames(), Some(target));

        let (_pos, pkt) = decoder
            .next_packet()
            .expect("decode first packet")
            .expect("packet present");
        let samples = pkt.samples().expect("pcm samples");
        for offset in 0..5 {
            assert_eq!(samples[offset * 2], 0.0, "frame before impulse non-zero");
            assert_eq!(samples[offset * 2 + 1], 0.0);
        }
        assert!(
            (samples[5 * 2] - (20_000.0 / 32768.0)).abs() < 1e-6,
            "left impulse misplaced or wrong amplitude: {}",
            samples[5 * 2]
        );
        assert!(
            (samples[5 * 2 + 1] - (-15_000.0 / 32768.0)).abs() < 1e-6,
            "right impulse misplaced or wrong amplitude: {}",
            samples[5 * 2 + 1]
        );

        // Seek to exact block boundary - 1 (frame 4095) and verify boundary continuity.
        assert_eq!(decoder.seek_exact(4095).expect("seek boundary-1"), 4095);
        let (_, pkt_a) = decoder.next_packet().unwrap().unwrap();
        let s_a = pkt_a.samples().unwrap();
        assert_eq!(s_a.len(), 2); // exactly 1 frame left in block 0
        assert!((s_a[0] - (10_000.0 / 32768.0)).abs() < 1e-6);

        let (_, pkt_b) = decoder.next_packet().unwrap().unwrap();
        let s_b = pkt_b.samples().unwrap();
        assert!((s_b[0] - (-10_000.0 / 32768.0)).abs() < 1e-6);
    }
}
