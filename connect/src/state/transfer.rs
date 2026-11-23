use std::collections::HashMap;

use crate::{
    core::{Error, SpotifyId, SpotifyUri},
    protocol::{context_track::ContextTrack, player::ProvidedTrack, transfer_state::TransferState},
    state::{
        context::ContextType,
        metadata::Metadata,
        options::ShuffleState,
        provider::{IsProvider, Provider},
        {ConnectState, StateError},
    },
};
use protobuf::MessageField;

fn is_transfer_mix_key(key: &str) -> bool {
    key.starts_with("audio.fade_")
        || key.starts_with("automix.")
        || key.starts_with("mix.")
        || matches!(key, "transition_recipe" | "is_mixed" | "mixer_enabled")
}

fn context_track_canonical_uri(track: &ContextTrack) -> Option<String> {
    if let Some(uri) = track.uri.as_ref().filter(|u| !u.is_empty()) {
        return Some(uri.clone());
    }
    let gid = track.gid.as_ref().filter(|g| !g.is_empty())?;
    let id = SpotifyId::from_raw(gid).ok()?;
    Some(SpotifyUri::Track { id }.to_uri())
}

fn insert_mix_metadata(target: &mut HashMap<String, String>, source: &HashMap<String, String>) {
    for (k, v) in source {
        if is_transfer_mix_key(k) && !v.is_empty() {
            match target.get(k) {
                Some(existing) if !existing.is_empty() => {}
                _ => {
                    target.insert(k.clone(), v.clone());
                }
            }
        }
    }
}

impl ConnectState {
    pub fn current_track_from_transfer(
        &self,
        transfer: &TransferState,
    ) -> Result<ProvidedTrack, Error> {
        let track = if transfer.queue.is_playing_queue.unwrap_or_default() {
            debug!("transfer track was used from the queue");
            transfer.queue.tracks.first()
        } else {
            debug!("transfer track was the current track");
            transfer.playback.current_track.as_ref()
        }
        .ok_or(StateError::CouldNotResolveTrackFromTransfer)?;

        self.context_to_provided_track(
            track,
            transfer.current_session.context.uri.as_deref(),
            None,
            None,
            transfer
                .queue
                .is_playing_queue
                .unwrap_or_default()
                .then_some(Provider::Queue),
        )
    }

    /// handles the initially transferable data
    pub fn handle_initial_transfer(
        &mut self,
        transfer: &mut TransferState,
        ctx_uri: Option<String>,
    ) {
        let current_context_metadata = self.context.as_ref().map(|c| c.metadata.clone());
        let player = self.player_mut();

        player.is_buffering = false;

        if let Some(options) = transfer.options.take() {
            player.options = MessageField::some(options.into());
        }
        player.is_paused = transfer.playback.is_paused.unwrap_or_default();
        player.is_playing = !player.is_paused;

        match transfer.playback.playback_speed {
            Some(speed) if speed != 0. => player.playback_speed = speed,
            _ => player.playback_speed = 1.,
        }

        let mut shuffle_seed = None;
        let mut initial_track = None;
        if let Some(session) = transfer.current_session.as_mut() {
            player.play_origin = session.play_origin.take().map(Into::into).into();
            player.suppressions = session.suppressions.take().map(Into::into).into();

            // maybe at some point we can use the shuffle seed provided by spotify,
            // but I doubt it, as spotify doesn't use true randomness but rather an algorithm
            // based shuffle
            trace!(
                "shuffle_seed: <{:?}> (spotify), <{:?}> (own)",
                session.shuffle_seed,
                session.context.get_shuffle_seed()
            );

            shuffle_seed = session
                .context
                .get_shuffle_seed()
                .and_then(|seed| seed.parse().ok());

            initial_track = session.context.get_initial_track().cloned();

            if let Some(ctx) = session.context.as_mut() {
                player.restrictions = ctx.restrictions.take().map(Into::into).into();
                for (key, value) in ctx.metadata.drain() {
                    player.context_metadata.insert(key, value);
                }
            }
        }

        const UNKNOWN_URI: &str = "spotify:unknown";
        // it's important to always set the url/uri to a value
        // so that the player doesn't go into an inactive state
        let uri = ctx_uri.unwrap_or(UNKNOWN_URI.into());

        player.context_url = format!("context://{uri}");
        player.context_uri = uri;

        if let Some(metadata) = current_context_metadata {
            for (key, value) in metadata {
                player.context_metadata.insert(key, value);
            }
        }

        self.transfer_shuffle = match (shuffle_seed, initial_track) {
            (Some(seed), Some(initial_track)) => Some(ShuffleState {
                seed,
                initial_track,
            }),
            _ => None,
        };

        self.clear_prev_track();
        self.clear_next_tracks();
        self.update_queue_revision()
    }

    /// completes the transfer, loading the queue and updating metadata
    pub fn finish_transfer(&mut self, transfer: TransferState) -> Result<(), Error> {
        let track = match self.player().track.as_ref() {
            None => self.current_track_from_transfer(&transfer)?,
            Some(track) => track.clone(),
        };

        let context_ty = if self.current_track(|t| t.is_from_autoplay()) {
            ContextType::Autoplay
        } else {
            ContextType::Default
        };

        self.set_active_context(context_ty);
        self.fill_up_context = context_ty;

        // Preserve pre-computed `audio.fade_*` / `automix.*` track metadata from the
        // transferring client across `/context-resolve` refreshes.
        self.merge_transfer_mix_metadata(&transfer);

        let ctx = self.get_context(self.active_context)?;

        let current_index = match transfer.current_session.current_uid.as_ref() {
            Some(uid) if track.is_queue() => Self::find_index_in_context(ctx, |c| &c.uid == uid)
                .map(|i| if i > 0 { i - 1 } else { i }),
            _ => Self::find_index_in_context(ctx, |c| c.uri == track.uri || c.uid == track.uid),
        };

        debug!(
            "active track is <{}> with index {current_index:?} in {:?} context, has {} tracks",
            track.uri,
            self.active_context,
            ctx.tracks.len()
        );

        if self.player().track.is_none() {
            self.set_track(track);
        }

        let current_index = current_index.ok();
        if let Some(current_index) = current_index {
            self.update_current_index(|i| i.track = current_index as u32);
        }

        debug!(
            "setting up next and prev: index is at {current_index:?} while shuffle {}",
            self.shuffling_context()
        );

        for (i, track) in transfer.queue.tracks.iter().enumerate() {
            if transfer.queue.is_playing_queue.unwrap_or_default() && i == 0 {
                // if we are currently playing from the queue,
                // don't add the first queued item, because we are currently playing that item
                continue;
            }

            if let Ok(queued_track) = self.context_to_provided_track(
                track,
                Some(self.context_uri()),
                None,
                None,
                Some(Provider::Queue),
            ) {
                self.add_to_queue(queued_track, false);
            }
        }

        if self.shuffling_context() {
            self.set_current_track(current_index.unwrap_or_default())?;
            self.set_shuffle(true);

            match self.transfer_shuffle.take() {
                None => self.shuffle_new(),
                Some(state) => self.shuffle_restore(state),
            }?
        } else {
            self.reset_playback_to_position(current_index)?;
        }

        self.update_restrictions();

        Ok(())
    }

    /// Merge `audio.fade_*` and `automix.*` per-track metadata sent inside `TransferState`
    /// onto the freshly resolved context tracks and current track without overwriting
    /// non-empty keys returned by `/context-resolve`.
    fn merge_transfer_mix_metadata(&mut self, transfer: &TransferState) {
        let mut by_uid: HashMap<&str, &HashMap<String, String>> = HashMap::new();
        let mut by_uri: HashMap<String, &HashMap<String, String>> = HashMap::new();

        let transferred_tracks = transfer.playback.current_track.as_ref().into_iter().chain(
            transfer
                .current_session
                .as_ref()
                .and_then(|s| s.context.as_ref())
                .into_iter()
                .flat_map(|ctx| ctx.pages.iter())
                .flat_map(|page| page.tracks.iter()),
        );

        for track in transferred_tracks {
            if !track
                .metadata
                .iter()
                .any(|(k, v)| is_transfer_mix_key(k) && !v.is_empty())
            {
                continue;
            }
            if let Some(uid) = track.uid.as_deref().filter(|u| !u.is_empty()) {
                by_uid.entry(uid).or_insert(&track.metadata);
            }
            if let Some(uri) = context_track_canonical_uri(track) {
                by_uri.entry(uri).or_insert(&track.metadata);
            }
        }

        // Also merge context-level mix metadata preserved from the transferring session
        // (drained into `player.context_metadata` by `handle_initial_transfer`) into the
        // resolved context without overwriting fresh keys.
        let transferred_ctx_mix: HashMap<String, String> = self
            .player()
            .context_metadata
            .iter()
            .filter(|(k, v)| is_transfer_mix_key(k) && !v.is_empty())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !transferred_ctx_mix.is_empty() {
            if let Ok(ctx) = self.get_context_mut(self.active_context) {
                insert_mix_metadata(&mut ctx.metadata, &transferred_ctx_mix);
            }
        }

        if by_uid.is_empty() && by_uri.is_empty() {
            return;
        }

        if let Ok(ctx) = self.get_context_mut(self.active_context) {
            for idx in 0..ctx.tracks.len() {
                if let Some(track) = ctx.tracks.get_mut(idx) {
                    if let Some(src) = by_uid
                        .get(track.uid.as_str())
                        .copied()
                        .or_else(|| by_uri.get(&track.uri).copied())
                    {
                        insert_mix_metadata(&mut track.metadata, src);
                    }
                }
            }
        }

        if let Some(current) = self.player_mut().track.as_mut() {
            if let Some(src) = by_uid
                .get(current.uid.as_str())
                .copied()
                .or_else(|| by_uri.get(&current.uri).copied())
            {
                insert_mix_metadata(&mut current.metadata, src);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        connect::Device, context::Context, context_page::ContextPage, playback::Playback,
        player::PlayerState, session::Session,
    };

    #[test]
    fn transfer_preserves_audio_fade_metadata_without_clobbering_resolved_recipes() {
        let mut transfer = TransferState::new();
        let mut tr_a = ContextTrack {
            uri: Some("spotify:track:6kPXvN6u28Ku8xcHtpL0yV".to_string()),
            uid: Some("uid-a".to_string()),
            ..Default::default()
        };
        tr_a.metadata.insert(
            "audio.fade_out_start_time".to_string(),
            "170444".to_string(),
        );
        tr_a.metadata
            .insert("audio.fade_out_duration".to_string(), "26483".to_string());
        tr_a.metadata.insert(
            "automix.auto_transition_recipe".to_string(),
            "stale-transfer-val".to_string(),
        );

        let mut tr_b = ContextTrack {
            uri: Some("spotify:track:685if6yGqGm0P88BoJrOiL".to_string()),
            uid: Some("uid-b".to_string()),
            ..Default::default()
        };
        tr_b.metadata
            .insert("audio.fade_in_start_time".to_string(), "12800".to_string());
        tr_b.metadata
            .insert("audio.fade_overlap".to_string(), "26483".to_string());

        transfer.playback = MessageField::some(Playback {
            current_track: MessageField::some(tr_a.clone()),
            ..Default::default()
        });
        transfer.current_session = MessageField::some(Session {
            context: MessageField::some(Context {
                uri: Some("spotify:playlist:test".to_string()),
                pages: vec![ContextPage {
                    tracks: vec![tr_a, tr_b],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        });

        let mut state = ConnectState::default();
        state.request.device = MessageField::some(Device {
            player_state: MessageField::some(PlayerState::new()),
            ..Default::default()
        });
        state.handle_initial_transfer(&mut transfer, Some("spotify:playlist:test".to_string()));
        // Pages must still be present on `transfer` after `handle_initial_transfer`.
        assert_eq!(transfer.current_session.context.pages.len(), 1);

        // Simulate `/context-resolve` populating fresh tracks without `audio.fade_*`.
        let mut resolved_a = ContextTrack {
            uri: Some("spotify:track:6kPXvN6u28Ku8xcHtpL0yV".to_string()),
            uid: Some("uid-a".to_string()),
            ..Default::default()
        };
        resolved_a.metadata.insert(
            "automix.auto_transition_recipe".to_string(),
            "fresh-resolved-recipe".to_string(),
        );
        let resolved_b = ContextTrack {
            uri: Some("spotify:track:685if6yGqGm0P88BoJrOiL".to_string()),
            uid: Some("uid-b".to_string()),
            ..Default::default()
        };
        let resolved_ctx = Context {
            uri: Some("spotify:playlist:test".to_string()),
            pages: vec![ContextPage {
                tracks: vec![resolved_a, resolved_b],
                ..Default::default()
            }],
            ..Default::default()
        };
        state
            .update_context(resolved_ctx, ContextType::Default)
            .expect("update_context");
        state.finish_transfer(transfer).expect("finish_transfer");

        let ctx = state.get_context(ContextType::Default).unwrap();
        let a = ctx.tracks.first().unwrap();
        let b = ctx.tracks.get(1).unwrap();
        assert_eq!(
            a.metadata
                .get("audio.fade_out_start_time")
                .map(String::as_str),
            Some("170444")
        );
        assert_eq!(
            a.metadata
                .get("automix.auto_transition_recipe")
                .map(String::as_str),
            Some("fresh-resolved-recipe"),
            "fresh /context-resolve recipe must not be overwritten"
        );
        assert_eq!(
            b.metadata
                .get("audio.fade_in_start_time")
                .map(String::as_str),
            Some("12800")
        );
        assert_eq!(
            b.metadata.get("audio.fade_overlap").map(String::as_str),
            Some("26483")
        );
    }
}
