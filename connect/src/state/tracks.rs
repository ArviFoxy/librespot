use crate::{
    core::{Error, SpotifyUri},
    protocol::player::ProvidedTrack,
    state::{
        ConnectState, SPOTIFY_MAX_NEXT_TRACKS_SIZE, SPOTIFY_MAX_PREV_TRACKS_SIZE, StateError,
        context::ContextType,
        metadata::Metadata,
        provider::{IsProvider, Provider},
    },
};
use protobuf::MessageField;
use rand::Rng;

// identifier used as part of the uid
pub const IDENTIFIER_DELIMITER: &str = "delimiter";

impl<'ct> ConnectState {
    fn new_delimiter(iteration: i64) -> ProvidedTrack {
        let mut delimiter = ProvidedTrack {
            uri: format!("spotify:{IDENTIFIER_DELIMITER}"),
            uid: format!("{IDENTIFIER_DELIMITER}{iteration}"),
            provider: Provider::Context.to_string(),
            ..Default::default()
        };
        delimiter.set_hidden(true);
        delimiter.set_iteration(iteration);

        delimiter
    }

    fn push_prev(&mut self, prev: ProvidedTrack) {
        let prev_tracks = self.prev_tracks_mut();
        // add prev track, while preserving a length of 10
        if prev_tracks.len() >= SPOTIFY_MAX_PREV_TRACKS_SIZE {
            // todo: O(n), but technically only maximal O(SPOTIFY_MAX_PREV_TRACKS_SIZE) aka O(10)
            let _ = prev_tracks.remove(0);
        }
        prev_tracks.push(prev)
    }

    fn get_next_track(&mut self) -> Option<ProvidedTrack> {
        if self.next_tracks().is_empty() {
            None
        } else {
            // todo: O(n), but technically only maximal O(SPOTIFY_MAX_NEXT_TRACKS_SIZE) aka O(80)
            Some(self.next_tracks_mut().remove(0))
        }
    }

    /// bottom => top, aka the last track of the list is the prev track
    fn prev_tracks_mut(&mut self) -> &mut Vec<ProvidedTrack> {
        &mut self.player_mut().prev_tracks
    }

    /// bottom => top, aka the last track of the list is the prev track
    pub(super) fn prev_tracks(&self) -> &Vec<ProvidedTrack> {
        &self.player().prev_tracks
    }

    /// top => bottom, aka the first track of the list is the next track
    fn next_tracks_mut(&mut self) -> &mut Vec<ProvidedTrack> {
        &mut self.player_mut().next_tracks
    }

    /// top => bottom, aka the first track of the list is the next track
    pub(super) fn next_tracks(&self) -> &Vec<ProvidedTrack> {
        &self.player().next_tracks
    }

    pub fn set_current_track_random(&mut self) -> Result<(), Error> {
        let max_tracks = self.get_context(self.active_context)?.tracks.len();
        let rng_track = rand::rng().random_range(0..max_tracks);
        self.set_current_track(rng_track)
    }

    pub fn set_current_track(&mut self, index: usize) -> Result<(), Error> {
        let context = self.get_context(self.active_context)?;

        let new_track = context
            .tracks
            .get(index)
            .ok_or(StateError::CanNotFindTrackInContext(
                Some(index),
                context.tracks.len(),
            ))?;

        debug!(
            "set track to: {} at {} of {} tracks",
            new_track.uri,
            index,
            context.tracks.len()
        );

        self.set_track(new_track.clone());

        self.update_current_index(|i| i.track = index as u32);

        Ok(())
    }

    /// Move to the next track
    ///
    /// Updates the current track to the next track. Adds the old track
    /// to prev tracks and fills up the next tracks from the current context
    pub fn next_track(&mut self) -> Result<Option<u32>, Error> {
        // when we skip in repeat track, we don't repeat the current track anymore
        if self.repeat_track() {
            self.set_repeat_track(false);
        }

        let old_track = self.player_mut().track.take();

        if let Some(old_track) = old_track {
            // only add songs from our context to our previous tracks
            if old_track.is_context() || old_track.is_autoplay() {
                self.push_prev(old_track)
            }
        }

        let new_track = loop {
            match self.get_next_track() {
                Some(next) if next.uid.starts_with(IDENTIFIER_DELIMITER) => {
                    self.push_prev(next);
                    continue;
                }
                Some(next) if next.is_unavailable() => continue,
                other => break other,
            };
        };

        let new_track = match new_track {
            None => return Ok(None),
            Some(t) => t,
        };

        self.fill_up_next_tracks()?;

        let update_index = if new_track.is_queue() {
            None
        } else if new_track.is_autoplay() {
            self.set_active_context(ContextType::Autoplay);
            None
        } else {
            match new_track.get_context_index() {
                Some(new_index) => Some(new_index as u32),
                None => {
                    error!("the given context track had no set context_index");
                    None
                }
            }
        };

        if let Some(update_index) = update_index {
            self.update_current_index(|i| i.track = update_index)
        } else {
            self.player_mut().index.clear()
        }

        self.set_track(new_track);
        self.update_restrictions();

        Ok(Some(self.player().index.track))
    }

    /// Move to the prev track
    ///
    /// Updates the current track to the prev track. Adds the old track
    /// to next tracks (when from the context) and fills up the prev tracks from the
    /// current context
    pub fn prev_track(&mut self) -> Result<Option<&MessageField<ProvidedTrack>>, Error> {
        let old_track = self.player_mut().track.take();

        if let Some(old_track) = old_track {
            if old_track.is_context() || old_track.is_autoplay() {
                // todo: O(n)
                self.next_tracks_mut().insert(0, old_track);
            }
        }

        // handle possible delimiter
        if matches!(self.prev_tracks().last(), Some(prev) if prev.uid.starts_with(IDENTIFIER_DELIMITER))
        {
            let delimiter = self
                .prev_tracks_mut()
                .pop()
                .expect("item that was prechecked");

            let next_tracks = self.next_tracks_mut();
            if next_tracks.len() >= SPOTIFY_MAX_NEXT_TRACKS_SIZE {
                let _ = next_tracks.pop();
            }
            // todo: O(n)
            next_tracks.insert(0, delimiter)
        }

        while self.next_tracks().len() > SPOTIFY_MAX_NEXT_TRACKS_SIZE {
            let _ = self.next_tracks_mut().pop();
        }

        let new_track = match self.prev_tracks_mut().pop() {
            None => return Ok(None),
            Some(t) => t,
        };

        if matches!(self.active_context, ContextType::Autoplay if new_track.is_context()) {
            // transition back to default context
            self.set_active_context(ContextType::Default);
        }

        self.fill_up_next_tracks()?;
        self.set_track(new_track);

        if self.player().index.track == 0 {
            warn!("prev: trying to skip into negative, index update skipped")
        } else {
            self.update_current_index(|i| i.track -= 1)
        }

        self.update_restrictions();

        Ok(Some(self.current_track(|t| t)))
    }

    pub fn current_track<F: Fn(&'ct MessageField<ProvidedTrack>) -> R, R>(
        &'ct self,
        access: F,
    ) -> R {
        access(&self.player().track)
    }

    pub fn set_track(&mut self, track: ProvidedTrack) {
        self.player_mut().track = MessageField::some(track)
    }

    pub fn set_next_tracks(&mut self, mut tracks: Vec<ProvidedTrack>) {
        // mobile only sends a set_queue command instead of an add_to_queue command
        // in addition to handling the mobile add_to_queue handling, this should also handle
        // a mass queue addition
        tracks
            .iter_mut()
            .filter(|t| t.is_from_queue())
            .for_each(|t| {
                t.set_provider(Provider::Queue);
                // technically we could preserve the queue-uid here,
                // but it seems to work without that, so we just override it
                t.uid = format!("q{}", self.queue_count);
                self.queue_count += 1;
            });

        // when you drag 'n drop the current track in the queue view into the "Next from: ..."
        // section, it is only send as an empty item with just the provider and metadata, so we have
        // to provide set the uri from the current track manually
        tracks
            .iter_mut()
            .filter(|t| t.uri.is_empty())
            .for_each(|t| t.uri = self.current_track(|ct| ct.uri.clone()));

        self.player_mut().next_tracks = tracks;
    }

    pub fn set_prev_tracks(&mut self, tracks: Vec<ProvidedTrack>) {
        self.player_mut().prev_tracks = tracks;
    }

    pub fn clear_prev_track(&mut self) {
        self.prev_tracks_mut().clear()
    }

    /// Remove all queued tracks
    ///
    /// Keeps the tracks from the context in the next tracks and fills
    /// them up from the current context afterwards
    pub fn clear_queue(&mut self) -> Result<(), Error> {
        self.next_tracks_mut().retain(|t| !t.is_queue());
        self.fill_up_next_tracks()?;
        self.update_queue_revision();
        Ok(())
    }

    pub fn clear_next_tracks(&mut self) {
        // respect queued track and don't throw them out of our next played tracks
        let first_non_queued_track = self
            .next_tracks()
            .iter()
            .enumerate()
            .find(|(_, track)| !track.is_queue());

        if let Some((non_queued_track, _)) = first_non_queued_track {
            while self.next_tracks().len() > non_queued_track
                && self.next_tracks_mut().pop().is_some()
            {}
        }
    }

    pub fn fill_up_next_tracks(&mut self) -> Result<(), Error> {
        let ctx = self.get_context(self.fill_up_context)?;
        let mut new_index = ctx.index.track as usize;
        let mut iteration = ctx.index.page;

        while self.next_tracks().len() < SPOTIFY_MAX_NEXT_TRACKS_SIZE {
            let ctx = self.get_context(self.fill_up_context)?;
            let track = match ctx.tracks.get(new_index) {
                None if self.repeat_context() => {
                    let delimiter = Self::new_delimiter(iteration.into());
                    iteration += 1;
                    new_index = 0;
                    delimiter
                }
                None if !matches!(self.fill_up_context, ContextType::Autoplay)
                    && self.autoplay_context.is_some()
                    && !self.repeat_context() =>
                {
                    self.update_context_index(self.fill_up_context, new_index)?;

                    // transition to autoplay as fill up context
                    self.fill_up_context = ContextType::Autoplay;
                    new_index = self.get_context(ContextType::Autoplay)?.index.track as usize;

                    // add delimiter to only display the current context
                    Self::new_delimiter(iteration.into())
                }
                None if self.autoplay_context.is_some() => {
                    match self
                        .get_context(ContextType::Autoplay)?
                        .tracks
                        .get(new_index)
                    {
                        None => break,
                        Some(ct) => {
                            new_index += 1;
                            ct.clone()
                        }
                    }
                }
                None => break,
                Some(ct) if ct.is_unavailable() || self.is_skip_track(ct, Some(iteration)) => {
                    debug!(
                        "skipped track {} during fillup as it's unavailable or should be skipped",
                        ct.uri
                    );
                    new_index += 1;
                    continue;
                }
                Some(ct) => {
                    new_index += 1;
                    ct.clone()
                }
            };

            self.next_tracks_mut().push(track);
        }

        debug!(
            "finished filling up next_tracks ({})",
            self.next_tracks().len()
        );

        self.update_context_index(self.fill_up_context, new_index)?;

        // the web-player needs a revision update, otherwise the queue isn't updated in the ui
        self.update_queue_revision();

        Ok(())
    }

    fn first_playable_next_track(&self) -> Option<&ProvidedTrack> {
        self.next_tracks().iter().find(|t| {
            !t.uid.starts_with(IDENTIFIER_DELIMITER) && !t.is_unavailable() && !t.uri.is_empty()
        })
    }

    pub fn preview_next_track(&mut self) -> Option<SpotifyUri> {
        let next = if self.repeat_track() {
            self.current_track(|t| &t.uri)
        } else {
            &self.first_playable_next_track()?.uri
        };

        SpotifyUri::from_uri(next).ok()
    }

    /// Return `(current_track, next_track, current_duration_ms)` for background transition
    /// resolution without mutating queue state.
    #[cfg(feature = "mix-transitions")]
    pub fn transition_pair_snapshot(&self) -> Option<(ProvidedTrack, ProvidedTrack, u32)> {
        let current = self.player().track.as_ref()?.clone();
        if current.uri.is_empty() {
            return None;
        }
        let next = if self.repeat_track() {
            current.clone()
        } else {
            self.first_playable_next_track()?.clone()
        };
        if next.uri.is_empty() {
            return None;
        }
        let duration_ms = self.player().duration.max(0) as u32;
        Some((current, next, duration_ms))
    }

    /// Atomically commit a dual-deck transition to `incoming_track_id` without issuing
    /// a redundant `Player::load`.
    ///
    /// Returns `Ok(true)` if `ConnectState` now points at `incoming_track_id` (either via
    /// repeat-track staying on the same track or by advancing once when `preview_next_track()`
    /// matches), and `Ok(false)` if the queue changed mid-overlap so the playlist is never
    /// accidentally drained.
    #[cfg(feature = "mix-transitions")]
    pub fn commit_transition_to(&mut self, incoming_track_id: &SpotifyUri) -> Result<bool, Error> {
        let incoming_uri = incoming_track_id.to_uri();
        if self.repeat_track() && self.current_track(|t| t.uri == incoming_uri) {
            return Ok(true);
        }
        if self.preview_next_track().as_ref() == Some(incoming_track_id) {
            self.next_track()?;
            return Ok(self.current_track(|t| t.uri == incoming_uri));
        }
        Ok(false)
    }

    pub fn has_next_tracks(&self, min: Option<usize>) -> bool {
        if let Some(min) = min {
            self.next_tracks().len() >= min
        } else {
            !self.next_tracks().is_empty()
        }
    }

    pub fn recent_track_uris(&self) -> Vec<String> {
        let mut prev = self
            .prev_tracks()
            .iter()
            .map(|t| t.uri.clone())
            .collect::<Vec<_>>();

        prev.push(self.current_track(|t| t.uri.clone()));
        prev
    }

    pub fn mark_unavailable(&mut self, id: &SpotifyUri) -> Result<(), Error> {
        let uri = id.to_uri();

        debug!("marking {uri} as unavailable");

        let next_tracks = self.next_tracks_mut();
        while let Some(pos) = next_tracks.iter().position(|t| t.uri == uri) {
            let _ = next_tracks.remove(pos);
        }

        for next_track in next_tracks {
            Self::mark_as_unavailable_for_match(next_track, &uri)
        }

        let prev_tracks = self.prev_tracks_mut();
        while let Some(pos) = prev_tracks.iter().position(|t| t.uri == uri) {
            let _ = prev_tracks.remove(pos);
        }

        for prev_track in prev_tracks {
            Self::mark_as_unavailable_for_match(prev_track, &uri)
        }

        self.unavailable_uri.push(uri);
        self.fill_up_next_tracks()?;
        self.update_queue_revision();

        Ok(())
    }

    pub fn add_to_queue(&mut self, mut track: ProvidedTrack, rev_update: bool) {
        track.uid = format!("q{}", self.queue_count);
        self.queue_count += 1;

        track.set_provider(Provider::Queue);
        if !track.is_from_queue() {
            track.set_from_queue(true);
        }

        let next_tracks = self.next_tracks_mut();
        if let Some(next_not_queued_track) = next_tracks.iter().position(|t| !t.is_queue()) {
            next_tracks.insert(next_not_queued_track, track);
        } else {
            next_tracks.push(track)
        }

        while next_tracks.len() > SPOTIFY_MAX_NEXT_TRACKS_SIZE {
            next_tracks.pop();
        }

        if rev_update {
            self.update_queue_revision();
        }
        self.update_restrictions();
    }
}

#[cfg(all(test, feature = "mix-transitions"))]
mod tests {
    use super::*;
    use crate::{
        protocol::{
            connect::Device,
            player::{ContextIndex, PlayerState},
        },
        shuffle_vec::ShuffleVec,
        state::context::StateContext,
    };
    use std::collections::HashMap;

    fn make_track(uri: &str, uid: &str, idx: usize) -> ProvidedTrack {
        let mut t = ProvidedTrack {
            uri: uri.to_string(),
            uid: uid.to_string(),
            ..Default::default()
        };
        t.set_provider(Provider::Context);
        t.set_context_index(idx);
        t
    }

    #[test]
    fn commit_transition_to_advances_once_respects_repeat_and_never_drains_on_mismatch() {
        let uri_a = "spotify:track:4cOdK2wGLETKBW3PvgPWqT";
        let uri_b = "spotify:track:0VjIjW4GlUZAMYd2vXMi3b";
        let uri_c = "spotify:track:11dFghVXANMlKmJXsNCbNl";
        let uri_d = "spotify:track:7qiZfU4dY1lWllzX7mPBI3";

        let tracks = vec![
            make_track(uri_a, "u1", 0),
            make_track(uri_b, "u2", 1),
            make_track(uri_c, "u3", 2),
        ];

        let mut state = ConnectState::default();
        state.request.device = MessageField::some(Device {
            player_state: MessageField::some(PlayerState::new()),
            ..Default::default()
        });
        state.context = Some(StateContext {
            tracks: ShuffleVec::from(tracks.clone()),
            metadata: HashMap::new(),
            restrictions: None,
            index: ContextIndex {
                page: 0,
                track: 3,
                ..Default::default()
            },
        });
        state.set_track(tracks[0].clone());
        state
            .next_tracks_mut()
            .extend([tracks[1].clone(), tracks[2].clone()]);

        // Mismatched track must NOT advance or drain the playlist.
        let id_d = SpotifyUri::from_uri(uri_d).unwrap();
        assert!(!state.commit_transition_to(&id_d).unwrap());
        assert_eq!(state.current_track(|t| t.uri.clone()), uri_a);
        assert_eq!(state.next_tracks().len(), 2);

        // Repeat-track on track A keeps track A active without popping track B.
        state.set_repeat_track(true);
        let id_a = SpotifyUri::from_uri(uri_a).unwrap();
        assert!(state.commit_transition_to(&id_a).unwrap());
        assert_eq!(state.current_track(|t| t.uri.clone()), uri_a);
        assert_eq!(state.next_tracks().len(), 2);
        state.set_repeat_track(false);

        // Matching next track B advances atomically from A -> B, moving A to prev and keeping C next.
        let id_b = SpotifyUri::from_uri(uri_b).unwrap();
        assert!(state.commit_transition_to(&id_b).unwrap());
        assert_eq!(state.current_track(|t| t.uri.clone()), uri_b);
        assert_eq!(state.prev_tracks().len(), 1);
        assert_eq!(state.prev_tracks()[0].uri, uri_a);
        assert_eq!(state.next_tracks().len(), 1);
        assert_eq!(state.next_tracks()[0].uri, uri_c);
    }
}
