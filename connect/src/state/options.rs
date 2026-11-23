use crate::{
    core::Error,
    protocol::player::ContextPlayerOptions,
    state::{
        ConnectState, StateError,
        context::{ContextType, ResetContext},
        metadata::Metadata,
    },
};
use protobuf::MessageField;
use rand::Rng;
use std::collections::HashMap;

#[derive(Default, Debug)]
pub(crate) struct ShuffleState {
    pub seed: u64,
    pub initial_track: String,
}

impl ConnectState {
    fn add_options_if_empty(&mut self) {
        if self.player().options.is_none() {
            self.player_mut().options = MessageField::some(ContextPlayerOptions::new())
        }
    }

    pub fn set_repeat_context(&mut self, repeat: bool) {
        self.add_options_if_empty();
        if let Some(options) = self.player_mut().options.as_mut() {
            options.repeating_context = repeat;
        }
    }

    pub fn set_repeat_track(&mut self, repeat: bool) {
        self.add_options_if_empty();
        if let Some(options) = self.player_mut().options.as_mut() {
            options.repeating_track = repeat;
        }
    }

    pub fn set_shuffle(&mut self, shuffle: bool) {
        self.add_options_if_empty();
        if let Some(options) = self.player_mut().options.as_mut() {
            options.shuffling_context = shuffle;
        }
    }

    pub fn reset_options(&mut self) {
        self.set_shuffle(false);
        self.set_repeat_track(false);
        self.set_repeat_context(false);
        self.clear_modes();
    }

    pub fn set_modes(&mut self, modes: HashMap<String, String>) {
        self.add_options_if_empty();
        if let Some(options) = self.player_mut().options.as_mut() {
            for (key, value) in modes {
                if value.is_empty() {
                    options.modes.remove(&key);
                } else {
                    options.modes.insert(key, value);
                }
            }
        }
    }

    pub fn clear_modes(&mut self) {
        if let Some(options) = self.player_mut().options.as_mut() {
            options.modes.clear();
        }
    }

    fn validate_shuffle_allowed(&self) -> Result<(), Error> {
        if let Some(reason) = self
            .player()
            .restrictions
            .disallow_toggling_shuffle_reasons
            .first()
        {
            Err(StateError::CurrentlyDisallowed {
                action: "shuffle",
                reason: reason.clone(),
            })?
        } else {
            Ok(())
        }
    }

    pub fn shuffle_restore(&mut self, shuffle_state: ShuffleState) -> Result<(), Error> {
        self.validate_shuffle_allowed()?;

        self.shuffle(shuffle_state.seed, &shuffle_state.initial_track)
    }

    pub fn shuffle_new(&mut self) -> Result<(), Error> {
        self.validate_shuffle_allowed()?;

        let new_seed = rand::rng().random_range(100_000_000_000..1_000_000_000_000);
        let current_track = self.current_track(|t| t.uri.clone());

        self.shuffle(new_seed, &current_track)
    }

    fn shuffle(&mut self, seed: u64, initial_track: &str) -> Result<(), Error> {
        self.clear_prev_track();
        self.clear_next_tracks();

        self.reset_context(ResetContext::DefaultIndex);

        let ctx = self.get_context_mut(ContextType::Default)?;
        ctx.tracks
            .shuffle_with_seed(seed, |f| f.uri == initial_track);

        ctx.set_initial_track(initial_track);
        ctx.set_shuffle_seed(seed);

        self.fill_up_next_tracks()?;

        Ok(())
    }

    pub fn shuffling_context(&self) -> bool {
        self.player().options.shuffling_context
    }

    pub fn repeat_context(&self) -> bool {
        self.player().options.repeating_context
    }

    #[allow(dead_code)]
    pub fn modes(&self) -> Option<&HashMap<String, String>> {
        self.player().options.as_ref().map(|o| &o.modes)
    }

    pub fn repeat_track(&self) -> bool {
        self.player().options.repeating_track
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{connect::Device, player::PlayerState};

    #[test]
    fn set_modes_merges_and_clears_empty_values() {
        let mut state = ConnectState::default();
        state.request.device = MessageField::some(Device {
            player_state: MessageField::some(PlayerState::new()),
            ..Default::default()
        });
        state.set_modes(HashMap::from([
            ("mix".to_string(), "on".to_string()),
            ("smart_shuffle".to_string(), "1".to_string()),
        ]));
        assert_eq!(
            state.modes().and_then(|m| m.get("mix")).map(String::as_str),
            Some("on")
        );

        state.set_modes(HashMap::from([("mix".to_string(), String::new())]));
        assert_eq!(state.modes().and_then(|m| m.get("mix")), None);
        assert_eq!(
            state
                .modes()
                .and_then(|m| m.get("smart_shuffle"))
                .map(String::as_str),
            Some("1")
        );

        state.reset_options();
        assert!(state.modes().is_none_or(HashMap::is_empty));
    }
}
