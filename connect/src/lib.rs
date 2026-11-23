#![warn(missing_docs)]
#![doc=include_str!("../README.md")]

#[macro_use]
extern crate log;

use librespot_core as core;
use librespot_playback as playback;
use librespot_protocol as protocol;

mod context_resolver;
#[cfg(feature = "mix-transitions")]
pub mod mix;
mod model;
mod shuffle_vec;
mod spirc;
mod state;

#[cfg(feature = "mix-transitions")]
pub use mix::{MixConfig, MixMode};
pub use model::*;
pub use spirc::*;
pub use state::*;
