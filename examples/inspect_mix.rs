use std::{collections::BTreeMap, env, fs, path::PathBuf, process::exit};

use data_encoding::{BASE64, BASE64_NOPAD, BASE64URL, BASE64URL_NOPAD, HEXLOWER};
use futures_util::StreamExt;
use librespot::{
    core::{
        SpotifyUri, authentication::Credentials, cache::Cache, config::SessionConfig,
        session::Session,
    },
    oauth::DeviceAuthClientBuilder,
    protocol::{
        audio_attributes_v2::AudioAttributes,
        automix_mode::AutomixMode,
        connect::{
            Capabilities, Cluster, Device, DeviceInfo, MemberType, PutStateReason, PutStateRequest,
        },
        cuepoints::Cuepoints,
        extension_kind::ExtensionKind,
        mix_beats::Beats,
        mix_state::MixState,
        mix_vocal_activity::VocalActivity,
        mixability::Mixability,
        playlist4_external::SelectedListContent,
        protobuf::{EnumOrUnknown, Message, MessageField, UnknownFields, UnknownValueRef},
        transition::Transition,
        transition_metadata_extension::TransitionMetadataExtensionResponse,
    },
};

fn usage(bin: &str) -> ! {
    eprintln!(
        "Usage: {bin} [--cache DIR | --token TOKEN] [--limit N] [--save-fixtures DIR] [--cluster] [PLAYLIST_URI_OR_ID]\n\
         \n\
         Inspects Spotify playlist4, context-resolve, extended-metadata, and (--cluster) live Connect cluster state.\n\
         If --cache DIR is given and contains no credentials yet, starts Spotify's device authorization flow\n\
         (spotify.com/pair) and stores credentials in DIR."
    );
    exit(2);
}

fn try_decode_base64(value: &str) -> Option<Vec<u8>> {
    let trimmed = value.trim();
    for enc in [BASE64, BASE64_NOPAD, BASE64URL, BASE64URL_NOPAD] {
        if let Ok(bytes) = enc.decode(trimmed.as_bytes()) {
            return Some(bytes);
        }
    }
    None
}

fn summarize_unknown_fields(unknown: &UnknownFields) -> Vec<String> {
    let mut out = Vec::new();
    for (field_num, val) in unknown.iter() {
        let desc = match val {
            UnknownValueRef::Fixed32(v) => format!("field {field_num} (fixed32={v})"),
            UnknownValueRef::Fixed64(v) => format!("field {field_num} (fixed64={v})"),
            UnknownValueRef::Varint(v) => format!("field {field_num} (varint={v})"),
            UnknownValueRef::LengthDelimited(b) => {
                let preview = match std::str::from_utf8(b) {
                    Ok(s) if s.chars().all(|c| !c.is_control()) => {
                        format!("\"{}\"", &s[..s.len().min(80)])
                    }
                    _ => format!("hex:{}", HEXLOWER.encode(&b[..b.len().min(32)])),
                };
                format!("field {field_num} (bytes[{}]: {preview})", b.len())
            }
        };
        out.push(desc);
    }
    out
}

fn decode_wire_summary(bytes: &[u8]) -> String {
    let mut dummy = MixState::new();
    if dummy.merge_from_bytes(bytes).is_ok() {
        let mut fields = Vec::new();
        if dummy.is_mixed {
            fields.push("field 1 (varint=1)".to_string());
        }
        fields.extend(summarize_unknown_fields(
            dummy.special_fields().unknown_fields(),
        ));
        if !fields.is_empty() {
            return fields.join(", ");
        }
    }
    format!(
        "raw[{}]: {}",
        bytes.len(),
        HEXLOWER.encode(&bytes[..bytes.len().min(48)])
    )
}

fn print_extension(kind: ExtensionKind, entity: &str, status: i32, type_url: &str, data: &[u8]) {
    if data.is_empty() && type_url.is_empty() {
        println!("  [{kind:?}] {entity} status={status} (empty)");
        return;
    }

    println!(
        "  [{kind:?}] {entity} status={status} bytes={} type_url={type_url}",
        data.len()
    );

    match kind {
        ExtensionKind::MIX_STATE => match MixState::parse_from_bytes(data) {
            Ok(v) => println!("    MixState {{ is_mixed: {} }}", v.is_mixed),
            Err(e) => println!("    parse error: {e}"),
        },
        ExtensionKind::AUTOMIX_MODE => match AutomixMode::parse_from_bytes(data) {
            Ok(v) => println!("    AutomixMode {{ style: {:?} }}", v.style),
            Err(e) => println!("    parse error: {e}"),
        },
        ExtensionKind::AUDIO_ATTRIBUTES_V2 => match AudioAttributes::parse_from_bytes(data) {
            Ok(v) => {
                let key = v.key.as_ref().map(|k| {
                    let camelot = k
                        .camelot_key
                        .as_ref()
                        .map(|c| c.value.as_str())
                        .unwrap_or("?");
                    format!("{} {:?} (camelot={camelot})", k.key, k.mode)
                });
                println!("    AudioAttributes {{ bpm: {:.2}, key: {key:?} }}", v.bpm);
            }
            Err(e) => println!("    parse error: {e}"),
        },
        ExtensionKind::MIXABILITY => match Mixability::parse_from_bytes(data) {
            Ok(v) => println!(
                "    Mixability {{ mixable: {}, genre_based_beatmatchability: {:.3} }}",
                v.mixable, v.genre_based_beatmatchability
            ),
            Err(e) => println!("    parse error: {e}"),
        },
        ExtensionKind::BEATS => match Beats::parse_from_bytes(data) {
            Ok(v) => {
                let first = v
                    .beats
                    .first()
                    .map(|b| (b.time, b.duration, b.value, b.downbeat_confidence));
                let last = v.beats.last().map(|b| (b.time, b.duration, b.value));
                println!(
                    "    Beats {{ beats_per_bar: {}, count: {}, hash: {:?}, first: {first:?}, last: {last:?} }}",
                    v.beats_per_bar,
                    v.beats.len(),
                    v.beats_hash
                );
            }
            Err(e) => println!("    parse error: {e}"),
        },
        ExtensionKind::VOCAL_ACTIVITY => match VocalActivity::parse_from_bytes(data) {
            Ok(v) => {
                let min = v.vocal_activity_probabilities.iter().copied().min();
                let max = v.vocal_activity_probabilities.iter().copied().max();
                println!(
                    "    VocalActivity {{ sr: {}, win: {}, start: {}, step: {}, count: {}, min: {min:?}, max: {max:?} }}",
                    v.source_sample_rate_hz,
                    v.smoothing_window_size,
                    v.first_window_sample_start,
                    v.samples_between_windows,
                    v.vocal_activity_probabilities.len()
                );
            }
            Err(e) => println!("    parse error: {e}"),
        },
        ExtensionKind::CUEPOINTS => match Cuepoints::parse_from_bytes(data) {
            Ok(v) => {
                println!(
                    "    Cuepoints {{ best_in: {:?}, best_out: {:?}, in_count: {}, out_count: {} }}",
                    v.best_fade_in_cuepoint.as_ref().map(|c| (
                        c.position_ms,
                        c.tempo_bpm,
                        c.origin,
                        c.confidence
                    )),
                    v.best_fade_out_cuepoint.as_ref().map(|c| (
                        c.position_ms,
                        c.tempo_bpm,
                        c.origin,
                        c.confidence
                    )),
                    v.fade_in_cuepoints.len(),
                    v.fade_out_cuepoints.len(),
                );
            }
            Err(e) => println!("    parse error: {e}"),
        },
        _ => {
            if let Ok(t) = TransitionMetadataExtensionResponse::parse_from_bytes(data) {
                if !t.transition_uri.is_empty() {
                    println!("    TransitionMetadataExtensionResponse: {t:#?}");
                    return;
                }
            }
            println!("    wire: {}", decode_wire_summary(data));
        }
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let bin = &args[0];

    let mut cache_dir: Option<PathBuf> = None;
    let mut token: Option<String> = None;
    let mut save_fixtures: Option<PathBuf> = None;
    let mut inspect_cluster = false;
    let mut limit: usize = 6;
    let mut playlist_arg: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cache" => {
                i += 1;
                cache_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| usage(bin))));
            }
            "--token" => {
                i += 1;
                token = Some(args.get(i).unwrap_or_else(|| usage(bin)).clone());
            }
            "--limit" => {
                i += 1;
                limit = args
                    .get(i)
                    .unwrap_or_else(|| usage(bin))
                    .parse()
                    .unwrap_or_else(|_| usage(bin));
            }
            "--save-fixtures" => {
                i += 1;
                save_fixtures = Some(PathBuf::from(args.get(i).unwrap_or_else(|| usage(bin))));
            }
            "--cluster" => {
                inspect_cluster = true;
            }
            "-h" | "--help" => usage(bin),
            arg if !arg.starts_with('-') && playlist_arg.is_none() => {
                playlist_arg = Some(arg.to_string());
            }
            _ => usage(bin),
        }
        i += 1;
    }

    if !inspect_cluster && playlist_arg.is_none() {
        usage(bin);
    }

    let playlist_id_and_uri = playlist_arg.map(|playlist_arg| {
        let playlist_uri = if playlist_arg.starts_with("spotify:") {
            SpotifyUri::from_uri(&playlist_arg).unwrap_or_else(|e| {
                eprintln!("Invalid playlist URI: {e}");
                exit(1);
            })
        } else {
            let id_str = playlist_arg
                .split("/playlist/")
                .nth(1)
                .unwrap_or(&playlist_arg)
                .split('?')
                .next()
                .unwrap_or(&playlist_arg);
            SpotifyUri::from_uri(&format!("spotify:playlist:{id_str}")).unwrap_or_else(|e| {
                eprintln!("Invalid playlist ID/URL: {e}");
                exit(1);
            })
        };

        let SpotifyUri::Playlist {
            id: playlist_id, ..
        } = playlist_uri
        else {
            eprintln!("Expected a playlist URI, got {playlist_uri:?}");
            exit(1);
        };
        let canonical = format!("spotify:playlist:{}", playlist_id.to_base62());
        (playlist_id, canonical)
    });

    let session_config = SessionConfig::default();
    let cache = cache_dir
        .as_ref()
        .map(|dir| Cache::new(Some(dir), None, None, None).expect("init cache"));

    let credentials = if let Some(tok) = token {
        Credentials::with_access_token(tok)
    } else if let Some(creds) = cache.as_ref().and_then(Cache::credentials) {
        creds
    } else {
        eprintln!("No cached credentials; starting Spotify device authorization flow...");
        let client = DeviceAuthClientBuilder::new(
            &session_config.client_id,
            vec![
                "streaming",
                "playlist-read-private",
                "playlist-read-collaborative",
            ],
        )
        .build()
        .expect("build device auth client");
        let auth = client.request_device_code().expect("request device code");
        eprintln!(
            "Authorize on any device at {} (code: {})",
            auth.url(),
            auth.user_code()
        );
        let tok = client.poll_for_token(&auth).expect("poll device token");
        Credentials::with_access_token(tok.access_token)
    };

    let session = Session::new(session_config, cache);
    let mut conn_id_rx = if inspect_cluster {
        Some(
            session
                .dealer()
                .listen_for("hm://pusher/v1/connections/", |msg| {
                    Ok(msg
                        .headers
                        .get("Spotify-Connection-Id")
                        .cloned()
                        .unwrap_or_default())
                })
                .expect("listen connection id"),
        )
    } else {
        None
    };

    if let Err(e) = session.connect(credentials, cache_dir.is_some()).await {
        eprintln!("Session connect failed: {e}");
        exit(1);
    }

    if let Some(mut rx) = conn_id_rx.take() {
        println!("=== 0. Connect Cluster State ===");
        session.dealer().start().await.expect("start dealer");
        if let Some(Ok(conn_id)) = rx.next().await {
            session.set_connection_id(&conn_id);
            let req = PutStateRequest {
                member_type: EnumOrUnknown::new(MemberType::ACTIVE_DEVICE_TRACKER),
                put_state_reason: EnumOrUnknown::new(PutStateReason::NEW_CONNECTION),
                is_active: false,
                device: MessageField::some(Device {
                    device_info: MessageField::some(DeviceInfo {
                        device_id: session.device_id().to_string(),
                        name: "inspect_mix".to_string(),
                        can_play: false,
                        client_id: session.client_id(),
                        capabilities: MessageField::some(Capabilities {
                            hidden: true,
                            can_be_player: false,
                            is_controllable: false,
                            is_observable: false,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            match session.spclient().put_connect_state_request(&req).await {
                Ok(raw) => match Cluster::parse_from_bytes(&raw) {
                    Ok(cluster) => {
                        println!("active_device_id = {:?}", cluster.active_device_id);
                        for (id, info) in &cluster.device {
                            let cap = info.capabilities.as_ref();
                            println!("\nDevice {id:?} (name={:?}):", info.name);
                            println!(
                                "  type={:?} software={:?} spirc={:?} client_id={:?} brand={:?} model={:?} license={:?} is_group={}",
                                info.device_type,
                                info.device_software_version,
                                info.spirc_version,
                                info.client_id,
                                info.brand,
                                info.model,
                                info.license,
                                info.is_group,
                            );
                            if !info.metadata_map.is_empty() {
                                println!("  metadata_map = {:?}", info.metadata_map);
                            }
                            if let Some(c) = cap {
                                println!("  capabilities = {c:#?}");
                                let unk =
                                    summarize_unknown_fields(c.special_fields().unknown_fields());
                                if !unk.is_empty() {
                                    println!("  capabilities unknown_fields = {unk:?}");
                                }
                            }
                            let unk_d =
                                summarize_unknown_fields(info.special_fields().unknown_fields());
                            if !unk_d.is_empty() {
                                println!("  device_info unknown_fields = {unk_d:?}");
                            }
                        }
                    }
                    Err(e) => eprintln!("Cluster parse failed: {e}"),
                },
                Err(e) => eprintln!("put_connect_state_request failed: {e}"),
            }
            let _ = session.spclient().delete_connect_state_request().await;
        }
    }

    let Some((playlist_id, canonical_playlist_uri)) = playlist_id_and_uri else {
        return;
    };

    if let Some(ref dir) = save_fixtures {
        fs::create_dir_all(dir).expect("create fixture dir");
    }

    println!(
        "=== 1. playlist4 (/playlist/v2/playlist/{}) ===",
        playlist_id.to_base62()
    );
    let mut track_uris: Vec<String> = Vec::new();
    match session.spclient().get_playlist(&playlist_id).await {
        Ok(raw) => {
            if let Some(ref dir) = save_fixtures {
                fs::write(dir.join("playlist4.bin"), &raw).expect("write playlist4 fixture");
            }
            match SelectedListContent::parse_from_bytes(&raw) {
                Ok(pl) => {
                    println!(
                        "length={} revision={} applied_lenses={:?}",
                        pl.length(),
                        HEXLOWER.encode(pl.revision()),
                        pl.applied_lenses.as_ref().map(|l| l
                            .states
                            .iter()
                            .map(|s| s.identifier.as_str())
                            .collect::<Vec<_>>())
                    );
                    let unknown_top =
                        summarize_unknown_fields(pl.special_fields().unknown_fields());
                    if !unknown_top.is_empty() {
                        println!("SelectedListContent unknown fields: {unknown_top:?}");
                    }
                    if let Some(attrs) = pl.attributes.as_ref() {
                        println!(
                            "attributes: format={:?} seq_ctx_template_bytes={}",
                            attrs.format,
                            attrs.sequence_context_template().len()
                        );
                        for fa in &attrs.format_attributes {
                            println!("  list format_attr: {:?} = {:?}", fa.key(), fa.value());
                        }
                        let unknown_la =
                            summarize_unknown_fields(attrs.special_fields().unknown_fields());
                        if !unknown_la.is_empty() {
                            println!("  ListAttributes unknown fields: {unknown_la:?}");
                        }
                    }
                    if let Some(contents) = pl.contents.as_ref() {
                        for (idx, item) in contents.items.iter().take(limit).enumerate() {
                            track_uris.push(item.uri().to_string());
                            let attrs = item.attributes.as_ref();
                            let item_id_hex = attrs
                                .map(|a| HEXLOWER.encode(a.item_id()))
                                .unwrap_or_default();
                            println!("  item[{idx}] uri={} item_id={item_id_hex}", item.uri());
                            if let Some(a) = attrs {
                                for fa in &a.format_attributes {
                                    let k = fa.key();
                                    let v = fa.value();
                                    println!("    format_attr: {k:?} = {v:?}");
                                    if k.contains("transition")
                                        || k.contains("recipe")
                                        || k.contains("mix")
                                    {
                                        if let Some(bytes) = try_decode_base64(v) {
                                            match Transition::parse_from_bytes(&bytes) {
                                                Ok(t) => {
                                                    println!("      decoded Transition: {t:#?}")
                                                }
                                                Err(e) => {
                                                    println!("      Transition decode err: {e}")
                                                }
                                            }
                                        }
                                    }
                                }
                                if !a.available_signals.is_empty() {
                                    println!(
                                        "    signals: {:?}",
                                        a.available_signals
                                            .iter()
                                            .map(|s| (&s.identifier, s.data.len()))
                                            .collect::<Vec<_>>()
                                    );
                                }
                                if !a.sequence_child_template().is_empty() {
                                    println!(
                                        "    sequence_child_template_bytes={}",
                                        a.sequence_child_template().len()
                                    );
                                }
                                let unknown_ia =
                                    summarize_unknown_fields(a.special_fields().unknown_fields());
                                if !unknown_ia.is_empty() {
                                    println!("    ItemAttributes unknown fields: {unknown_ia:?}");
                                }
                            }
                        }
                    }
                }
                Err(e) => eprintln!("SelectedListContent parse failed: {e}"),
            }
        }
        Err(e) => eprintln!("get_playlist failed: {e}"),
    }

    println!("\n=== 2. context-resolve (/context-resolve/v1/{canonical_playlist_uri}) ===");
    match session
        .spclient()
        .get_context(&canonical_playlist_uri)
        .await
    {
        Ok(ctx) => {
            let sorted_meta: BTreeMap<_, _> = ctx.metadata.iter().collect();
            println!("context.metadata = {sorted_meta:#?}");
            if let Some(page) = ctx.pages.first() {
                for (idx, tr) in page.tracks.iter().take(limit).enumerate() {
                    let meta: BTreeMap<_, _> = tr.metadata.iter().collect();
                    println!(
                        "  track[{idx}] uri={:?} uid={:?} metadata={meta:?}",
                        tr.uri, tr.uid
                    );
                }
            }
        }
        Err(e) => eprintln!("get_context failed: {e}"),
    }

    println!("\n=== 3. extended-metadata (playlist extensions) ===");
    let playlist_kinds = vec![
        ExtensionKind::MIX_STATE,
        ExtensionKind::AUTOMIX_MODE,
        ExtensionKind::PLAYLIST_ATTRIBUTES_V2,
        ExtensionKind::LIST_ATTRIBUTES_V2,
        ExtensionKind::LIST_METADATA,
        ExtensionKind::LIST_METADATA_V2,
        ExtensionKind::TRANSITION_DATA,
        ExtensionKind::TRANSITION_MAPS,
        ExtensionKind::PLAYLIST_TUNER,
        ExtensionKind::AUTO_LENS,
    ];
    match session
        .spclient()
        .get_extended_metadata_entries(vec![(canonical_playlist_uri.clone(), playlist_kinds)])
        .await
    {
        Ok(entries) => {
            for e in entries {
                if let Some(ref dir) = save_fixtures {
                    if !e.data.is_empty() {
                        let fname = format!("playlist_{:?}.bin", e.kind);
                        fs::write(dir.join(fname), &e.data).expect("write fixture");
                    }
                }
                print_extension(e.kind, &e.entity_uri, e.status_code, &e.type_url, &e.data);
            }
        }
        Err(e) => eprintln!("playlist extended-metadata failed: {e}"),
    }

    println!(
        "\n=== 4. extended-metadata (track extensions, first {} tracks) ===",
        track_uris.len()
    );
    if !track_uris.is_empty() {
        let track_kinds = vec![
            ExtensionKind::BEATS,
            ExtensionKind::AUDIO_ATTRIBUTES_V2,
            ExtensionKind::MIXABILITY,
            ExtensionKind::VOCAL_ACTIVITY,
            ExtensionKind::CUEPOINTS,
            ExtensionKind::LIST_TUNER_AUDIO_ANALYSIS,
            ExtensionKind::LIST_TUNER_CUEPOINTS,
            ExtensionKind::TRANSITION_DATA,
            ExtensionKind::COMMON_TRANSITION_POINT,
            ExtensionKind::TRACK_PAIR_TRANSITION,
            ExtensionKind::TRANSITION_MAPS,
            ExtensionKind::SPEEDABILITY,
            ExtensionKind::TRACK_SECTIONS,
        ];
        let reqs: Vec<_> = track_uris
            .iter()
            .map(|u| (u.clone(), track_kinds.clone()))
            .collect();
        match session.spclient().get_extended_metadata_entries(reqs).await {
            Ok(entries) => {
                for e in entries {
                    if let Some(ref dir) = save_fixtures {
                        if !e.data.is_empty() {
                            let safe_id = e.entity_uri.rsplit(':').next().unwrap_or("entity");
                            let fname = format!("track_{safe_id}_{:?}.bin", e.kind);
                            fs::write(dir.join(fname), &e.data).expect("write fixture");
                        }
                    }
                    print_extension(e.kind, &e.entity_uri, e.status_code, &e.type_url, &e.data);
                }
            }
            Err(e) => eprintln!("track extended-metadata failed: {e}"),
        }
    }
}
