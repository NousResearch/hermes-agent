"""Native Spotify tools for Hermes."""

from __future__ import annotations

from typing import Any, Dict, List

from hermes_cli.auth import get_auth_status
from tools.providers.spotify_client import (
    SpotifyAPIError,
    SpotifyAuthRequiredError,
    SpotifyClient,
    SpotifyError,
    normalize_spotify_id,
    normalize_spotify_uri,
    normalize_spotify_uris,
)
from tools.registry import registry, tool_error, tool_result


def _check_spotify_available() -> bool:
    try:
        return bool(get_auth_status("spotify").get("logged_in"))
    except Exception:
        return False


def _spotify_client() -> SpotifyClient:
    return SpotifyClient()


def _spotify_tool_error(exc: Exception) -> str:
    if isinstance(exc, (SpotifyError, SpotifyAuthRequiredError)):
        return tool_error(str(exc))
    if isinstance(exc, SpotifyAPIError):
        return tool_error(str(exc), status_code=exc.status_code)
    return tool_error(f"Spotify tool failed: {type(exc).__name__}: {exc}")


def _coerce_limit(raw: Any, *, default: int = 20, minimum: int = 1, maximum: int = 50) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        cleaned = raw.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return default


def _as_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _describe_empty_playback(payload: Any, *, action: str) -> dict | None:
    if not isinstance(payload, dict) or not payload.get("empty"):
        return None
    if action == "get_currently_playing":
        return {
            "success": True,
            "action": action,
            "is_playing": False,
            "status_code": payload.get("status_code", 204),
            "message": payload.get("message") or "Spotify is not currently playing anything.",
        }
    if action == "get_state":
        return {
            "success": True,
            "action": action,
            "has_active_device": False,
            "status_code": payload.get("status_code", 204),
            "message": payload.get("message") or "No active Spotify playback session was found.",
        }
    return None


def _handle_spotify_playback(args: dict, **kw) -> str:
    action = str(args.get("action") or "get_state").strip().lower()
    client = _spotify_client()
    try:
        if action == "get_state":
            payload = client.get_playback_state(market=args.get("market"))
            empty_result = _describe_empty_playback(payload, action=action)
            return tool_result(empty_result or payload)
        if action == "get_currently_playing":
            payload = client.get_currently_playing(market=args.get("market"))
            empty_result = _describe_empty_playback(payload, action=action)
            return tool_result(empty_result or payload)
        if action == "play":
            offset = args.get("offset")
            if isinstance(offset, dict):
                payload_offset = {k: v for k, v in offset.items() if v is not None}
            else:
                payload_offset = None
            uris = normalize_spotify_uris(_as_list(args.get("uris")), "track") if args.get("uris") else None
            context_uri = None
            if args.get("context_uri"):
                raw_context = str(args.get("context_uri"))
                context_type = None
                if raw_context.startswith("spotify:album:") or "/album/" in raw_context:
                    context_type = "album"
                elif raw_context.startswith("spotify:playlist:") or "/playlist/" in raw_context:
                    context_type = "playlist"
                elif raw_context.startswith("spotify:artist:") or "/artist/" in raw_context:
                    context_type = "artist"
                context_uri = normalize_spotify_uri(raw_context, context_type)
            result = client.start_playback(
                device_id=args.get("device_id"),
                context_uri=context_uri,
                uris=uris,
                offset=payload_offset,
                position_ms=args.get("position_ms"),
            )
            return tool_result({"success": True, "action": action, "result": result})
        if action == "pause":
            result = client.pause_playback(device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        if action == "next":
            result = client.skip_next(device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        if action == "previous":
            result = client.skip_previous(device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        if action == "seek":
            if args.get("position_ms") is None:
                return tool_error("position_ms is required for action='seek'")
            result = client.seek(position_ms=int(args["position_ms"]), device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        if action == "set_repeat":
            state = str(args.get("state") or "").strip().lower()
            if state not in {"track", "context", "off"}:
                return tool_error("state must be one of: track, context, off")
            result = client.set_repeat(state=state, device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        if action == "set_shuffle":
            result = client.set_shuffle(state=_coerce_bool(args.get("state")), device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        if action == "set_volume":
            if args.get("volume_percent") is None:
                return tool_error("volume_percent is required for action='set_volume'")
            result = client.set_volume(volume_percent=max(0, min(100, int(args["volume_percent"]))), device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "result": result})
        return tool_error(f"Unknown spotify_playback action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_devices(args: dict, **kw) -> str:
    action = str(args.get("action") or "list").strip().lower()
    client = _spotify_client()
    try:
        if action == "list":
            return tool_result(client.get_devices())
        if action == "transfer":
            device_id = str(args.get("device_id") or "").strip()
            if not device_id:
                return tool_error("device_id is required for action='transfer'")
            result = client.transfer_playback(device_id=device_id, play=_coerce_bool(args.get("play")))
            return tool_result({"success": True, "action": action, "result": result})
        return tool_error(f"Unknown spotify_devices action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_queue(args: dict, **kw) -> str:
    action = str(args.get("action") or "get").strip().lower()
    client = _spotify_client()
    try:
        if action == "get":
            return tool_result(client.get_queue())
        if action == "add":
            uri = normalize_spotify_uri(str(args.get("uri") or ""), None)
            result = client.add_to_queue(uri=uri, device_id=args.get("device_id"))
            return tool_result({"success": True, "action": action, "uri": uri, "result": result})
        return tool_error(f"Unknown spotify_queue action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_search(args: dict, **kw) -> str:
    client = _spotify_client()
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")
    raw_types = _as_list(args.get("types") or args.get("type") or ["track"])
    search_types = [value.lower() for value in raw_types if value.lower() in {"album", "artist", "playlist", "track", "show", "episode", "audiobook"}]
    if not search_types:
        return tool_error("types must contain one or more of: album, artist, playlist, track, show, episode, audiobook")
    try:
        return tool_result(client.search(
            query=query,
            search_types=search_types,
            limit=_coerce_limit(args.get("limit"), default=10),
            offset=max(0, int(args.get("offset") or 0)),
            market=args.get("market"),
            include_external=args.get("include_external"),
        ))
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_playlists(args: dict, **kw) -> str:
    action = str(args.get("action") or "list").strip().lower()
    client = _spotify_client()
    try:
        if action == "list":
            return tool_result(client.get_my_playlists(
                limit=_coerce_limit(args.get("limit"), default=20),
                offset=max(0, int(args.get("offset") or 0)),
            ))
        if action == "get":
            playlist_id = normalize_spotify_id(str(args.get("playlist_id") or ""), "playlist")
            return tool_result(client.get_playlist(playlist_id=playlist_id, market=args.get("market")))
        if action == "create":
            name = str(args.get("name") or "").strip()
            if not name:
                return tool_error("name is required for action='create'")
            return tool_result(client.create_playlist(
                name=name,
                public=_coerce_bool(args.get("public")),
                collaborative=_coerce_bool(args.get("collaborative")),
                description=args.get("description"),
            ))
        if action == "add_items":
            playlist_id = normalize_spotify_id(str(args.get("playlist_id") or ""), "playlist")
            uris = normalize_spotify_uris(_as_list(args.get("uris")))
            return tool_result(client.add_playlist_items(
                playlist_id=playlist_id,
                uris=uris,
                position=args.get("position"),
            ))
        if action == "remove_items":
            playlist_id = normalize_spotify_id(str(args.get("playlist_id") or ""), "playlist")
            uris = normalize_spotify_uris(_as_list(args.get("uris")))
            return tool_result(client.remove_playlist_items(
                playlist_id=playlist_id,
                uris=uris,
                snapshot_id=args.get("snapshot_id"),
            ))
        if action == "update_details":
            playlist_id = normalize_spotify_id(str(args.get("playlist_id") or ""), "playlist")
            return tool_result(client.update_playlist_details(
                playlist_id=playlist_id,
                name=args.get("name"),
                public=args.get("public"),
                collaborative=args.get("collaborative"),
                description=args.get("description"),
            ))
        return tool_error(f"Unknown spotify_playlists action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_albums(args: dict, **kw) -> str:
    action = str(args.get("action") or "get").strip().lower()
    client = _spotify_client()
    try:
        album_id = normalize_spotify_id(str(args.get("album_id") or args.get("id") or ""), "album")
        if action == "get":
            return tool_result(client.get_album(album_id=album_id, market=args.get("market")))
        if action == "tracks":
            return tool_result(client.get_album_tracks(
                album_id=album_id,
                limit=_coerce_limit(args.get("limit"), default=20),
                offset=max(0, int(args.get("offset") or 0)),
                market=args.get("market"),
            ))
        return tool_error(f"Unknown spotify_albums action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_saved_tracks(args: dict, **kw) -> str:
    action = str(args.get("action") or "list").strip().lower()
    client = _spotify_client()
    try:
        if action == "list":
            return tool_result(client.get_saved_tracks(
                limit=_coerce_limit(args.get("limit"), default=20),
                offset=max(0, int(args.get("offset") or 0)),
                market=args.get("market"),
            ))
        if action == "save":
            uris = normalize_spotify_uris(_as_list(args.get("uris") or args.get("items")), "track")
            return tool_result(client.save_library_items(uris=uris))
        if action == "remove":
            track_ids = [normalize_spotify_id(item, "track") for item in _as_list(args.get("ids") or args.get("items"))]
            if not track_ids:
                return tool_error("ids/items is required for action='remove'")
            return tool_result(client.remove_saved_tracks(track_ids=track_ids))
        return tool_error(f"Unknown spotify_saved_tracks action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_saved_albums(args: dict, **kw) -> str:
    action = str(args.get("action") or "list").strip().lower()
    client = _spotify_client()
    try:
        if action == "list":
            return tool_result(client.get_saved_albums(
                limit=_coerce_limit(args.get("limit"), default=20),
                offset=max(0, int(args.get("offset") or 0)),
                market=args.get("market"),
            ))
        if action == "save":
            uris = normalize_spotify_uris(_as_list(args.get("uris") or args.get("items")), "album")
            return tool_result(client.save_library_items(uris=uris))
        if action == "remove":
            album_ids = [normalize_spotify_id(item, "album") for item in _as_list(args.get("ids") or args.get("items"))]
            if not album_ids:
                return tool_error("ids/items is required for action='remove'")
            return tool_result(client.remove_saved_albums(album_ids=album_ids))
        return tool_error(f"Unknown spotify_saved_albums action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


def _handle_spotify_activity(args: dict, **kw) -> str:
    action = str(args.get("action") or "now_playing").strip().lower()
    client = _spotify_client()
    try:
        if action == "now_playing":
            payload = client.get_currently_playing(market=args.get("market"))
            if isinstance(payload, dict) and payload.get("empty"):
                return tool_result({
                    "success": True,
                    "action": action,
                    "is_playing": False,
                    "status_code": payload.get("status_code", 204),
                    "message": payload.get("message") or "Spotify is not currently playing anything.",
                })
            return tool_result(payload)
        if action == "recently_played":
            after = args.get("after")
            before = args.get("before")
            if after and before:
                return tool_error("Provide only one of 'after' or 'before'")
            return tool_result(client.get_recently_played(
                limit=_coerce_limit(args.get("limit"), default=20),
                after=int(after) if after is not None else None,
                before=int(before) if before is not None else None,
            ))
        return tool_error(f"Unknown spotify_activity action: {action}")
    except Exception as exc:
        return _spotify_tool_error(exc)


COMMON_STRING = {"type": "string"}

SPOTIFY_PLAYBACK_SCHEMA = {
    "name": "spotify_playback",
    "description": "Control Spotify playback or inspect the active playback state.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get_state", "get_currently_playing", "play", "pause", "next", "previous", "seek", "set_repeat", "set_shuffle", "set_volume"]},
            "device_id": COMMON_STRING,
            "market": COMMON_STRING,
            "context_uri": COMMON_STRING,
            "uris": {"type": "array", "items": COMMON_STRING},
            "offset": {"type": "object"},
            "position_ms": {"type": "integer"},
            "state": {"description": "For set_repeat use track/context/off. For set_shuffle use boolean-like true/false.", "oneOf": [{"type": "string"}, {"type": "boolean"}]},
            "volume_percent": {"type": "integer"},
        },
        "required": ["action"],
    },
}

SPOTIFY_DEVICES_SCHEMA = {
    "name": "spotify_devices",
    "description": "List Spotify Connect devices or transfer playback to a different device.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "transfer"]},
            "device_id": COMMON_STRING,
            "play": {"type": "boolean"},
        },
        "required": ["action"],
    },
}

SPOTIFY_QUEUE_SCHEMA = {
    "name": "spotify_queue",
    "description": "Inspect the user's Spotify queue or add an item to it.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "add"]},
            "uri": COMMON_STRING,
            "device_id": COMMON_STRING,
        },
        "required": ["action"],
    },
}

SPOTIFY_SEARCH_SCHEMA = {
    "name": "spotify_search",
    "description": "Search the Spotify catalog for tracks, albums, artists, playlists, shows, or episodes.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": COMMON_STRING,
            "types": {"type": "array", "items": COMMON_STRING},
            "type": COMMON_STRING,
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "market": COMMON_STRING,
            "include_external": COMMON_STRING,
        },
        "required": ["query"],
    },
}

SPOTIFY_PLAYLISTS_SCHEMA = {
    "name": "spotify_playlists",
    "description": "List, inspect, create, update, and modify Spotify playlists.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "get", "create", "add_items", "remove_items", "update_details"]},
            "playlist_id": COMMON_STRING,
            "market": COMMON_STRING,
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "name": COMMON_STRING,
            "description": COMMON_STRING,
            "public": {"type": "boolean"},
            "collaborative": {"type": "boolean"},
            "uris": {"type": "array", "items": COMMON_STRING},
            "position": {"type": "integer"},
            "snapshot_id": COMMON_STRING,
        },
        "required": ["action"],
    },
}

SPOTIFY_ALBUMS_SCHEMA = {
    "name": "spotify_albums",
    "description": "Fetch Spotify album metadata or album tracks.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "tracks"]},
            "album_id": COMMON_STRING,
            "id": COMMON_STRING,
            "market": COMMON_STRING,
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
        "required": ["action"],
    },
}

SPOTIFY_SAVED_TRACKS_SCHEMA = {
    "name": "spotify_saved_tracks",
    "description": "List, save, or remove the user's saved Spotify tracks.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "save", "remove"]},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "market": COMMON_STRING,
            "uris": {"type": "array", "items": COMMON_STRING},
            "ids": {"type": "array", "items": COMMON_STRING},
            "items": {"type": "array", "items": COMMON_STRING},
        },
        "required": ["action"],
    },
}

SPOTIFY_SAVED_ALBUMS_SCHEMA = {
    "name": "spotify_saved_albums",
    "description": "List, save, or remove the user's saved Spotify albums.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "save", "remove"]},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "market": COMMON_STRING,
            "uris": {"type": "array", "items": COMMON_STRING},
            "ids": {"type": "array", "items": COMMON_STRING},
            "items": {"type": "array", "items": COMMON_STRING},
        },
        "required": ["action"],
    },
}

SPOTIFY_ACTIVITY_SCHEMA = {
    "name": "spotify_activity",
    "description": "Inspect now playing or recently played Spotify activity.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["now_playing", "recently_played"]},
            "market": COMMON_STRING,
            "limit": {"type": "integer"},
            "after": {"type": "integer"},
            "before": {"type": "integer"},
        },
        "required": ["action"],
    },
}


registry.register(name="spotify_playback", toolset="spotify", schema=SPOTIFY_PLAYBACK_SCHEMA, handler=_handle_spotify_playback, check_fn=_check_spotify_available, emoji="🎵")
registry.register(name="spotify_devices", toolset="spotify", schema=SPOTIFY_DEVICES_SCHEMA, handler=_handle_spotify_devices, check_fn=_check_spotify_available, emoji="🔈")
registry.register(name="spotify_queue", toolset="spotify", schema=SPOTIFY_QUEUE_SCHEMA, handler=_handle_spotify_queue, check_fn=_check_spotify_available, emoji="📻")
registry.register(name="spotify_search", toolset="spotify", schema=SPOTIFY_SEARCH_SCHEMA, handler=_handle_spotify_search, check_fn=_check_spotify_available, emoji="🔎")
registry.register(name="spotify_playlists", toolset="spotify", schema=SPOTIFY_PLAYLISTS_SCHEMA, handler=_handle_spotify_playlists, check_fn=_check_spotify_available, emoji="📚")
registry.register(name="spotify_albums", toolset="spotify", schema=SPOTIFY_ALBUMS_SCHEMA, handler=_handle_spotify_albums, check_fn=_check_spotify_available, emoji="💿")
registry.register(name="spotify_saved_tracks", toolset="spotify", schema=SPOTIFY_SAVED_TRACKS_SCHEMA, handler=_handle_spotify_saved_tracks, check_fn=_check_spotify_available, emoji="❤️")
registry.register(name="spotify_saved_albums", toolset="spotify", schema=SPOTIFY_SAVED_ALBUMS_SCHEMA, handler=_handle_spotify_saved_albums, check_fn=_check_spotify_available, emoji="💽")
registry.register(name="spotify_activity", toolset="spotify", schema=SPOTIFY_ACTIVITY_SCHEMA, handler=_handle_spotify_activity, check_fn=_check_spotify_available, emoji="🕘")
