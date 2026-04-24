"""Thin Spotify Web API helper used by Hermes native tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import httpx

from hermes_cli.auth import (
    AuthError,
    resolve_spotify_runtime_credentials,
)


class SpotifyError(RuntimeError):
    """Base Spotify tool error."""


class SpotifyAuthRequiredError(SpotifyError):
    """Raised when the user needs to authenticate with Spotify first."""


class SpotifyAPIError(SpotifyError):
    """Structured Spotify API failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.path = None


class SpotifyClient:
    def __init__(self) -> None:
        self._runtime = self._resolve_runtime(refresh_if_expiring=True)

    def _resolve_runtime(self, *, force_refresh: bool = False, refresh_if_expiring: bool = True) -> Dict[str, Any]:
        try:
            return resolve_spotify_runtime_credentials(
                force_refresh=force_refresh,
                refresh_if_expiring=refresh_if_expiring,
            )
        except AuthError as exc:
            raise SpotifyAuthRequiredError(str(exc)) from exc

    @property
    def base_url(self) -> str:
        return str(self._runtime.get("base_url") or "").rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._runtime['access_token']}",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        allow_retry_on_401: bool = True,
        empty_response: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        response = httpx.request(
            method,
            url,
            headers=self._headers(),
            params=_strip_none(params),
            json=_strip_none(json_body) if json_body is not None else None,
            timeout=30.0,
        )
        if response.status_code == 401 and allow_retry_on_401:
            self._runtime = self._resolve_runtime(force_refresh=True, refresh_if_expiring=True)
            return self.request(
                method,
                path,
                params=params,
                json_body=json_body,
                allow_retry_on_401=False,
            )
        if response.status_code >= 400:
            self._raise_api_error(response, method=method, path=path)
        if response.status_code == 204 or not response.content:
            return empty_response or {"success": True, "status_code": response.status_code, "empty": True}
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return {"success": True, "text": response.text}

    def _raise_api_error(self, response: httpx.Response, *, method: str, path: str) -> None:
        detail = response.text.strip()
        message = _friendly_spotify_error_message(
            status_code=response.status_code,
            detail=_extract_spotify_error_detail(response, fallback=detail),
            method=method,
            path=path,
            retry_after=response.headers.get("Retry-After"),
        )
        error = SpotifyAPIError(message, status_code=response.status_code, response_body=detail)
        error.path = path
        raise error

    def get_devices(self) -> Any:
        return self.request("GET", "/me/player/devices")

    def transfer_playback(self, *, device_id: str, play: bool = False) -> Any:
        return self.request("PUT", "/me/player", json_body={
            "device_ids": [device_id],
            "play": play,
        })

    def get_playback_state(self, *, market: Optional[str] = None) -> Any:
        return self.request(
            "GET",
            "/me/player",
            params={"market": market},
            empty_response={
                "status_code": 204,
                "empty": True,
                "message": "No active Spotify playback session was found. Open Spotify on a device and start playback, or transfer playback to an available device.",
            },
        )

    def get_currently_playing(self, *, market: Optional[str] = None) -> Any:
        return self.request(
            "GET",
            "/me/player/currently-playing",
            params={"market": market},
            empty_response={
                "status_code": 204,
                "empty": True,
                "message": "Spotify is not currently playing anything. Start playback in Spotify and try again.",
            },
        )

    def start_playback(
        self,
        *,
        device_id: Optional[str] = None,
        context_uri: Optional[str] = None,
        uris: Optional[list[str]] = None,
        offset: Optional[Dict[str, Any]] = None,
        position_ms: Optional[int] = None,
    ) -> Any:
        return self.request(
            "PUT",
            "/me/player/play",
            params={"device_id": device_id},
            json_body={
                "context_uri": context_uri,
                "uris": uris,
                "offset": offset,
                "position_ms": position_ms,
            },
        )

    def pause_playback(self, *, device_id: Optional[str] = None) -> Any:
        return self.request("PUT", "/me/player/pause", params={"device_id": device_id})

    def skip_next(self, *, device_id: Optional[str] = None) -> Any:
        return self.request("POST", "/me/player/next", params={"device_id": device_id})

    def skip_previous(self, *, device_id: Optional[str] = None) -> Any:
        return self.request("POST", "/me/player/previous", params={"device_id": device_id})

    def seek(self, *, position_ms: int, device_id: Optional[str] = None) -> Any:
        return self.request("PUT", "/me/player/seek", params={
            "position_ms": position_ms,
            "device_id": device_id,
        })

    def set_repeat(self, *, state: str, device_id: Optional[str] = None) -> Any:
        return self.request("PUT", "/me/player/repeat", params={"state": state, "device_id": device_id})

    def set_shuffle(self, *, state: bool, device_id: Optional[str] = None) -> Any:
        return self.request("PUT", "/me/player/shuffle", params={"state": str(bool(state)).lower(), "device_id": device_id})

    def set_volume(self, *, volume_percent: int, device_id: Optional[str] = None) -> Any:
        return self.request("PUT", "/me/player/volume", params={
            "volume_percent": volume_percent,
            "device_id": device_id,
        })

    def get_queue(self) -> Any:
        return self.request("GET", "/me/player/queue")

    def add_to_queue(self, *, uri: str, device_id: Optional[str] = None) -> Any:
        return self.request("POST", "/me/player/queue", params={"uri": uri, "device_id": device_id})

    def search(
        self,
        *,
        query: str,
        search_types: list[str],
        limit: int = 10,
        offset: int = 0,
        market: Optional[str] = None,
        include_external: Optional[str] = None,
    ) -> Any:
        return self.request("GET", "/search", params={
            "q": query,
            "type": ",".join(search_types),
            "limit": limit,
            "offset": offset,
            "market": market,
            "include_external": include_external,
        })

    def get_my_playlists(self, *, limit: int = 20, offset: int = 0) -> Any:
        return self.request("GET", "/me/playlists", params={"limit": limit, "offset": offset})

    def get_playlist(self, *, playlist_id: str, market: Optional[str] = None) -> Any:
        return self.request("GET", f"/playlists/{playlist_id}", params={"market": market})

    def create_playlist(
        self,
        *,
        name: str,
        public: bool = False,
        collaborative: bool = False,
        description: Optional[str] = None,
    ) -> Any:
        return self.request("POST", "/me/playlists", json_body={
            "name": name,
            "public": public,
            "collaborative": collaborative,
            "description": description,
        })

    def add_playlist_items(
        self,
        *,
        playlist_id: str,
        uris: list[str],
        position: Optional[int] = None,
    ) -> Any:
        return self.request("POST", f"/playlists/{playlist_id}/items", json_body={
            "uris": uris,
            "position": position,
        })

    def remove_playlist_items(
        self,
        *,
        playlist_id: str,
        uris: list[str],
        snapshot_id: Optional[str] = None,
    ) -> Any:
        return self.request("DELETE", f"/playlists/{playlist_id}/items", json_body={
            "items": [{"uri": uri} for uri in uris],
            "snapshot_id": snapshot_id,
        })

    def update_playlist_details(
        self,
        *,
        playlist_id: str,
        name: Optional[str] = None,
        public: Optional[bool] = None,
        collaborative: Optional[bool] = None,
        description: Optional[str] = None,
    ) -> Any:
        return self.request("PUT", f"/playlists/{playlist_id}", json_body={
            "name": name,
            "public": public,
            "collaborative": collaborative,
            "description": description,
        })

    def get_album(self, *, album_id: str, market: Optional[str] = None) -> Any:
        return self.request("GET", f"/albums/{album_id}", params={"market": market})

    def get_album_tracks(self, *, album_id: str, limit: int = 20, offset: int = 0, market: Optional[str] = None) -> Any:
        return self.request("GET", f"/albums/{album_id}/tracks", params={
            "limit": limit,
            "offset": offset,
            "market": market,
        })

    def get_saved_tracks(self, *, limit: int = 20, offset: int = 0, market: Optional[str] = None) -> Any:
        return self.request("GET", "/me/tracks", params={"limit": limit, "offset": offset, "market": market})

    def save_library_items(self, *, uris: list[str]) -> Any:
        return self.request("PUT", "/me/library", params={"uris": ",".join(uris)})

    def library_contains(self, *, uris: list[str]) -> Any:
        return self.request("GET", "/me/library/contains", params={"uris": ",".join(uris)})

    def get_saved_albums(self, *, limit: int = 20, offset: int = 0, market: Optional[str] = None) -> Any:
        return self.request("GET", "/me/albums", params={"limit": limit, "offset": offset, "market": market})

    def remove_saved_tracks(self, *, track_ids: list[str]) -> Any:
        uris = [f"spotify:track:{track_id}" for track_id in track_ids]
        return self.request("DELETE", "/me/library", params={"uris": ",".join(uris)})

    def remove_saved_albums(self, *, album_ids: list[str]) -> Any:
        uris = [f"spotify:album:{album_id}" for album_id in album_ids]
        return self.request("DELETE", "/me/library", params={"uris": ",".join(uris)})

    def get_recently_played(
        self,
        *,
        limit: int = 20,
        after: Optional[int] = None,
        before: Optional[int] = None,
    ) -> Any:
        return self.request("GET", "/me/player/recently-played", params={
            "limit": limit,
            "after": after,
            "before": before,
        })


def _extract_spotify_error_detail(response: httpx.Response, *, fallback: str) -> str:
    detail = fallback
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                detail = str(error_obj.get("message") or detail)
            elif isinstance(error_obj, str):
                detail = error_obj
    except Exception:
        pass
    return detail.strip()


def _friendly_spotify_error_message(
    *,
    status_code: int,
    detail: str,
    method: str,
    path: str,
    retry_after: Optional[str],
) -> str:
    normalized_detail = detail.lower()
    is_playback_path = path.startswith("/me/player")

    if status_code == 401:
        return "Spotify authentication failed or expired. Run `hermes auth spotify` again."

    if status_code == 403:
        if is_playback_path:
            return (
                "Spotify rejected this playback request. Playback control usually requires a Spotify Premium account "
                "and an active Spotify Connect device."
            )
        if "scope" in normalized_detail or "permission" in normalized_detail:
            return "Spotify rejected the request because the current auth scope is insufficient. Re-run `hermes auth spotify` to refresh permissions."
        return "Spotify rejected the request. The account may not have permission for this action."

    if status_code == 404:
        if is_playback_path:
            return "Spotify could not find an active playback device or player session for this request."
        return "Spotify resource not found."

    if status_code == 429:
        message = "Spotify rate limit exceeded."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        return message

    if detail:
        return detail
    return f"Spotify API request failed with status {status_code}."


def _strip_none(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not payload:
        return {}
    return {key: value for key, value in payload.items() if value is not None}


def normalize_spotify_id(value: str, expected_type: Optional[str] = None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise SpotifyError("Spotify id/uri/url is required.")
    if cleaned.startswith("spotify:"):
        parts = cleaned.split(":")
        if len(parts) >= 3:
            item_type = parts[1]
            if expected_type and item_type != expected_type:
                raise SpotifyError(f"Expected a Spotify {expected_type}, got {item_type}.")
            return parts[2]
    if "open.spotify.com" in cleaned:
        parsed = urlparse(cleaned)
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2:
            item_type, item_id = path_parts[0], path_parts[1]
            if expected_type and item_type != expected_type:
                raise SpotifyError(f"Expected a Spotify {expected_type}, got {item_type}.")
            return item_id
    return cleaned


def normalize_spotify_uri(value: str, expected_type: Optional[str] = None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise SpotifyError("Spotify URI/url/id is required.")
    if cleaned.startswith("spotify:"):
        if expected_type:
            parts = cleaned.split(":")
            if len(parts) >= 3 and parts[1] != expected_type:
                raise SpotifyError(f"Expected a Spotify {expected_type}, got {parts[1]}.")
        return cleaned
    item_id = normalize_spotify_id(cleaned, expected_type)
    if expected_type:
        return f"spotify:{expected_type}:{item_id}"
    return cleaned


def normalize_spotify_uris(values: Iterable[str], expected_type: Optional[str] = None) -> list[str]:
    uris: list[str] = []
    for value in values:
        uri = normalize_spotify_uri(str(value), expected_type)
        if uri not in uris:
            uris.append(uri)
    if not uris:
        raise SpotifyError("At least one Spotify item is required.")
    return uris


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
