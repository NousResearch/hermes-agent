from __future__ import annotations

import json

import pytest

from hermes_cli.auth import AuthError
from plugins.spotify import client as spotify_mod
from plugins.spotify import tools as spotify_tool


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, *, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {"content-type": "application/json"}
        self.content = self.text.encode("utf-8") if self.text else b""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _StubSpotifyClient:
    def __init__(self, payload):
        self.payload = payload

    def get_currently_playing(self, *, market=None):
        return self.payload


def test_spotify_client_retries_once_after_401(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    tokens = iter([
        {
            "access_token": "token-1",
            "base_url": "https://api.spotify.com/v1",
        },
        {
            "access_token": "token-2",
            "base_url": "https://api.spotify.com/v1",
        },
    ])

    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        lambda **kwargs: next(tokens),
    )

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            return _FakeResponse(401, {"error": {"message": "expired token"}})
        return _FakeResponse(200, {"devices": [{"id": "dev-1"}]})

    monkeypatch.setattr(spotify_mod.httpx, "request", fake_request)

    client = spotify_mod.SpotifyClient()
    payload = client.get_devices()

    assert payload["devices"][0]["id"] == "dev-1"
    assert calls == ["Bearer token-1", "Bearer token-2"]


def test_normalize_spotify_uri_accepts_urls() -> None:
    uri = spotify_mod.normalize_spotify_uri(
        "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P",
        "track",
    )
    assert uri == "spotify:track:7ouMYWpwJ422jRcDASZB7P"


def test_get_currently_playing_returns_explanatory_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        lambda **kwargs: {
            "access_token": "token-1",
            "base_url": "https://api.spotify.com/v1",
        },
    )

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        return _FakeResponse(204, None, text="", headers={"content-type": "application/json"})

    monkeypatch.setattr(spotify_mod.httpx, "request", fake_request)

    client = spotify_mod.SpotifyClient()
    payload = client.get_currently_playing()

    assert payload == {
        "status_code": 204,
        "empty": True,
        "message": "Spotify is not currently playing anything. Start playback in Spotify and try again.",
    }


def test_client_wraps_invalid_grant_as_spotify_auth_required_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SpotifyClient._resolve_runtime wraps AuthError(code=spotify_refresh_invalid_grant) into SpotifyAuthRequiredError."""

    def _raise_invalid_grant(**kwargs):
        raise AuthError(
            "Spotify refresh token has expired or was revoked. Run `hermes auth spotify` again.",
            provider="spotify",
            code="spotify_refresh_invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        _raise_invalid_grant,
    )
    with pytest.raises(spotify_mod.SpotifyAuthRequiredError, match="expired or was revoked"):
        spotify_mod.SpotifyClient()


# ── Regression tests for Spotify URI normalization & device guidance ─────────
# Covers the normalizer contract used by the playback/queue tools, the
# dedup + non-empty contract preserved by normalize_spotify_uris in the play
# path, and the active-device guard that distinguishes the active device from
# merely-listed devices.


def test_normalize_spotify_uri_bare_id_prefixes_expected_type() -> None:
    result = spotify_mod.normalize_spotify_uri("7ouMYWpwJ422jRcDASZB7P", "track")
    assert result == "spotify:track:7ouMYWpwJ422jRcDASZB7P"


def test_normalize_spotify_uri_returns_native_uri_unchanged() -> None:
    uri = "spotify:album:0sNOF9WDwhWunNAHPD3Baj"
    assert spotify_mod.normalize_spotify_uri(uri, "album") == uri


def test_normalize_spotify_uri_open_url_canonicalizes() -> None:
    url = "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P?si=abc"
    assert spotify_mod.normalize_spotify_uri(url, "track") == "spotify:track:7ouMYWpwJ422jRcDASZB7P"


def test_normalize_spotify_uri_rejects_type_mismatch() -> None:
    with pytest.raises(spotify_mod.SpotifyError, match="Expected a Spotify track"):
        spotify_mod.normalize_spotify_uri("spotify:album:abc", "track")


def test_normalize_spotify_uri_empty_string_raises() -> None:
    with pytest.raises(spotify_mod.SpotifyError, match="Spotify URI/url/id is required"):
        spotify_mod.normalize_spotify_uri("", None)


def test_normalize_spotify_uri_none_raises() -> None:
    with pytest.raises(spotify_mod.SpotifyError, match="Spotify URI/url/id is required"):
        spotify_mod.normalize_spotify_uri(None, None)  # type: ignore[arg-type]


def test_normalize_spotify_uris_deduplicates_and_rejects_empty() -> None:
    # Deduplicates repeated entries while preserving order.
    result = spotify_mod.normalize_spotify_uris(
        ["7ouMYWpwJ422jRcDASZB7P", "7ouMYWpwJ422jRcDASZB7P"], "track"
    )
    assert result == ["spotify:track:7ouMYWpwJ422jRcDASZB7P"]
    # Rejects an empty collection instead of forwarding an empty list to the API.
    with pytest.raises(spotify_mod.SpotifyError, match="At least one Spotify item is required"):
        spotify_mod.normalize_spotify_uris([], "track")


def test_handle_spotify_queue_add_canonicalizes_bare_id_to_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue add must promote a bare search-result ID to a full track URI,
    since Spotify's POST /me/player/queue rejects anything other than a full
    spotify:track:<id> URI."""
    seen_uris: list[str] = []

    class _QueueStub:
        def get_devices(self):
            return {"devices": [{"id": "dev-1", "is_active": True}]}

        def add_to_queue(self, *, uri, device_id=None):
            seen_uris.append(uri)
            return {"snapshot_id": "snap-1"}

    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: _QueueStub())
    response = json.loads(
        spotify_tool._handle_spotify_queue(
            {"action": "add", "uri": "7ouMYWpwJ422jRcDASZB7P", "device_id": "dev-1"}
        )
    )
    assert response["success"] is True
    assert response["uri"] == "spotify:track:7ouMYWpwJ422jRcDASZB7P"
    assert seen_uris == ["spotify:track:7ouMYWpwJ422jRcDASZB7P"]


def test_handle_spotify_queue_add_passes_native_track_uri_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_uris: list[str] = []

    class _QueueStub:
        def add_to_queue(self, *, uri, device_id=None):
            seen_uris.append(uri)
            return {"snapshot_id": "snap-1"}

    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: _QueueStub())
    response = json.loads(
        spotify_tool._handle_spotify_queue(
            {"action": "add", "uri": "spotify:track:7ouMYWpwJ422jRcDASZB7P", "device_id": "dev-1"}
        )
    )
    assert response["uri"] == "spotify:track:7ouMYWpwJ422jRcDASZB7P"
    assert seen_uris == ["spotify:track:7ouMYWpwJ422jRcDASZB7P"]


def test_handle_spotify_queue_add_blocks_when_no_active_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _QueueStub:
        def get_devices(self):
            # Devices listed, but none active — must NOT be treated as available.
            return {"devices": [{"id": "dev-1", "is_active": False}]}

    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: _QueueStub())
    response = json.loads(
        spotify_tool._handle_spotify_queue({"action": "add", "uri": "spotify:track:abc"})
    )
    assert "error" in response
    assert "No active Spotify playback device" in response["error"]
