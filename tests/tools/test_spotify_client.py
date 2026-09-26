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
    payload = client.request("GET", "/me/player/devices")

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
