from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

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
        assert headers is not None
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


def test_spotify_playlists_upload_cover_converts_and_sends_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = iter([
        {"access_token": "token-1", "base_url": "https://api.spotify.com/v1"},
        {"access_token": "token-2", "base_url": "https://api.spotify.com/v1"},
    ])
    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        lambda **kwargs: next(tokens),
    )
    calls: list[dict] = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return _FakeResponse(401 if len(calls) == 1 else 202, None, text="", headers={})

    monkeypatch.setattr(spotify_mod.httpx, "request", fake_request)
    source = BytesIO()
    Image.new("RGBA", (600, 400), (40, 80, 120, 128)).save(source, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(source.getvalue()).decode("ascii")

    result = json.loads(spotify_tool._handle_spotify_playlists({
        "action": "upload_cover",
        "playlist_id": "spotify:playlist:playlist-1",
        "image_url": data_url,
    }))

    assert result["success"] is True
    assert result["action"] == "upload_cover"
    assert len(calls) == 2
    assert [call["headers"]["Authorization"] for call in calls] == [
        "Bearer token-1", "Bearer token-2",
    ]
    assert all(call["method"] == "PUT" for call in calls)
    assert all(call["url"].endswith("/playlists/playlist-1/images") for call in calls)
    assert all(call["headers"]["Content-Type"] == "image/jpeg" for call in calls)
    assert all("json" not in call for call in calls)
    assert calls[0]["content"] == calls[1]["content"]
    jpeg = base64.b64decode(calls[0]["content"], validate=True)
    assert jpeg.startswith(b"\xff\xd8\xff")
    assert len(calls[0]["content"].encode("ascii")) <= spotify_tool._SPOTIFY_COVER_MAX_PAYLOAD_BYTES
    params = spotify_tool.SPOTIFY_PLAYLISTS_SCHEMA["parameters"]["properties"]
    assert "upload_cover" in params["action"]["enum"]
    assert params["image_url"]["type"] == "string"
    with pytest.raises(spotify_mod.SpotifyError, match="approved generated-media cache"):
        spotify_tool._prepare_playlist_cover("/tmp/private-photo.jpg")


def test_spotify_playlist_cover_accepts_only_profile_media_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "profile"
    approved = hermes_home / "cache" / "images" / "generated.png"
    approved.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), (40, 80, 120)).save(approved, format="PNG")
    outside = tmp_path / "private.png"
    Image.new("RGB", (32, 32), (120, 80, 40)).save(outside, format="PNG")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # The Spotify boundary must stay cache-only even for a local terminal backend.
    monkeypatch.setenv("TERMINAL_ENV", "local")

    encoded = spotify_tool._prepare_playlist_cover(str(approved))
    assert base64.b64decode(encoded, validate=True).startswith(b"\xff\xd8\xff")
    encoded_uri = spotify_tool._prepare_playlist_cover(approved.as_uri())
    assert base64.b64decode(encoded_uri, validate=True).startswith(b"\xff\xd8\xff")
    with pytest.raises(spotify_mod.SpotifyError, match="approved generated-media cache"):
        spotify_tool._prepare_playlist_cover(str(outside))

    # Agent-visible Docker cache paths map back to the same approved host cache.
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    encoded_docker = spotify_tool._prepare_playlist_cover(
        "/root/.hermes/cache/images/generated.png")
    assert base64.b64decode(encoded_docker, validate=True).startswith(b"\xff\xd8\xff")
    with pytest.raises(spotify_mod.SpotifyError, match="approved generated-media cache"):
        spotify_tool._prepare_playlist_cover(str(outside))


def test_spotify_playlist_cover_rejects_cache_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "profile"
    cache = hermes_home / "cache"
    cache.mkdir(parents=True)
    outside = tmp_path / "private.png"
    Image.new("RGB", (32, 32), (120, 80, 40)).save(outside, format="PNG")
    link = cache / "generated.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    with pytest.raises(spotify_mod.SpotifyError, match="approved generated-media cache"):
        spotify_tool._prepare_playlist_cover(str(link))

    # Even a symlink that resolves to another location inside the cache must not
    # pass the descriptor walk; otherwise a post-validation swap could escape.
    real_dir = cache / "real"
    real_dir.mkdir()
    Image.new("RGB", (32, 32), (40, 80, 120)).save(
        real_dir / "generated.png", format="PNG")
    alias = cache / "alias"
    alias.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(spotify_mod.SpotifyError, match="without following links"):
        spotify_tool._prepare_playlist_cover(str(alias / "generated.png"))

    # The allowlisted root itself may not redirect outside the profile.
    outside_cache = tmp_path / "outside-cache"
    outside_cache.mkdir()
    Image.new("RGB", (32, 32), (40, 80, 120)).save(
        outside_cache / "generated.png", format="PNG")
    root_link = hermes_home / "image_cache"
    root_link.symlink_to(outside_cache, target_is_directory=True)
    with pytest.raises(spotify_mod.SpotifyError, match="approved generated-media cache"):
        spotify_tool._prepare_playlist_cover(str(root_link / "generated.png"))

    hard_link = cache / "hard-linked.png"
    try:
        os.link(outside, hard_link)
    except OSError:
        pytest.skip("hard links unavailable")
    with pytest.raises(spotify_mod.SpotifyError, match="without following links"):
        spotify_tool._prepare_playlist_cover(str(hard_link))
