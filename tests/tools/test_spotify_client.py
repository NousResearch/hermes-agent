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


def test_normalize_spotify_id_accepts_plain_and_locale_prefixed_urls() -> None:
    """A share link copied from a localized web player must parse the same.

    Spotify hands out /intl-<locale>/<type>/<id> from every non-English web
    player, and its router treats any intl-* first segment as a locale. Reading
    the path positionally without dropping that segment mistakes the locale for
    the type.
    """
    plain = "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P"
    assert spotify_mod.normalize_spotify_id(plain, "track") == "7ouMYWpwJ422jRcDASZB7P"

    localized = "https://open.spotify.com/intl-de/track/7ouMYWpwJ422jRcDASZB7P"
    assert spotify_mod.normalize_spotify_id(localized, "track") == "7ouMYWpwJ422jRcDASZB7P"

    # Locale subtags are not always two letters: /intl-pt-br/ and /intl-zh-tw/
    # are both live share-url shapes.
    subtagged = "https://open.spotify.com/intl-pt-br/album/1DFixLWuPkv3KT3TnV35m3"
    assert spotify_mod.normalize_spotify_id(subtagged, "album") == "1DFixLWuPkv3KT3TnV35m3"

    # Without expected_type the old code returned the *type* segment as the id
    # instead of raising, so this asserts the quiet half of the bug too.
    assert spotify_mod.normalize_spotify_id(localized) == "7ouMYWpwJ422jRcDASZB7P"

    # The ?si= share tracker rides along on copied links.
    tracked = "https://open.spotify.com/intl-de/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc123"
    assert spotify_mod.normalize_spotify_id(tracked, "playlist") == "37i9dQZF1DXcBWIGoYBM5M"

    # A wrong type is still rejected once the locale is out of the way.
    with pytest.raises(spotify_mod.SpotifyError):
        spotify_mod.normalize_spotify_id(localized, "album")


def test_normalize_spotify_uri_accepts_locale_prefixed_urls() -> None:
    for url, item_type, expected in (
        (
            "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P",
            "track",
            "spotify:track:7ouMYWpwJ422jRcDASZB7P",
        ),
        (
            "https://open.spotify.com/intl-de/track/7ouMYWpwJ422jRcDASZB7P",
            "track",
            "spotify:track:7ouMYWpwJ422jRcDASZB7P",
        ),
        (
            "https://open.spotify.com/intl-pt-br/album/1DFixLWuPkv3KT3TnV35m3",
            "album",
            "spotify:album:1DFixLWuPkv3KT3TnV35m3",
        ),
        (
            "https://open.spotify.com/intl-de/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc123",
            "playlist",
            "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
        ),
    ):
        assert spotify_mod.normalize_spotify_uri(url, item_type) == expected


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
