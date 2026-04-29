from __future__ import annotations

import json

import pytest

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


@pytest.mark.parametrize(
    ("status_code", "path", "payload", "expected"),
    [
        (
            403,
            "/me/player/play",
            {"error": {"message": "Premium required"}},
            "Spotify rejected this playback request. Playback control usually requires a Spotify Premium account and an active Spotify Connect device.",
        ),
        (
            404,
            "/me/player",
            {"error": {"message": "Device not found"}},
            "Spotify could not find an active playback device or player session for this request.",
        ),
        (
            429,
            "/search",
            {"error": {"message": "rate limit"}},
            "Spotify rate limit exceeded. Retry after 7 seconds.",
        ),
    ],
)
def test_spotify_client_formats_friendly_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    path: str,
    payload: dict,
    expected: str,
) -> None:
    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        lambda **kwargs: {
            "access_token": "token-1",
            "base_url": "https://api.spotify.com/v1",
        },
    )

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        return _FakeResponse(status_code, payload, headers={"content-type": "application/json", "Retry-After": "7"})

    monkeypatch.setattr(spotify_mod.httpx, "request", fake_request)

    client = spotify_mod.SpotifyClient()
    with pytest.raises(spotify_mod.SpotifyAPIError) as exc:
        client.request("GET", path)

    assert str(exc.value) == expected


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


def test_spotify_playback_get_currently_playing_returns_explanatory_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        spotify_tool,
        "_spotify_client",
        lambda: _StubSpotifyClient({
            "status_code": 204,
            "empty": True,
            "message": "Spotify is not currently playing anything. Start playback in Spotify and try again.",
        }),
    )

    payload = json.loads(spotify_tool._handle_spotify_playback({"action": "get_currently_playing"}))

    assert payload == {
        "success": True,
        "action": "get_currently_playing",
        "is_playing": False,
        "status_code": 204,
        "message": "Spotify is not currently playing anything. Start playback in Spotify and try again.",
    }


def test_library_contains_uses_generic_library_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        lambda **kwargs: {
            "access_token": "token-1",
            "base_url": "https://api.spotify.com/v1",
        },
    )

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        seen.append((method, url, params))
        return _FakeResponse(200, [True])

    monkeypatch.setattr(spotify_mod.httpx, "request", fake_request)

    client = spotify_mod.SpotifyClient()
    payload = client.library_contains(uris=["spotify:album:abc", "spotify:track:def"])

    assert payload == [True]
    assert seen == [
        (
            "GET",
            "https://api.spotify.com/v1/me/library/contains",
            {"uris": "spotify:album:abc,spotify:track:def"},
        )
    ]


@pytest.mark.parametrize(
    ("method_name", "item_key", "item_value", "expected_uris"),
    [
        ("remove_saved_tracks", "track_ids", ["track-a", "track-b"], ["spotify:track:track-a", "spotify:track:track-b"]),
        ("remove_saved_albums", "album_ids", ["album-a"], ["spotify:album:album-a"]),
    ],
)
def test_library_remove_uses_generic_library_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    item_key: str,
    item_value: list[str],
    expected_uris: list[str],
) -> None:
    seen: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        spotify_mod,
        "resolve_spotify_runtime_credentials",
        lambda **kwargs: {
            "access_token": "token-1",
            "base_url": "https://api.spotify.com/v1",
        },
    )

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        seen.append((method, url, params))
        return _FakeResponse(200, {})

    monkeypatch.setattr(spotify_mod.httpx, "request", fake_request)

    client = spotify_mod.SpotifyClient()
    getattr(client, method_name)(**{item_key: item_value})

    assert seen == [
        (
            "DELETE",
            "https://api.spotify.com/v1/me/library",
            {"uris": ",".join(expected_uris)},
        )
    ]



def test_spotify_library_tracks_list_routes_to_saved_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _LibStub:
        def get_saved_tracks(self, **kw):
            seen.append("tracks")
            return {"items": [], "total": 0}

        def get_saved_albums(self, **kw):
            seen.append("albums")
            return {"items": [], "total": 0}

    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: _LibStub())
    json.loads(spotify_tool._handle_spotify_library({"kind": "tracks", "action": "list"}))
    assert seen == ["tracks"]


def test_spotify_library_albums_list_routes_to_saved_albums(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _LibStub:
        def get_saved_tracks(self, **kw):
            seen.append("tracks")
            return {"items": [], "total": 0}

        def get_saved_albums(self, **kw):
            seen.append("albums")
            return {"items": [], "total": 0}

    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: _LibStub())
    json.loads(spotify_tool._handle_spotify_library({"kind": "albums", "action": "list"}))
    assert seen == ["albums"]


def test_spotify_library_rejects_missing_kind() -> None:
    payload = json.loads(spotify_tool._handle_spotify_library({"action": "list"}))
    assert "kind" in (payload.get("error") or "").lower()


def test_spotify_playback_recently_played_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """recently_played is now an action on spotify_playback (folded from spotify_activity)."""
    seen: list[dict] = []

    class _RecentStub:
        def get_recently_played(self, **kw):
            seen.append(kw)
            return {"items": [{"track": {"name": "x"}}]}

    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: _RecentStub())
    payload = json.loads(spotify_tool._handle_spotify_playback({"action": "recently_played", "limit": 5}))
    assert seen and seen[0]["limit"] == 5
    assert isinstance(payload, dict)
