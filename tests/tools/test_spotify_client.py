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


def _queue_stub(*, active_devices=True, devices_error=None):
    """A request()-based SpotifyClient stub: records queued URIs, answers device lookups."""
    seen_uris: list[str] = []

    class _Stub:
        def request(self, method, path, params=None, json_body=None):
            if method == "GET" and path == "/me/player/devices":
                if devices_error is not None:
                    raise devices_error
                return {"devices": [{"id": "dev-1", "is_active": active_devices}]}
            if method == "POST" and path == "/me/player/queue":
                seen_uris.append((params or {}).get("uri"))
                return {"snapshot_id": "snap-1"}
            raise AssertionError(f"unexpected request: {method} {path}")

    return _Stub(), seen_uris


def test_handle_spotify_queue_add_canonicalizes_bare_id_to_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue add must promote a bare search-result ID to a full track URI,
    since Spotify's POST /me/player/queue rejects anything other than a full
    spotify:track:<id> URI."""
    stub, seen_uris = _queue_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
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
    stub, seen_uris = _queue_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
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
    """No device_id and only inactive devices: must NOT reach the queue API."""
    stub, seen_uris = _queue_stub(active_devices=False)
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_queue({"action": "add", "uri": "spotify:track:abc"})
    )
    assert "error" in response
    assert "No active Spotify playback device" in response["error"]
    assert seen_uris == []


# ── Handler-level coverage requested by the cross-PR triage review ─────────
# Covers the four gaps the consolidation review asked to confirm:
#   1. empty playback input (uris=[""] must not reach start_playback as [])
#   2. inactive-only device lists block playback, not just queue
#   3. lookup failures propagate instead of being masked as "no active device"
#   4. queue raw-ID coercion (covered above — listed here for completeness)


def _play_stub(*, active_devices=True, devices_error=None):
    """A request()-based playback stub: answers device lookups, records play bodies."""
    plays: list[dict] = []

    class _Stub:
        def request(self, method, path, params=None, json_body=None):
            if method == "GET" and path == "/me/player/devices":
                if devices_error is not None:
                    raise devices_error
                return {"devices": [{"id": "dev-1", "is_active": active_devices}]}
            if method == "PUT" and path == "/me/player/play":
                plays.append({"params": params or {}, "body": json_body or {}})
                return {}
            raise AssertionError(f"unexpected request: {method} {path}")

    return _Stub(), plays


def test_playback_play_rejects_empty_uris_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truthy uris=[''] must NOT reach start_playback as an empty list.

    The play path routes through normalize_spotify_uris (via _as_list, which
    drops blank entries), so the batch normalizer's non-empty contract still
    rejects it. Guards against the regression flagged at tools.py:134.
    """
    stub, plays = _play_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_playback({"action": "play", "uris": [""]})
    )
    assert "error" in response
    assert "At least one Spotify item" in response["error"]
    # Crucially, the API client was never called with an empty uris list.
    assert plays == []


def test_playback_play_blocks_when_only_inactive_devices_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """playback 'play' with no device_id and only inactive devices is blocked.

    A nonempty device list does not establish the active device required when
    device_id is omitted (docs contract at spotify.md:106).
    """
    stub, plays = _play_stub(active_devices=False)
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_playback(
            {"action": "play", "uris": ["spotify:track:abc"]}
        )
    )
    assert "error" in response
    assert "No active Spotify playback device" in response["error"]
    assert plays == []


def test_playback_play_propagates_device_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_devices() failures must surface, not be masked as 'no active device'.

    _has_active_device deliberately does NOT swallow lookup errors; the
    caller's top-level except surfaces them via _spotify_tool_error so the
    real cause (e.g. a 401 / network failure) stays visible.
    """
    stub, plays = _play_stub(
        devices_error=spotify_mod.SpotifyAPIError(
            "Spotify rejected this playback request. "
            "Playback control usually requires a Spotify Premium account "
            "and an active Spotify Connect device.",
            status_code=403,
        )
    )
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_playback(
            {"action": "play", "uris": ["spotify:track:abc"]}
        )
    )
    assert "error" in response
    # The real cause is surfaced, NOT the misleading "no active device" message.
    assert "No active Spotify playback device" not in response["error"]
    assert plays == []


def test_playback_play_proceeds_when_active_device_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: playback 'play' proceeds when a device with is_active=True exists."""
    stub, plays = _play_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_playback(
            {"action": "play", "uris": ["spotify:track:abc"]}
        )
    )
    assert response.get("success") is True
    assert plays and plays[0]["body"]["uris"] == ["spotify:track:abc"]


# ── Queue add must infer the item type instead of forcing "track" ────────────
# Spotify's POST /v1/me/player/queue accepts tracks *and* episodes, so a typed
# episode URI must survive the handler while unqueueable types still error and
# bare search-result IDs still default to a track.


@pytest.mark.parametrize(
    ("raw_uri", "expected_uri"),
    [
        ("spotify:track:7ouMYWpwJ422jRcDASZB7P", "spotify:track:7ouMYWpwJ422jRcDASZB7P"),
        ("spotify:episode:512ojhOuo1ktJprKbVcKyQ", "spotify:episode:512ojhOuo1ktJprKbVcKyQ"),
        (
            "https://open.spotify.com/episode/512ojhOuo1ktJprKbVcKyQ?si=abc",
            "spotify:episode:512ojhOuo1ktJprKbVcKyQ",
        ),
        ("7ouMYWpwJ422jRcDASZB7P", "spotify:track:7ouMYWpwJ422jRcDASZB7P"),
    ],
)
def test_handle_spotify_queue_add_infers_type_per_input(
    monkeypatch: pytest.MonkeyPatch, raw_uri: str, expected_uri: str
) -> None:
    stub, seen_uris = _queue_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_queue(
            {"action": "add", "uri": raw_uri, "device_id": "dev-1"}
        )
    )
    assert response["success"] is True
    assert response["uri"] == expected_uri
    # A raw open.spotify.com URL is never handed to Spotify as-is (Bad URI).
    assert seen_uris == [expected_uri]


@pytest.mark.parametrize(
    "raw_uri",
    [
        "spotify:album:0sNOF9WDwhWunNAHPD3Baj",
        "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
    ],
)
def test_handle_spotify_queue_add_rejects_unqueueable_type(
    monkeypatch: pytest.MonkeyPatch, raw_uri: str
) -> None:
    """Types Spotify cannot queue must fail before the API call."""
    stub, seen_uris = _queue_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_queue(
            {"action": "add", "uri": raw_uri, "device_id": "dev-1"}
        )
    )
    assert "error" in response
    assert "only queue a track or an episode" in response["error"]
    assert seen_uris == []


@pytest.mark.parametrize(
    "raw_uri",
    [
        "spotify:EPISODE:512ojhOuo1ktJprKbVcKyQ",
        "https://open.spotify.com/Track/7ouMYWpwJ422jRcDASZB7P",
    ],
)
def test_handle_spotify_queue_add_rejects_miscased_type(
    monkeypatch: pytest.MonkeyPatch, raw_uri: str
) -> None:
    """A queueable type in the wrong case fails locally, not as a Bad URI."""
    stub, seen_uris = _queue_stub()
    monkeypatch.setattr(spotify_tool, "_spotify_client", lambda: stub)
    response = json.loads(
        spotify_tool._handle_spotify_queue(
            {"action": "add", "uri": raw_uri, "device_id": "dev-1"}
        )
    )
    assert "error" in response
    assert "Expected a Spotify" in response["error"]
    assert seen_uris == []


def test_spotify_uri_type_reads_type_from_uri_and_url() -> None:
    assert spotify_mod.spotify_uri_type("spotify:episode:abc") == "episode"
    assert spotify_mod.spotify_uri_type("https://open.spotify.com/track/abc?si=x") == "track"
    # A bare id carries no type information.
    assert spotify_mod.spotify_uri_type("7ouMYWpwJ422jRcDASZB7P") is None
