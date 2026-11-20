"""WhatsApp media delivery for send_message (#19105).

Covers two layers:

* ``_bridge_media_type`` — extension/voice/force_document -> bridge mediaType.
* ``_standalone_send`` — text-first then per-file ``/send-media`` uploads,
  media-only (skip ``/send``), and missing-file errors. The bridge HTTP calls
  are mocked at the ``aiohttp.ClientSession`` boundary.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.whatsapp.adapter import _bridge_media_type, _standalone_send
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


# ---------------------------------------------------------------------------
# _bridge_media_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,is_voice,force_document,expected",
    [
        ("a.png", False, False, "image"),
        ("a.JPG", False, False, "image"),
        ("a.jpeg", False, False, "image"),
        ("a.webp", False, False, "image"),
        ("a.gif", False, False, "image"),
        ("a.mp4", False, False, "video"),
        ("a.mov", False, False, "video"),
        ("a.webm", False, False, "video"),
        ("a.ogg", True, False, "audio"),
        ("a.opus", False, False, "audio"),
        ("a.mp3", False, False, "audio"),
        ("a.wav", False, False, "audio"),
        ("a.pdf", False, False, "document"),
        ("a.zip", False, False, "document"),
        # force_document overrides everything
        ("a.png", False, True, "document"),
        ("a.mp4", False, True, "document"),
        # is_voice wins over a video extension
        ("a.mp4", True, False, "audio"),
    ],
)
def test_bridge_media_type(path, is_voice, force_document, expected):
    assert _bridge_media_type(path, is_voice, force_document) == expected


# ---------------------------------------------------------------------------
# _standalone_send — bridge HTTP mocked
# ---------------------------------------------------------------------------


def _resp(status, json_data=None, text_data=None):
    r = AsyncMock()
    r.status = status
    r.json = AsyncMock(return_value=json_data or {})
    r.text = AsyncMock(return_value=text_data or "")
    return r


def _session_with(responses, health=None, *, token="standalone-test-token-0123456789abcdef"):
    """Build a mocked aiohttp.ClientSession: GET /auth-proof answers the challenge with *token* (the bridge the
    adapter spawned), GET /health answers *health* (default an empty 200), POSTs return *responses* in order; every
    POST and the /health GET are recorded as (url, json_payload) — the proof handshake is not part of the recorded
    delivery sequence."""
    from plugins.platforms.whatsapp.adapter import _bridge_auth_proof

    calls = []
    idx = [0]

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        r = responses[idx[0]] if idx[0] < len(responses) else responses[-1]
        idx[0] += 1
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=r)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    def _get(url, **kwargs):
        if url.endswith("/auth-proof"):
            challenge = (kwargs.get("headers") or {})["X-Hermes-Bridge-Challenge"]
            resp = _resp(200, {"authProof": _bridge_auth_proof(token, challenge)})
        else:
            calls.append((url, kwargs.get("json")))
            resp = health or _resp(200)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session.get = MagicMock(side_effect=_get)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, calls


@pytest.fixture(autouse=True)
def _persisted_bridge_token(monkeypatch):
    """Standalone senders only READ the token the gateway persisted; stand in for that file."""
    monkeypatch.setattr("plugins.platforms.whatsapp.adapter._read_bridge_token",
                        lambda session_path: "standalone-test-token-0123456789abcdef")


def _pconfig():
    return SimpleNamespace(token="", extra={"bridge_port": 3000})


def _tmpfile(suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def test_text_plus_mixed_media_routes_native_types():
    img = _tmpfile(".png")
    vid = _tmpfile(".mp4")
    voice = _tmpfile(".ogg")
    try:
        session_ctx, calls = _session_with(
            [
                _resp(200, {"messageId": "t1"}),
                _resp(200, {"messageId": "m1"}),
                _resp(200, {"messageId": "m2"}),
                _resp(200, {"messageId": "m3"}),
            ]
        )
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "12345",
                    "hello",
                    media_files=[(img, False), (vid, False), (voice, True)],
                )
            )
        assert res["success"] is True
        # health check, text first, then three media uploads in order
        assert calls[0][0].endswith("/health")
        assert calls[1][0].endswith("/send")
        assert calls[1][1]["message"] == "hello"
        media_types = [c[1]["mediaType"] for c in calls if c[0].endswith("/send-media")]
        assert media_types == ["image", "video", "audio"]
        # chat id normalized to a WhatsApp JID
        assert "@" in calls[1][1]["chatId"]
    finally:
        for p in (img, vid, voice):
            os.unlink(p)


def test_missing_captioned_file_falls_back_to_text():
    """If the single captioned file is missing, the caption is delivered as a
    plain /send message rather than being silently lost (W1)."""
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                "12345",
                "",
                media_files=[("/no/such/file.png", False)],
                caption="floor plan",
            )
        )
    # The send still surfaces the missing-file error...
    assert "error" in res
    assert "not found" in res["error"]
    # ...but the caption text was delivered on its own first.
    assert [url for url, _ in calls] == ["http://127.0.0.1:3000/health", "http://127.0.0.1:3000/send"]
    assert calls[1][1]["message"] == "floor plan"


def test_standalone_send_uses_persisted_secondary_bridge_port(tmp_path, monkeypatch):
    """A secondary's automatic port must route standalone sends to its bridge."""
    home = tmp_path / ".hermes"
    record = home / "platforms" / "whatsapp" / "bridge_port"
    record.parent.mkdir(parents=True)
    record.write_text("3042", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        result = asyncio.run(_standalone_send(SimpleNamespace(token="", extra={}), "12345", "hello"))
    assert result["success"] is True
    assert [url for url, _ in calls] == ["http://127.0.0.1:3042/health", "http://127.0.0.1:3042/send"]


@pytest.mark.parametrize(
    "persisted_port,extra,expected_port",
    [(3057, {}, 3057), (3057, {"bridge_port": 3061}, 3061), (None, {}, 3000)],
    ids=["persisted", "explicit-wins", "unallocated-default"],
)
def test_secondary_standalone_sends_use_active_profile_port_for_text_media_and_mentions(
    tmp_path, monkeypatch, persisted_port, extra, expected_port,
):
    """The active multiplex profile, not launch HOME, owns all standalone routes."""
    launch_home = tmp_path / "launch"
    secondary_home = tmp_path / "secondary"
    record = secondary_home / "platforms" / "whatsapp" / "bridge_port"
    secondary_home.mkdir()
    if persisted_port is not None:
        record.parent.mkdir(parents=True)
        record.write_text(str(persisted_port), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    media = tmp_path / "image.png"
    media.write_bytes(b"image")
    for home, config_extra, port in (
        (launch_home, {}, 3000),
        (secondary_home, extra, expected_port),
        (launch_home, {}, 3000),
    ):
        session_ctx, calls = _session_with(
            [_resp(200, {"messageId": "text"}), _resp(200, {"messageId": "media"})],
            health=_resp(200, {"capabilities": {"outboundMentions": True}}),
        )
        override = set_hermes_home_override(home)
        try:
            with patch("aiohttp.ClientSession", return_value=session_ctx):
                result = asyncio.run(_standalone_send(
                    SimpleNamespace(token="", extra=config_extra), "12345", "hello",
                    media_files=[(str(media), False)], mentions=["12345"],
                ))
        finally:
            reset_hermes_home_override(override)
        assert result.get("success") is True, result
        assert [url for url, _ in calls] == [
            f"http://127.0.0.1:{port}/health",
            f"http://127.0.0.1:{port}/send",
            f"http://127.0.0.1:{port}/send-media",
        ]
        assert calls[1][1]["mentions"] == ["12345@s.whatsapp.net"]
        assert calls[2][1]["filePath"] == str(media)
    assert not (launch_home / "platforms/whatsapp/bridge_port").exists()
    if persisted_port is None:
        assert not record.exists()
    else:
        assert record.read_text(encoding="utf-8") == str(persisted_port)


@pytest.mark.parametrize("reported", ["own", "other", None, "unhealthy"],
                         ids=["own-session", "other-profile-session", "pre-session-bridge", "health-503"])
def test_standalone_send_posts_only_through_this_profiles_bridge(tmp_path, reported):
    """Profiles sharing a bridge_port must not send from each other's WhatsApp account, text or media."""
    own = tmp_path / "b" / "session"
    media = tmp_path / "image.png"
    media.write_bytes(b"image")
    sessions = {"own": own, "other": tmp_path / "a" / "session"}
    health = _resp(503) if reported == "unhealthy" else _resp(200, {"session": str(sessions[reported])} if reported else {})
    session_ctx, calls = _session_with(
        [_resp(200, {"messageId": "text"}), _resp(200, {"messageId": "media"})], health=health)
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        result = asyncio.run(_standalone_send(
            SimpleNamespace(token="", extra={"bridge_port": 3000, "session_path": str(own)}), "12345", "hello",
            media_files=[(str(media), False)],
        ))
    posted = [url for url, _ in calls if not url.endswith("/health")]
    if reported in ("other", "unhealthy"):
        assert "nothing was sent" in result["error"]
        assert posted == []
    else:
        assert result.get("success") is True, result
        assert posted == ["http://127.0.0.1:3000/send", "http://127.0.0.1:3000/send-media"]
