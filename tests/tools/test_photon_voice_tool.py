"""photon_voice tool — service-gated voice-note send over Photon Spectrum.

Covers the schema/availability boundary (check_fn), chat resolution, and the
live-adapter send path. Mirrors tests/tools/test_send_message_react.py: the
message-id resolution + adapter dispatch are exercised against a fake gateway
runner, and the handler is invoked through its registered entry (JSON-string
result, not a raw dict/SendResult) so dispatch normalization is covered too.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.photon_voice_tool as m
from tools.registry import registry


# ---------------------------------------------------------------------------
# Fake gateway plumbing
# ---------------------------------------------------------------------------


class _FakeSendResult:
    """Stand-in for gateway.platforms.base.SendResult."""

    def __init__(self, success=True, message_id=None, error=None, retryable=False):
        self.success = success
        self.message_id = message_id
        self.error = error
        self.retryable = retryable


class _FakePhotonAdapter:
    """Records send_voice calls; returns a configurable SendResult."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result or _FakeSendResult(success=True, message_id="spc-msg-abc")

    async def send_voice(self, chat_id, audio_path, caption=None):
        self.calls.append((chat_id, audio_path, caption))
        return self._result


def _runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform("photon"): adapter})


def _home_channel(chat_id):
    return SimpleNamespace(chat_id=chat_id)


# ---------------------------------------------------------------------------
# check_fn — service gating
# ---------------------------------------------------------------------------


def test_check_fn_false_without_live_adapter():
    with patch("gateway.run._gateway_runner_ref", return_value=None):
        assert m._photon_voice_check() is False


def test_check_fn_true_with_live_adapter():
    with patch(
        "gateway.run._gateway_runner_ref", return_value=_runner_with(_FakePhotonAdapter())
    ):
        assert m._photon_voice_check() is True


# ---------------------------------------------------------------------------
# Argument validation (no adapter touched)
# ---------------------------------------------------------------------------


def test_missing_audio_path_errors():
    out = m._handle_photon_voice({})
    res = json.loads(out)
    assert res.get("success") is not True
    assert "error" in res
    assert "audio_path" in res["error"]


def test_nonexistent_audio_path_errors(tmp_path):
    bad = str(tmp_path / "nope.mp3")
    out = m._handle_photon_voice({"audio_path": bad})
    res = json.loads(out)
    assert res.get("success") is not True
    assert "does not exist" in res["error"]


# ---------------------------------------------------------------------------
# Chat resolution + dispatch
# ---------------------------------------------------------------------------


def test_explicit_chat_id_used(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    adapter = _FakePhotonAdapter()
    with patch(
        "gateway.run._gateway_runner_ref", return_value=_runner_with(adapter)
    ):
        out = m._handle_photon_voice(
            {"audio_path": str(audio), "chat_id": "any;-;+15551234567"}
        )
    res = json.loads(out)
    assert res["success"] is True
    assert res["message_id"] == "spc-msg-abc"
    assert adapter.calls[0][0] == "any;-;+15551234567"
    assert adapter.calls[0][1] == str(audio)


def test_defaults_to_home_channel(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    adapter = _FakePhotonAdapter()
    with patch(
        "gateway.run._gateway_runner_ref", return_value=_runner_with(adapter)
    ), patch(
        "gateway.config.load_gateway_config",
        return_value=SimpleNamespace(
            get_home_channel=lambda _p: _home_channel("home;-;+15559998888")
        ),
    ):
        out = m._handle_photon_voice({"audio_path": str(audio)})
    res = json.loads(out)
    assert res["success"] is True
    assert adapter.calls[0][0] == "home;-;+15559998888"


def test_caption_passed_through(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    adapter = _FakePhotonAdapter()
    with patch(
        "gateway.run._gateway_runner_ref", return_value=_runner_with(adapter)
    ):
        m._handle_photon_voice(
            {
                "audio_path": str(audio),
                "chat_id": "any;-;+15551234567",
                "caption": "  spoken follow-up  ",
            }
        )
    assert adapter.calls[0][2] == "spoken follow-up"


def test_send_failure_surfaced(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    adapter = _FakePhotonAdapter(
        _FakeSendResult(success=False, error="sidecar 503", retryable=True)
    )
    with patch(
        "gateway.run._gateway_runner_ref", return_value=_runner_with(adapter)
    ):
        out = m._handle_photon_voice(
            {"audio_path": str(audio), "chat_id": "any;-;+15551234567"}
        )
    res = json.loads(out)
    assert res["success"] is False
    assert res["error"] == "sidecar 503"
    assert res["retryable"] is True


def test_no_live_adapter_errors(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    with patch("gateway.run._gateway_runner_ref", return_value=None):
        out = m._handle_photon_voice(
            {"audio_path": str(audio), "chat_id": "any;-;+15551234567"}
        )
    res = json.loads(out)
    assert res.get("success") is not True
    assert "live Photon adapter" in res["error"]


# ---------------------------------------------------------------------------
# Registered entry round-trips through dispatch (JSON string, not dict)
# ---------------------------------------------------------------------------


def test_registered_entry_dispatch(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    adapter = _FakePhotonAdapter()
    with patch(
        "gateway.run._gateway_runner_ref", return_value=_runner_with(adapter)
    ):
        entry = registry.get_entry("photon_voice")
        assert entry is not None
        assert entry.is_async is False
        # dispatch normalizes the result; handler returns a JSON string already,
        # so dispatch must pass it through unchanged.
        result = registry.dispatch("photon_voice", {"audio_path": str(audio), "chat_id": "x"})
    parsed = json.loads(result)
    assert parsed["success"] is True
