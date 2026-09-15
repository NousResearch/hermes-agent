"""voice.grok.* JSON-RPC handlers: the renderer-facing surface of the Grok-Live engine.

No network, no real bridge: ``tools.voice_live_grok_bridge.GrokLiveBridge`` is patched with a
recording fake and ``tools.wake_word`` is replaced in ``sys.modules`` (the mic lease must not
touch real audio). Every handler result is validated against its declared contract Result
model — under extra="forbid" a drifted key is exactly the bug the contract system exists to
catch.
"""

import base64
import sys
import types

import pytest

import tools.voice_live_grok as grok_config
import tools.voice_live_grok_bridge as bridge_module
from tui_gateway import server
from tui_gateway.contracts import EVENTS, METHODS
from tui_gateway.contracts.prompt_voice_grok import (
    VoiceGrokAudioResult, VoiceGrokMuteResult, VoiceGrokStartResult, VoiceGrokStatusResult,
    VoiceGrokStopResult)

METHOD_NAMES = ("voice.grok.status", "voice.grok.start", "voice.grok.stop",
                "voice.grok.audio", "voice.grok.mute")
EVENT_NAMES = ("voice.grok.audio", "voice.grok.transcript", "voice.grok.state",
               "voice.grok.delegation")


class FakeBridge:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.muted = None
        self.mic = []
        FakeBridge.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=10.0):
        self.stopped = True
        return True

    @property
    def alive(self):
        return self.started and not self.stopped

    def send_mic(self, pcm):
        self.mic.append(pcm)
        return {"accepted": True}

    def set_muted(self, muted):
        self.muted = bool(muted)
        return self.muted


@pytest.fixture
def grok_rpc(monkeypatch):
    """Fake bridge class + fake wake-word module; clean per-test bridge registry."""
    FakeBridge.instances = []
    monkeypatch.setattr(bridge_module, "GrokLiveBridge", FakeBridge)
    wake_calls = {"paused": [], "resumed": []}
    fake_wake = types.ModuleType("tools.wake_word")
    fake_wake.pause_listening = lambda owner=None: wake_calls["paused"].append(owner) or True
    fake_wake.resume_listening = lambda owner=None: wake_calls["resumed"].append(owner) or True
    monkeypatch.setitem(sys.modules, "tools.wake_word", fake_wake)
    with server._grok_bridges_lock:
        server._grok_bridges.clear()
        server._grok_wake_owners.clear()
    yield wake_calls
    with server._grok_bridges_lock:
        server._grok_bridges.clear()
        server._grok_wake_owners.clear()


def call(name, params):
    return server._methods[name](1, params)


# ── wire surface ─────────────────────────────────────────────────────────────


def test_every_voice_grok_method_and_event_is_declared_and_registered():
    for name in METHOD_NAMES:
        assert name in METHODS, f"{name} missing from the contract catalog"
        assert name in server._methods, f"{name} has no registered handler"
    for name in EVENT_NAMES:
        assert name in EVENTS, f"{name} missing from the event catalog"


# ── voice.grok.status ──────────────────────────────────────────────────────────


def test_status_mirrors_the_resolver_without_the_credential(grok_rpc, monkeypatch):
    verdict = {"mode": "grok-live", "available": True, "reason": None,
               "model": "grok-voice-latest", "voice": "eve"}
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: verdict)
    answer = call("voice.grok.status", {})
    VoiceGrokStatusResult.model_validate(answer["result"])
    assert answer["result"] == verdict
    # The status contract is the non-secret readiness verdict (SPEC §8.1/§11).
    assert set(answer["result"]) == {"mode", "available", "reason", "model", "voice"}


def test_status_surfaces_resolution_failure_as_an_error(grok_rpc, monkeypatch):
    def boom():
        raise RuntimeError("config exploded")
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", boom)
    assert call("voice.grok.status", {})["error"]["code"] == 5026


# ── voice.grok.start ─────────────────────────────────────────────────────────────


def test_start_without_a_credential_fails_fast(grok_rpc, monkeypatch):
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": False,
        "reason": "no xAI credential (SuperGrok login via `hermes login` or set XAI_API_KEY)",
        "model": "m", "voice": "v"})
    answer = call("voice.grok.start", {"session_id": "s1"})
    VoiceGrokStartResult.model_validate(answer["result"])
    assert answer["result"]["started"] is False
    assert "no xAI credential" in answer["result"]["reason"]
    assert FakeBridge.instances == []  # no bridge, no wake pause
    assert grok_rpc["paused"] == []


def test_start_creates_the_bridge_pauses_wake_and_answers_connecting(grok_rpc, monkeypatch):
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": True, "reason": None, "model": "m", "voice": "v"})
    answer = call("voice.grok.start", {"session_id": "s1"})
    VoiceGrokStartResult.model_validate(answer["result"])
    assert answer["result"] == {"started": True, "state": "connecting"}
    (bridge,) = FakeBridge.instances
    assert bridge.started and bridge.kwargs["session_id"] == "s1"
    assert server._grok_bridges["s1"] is bridge
    assert len(grok_rpc["paused"]) == 1  # mic lease taken (SPEC §6)


def test_start_twice_reuses_the_live_bridge(grok_rpc, monkeypatch):
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": True, "reason": None, "model": "m", "voice": "v"})
    call("voice.grok.start", {"session_id": "s1"})
    answer = call("voice.grok.start", {"session_id": "s1"})
    assert answer["result"]["reused"] is True
    assert len(FakeBridge.instances) == 1


def test_start_requires_a_session_id(grok_rpc):
    assert call("voice.grok.start", {})["error"]["code"] == 4001


# ── voice.grok.audio ─────────────────────────────────────────────────────────────


def _started_bridge(grok_rpc, monkeypatch):
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": True, "reason": None, "model": "m", "voice": "v"})
    call("voice.grok.start", {"session_id": "s1"})
    return FakeBridge.instances[0]


def test_audio_relays_decoded_pcm_to_the_bridge(grok_rpc, monkeypatch):
    bridge = _started_bridge(grok_rpc, monkeypatch)
    pcm = b"\x01\x00" * 2400
    answer = call("voice.grok.audio", {"session_id": "s1",
                                       "pcm_b64": base64.b64encode(pcm).decode(), "seq": 7})
    VoiceGrokAudioResult.model_validate(answer["result"])
    assert answer["result"] == {"accepted": True}
    assert bridge.mic == [pcm]


def test_audio_for_an_unknown_session_is_not_running_not_an_error(grok_rpc):
    answer = call("voice.grok.audio", {"session_id": "ghost",
                                       "pcm_b64": base64.b64encode(b"\x00" * 100).decode()})
    assert answer["result"] == {"accepted": False, "reason": "not_running"}


def test_audio_rejects_malformed_frames(grok_rpc, monkeypatch):
    _started_bridge(grok_rpc, monkeypatch)
    assert call("voice.grok.audio", {"session_id": "s1"})["error"]["code"] == 4001
    assert call("voice.grok.audio", {"session_id": "s1", "pcm_b64": "a"})["error"]["code"] == 4001
    oversized = base64.b64encode(b"\x00" * 96001).decode()
    assert call("voice.grok.audio", {"session_id": "s1", "pcm_b64": oversized})["error"]["code"] == 4001


# ── voice.grok.mute / voice.grok.stop ────────────────────────────────────────────


def test_mute_toggles_the_bridge(grok_rpc, monkeypatch):
    bridge = _started_bridge(grok_rpc, monkeypatch)
    answer = call("voice.grok.mute", {"session_id": "s1", "muted": True})
    VoiceGrokMuteResult.model_validate(answer["result"])
    assert answer["result"] == {"muted": True} and bridge.muted is True
    assert call("voice.grok.mute", {"session_id": "ghost", "muted": True})["error"]["code"] == 4021


def test_stop_tears_down_the_bridge_and_resumes_wake(grok_rpc, monkeypatch):
    bridge = _started_bridge(grok_rpc, monkeypatch)
    answer = call("voice.grok.stop", {"session_id": "s1"})
    VoiceGrokStopResult.model_validate(answer["result"])
    assert answer["result"] == {"stopped": True}
    assert bridge.stopped and "s1" not in server._grok_bridges
    assert grok_rpc["resumed"] == grok_rpc["paused"]  # lease handed back (SPEC §6)


def test_stop_without_a_session_is_a_clean_not_running(grok_rpc):
    answer = call("voice.grok.stop", {"session_id": "ghost"})
    assert answer["result"] == {"stopped": False, "reason": "not_running"}
