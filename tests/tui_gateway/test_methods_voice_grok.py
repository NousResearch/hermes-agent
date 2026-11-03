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
        self.session_id = kwargs["session_id"]
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

    def rekey(self, session_id):
        self.session_id = session_id

    def speak_reply(self, text):
        self.spoken = text
        return True


@pytest.fixture
def grok_rpc(monkeypatch):
    """Fake bridge class + fake wake-word module; clean per-test bridge registry."""
    FakeBridge.instances = []
    monkeypatch.setattr(bridge_module, "GrokLiveBridge", FakeBridge)
    monkeypatch.setattr(grok_config, "_resolve_grok_credentials", lambda live=None: "test-token")
    wake_calls = {"paused": [], "resumed": []}
    fake_wake = types.ModuleType("tools.wake_word")
    fake_wake.pause_listening = lambda owner=None: wake_calls["paused"].append(owner) or True
    fake_wake.resume_listening = lambda owner=None: wake_calls["resumed"].append(owner) or True
    monkeypatch.setitem(sys.modules, "tools.wake_word", fake_wake)
    with server._grok_bridges_lock:
        server._grok_bridges.clear()
        server._grok_wake_owners.clear()
        server._grok_client_transports.clear()
    with server._grok_alias_lock:
        server._grok_alias.clear()
    yield wake_calls
    with server._grok_bridges_lock:
        server._grok_bridges.clear()
        server._grok_wake_owners.clear()
        server._grok_client_transports.clear()
    with server._grok_alias_lock:
        server._grok_alias.clear()


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


def test_disconnect_stops_only_that_clients_bridges_and_releases_its_lease(grok_rpc, monkeypatch):
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": True, "reason": None, "model": "m", "voice": "v"})
    owner, other = object(), object()
    current = [owner]
    monkeypatch.setattr(server, "_caller_transport", lambda: current[0])
    call("voice.grok.start", {"session_id": "s1"})
    current[0] = other
    call("voice.grok.start", {"session_id": "s2"})

    server.unregister_live_transport(owner)

    assert FakeBridge.instances[0].stopped is True
    assert FakeBridge.instances[1].stopped is False
    assert "s1" not in server._grok_bridges
    assert "s2" in server._grok_bridges
    assert grok_rpc["resumed"] == [owner]


def test_stop_without_a_session_is_a_clean_not_running(grok_rpc):
    answer = call("voice.grok.stop", {"session_id": "ghost"})
    assert answer["result"] == {"stopped": False, "reason": "not_running"}


def test_every_handler_is_profile_scoped():
    """Every RPC reads config/credentials and must use the requested profile's runtime scope."""
    from tui_gateway import methods_voice_grok

    pending = dict(methods_voice_grok._registry._pending)
    for name in (*METHOD_NAMES, "voice.grok.speak", "voice.grok.rekey"):
        assert getattr(pending[name], "_hermes_profile_scoped", False), name


def test_dead_bridge_reap_releases_the_wake_lease(grok_rpc, monkeypatch):
    bridge = _started_bridge(grok_rpc, monkeypatch)
    bridge.stopped = True

    answer = call("voice.grok.audio", {
        "session_id": "s1", "pcm_b64": base64.b64encode(b"\x00\x00").decode()
    })

    assert answer["result"] == {"accepted": False, "reason": "not_running"}
    assert grok_rpc["resumed"] == grok_rpc["paused"]
    assert "s1" not in server._grok_client_transports


def test_rekey_updates_bridge_identity_and_emits_under_the_real_session(grok_rpc, monkeypatch):
    frames = []
    transport = types.SimpleNamespace(write=frames.append)
    monkeypatch.setattr(server, "_caller_transport", lambda: transport)
    monkeypatch.setitem(server._sessions, "real-1", {"transport": transport})
    monkeypatch.setattr("tui_gateway.event_replay._stamp_event", lambda frame: None)
    bridge = _started_bridge(grok_rpc, monkeypatch)

    answer = call("voice.grok.rekey", {"from_session_id": "s1", "to_session_id": "real-1"})

    assert answer["result"] == {"rekeyed": True}
    assert bridge.session_id == "real-1"
    bridge.kwargs["emit"]("voice.grok.state", {
        "session_id": bridge.session_id, "state": "listening"
    })
    assert frames[-1]["params"]["session_id"] == "real-1"
    assert frames[-1]["params"]["session_id"] == frames[-1]["params"]["payload"]["session_id"]


def test_other_client_cannot_control_inject_or_steal_a_voice_bridge(grok_rpc, monkeypatch):
    owner, intruder = object(), object()
    current = [owner]
    monkeypatch.setattr(server, "_caller_transport", lambda: current[0])
    bridge = _started_bridge(grok_rpc, monkeypatch)
    monkeypatch.setitem(server._sessions, "real-1", {"transport": owner})
    pcm = base64.b64encode(b"\x00\x00").decode()

    current[0] = intruder
    attempts = [
        ("voice.grok.start", {"session_id": "s1"}),
        ("voice.grok.start", {"session_id": "real-1"}),
        ("voice.grok.audio", {"session_id": "s1", "pcm_b64": pcm}),
        ("voice.grok.mute", {"session_id": "s1", "muted": True}),
        ("voice.grok.speak", {"session_id": "s1", "text": "attacker controlled"}),
        ("voice.grok.rekey", {"from_session_id": "s1", "to_session_id": "real-1"}),
        ("voice.grok.stop", {"session_id": "s1"}),
    ]
    for name, params in attempts:
        assert call(name, params)["error"]["code"] == 4021, name
    assert bridge.mic == [] and bridge.muted is None and not bridge.stopped
    assert server._grok_client_transports["s1"] is owner

    current[0] = owner
    assert call("voice.grok.audio", {"session_id": "s1", "pcm_b64": pcm})["result"]["accepted"]


def test_start_failure_rolls_back_the_wake_lease(grok_rpc, monkeypatch):
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": True, "reason": None, "model": "m", "voice": "v"})

    def fail_start(self):
        raise RuntimeError("thread failed")

    monkeypatch.setattr(FakeBridge, "start", fail_start)
    answer = call("voice.grok.start", {"session_id": "s1"})
    assert answer["error"]["code"] == 5026
    assert grok_rpc["resumed"] == grok_rpc["paused"]
    assert "s1" not in server._grok_bridges
    assert "s1" not in server._grok_client_transports
