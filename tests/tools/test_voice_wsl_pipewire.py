"""Regression: WSL voice detection must honor PIPEWIRE_REMOTE, not only PULSE_SERVER.

detect_audio_environment() honors forwarded audio (has_forwarded_audio =
PULSE_SERVER or PIPEWIRE_REMOTE or a reachable socket) in the SSH and container
blocks, but the WSL block previously checked only PULSE_SERVER — so a WSL user
with PipeWire forwarding (PIPEWIRE_REMOTE) was wrongly blocked from voice mode.
These tests patch the voice module's WSL predicate while retaining native Linux gating.
"""
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.platforms("linux")


def _force_wsl(monkeypatch):
    import tools.voice_mode as voice_mode

    monkeypatch.setattr(voice_mode, "is_wsl", lambda: True)


def _base(monkeypatch):
    for v in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION", "PULSE_SERVER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("PIPEWIRE_REMOTE", raising=False)
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)
    monkeypatch.setattr("tools.voice_mode._pulse_socket_reachable", lambda: False)
    sd = MagicMock(); sd.query_devices.return_value = [{"name": "dev"}]
    monkeypatch.setattr("tools.voice_mode._import_audio", lambda: (sd, MagicMock()))


def test_wsl_with_pipewire_remote_allows_voice(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setenv("PIPEWIRE_REMOTE", "/run/user/1000/pipewire-0")
    _force_wsl(monkeypatch)
    from tools.voice_mode import detect_audio_environment
    result = detect_audio_environment()
    assert result["available"] is True, result["warnings"]


def test_wsl_with_pulse_server_still_allows_voice(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setenv("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")
    _force_wsl(monkeypatch)
    from tools.voice_mode import detect_audio_environment
    assert detect_audio_environment()["available"] is True


def test_wsl_without_forwarding_still_blocks(monkeypatch):
    _base(monkeypatch)
    _force_wsl(monkeypatch)  # no PULSE_SERVER, no PIPEWIRE_REMOTE
    from tools.voice_mode import detect_audio_environment
    res = detect_audio_environment()
    assert res["available"] is False
    assert any("WSL" in w for w in res["warnings"])
