"""STT provider RESOLUTION never installs; transcription still does.

Regression: ``_detect_local_backend`` lazy-installed faster-whisper, and it sits in the provider
resolution chain every status surface uses — ``wake.status`` (fired on every desktop
gateway-open), ``/voice status`` and ``GET /api/audio/voice-config``. Each of those could start a
full dependency rebuild, holding the per-install lock long enough to keep a sibling profile's
backend off its port.
"""

import pytest

from tools import transcription_tools


@pytest.fixture
def no_faster_whisper(monkeypatch):
    """A machine without faster-whisper or a whisper CLI, recording install attempts."""
    attempts: list[int] = []
    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", False)
    monkeypatch.setattr(transcription_tools, "_has_local_command", lambda: False)
    monkeypatch.setattr(transcription_tools, "_try_lazy_install_stt",
                        lambda: attempts.append(1) or False)
    return transcription_tools, attempts


def test_resolution_reports_unavailable_without_installing(no_faster_whisper):
    tools, attempts = no_faster_whisper
    assert tools._detect_local_backend() is None
    assert tools._resolve_explicit_local() == "none"
    assert attempts == []


def test_transcription_is_what_installs(no_faster_whisper):
    tools, attempts = no_faster_whisper
    result = tools._transcribe_local("/tmp/absent.wav", "base")
    assert attempts == [1], "the action path lost its on-demand install"
    assert result["success"] is False
    assert "faster-whisper not installed" in result["error"]


def test_readiness_reprobes_instead_of_answering_from_an_import_snapshot(monkeypatch):
    """#81235: a package installed while the gateway runs must count without a restart.

    ``_HAS_*`` was an import-time bool, so the resolution chain (and the wake-word arming
    prerequisite that reads it) kept reporting the provider missing until a relaunch.
    """
    tools = transcription_tools
    monkeypatch.setattr(tools, "_has_local_command", lambda: False)
    monkeypatch.delitem(tools.__dict__, "_HAS_FASTER_WHISPER", raising=False)
    monkeypatch.setattr(tools, "_safe_find_spec", lambda name: name == "faster_whisper")
    assert tools._HAS_FASTER_WHISPER is True, "outside readers must re-probe"
    assert tools._detect_local_backend() == "local", "in-module readers must re-probe too"

    monkeypatch.setattr(tools, "_safe_find_spec", lambda name: False)
    assert tools._HAS_FASTER_WHISPER is False
    assert tools._detect_local_backend() is None
