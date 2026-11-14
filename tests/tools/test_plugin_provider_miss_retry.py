"""One forced plugin re-discovery per unregistered provider NAME, not per call (#150621).

The TTS / STT / image_gen / video_gen dispatchers all did ``_ensure_plugins_discovered(force=True)``
on a registry miss. That pass unloads every plugin of the profile — teardown runs, ``ctx.spawn_task``
work is cancelled — and the name is still unregistered afterwards, so a ``<category>.provider`` typo
paid a full plugin reload on every call (in the gateway: every inbound voice note, every spoken
reply). :mod:`hermes_cli.plugin_provider_retry` remembers the miss together with a stamp of the
plugin sources and skips the pass while that stamp is unchanged.

These tests count real ``force=True`` discovery calls through each dispatcher, so a regression that
re-introduces a per-call pass fails on the count rather than on an internal detail.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.plugins as plugins_module
from hermes_cli import plugin_provider_retry as retry


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Each test gets its own profile home, so the memo (keyed by home) cannot leak between tests."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(retry, "get_hermes_home", lambda: tmp_path)
    retry.forget_misses()
    yield
    retry.forget_misses()


@pytest.fixture
def forced(monkeypatch):
    """Record ``_ensure_plugins_discovered``'s ``force`` flag for every call a dispatcher makes."""
    calls: list[bool] = []

    def _ensure(force: bool = False):
        calls.append(bool(force))

    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", _ensure)
    return calls


def test_tts_lookup_forces_one_rediscovery_for_a_name_that_never_registers(forced):
    from tools.tts_tool_plugins import _lookup_plugin_provider

    for _ in range(3):
        assert _lookup_plugin_provider("ghost-tts", retry=True) is None

    assert forced.count(True) == 1


def test_a_new_plugin_directory_rearms_the_forced_retry(forced, tmp_path):
    from tools.tts_tool_plugins import _lookup_plugin_provider

    assert _lookup_plugin_provider("ghost-tts", retry=True) is None
    assert forced.count(True) == 1

    # A plugin installed mid-session is exactly what the forced pass exists for: the plugin roots
    # gain an entry, the stamp changes, and the next miss retries once more.
    (tmp_path / "plugins" / "late-tts-backend").mkdir(parents=True)
    assert _lookup_plugin_provider("ghost-tts", retry=True) is None

    assert forced.count(True) == 2


def test_a_registered_provider_never_forces_a_rediscovery(forced):
    from agent import tts_registry

    class _TTS(tts_registry.TTSProvider):
        @property
        def name(self) -> str:
            return "ghost-registered"

        def synthesize(self, text, output_path, **kwargs):
            return output_path

    tts_registry.register_provider(_TTS())  # type: ignore[attr-defined]
    try:
        from tools.tts_tool_plugins import _lookup_plugin_provider

        assert _lookup_plugin_provider("ghost-registered", retry=True) is not None
        assert forced.count(True) == 0
    finally:
        tts_registry._reset_for_tests()  # type: ignore[attr-defined]


def test_stt_dispatch_and_requirements_check_share_one_forced_pass(forced):
    from tools.transcription_command import _dispatch_to_plugin_provider
    from tools.voice_mode import _check_plugin_stt_provider

    assert _dispatch_to_plugin_provider("/tmp/voice.wav", "ghost-stt") is None
    assert _check_plugin_stt_provider("ghost-stt") is False

    assert forced.count(True) == 1


def test_image_gen_dispatch_forces_one_rediscovery(forced, monkeypatch):
    from tools import image_generation_tool

    monkeypatch.setattr(image_generation_tool, "_plugin_provider_name", lambda: "ghost-image")
    for _ in range(3):
        raw = image_generation_tool._dispatch_to_plugin_provider("a cat", "square")
        assert raw is not None
        assert json.loads(raw)["error_type"] == "provider_not_registered"

    assert forced.count(True) == 1


def test_video_gen_resolution_forces_one_rediscovery(forced, monkeypatch):
    from tools import video_generation_tool

    monkeypatch.setattr("agent.video_gen_registry.configured_provider_name", lambda *a, **k: "ghost-video")
    monkeypatch.setattr("agent.provider_registry.configured_provider_name", lambda *a, **k: "ghost-video")
    for _ in range(3):
        assert video_generation_tool._resolve_active_provider() is None

    assert forced.count(True) == 1


def test_the_miss_memo_is_per_provider_kind():
    retry.remember_miss("stt", "shared-name")

    assert retry.forced_refresh_needed("tts", "shared-name") is True
    assert retry.forced_refresh_needed("stt", "shared-name") is False
