"""Regression tests for #81235: a TTS/STT package installed into the venv
after gateway startup must be picked up by the next provider-status check
without a gateway restart.

Before the fix, ``tools/transcription_tools`` cached its ``_HAS_*``
optional-package flags at import time.  A long-lived gateway process kept
reporting a newly-installed provider as unavailable ("Piper provider
selected but 'piper-tts' package not installed...") until a manual
``launchctl kickstart`` restarted it — and the same stale cache silently
blocked the wake-word listener from arming, because STT+TTS readiness is
its arming prerequisite.
"""

from unittest.mock import patch

import tools.transcription_tools as tt
import tools.voice_mode as vm


class TestRecheckPackageAvailability:
    def test_refreshes_flags_from_environment(self, monkeypatch):
        # Simulate a gateway that started before faster-whisper was
        # installed: the import-time flag is False.
        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
        monkeypatch.setattr(tt, "_HAS_OPENAI", False)
        monkeypatch.setattr(tt, "_HAS_MISTRAL", False)
        monkeypatch.setattr(tt, "_HAS_PILK", False)

        # After the operator installs the package, find_spec starts
        # succeeding.
        with patch.object(tt, "_safe_find_spec", return_value=True):
            tt._recheck_package_availability()

        assert tt._HAS_FASTER_WHISPER is True
        assert tt._HAS_OPENAI is True

    def test_probe_failure_keeps_previous_values(self, monkeypatch):
        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", True)
        with patch.object(tt, "_safe_find_spec", side_effect=ValueError("boom")):
            tt._recheck_package_availability()
        # The cached value survives a probe failure.
        assert tt._HAS_FASTER_WHISPER is True

    def test_picks_up_newly_installed_package_in_get_provider_path(self, monkeypatch):
        """The issue's scenario: explicit local provider was 'none' at
        startup; after the package lands, the refresh updates the cached
        flags so _get_provider resolves it without a process restart."""
        # Startup state: faster-whisper not installed.
        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
        monkeypatch.setattr(tt, "_HAS_OPENAI", False)
        monkeypatch.setattr(tt, "_HAS_MISTRAL", False)
        monkeypatch.setattr(tt, "_HAS_PILK", False)

        # Before the refresh, the stale cache says the provider is absent.
        with patch.object(tt, "_try_lazy_install_stt", return_value=False), patch.object(
            tt, "_has_local_command", return_value=False
        ):
            assert tt._get_provider({"enabled": True, "provider": "local"}) == "none"

        # Operator installs faster-whisper mid-process.
        def _find_spec(name):
            return name == "faster_whisper"

        with patch.object(tt, "_safe_find_spec", side_effect=_find_spec):
            tt._recheck_package_availability()

        with patch.object(tt, "_try_lazy_install_stt", return_value=False), patch.object(
            tt, "_has_local_command", return_value=False
        ):
            provider = tt._get_provider({"enabled": True, "provider": "local"})

        assert provider == "local"

    def test_voice_requirements_refreshes_before_provider_resolution(self, monkeypatch):
        """check_voice_requirements (the UI/wake-word path from the issue)
        re-probes packages before resolving the STT provider."""
        calls = []

        def _tracked_recheck():
            calls.append(1)

        monkeypatch.setattr(tt, "_recheck_package_availability", _tracked_recheck)
        # Force the default path: audio present, STT enabled with a native
        # provider so no plugin dispatch is involved.
        monkeypatch.setattr(vm, "_audio_available", lambda: True)
        monkeypatch.setattr(vm, "detect_audio_environment", lambda: {"available": True, "warnings": [], "notices": []})
        monkeypatch.setattr(tt, "is_stt_enabled", lambda cfg: True)
        monkeypatch.setattr(tt, "_get_provider", lambda cfg: "groq")

        result = vm.check_voice_requirements()

        assert calls, "check_voice_requirements must re-probe packages"
        assert result["stt_available"] is True
