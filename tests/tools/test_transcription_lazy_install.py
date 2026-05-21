"""Regression tests for the STT lazy-install path (#29782).

These tests cover the behaviour added in the fix:

  * ``_ensure_faster_whisper`` calls ``tools.lazy_deps.ensure`` when
    faster-whisper is missing and flips ``_HAS_FASTER_WHISPER`` to True
    on success.
  * ``_ensure_faster_whisper`` is a no-op when faster-whisper is already
    present (short-circuit).
  * ``_ensure_faster_whisper`` skips the install attempt under
    ``PYTEST_CURRENT_TEST`` so existing tests that patch
    ``_HAS_FASTER_WHISPER`` to False don't trigger real installs.
  * ``_get_provider`` invokes the helper for explicit ``provider="local"``
    and for the auto-detect "everything else is missing" final fallback,
    but NOT when a cloud provider would have satisfied the request.

Before the fix, the gateway emitted a "no STT provider is configured"
message telling Docker users to run ``pip install faster-whisper`` — but
the published Docker image doesn't ship ``pip`` inside the venv, so the
advice was unactionable. The lazy_deps machinery for ``stt.faster_whisper``
existed but was never wired up to the transcription pipeline.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean STT-related env vars so tests are deterministic."""
    for var in (
        "VOICE_TOOLS_OPENAI_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "MISTRAL_API_KEY",
        "HERMES_LOCAL_STT_COMMAND",
        "HERMES_LOCAL_STT_LANGUAGE",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# _ensure_faster_whisper
# ---------------------------------------------------------------------------


class TestEnsureFasterWhisper:
    """Behavioural tests for the lazy-install helper itself."""

    def test_short_circuits_when_already_installed(self, monkeypatch):
        """When the module flag is True, ensure() must NOT be called."""
        # Drop PYTEST_CURRENT_TEST so the pytest fast-path doesn't fire.
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        from tools import transcription_tools as tt

        sentinel = MagicMock()
        with patch.object(tt, "_HAS_FASTER_WHISPER", True), \
             patch("tools.lazy_deps.ensure", sentinel):
            assert tt._ensure_faster_whisper() is True
            sentinel.assert_not_called()

    def test_skips_install_under_pytest(self):
        """Pytest fast-path: PYTEST_CURRENT_TEST set + flag False = no install attempt."""
        # autouse fixture clean_env doesn't touch PYTEST_CURRENT_TEST; pytest itself sets it.
        assert os.environ.get("PYTEST_CURRENT_TEST"), \
            "PYTEST_CURRENT_TEST must be set by pytest for this guard test to be meaningful"
        from tools import transcription_tools as tt

        sentinel = MagicMock()
        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch("tools.lazy_deps.ensure", sentinel):
            assert tt._ensure_faster_whisper() is False
            sentinel.assert_not_called()

    def test_calls_lazy_ensure_when_missing(self, monkeypatch):
        """In a non-pytest environment with the flag False, lazy_deps.ensure must be invoked."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        from tools import transcription_tools as tt

        ensure_mock = MagicMock(return_value=None)
        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch("tools.lazy_deps.ensure", ensure_mock), \
             patch.object(tt, "_safe_find_spec", return_value=True):
            result = tt._ensure_faster_whisper(prompt=False)

        ensure_mock.assert_called_once_with("stt.faster_whisper", prompt=False)
        assert result is True

    def test_flips_module_flag_after_successful_install(self, monkeypatch):
        """A successful install must update ``_HAS_FASTER_WHISPER`` so later checks pass."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        from tools import transcription_tools as tt

        original = tt._HAS_FASTER_WHISPER
        try:
            tt._HAS_FASTER_WHISPER = False
            with patch("tools.lazy_deps.ensure", return_value=None), \
                 patch.object(tt, "_safe_find_spec", return_value=True):
                tt._ensure_faster_whisper()
            assert tt._HAS_FASTER_WHISPER is True
        finally:
            tt._HAS_FASTER_WHISPER = original

    def test_swallows_install_failure(self, monkeypatch, caplog):
        """An install failure logs a warning and leaves the flag False — no crash."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        from tools import transcription_tools as tt

        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch("tools.lazy_deps.ensure", side_effect=RuntimeError("pip failed")):
            result = tt._ensure_faster_whisper()

        assert result is False
        # The warning carries the manual-install hint with the correct command.
        assert any("uv pip install faster-whisper" in r.getMessage()
                   for r in caplog.records), \
            "Warning must mention `uv pip install` (not bare `pip`) so Docker users get actionable advice"

    def test_no_crash_when_lazy_deps_unavailable(self, monkeypatch):
        """If ``tools.lazy_deps`` itself can't be imported, return cleanly."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        from tools import transcription_tools as tt

        # Simulate lazy_deps import failure by removing it from sys.modules
        # and inserting a path entry that will raise on import. Simpler:
        # patch the import inside the helper. The helper does
        # `from tools.lazy_deps import ensure as _lazy_ensure` — wrap it.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tools.lazy_deps":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(builtins, "__import__", side_effect=fake_import):
            assert tt._ensure_faster_whisper() is False


# ---------------------------------------------------------------------------
# _get_provider integration with the lazy-install helper
# ---------------------------------------------------------------------------


class TestGetProviderTriggersLazyInstall:
    """Verify _get_provider invokes _ensure_faster_whisper at the right moments."""

    def test_explicit_local_invokes_lazy_install(self):
        """provider=local must attempt the lazy install before degrading to none."""
        from tools import transcription_tools as tt

        helper = MagicMock(return_value=False)
        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_ensure_faster_whisper", helper), \
             patch.object(tt, "_has_local_command", return_value=False):
            result = tt._get_provider({"provider": "local"})

        helper.assert_called_once_with(prompt=False)
        assert result == "none"

    def test_explicit_local_with_successful_lazy_install_returns_local(self):
        """When lazy install succeeds (helper flips flag to True), provider is 'local'."""
        from tools import transcription_tools as tt

        def helper(*, prompt: bool = False) -> bool:
            tt._HAS_FASTER_WHISPER = True
            return True

        original = tt._HAS_FASTER_WHISPER
        try:
            tt._HAS_FASTER_WHISPER = False
            with patch.object(tt, "_ensure_faster_whisper", side_effect=helper):
                result = tt._get_provider({"provider": "local"})
            assert result == "local"
        finally:
            tt._HAS_FASTER_WHISPER = original

    def test_explicit_local_command_invokes_lazy_install_when_command_missing(self):
        """provider=local_command falls back to local; that fallback must try the install."""
        from tools import transcription_tools as tt

        helper = MagicMock(return_value=False)
        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_ensure_faster_whisper", helper), \
             patch.object(tt, "_has_local_command", return_value=False):
            tt._get_provider({"provider": "local_command"})

        helper.assert_called_once_with(prompt=False)

    def test_explicit_openai_does_NOT_invoke_lazy_install(self, monkeypatch):
        """Cloud providers must not trigger surprise STT installs."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        from tools import transcription_tools as tt

        helper = MagicMock()
        with patch.object(tt, "_HAS_OPENAI", True), \
             patch.object(tt, "_ensure_faster_whisper", helper):
            result = tt._get_provider({"provider": "openai"})

        helper.assert_not_called()
        assert result == "openai"

    def test_auto_detect_with_cloud_key_does_NOT_invoke_lazy_install(self, monkeypatch):
        """Auto-detect picks the cloud provider before reaching the lazy-install fallback."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from tools import transcription_tools as tt

        helper = MagicMock()
        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_has_local_command", return_value=False), \
             patch.object(tt, "_HAS_OPENAI", True), \
             patch.object(tt, "_ensure_faster_whisper", helper):
            result = tt._get_provider({})

        helper.assert_not_called()
        assert result == "groq"

    def test_auto_detect_last_resort_attempts_lazy_install(self, monkeypatch):
        """When nothing else works, auto-detect tries the install as a last resort."""
        from tools import transcription_tools as tt

        helper = MagicMock(return_value=False)
        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_has_local_command", return_value=False), \
             patch.object(tt, "_HAS_OPENAI", False), \
             patch.object(tt, "_ensure_faster_whisper", helper), \
             patch("tools.xai_http.resolve_xai_http_credentials",
                   return_value={"api_key": None}):
            result = tt._get_provider({})

        helper.assert_called_once_with(prompt=False)
        assert result == "none"

    def test_auto_detect_last_resort_succeeds(self):
        """When the last-resort lazy install succeeds, auto-detect returns 'local'."""
        from tools import transcription_tools as tt

        def helper(*, prompt: bool = False) -> bool:
            tt._HAS_FASTER_WHISPER = True
            return True

        original = tt._HAS_FASTER_WHISPER
        try:
            tt._HAS_FASTER_WHISPER = False
            with patch.object(tt, "_has_local_command", return_value=False), \
                 patch.object(tt, "_HAS_OPENAI", False), \
                 patch.object(tt, "_ensure_faster_whisper", side_effect=helper), \
                 patch("tools.xai_http.resolve_xai_http_credentials",
                       return_value={"api_key": None}):
                result = tt._get_provider({})
            assert result == "local"
        finally:
            tt._HAS_FASTER_WHISPER = original
