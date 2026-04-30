"""Regression tests for the transcription_tools variant of #17140.

Same class of bug as ``tools/tts_tool.py`` (fixed in PR #17163): the STT
provider call sites read API keys via ``os.getenv()``, which bypasses
``~/.hermes/.env`` entries. These tests confirm each STT provider now
consults ``get_env_value()`` and the provider auto-detect + explicit
selection gate (``_get_provider``) do the same.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Strip every STT-related env var so the test really exercises the
    dotenv code path. If any of these survive into the test, the assertion
    that ``get_env_value`` was consulted becomes meaningless because
    ``os.environ`` already satisfies the lookup.
    """
    for key in (
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_API_KEY",
        "XAI_STT_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestProviderSelectionGate:
    """``_get_provider`` picks the STT backend. If it only consulted
    ``os.environ`` a user with keys in ``~/.hermes/.env`` would be told
    "no STT available" even though the actual transcribe call would
    succeed. The gate lives behind ``is_stt_enabled(stt_config)``, so
    configure ``{"enabled": True, "provider": ...}`` for explicit tests.
    """

    def test_import_after_config_env_patch_uses_restored_dotenv_loader(self):
        """Importing STT while hermes_cli.config.get_env_value is patched must
        not freeze that temporary helper into this module forever.
        """
        import importlib
        import hermes_cli.config as config_mod
        from tools import transcription_tools as tt

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config_mod, "get_env_value", lambda name, default=None: "")
            tt = importlib.reload(tt)

        try:
            with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
                 patch.object(tt, "_HAS_OPENAI", True), \
                 patch.object(tt, "_has_local_command", return_value=False), \
                 patch("hermes_cli.config.load_env",
                       return_value={"GROQ_API_KEY": "dotenv-secret"}):
                assert tt._get_provider({"enabled": True, "provider": "groq"}) == "groq"
        finally:
            importlib.reload(tt)

    def test_explicit_groq_sees_dotenv(self):
        from tools import transcription_tools as tt

        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_HAS_OPENAI", True), \
             patch.object(tt, "_has_local_command", return_value=False), \
             patch("hermes_cli.config.load_env",
                   return_value={"GROQ_API_KEY": "dotenv-secret"}):
            assert tt._get_provider({"enabled": True, "provider": "groq"}) == "groq"

    def test_explicit_mistral_sees_dotenv(self):
        from tools import transcription_tools as tt

        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_HAS_MISTRAL", True), \
             patch.object(tt, "_has_local_command", return_value=False), \
             patch("hermes_cli.config.load_env",
                   return_value={"MISTRAL_API_KEY": "dotenv-secret"}):
            assert tt._get_provider({"enabled": True, "provider": "mistral"}) == "mistral"

    def test_explicit_xai_sees_dotenv(self):
        from tools import transcription_tools as tt

        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_has_local_command", return_value=False), \
             patch("hermes_cli.config.load_env",
                   return_value={"XAI_API_KEY": "dotenv-secret"}):
            assert tt._get_provider({"enabled": True, "provider": "xai"}) == "xai"

    def test_auto_detect_sees_dotenv_groq(self):
        """No local backend, no explicit provider — auto-detect should fall
        through to Groq when its key lives in dotenv only. Before the fix
        it would return 'none'."""
        from tools import transcription_tools as tt

        with patch.object(tt, "_HAS_FASTER_WHISPER", False), \
             patch.object(tt, "_HAS_OPENAI", True), \
             patch.object(tt, "_HAS_MISTRAL", False), \
             patch.object(tt, "_has_local_command", return_value=False), \
             patch.object(tt, "_has_openai_audio_backend", return_value=False), \
             patch("hermes_cli.config.load_env",
                   return_value={"GROQ_API_KEY": "dotenv-secret"}):
            # No "provider" key → explicit=False → auto-detect branch
            assert tt._get_provider({"enabled": True}) == "groq"


class TestTranscribeCallSitesReadDotenv:
    """The actual transcribe functions must forward the dotenv-resolved
    key into the provider SDK / HTTP call. We mock ``get_env_value`` and
    capture what gets passed through."""

    def test_transcribe_groq_forwards_dotenv_key(self):
        from tools import transcription_tools as tt

        seen_keys: list = []

        class FakeOpenAIClient:
            def __init__(self, *, api_key=None, base_url=None, timeout=None, max_retries=None):
                seen_keys.append(api_key)
                self.audio = MagicMock()
                self.audio.transcriptions.create.return_value = "hello"
            def close(self):
                pass

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI = FakeOpenAIClient
        fake_openai_module.APIError = Exception
        fake_openai_module.APIConnectionError = Exception
        fake_openai_module.APITimeoutError = Exception

        with patch.object(tt, "get_env_value", return_value="groq-dotenv-key"), \
             patch.object(tt, "_HAS_OPENAI", True), \
             patch.dict("sys.modules", {"openai": fake_openai_module}), \
             patch("builtins.open", MagicMock()):
            result = tt._transcribe_groq("/tmp/fake.mp3", "whisper-large-v3-turbo")

        assert result["success"] is True
        assert seen_keys == ["groq-dotenv-key"]

    def test_transcribe_mistral_forwards_dotenv_key(self):
        from tools import transcription_tools as tt

        seen_keys: list = []

        class FakeMistralClient:
            def __init__(self, *, api_key=None):
                seen_keys.append(api_key)
                self.audio = MagicMock()
                completion = MagicMock()
                completion.text = "hi"
                self.audio.transcriptions.complete.return_value = completion
            def __enter__(self): return self
            def __exit__(self, *a): return False

        fake_client_module = MagicMock()
        fake_client_module.Mistral = FakeMistralClient

        with patch.object(tt, "get_env_value", return_value="mistral-dotenv-key"), \
             patch.dict("sys.modules", {"mistralai.client": fake_client_module}), \
             patch("builtins.open", MagicMock()):
            result = tt._transcribe_mistral("/tmp/fake.mp3", "voxtral-mini-latest")

        assert result["success"] is True
        assert seen_keys == ["mistral-dotenv-key"]

    def test_transcribe_xai_forwards_dotenv_key(self):
        from tools import transcription_tools as tt

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            response.json.return_value = {"text": "hello"}
            return response

        # get_env_value is consulted for both XAI_API_KEY and XAI_STT_BASE_URL.
        # Return the key for the first call, None for base-url override
        # (so it defaults to the module-level XAI_STT_BASE_URL).
        def fake_get_env_value(name, default=None):
            if name == "XAI_API_KEY":
                return "xai-dotenv-key"
            return None

        with patch.object(tt, "get_env_value", side_effect=fake_get_env_value), \
             patch("requests.post", side_effect=fake_post), \
             patch("builtins.open", MagicMock()):
            result = tt._transcribe_xai("/tmp/fake.mp3", "grok-stt")

        assert result["success"] is True
        assert captured["headers"]["Authorization"] == "Bearer xai-dotenv-key"


class TestEndToEndRegressionGuard:
    """End-to-end probe: patch ``hermes_cli.config.load_env`` to simulate
    ``~/.hermes/.env`` carrying the key while ``os.environ`` does not.
    Before the fix ``_transcribe_xai`` called ``os.getenv("XAI_API_KEY")``
    directly and returned ``XAI_API_KEY not set``."""

    def test_xai_key_only_in_dotenv_before_fix(self, monkeypatch):
        from tools import transcription_tools as tt

        monkeypatch.delenv("XAI_API_KEY", raising=False)

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            response.json.return_value = {"text": "ok"}
            return response

        with patch("hermes_cli.config.load_env",
                   return_value={"XAI_API_KEY": "dotenv-secret"}):
            # Sanity: get_env_value resolves through load_env when
            # os.environ is empty.
            from hermes_cli.config import get_env_value as live_get
            assert live_get("XAI_API_KEY") == "dotenv-secret"

            with patch("requests.post", side_effect=fake_post), \
                 patch("builtins.open", MagicMock()):
                result = tt._transcribe_xai("/tmp/fake.mp3", "grok-stt")

        assert result["success"] is True
        assert captured["headers"]["Authorization"] == "Bearer dotenv-secret"
