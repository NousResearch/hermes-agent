"""Regression tests for #17140.

TTS provider tools must resolve API keys from ``~/.hermes/.env`` (via
``hermes_cli.config.get_env_value``) and not only from ``os.environ`` —
otherwise users who keep their keys in the dotenv file see "API key not set"
errors even though the key is configured. Same class of bug as #15914 (auth)
already addressed for ``agent/credential_pool`` and ``hermes_cli/auth``.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Strip every TTS-related env var so the test really exercises the
    dotenv code path. If any of these survive into the test, the assertion
    that ``get_env_value`` was consulted becomes meaningless because
    ``os.environ`` already satisfies the lookup.
    """
    for key in (
        "ELEVENLABS_API_KEY",
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "MINIMAX_API_KEY",
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


class TestDotenvFallbackPerProvider:
    """For each affected provider, when only ``~/.hermes/.env`` carries the
    key, the provider must find it. These per-provider tests model that
    dotenv-backed lookup by mocking ``tools.tts_tool.get_env_value`` directly;
    the separate regression-guard tests cover the lower-level
    ``hermes_cli.config.load_env`` integration. Before the fix, ``os.getenv``
    returned ``None`` and the provider raised
    ``ValueError("X_API_KEY not set")``.
    """

    def test_elevenlabs_reads_dotenv_key(self, tmp_path):
        from tools import tts_tool

        with patch.object(tts_tool, "get_env_value", return_value="el-dotenv-key"), \
             patch.object(tts_tool, "_import_elevenlabs") as mock_import:
            mock_client = MagicMock()
            mock_client.text_to_speech.convert.return_value = iter([b"audio"])
            mock_import.return_value = MagicMock(return_value=mock_client)

            output = str(tmp_path / "out.mp3")
            tts_tool._generate_elevenlabs("hi", output, {})

            mock_import.return_value.assert_called_once_with(api_key="el-dotenv-key")

    def test_xai_reads_dotenv_key(self, tmp_path):
        from tools import tts_tool

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            response = MagicMock()
            response.content = b"audio"
            response.raise_for_status = MagicMock()
            return response

        with patch.object(tts_tool, "get_env_value", return_value="xai-dotenv-key"), \
             patch("requests.post", side_effect=fake_post):
            tts_tool._generate_xai_tts("hi", str(tmp_path / "out.mp3"), {})

        assert captured["headers"]["Authorization"] == "Bearer xai-dotenv-key"

    def test_minimax_reads_dotenv_key(self, tmp_path):
        from tools import tts_tool

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            response = MagicMock()
            response.json.return_value = {
                "data": {"audio": b"\x00\x01".hex()},
                "base_resp": {"status_code": 0},
            }
            response.raise_for_status = MagicMock()
            return response

        with patch.object(tts_tool, "get_env_value", return_value="mm-dotenv-key"), \
             patch("requests.post", side_effect=fake_post):
            tts_tool._generate_minimax_tts("hi", str(tmp_path / "out.mp3"), {})

        assert captured["headers"]["Authorization"] == "Bearer mm-dotenv-key"

    def test_mistral_reads_dotenv_key(self, tmp_path):
        import base64

        from tools import tts_tool

        seen_keys: list = []

        def fake_mistral_factory(*, api_key=None):
            seen_keys.append(api_key)
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.audio.speech.complete.return_value = MagicMock(
                audio_data=base64.b64encode(b"data").decode()
            )
            return client

        with patch.object(tts_tool, "get_env_value", return_value="mistral-dotenv-key"), \
             patch.object(tts_tool, "_import_mistral_client", return_value=fake_mistral_factory):
            tts_tool._generate_mistral_tts("hi", str(tmp_path / "out.mp3"), {})

        assert seen_keys == ["mistral-dotenv-key"]

    def test_gemini_reads_dotenv_key(self, tmp_path):
        from tools import tts_tool

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["params"] = kwargs.get("params", {})
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "data": "AAAA",
                                        "mimeType": "audio/L16;codec=pcm;rate=24000",
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            response.raise_for_status = MagicMock()
            return response

        # GEMINI_API_KEY hits the first branch; GOOGLE_API_KEY would only be
        # consulted if the first returned None. Use a side-effect-style mock
        # to verify the lookup order matches the production code.
        seen_lookups: list = []

        def fake_get_env_value(key):
            seen_lookups.append(key)
            if key == "GEMINI_API_KEY":
                return "gemini-dotenv-key"
            return None

        with patch.object(tts_tool, "get_env_value", side_effect=fake_get_env_value), \
             patch("requests.post", side_effect=fake_post):
            tts_tool._generate_gemini_tts("hi", str(tmp_path / "out.wav"), {})

        assert "GEMINI_API_KEY" in seen_lookups
        assert captured["params"]["key"] == "gemini-dotenv-key"


class TestRegressionGuard:
    """Goal-backward proof that the old behaviour ('only check ``os.environ``')
    breaks reading from a dotenv-only key, and the new behaviour fixes it.
    Implemented as an end-to-end probe that patches
    ``hermes_cli.config.load_env`` to simulate ``~/.hermes/.env`` carrying the
    key while ``os.environ`` does not.
    """

    def test_minimax_missing_when_only_in_dotenv_before_fix(self, tmp_path, monkeypatch):
        from tools import tts_tool

        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

        # Simulate ~/.hermes/.env carrying the key (load_env returns the dict
        # that get_env_value falls back to). The pre-fix ``os.getenv`` call
        # ignores this entirely and raises ValueError.
        with patch(
            "hermes_cli.config.load_env",
            return_value={"MINIMAX_API_KEY": "dotenv-secret"},
        ):
            # Sanity-check: get_env_value resolves through load_env when
            # os.environ is empty.
            from hermes_cli.config import get_env_value as live_get
            assert live_get("MINIMAX_API_KEY") == "dotenv-secret"

            # And the production code path now consumes the resolved value
            # instead of raising "MINIMAX_API_KEY not set".
            captured: dict = {}

            def fake_post(url, **kwargs):
                captured["headers"] = kwargs.get("headers", {})
                response = MagicMock()
                response.json.return_value = {
                    "data": {"audio": b"\x00".hex()},
                    "base_resp": {"status_code": 0},
                }
                response.raise_for_status = MagicMock()
                return response

            with patch("requests.post", side_effect=fake_post):
                tts_tool._generate_minimax_tts(
                    "hi", str(tmp_path / "out.mp3"), {}
                )

            assert captured["headers"]["Authorization"] == "Bearer dotenv-secret"

    def test_check_tts_requirements_sees_dotenv_minimax(self, monkeypatch):
        """``check_tts_requirements`` is the gate that decides whether
        ``/voice on`` is even offered. If it only checked ``os.environ`` it
        would say "no provider available" for users who keep MINIMAX_API_KEY
        in ``~/.hermes/.env``, even though the dispatcher would later succeed.
        """
        from tools import tts_tool

        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

        with patch(
            "hermes_cli.config.load_env",
            return_value={"MINIMAX_API_KEY": "dotenv-secret"},
        ), patch.object(tts_tool, "_import_edge_tts", side_effect=ImportError), \
             patch.object(tts_tool, "_import_elevenlabs", side_effect=ImportError), \
             patch.object(tts_tool, "_import_openai_client", side_effect=ImportError), \
             patch.object(tts_tool, "_check_neutts_available", return_value=False), \
             patch.object(tts_tool, "_check_kittentts_available", return_value=False):
            assert tts_tool.check_tts_requirements() is True
