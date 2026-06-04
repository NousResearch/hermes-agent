"""Tests for the configurable ElevenLabs TTS base URL.

Covers ``_build_elevenlabs_client`` (the resolver used by both the sync
``_generate_elevenlabs`` handler and the streaming ``stream_tts_to_speaker``
path) and the end-to-end wiring through ``_generate_elevenlabs``.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_elevenlabs_module():
    """Patch the lazy SDK import + environment class.

    ``_build_elevenlabs_client`` calls ``_import_elevenlabs()`` for the client
    class and imports ``elevenlabs.environment.ElevenLabsEnvironment`` for the
    custom-origin branch. We stub both so the tests need no real SDK install.
    """
    mock_client = MagicMock()
    mock_cls = MagicMock(return_value=mock_client)

    class FakeEnvironment:
        def __init__(self, *, base, wss):
            self.base = base
            self.wss = wss

    fake_env_module = MagicMock()
    fake_env_module.ElevenLabsEnvironment = FakeEnvironment

    with patch("tools.tts_tool._import_elevenlabs", return_value=mock_cls), patch.dict(
        "sys.modules", {"elevenlabs.environment": fake_env_module}
    ):
        yield mock_cls, mock_client, FakeEnvironment


class TestBuildElevenLabsClient:
    def test_default_omits_environment(self, mock_elevenlabs_module):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        _build_elevenlabs_client("key", {})

        mock_cls.assert_called_once_with(api_key="key")

    def test_explicit_default_url_omits_environment(self, mock_elevenlabs_module):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        _build_elevenlabs_client("key", {"base_url": "https://api.elevenlabs.io"})

        mock_cls.assert_called_once_with(api_key="key")

    def test_config_base_url_sets_environment(self, mock_elevenlabs_module):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, FakeEnvironment = mock_elevenlabs_module
        _build_elevenlabs_client("key", {"base_url": "https://proxy.example.com"})

        _name, kwargs = mock_cls.call_args
        assert kwargs["api_key"] == "key"
        env = kwargs["environment"]
        assert isinstance(env, FakeEnvironment)
        assert env.base == "https://proxy.example.com"
        assert env.wss == "wss://proxy.example.com"

    def test_trailing_v1_is_stripped(self, mock_elevenlabs_module):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        _build_elevenlabs_client("key", {"base_url": "https://proxy.example.com/v1/"})

        env = mock_cls.call_args[1]["environment"]
        assert env.base == "https://proxy.example.com"

    def test_trailing_slash_is_stripped(self, mock_elevenlabs_module):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        _build_elevenlabs_client("key", {"base_url": "https://proxy.example.com/"})

        env = mock_cls.call_args[1]["environment"]
        assert env.base == "https://proxy.example.com"

    def test_env_var_used_when_config_absent(self, mock_elevenlabs_module, monkeypatch):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        monkeypatch.setenv("ELEVENLABS_BASE_URL", "https://env.example.com")
        _build_elevenlabs_client("key", {})

        env = mock_cls.call_args[1]["environment"]
        assert env.base == "https://env.example.com"

    def test_config_overrides_env_var(self, mock_elevenlabs_module, monkeypatch):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        monkeypatch.setenv("ELEVENLABS_BASE_URL", "https://env.example.com")
        _build_elevenlabs_client("key", {"base_url": "https://config.example.com"})

        env = mock_cls.call_args[1]["environment"]
        assert env.base == "https://config.example.com"

    def test_http_origin_derives_ws_scheme(self, mock_elevenlabs_module):
        from tools.tts_tool import _build_elevenlabs_client

        mock_cls, _client, _env = mock_elevenlabs_module
        _build_elevenlabs_client("key", {"base_url": "http://localhost:8080"})

        env = mock_cls.call_args[1]["environment"]
        assert env.base == "http://localhost:8080"
        assert env.wss == "ws://localhost:8080"


class TestGenerateElevenLabsUsesBaseUrl:
    def test_generate_passes_environment(self, tmp_path, mock_elevenlabs_module, monkeypatch):
        from tools.tts_tool import _generate_elevenlabs

        mock_cls, mock_client, _env = mock_elevenlabs_module
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        mock_client.text_to_speech.convert.return_value = [b"audio-bytes"]

        output_path = str(tmp_path / "out.mp3")
        config = {"elevenlabs": {"base_url": "https://proxy.example.com"}}
        result = _generate_elevenlabs("Hello", output_path, config)

        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == b"audio-bytes"
        env = mock_cls.call_args[1]["environment"]
        assert env.base == "https://proxy.example.com"

    def test_generate_default_omits_environment(
        self, tmp_path, mock_elevenlabs_module, monkeypatch
    ):
        from tools.tts_tool import _generate_elevenlabs

        mock_cls, mock_client, _env = mock_elevenlabs_module
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        mock_client.text_to_speech.convert.return_value = [b"audio-bytes"]

        _generate_elevenlabs("Hello", str(tmp_path / "out.mp3"), {})

        mock_cls.assert_called_once_with(api_key="test-key")

    def test_missing_api_key_raises(self, tmp_path, mock_elevenlabs_module):
        from tools.tts_tool import _generate_elevenlabs

        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            _generate_elevenlabs("Hello", str(tmp_path / "out.mp3"), {})
