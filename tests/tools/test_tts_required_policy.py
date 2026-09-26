"""Required TTS policy stays closed when a configured plugin fails to load."""
import json
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import _reset_plugin_managers_for_tests


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    _reset_plugin_managers_for_tests()
    a, b = tmp_path / "a", tmp_path / "b"
    for home in (a, b):
        home.mkdir()
    yield a, b
    _reset_plugin_managers_for_tests()


def configure(home, *, required=None, broken=False):
    cfg = {"tts": {"provider": "edge"}}
    if required is not None:
        cfg["tts"]["pre_synthesis"] = {"required_plugins": required}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    if broken:
        plugin = home / "plugins" / "sample-policy"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: sample-policy\nversion: 1.0.0\n")
        (plugin / "__init__.py").write_text(
            "def register(ctx):\n"
            "    ctx.register_hook('pre_tts_synthesis', lambda text: None)\n"
            "    raise ImportError('broken policy import')\n"
        )
        cfg["plugins"] = {"enabled": ["sample-policy"]}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def test_required_failed_plugin_blocks_manual_before_provider_and_is_profile_scoped(homes, monkeypatch, tmp_path):
    from tools import tts_tool
    a, b = homes
    configure(a, required=["sample-policy"], broken=True)
    configure(b)
    calls = []
    async def generate(text, path, config):
        calls.append(text)
        Path(path).write_bytes(b"audio")
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", generate)
    for index, (home, expected) in enumerate(((a, False), (b, True), (a, False))):
        token = set_hermes_home_override(home)
        try:
            result = json.loads(tts_tool.text_to_speech_tool("Price 123", output_path=str(tmp_path / "audio.mp3")))
            assert result["success"] is expected, result
            if not expected:
                assert "sample-policy" in result["error"]
                assert len(calls) == (1 if index == 2 else 0)
        finally:
            reset_hermes_home_override(token)


def test_required_failed_plugin_blocks_gateway_auto_and_streaming_client(homes, monkeypatch, tmp_path):
    from tools import tts_tool
    from tools.voice_client_config import resolve_client_voice_config
    from model_tools import handle_function_call
    a, b = homes
    configure(a, required=["sample-policy"], broken=True)
    configure(b)
    cfg = yaml.safe_load((b / "config.yaml").read_text())
    cfg["tts"]["provider"] = "openai"
    (b / "config.yaml").write_text(yaml.safe_dump(cfg))
    cfg = yaml.safe_load((a / "config.yaml").read_text())
    cfg["tts"]["provider"] = "openai"
    (a / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    token = set_hermes_home_override(a)
    try:
        calls = []
        async def generate(text, path, config):
            calls.append(text)
            Path(path).write_bytes(b"audio")
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", generate)
        result = json.loads(handle_function_call("text_to_speech", {"text": "Price 123", "output_path": str(tmp_path / "auto.mp3")}))
        assert not result["success"] and "sample-policy" in result["error"]
        assert not calls
        assert resolve_client_voice_config()["tts"]["mode"] == "relay"
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(b)
    try:
        assert resolve_client_voice_config()["tts"]["mode"] == "direct"
    finally:
        reset_hermes_home_override(token)


def test_required_missing_plugin_blocks_without_any_provider_call(homes, monkeypatch, tmp_path):
    from tools import tts_tool
    a, _ = homes
    configure(a, required=["not-installed"])
    invoked = []
    monkeypatch.setattr(tts_tool, "_run_edge_tts", lambda *args: invoked.append(args))
    token = set_hermes_home_override(a)
    try:
        result = json.loads(tts_tool.text_to_speech_tool("Plain words", output_path=str(tmp_path / "never.mp3")))
        assert not result["success"] and "not-installed" in result["error"]
        assert not invoked
    finally:
        reset_hermes_home_override(token)


def test_required_failed_plugin_blocks_gateway_pcm_stream_before_provider(homes):
    import asyncio
    from gateway.streaming_tts_consumer import StreamingTTSConsumer
    a, _ = homes
    configure(a, required=["sample-policy"], broken=True)
    consumer = StreamingTTSConsumer.__new__(StreamingTTSConsumer)
    class Streamer:
        def __init__(self):
            self.requests = []
        def stream(self, text):
            self.requests.append(text)
            yield b"audio"
    streamer = Streamer()
    consumer._streamer = streamer
    consumer._handle = None
    consumer._strip_markdown = lambda text: text
    consumer._provider = "edge"
    token = set_hermes_home_override(a)
    try:
        with pytest.raises(ValueError, match="sample-policy"):
            asyncio.run(consumer._synthesise_and_write("Price 123"))
        assert not streamer.requests
    finally:
        reset_hermes_home_override(token)


def test_required_healthy_plugin_runs_and_default_policy_remains_unchanged(homes):
    from tools.tts_synthesis_policy import enforce_pre_synthesis
    a, b = homes
    configure(a, required=["sample-policy"])
    configure(b)
    plugin = a / "plugins" / "sample-policy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: sample-policy\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_hook('pre_tts_synthesis', lambda text: "
        "{'action': 'block', 'message': 'denied'} if any(c.isnumeric() for c in text) else None)\n"
    )
    cfg = yaml.safe_load((a / "config.yaml").read_text())
    cfg["plugins"] = {"enabled": ["sample-policy"]}
    (a / "config.yaml").write_text(yaml.safe_dump(cfg))
    for home, blocked in ((a, True), (b, False), (a, True)):
        token = set_hermes_home_override(home)
        try:
            if blocked:
                assert enforce_pre_synthesis("Price one", "edge") == "Price one"
                with pytest.raises(ValueError, match="denied"):
                    enforce_pre_synthesis("Price 123", "edge")
            else:
                assert enforce_pre_synthesis("Price 123", "edge") == "Price 123"
        finally:
            reset_hermes_home_override(token)


def test_required_healthy_plugin_must_register_the_hook(homes):
    from tools.tts_synthesis_policy import enforce_pre_synthesis
    a, _ = homes
    configure(a, required=["sample-policy"])
    plugin = a / "plugins" / "sample-policy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: sample-policy\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text("def register(ctx):\n    ctx.register_hook('post_tool_call', lambda **kwargs: None)\n")
    cfg = yaml.safe_load((a / "config.yaml").read_text())
    cfg["plugins"] = {"enabled": ["sample-policy"]}
    (a / "config.yaml").write_text(yaml.safe_dump(cfg))
    token = set_hermes_home_override(a)
    try:
        with pytest.raises(ValueError, match="sample-policy"):
            enforce_pre_synthesis("Words", "edge")
    finally:
        reset_hermes_home_override(token)
