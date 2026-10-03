"""The synthesis boundary applies equally to direct (gateway) and tool TTS."""
import json
from pathlib import Path
import pytest

from hermes_cli.plugins import PluginManager
from tools import tts_tool


def test_pre_synthesis_blocks_direct_and_tool_without_touching_provider(monkeypatch, tmp_path):
    manager = PluginManager()
    seen = []
    manager._hooks["pre_tts_synthesis"] = [
        lambda text, provider: seen.append((text, provider)) or
        ({"action": "block", "message": "raw number"} if any(c.isnumeric() for c in text) else None)
    ]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {
        "provider": "elevenlabs", "elevenlabs": {"voice_id": "chosen", "model_id": "chosen-model"}})
    generated = []
    def synthesize(text, path, config):
        generated.append((text, config["elevenlabs"]))
        Path(path).write_bytes(b"audio")
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_elevenlabs", synthesize)
    for caller in (lambda text: tts_tool.text_to_speech_tool(text, output_path=str(tmp_path / "direct.mp3")),
                   lambda text: __import__("model_tools").handle_function_call("text_to_speech", {"text": text, "output_path": str(tmp_path / "tool.mp3")})):
        denied = json.loads(caller("Цена 123"))
        assert not denied["success"] and "raw number" in denied["error"]
        assert json.loads(caller("Сто двадцать три"))["success"]
    assert len(generated) == 2
    assert all(cfg == {"voice_id": "chosen", "model_id": "chosen-model"} for _, cfg in generated)
    assert seen == [("Цена 123", "elevenlabs"), ("Сто двадцать три", "elevenlabs")] * 2


def test_pre_synthesis_callback_failure_blocks_before_output(monkeypatch, tmp_path):
    manager = PluginManager()
    def broken(text):
        raise RuntimeError("policy unavailable")
    manager._hooks["pre_tts_synthesis"] = [broken]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    path = tmp_path / "never.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("All words", output_path=str(path)))
    assert not result["success"] and "policy unavailable" in result["error"]
    assert not path.exists()


@pytest.mark.parametrize("directive", ["allow", [], {"action": "unknown"}, {"text": 123}, {"action": "allow", "text": "changed"}])
def test_malformed_policy_directive_fails_closed(monkeypatch, tmp_path, directive):
    manager = PluginManager()
    manager._hooks["pre_tts_synthesis"] = [lambda text: directive]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    path = tmp_path / "never.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("Plain words", output_path=str(path)))
    assert not result["success"] and "pre_tts_synthesis" in result["error"]
    assert not path.exists()


def test_rewrite_after_guard_is_rechecked_before_synthesis(monkeypatch, tmp_path):
    manager = PluginManager()
    manager._hooks["pre_tts_synthesis"] = [
        lambda text: {"action": "block", "message": "raw number"} if any(c.isnumeric() for c in text) else None,
        lambda text: {"text": "Price 123"} if text == "Price one" else None,
    ]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    path = tmp_path / "never.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("Price one", output_path=str(path)))
    assert not result["success"] and "raw number" in result["error"]
    assert not path.exists()


def test_all_guards_see_rewritten_script_in_order(monkeypatch):
    from tools.tts_synthesis_policy import enforce_pre_synthesis
    manager = PluginManager()
    seen = []
    def guard(text, provider):
        seen.append(("guard", text, provider))
    def rewrite(text, provider):
        seen.append(("rewrite", text, provider))
        return {"text": "Price one"} if text == "Price 1" else None
    manager._hooks["pre_tts_synthesis"] = [guard, rewrite]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    assert enforce_pre_synthesis("Price 1", "elevenlabs") == "Price one"
    assert seen == [("guard", "Price 1", "elevenlabs"),
                    ("rewrite", "Price 1", "elevenlabs"),
                    ("guard", "Price one", "elevenlabs"),
                    ("rewrite", "Price one", "elevenlabs")]


def test_late_guard_veto_overrides_earlier_rewrite(monkeypatch):
    from tools.tts_synthesis_policy import enforce_pre_synthesis
    manager = PluginManager()
    manager._hooks["pre_tts_synthesis"] = [
        lambda text: {"text": "Price one"} if text == "Price 1" else None,
        lambda text: {"action": "block", "message": "denied"} if text == "Price 1" else None,
    ]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    with pytest.raises(ValueError, match="denied"):
        enforce_pre_synthesis("Price 1", "elevenlabs")


def test_policy_timeout_fails_closed(monkeypatch):
    import time
    from tools.tts_synthesis_policy import enforce_pre_synthesis
    manager = PluginManager()
    manager._hooks["pre_tts_synthesis"] = [lambda text: time.sleep(.1)]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: .01)
    with pytest.raises(ValueError):
        enforce_pre_synthesis("Price one", "elevenlabs")


def test_speaker_streamer_does_not_call_backend_for_denied_sentence(monkeypatch):
    import queue
    import threading
    from tools import tts_tool_speaker
    manager = PluginManager()
    manager._hooks["pre_tts_synthesis"] = [lambda text: {"action": "block", "message": "raw number"}]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    class Streamer:
        requests = []
        sample_rate = 24000
        channels = 1
        def stream(self, text):
            self.requests.append(text)
            yield b"\x00\x00"

    streamer = Streamer()
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg, preferred=None: streamer)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    monkeypatch.setattr(tts_tool_speaker._StreamerPlayback, "_device_usable", lambda self: False)
    monkeypatch.setattr(tts_tool_speaker, "_play_via_tempfile", lambda *args: None)
    text_queue = queue.Queue()
    text_queue.put("Price 123.")
    text_queue.put(None)
    done = threading.Event()
    tts_tool_speaker.stream_tts_to_speaker(text_queue, threading.Event(), done)
    assert done.is_set() and streamer.requests == []


def test_speaker_policy_sees_selected_streamer_not_sync_default(monkeypatch):
    import queue
    import threading
    from tools import tts_tool_speaker
    manager = PluginManager()
    seen = []
    manager._hooks["pre_tts_synthesis"] = [
        lambda text, provider: seen.append(provider) or
        ({"action": "block"} if provider == "openai" else None)]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    class Streamer:
        provider_name = "openai"
        requests = []
        sample_rate = 24000
        channels = 1
        def stream(self, text):
            self.requests.append(text)
            yield b"\x00\x00"

    streamer = Streamer()
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg, preferred=None: streamer)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool_speaker._StreamerPlayback, "_device_usable", lambda self: False)
    text_queue = queue.Queue()
    text_queue.put("Hello there.")
    text_queue.put(None)
    tts_tool_speaker.stream_tts_to_speaker(text_queue, threading.Event(), threading.Event())
    assert seen == ["openai"] and streamer.requests == []


def test_policy_dispatch_failure_is_not_synthesized(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("plugin manager unavailable")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", unavailable)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    path = tmp_path / "never.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("Plain words", output_path=str(path)))
    assert not result["success"] and "plugin manager unavailable" in result["error"]
    assert not path.exists()
