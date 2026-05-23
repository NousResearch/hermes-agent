import json
from pathlib import Path


def test_voicebox_speak_to_file_posts_speak_polls_and_downloads(tmp_path, monkeypatch):
    from tools import voicebox_tool

    calls = []

    def fake_http_json(method, path, *, payload=None, config=None, timeout=None):
        calls.append((method, path, payload, config))
        if path == "/speak":
            return {"id": "gen-1", "status": "generating"}
        raise AssertionError(path)

    monkeypatch.setattr(voicebox_tool, "_http_json", fake_http_json)
    monkeypatch.setattr(voicebox_tool, "_poll_generation", lambda generation_id, **kw: {"id": generation_id, "status": "completed"})
    monkeypatch.setattr(voicebox_tool, "_http_bytes", lambda path, **kw: b"RIFFfake-wav")

    out = tmp_path / "voice.wav"
    result = voicebox_tool.voicebox_speak_to_file(
        "Build complete",
        str(out),
        profile="Morgan",
        engine="qwen",
        language="en",
        tts_config={"voicebox": {"base_url": "http://127.0.0.1:17493", "client_id": "test-client"}},
    )

    assert result["success"] is True
    assert result["generation_id"] == "gen-1"
    assert out.read_bytes() == b"RIFFfake-wav"
    assert calls[0][0] == "POST"
    assert calls[0][1] == "/speak"
    assert calls[0][2]["profile"] == "Morgan"
    assert calls[0][2]["engine"] == "qwen"


def test_voicebox_status_reports_setup_needed_on_connection_error(monkeypatch):
    from tools import voicebox_tool

    monkeypatch.setattr(voicebox_tool, "_http_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("down")))

    data = json.loads(voicebox_tool.voicebox_status_tool())

    assert data["success"] is False
    assert "setup_needed" in data
    assert "down" in data["error"]


def test_voicebox_transcribe_posts_multipart(tmp_path, monkeypatch):
    from tools import voicebox_tool

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFfake")
    captured = {}

    def fake_post(path, file_path, *, fields=None, config=None, timeout=None):
        captured["path"] = path
        captured["file_path"] = file_path
        captured["fields"] = fields
        return {"text": "hello", "duration": 1.0}

    monkeypatch.setattr(voicebox_tool, "_post_multipart_file", fake_post)

    data = json.loads(voicebox_tool.voicebox_transcribe_tool(str(audio), language="en", model="turbo"))

    assert data["success"] is True
    assert data["text"] == "hello"
    assert captured == {"path": "/transcribe", "file_path": str(audio), "fields": {"language": "en", "model": "turbo"}}


def test_tts_provider_voicebox_uses_voicebox_sidecar(tmp_path, monkeypatch):
    from tools import tts_tool

    out = tmp_path / "tts.wav"

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "voicebox", "voicebox": {"profile": "Morgan"}})
    monkeypatch.setattr(tts_tool, "_convert_to_opus", lambda path: None)

    from tools import voicebox_tool

    def fake_speak(text, output_path=None, **kwargs):
        assert output_path is not None
        Path(str(output_path)).write_bytes(b"RIFFfake-wav")
        return {"success": True, "file_path": str(output_path), "generation_id": "gen-2"}

    monkeypatch.setattr(voicebox_tool, "voicebox_speak_to_file", fake_speak)

    data = json.loads(tts_tool.text_to_speech_tool("Hello", output_path=str(out)))

    assert data["success"] is True
    assert data["provider"] == "voicebox"
    assert data["file_path"] == str(out)
    assert out.read_bytes() == b"RIFFfake-wav"
