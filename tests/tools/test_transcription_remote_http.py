"""Exercise remote audio over a real local HTTP transport."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from tools import transcription_remote as remote
from tools import transcription_tools as stt


def test_remote_download_redirect_source_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    payload = b"ID3" + b"audio fixture" * 16
    requests = []
    downloaded = []

    class Handler(BaseHTTPRequestHandler):
        def serve(self, body=False):
            requests.append((self.command, self.path))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/audio.mp3")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if body and self.path != "/redirect":
                self.wfile.write(payload)

        def do_HEAD(self):
            self.serve()

        def do_GET(self):
            self.serve(body=True)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    # Only the controlled server is allowed; requests, redirect handling, file
    # validation and temporary-file cleanup all use their real implementations.
    monkeypatch.setattr(remote, "is_safe_url", lambda value: value.startswith(url + "/"))
    monkeypatch.setattr(stt, "_load_stt_config", lambda: {"provider": "local"})
    monkeypatch.setattr(stt, "_get_provider", lambda _config: "local")

    def transcribe(path, model, source):
        downloaded.append(Path(path))
        assert Path(path).read_bytes() == payload
        assert model == "test-model"
        assert source == "gateway"
        return {"success": True, "transcript": "local transport", "provider": "local"}

    monkeypatch.setattr(stt, "_transcribe_prepared_audio", transcribe)
    try:
        result = stt.transcribe_audio(url + "/redirect", "test-model", source="gateway")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result["success"] is True
    assert result["source_url"] == url + "/audio.mp3"
    assert requests == [("HEAD", "/redirect"), ("HEAD", "/audio.mp3"), ("GET", "/audio.mp3")]
    assert downloaded and all(not path.exists() and not path.parent.exists() for path in downloaded)


def test_direct_groq_preserves_model_language_prompt_and_source(monkeypatch):
    from unittest.mock import Mock

    config = {"provider": "groq", "groq": {"model": "configured-model", "language": "en"},
              "prompt": "configured vocabulary"}
    monkeypatch.setattr(stt, "_load_stt_config", lambda: config)
    monkeypatch.setattr(stt, "_get_provider", lambda _config: "groq")
    monkeypatch.setattr(stt, "_resolve_provider_key", lambda *_args: "test-key")
    monkeypatch.setattr(remote, "_probe_remote_audio_url", lambda url: {
        "success": True, "url": url, "content_type": "audio/mpeg", "content_length": 123})
    hook = Mock(return_value=(None, "fr", "hook vocabulary"))
    limit = Mock(return_value="limited vocabulary")
    monkeypatch.setattr(remote, "_apply_pre_transcription_hook", hook)
    monkeypatch.setattr(remote, "_enforce_prompt_length_limit", limit)
    response = Mock(status_code=200, text="bonjour")
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    url = "https://audio.example.test/clip.mp3"
    result = stt.transcribe_audio(url, source="gateway")
    assert result["success"] is True
    hook.assert_called_once_with(file_path=url, provider="groq", model=None,
                                 language="en", prompt="configured vocabulary", source="gateway")
    limit.assert_called_once_with("hook vocabulary", "groq")
    fields = post.call_args.kwargs["files"]
    assert fields["model"] == (None, "configured-model")
    assert fields["language"] == (None, "fr")
    assert fields["prompt"] == (None, "limited vocabulary")
    hook.return_value = ("hook-model", None, None)
    limit.return_value = None
    stt.transcribe_audio(url, "caller-model", source="voice_mode")
    assert hook.call_args.kwargs["model"] == "caller-model"
    assert post.call_args.kwargs["files"]["model"] == (None, "hook-model")
    assert post.call_args.kwargs["files"]["language"] == (None, "en")
