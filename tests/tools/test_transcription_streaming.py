"""Live STT sessions: wire behaviour over a real socket, and the recorder hand-off."""

from __future__ import annotations

import json
import threading

from websockets.sync.server import serve

from tools import transcription_streaming as ts
from tools import voice_mode


def _xai_like_server(received: list):
    """xAI ``/v1/stt`` shape: created first, a partial per audio frame, stitched final on audio.done."""

    def handler(ws):
        ws.send(json.dumps({"type": "transcript.created"}))
        words = []
        for raw in ws:
            if isinstance(raw, bytes):
                received.append(raw)
                words.append(f"w{len(words)}")
                ws.send(json.dumps({"type": "transcript.partial", "text": " ".join(words),
                                    "is_final": False, "speech_final": False}))
            elif json.loads(raw).get("type") == "audio.done":
                ws.send(json.dumps({"type": "transcript.partial", "text": " ".join(words),
                                    "is_final": True, "speech_final": True}))
                ws.send(json.dumps({"type": "transcript.done", "text": ""}))
                return

    server = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_live_session_streams_partials_before_end_and_keeps_early_audio():
    received: list = []
    server = _xai_like_server(received)
    port = server.socket.getsockname()[1]
    partials: list = []
    seen_partial = threading.Event()

    def on_partial(text):
        partials.append(text)
        seen_partial.set()

    session = ts.XAIStreamingSession("k", f"http://127.0.0.1:{port}/v1", "grok-voice-transcribe-2.0", "en",
                                     on_partial).start()
    session.push_audio(b"\x01\x00" * 160)  # before transcript.created: must be held, not dropped
    session.push_audio(b"\x02\x00" * 160)
    assert seen_partial.wait(5), "a partial must arrive while audio is still streaming"
    result = session.finalize(timeout=5)
    server.shutdown()

    assert result == {"success": True, "transcript": "w0 w1", "provider": "xai"}
    assert len(received) == 2
    assert partials[-1] == "w0 w1"


class _Session:
    def __init__(self, result):
        self.result, self.ended = result, False

    def end_audio(self):
        self.ended = True

    def finalize(self):
        return self.result

    def cancel(self):
        pass


def test_recording_uses_the_live_transcript_and_falls_back_to_the_file(tmp_path, monkeypatch):
    uploads: list = []
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio",
                        lambda path, **kw: uploads.append(path) or {"success": True, "transcript": "from file"})
    live_wav, failed_wav = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")

    live = _Session({"success": True, "transcript": "from live", "provider": "xai"})
    voice_mode._park_live_session(live_wav, live)
    assert live.ended, "parking ends the audio so the provider flushes while the caller works"
    assert voice_mode.transcribe_recording(live_wav)["transcript"] == "from live"
    assert uploads == []

    voice_mode._park_live_session(failed_wav, _Session({"success": False, "transcript": "", "error": "socket"}))
    assert voice_mode.transcribe_recording(failed_wav)["transcript"] == "from file"
    assert uploads == [failed_wav]
