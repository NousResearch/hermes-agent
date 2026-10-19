"""Push-to-talk events for a shared-owner session reach the client that started the capture.

On the shared gateway the session lives in the authority, not in this process's ``_sessions``, so
``write_json`` cannot route a ``voice.transcript`` by session id. The STT callback runs on a worker
thread with no transport bound; before the fix the frame fell through to stdio and the TUI never
received its transcript.
"""

import sys
import threading
import types

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class _Sink:
    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass


def test_voice_transcript_for_an_unregistered_session_goes_to_the_capturing_client(monkeypatch):
    callbacks = {}
    monkeypatch.setitem(sys.modules, "hermes_cli.voice", types.SimpleNamespace(
        start_continuous=lambda **kw: callbacks.update(kw), stop_continuous=lambda **kw: None))
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", dict)
    ws, stdio = _Sink(), _Sink()
    monkeypatch.setattr(server, "_stdio_transport", stdio)
    monkeypatch.setattr(server, "_voice_event_sid", "")
    assert "owner-sid" not in server._sessions

    token = bind_transport(ws)
    try:
        resp = server.handle_request({"id": "r", "method": "voice.record",
                                      "params": {"action": "start", "session_id": "owner-sid"}})
    finally:
        reset_transport(token)
    assert resp["result"]["status"] == "recording"

    worker = threading.Thread(target=callbacks["on_transcript"], args=("hello there",))
    worker.start()
    worker.join()

    transcripts = [f["params"] for f in ws.frames if f.get("params", {}).get("type") == "voice.transcript"]
    assert [(t["session_id"], t["payload"]["text"]) for t in transcripts] == [("owner-sid", "hello there")]
    assert not [f for f in stdio.frames if f.get("params", {}).get("type") == "voice.transcript"]
