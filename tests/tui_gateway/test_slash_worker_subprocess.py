from __future__ import annotations

import io


class _FakeProc:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def poll(self):
        return 0


def test_slash_worker_decodes_pipes_as_utf8_with_replacement(monkeypatch):
    from tui_gateway import server

    popen_kwargs = {}

    def fake_popen(_argv, **kwargs):
        popen_kwargs.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    worker = server._SlashWorker("session-key", "test-model")
    worker.close()

    assert popen_kwargs["text"] is True
    assert popen_kwargs["encoding"] == "utf-8"
    assert popen_kwargs["errors"] == "replace"
