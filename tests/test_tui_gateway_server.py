import json
import threading
import time
from unittest.mock import patch

from tui_gateway import server


class _ChunkyStdout:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        for ch in text:
            self.parts.append(ch)
            time.sleep(0.0001)
        return len(text)

    def flush(self) -> None:
        return None


class _BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def test_write_json_serializes_concurrent_writes(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server.sys, "stdout", out)

    threads = [
        threading.Thread(target=server.write_json, args=({"seq": i, "text": "x" * 24},))
        for i in range(8)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    lines = "".join(out.parts).splitlines()

    assert len(lines) == 8
    assert {json.loads(line)["seq"] for line in lines} == set(range(8))


def test_write_json_returns_false_on_broken_pipe(monkeypatch):
    monkeypatch.setattr(server.sys, "stdout", _BrokenStdout())

    assert server.write_json({"ok": True}) is False


def test_status_callback_emits_kind_and_text():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("context_pressure", "85% to compaction")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "context_pressure", "text": "85% to compaction"},
    )


def test_status_callback_accepts_single_message_argument():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("thinking...")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "status", "text": "thinking..."},
    )
