"""`agent.terminal.output` tells the desktop viewer whether the chunk came from a PTY."""

import threading
import time
from types import SimpleNamespace

from tools import process_registry as registry_mod
from tools.process_registry import ProcessSession
from tui_gateway import session_notifications
from tui_gateway.contracts.events import TerminalOutputPayload
from tui_gateway.method_ctx import rebind


def _wire(monkeypatch):
    emitted = []
    monkeypatch.setattr(registry_mod.process_registry, "on_output", None)
    monkeypatch.setattr(registry_mod.process_registry, "on_close", None)
    wire = rebind(session_notifications._wire_desktop_sinks, {
        **vars(session_notifications),
        "_emit": lambda event, sid, payload=None: emitted.append((event, payload)),
        "_sessions_lock": threading.Lock(), "_sessions": {},
        "_desktop_ui_wired": True,
    })
    wire()
    return emitted


def _session(sid, pty):
    session = ProcessSession(id=sid, command="cmd", task_id="t", started_at=time.time())
    session._pty = pty
    return session


def test_live_chunk_carries_the_pty_flag(monkeypatch):
    emitted = _wire(monkeypatch)
    sink = registry_mod.process_registry.on_output

    sink(_session("proc_pty", SimpleNamespace()), "\x1b[2;3H\x1b[K\n")
    sink(_session("proc_pipe", None), "line\n")

    payloads = {p["process_id"]: p for event, p in emitted if event == "agent.terminal.output"}
    assert payloads["proc_pty"]["pty"] is True
    assert payloads["proc_pipe"]["pty"] is False
    for payload in payloads.values():
        TerminalOutputPayload.model_validate(payload)
