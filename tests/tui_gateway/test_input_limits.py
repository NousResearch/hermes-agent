"""Size caps on the stdio transport and on command fields (tui_gateway/_input_limits.py)."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import _input_limits as limits

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


class _CountingStream(io.StringIO):
    """StringIO that records the largest single readline() result it handed out."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.largest = 0

    def readline(self, size=-1):
        out = super().readline(size)
        self.largest = max(self.largest, len(out))
        return out


# ── defaults ─────────────────────────────────────────────────────────


def test_default_frame_cap_matches_websocket_attachment_cap(monkeypatch):
    from hermes_cli.web_server_boundary import _DESKTOP_ATTACHMENT_WS_MAX_BYTES

    monkeypatch.delenv("HERMES_TUI_MAX_FRAME_BYTES", raising=False)
    assert limits.max_frame_bytes() == _DESKTOP_ATTACHMENT_WS_MAX_BYTES


@pytest.mark.parametrize("raw", ["", "0", "-5", "nope"])
def test_invalid_env_override_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("HERMES_TUI_MAX_FIELD_BYTES", raw)
    assert limits.max_field_bytes() == limits._DEFAULT_FIELD_BYTES


# ── read_frame ───────────────────────────────────────────────────────


def test_read_frame_passes_lines_at_the_cap_and_eof(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_MAX_FRAME_BYTES", "8")
    stream = io.StringIO("1234567\n" + "abc\n")
    assert limits.read_frame(stream) == "1234567\n"  # 8 chars incl. newline == cap
    assert limits.read_frame(stream) == "abc\n"
    assert limits.read_frame(stream) == ""


def test_read_frame_drains_oversized_line_without_buffering_it(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_MAX_FRAME_BYTES", "16")
    stream = _CountingStream("x" * 500 + "\n" + '{"id": 2}\n')
    with pytest.raises(limits.FrameTooLarge) as info:
        limits.read_frame(stream)
    assert info.value.size == 501 and info.value.limit == 16
    assert stream.largest <= 17  # never more than one cap-sized chunk in memory
    assert limits.read_frame(stream) == '{"id": 2}\n'  # next request starts clean


def test_read_frame_oversized_tail_without_newline_then_eof(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_MAX_FRAME_BYTES", "4")
    stream = io.StringIO("y" * 50)
    with pytest.raises(limits.FrameTooLarge):
        limits.read_frame(stream)
    assert limits.read_frame(stream) == ""


# ── field_error ──────────────────────────────────────────────────────


def test_field_error_counts_utf8_bytes(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_MAX_FIELD_BYTES", "8")
    assert limits.field_error("arg", "abcdefgh") is None
    assert limits.field_error("arg", "😀😀") is None  # 8 bytes
    msg = limits.field_error("arg", "😀😀😀")  # 3 chars, 12 bytes
    assert msg == "arg too long: 12 bytes (limit 8)"
    assert limits.field_error("arg", None) is None
    assert limits.field_error("arg", 123) is None


# ── stdio entry loop ─────────────────────────────────────────────────


def test_entry_rejects_oversized_frame_and_keeps_serving(monkeypatch):
    from tui_gateway import entry

    monkeypatch.setenv("HERMES_TUI_MAX_FRAME_BYTES", "64")
    replies: list[dict] = []
    big = json.dumps({"jsonrpc": "2.0", "id": "big", "method": "ping", "params": {"pad": "z" * 500}})
    ok = json.dumps({"jsonrpc": "2.0", "id": "next", "method": "ping"})

    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(entry.server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *_args: False)
    monkeypatch.setattr(entry, "write_json", lambda payload: replies.append(payload) or True)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(big + "\n" + ok + "\n"))
    seen: list = []
    monkeypatch.setattr(entry, "dispatch",
                        lambda req: seen.append(req["id"]) or {"jsonrpc": "2.0", "id": req["id"], "result": {}})

    entry.main()

    assert seen == ["next"]  # the oversized request never reached dispatch
    rejected = [r for r in replies if "error" in r]
    assert len(rejected) == 1
    assert rejected[0]["id"] is None and rejected[0]["error"]["code"] == -32600
    assert {"jsonrpc": "2.0", "id": "next", "result": {}} in replies


# ── command-field RPCs ───────────────────────────────────────────────


@pytest.fixture()
def server():
    # Same import isolation as tests/tui_gateway/test_protocol.py's fixture.
    import tui_gateway.server_requests  # noqa: F401
    import tui_gateway.transport  # noqa: F401
    import hermes_constants

    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test"),
                                      get_process_hermes_home=hermes_constants.get_process_hermes_home),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)


def _call(server, method, params):
    return server._methods[method]("r1", params)


@pytest.mark.parametrize("method,params", [
    ("command.dispatch", {"name": "N" * 33, "arg": ""}),
    ("command.dispatch", {"name": "help", "arg": "A" * 33}),
    ("shell.exec", {"command": "echo " + "C" * 40}),
])
def test_oversized_command_fields_are_rejected_up_front(server, monkeypatch, method, params):
    monkeypatch.setenv("HERMES_TUI_MAX_FIELD_BYTES", "32")
    # Nothing past the guard may run: resolution, approval detection, the shell.
    monkeypatch.setattr(server, "_resolve_name", MagicMock(side_effect=AssertionError("resolved")))
    monkeypatch.setattr(server, "_tools_mod", MagicMock(side_effect=AssertionError("approval ran")))
    resp = _call(server, method, params)
    assert resp["error"]["code"] == 4000
    assert "too long" in resp["error"]["message"]


def test_slash_exec_rejects_oversized_command_before_worker(server, monkeypatch):
    monkeypatch.setenv("HERMES_TUI_MAX_FIELD_BYTES", "32")
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: ({"session_key": "k"}, None))
    monkeypatch.setattr(server, "_live_slash_command_output",
                        MagicMock(side_effect=AssertionError("command was routed")))
    resp = _call(server, "slash.exec", {"session_id": "s", "command": "/echo " + "S" * 40})
    assert resp["error"]["code"] == 4000


def test_fields_at_the_cap_still_reach_the_handler(server, monkeypatch):
    monkeypatch.setenv("HERMES_TUI_MAX_FIELD_BYTES", "32")
    reached = MagicMock(side_effect=lambda name: name)
    monkeypatch.setattr(server, "_resolve_name", reached)
    _call(server, "command.dispatch", {"name": "n" * 32, "arg": "a" * 32})
    reached.assert_called_once_with("n" * 32)
