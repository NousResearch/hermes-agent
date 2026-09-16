"""Regression test: an inline stdio handler raising must not kill the gateway.

``entry.main()``'s read loop used to call ``dispatch(req)`` bare, so any
exception from an inline (non-pool) RPC handler unwound out of ``main()``
and terminated the gateway child — in-flight replies were lost and the TUI
stayed wedged until a restart. The ws.py transport already degraded a
dispatch crash to one JSON-RPC error reply; the stdio loop must do the
same: reply ``-32603`` with the request's id, keep the loop alive, and
still serve later requests normally.

Harness: same style as tests/tui_gateway/test_entry_picker_prewarm.py —
import ``tui_gateway.entry`` and monkeypatch its module attributes, running
the real ``main()`` with stubbed I/O collaborators (no subprocess, no real
gateway).
"""

from __future__ import annotations

import io
import json

from tui_gateway import entry


def _run_main(monkeypatch, replies, stdin_text, *, crash_log=None):
    """Run entry.main() with stubbed collaborators, capturing every write_json
    payload dict into *replies*. Returns normally once stdin hits EOF."""

    def _write_json(payload):
        replies.append(payload)
        return True

    if crash_log is None:

        def _no_crash_log(*args, **kwargs):
            return None

        crash_log = _no_crash_log

    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
    monkeypatch.setattr(entry, "_append_crash_log", crash_log)
    monkeypatch.setattr(entry, "write_json", _write_json)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(stdin_text))

    entry.main()


def test_inline_handler_crash_degrades_to_error_reply_and_loop_survives(monkeypatch):
    """A raising inline handler produces exactly one -32603 reply carrying the
    request's id, the loop survives, and a later request is served normally."""
    bad = {"jsonrpc": "2.0", "method": "paste.clipboard_image", "id": 41}
    good = {"jsonrpc": "2.0", "method": "echo.ok", "id": 42}
    good_resp = {"jsonrpc": "2.0", "result": {"ok": True}, "id": 42}
    stdin_text = json.dumps(bad) + "\n" + json.dumps(good) + "\n"

    def _dispatch(req):
        if req.get("id") == 41:
            raise NameError("handler exploded during image paste")
        return good_resp

    monkeypatch.setattr(entry, "dispatch", _dispatch)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *a: False)

    replies: list = []
    _run_main(monkeypatch, replies, stdin_text)  # returning at all proves survival

    error_replies = [r for r in replies if r.get("error", {}).get("code") == -32603]
    assert len(error_replies) == 1, f"want exactly one -32603 reply, got {replies!r}"
    assert error_replies[0]["id"] == 41
    assert good_resp in replies, "a request after the crash must still be served"


def test_crashing_dispatch_writes_crash_log_breadcrumb(monkeypatch):
    """The crash breadcrumb (forensics for the degraded reply) is appended via
    _append_crash_log with the failing method named in the header."""
    breadcrumbs: list = []

    def _crash_log(header, dump=None):
        breadcrumbs.append(header)

    monkeypatch.setattr(
        entry, "dispatch", lambda req: (_ for _ in ()).throw(NameError("boom"))
    )
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *a: False)

    replies: list = []
    _run_main(
        monkeypatch,
        replies,
        json.dumps({"jsonrpc": "2.0", "method": "paste.clipboard_image", "id": 7})
        + "\n",
        crash_log=_crash_log,
    )

    assert len(breadcrumbs) == 1, f"want one breadcrumb, got {breadcrumbs!r}"
    assert "paste.clipboard_image" in breadcrumbs[0]
    assert replies[-1]["error"]["code"] == -32603
