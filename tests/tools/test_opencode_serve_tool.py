"""Tests for tools/opencode_serve_tool.py."""

from __future__ import annotations

import json

import httpx
import pytest

from tools.opencode_serve_tool import (
    MAX_SUMMARY_CHARS,
    _diff_summary,
    _save_sessions,
    _session_store,
    _text_from_parts,
    opencode_run,
    opencode_status,
)

SERVER_URL = "http://opencode.test:4096"


@pytest.fixture
def opencode_env(monkeypatch):
    """Point the tool at a fake server. Conftest strips credential env vars,
    so these must be (re)set per test."""
    monkeypatch.setenv("OPENCODE_SERVER_URL", SERVER_URL)
    monkeypatch.setenv("OPENCODE_SERVER_USERNAME", "opencode")
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "s3cret")
    monkeypatch.delenv("OPENCODE_DEFAULT_PROJECT", raising=False)


@pytest.fixture
def mock_server(monkeypatch):
    """Swap the module's httpx.Client for a MockTransport-backed client.

    opencode_run/opencode_status construct their own client internally, so
    patch the class the module calls. Capture the real client first to avoid
    recursion when constructing the transport-backed replacement.
    """
    real_client = httpx.Client

    def install(handler):
        def factory(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr("tools.opencode_serve_tool.httpx.Client", factory)

    return install


def _route(routes):
    """Build a MockTransport handler from a {(method, path): fn} map."""

    def handler(request: httpx.Request) -> httpx.Response:
        fn = routes.get((request.method, request.url.path))
        if fn is None:
            return httpx.Response(404, json={"error": "no route"})
        return fn(request)

    return handler


# ── helpers ────────────────────────────────────────────────────────────────


def test_text_from_parts_joins_and_filters():
    parts = [
        {"type": "text", "text": "hello"},
        {"type": "tool", "text": "not a text part"},
        {"type": "text", "text": "  world  "},
        {"type": "text", "text": ""},
    ]
    assert _text_from_parts(parts) == "hello\n  world"


def test_text_from_parts_truncates_long_output():
    long = "x" * (MAX_SUMMARY_CHARS + 500)
    out = _text_from_parts([{"type": "text", "text": long}])
    assert out.startswith("x" * MAX_SUMMARY_CHARS)
    assert "truncated; use opencode_status for the rest" in out


def test_diff_summary_formats_file_stats():
    client = httpx.Client(
        base_url=SERVER_URL,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json=[
                    {"path": "tools/foo.py", "additions": 3, "deletions": 1},
                    {"path": "docs/x.md", "additions": 0, "deletions": 0},
                ],
            )
        )
    )
    out = _diff_summary(client, "sess-1", "msg-1")
    assert "Files changed:" in out
    assert "- tools/foo.py (+3 -1)" in out
    assert "- docs/x.md (+0 -0)" in out


def test_diff_summary_empty_and_error_cases():
    empty = httpx.Client(base_url=SERVER_URL, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[])))
    assert _diff_summary(empty, "sess-1", "msg-1") == ""
    # no message id → no diff request at all
    assert _diff_summary(empty, "sess-1", None) == ""
    err = httpx.Client(base_url=SERVER_URL, transport=httpx.MockTransport(lambda req: httpx.Response(500)))
    assert _diff_summary(err, "sess-1", "msg-1") == ""


# ── opencode_run ───────────────────────────────────────────────────────────


def test_opencode_run_blocking_happy_path(opencode_env, mock_server):
    mock_server(
        _route(
            {
                ("GET", "/global/health"): lambda req: httpx.Response(200, json={"ok": True}),
                ("POST", "/session"): lambda req: httpx.Response(200, json={"id": "sess-1"}),
                ("POST", "/session/sess-1/message"): lambda req: httpx.Response(
                    200,
                    json={
                        "info": {"id": "msg-1"},
                        "parts": [{"type": "text", "text": "Task done"}],
                    },
                ),
                ("GET", "/session/sess-1/diff"): lambda req: httpx.Response(
                    200, json=[{"path": "tools/foo.py", "additions": 3, "deletions": 1}]
                ),
            }
        )
    )

    result = opencode_run("fix the bug", project="/srv/proj")

    assert "[session sess-1," in result
    assert "Task done" in result
    assert "Files changed:" in result
    assert "- tools/foo.py (+3 -1)" in result
    # session mapping persisted via the atomic store
    assert json.loads(_session_store().read_text()) == {"/srv/proj": "sess-1"}


def test_opencode_run_background_dispatches(opencode_env, mock_server):
    mock_server(
        _route(
            {
                ("GET", "/global/health"): lambda req: httpx.Response(200, json={"ok": True}),
                ("POST", "/session"): lambda req: httpx.Response(200, json={"id": "sess-1"}),
                ("POST", "/session/sess-1/prompt_async"): lambda req: httpx.Response(204),
            }
        )
    )

    result = opencode_run("long task", project="/srv/proj", background=True)

    assert "Dispatched to opencode session sess-1 in background." in result


def test_opencode_run_reuses_existing_session(opencode_env, mock_server):
    _save_sessions({"/srv/proj": "sess-old"})
    posts = {"count": 0}

    def create(request):
        posts["count"] += 1
        return httpx.Response(200, json={"id": "sess-new"})

    mock_server(
        _route(
            {
                ("GET", "/global/health"): lambda req: httpx.Response(200, json={"ok": True}),
                ("GET", "/session/sess-old"): lambda req: httpx.Response(200, json={"id": "sess-old"}),
                ("POST", "/session"): create,
                ("POST", "/session/sess-old/message"): lambda req: httpx.Response(
                    200,
                    json={"info": {"id": "m"}, "parts": [{"type": "text", "text": "ok"}]},
                ),
            }
        )
    )

    result = opencode_run("x", project="/srv/proj")

    assert posts["count"] == 0
    assert "[session sess-old," in result
    assert json.loads(_session_store().read_text()) == {"/srv/proj": "sess-old"}


def test_opencode_run_inactive_without_env(monkeypatch):
    monkeypatch.delenv("OPENCODE_SERVER_URL", raising=False)
    result = opencode_run("x", project="/srv/proj")
    assert "OPENCODE_SERVER_URL is not set" in result


def test_opencode_run_connect_error(opencode_env, mock_server):
    def unreachable(request):
        raise httpx.ConnectError("connection refused", request=request)

    mock_server(unreachable)
    result = opencode_run("x", project="/srv/proj")
    assert "cannot reach opencode server" in result


# ── opencode_status ────────────────────────────────────────────────────────


def test_opencode_status_last_message_fallback(opencode_env, mock_server):
    _save_sessions({"/srv/proj": "sess-1"})
    mock_server(
        _route(
            {
                ("GET", "/session/status"): lambda req: httpx.Response(200, json={}),
                ("GET", "/session/sess-1/message"): lambda req: httpx.Response(
                    200,
                    json=[
                        {
                            "info": {"role": "assistant"},
                            "parts": [{"type": "text", "text": "All good"}],
                        }
                    ],
                ),
            }
        )
    )

    result = opencode_status(project="/srv/proj")

    assert "Session sess-1" in result
    assert "role=assistant" in result
    assert "All good" in result


def test_opencode_status_reports_session_map(opencode_env, mock_server):
    _save_sessions({"/srv/proj": "sess-1"})
    mock_server(
        _route(
            {
                ("GET", "/session/status"): lambda req: httpx.Response(
                    200, json={"sess-1": {"state": "idle"}}
                ),
            }
        )
    )

    result = opencode_status(project="/srv/proj")

    assert "Session sess-1 status:" in result
    assert '"state": "idle"' in result


def test_opencode_status_no_session(opencode_env):
    result = opencode_status(project="/srv/proj")
    assert "No opencode session for project /srv/proj yet" in result
