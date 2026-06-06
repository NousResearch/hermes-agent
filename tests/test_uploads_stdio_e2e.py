"""E2E test against a real tui_gateway subprocess.

Spawns `python -m tui_gateway.entry` as a subprocess, exchanges real
JSON-RPC over stdio (the way the TUI client does in production), and
verifies the file.* handlers behave correctly under real I/O.

This is the strongest test we can run in CI without the full TUI
client: the gateway is the same code the TS client talks to, just
exercised from Python instead of TypeScript.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

GATEWAY_CMD = [sys.executable, "-m", "tui_gateway.entry"]


def _read_event(proc, timeout=10.0):
    """Read one JSON line from gateway stdout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON line (shouldn't happen, but be tolerant)
            continue
    raise TimeoutError("no response from gateway within timeout")


def _send(proc, req):
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()


def _make_session_id():
    return f"e2e-stdio-{os.getpid()}-{int(time.time() * 1000) % 100000}"


def _open_session(gateway, label="e2e"):
    """Create a session via the real JSON-RPC session.create method.

    The server generates the session id; we capture it from the
    response. Returns the session id string.
    """
    _send(gateway, {"id": "sc1", "method": "session.create", "params": {"label": label, "cols": 80}})
    resp = _read_event(gateway)
    assert "result" in resp, f"session.create failed: {resp}"
    assert "session_id" in resp["result"]
    return resp["result"]["session_id"]


@pytest.fixture
def gateway(tmp_path):
    """Spawn a tui_gateway subprocess; clean up on teardown."""
    env = os.environ.copy()
    env["HERMES_SANDBOX_ROOT"] = str(tmp_path / "sandbox")
    # Disable MCP discovery noise (we don't have any configured)
    env.setdefault("HERMES_HOME", str(tmp_path / "hermes-home"))
    proc = subprocess.Popen(
        GATEWAY_CMD,
        cwd="/opt/data/home/worktree/hermes-tui-uploads",
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )
    # First message should be gateway.ready
    ready = _read_event(proc)
    assert ready.get("method") == "event", f"expected event, got {ready}"
    assert ready["params"]["type"] == "gateway.ready"
    yield proc
    # Teardown — close stdin so the loop terminates, then wait
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_session_create_returns_id(gateway):
    sid = _open_session(gateway, label="e2e")
    assert isinstance(sid, str) and len(sid) > 0


def test_file_attach_through_real_stdio(gateway, tmp_path):
    sid = _open_session(gateway, label="e2e-attach")

    # Create a real file to attach
    md = tmp_path / "stdio.md"
    md.write_bytes(b"# Stdio test\n\nThis file went through a real subprocess.\n")

    # Attach via real JSON-RPC
    _send(gateway, {
        "id": "fa1",
        "method": "file.attach",
        "params": {"session_id": sid, "path": str(md)},
    })
    resp = _read_event(gateway)
    assert "result" in resp, f"attach failed: {resp}"
    assert resp["result"]["attached"] is True
    assert resp["result"]["mime_type"] == "text/markdown"
    assert resp["result"]["kind"] == "TEXT"
    assert "preview_text" in resp["result"]
    assert "Stdio test" in resp["result"]["preview_text"]


def test_file_list_reflects_attach(gateway, tmp_path):
    sid = _open_session(gateway, label="list")

    md = tmp_path / "listable.md"
    md.write_bytes(b"# List test\n")

    _send(gateway, {"id": "fa1", "method": "file.attach", "params": {"session_id": sid, "path": str(md)}})
    attach_resp = _read_event(gateway)
    assert "result" in attach_resp
    file_id = attach_resp["result"]["id"]

    _send(gateway, {"id": "fl1", "method": "file.list", "params": {"session_id": sid}})
    list_resp = _read_event(gateway)
    assert "result" in list_resp
    files = list_resp["result"]["files"]
    assert len(files) == 1
    assert files[0]["id"] == file_id


def test_file_detach_removes_from_list(gateway, tmp_path):
    sid = _open_session(gateway, label="detach")

    f = tmp_path / "detach.md"
    f.write_bytes(b"# Detach test\n")
    _send(gateway, {"id": "fa1", "method": "file.attach", "params": {"session_id": sid, "path": str(f)}})
    attach_resp = _read_event(gateway)
    file_id = attach_resp["result"]["id"]

    _send(gateway, {"id": "fd1", "method": "file.detach", "params": {"session_id": sid, "id": file_id}})
    detach_resp = _read_event(gateway)
    assert "result" in detach_resp
    assert detach_resp["result"]["detached"] is True

    _send(gateway, {"id": "fl1", "method": "file.list", "params": {"session_id": sid}})
    list_resp = _read_event(gateway)
    assert list_resp["result"]["files"] == []


def test_spoof_rejected_via_stdio(gateway, tmp_path):
    """An ELF binary renamed to .png is rejected by the gateway."""
    sid = _open_session(gateway, label="spoof")

    evil = tmp_path / "fake.png"
    evil.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 100)

    _send(gateway, {"id": "fa1", "method": "file.attach", "params": {"session_id": sid, "path": str(evil)}})
    resp = _read_event(gateway)
    assert "error" in resp, f"expected error, got: {resp}"
    assert "not allowed" in resp["error"]["message"].lower()
