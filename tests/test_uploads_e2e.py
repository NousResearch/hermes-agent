"""End-to-end test: full file.attach → file.list → file.detach cycle.

Exercises the whole pipeline (whitelist + sandbox + magic-byte
detection + cleanup) as a real user would experience it. The
gateway handlers are imported directly so we test the same
code path the TUI client hits over JSON-RPC.
"""

import pytest

from tui_gateway import server


def _session_dict(**extra):
    return {
        "agent": __import__("types").SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def test_attach_list_detach_cycle(monkeypatch, tmp_path):
    """Attach a markdown file, list it, detach it, list is empty."""
    sid = "e2e-cycle"
    monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
    server._sessions[sid] = _session_dict()

    # 1. Attach.
    md = tmp_path / "cycle.md"
    md.write_bytes(b"# E2E\n\nThis file goes through the full pipeline.\n")
    attach_resp = server.handle_request(
        {"id": "1", "method": "file.attach", "params": {"session_id": sid, "path": str(md)}}
    )
    assert "result" in attach_resp, f"attach failed: {attach_resp}"
    file_id = attach_resp["result"]["id"]
    assert attach_resp["result"]["attached"] is True
    assert attach_resp["result"]["mime_type"] == "text/markdown"
    assert attach_resp["result"]["kind"] == "TEXT"
    assert attach_resp["result"]["size_bytes"] > 0
    assert "preview_text" in attach_resp["result"]
    assert "E2E" in attach_resp["result"]["preview_text"]

    # 2. List.
    list_resp = server.handle_request(
        {"id": "2", "method": "file.list", "params": {"session_id": sid}}
    )
    assert "result" in list_resp
    files = list_resp["result"]["files"]
    assert len(files) == 1
    assert files[0]["id"] == file_id
    # The list response shows the stored filename (sha16 prefix + ext)
    # — that's the canonical name in the sandbox. The user-facing
    # original name ("cycle.md") is in the attach response.
    assert files[0]["name"].endswith(".md")
    assert files[0]["mime_type"] == "text/markdown"

    # 3. Detach.
    detach_resp = server.handle_request(
        {"id": "3", "method": "file.detach", "params": {"session_id": sid, "id": file_id}}
    )
    assert "result" in detach_resp
    assert detach_resp["result"]["detached"] is True

    # 4. List again — empty.
    list_resp2 = server.handle_request(
        {"id": "4", "method": "file.list", "params": {"session_id": sid}}
    )
    assert list_resp2["result"]["files"] == []


def test_spoofed_extension_rejected(monkeypatch, tmp_path):
    """An ELF binary renamed to .png is rejected by MIME detection."""
    sid = "e2e-spoof"
    monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
    server._sessions[sid] = _session_dict()

    # Real ELF magic bytes, bad extension.
    evil = tmp_path / "innocent.png"
    evil.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 100)

    resp = server.handle_request(
        {"id": "1", "method": "file.attach", "params": {"session_id": sid, "path": str(evil)}}
    )
    assert "error" in resp, f"expected error, got: {resp}"
    assert "not allowed" in resp["error"]["message"].lower()


def test_size_limit_enforced(monkeypatch, tmp_path):
    """A file over MAX_UPLOAD_SIZE_BYTES is rejected before copy."""
    import cli

    sid = "e2e-size"
    monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "MAX_UPLOAD_SIZE_BYTES", 10)  # tiny
    server._sessions[sid] = _session_dict()

    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)

    resp = server.handle_request(
        {"id": "1", "method": "file.attach", "params": {"session_id": sid, "path": str(big)}}
    )
    assert "error" in resp
    assert "size" in resp["error"]["message"].lower()


def test_quotes_and_spaces_in_path(monkeypatch, tmp_path):
    """A path with spaces and quotes (e.g. macOS screenshots) attaches."""
    sid = "e2e-quoted"
    monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
    server._sessions[sid] = _session_dict()

    # Simulate a screenshot with spaces.
    weird = tmp_path / "Screenshot 2026-04-21 at 1.04.43 PM.png"
    weird.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfe\x02\xfe\xa3\x9b"
        b"\xe0\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    # User pastes the path with surrounding double quotes.
    resp = server.handle_request(
        {
            "id": "1",
            "method": "file.attach",
            "params": {"session_id": sid, "path": f'"{weird}"'},
        }
    )
    assert "result" in resp, f"failed: {resp}"
    assert resp["result"]["attached"] is True
    assert "Screenshot" in resp["result"]["name"]
