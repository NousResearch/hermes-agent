"""Remote-mode local folder upload via ``file.attach`` + ``extract:true`` (#120449).

A desktop on a remote backend (SSH/URL) cannot hand the agent a local folder:
dropped/picked folders become ``@folder:`` refs to paths the backend can't see.
The fix reuses the existing single-file upload channel: the client zips the
folder and the gateway expands the archive server-side into ``attachments/``.
"""

import base64
import io
import sys
import threading
import types
import zipfile

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
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


def _zip_data_url(files: dict[str, bytes]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return "data:application/zip;base64," + base64.b64encode(buf.getvalue()).decode()


def _remote_cli(monkeypatch):
    """Pretend the client path is invisible from the gateway (remote backend)."""
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)


def test_file_attach_extract_expands_zip_into_session_attachments(monkeypatch, tmp_path):
    """Remote folder upload: zip bytes + extract:true → @folder: ref, structure kept."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    _remote_cli(monkeypatch)
    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "C:/Users/alice/Documents/invoices",
                    "name": "invoices.zip",
                    "data_url": _zip_data_url(
                        {
                            "jan.pdf": b"%PDF-1.4 fake",
                            "feb/scan.txt": b"scanned",
                            ".git/HEAD": b"ref: refs/heads/main",
                            "node_modules/dep/index.js": b"junk",
                        }
                    ),
                    "extract": True,
                },
            }
        )
        result = resp["result"]
        assert result["attached"] is True
        assert result["extracted"] is True
        assert result["file_count"] == 2
        folder = home / "attachments" / "invoices"
        assert result["path"] == str(folder)
        assert result["ref_text"] == f"@folder:{folder}"
        assert (folder / "jan.pdf").read_bytes() == b"%PDF-1.4 fake"
        assert (folder / "feb" / "scan.txt").read_text() == "scanned"
        # Dependency/VCS trees never cross the wire's staging.
        assert not (folder / ".git").exists()
        assert not (folder / "node_modules").exists()
        # The intermediate archive is not kept alongside the expansion.
        assert not (home / "attachments" / "invoices.zip").exists()
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_extract_rejects_zip_slip(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    _remote_cli(monkeypatch)
    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    try:
        resp = server.handle_request(
            {
                "id": "2",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "C:/Users/alice/evil",
                    "name": "evil.zip",
                    "data_url": _zip_data_url({"../../evil.txt": b"x", "ok.txt": b"y"}),
                    "extract": True,
                },
            }
        )
        assert "error" in resp
        assert not (home / "attachments" / "evil").exists()
        assert not (workspace / "evil.txt").exists()
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_extract_rejects_non_zip(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    _remote_cli(monkeypatch)
    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    try:
        resp = server.handle_request(
            {
                "id": "3",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "C:/Users/alice/notes.txt",
                    "name": "notes.txt",
                    "data_url": "data:text/plain;base64,aGVsbG8=",
                    "extract": True,
                },
            }
        )
        assert "error" in resp
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_without_extract_keeps_single_file_behavior(monkeypatch, tmp_path):
    """No extract flag → the pre-existing single-file staging is untouched."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    _remote_cli(monkeypatch)
    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    try:
        resp = server.handle_request(
            {
                "id": "4",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "C:/Users/alice/report.txt",
                    "name": "report.txt",
                    "data_url": "data:text/plain;base64,aGVsbG8gd29ybGQ=",
                },
            }
        )
        result = resp["result"]
        assert result["attached"] is True
        assert result.get("extracted") is not True
        assert result["ref_text"].startswith("@file:")
        assert (home / "attachments" / "report.txt").read_text() == "hello world"
    finally:
        server._sessions.pop("sid", None)
