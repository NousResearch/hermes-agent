"""Remote attachment RPCs must never read gateway-host paths."""

import sys
import types

from tui_gateway import server
from tests.tui_gateway.test_tui_gateway_server import _attach_bytes_cli, _session


class PeerWS:
    def __init__(self, peer):
        self._peer = peer

    def write(self, _obj):
        return True


def dispatch(method, params, *, peer="203.0.113.9:54321"):
    return server.dispatch(
        {"id": "1", "method": method, "params": {"session_id": "sid", **params}},
        transport=PeerWS(peer),
    )


def test_image_host_path_refused_for_remote_peer():
    server._sessions["sid"] = _session()
    try:
        result = dispatch("image.attach", {"path": "/tmp/cat.png"})
        assert result["error"]["code"] == 4016
        assert "remote connection" in result["error"]["message"]
        assert server._sessions["sid"]["attached_images"] == []
    finally:
        server._sessions.pop("sid", None)


def test_pdf_host_path_refused_for_remote_peer(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/pdftoppm")
    server._sessions["sid"] = _session()
    try:
        result = dispatch("pdf.attach", {"path": "/etc/anything.pdf"})
        assert result["error"]["code"] == 4016
        assert "remote connection" in result["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_pdf_host_path_still_resolves_for_loopback(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/pdftoppm")
    server._sessions["sid"] = _session()
    try:
        result = dispatch("pdf.attach", {"path": "/nonexistent/file.pdf"}, peer="127.0.0.1:54321")
        assert result["error"]["code"] == 4016
        assert "PDF not found" in result["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_file_host_path_refused_for_remote_peer(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    source = tmp_path / "outside.txt"
    source.write_text("gateway secret", encoding="utf-8")
    fake_cli = types.ModuleType("cli")
    fake_cli._resolve_attachment_path = lambda raw: source
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    try:
        result = dispatch("file.attach", {"path": str(source)})
        assert result["error"]["code"] == 5028
        assert "remote connection" in result["error"]["message"]
        assert not (home / "attachments" / "outside.txt").exists()
    finally:
        server._sessions.pop("sid", None)


def test_remote_file_upload_ignores_gateway_host_path(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    source = tmp_path / "outside.txt"
    source.write_text("gateway secret", encoding="utf-8")
    fake_cli = types.ModuleType("cli")
    fake_cli._resolve_attachment_path = lambda raw: source
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    try:
        result = dispatch("file.attach", {
            "path": str(source), "data_url": "data:text/plain;base64,Y2xpZW50IGJ5dGVz",
        })
        stored = home / "attachments" / "attachment"
        assert result["result"]["attached"] is True
        assert result["result"]["uploaded"] is True
        assert result["result"]["path"] == str(stored)
        assert stored.read_text(encoding="utf-8") == "client bytes"
    finally:
        server._sessions.pop("sid", None)


def test_remote_drop_preflight_never_resolves_host_path(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: (_ for _ in ()).throw(
        AssertionError("remote drop preflight must not resolve host paths")
    )
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    server._sessions["sid"] = _session()
    try:
        for text in ("/tmp/cat.png", "ordinary prompt text"):
            result = dispatch("input.detect_drop", {"text": text})
            assert result["result"] == {"matched": False}
            assert server._sessions["sid"]["attached_images"] == []
    finally:
        server._sessions.pop("sid", None)
