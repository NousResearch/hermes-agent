"""Tests for the assets.* JSON-RPC surface (server-authoritative asset index).

Everything is pinned to tmp: a real SessionDB with MEDIA-tagged assistant
messages, a tmp attachments dir, and a tmp projects.db so project grouping
runs against real project_for_path semantics.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as server


def _call(method, params=None):
    handler = server._methods[method]
    return handler(1, params or {})


def _ok(method, params=None):
    resp = _call(method, params)
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.fixture()
def assets_env(tmp_path, monkeypatch):
    """SessionDB with one MEDIA-delivering session, tmp attachments + projects.db."""
    from hermes_cli import projects_db as pdb
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    workdir = tmp_path / "work"
    workdir.mkdir()
    delivered = workdir / "report.pdf"
    delivered.write_bytes(b"%PDF-1.4 fake")
    gone = workdir / "gone.png"
    gone.write_bytes(b"png")
    db.create_session("s-1", "cli", cwd=str(workdir))
    db.append_message("s-1", "assistant",
                     f"Here you go.\nMEDIA:{delivered}\nMEDIA:{gone}\n"
                     "MEDIA: not-a-real-tag-in-prose")
    gone.unlink()  # delivered-then-deleted: must drop out of the index

    attachments = tmp_path / "attachments"
    attachments.mkdir()
    upload = attachments / "photo.jpg"
    upload.write_bytes(b"jpegbytes")

    import tui_gateway.methods_assets as assets  # noqa: F401  (import for module presence)
    # Handlers are rebound onto server's globals by bind_module — patch the
    # binding production reads (server._attachments_root), not the defining
    # module, or the patch is invisible at call time.
    monkeypatch.setattr(server, "_attachments_root", lambda: attachments)

    # projects.db at tmp: one project owning the workdir.
    monkeypatch.setattr(pdb, "get_hermes_home", lambda: tmp_path)
    with pdb.connect_closing() as conn:
        project_id = pdb.create_project(conn, name="Work Project",
                                       primary_path=str(workdir))

    monkeypatch.setattr(server, "_get_db", lambda: db)
    yield {"db": db, "delivered": delivered, "upload": upload,
           "workdir": workdir, "project_id": project_id}
    db.close()


def test_assets_methods_registered():
    for m in ("assets.status", "assets.list"):
        assert m in server._methods


def test_status_reports_attachments_dir(assets_env):
    result = _ok("assets.status")
    assert result["available"] is True
    assert result["attachments_present"] is True


def test_list_finds_media_and_attachment(assets_env):
    result = _ok("assets.list")
    by_name = {r["name"]: r for r in result["assets"]}
    assert "report.pdf" in by_name, result["assets"]
    assert "photo.jpg" in by_name
    assert by_name["report.pdf"]["kind"] == "artifact"  # pdf: not a media mime
    assert by_name["photo.jpg"]["kind"] == "attachment"
    assert by_name["report.pdf"]["session_id"] == "s-1"
    # Deleted-then-delivered file is absent; prose MEDIA: never matched.
    assert "gone.png" not in by_name
    assert all("not-a-real-tag-in-prose" not in r["name"] for r in result["assets"])
    assert result["total"] == len(result["assets"])


def test_kind_filter(assets_env):
    result = _ok("assets.list", {"kind": "attachment"})
    assert [r["name"] for r in result["assets"]] == ["photo.jpg"]
    resp = _call("assets.list", {"kind": "bogus"})
    assert resp["error"]["code"] == 5091


def test_project_grouping(assets_env):
    result = _ok("assets.list", {"project_id": assets_env["project_id"]})
    names = {r["name"] for r in result["assets"]}
    assert "report.pdf" in names  # lives under the project's primary path
    assert "photo.jpg" not in names  # outside the project
