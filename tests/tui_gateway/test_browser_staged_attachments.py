"""Exercise browser staging and file.attach through their real public routes."""
from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest


@pytest.fixture
def attachment_runtime(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Real imports after both home sources are isolated. No model/tool client is
    # needed: file.attach only requires a live session record and a filesystem.
    from fastapi.testclient import TestClient
    from hermes_cli import install_identity, profile_incarnation, web_server
    from tui_gateway import server

    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "staged-attachment-test")
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(install_identity, "_INSTALL_ID_CACHE", {"root": None, "value": None})

    def profile_home(profile):
        target = home / "profiles" / profile if profile else home
        target.mkdir(parents=True, exist_ok=True)
        if profile:
            (target / "config.yaml").write_text("{}\n", encoding="utf-8")
        return target

    def add_session(profile=None, sid="runtime"):
        target = profile_home(profile)
        server._sessions[sid] = {
            "agent": SimpleNamespace(),
            "session_key": sid,
            "cwd": str(workspace),
            "profile_home": str(target),
            "profile_incarnation": profile_incarnation.ensure_profile_incarnation(target),
        }
        return target

    with TestClient(web_server.app) as client:
        def upload(profile=None, payload=b"browser payload"):
            profile_home(profile)
            response = client.post(
                "/api/chat/file-upload",
                params={"profile": profile} if profile else None,
                files={"file": ("notes.txt", payload, "text/plain")},
                headers={"X-Hermes-Session-Token": "staged-attachment-test"},
            )
            assert response.status_code == 200, response.text
            return response.json()

        def attach(staged, sid="runtime", **extra):
            return server.handle_request({
                "id": "attach",
                "method": "file.attach",
                "params": {"session_id": sid, "staged_upload": staged, **extra},
            })

        yield SimpleNamespace(
            add_session=add_session,
            attach=attach,
            home=home,
            install_identity=install_identity,
            profile_incarnation=profile_incarnation,
            profile_home=profile_home,
            server=server,
            upload=upload,
        )


@pytest.mark.parametrize(
    ("source_profile", "target_profile"),
    [(None, None), ("worker", "worker"), ("foreground", "tile-owner")],
)
def test_uploaded_file_reaches_the_owning_session_cache_without_retransmission(
    attachment_runtime, source_profile, target_profile
):
    runtime = attachment_runtime
    uploaded = runtime.upload(source_profile)
    target_home = runtime.add_session(target_profile)
    receipt = uploaded["staged_upload"]
    assert receipt["path"] == uploaded["path"]
    assert receipt["profile_home"] == str(runtime.profile_home(source_profile))
    assert receipt["profile_incarnation"] == runtime.profile_incarnation.read_profile_incarnation(
        runtime.profile_home(source_profile)
    )
    response = runtime.attach(receipt)
    assert "error" not in response, response
    stored = Path(response["result"]["path"])
    # This is the same cache directory mounted into container/SSH backends.
    from tools.credential_files import _CACHE_DIRS
    assert stored.parent.name in {relative for relative, _ in _CACHE_DIRS}
    assert stored.parent == target_home / "attachments"
    assert response["result"]["uploaded"] is True
    assert stored.read_bytes() == Path(uploaded["path"]).read_bytes() == b"browser payload"
    assert response["result"]["ref_path"] == str(stored)


def test_source_recreated_with_same_upload_name_rejects_old_provenance(attachment_runtime):
    runtime = attachment_runtime
    uploaded = runtime.upload("worker")
    source = Path(uploaded["path"])
    source_home = runtime.profile_home("worker")
    shutil.rmtree(source_home)
    runtime.profile_home("worker")
    runtime.profile_incarnation.write_fresh_profile_incarnation(source_home)
    # Simulate restore/clone-all preserving old upload filenames. The current
    # target session is fresh, so only checking its incarnation misses this.
    source.parent.mkdir()
    source.write_bytes(b"replacement generation")
    target_home = runtime.add_session("worker")
    response = runtime.attach(uploaded["staged_upload"])
    assert "error" in response
    assert "incarnation" in response["error"]["message"].lower()
    assert not (target_home / "attachments").exists()
    assert source.read_bytes() == b"replacement generation"


def test_source_provenance_cannot_cross_a_backend_replacement(attachment_runtime):
    runtime = attachment_runtime
    uploaded = runtime.upload()
    target_home = runtime.add_session()
    # A replacement backend at the same URL/path has a different persisted
    # install identity. Existing source bytes alone must not authorize reuse.
    (runtime.home / "install_id").write_text(uuid.uuid4().hex + "\n", encoding="utf-8")
    runtime.install_identity._INSTALL_ID_CACHE.clear()
    response = runtime.attach(uploaded["staged_upload"])
    assert "error" in response
    assert "another Hermes backend" in response["error"]["message"]
    assert not (target_home / "attachments").exists()
    assert Path(uploaded["path"]).is_file()


def test_source_remains_reusable_after_process_cache_and_runtime_recovery(attachment_runtime):
    runtime = attachment_runtime
    uploaded = runtime.upload("worker")
    runtime.add_session("worker", "old-runtime")
    runtime.server._sessions.clear()
    runtime.install_identity._INSTALL_ID_CACHE.clear()
    target_home = runtime.add_session("worker", "new-runtime")
    response = runtime.attach(uploaded["staged_upload"], sid="new-runtime")
    assert "error" not in response, response
    assert Path(response["result"]["path"]).parent == target_home / "attachments"
    assert Path(response["result"]["path"]).read_bytes() == b"browser payload"


def test_failed_staged_attach_keeps_source_available_for_retry(attachment_runtime, monkeypatch):
    runtime = attachment_runtime
    uploaded = runtime.upload("worker")
    target_home = runtime.add_session("worker")
    write_bytes = Path.write_bytes

    def fail_target(path, data):
        if path.parent == target_home / "attachments":
            raise PermissionError("temporary target failure")
        return write_bytes(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail_target)
        failed = runtime.attach(uploaded["staged_upload"])
        assert "temporary target failure" in failed["error"]["message"]
    assert Path(uploaded["path"]).read_bytes() == b"browser payload"
    recovered = runtime.attach(uploaded["staged_upload"])
    assert recovered["result"]["attached"] is True
    assert Path(recovered["result"]["path"]).read_bytes() == b"browser payload"


def test_missing_staged_source_never_falls_back_to_supplied_bytes(attachment_runtime):
    runtime = attachment_runtime
    uploaded = runtime.upload("worker")
    target_home = runtime.add_session("worker")
    Path(uploaded["path"]).unlink()
    response = runtime.attach(
        uploaded["staged_upload"],
        path=uploaded["path"],
        data_url="data:text/plain;base64,d3Jvbmc=",
    )
    assert "error" in response
    assert not (target_home / "attachments").exists()


def test_changed_staged_file_still_obeys_browser_size_limit(attachment_runtime):
    from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES

    runtime = attachment_runtime
    uploaded = runtime.upload("worker")
    target_home = runtime.add_session("worker")
    with Path(uploaded["path"]).open("wb") as handle:
        handle.truncate(WEBAPP_ATTACHMENT_MAX_BYTES + 1)
    response = runtime.attach(uploaded["staged_upload"])
    assert "error" in response
    assert "size limit" in response["error"]["message"]
    assert not (target_home / "attachments").exists()
