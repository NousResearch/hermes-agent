"""Profile editor writes resolve and publish under one lifecycle admission."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from hermes_cli import profile_lifecycle
from hermes_cli.profile_incarnation import ensure_profile_incarnation, write_fresh_profile_incarnation
import tui_gateway.server as server


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    path = root / "profiles" / "editor"
    (path / "assets").mkdir(parents=True)
    (path / "assets" / "avatar.png").write_bytes(b"old avatar")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ensure_profile_incarnation(path)
    return path


@pytest.mark.parametrize("operation", ["upload", "clear", "configure"])
def test_editor_publication_excludes_profile_replacement(home, tmp_path, monkeypatch, operation):
    entered, release = threading.Event(), threading.Event()
    name = "_configure_ui_meta" if operation == "configure" else "_unlink_asset_files"
    original = getattr(server, name)

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, name, pause)
    params = {"name": "editor"}
    method = "profiles.set_asset"
    if operation == "upload":
        import base64
        params["data"] = base64.b64encode(b"\x89PNG\r\n\x1a\nnew avatar").decode()
    elif operation == "clear":
        params["clear"] = True
    else:
        method = "profiles.configure"
        params.update(ui_meta={"label": "old generation"}, soul="old soul", description="old description")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(server._methods[method], 1, params)
        assert entered.wait(5)
        try:
            # This is the same cross-thread/process authority create/delete use.
            # A replacement cannot pass its publication gate until the write ends.
            with pytest.raises(TimeoutError):
                with profile_lifecycle.profile_lifecycle_lease(home, timeout=0.05):
                    pass
        finally:
            release.set()
        assert future.result(timeout=5)["result"]["ok"] is True

    # Once the old request finishes, replacement and a fresh editor both work.
    with profile_lifecycle.profile_lifecycle_lease(home):
        home.rename(tmp_path / "retired")
        home.mkdir()
        write_fresh_profile_incarnation(home)
        (home / "SOUL.md").write_text("successor")
    fresh = server._methods["profiles.configure"](2, {"name": "editor", "soul": "fresh edit"})
    assert fresh["result"]["applied"]["soul"] is True
    assert (home / "SOUL.md").read_text() == "fresh edit"
    assert not (home / "assets").exists()
