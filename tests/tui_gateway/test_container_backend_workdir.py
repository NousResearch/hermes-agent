"""#103147: a container backend's ``terminal.cwd`` (docker ``/workspace``) lives inside the sandbox.

The gateway used to host-validate it with ``os.path.isdir``, fail, and fall back to its own launch
directory ($HOME for the desktop). Every host file then looked "inside the workspace", so
``file.attach`` skipped staging into the bind-mounted ``attachments/`` dir and the agent was handed a
host path that does not exist in the container.
"""

import os
import uuid
from pathlib import Path

import pytest

from agent.context_references import preprocess_context_references
from tui_gateway import server


def _container_dir() -> str:
    path = f"/hermes-test-workspace-{uuid.uuid4().hex[:8]}"
    assert not os.path.isdir(path)
    return path


def _write_cfg(home: Path, body: str) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    """A temp launch HERMES_HOME the gateway and the cache-mount mapper both read."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    # The desktop's gateway runs from the user's home: a host dir holding the user's files.
    launch_dir = tmp_path / "user-home"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    return home


@pytest.mark.parametrize("bridged", [False, True], ids=["config-only", "env-bridged"])
def test_container_cwd_is_the_session_workspace(launch_home, monkeypatch, bridged):
    container = _container_dir()
    if bridged:  # launchers bridge terminal.* into TERMINAL_ENV / TERMINAL_CWD
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CWD", container)
    else:  # in-process desktop gateway: only config.yaml says docker
        _write_cfg(launch_home, f"terminal:\n  backend: docker\n  cwd: {container}\n")

    assert server._completion_cwd({}) == container
    # Same answer the terminal tool already gets.
    assert server._terminal_task_cwd(None) == container
    session = {"cwd": server._completion_cwd({}), "profile_home": None}
    assert not server._session_is_local_backend(session)
    # Never "healed" to the nearest host ancestor (``/``).
    assert server._display_session_cwd(session) == container


def test_local_backend_missing_cwd_still_falls_back(launch_home):
    _write_cfg(launch_home, f"terminal:\n  backend: local\n  cwd: {_container_dir()}\n")
    assert server._completion_cwd({}) == os.getcwd()


def test_container_backend_host_cwd_keeps_host_path(launch_home, tmp_path):
    """A host dir (the ``docker_mount_cwd_to_workspace`` source) is still validated and kept."""
    project = tmp_path / "project"
    project.mkdir()
    _write_cfg(launch_home, f"terminal:\n  backend: docker\n  cwd: {project}\n")
    assert server._completion_cwd({}) == str(project)


def test_named_container_profile_uses_its_container_cwd(launch_home, tmp_path, monkeypatch):
    container = _container_dir()
    profile = _write_cfg(tmp_path / "box", f"terminal:\n  backend: docker\n  cwd: {container}\n")
    monkeypatch.setattr(server, "_profile_home", lambda name: profile if name == "box" else None)
    inherited = str(tmp_path / "user-home")

    assert server._completion_cwd({"profile": "box", "cwd": inherited, "cwd_explicit": False}) == container
    assert server._completion_cwd({"profile": "box"}) == container
    assert not server._session_is_local_backend({"cwd": container, "profile_home": str(profile)})


def test_attached_host_file_is_staged_and_agent_sees_container_path(launch_home, monkeypatch):
    container = _container_dir()
    _write_cfg(launch_home, f"terminal:\n  backend: docker\n  cwd: {container}\n")
    report = Path.cwd() / "Downloads" / "report.pdf"
    report.parent.mkdir()
    report.write_bytes(b"%PDF-1.4\x00\x01binary")
    session = {"cwd": server._completion_cwd({}), "profile_home": None}
    monkeypatch.setitem(server._sessions, "container-attach", session)

    stored, uploaded = server._stage_session_file_attachment(
        session, raw_path=str(report), data_url="", name="",
        owner=server._attachment_owner(session, "container-attach"))

    assert uploaded is True
    assert stored.parent == (launch_home / "attachments").resolve()
    assert stored.read_bytes() == report.read_bytes()
    ref_path = server._attachment_ref_path(session, stored)
    assert ref_path == str(stored)  # absolute, never relative to the host launch dir

    # The turn expands the ref against the session cwd; the agent gets the mounted path.
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    cwd = server._session_cwd(session)
    ctx = preprocess_context_references(
        f"summarise @file:{server._format_ref_value(ref_path)}", cwd=cwd, allowed_root=cwd,
        context_length=200_000)
    assert not ctx.warnings, ctx.warnings
    assert f"available on disk at `/root/.hermes/attachments/{stored.name}`" in ctx.message
