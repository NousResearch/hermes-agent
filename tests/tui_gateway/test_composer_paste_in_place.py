"""#153415: a Desktop-managed large paste must be referenced where it lives, not copied.

``writeComposerPaste`` keeps the ``.txt`` under ``<HERMES_HOME>/composer-pastes``
(``apps/desktop/electron/composer-paste.ts``) and the chip is attached through the ordinary
``file.attach`` path. That path lives outside every chat workspace by construction, so
``_stage_session_file_attachment`` copied it into ``attachments/`` under the same generated name —
two identical files per paste, and ``attachments/`` stopped meaning "files the user dropped".

The paste dir is the one staging root that needs no copy: the workspace guard admits it in place
(``agent/context_references.py::_resolve_path``) and ``tools/credential_files.py::_CACHE_DIRS``
mounts it into container/remote backends (#110174), which is exactly what the copy was for.
"""

import os

import pytest

from agent.context_references import preprocess_context_references
from tui_gateway import server


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    """A temp launch HERMES_HOME the gateway and the cache-mount mapper both read."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return home


def _paste_file(home, name="pasted_content_20261111_101500_ab12cd.txt"):
    """The file the desktop writes for a large paste, in the shape it really has."""
    pastes = home / "composer-pastes"
    pastes.mkdir(parents=True, exist_ok=True)
    paste = pastes / name
    paste.write_text("a long pasted body\n", encoding="utf-8")
    return paste


def _attachments(home):
    root = home / "attachments"
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def test_paste_is_referenced_in_place_not_copied_into_attachments(launch_home, tmp_path):
    paste = _paste_file(launch_home)
    project = tmp_path / "project"
    session = {"cwd": str(project), "profile_home": None}

    stored, uploaded = server._stage_session_file_attachment(
        session, raw_path=str(paste), data_url="", name="")

    assert stored == paste.resolve()
    assert uploaded is False
    assert _attachments(launch_home) == []
    assert paste.read_text(encoding="utf-8") == "a long pasted body\n"


def test_profile_session_still_reaches_an_installation_home_paste(launch_home, tmp_path):
    """The desktop writes pastes under the installation home; a profile session's own
    ``profile_home`` is a different tree, and the ref has to keep pointing at the paste."""
    paste = _paste_file(launch_home)
    profile_home = launch_home / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    session = {"cwd": str(tmp_path / "project"), "profile_home": str(profile_home)}

    stored, uploaded = server._stage_session_file_attachment(
        session, raw_path=str(paste), data_url="", name="")

    assert stored == paste.resolve()
    assert uploaded is False
    assert _attachments(launch_home) == []


def test_ordinary_out_of_workspace_file_is_still_staged(launch_home, tmp_path):
    """Nothing changes for a real drop: it is copied into ``attachments/`` as before."""
    dropped = tmp_path / "Downloads" / "report.pdf"
    dropped.parent.mkdir(parents=True)
    dropped.write_bytes(b"%PDF-1.4\x00\x01binary")
    session = {"cwd": str(tmp_path / "project"), "profile_home": None}

    stored, uploaded = server._stage_session_file_attachment(
        session, raw_path=str(dropped), data_url="", name="")

    assert uploaded is True
    assert stored.parent == (launch_home / "attachments").resolve()
    assert stored.read_bytes() == dropped.read_bytes()


def test_container_backend_resolves_the_paste_ref_without_a_copy(launch_home, monkeypatch, tmp_path):
    """The reason the copy existed: the bytes must still reach the agent. They do — the ref
    expands from the ``composer-pastes`` cache mount (#110174), inlined as text."""
    paste = _paste_file(launch_home)
    session = {"cwd": str(tmp_path / "project"), "profile_home": None}
    stored, uploaded = server._stage_session_file_attachment(
        session, raw_path=str(paste), data_url="", name="")
    assert uploaded is False

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    cwd = server._session_cwd(session)
    ctx = preprocess_context_references(
        f"summarise @file:{server._format_ref_value(str(stored))}", cwd=cwd, allowed_root=cwd,
        context_length=200_000)

    assert not ctx.warnings, ctx.warnings
    assert not ctx.blocked
    assert "a long pasted body" in ctx.message


def test_window_mount_covers_composer_pastes(launch_home):
    """Guard the premise of the test above rather than trusting the comment."""
    from tools.credential_files import get_cache_directory_mounts

    mounts = get_cache_directory_mounts()
    assert "/root/.hermes/composer-pastes" in {m["container_path"] for m in mounts}
    assert any(os.path.realpath(m["host_path"]) == os.path.realpath(str(launch_home / "composer-pastes"))
               for m in mounts)
