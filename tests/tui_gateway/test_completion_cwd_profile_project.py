"""#79406: a new session bound to another profile must take its workspace from THAT
profile's own ``projects.db`` (``active_id`` → ``primary_path``), never from the cwd
inherited across a profile switch.

The desktop seeds a new chat's cwd from its app-global workspace, which after a
profile switch still holds the PREVIOUS profile's active project directory. The
profile's ``terminal.cwd`` already wins over that inherited path (#52589), but a
profile that configures no ``terminal.cwd`` fell straight through to the inherited
cwd — so Profile B's new session ran inside Profile A's project and loaded Profile
A's ``AGENTS.md`` as Project Context.
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli import projects_db as pdb
from tui_gateway import server


def _profile_home(tmp_path: Path, name: str, *, active_project: Path | None = None,
                  terminal_cwd: str | None = None) -> Path:
    """A named profile home: config.yaml (+ optional terminal.cwd) and, when asked,
    a projects.db carrying one project pinned as the active pointer."""
    home = tmp_path / name
    home.mkdir(exist_ok=True)
    cfg = "terminal:\n  cwd: %s\n" % terminal_cwd if terminal_cwd else "{}\n"
    (home / "config.yaml").write_text(cfg, encoding="utf-8")

    if active_project is not None:
        with pdb.connect_closing(home / "projects.db") as conn:
            project_id = pdb.create_project(conn, name="Project", primary_path=str(active_project))
            pdb.set_active(conn, project_id)

    return home


def _bind_profiles(monkeypatch, mapping: dict[str, Path | None]) -> None:
    monkeypatch.setattr(server, "_profile_home", lambda name: mapping.get((name or "").strip()))


def test_completion_cwd_uses_target_profile_active_project_over_inherited_cwd(monkeypatch, tmp_path):
    """Profile switch A → B: B's active project wins over the inherited cwd."""
    profile_a_project = tmp_path / "domain" / "subproject"
    profile_a_project.mkdir(parents=True)
    profile_b_project = tmp_path / "workspace"
    profile_b_project.mkdir()
    home_b = _profile_home(tmp_path, "profile-b", active_project=profile_b_project)

    _bind_profiles(monkeypatch, {"b": home_b})

    # The desktop's inherited cwd is still Profile A's project directory.
    assert server._completion_cwd({
        "profile": "b", "cwd": str(profile_a_project), "cwd_explicit": False
    }) == str(profile_b_project)


def test_completion_cwd_active_project_not_used_for_an_explicit_pick(monkeypatch, tmp_path):
    """A deliberate workspace pick (``cwd_explicit``) still owns the session cwd."""
    picked = tmp_path / "picked"
    picked.mkdir()
    profile_b_project = tmp_path / "workspace"
    profile_b_project.mkdir()
    home_b = _profile_home(tmp_path, "profile-b", active_project=profile_b_project)

    _bind_profiles(monkeypatch, {"b": home_b})

    assert server._completion_cwd({
        "profile": "b", "cwd": str(picked), "cwd_explicit": True
    }) == str(picked)


def test_completion_cwd_profile_terminal_cwd_still_beats_active_project(monkeypatch, tmp_path):
    """An explicitly configured ``terminal.cwd`` outranks the active project pointer."""
    configured = tmp_path / "configured"
    configured.mkdir()
    profile_b_project = tmp_path / "workspace"
    profile_b_project.mkdir()
    home_b = _profile_home(tmp_path, "profile-b", active_project=profile_b_project,
                           terminal_cwd=str(configured))

    _bind_profiles(monkeypatch, {"b": home_b})

    assert server._completion_cwd({
        "profile": "b", "cwd": str(tmp_path / "inherited"), "cwd_explicit": False
    }) == str(configured)


def test_completion_cwd_without_active_project_keeps_inherited_workspace(monkeypatch, tmp_path):
    """Empty project store: the inherited workspace is kept, exactly as before
    (#52589 — the profile override only applies when the profile owns one)."""
    inherited = tmp_path / "workspace"
    inherited.mkdir()
    home_b = _profile_home(tmp_path, "profile-b")

    _bind_profiles(monkeypatch, {"b": home_b})

    assert server._completion_cwd({
        "profile": "b", "cwd": str(inherited), "cwd_explicit": False
    }) == str(inherited)


def test_completion_cwd_ignores_active_project_pointing_at_a_missing_dir(monkeypatch, tmp_path):
    """A stale pointer (deleted checkout) must not win over the inherited workspace."""
    inherited = tmp_path / "workspace"
    inherited.mkdir()
    ghost = tmp_path / "removed-project"
    home_b = _profile_home(tmp_path, "profile-b", active_project=ghost)

    _bind_profiles(monkeypatch, {"b": home_b})

    assert server._completion_cwd({
        "profile": "b", "cwd": str(inherited), "cwd_explicit": False
    }) == str(inherited)


def test_completion_cwd_leaves_launch_profile_resolution_untouched(monkeypatch, tmp_path):
    """No ``profile`` param → the launch profile's own resolution is unchanged."""
    inherited = tmp_path / "workspace"
    inherited.mkdir()

    monkeypatch.setattr(server, "_profile_home", lambda _name: None)

    assert server._completion_cwd({"cwd": str(inherited), "cwd_explicit": False}) == str(inherited)
