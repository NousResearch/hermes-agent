"""Tests for the projects.* JSON-RPC methods on the tui_gateway server."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
import tui_gateway.server as server


def _call(method, params=None):
    handler = server._methods[method]
    resp = handler(1, params or {})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


def test_methods_registered():
    for m in (
        "projects.list",
        "projects.create",
        "projects.get",
        "projects.update",
        "projects.add_folder",
        "projects.remove_folder",
        "projects.set_primary",
        "projects.archive",
        "projects.set_active",
        "projects.for_cwd",
    ):
        assert m in server._methods


def test_for_cwd_is_a_long_handler():
    # git-probe handler must run off the dispatch thread.
    assert "projects.for_cwd" in server._LONG_HANDLERS


def test_repo_root_cache_does_not_freeze_a_not_yet_repo(monkeypatch):
    # We `git init` a new project's folder on first worktree; the cache must not
    # have frozen the pre-init "" result, or the main lane mislabels by basename.
    # Negative results are TTL-cached; TTL=0 here makes them expire immediately so
    # this verifies the "never permanently frozen" contract directly.
    from tui_gateway import git_probe

    monkeypatch.setattr(git_probe, "_NEG_TTL", 0)
    cwd = "/tmp/baby pics"
    git_probe.invalidate()
    state = {"root": ""}  # flips once the folder becomes a repo
    monkeypatch.setattr(git_probe, "run_git", lambda c, *a: state["root"] if c == cwd else "")

    assert git_probe.repo_root(cwd) == ""  # pre-init: not a repo (expires at once)

    state["root"] = cwd  # `git init` happened
    assert git_probe.repo_root(cwd) == cwd  # re-probed, not frozen
    assert git_probe.repo_root(cwd) == cwd  # now cached


def test_negative_results_are_ttl_cached_then_re_probed(monkeypatch):
    # A non-repo cwd is re-derived on every session in a project-tree build, so a
    # "not a repo" answer must be cached briefly to avoid re-spawning git dozens
    # of times — but only until the TTL elapses, so a folder that later becomes a
    # repo is still picked up.
    from tui_gateway import git_probe

    git_probe.invalidate()
    calls = {"n": 0}

    def probe(_cwd, *_a):
        calls["n"] += 1
        return ""  # never a repo

    monkeypatch.setattr(git_probe, "run_git", probe)
    monkeypatch.setattr(git_probe, "_NEG_TTL", 1000)  # effectively no expiry here

    cwd = "/not/a/repo"
    assert git_probe.repo_root(cwd) == ""
    for _ in range(10):
        assert git_probe.repo_root(cwd) == ""
    assert calls["n"] == 1  # cached: probed once, not 11 times

    # Once the TTL lapses, the next lookup re-probes (a `git init` may have run).
    monkeypatch.setattr(git_probe, "_NEG_TTL", 0)
    git_probe._cache._neg[cwd] = 0.0  # force-expire the cached negative
    assert git_probe.repo_root(cwd) == ""
    assert calls["n"] == 2


def test_repo_root_cache_is_single_flight(monkeypatch):
    # Concurrent identical probes share one git invocation (gateway long handlers
    # run on worker threads).
    import threading

    from tui_gateway import git_probe

    git_probe.invalidate()
    calls = {"n": 0}
    started = threading.Event()

    def slow(_cwd, *_a):
        calls["n"] += 1
        started.set()
        time = __import__("time")
        time.sleep(0.05)
        return "/repo"

    monkeypatch.setattr(git_probe, "run_git", slow)
    out: list[str] = []
    threads = [threading.Thread(target=lambda: out.append(git_probe.repo_root("/repo/x"))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert out == ["/repo"] * 6
    assert calls["n"] == 1


def test_warm_roots_probes_in_parallel_and_fills_the_cache(monkeypatch):
    # Cold first paint must not serialize one git subprocess per cwd.
    import threading
    import time

    from tui_gateway import git_probe

    git_probe.invalidate()
    lock = threading.Lock()
    live = {"now": 0, "peak": 0, "calls": 0}

    def slow(cwd, *_a):
        with lock:
            live["now"] += 1
            live["calls"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.02)
        with lock:
            live["now"] -= 1
        return cwd  # show-toplevel → cwd is its own root

    monkeypatch.setattr(git_probe, "run_git", slow)
    cwds = [f"/repo{i}" for i in range(8)]
    git_probe.warm_roots(cwds, max_workers=8)

    assert live["peak"] > 1  # ran concurrently, not serialized
    # Cache is warm: resolving again triggers no further probes.
    before = live["calls"]
    assert git_probe.repo_root("/repo0") == "/repo0"
    assert live["calls"] == before


def test_create_list_roundtrip(tmp_path):
    created = _call("projects.create", {"name": "Demo", "folders": [str(tmp_path)], "use": True})
    assert created["project"]["slug"] == "demo"

    listing = _call("projects.list")
    assert [p["slug"] for p in listing["projects"]] == ["demo"]
    assert listing["active_id"] == created["project"]["id"]


def test_add_folder_and_for_cwd(tmp_path):
    folder = tmp_path / "repo"
    folder.mkdir()
    pid = _call("projects.create", {"name": "Repo", "folders": [str(folder)]})["project"]["id"]

    nested = folder / "src"
    nested.mkdir()
    resolved = _call("projects.for_cwd", {"cwd": str(nested)})
    assert resolved["project"]["id"] == pid
    # branch key is present (empty string when not a git repo).
    assert "branch" in resolved


def test_update_and_archive(tmp_path):
    pid = _call("projects.create", {"name": "Orig", "folders": [str(tmp_path)]})["project"]["id"]

    updated = _call("projects.update", {"id": pid, "name": "Renamed"})
    assert updated["project"]["name"] == "Renamed"

    payload = _call("projects.archive", {"id": pid})
    assert all(p["id"] != pid or p["archived"] for p in payload["projects"])


def test_get_unknown_returns_error():
    resp = server._methods["projects.get"](1, {"id": "nope"})
    assert "error" in resp


def test_delete_removes_project(tmp_path):
    pid = _call("projects.create", {"name": "Doomed", "folders": [str(tmp_path)]})["project"]["id"]
    payload = _call("projects.delete", {"id": pid})

    assert all(p["id"] != pid for p in payload["projects"])
    assert "projects.delete" in server._methods


def test_discover_repos_is_registered_long_handler():
    assert "projects.discover_repos" in server._methods
    assert "projects.discover_repos" in server._LONG_HANDLERS
    assert "projects.record_repos" in server._methods
    assert "projects.record_repos" in server._LONG_HANDLERS


def test_record_repos_persists_and_shows_zero_session_repo(tmp_path):
    repo = tmp_path / "fresh-repo"
    repo.mkdir()

    # Repo-first: a scanned repo with no hermes sessions still surfaces.
    _call("projects.record_repos", {"repos": [{"root": str(repo), "label": "fresh-repo"}]})

    by_label = {r["label"]: r for r in _call("projects.discover_repos")["repos"]}
    assert "fresh-repo" in by_label
    assert by_label["fresh-repo"]["sessions"] == 0


def test_discover_repos_from_full_history(tmp_path):
    repo = tmp_path / "myrepo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    plain = tmp_path / "plain"
    plain.mkdir()

    db = server._get_db()
    db.create_session("s1", "cli", cwd=str(repo))
    db.create_session("s2", "cli", cwd=str(repo / "src"))
    db.create_session("s3", "cli", cwd=str(plain))  # not a git repo → excluded

    repos = _call("projects.discover_repos")["repos"]
    by_label = {r["label"]: r for r in repos}

    assert "myrepo" in by_label
    assert by_label["myrepo"]["sessions"] == 2  # both repo cwds aggregate
    assert "plain" not in by_label  # non-git dir never promoted

    # The probe is persisted back onto the session rows (membership at the source).
    assert os.path.realpath(db.get_session("s1")["git_repo_root"]) == os.path.realpath(str(repo))


def _profile_home(tmp_path: Path, name: str) -> Path:
    home = tmp_path / name
    home.mkdir(parents=True, exist_ok=True)
    return home


def _create_project(home: Path, name: str, folder: Path, *, use: bool = False) -> dict:
    token = set_hermes_home_override(home)
    server._db = None
    server._db_error = None
    try:
        return _call("projects.create", {"name": name, "folders": [str(folder)], "use": use})["project"]
    finally:
        server._db = None
        server._db_error = None
        reset_hermes_home_override(token)


def _create_session(home: Path, session_id: str, cwd: Path) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, "cli", cwd=str(cwd))
        db.append_message(session_id, "user", f"hello from {session_id}")
    finally:
        db.close()


def test_projects_rpc_profile_scope_isolates_projects_and_sessions(monkeypatch, tmp_path):
    launch_home = _profile_home(tmp_path, "launch")
    coder_home = _profile_home(tmp_path, "coder")
    launch_repo = tmp_path / "launch-repo"
    coder_repo = tmp_path / "coder-repo"
    launch_repo.mkdir()
    coder_repo.mkdir()

    launch_project = _create_project(launch_home, "Launch", launch_repo, use=True)
    coder_project = _create_project(coder_home, "Coder", coder_repo, use=True)
    _create_session(launch_home, "launch-session", launch_repo)
    _create_session(coder_home, "coder-session", coder_repo)

    token = set_hermes_home_override(launch_home)
    server._db = None
    server._db_error = None
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: coder_home if name == "coder" else launch_home if name == "default" else tmp_path / name,
    )
    try:
        from hermes_state import SessionDB

        server._db = SessionDB(db_path=launch_home / "state.db")
        launch_listing = _call("projects.list")
        coder_listing = _call("projects.list", {"profile": "coder"})
        launch_tree = _call("projects.tree")
        coder_tree = _call("projects.tree", {"profile": "coder"})
        coder_sessions = _call("projects.project_sessions", {"profile": "coder", "project_id": coder_project["id"]})
    finally:
        if server._db is not None:
            server._db.close()
        server._db = None
        server._db_error = None
        reset_hermes_home_override(token)

    assert [p["name"] for p in launch_listing["projects"]] == ["Launch"]
    assert [p["name"] for p in coder_listing["projects"]] == ["Coder"]
    assert launch_listing["active_id"] == launch_project["id"]
    assert coder_listing["active_id"] == coder_project["id"]
    assert [p["label"] for p in launch_tree["projects"]] == ["Launch"]
    assert [p["label"] for p in coder_tree["projects"]] == ["Coder"]
    assert launch_tree["projects"][0]["sessionCount"] == 1
    assert coder_tree["projects"][0]["sessionCount"] == 1
    assert coder_sessions["project"]["id"] == coder_project["id"]
    assert coder_sessions["project"]["sessionCount"] == 1
    assert coder_sessions["project"]["repos"][0]["groups"][0]["sessions"][0]["id"] == "coder-session"


def test_projects_record_repos_writes_to_target_profile(monkeypatch, tmp_path):
    launch_home = _profile_home(tmp_path, "launch")
    coder_home = _profile_home(tmp_path, "coder")
    launch_repo = tmp_path / "launch-scan"
    coder_repo = tmp_path / "coder-scan"
    launch_repo.mkdir()
    coder_repo.mkdir()

    token = set_hermes_home_override(launch_home)
    server._db = None
    server._db_error = None
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: coder_home if name == "coder" else launch_home if name == "default" else tmp_path / name,
    )
    try:
        from hermes_state import SessionDB

        server._db = SessionDB(db_path=launch_home / "state.db")
        _call("projects.record_repos", {"repos": [{"root": str(launch_repo), "label": "launch"}]})
        _call("projects.record_repos", {"profile": "coder", "repos": [{"root": str(coder_repo), "label": "coder"}]})

        launch_repos = _call("projects.discover_repos")["repos"]
        coder_repos = _call("projects.discover_repos", {"profile": "coder"})["repos"]
    finally:
        if server._db is not None:
            server._db.close()
        server._db = None
        server._db_error = None
        reset_hermes_home_override(token)

    assert [repo["label"] for repo in launch_repos] == ["launch"]
    assert [repo["label"] for repo in coder_repos] == ["coder"]
