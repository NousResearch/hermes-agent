"""Tests for the projects.* JSON-RPC methods on the tui_gateway server."""

from __future__ import annotations

import contextlib
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


@pytest.fixture(autouse=True)
def _fast_git_probe(monkeypatch):
    """Replace real git subprocess probes with a cheap .git-directory check.

    The record/discover RPC paths probe every distinct session cwd in the DB
    with a real ``git`` subprocess; on a warm session DB that made single
    tests take 10-80s. Behavior under test (policy gating, cache merging,
    ranking) only needs root resolution, not real git.
    """
    from tui_gateway import git_probe

    git_probe.invalidate()

    def _fake_run_git(cwd, *_a):
        d = str(cwd)
        while d and d not in ("/", os.path.dirname(d)):
            if os.path.isdir(os.path.join(d, ".git")):
                return d
            d = os.path.dirname(d)
        return ""

    monkeypatch.setattr(git_probe, "run_git", _fake_run_git)
    yield
    git_probe.invalidate()


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


def test_missing_directory_costs_no_subprocess(monkeypatch):
    # Deleted worktrees dominate a long session history's cwds, and `git -C` on
    # one can only fail — so it must never reach the fork.
    from tui_gateway import git_probe

    def boom(*_a, **_kw):
        raise AssertionError("spawned git for a directory that does not exist")

    monkeypatch.setattr(git_probe, "bounded_git_probe", boom)

    assert git_probe.run_git("/gone/worktree", "rev-parse", "--show-toplevel") == ""


def test_non_repo_cwd_is_not_probed_for_a_common_dir(monkeypatch, tmp_path):
    # `warm_roots` only reaches `common_repo_root` for cwds that ARE repos, so a
    # common-dir probe here is one the warm can't absorb: it runs serially on
    # the discovery pass, once per non-repo cwd.
    from tui_gateway import git_probe

    git_probe.invalidate()
    asked = []

    def probe(cwd, *args):
        asked.append(args[-1])
        return ""  # not a repo, whatever we ask

    monkeypatch.setattr(git_probe, "run_git", probe)

    assert git_probe.common_repo_root(str(tmp_path)) == ""
    assert asked == ["--show-toplevel"]


def test_tree_build_warms_every_path_it_will_resolve(monkeypatch, tmp_path):
    # build_tree resolves declared project folders and discovered repo roots as
    # well as session cwds. Anything left out of the warm is probed one
    # directory at a time while the sidebar shows a skeleton.
    from tui_gateway import git_probe

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    _call("projects.create", {"name": "Repo", "folders": [str(repo)]})

    warmed: list[str] = []
    real_warm = git_probe.warm_roots

    def recording_warm(cwds, **kw):
        paths = list(cwds)
        warmed.extend(paths)
        return real_warm(paths, **kw)

    monkeypatch.setattr(git_probe, "warm_roots", recording_warm)

    server._build_project_tree(
        server._get_db(), preview_limit=3, hydrate=False, session_limit=5, include_discovered=True
    )

    assert str(repo) in warmed


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


def test_project_info_for_cwd_returns_status_payload(tmp_path):
    # The status-surface resolver returns the owning project's identity for a
    # nested cwd — the shape the TUI status label + /status read.
    folder = tmp_path / "repo"
    folder.mkdir()
    created = _call("projects.create", {"name": "Repo", "folders": [str(folder)]})["project"]

    nested = folder / "src"
    nested.mkdir()

    assert server._project_info_for_cwd(str(nested)) == {
        "id": created["id"],
        "slug": "repo",
        "name": "Repo",
        "primary_path": str(folder),
    }


def test_session_info_carries_project_for_owned_cwd(tmp_path):
    # session.info threads the resolved project through so the desktop/TUI can
    # name the workspace without a second round-trip.
    folder = tmp_path / "proj"
    folder.mkdir()
    _call("projects.create", {"name": "Proj", "folders": [str(folder)]})

    info = server._session_info(None, {"cwd": str(folder), "session_key": "s1"})
    assert info["project"] == {
        "id": info["project"]["id"],
        "slug": "proj",
        "name": "Proj",
        "primary_path": str(folder),
    }
    assert info["project"]["name"] == "Proj"


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


def test_scan_time_is_not_treated_as_session_activity(tmp_path):
    """A scanned repo with no sessions must not rank as recently active.

    ``discovered_repos.last_seen`` records when the disk scan last saw the
    directory. Folding it into ``last_active`` stamped every scanned checkout
    with the scan time — i.e. "just now" — so repos the user has never opened
    in Hermes outranked the ones they actually work in.
    """
    worked_in = tmp_path / "worked-in"
    worked_in.mkdir()
    subprocess.run(["git", "init"], cwd=worked_in, check=True, capture_output=True)
    server._get_db().create_session("worked-in-session", "cli", cwd=str(worked_in))

    never_opened = tmp_path / "never-opened"
    never_opened.mkdir()

    _call(
        "projects.record_repos",
        {"repos": [{"root": str(never_opened)}, {"root": str(worked_in)}]},
    )

    by_root = {r["root"]: r for r in _call("projects.discover_repos")["repos"]}
    idle = by_root[str(never_opened)]
    active = by_root[str(worked_in)]

    assert idle["sessions"] == 0
    # A repo with no sessions has no activity to report...
    assert idle["last_active"] == 0
    # ...so the repo the user actually worked in sorts ahead of it.
    assert active["last_active"] > idle["last_active"]


def test_remote_scan_failure_merges_instead_of_replacing_cache(tmp_path, monkeypatch):
    """A backend scan that can't fully walk its roots must NOT wipe the cache.

    `projects.discover_repos` with `scan:true` asks the remote host to scan its
    own discovery roots. When one root fails to walk, the scan result is not the
    authoritative full universe — the previously cached repos must survive so a
    failed remote refresh can't blank the sidebar back to the silent, empty
    state of #81723 (regression for MEDIUM: `replace=True` was wiping on every
    call regardless of success).
    """
    from hermes_cli import projects_db as pdb
    import tui_gateway.server as server

    def _git_repo(path):
        repo = path
        repo.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        return str(repo)

    # A cached repo the partial scan never visits, parked OUTSIDE the scan roots.
    seed = _git_repo(tmp_path / "elsewhere" / "seed-repo")
    with pdb.connect_closing() as conn:
        pdb.record_discovered_repos(conn, [(seed, "seed-repo")])
        seeded = [r["root"] for r in pdb.list_discovered_repos(conn)]
    assert seed in seeded

    good = _git_repo(tmp_path / "good-repo")

    # Force one root to fail to walk: the scan becomes non-authoritative, so it
    # must merge into the cache, never wipe it.
    real_walk = os.walk
    bad_root = str(tmp_path / "unwalkable")

    def _flaky_walk(top, *a, **k):
        if top == bad_root:
            raise OSError("boom")
        yield from real_walk(top, *a, **k)

    monkeypatch.setattr(server.os, "walk", _flaky_walk)

    policy = {"enabled": True, "roots": [good, bad_root], "exclude_paths": []}

    with pdb.connect_closing() as conn:
        authoritative = server._scan_discovered_repos_remote(conn, policy)
        joined = [r["root"] for r in pdb.list_discovered_repos(conn)]

    # The scan found the good repo and merged it, but the failed root means the
    # result is not authoritative, so it must NOT have replaced the cache.
    assert not authoritative
    assert good in joined
    # The seeded repo that the partial scan never saw is still cached.
    assert seed in joined


def test_remote_scan_missing_root_does_not_wipe_cache(tmp_path):
    """A configured discovery root missing on disk must NOT wipe the cache.

    ``os.walk`` on a non-existent root silently yields nothing instead of
    raising, so a temporarily unavailable root (unmounted volume, moved path)
    would otherwise make the scan look like a genuinely empty authoritative
    set and DELETE-replace every cached repo that lived under it. The missing
    root must contribute nothing, and the scan must merge — never wipe.
    """
    from hermes_cli import projects_db as pdb
    import tui_gateway.server as server

    def _git_repo(path):
        repo = path
        repo.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        return str(repo)

    # A cached repo the scan never visits, parked OUTSIDE the scan roots.
    seed = _git_repo(tmp_path / "elsewhere" / "seed-repo")
    with pdb.connect_closing() as conn:
        pdb.record_discovered_repos(conn, [(seed, "seed-repo")])
        seeded = [r["root"] for r in pdb.list_discovered_repos(conn)]
    assert seed in seeded

    good = _git_repo(tmp_path / "good-repo")

    # This root is configured but does NOT exist on disk. os.walk on it yields
    # nothing silently — without the guard the scan would stay authoritative
    # and wipe the cache.
    missing_root = str(tmp_path / "missing-root")

    policy = {"enabled": True, "roots": [good, missing_root], "exclude_paths": []}

    with pdb.connect_closing() as conn:
        authoritative = server._scan_discovered_repos_remote(conn, policy)
        joined = [r["root"] for r in pdb.list_discovered_repos(conn)]

    # The missing root is not authoritative, so the scan must merge, not wipe.
    assert not authoritative
    assert good in joined
    # The seeded repo the missing root would have wiped is still cached.
    assert seed in joined


def test_remote_scan_full_authoritative_replaces_cache(tmp_path):
    """Only a fully-walked scan may replace the stale cache."""
    from hermes_cli import projects_db as pdb
    import tui_gateway.server as server

    def _git_repo(path):
        repo = path
        repo.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        return str(repo)

    # Park the stale repo OUTSIDE the scan root so the authoritative scan no
    # longer sees it, and the fresh repo inside the root it walks.
    stale = _git_repo(tmp_path / "outside" / "stale-repo")
    scandir = tmp_path / "scandir"
    scandir.mkdir()
    fresh = _git_repo(scandir / "fresh-repo")

    with pdb.connect_closing() as conn:
        pdb.record_discovered_repos(conn, [(stale, "stale-repo")])

    policy = {"enabled": True, "roots": [str(scandir)], "exclude_paths": []}

    with pdb.connect_closing() as conn:
        authoritative = server._scan_discovered_repos_remote(conn, policy)
        joined = [r["root"] for r in pdb.list_discovered_repos(conn)]

    assert authoritative
    assert fresh in joined
    # A full, authoritative scan replaced the stale cache: the old repo the
    # scan no longer saw is gone from the authoritative set.
    assert stale not in joined


def test_terminal_session_persists_its_launch_cwd():
    """A terminal session's cwd IS its workspace, so the row must record it.

    The user cd'd into that directory before running hermes. Dropping it left
    the row with no cwd and no git_repo_root, so the sidebar could never place
    the session under its project.
    """
    for source in ("tui", "cli"):
        assert server._persisted_session_cwd(
            {"source": source, "cwd": "/somewhere/a-repo"}
        ) == "/somewhere/a-repo"


def test_desktop_launch_cwd_is_not_persisted_as_a_workspace():
    # The desktop launches from wherever the bundle was opened, so an unpicked
    # cwd is an artifact — those chats belong under "No workspace".
    assert server._persisted_session_cwd({"source": "desktop", "cwd": "/opt/whatever"}) is None

    # An explicit pick is always honored, desktop included.
    assert server._persisted_session_cwd(
        {"source": "desktop", "cwd": "/picked/repo", "explicit_cwd": True}
    ) == "/picked/repo"


def test_desktop_launch_cwd_is_marked_as_context_artifact():
    assert server._context_cwd_is_launch_artifact(
        {"source": "desktop", "cwd": "/opt/hermes"}
    ) is True


def test_explicit_desktop_and_terminal_cwds_are_context_workspaces():
    assert server._context_cwd_is_launch_artifact(
        {"source": "desktop", "cwd": "/picked/repo", "explicit_cwd": True}
    ) is False
    assert server._context_cwd_is_launch_artifact(
        {"source": "tui", "cwd": "/opt/hermes"}
    ) is False


def test_home_container_dirs_are_never_a_workspace(tmp_path):
    """`/home` and `/Users` hold homes; they are not workspaces themselves.

    A session whose cwd is one of them used to be promoted to its own auto
    project, so the sidebar showed a second row labelled "home" sitting right
    next to the synthetic Home bucket. Both POSIX spellings are excluded on
    every host: either can reach a local row (macOS ships an empty `/home`
    stub) or arrive from a container/remote shell.
    """
    home = os.path.realpath(os.path.expanduser("~"))

    for path in (os.sep, home, os.path.dirname(home), "/home", "/Users"):
        assert server._is_session_cwd_junk(path), path
        assert server._is_repo_junk(path), path

    # An ordinary directory is still a workspace.
    workspace = tmp_path / "a-repo"
    workspace.mkdir()
    assert not server._is_session_cwd_junk(str(workspace))
    assert not server._is_repo_junk(str(workspace))


def test_disabled_discovery_clears_cache_and_rejects_new_scan(monkeypatch, tmp_path):
    repo = tmp_path / "cached-repo"
    repo.mkdir()
    session_repo = tmp_path / "session-repo"
    session_repo.mkdir()
    subprocess.run(
        ["git", "init"], cwd=session_repo, check=True, capture_output=True
    )
    server._get_db().create_session("session-repo", "cli", cwd=str(session_repo))
    _call("projects.record_repos", {"repos": [{"root": str(repo)}]})

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "desktop": {
                "repo_scan_enabled": False,
                "repo_scan_roots": [],
                "repo_scan_exclude_paths": [],
            }
        },
    )
    result = _call(
        "projects.record_repos",
        {
            "repos": [{"root": str(repo)}],
            "discovery_policy": {
                "enabled": False,
                "roots": [],
                "exclude_paths": [],
            },
        },
    )

    assert result["accepted"] is False
    assert all(item["root"] != str(repo) for item in result["repos"])
    assert any(item["root"] == str(session_repo) for item in result["repos"])


def test_nondefault_policy_rejects_stale_or_legacy_results(monkeypatch, tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    policy = {
        "enabled": True,
        "roots": [str(root)],
        "exclude_paths": [],
    }
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "desktop": {
                "repo_scan_enabled": True,
                "repo_scan_roots": [str(root)],
                "repo_scan_exclude_paths": [],
            }
        },
    )

    legacy = _call("projects.record_repos", {"repos": [{"root": str(root)}]})
    stale = _call(
        "projects.record_repos",
        {
            "repos": [{"root": str(root)}],
            "discovery_policy": {**policy, "roots": [str(tmp_path / "other")]},
        },
    )
    accepted = _call(
        "projects.record_repos",
        {"repos": [{"root": str(root)}], "discovery_policy": policy},
    )

    assert legacy["accepted"] is False
    assert stale["accepted"] is False
    assert accepted["accepted"] is True
    assert any(item["root"] == str(root) for item in accepted["repos"])


def _profile_dir(tmp_path: Path, name: str) -> Path:
    home = tmp_path / "homes" / name
    home.mkdir(parents=True, exist_ok=True)
    return home


def _bind_profiles(monkeypatch, tmp_path: Path, homes: dict[str, Path]) -> None:
    """Resolve profile names to this test's throwaway homes.

    Unmapped names resolve to a path that does not exist, which is how the
    gateway detects "not a real profile on this host" and stays on launch.
    """
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: homes.get(name, tmp_path / "homes" / "missing" / name),
    )


def _create_project(home: Path, name: str, folder: Path, *, use: bool = False) -> dict:
    """Create a project in ``home``'s projects.db via the real RPC."""
    token = set_hermes_home_override(home)
    try:
        return _call(
            "projects.create", {"name": name, "folders": [str(folder)], "use": use}
        )["project"]
    finally:
        reset_hermes_home_override(token)


def _create_session(home: Path, session_id: str, cwd: Path) -> None:
    """Seed one message-bearing session in ``home``'s state.db."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, "cli", cwd=str(cwd))
        db.append_message(session_id, "user", f"hello from {session_id}")
    finally:
        db.close()


@contextlib.contextmanager
def _serving_launch_profile(launch_home: Path):
    """Run the handlers as a backend launched under ``launch_home``."""
    from hermes_state import SessionDB

    token = set_hermes_home_override(launch_home)
    prev_db, prev_error = server._db, server._db_error
    server._db = SessionDB(db_path=launch_home / "state.db")
    server._db_error = None
    try:
        yield
    finally:
        server._db.close()
        server._db, server._db_error = prev_db, prev_error
        reset_hermes_home_override(token)


def _cached_repo_labels(home: Path) -> list[str]:
    """Labels in ``home``'s discovered-repo cache, read straight off disk."""
    from hermes_cli import projects_db as pdb

    with pdb.connect_closing(home / "projects.db") as conn:
        return sorted(str(entry.get("label") or "") for entry in pdb.list_discovered_repos(conn))


def test_projects_reads_are_scoped_to_the_requested_profile(monkeypatch, tmp_path):
    """A ``profile`` param reads that profile's projects.db AND its state.db."""
    launch_home = _profile_dir(tmp_path, "launch")
    coder_home = _profile_dir(tmp_path, "coder")
    launch_repo = tmp_path / "repos" / "launch-repo"
    coder_repo = tmp_path / "repos" / "coder-repo"
    launch_repo.mkdir(parents=True)
    coder_repo.mkdir(parents=True)
    _bind_profiles(monkeypatch, tmp_path, {"default": launch_home, "coder": coder_home})

    launch_project = _create_project(launch_home, "Launch", launch_repo, use=True)
    coder_project = _create_project(coder_home, "Coder", coder_repo, use=True)
    _create_session(launch_home, "launch-session", launch_repo)
    _create_session(coder_home, "coder-session", coder_repo)

    with _serving_launch_profile(launch_home):
        launch_listing = _call("projects.list")
        coder_listing = _call("projects.list", {"profile": "coder"})
        launch_tree = _call("projects.tree")
        coder_tree = _call("projects.tree", {"profile": "coder"})
        coder_sessions = _call(
            "projects.project_sessions",
            {"profile": "coder", "project_id": coder_project["id"]},
        )
        # The override must not leak: the very next unscoped read is launch again.
        launch_again = _call("projects.list")

    assert [p["name"] for p in launch_listing["projects"]] == ["Launch"]
    assert [p["name"] for p in coder_listing["projects"]] == ["Coder"]
    assert launch_listing["active_id"] == launch_project["id"]
    assert coder_listing["active_id"] == coder_project["id"]
    assert launch_again == launch_listing

    assert [p["label"] for p in launch_tree["projects"]] == ["Launch"]
    assert [p["label"] for p in coder_tree["projects"]] == ["Coder"]
    # Session counts prove the SESSION db was swapped too, not just projects.db.
    assert launch_tree["projects"][0]["sessionCount"] == 1
    assert coder_tree["projects"][0]["sessionCount"] == 1
    assert launch_tree["scoped_session_ids"] == ["launch-session"]
    assert coder_tree["scoped_session_ids"] == ["coder-session"]

    assert coder_sessions["project"]["id"] == coder_project["id"]
    assert coder_sessions["project"]["sessionCount"] == 1
    lane = coder_sessions["project"]["repos"][0]["groups"][0]
    assert [s["id"] for s in lane["sessions"]] == ["coder-session"]


def test_projects_tree_is_scoped_to_the_requested_profile(monkeypatch, tmp_path):
    """``projects.tree`` on its own reads the requested profile's stores."""
    launch_home = _profile_dir(tmp_path, "launch")
    coder_home = _profile_dir(tmp_path, "coder")
    launch_repo = tmp_path / "repos" / "tree-launch"
    coder_repo = tmp_path / "repos" / "tree-coder"
    launch_repo.mkdir(parents=True)
    coder_repo.mkdir(parents=True)
    _bind_profiles(monkeypatch, tmp_path, {"default": launch_home, "coder": coder_home})

    _create_project(launch_home, "Launch", launch_repo, use=True)
    _create_project(coder_home, "Coder", coder_repo, use=True)
    _create_session(launch_home, "tree-launch-session", launch_repo)
    _create_session(coder_home, "tree-coder-session", coder_repo)

    with _serving_launch_profile(launch_home):
        coder_tree = _call("projects.tree", {"profile": "coder"})
        launch_tree = _call("projects.tree")

    assert [p["label"] for p in coder_tree["projects"]] == ["Coder"]
    assert coder_tree["scoped_session_ids"] == ["tree-coder-session"]
    assert [p["label"] for p in launch_tree["projects"]] == ["Launch"]
    assert launch_tree["scoped_session_ids"] == ["tree-launch-session"]


def test_project_sessions_is_scoped_to_the_requested_profile(monkeypatch, tmp_path):
    """``projects.project_sessions`` on its own hydrates from the requested profile."""
    launch_home = _profile_dir(tmp_path, "launch")
    coder_home = _profile_dir(tmp_path, "coder")
    launch_repo = tmp_path / "repos" / "drill-launch"
    coder_repo = tmp_path / "repos" / "drill-coder"
    launch_repo.mkdir(parents=True)
    coder_repo.mkdir(parents=True)
    _bind_profiles(monkeypatch, tmp_path, {"default": launch_home, "coder": coder_home})

    launch_project = _create_project(launch_home, "Launch", launch_repo, use=True)
    coder_project = _create_project(coder_home, "Coder", coder_repo, use=True)
    _create_session(launch_home, "drill-launch-session", launch_repo)
    _create_session(coder_home, "drill-coder-session", coder_repo)

    with _serving_launch_profile(launch_home):
        coder_drill = _call(
            "projects.project_sessions",
            {"profile": "coder", "project_id": coder_project["id"]},
        )
        launch_drill = _call(
            "projects.project_sessions", {"project_id": launch_project["id"]}
        )

    assert coder_drill["project"] is not None
    assert coder_drill["project"]["id"] == coder_project["id"]
    coder_lane = coder_drill["project"]["repos"][0]["groups"][0]
    assert [s["id"] for s in coder_lane["sessions"]] == ["drill-coder-session"]

    assert launch_drill["project"]["id"] == launch_project["id"]
    launch_lane = launch_drill["project"]["repos"][0]["groups"][0]
    assert [s["id"] for s in launch_lane["sessions"]] == ["drill-launch-session"]


def test_record_repos_writes_to_the_requested_profiles_projects_db(monkeypatch, tmp_path):
    """The scan cache is per-profile: a scoped write must not land on launch."""
    launch_home = _profile_dir(tmp_path, "launch")
    coder_home = _profile_dir(tmp_path, "coder")
    launch_repo = tmp_path / "repos" / "launch-scan"
    coder_repo = tmp_path / "repos" / "coder-scan"
    launch_repo.mkdir(parents=True)
    coder_repo.mkdir(parents=True)
    _bind_profiles(monkeypatch, tmp_path, {"default": launch_home, "coder": coder_home})

    with _serving_launch_profile(launch_home):
        _call("projects.record_repos", {"repos": [{"root": str(launch_repo), "label": "launch"}]})
        _call(
            "projects.record_repos",
            {"profile": "coder", "repos": [{"root": str(coder_repo), "label": "coder"}]},
        )

        launch_repos = _call("projects.discover_repos")["repos"]
        coder_repos = _call("projects.discover_repos", {"profile": "coder"})["repos"]

    assert [repo["label"] for repo in launch_repos] == ["launch"]
    assert [repo["label"] for repo in coder_repos] == ["coder"]
    assert _cached_repo_labels(launch_home) == ["launch"]
    assert _cached_repo_labels(coder_home) == ["coder"]


def test_projects_without_a_profile_stay_on_the_launch_home(monkeypatch, tmp_path):
    """Omitted/blank/unknown profile is a no-op — the pre-scoping behavior."""
    launch_home = _profile_dir(tmp_path, "launch")
    coder_home = _profile_dir(tmp_path, "coder")
    repo = tmp_path / "repos" / "launch-only"
    repo.mkdir(parents=True)
    _bind_profiles(monkeypatch, tmp_path, {"default": launch_home, "coder": coder_home})

    with _serving_launch_profile(launch_home):
        created = _call(
            "projects.create", {"name": "Launch only", "folders": [str(repo)], "use": True}
        )["project"]
        _call("projects.record_repos", {"repos": [{"root": str(repo), "label": "only"}]})

        omitted = _call("projects.list")
        blank = _call("projects.list", {"profile": ""})
        unknown = _call("projects.list", {"profile": "not-a-profile"})

    assert [p["name"] for p in omitted["projects"]] == ["Launch only"]
    assert blank == omitted
    assert unknown == omitted
    assert omitted["active_id"] == created["id"]

    assert _cached_repo_labels(launch_home) == ["only"]
    assert not (coder_home / "projects.db").exists()
    assert not (Path(os.environ["HERMES_HOME"]) / "projects.db").exists()


