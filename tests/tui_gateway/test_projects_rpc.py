"""Tests for the projects.* JSON-RPC methods on the tui_gateway server."""

from __future__ import annotations

import os
import subprocess

import pytest

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


