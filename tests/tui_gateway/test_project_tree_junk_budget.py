"""Per-build junk-classification work budgets through the rebound server facade."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import project_tree, server


def _install_inputs(monkeypatch, sessions):
    # Keep gathering/probing synthetic; exercise the real facade and authoritative builder.
    monkeypatch.setattr(server, "_project_tree_inputs", lambda *a, **k: (sessions, [], [], "active"))
    monkeypatch.setattr(server, "git_probe", SimpleNamespace(
        warm_roots=lambda paths: None, resolve=lambda path: None))


def _build():
    tree, active_id = server._build_project_tree(
        None, preview_limit=2, hydrate=True, session_limit=1000, include_discovered=True)
    assert active_id == "active"
    return tree


def _rows(path, prefix, count, *, root=""):
    return [{"id": f"{prefix}-{i}", "cwd": path, "git_repo_root": root,
             "git_branch": "main", "last_active": i, "source": "cli"}
            for i in range(count)]


def test_repeated_paths_classified_once_per_build_with_fresh_next_build(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = tmp_path / "profile"
    notes = profile / "notes"
    repo = tmp_path / "repo"
    notes.mkdir(parents=True)
    repo.mkdir()
    profile, notes, repo = str(profile), str(notes), str(repo)
    sessions = (_rows(repo, "repo", 24, root=repo)
                + _rows(notes, "hidden-repo", 24, root=notes)
                + _rows(notes, "notes", 24) + _rows(profile, "home", 24))
    _install_inputs(monkeypatch, sessions)
    calls = Counter()
    repo_junk, cwd_junk = server._is_repo_junk, server._is_session_cwd_junk

    def classify_root(path):
        calls["root", path] += 1
        return repo_junk(path)

    def classify_cwd(path):
        calls["cwd", path] += 1
        return cwd_junk(path)

    monkeypatch.setattr(server, "_is_repo_junk", classify_root)
    monkeypatch.setattr(server, "_is_session_cwd_junk", classify_cwd)
    token = set_hermes_home_override(Path(profile))
    try:
        first = _build()
    finally:
        reset_hermes_home_override(token)
    first_calls = calls.copy()
    calls.clear()
    token = set_hermes_home_override(tmp_path / "other-profile")
    try:
        second = _build()
    finally:
        reset_hermes_home_override(token)

    # A descendant is junk as a git root, but a legitimate non-git workspace.
    assert [p["id"] for p in first["projects"]] == [project_tree.NO_PROJECT_ID, repo, notes]
    assert [p["sessionCount"] for p in first["projects"]] == [48, 24, 24]
    assert [s["id"] for s in first["projects"][1]["repos"][0]["groups"][0]["sessions"]] == [
        f"repo-{i}" for i in reversed(range(24))]
    assert [s["id"] for s in first["projects"][2]["previewSessions"]] == ["notes-23", "notes-22"]
    # Changing profile must re-evaluate both previously true and false results.
    assert [p["id"] for p in second["projects"]] == [repo, notes, profile]
    assert [p["sessionCount"] for p in second["projects"]] == [24, 48, 24]
    for tree in (first, second):
        assert Counter(tree["scoped_session_ids"]) == Counter(s["id"] for s in sessions)
    expected = Counter({("root", repo): 1, ("root", notes): 1,
                        ("cwd", notes): 1, ("cwd", profile): 1})
    assert (first_calls, calls) == (expected, expected)


def test_overlapping_profile_builds_keep_independent_classifications(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = tmp_path / "profile-a"
    path.mkdir()
    path = str(path)
    sessions = _rows(path, "git", 12, root=path) + _rows(path, "cwd", 12)
    _install_inputs(monkeypatch, sessions)
    calls = Counter()
    nested = []
    profile = "outer"
    repo_junk, cwd_junk = server._is_repo_junk, server._is_session_cwd_junk

    def classify_root(root):
        nonlocal profile
        calls[profile, "root", root] += 1
        if profile == "outer" and not nested:
            # Re-enter while the outer build's callback is active, under a different home.
            token = set_hermes_home_override(tmp_path / "profile-b")
            profile = "inner"
            try:
                nested.append(_build())
            finally:
                profile = "outer"
                reset_hermes_home_override(token)
        return repo_junk(root)

    def classify_cwd(cwd):
        calls[profile, "cwd", cwd] += 1
        return cwd_junk(cwd)

    monkeypatch.setattr(server, "_is_repo_junk", classify_root)
    monkeypatch.setattr(server, "_is_session_cwd_junk", classify_cwd)
    token = set_hermes_home_override(Path(path))
    try:
        outer = _build()
    finally:
        reset_hermes_home_override(token)
    assert [p["id"] for p in outer["projects"]] == [project_tree.NO_PROJECT_ID]
    assert [p["id"] for p in nested[0]["projects"]] == [path]
    for tree in (outer, nested[0]):
        assert tree["projects"][0]["sessionCount"] == 24
        assert Counter(tree["scoped_session_ids"]) == Counter(s["id"] for s in sessions)
    assert calls == Counter({(owner, kind, path): 1
                             for owner in ("outer", "inner") for kind in ("root", "cwd")})


def test_missing_auto_roots_skip_policy_but_recovered_worktrees_keep_cwd_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    profile = tmp_path / "profile"
    notes = profile / "notes"
    notes.mkdir(parents=True)
    profile, notes = str(profile), str(notes)
    explicit = str(tmp_path / "missing-explicit")
    recovered = profile + "-feature/src"
    # Distinct missing paths defeat memoization: none is eligible for an auto project.
    sessions = [row for i in range(12) for row in (
        _rows(str(tmp_path / "absent" / f"git{i}"), f"git{i}", 1,
              root=str(tmp_path / "absent" / f"git{i}"))
        + _rows(str(tmp_path / "absent" / f"cwd{i}"), f"cwd{i}", 1))]
    sessions += (_rows(explicit, "explicit", 1, root=explicit) + _rows(notes, "notes", 1)
                 + _rows(recovered, "recovered", 1) + _rows(profile, "junk-git", 1, root=profile)
                 + _rows(str(tmp_path), "junk-cwd", 1))
    _install_inputs(monkeypatch, sessions)
    projects = [{"id": "explicit", "name": "Explicit", "primary_path": explicit,
                 "folders": [{"path": explicit}]}]
    monkeypatch.setattr(server, "_project_tree_inputs", lambda *a, **k: (sessions, projects, [], "active"))
    monkeypatch.setattr(server.git_probe, "resolve", lambda path: (
        {"repo_root": profile, "worktree_root": profile} if path == profile else None))
    calls = Counter()
    repo_junk, cwd_junk = server._is_repo_junk, server._is_session_cwd_junk

    def classify_root(path):
        calls["root", path] += 1
        return repo_junk(path)

    def classify_cwd(path):
        calls["cwd", path] += 1
        return cwd_junk(path)

    monkeypatch.setattr(server, "_is_repo_junk", classify_root)
    monkeypatch.setattr(server, "_is_session_cwd_junk", classify_cwd)
    token = set_hermes_home_override(Path(profile))
    try:
        tree = _build()
    finally:
        reset_hermes_home_override(token)
    assert [p["id"] for p in tree["projects"]] == [project_tree.NO_PROJECT_ID, "explicit", notes, profile]
    assert [p["sessionCount"] for p in tree["projects"]] == [26, 1, 1, 1]
    # The missing sibling folds into its surviving parent, even though the parent itself
    # would be junk. The policy must still see the ORIGINAL cwd, not that parent.
    assert tree["projects"][-1]["repos"][0]["groups"][0]["path"] == profile
    assert tree["projects"][-1]["sessionIds"] == ["recovered-0"]
    assert Counter(tree["scoped_session_ids"]) == Counter(s["id"] for s in sessions)
    assert calls == Counter({("root", profile): 1, ("cwd", str(tmp_path)): 1,
                             ("cwd", notes): 1, ("cwd", recovered): 1})


@pytest.mark.platforms("posix")
def test_existing_symlinks_still_obey_distinct_root_and_cwd_policies(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = tmp_path / "profile"
    notes = profile / "notes"
    notes.mkdir(parents=True)
    home_link, notes_link = tmp_path / "home-link", tmp_path / "notes-link"
    home_link.symlink_to(profile, target_is_directory=True)
    notes_link.symlink_to(notes, target_is_directory=True)
    home_link, notes_link = str(home_link), str(notes_link)
    sessions = (_rows(home_link, "home-git", 1, root=home_link) + _rows(home_link, "home-cwd", 1)
                + _rows(notes_link, "notes-git", 1, root=notes_link) + _rows(notes_link, "notes-cwd", 1))
    _install_inputs(monkeypatch, sessions)
    token = set_hermes_home_override(profile)
    try:
        tree = _build()
    finally:
        reset_hermes_home_override(token)
    assert [p["id"] for p in tree["projects"]] == [project_tree.NO_PROJECT_ID, notes_link]
    assert [p["sessionCount"] for p in tree["projects"]] == [3, 1]
    assert tree["projects"][1]["sessionIds"] == ["notes-cwd-0"]
    assert Counter(tree["scoped_session_ids"]) == Counter(s["id"] for s in sessions)
