"""Regression tests for file-tool path resolution base correctness.

The bug (observed in a worktree dev session, May 2026): when the resolution
base for a relative path is itself RELATIVE — e.g. ``TERMINAL_CWD="."`` from a
stale config — ``_resolve_path_for_task`` resolved the path against the agent's
PROCESS cwd instead of the intended workspace. In a git-worktree session this
silently routed ``patch``/``write_file`` edits into the *main* checkout: the
write landed, self-verified, and reported success — against the wrong file.
The agent then grepped the worktree, saw nothing, and concluded the patch tool
had silently no-op'd. It hadn't; it wrote to the wrong place.

Core invariant these tests pin:
  The resolution base for a relative path MUST always be absolute. A relative
  ``TERMINAL_CWD`` (``.``, ``./sub``, ``..``) must be anchored deterministically,
  never left to resolve against whatever the process cwd happens to be.
"""

import os
from pathlib import Path, PurePosixPath

import pytest

import tools.file_tools as ft
import tools.terminal_tool as terminal_tool


@pytest.fixture
def _isolated_cwd(tmp_path, monkeypatch):
    """Two checkouts: workspace (intended) + decoy (process cwd)."""
    workspace = tmp_path / "workspace"
    decoy = tmp_path / "decoy"
    workspace.mkdir()
    decoy.mkdir()
    (workspace / "target.py").write_text("WORKSPACE_ORIGINAL\n")
    (decoy / "target.py").write_text("DECOY_ORIGINAL\n")
    # Process cwd = decoy, analogous to "main repo" while the terminal is in
    # the worktree.
    monkeypatch.chdir(decoy)
    # No live-terminal-cwd tracking recorded yet (fresh-session condition).
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    return workspace, decoy


def test_relative_terminal_cwd_anchors_to_absolute_not_process_cwd(
    _isolated_cwd, monkeypatch
):
    """TERMINAL_CWD='.' must NOT silently mean 'the agent process cwd'.

    A relative base is meaningless as a resolution anchor. The resolver must
    make it absolute deterministically. We assert the resolved path is
    absolute and stable regardless of where os.getcwd() points.
    """
    workspace, decoy = _isolated_cwd
    # Poison config: literal relative '.'
    monkeypatch.setenv("TERMINAL_CWD", ".")

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved.is_absolute(), f"resolution base leaked a relative path: {resolved}"
    # The exact anchor for a bare '.' is the process cwd resolved to absolute —
    # that is acceptable as long as it is ABSOLUTE and stable. The bug was that
    # a relative base produced surprising results; the fix is that the base is
    # always absolutised. (We do not require it to point at the workspace here —
    # that's what live-cwd tracking is for; see the next test.)
    assert str(resolved) == str((Path(os.getcwd()) / "target.py").resolve())


def test_live_tracking_cwd_wins_over_relative_terminal_cwd(_isolated_cwd, monkeypatch):
    """When the terminal reports its absolute cwd, that is authoritative.

    This is the real-world fix: the terminal's tracked absolute cwd (the
    worktree) must override a stale relative TERMINAL_CWD so edits land where
    the agent is actually working.
    """
    workspace, decoy = _isolated_cwd
    monkeypatch.setenv("TERMINAL_CWD", ".")
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace)
    )

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")


def test_absolute_terminal_cwd_used_verbatim(_isolated_cwd, monkeypatch):
    """An absolute TERMINAL_CWD is the resolution base (no live tracking)."""
    workspace, decoy = _isolated_cwd
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")


def test_absolute_input_path_ignores_base(_isolated_cwd, monkeypatch):
    """An absolute input path is never re-anchored."""
    workspace, decoy = _isolated_cwd
    monkeypatch.setenv("TERMINAL_CWD", ".")
    abs_target = str(workspace / "target.py")

    resolved = ft._resolve_path_for_task(abs_target, task_id="default")

    assert resolved == Path(abs_target).resolve()


def test_container_absolute_input_path_does_not_follow_host_symlink(tmp_path, monkeypatch):
    """Docker paths are sandbox-local and must not be host-dereferenced.

    A user may have a host symlink at a container-looking path such as
    ``/workspace/projects``. For Docker file ops, resolving that symlink on the
    host rewrites the path before Docker sees it, making file tools and terminal
    disagree about where the file lives.
    """
    host_project = tmp_path / "host-project"
    host_project.mkdir()
    container_mount = tmp_path / "workspace-projects"
    container_mount.symlink_to(host_project, target_is_directory=True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "docker"})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})

    container_path = container_mount / "oilsands-sim" / "README.md"
    resolved = ft._resolve_path_for_task(str(container_path), task_id="default")

    assert resolved == container_path
    assert resolved != (host_project / "oilsands-sim" / "README.md")


def test_container_path_normalization_uses_posix_path_syntax():
    resolved = ft._normalize_without_host_deref("/workspace/projects/foo/../bar")

    assert resolved == PurePosixPath("/workspace/projects/bar")
    assert str(resolved) == "/workspace/projects/bar"


def test_container_relative_path_keeps_container_cwd_symlink(tmp_path, monkeypatch):
    """Relative Docker paths should stay under the container cwd textually."""
    host_project = tmp_path / "host-project"
    host_project.mkdir()
    container_mount = tmp_path / "workspace-projects"
    container_mount.symlink_to(host_project, target_is_directory=True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "docker"})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(container_mount))

    resolved = ft._resolve_path_for_task("oilsands-sim/README.md", task_id="default")

    assert resolved == container_mount / "oilsands-sim" / "README.md"
    assert resolved != host_project / "oilsands-sim" / "README.md"


class _DummyDockerEnvironment:
    cwd = "/workspace"
    cwd_owner = "default"


def test_container_path_detection_uses_live_docker_environment(monkeypatch):
    """A live DockerEnvironment-shaped env should beat config fallback."""
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": _DummyDockerEnvironment()},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: (_ for _ in ()).throw(AssertionError("should not read config")),
    )
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    assert ft._uses_container_paths("default") is True


def test_resolution_base_always_absolute_no_terminal_cwd(_isolated_cwd, monkeypatch):
    """With TERMINAL_CWD unset, the base falls back to an ABSOLUTE process cwd."""
    workspace, decoy = _isolated_cwd
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved.is_absolute()
    assert str(resolved) == str((Path(os.getcwd()) / "target.py").resolve())


# ── B-(ii): workspace-divergence warning ────────────────────────────────────


def test_warning_fires_when_relative_path_escapes_workspace(_isolated_cwd, monkeypatch):
    """Relative path resolving outside the live workspace must warn."""
    workspace, decoy = _isolated_cwd
    # Live cwd = workspace, but the relative path resolves to decoy (process cwd)
    # because TERMINAL_CWD is the poison '.'.  Simulate by pointing live tracking
    # at workspace while the resolved path is under decoy.
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace)
    )
    resolved_in_decoy = decoy / "target.py"

    warn = ft._path_resolution_warning(
        "target.py", resolved_in_decoy, task_id="default"
    )

    assert warn is not None
    assert "OUTSIDE the active workspace" in warn
    assert str(decoy) in warn
    assert str(workspace) in warn


def test_no_warning_when_relative_path_inside_workspace(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace)
    )
    resolved_in_workspace = workspace / "target.py"

    warn = ft._path_resolution_warning(
        "target.py", resolved_in_workspace, task_id="default"
    )

    assert warn is None


def test_no_warning_for_absolute_input(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace)
    )

    warn = ft._path_resolution_warning(
        str(decoy / "target.py"), decoy / "target.py", task_id="default"
    )

    assert warn is None


def test_no_warning_when_no_live_cwd(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    warn = ft._path_resolution_warning(
        "target.py", decoy / "target.py", task_id="default"
    )

    assert warn is None


# ── Fix C: sentinel TERMINAL_CWD + empty-registry worktree anchoring ─────────
# (May 2026 follow-up: PR #35399 made misroutes visible via resolved_path but
# the divergence warning only fired when the live terminal cwd was known. A
# worktree session whose terminal registry is still empty — no `cd` run yet —
# got neither a worktree anchor nor a warning, so a relative edit silently
# landed in main. These tests pin the sentinel handling + empty-registry
# anchoring + early warning.)


@pytest.mark.parametrize("sentinel", ["", ".", "./", "auto", "cwd", "CWD", "Auto"])
def test_sentinel_terminal_cwd_is_treated_as_unset(_isolated_cwd, monkeypatch, sentinel):
    """Sentinel TERMINAL_CWD values are NOT used as a directory anchor.

    They fall through to the (absolute) process cwd, exactly as if unset —
    never resolved as a literal relative directory.
    """
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.setenv("TERMINAL_CWD", sentinel)

    assert ft._configured_terminal_cwd() is None
    resolved = ft._resolve_path_for_task("target.py", task_id="default")
    assert resolved.is_absolute()
    assert resolved == (decoy / "target.py").resolve()


def test_relative_nonsentinel_terminal_cwd_rejected(_isolated_cwd, monkeypatch):
    """A relative (but non-sentinel) TERMINAL_CWD is still rejected as an anchor.

    A relative anchor is ambiguous (relative to which cwd?), which is the exact
    ambiguity that misroutes edits. It must fall through to the process cwd, not
    be joined onto it as a literal subdir.
    """
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.setenv("TERMINAL_CWD", "some/rel/path")

    assert ft._configured_terminal_cwd() is None
    resolved = ft._resolve_path_for_task("target.py", task_id="default")
    assert resolved == (decoy / "target.py").resolve()


def test_absolute_terminal_cwd_anchors_with_empty_registry(_isolated_cwd, monkeypatch):
    """The incident-preventing case: worktree session, registry still empty.

    With no live terminal cwd recorded yet but an absolute TERMINAL_CWD (the
    worktree path cli.py/main.py set for `-w`), a relative edit must land in the
    worktree — not the process cwd (main repo).
    """
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")
    assert not str(resolved).startswith(str(decoy))


def test_registered_task_cwd_override_anchors_before_terminal_env_exists(_isolated_cwd, monkeypatch):
    """TUI/Desktop sessions register cwd by raw session key before tools run.

    CWD-only overrides collapse to the shared terminal environment key, but the
    file resolver must still read the raw task/session override before falling
    back to TERMINAL_CWD or the process cwd.
    """
    workspace, decoy = _isolated_cwd
    task_id = "desktop-session-cwd"
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(workspace)})

    resolved = ft._resolve_path_for_task("target.py", task_id=task_id)

    assert terminal_tool._resolve_container_task_id(task_id) == "default"
    assert resolved == (workspace / "target.py")
    assert not str(resolved).startswith(str(decoy))


def test_warning_fires_from_terminal_cwd_when_registry_empty(_isolated_cwd, monkeypatch):
    """Divergence warning must fire even before any terminal command runs.

    PR #35399's warning required a live terminal cwd; a fresh worktree session
    (empty registry) silently misrouted with no warning. Now the warning falls
    back to the absolute TERMINAL_CWD anchor, so an edit aimed outside the
    worktree is flagged on the very first write.
    """
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    # Relative path that escapes the worktree into the decoy/main checkout.
    escaping = os.path.relpath(str(decoy / "target.py"), str(workspace))
    resolved = ft._resolve_path_for_task(escaping, task_id="default")

    warn = ft._path_resolution_warning(escaping, resolved, task_id="default")

    assert warn is not None
    assert "OUTSIDE the active workspace" in warn
    assert str(workspace) in warn


def test_live_cwd_still_wins_over_absolute_terminal_cwd(_isolated_cwd, monkeypatch):
    """When both are present, the live terminal cwd remains authoritative."""
    workspace, decoy = _isolated_cwd
    other = decoy.parent / "other"
    other.mkdir()
    # Live cwd = workspace; TERMINAL_CWD points elsewhere — live must win.
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(other))

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")


# ── Fix A: write_file / patch report the resolved ABSOLUTE path ──────────────


def test_write_file_reports_resolved_absolute_path(_isolated_cwd, monkeypatch):
    """write_file_tool must put the absolute on-disk path in files_modified."""
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace)
    )

    import json

    out = json.loads(ft.write_file_tool("newfile.txt", "hello\n", task_id="t1"))

    expected = str((workspace / "newfile.txt").resolve())
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    assert (workspace / "newfile.txt").read_text() == "hello\n"


def test_patch_reports_resolved_absolute_path(_isolated_cwd, monkeypatch):
    """patch_tool (replace mode) must put the absolute on-disk path in files_modified."""
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace)
    )

    import json

    out = json.loads(
        ft.patch_tool(
            mode="replace",
            path="target.py",
            old_string="WORKSPACE_ORIGINAL",
            new_string="WORKSPACE_PATCHED",
            task_id="t1",
        )
    )

    expected = str((workspace / "target.py").resolve())
    assert not out.get("error"), out
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    assert "WORKSPACE_PATCHED" in (workspace / "target.py").read_text()
    # And the decoy copy is untouched.
    assert (decoy / "target.py").read_text() == "DECOY_ORIGINAL\n"


# ── Fix D: shared terminal env must not leak its cwd across worktree sessions ─
# (June 2026: two desktop sessions, each on its own worktree, share the single
# "default" terminal environment. Its `cwd` tracks whichever session ran the
# last command, so a file edit from the OTHER session resolved against that
# foreign cwd and silently landed in the wrong worktree. terminal_tool now
# stamps env.cwd_owner with the driving session; file tools trust the shared
# env's live cwd only when the resolving session owns it.)


class _FakeOwnedEnv:
    def __init__(self, cwd: str, cwd_owner: str):
        self.cwd = cwd
        self.cwd_owner = cwd_owner


@pytest.fixture
def _two_worktree_sessions(tmp_path, monkeypatch):
    """Two worktree sessions sharing one terminal env owned by session B."""
    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    main = tmp_path / "main"
    for d in (wt_a, wt_b, main):
        d.mkdir()
        (d / "target.py").write_text(f"{d.name}\n")
    monkeypatch.chdir(main)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    # Both sessions register their worktree cwd (TUI/desktop registration path).
    terminal_tool.register_task_env_overrides("sess-a", {"cwd": str(wt_a)})
    terminal_tool.register_task_env_overrides("sess-b", {"cwd": str(wt_b)})
    # The shared "default" env: session B ran the last command, so its live cwd
    # is wt_b and B owns it.
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": _FakeOwnedEnv(str(wt_b), "sess-b")},
    )
    return wt_a, wt_b, main


def test_live_cwd_ignored_for_non_owning_session(_two_worktree_sessions):
    wt_a, wt_b, _main = _two_worktree_sessions
    # Owner sees the live cwd; the other session must NOT inherit it.
    assert ft._get_live_tracking_cwd("sess-b") == str(wt_b)
    assert ft._get_live_tracking_cwd("sess-a") is None


def test_resolution_routes_to_resolving_sessions_worktree(_two_worktree_sessions):
    """The wrong-worktree fix: A resolves into wt_a, not the shared env's wt_b."""
    wt_a, wt_b, _main = _two_worktree_sessions
    # Session A does not own the shared env → falls back to its own registered
    # worktree cwd instead of B's live cwd.
    resolved_a = ft._resolve_path_for_task("target.py", task_id="sess-a")
    assert resolved_a == (wt_a / "target.py")
    assert not str(resolved_a).startswith(str(wt_b))


def test_owning_session_still_resolves_against_live_cwd(_two_worktree_sessions):
    """No regression: the owner keeps resolving against the live cwd."""
    wt_a, wt_b, _main = _two_worktree_sessions
    resolved_b = ft._resolve_path_for_task("target.py", task_id="sess-b")
    assert resolved_b == (wt_b / "target.py")
    assert not str(resolved_b).startswith(str(wt_a))


def test_unknown_owner_keeps_prior_single_session_behavior(tmp_path, monkeypatch):
    """An env with no owner (CLI / legacy) still yields its live cwd."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": _FakeOwnedEnv(str(ws), "")},
    )
    assert ft._get_live_tracking_cwd("default") == str(ws)
    assert ft._get_live_tracking_cwd("any-session") == str(ws)


def test_preserved_cwd_does_not_override_non_owning_sessions_worktree(
    _two_worktree_sessions, monkeypatch
):
    """#26211 belt-and-suspenders must not break worktree isolation.

    The owner (session B) doing an owned live read mirrors wt_b into the shared
    _last_known_cwd['default'] registry. Session A — which does NOT own the env
    but HAS its own registered worktree (wt_a) — must still resolve into wt_a,
    not inherit B's preserved cwd through the shared-container key. The
    session-specific registered override must beat the durable shared anchor.
    """
    wt_a, wt_b, _main = _two_worktree_sessions
    monkeypatch.setattr(ft, "_last_known_cwd", {})

    # Owner B resolves first — this mirrors wt_b into _last_known_cwd['default'].
    assert ft._resolve_path_for_task("target.py", task_id="sess-b") == (wt_b / "target.py")
    assert ft._last_known_cwd.get("default") == str(wt_b)

    # A still routes to its own registered worktree despite the shared anchor.
    resolved_a = ft._resolve_path_for_task("target.py", task_id="sess-a")
    assert resolved_a == (wt_a / "target.py")
    assert not str(resolved_a).startswith(str(wt_b))
