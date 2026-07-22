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
import sys
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
    # No session cwd recorded yet (fresh-session condition).
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    return workspace, decoy


def test_relative_terminal_cwd_anchors_to_absolute_not_process_cwd(_isolated_cwd, monkeypatch):
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
    terminal_tool.record_session_cwd("default", str(workspace))

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
    terminal_tool.record_session_cwd("default", str(container_mount))

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
    # because TERMINAL_CWD is the poison '.'.  Simulate by recording workspace
    # as the session cwd while the resolved path is under decoy.
    terminal_tool.record_session_cwd("default", str(workspace))
    resolved_in_decoy = decoy / "target.py"

    warn = ft._path_resolution_warning("target.py", resolved_in_decoy, task_id="default")

    assert warn is not None
    assert "OUTSIDE the active workspace" in warn
    assert str(decoy) in warn
    assert str(workspace) in warn


def test_no_warning_when_relative_path_inside_workspace(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))
    resolved_in_workspace = workspace / "target.py"

    warn = ft._path_resolution_warning("target.py", resolved_in_workspace, task_id="default")

    assert warn is None


def test_no_warning_for_absolute_input(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    warn = ft._path_resolution_warning(str(decoy / "target.py"), decoy / "target.py", task_id="default")

    assert warn is None


def test_no_warning_when_no_live_cwd(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    warn = ft._path_resolution_warning("target.py", decoy / "target.py", task_id="default")

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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    # Recorded session cwd = workspace; TERMINAL_CWD points elsewhere — record wins.
    terminal_tool.record_session_cwd("default", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(other))

    resolved = ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")


# ── Fix A: write_file / patch report the resolved ABSOLUTE path ──────────────


def test_write_file_reports_resolved_absolute_path(_isolated_cwd, monkeypatch):
    """write_file_tool must put the absolute on-disk path in files_modified."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("t1", str(workspace))

    import json
    out = json.loads(ft.write_file_tool("newfile.txt", "hello\n", task_id="t1"))

    expected = str((workspace / "newfile.txt").resolve())
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    assert (workspace / "newfile.txt").read_text() == "hello\n"


def test_patch_reports_resolved_absolute_path(_isolated_cwd, monkeypatch):
    """patch_tool (replace mode) must put the absolute on-disk path in files_modified."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("t1", str(workspace))

    import json
    out = json.loads(ft.patch_tool(
        mode="replace", path="target.py",
        old_string="WORKSPACE_ORIGINAL", new_string="WORKSPACE_PATCHED",
        task_id="t1",
    ))

    expected = str((workspace / "target.py").resolve())
    assert not out.get("error"), out
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    assert "WORKSPACE_PATCHED" in (workspace / "target.py").read_text()
    # And the decoy copy is untouched.
    assert (decoy / "target.py").read_text() == "DECOY_ORIGINAL\n"


# ── Cross-session isolation: one session's cwd never leaks into another ──────
# (June 2026 bug class: two desktop sessions, each on its own worktree, shared
# the single "default" terminal environment and could inherit each other's cwd.
# The per-session record store solves this structurally: each session's cd
# state lives in its own record, keyed by the raw session id.)


@pytest.fixture
def _two_worktree_sessions(tmp_path, monkeypatch):
    """Two worktree sessions: B has cd'd (record), both registered overrides."""
    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    main = tmp_path / "main"
    for d in (wt_a, wt_b, main):
        d.mkdir()
        (d / "target.py").write_text(f"{d.name}\n")
    monkeypatch.chdir(main)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    # Both sessions register their worktree cwd (TUI/desktop registration path;
    # registration seeds each session's record).
    terminal_tool.register_task_env_overrides("sess-a", {"cwd": str(wt_a)})
    terminal_tool.register_task_env_overrides("sess-b", {"cwd": str(wt_b)})
    # Session B ran the last command; the shared env's live cwd is wt_b but
    # only B's RECORD carries it.
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": _FakeEnv(str(wt_b))},
    )
    return wt_a, wt_b, main


class _FakeEnv:
    def __init__(self, cwd: str):
        self.cwd = cwd


def test_resolution_routes_to_resolving_sessions_worktree(_two_worktree_sessions):
    """The wrong-worktree fix: A resolves into wt_a, not the shared env's wt_b."""
    wt_a, wt_b, _main = _two_worktree_sessions
    resolved_a = ft._resolve_path_for_task("target.py", task_id="sess-a")
    assert resolved_a == (wt_a / "target.py")
    assert not str(resolved_a).startswith(str(wt_b))


def test_session_with_cd_record_resolves_against_it(_two_worktree_sessions):
    """B's record (its own cd state) is authoritative for B."""
    wt_a, wt_b, _main = _two_worktree_sessions
    resolved_b = ft._resolve_path_for_task("target.py", task_id="sess-b")
    assert resolved_b == (wt_b / "target.py")
    assert not str(resolved_b).startswith(str(wt_a))


def test_sessions_cd_updates_only_its_own_resolution(_two_worktree_sessions, tmp_path):
    """B cd's elsewhere → B's resolution follows, A's is untouched."""
    wt_a, wt_b, _main = _two_worktree_sessions
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    terminal_tool.record_session_cwd("sess-b", str(elsewhere))

    assert ft._resolve_path_for_task("f.py", task_id="sess-b") == (elsewhere / "f.py")
    assert ft._resolve_path_for_task("f.py", task_id="sess-a") == (wt_a / "f.py")


def test_unregistered_session_never_inherits_another_sessions_record(
    _two_worktree_sessions, monkeypatch
):
    """Session C: no record, no override. Must NOT inherit A's or B's cwd."""
    wt_a, wt_b, main = _two_worktree_sessions
    resolved = ft._resolve_path_for_task("target.py", task_id="sess-c")
    assert not str(resolved).startswith(str(wt_a))
    assert not str(resolved).startswith(str(wt_b))
    assert resolved == (main / "target.py").resolve()


# ── #67185: cwd-shaped relative input (absolute path missing leading /) ──────
# A model echoing an absolute path without its leading separator
# (``home/user/dev/notes/x.md``) joins onto the base and silently creates a
# DOUBLED tree inside the workspace, so the out-of-workspace warning above
# never fires. These tests pin the doubled-path warning.


def _workspace_echo_input(workspace: Path, *tail: str) -> str:
    """Build a relative input that replays *workspace*'s own directories."""
    return str(Path(*workspace.parts[1:], *tail))


def test_warning_fires_for_cwd_shaped_relative_input(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    echo_input = _workspace_echo_input(workspace, "notes", "x.md")
    resolved = ft._resolve_path_for_task(echo_input, task_id="default")

    # Sanity: this is the doubled path, INSIDE the workspace.
    assert resolved == workspace.joinpath(*workspace.parts[1:], "notes", "x.md")

    warn = ft._path_resolution_warning(echo_input, resolved, task_id="default")

    assert warn is not None
    assert warn.startswith(ft._DOUBLED_PATH_MARKER)
    # Paths are embedded repr-style ({...!r}), matching the existing
    # out-of-workspace message format.
    assert repr(str(resolved)) in warn
    # The suggested fix names the intended absolute path.
    assert repr(str(workspace / "notes" / "x.md")) in warn


def test_warning_fires_for_bare_workspace_echo(_isolated_cwd, monkeypatch):
    """The echo with no trailing segments is still a doubled target."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    echo_input = _workspace_echo_input(workspace)
    resolved = ft._resolve_path_for_task(echo_input, task_id="default")

    warn = ft._path_resolution_warning(echo_input, resolved, task_id="default")

    assert warn is not None
    assert warn.startswith(ft._DOUBLED_PATH_MARKER)


def test_no_warning_for_ordinary_relative_path(_isolated_cwd, monkeypatch):
    """A first segment merely equal to the workspace's first segment is fine."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    # Shares only a partial prefix with the workspace parts — not a full echo.
    partial = str(Path(workspace.parts[1], "notes", "x.md"))
    resolved = ft._resolve_path_for_task(partial, task_id="default")

    warn = ft._path_resolution_warning(partial, resolved, task_id="default")

    assert warn is None


def test_no_warning_for_subdir_named_like_root_tail(_isolated_cwd, monkeypatch):
    """workspace/<its-own-basename>/f.py is a legitimate subdirectory."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    inner = str(Path(workspace.name, "f.py"))
    resolved = ft._resolve_path_for_task(inner, task_id="default")

    warn = ft._path_resolution_warning(inner, resolved, task_id="default")

    assert warn is None


def test_write_file_surfaces_doubled_path_warning(_isolated_cwd, monkeypatch):
    """End to end: write_file_tool returns _warning for the doubled path."""
    import json

    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("t-echo", str(workspace))

    echo_input = _workspace_echo_input(workspace, "notes", "x.md")
    out = json.loads(ft.write_file_tool(echo_input, "digest\n", task_id="t-echo"))

    assert not out.get("error")
    assert out.get("_warning", "").startswith(ft._DOUBLED_PATH_MARKER)
    assert repr(str(workspace / "notes" / "x.md")) in out["_warning"]


def test_cwd_echo_warning_container_posix_semantics():
    """Container roots are PurePosixPath — the echo check must stay POSIX."""
    root = PurePosixPath("/home/user/dev")
    resolved = PurePosixPath("/home/user/dev/home/user/dev/notes/x.md")

    warn = ft._cwd_echo_warning("home/user/dev/notes/x.md", resolved, root)

    assert warn is not None
    assert warn.startswith(ft._DOUBLED_PATH_MARKER)
    assert "'/home/user/dev/notes/x.md'" in warn


def test_cwd_echo_warning_container_no_false_positive():
    root = PurePosixPath("/home/user/dev")

    warn = ft._cwd_echo_warning("notes/x.md", PurePosixPath("/home/user/dev/notes/x.md"), root)

    assert warn is None


def test_doubled_path_warning_wins_over_cross_agent_staleness(_isolated_cwd, monkeypatch):
    """A misdirected write (#67185) must not be masked by staleness warnings.

    In the issue's cron scenario the doubled target is written repeatedly by
    fresh sessions, so the cross-agent staleness warning fires on every run
    after the first — and previously would have shadowed the doubled-path
    warning entirely.
    """
    import json

    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("t-prio", str(workspace))
    monkeypatch.setattr(
        ft.file_state, "check_stale", lambda task_id, resolved: "sibling subagent touched this file"
    )

    echo_input = _workspace_echo_input(workspace, "notes", "x.md")
    out = json.loads(ft.write_file_tool(echo_input, "digest\n", task_id="t-prio"))

    assert out.get("_warning", "").startswith(ft._DOUBLED_PATH_MARKER)


@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")
def test_warning_fires_when_workspace_root_is_a_symlink(tmp_path, monkeypatch):
    """The echo check must also match the symlinked (display) form of the root.

    The model echoes the cwd the terminal reports — the symlink path — while
    the containment root is resolve()d to the real path, so comparing against
    the resolved form alone misses the echo.
    """
    real = tmp_path / "real-workspace"
    real.mkdir()
    link = tmp_path / "linked-workspace"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    terminal_tool.record_session_cwd("default", str(link))

    echo_input = _workspace_echo_input(link, "notes", "x.md")
    resolved = ft._resolve_path_for_task(echo_input, task_id="default")

    warn = ft._path_resolution_warning(echo_input, resolved, task_id="default")

    assert warn is not None
    assert warn.startswith(ft._DOUBLED_PATH_MARKER)


def test_patch_tool_doubled_path_warning_wins_over_staleness(_isolated_cwd, monkeypatch):
    """patch_tool must apply the same precedence as write_file_tool."""
    import json

    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("t-patch-prio", str(workspace))
    monkeypatch.setattr(
        ft.file_state, "check_stale", lambda task_id, resolved: "sibling subagent touched this file"
    )

    echo_input = _workspace_echo_input(workspace, "target.py")
    doubled = workspace.joinpath(*workspace.parts[1:], "target.py")
    doubled.parent.mkdir(parents=True)
    doubled.write_text("OLD\n")

    out = json.loads(
        ft.patch_tool(
            mode="replace",
            path=echo_input,
            old_string="OLD",
            new_string="NEW",
            task_id="t-patch-prio",
        )
    )

    warning = out.get("_warning") or " ".join(out.get("_warnings", []))
    assert ft._DOUBLED_PATH_MARKER in warning


def test_dotdot_that_undoes_the_echo_does_not_warn(_isolated_cwd, monkeypatch):
    """'<echo>/../../..' collapses back out of the doubled tree — no warning."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    n = len(workspace.parts) - 1
    collapsing = str(Path(*workspace.parts[1:], *[".."] * n, "plain.md"))
    resolved = ft._resolve_path_for_task(collapsing, task_id="default")

    warn = ft._path_resolution_warning(collapsing, resolved, task_id="default")

    assert warn is None


def test_dotdot_hidden_echo_is_still_detected(_isolated_cwd, monkeypatch):
    """'junk/../<echo>/x' lexically collapses to the echo — must warn."""
    workspace, decoy = _isolated_cwd
    terminal_tool.record_session_cwd("default", str(workspace))

    hidden = str(Path("junk", "..", *workspace.parts[1:], "notes", "x.md"))
    resolved = ft._resolve_path_for_task(hidden, task_id="default")

    warn = ft._path_resolution_warning(hidden, resolved, task_id="default")

    assert warn is not None
    assert warn.startswith(ft._DOUBLED_PATH_MARKER)


@pytest.mark.skipif(sys.platform != "win32", reason="UNC path semantics are Windows-specific")
def test_unc_root_echo_detection_and_no_false_positive():
    """UNC anchors hold server+share — they must join the comparison parts."""
    root = Path(r"\\server\share\work")
    doubled = Path(r"\\server\share\work\server\share\work\notes\x.md")

    warn = ft._cwd_echo_warning(r"server\share\work\notes\x.md", doubled, root)
    assert warn is not None
    assert warn.startswith(ft._DOUBLED_PATH_MARKER)
    assert repr(r"\\server\share\work\notes\x.md") in warn

    # A path that merely starts with the root's LAST component is legitimate.
    ok = ft._cwd_echo_warning(r"work\x.md", Path(r"\\server\share\work\work\x.md"), root)
    assert ok is None
