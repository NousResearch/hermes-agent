"""Regression tests for file-tool path resolution correctness.

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

Remote backend invariant these tests pin:
  Paths passed to non-local file backends MUST stay backend-lexical. Host
  canonicalization, host mtime tracking, and host ``~`` expansion must not leak
  into Docker/SSH/Modal/Daytona file operations.
"""

import os
import ntpath
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
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))

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
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))
    resolved_in_decoy = decoy / "target.py"

    warn = ft._path_resolution_warning("target.py", resolved_in_decoy, task_id="default")

    assert warn is not None
    assert "OUTSIDE the active workspace" in warn
    assert str(decoy) in warn
    assert str(workspace) in warn


def test_no_warning_when_relative_path_inside_workspace(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))
    resolved_in_workspace = workspace / "target.py"

    warn = ft._path_resolution_warning("target.py", resolved_in_workspace, task_id="default")

    assert warn is None


def test_no_warning_for_absolute_input(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))

    warn = ft._path_resolution_warning(str(decoy / "target.py"), decoy / "target.py", task_id="default")

    assert warn is None


def test_no_warning_when_no_live_cwd(_isolated_cwd, monkeypatch):
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
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
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))

    import json
    out = json.loads(ft.write_file_tool("newfile.txt", "hello\n", task_id="t1"))

    expected = str((workspace / "newfile.txt").resolve())
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    assert (workspace / "newfile.txt").read_text() == "hello\n"


def test_patch_reports_resolved_absolute_path(_isolated_cwd, monkeypatch):
    """patch_tool (replace mode) must put the absolute on-disk path in files_modified."""
    workspace, decoy = _isolated_cwd
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": str(workspace))

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


class _FakeRemoteFileOps:
    """FileOperations stand-in whose env is deliberately not LocalEnvironment."""

    env = object()

    def __init__(self):
        self.read_paths: list[str] = []
        self.write_paths: list[str] = []
        self.patch_paths: list[str] = []
        self.patch_payloads: list[str] = []
        self.search_paths: list[str] = []

    def read_file(self, path: str, offset: int = 1, limit: int = 500):
        from tools.file_operations import ReadResult

        self.read_paths.append(path)
        return ReadResult(content="1|old", total_lines=1, file_size=3)

    def write_file(self, path: str, content: str):
        from tools.file_operations import WriteResult

        self.write_paths.append(path)
        return WriteResult(bytes_written=len(content))

    def patch_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        from tools.file_operations import PatchResult

        self.patch_paths.append(path)
        return PatchResult(success=True, files_modified=[path])

    def patch_v4a(self, patch: str):
        from tools.file_operations import PatchResult

        self.patch_payloads.append(patch)
        return PatchResult(success=True, files_modified=[])

    def search(
        self,
        pattern: str,
        path: str = ".",
        target: str = "content",
        file_glob: str | None = None,
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
    ):
        from tools.file_operations import SearchResult

        self.search_paths.append(path)
        return SearchResult(matches=[])


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


def test_remote_write_file_preserves_backend_absolute_path(monkeypatch):
    """Remote absolute paths must not be canonicalized through host symlinks."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    # macOS 10.15+ resolves host /home through /System/Volumes/Data/home.  The
    # terminal backend still expects the literal remote /home path.
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": Path(
            "/System/Volumes/Data/home/myuser/now.txt"
        ),
    )

    out = json.loads(
        ft.write_file_tool(
            "/home/myuser/now.txt",
            "stamp\n",
            task_id="remote-task",
        )
    )

    assert not out.get("error"), out
    assert "_warning" not in out
    assert ops.write_paths == ["/home/myuser/now.txt"]
    assert out["resolved_path"] == "/home/myuser/now.txt"
    assert out["files_modified"] == ["/home/myuser/now.txt"]


def test_remote_write_guards_do_not_host_resolve_backend_path(monkeypatch):
    """SSH-style remote paths use the same lexical path for guards and writes."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    def host_resolve_should_not_run(filepath, task_id="default"):
        raise AssertionError("remote write guard must not host-resolve backend paths")

    monkeypatch.setattr(ft, "_resolve_path_for_task", host_resolve_should_not_run)

    out = json.loads(
        ft.write_file_tool(
            "/home/myuser/now.txt",
            "stamp\n",
            task_id="ssh-task",
        )
    )

    assert not out.get("error"), out
    assert ops.write_paths == ["/home/myuser/now.txt"]


def test_remote_patch_replace_preserves_backend_absolute_path(monkeypatch):
    """replace-mode patch should pass backend paths through like write_file."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": Path(
            "/System/Volumes/Data/home/myuser/app.py"
        ),
    )

    out = json.loads(
        ft.patch_tool(
            mode="replace",
            path="/home/myuser/app.py",
            old_string="old",
            new_string="new",
            task_id="remote-task",
        )
    )

    assert not out.get("error"), out
    assert "_warning" not in out
    assert ops.patch_paths == ["/home/myuser/app.py"]
    assert out["resolved_path"] == "/home/myuser/app.py"
    assert out["files_modified"] == ["/home/myuser/app.py"]


def test_remote_patch_guards_do_not_host_resolve_backend_path(monkeypatch):
    """Remote patch preflight uses the backend path detector consistently."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    def host_resolve_should_not_run(filepath, task_id="default"):
        raise AssertionError("remote patch guard must not host-resolve backend paths")

    monkeypatch.setattr(ft, "_resolve_path_for_task", host_resolve_should_not_run)

    out = json.loads(
        ft.patch_tool(
            mode="replace",
            path="/home/myuser/app.py",
            old_string="old",
            new_string="new",
            task_id="ssh-task",
        )
    )

    assert not out.get("error"), out
    assert ops.patch_paths == ["/home/myuser/app.py"]


def test_remote_write_file_uses_backend_cwd_without_host_resolve(monkeypatch):
    """Relative remote paths are joined lexically to the backend cwd."""
    import json

    ops = _FakeRemoteFileOps()
    ops.cwd = "/home/myuser/project"
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": Path(
            "/System/Volumes/Data/home/myuser/project/nested/now.txt"
        ),
    )

    out = json.loads(
        ft.write_file_tool(
            "nested/now.txt",
            "stamp\n",
            task_id="remote-task",
        )
    )

    expected = "/home/myuser/project/nested/now.txt"
    assert not out.get("error"), out
    assert "_warning" not in out
    assert ops.write_paths == [expected]
    assert out["resolved_path"] == expected
    assert out["files_modified"] == [expected]


def test_remote_relative_without_backend_root_does_not_use_host_cwd(monkeypatch, tmp_path):
    """Without a remote root, relative paths stay relative rather than host-anchored."""
    host_cwd = tmp_path / "host-project"
    host_cwd.mkdir()
    monkeypatch.chdir(host_cwd)
    ops = _FakeRemoteFileOps()

    assert (
        ft._lexical_path_for_task("nested/app.py", "remote-task", ops)
        == "nested/app.py"
    )


def test_remote_relative_path_uses_last_known_cwd_after_env_rebuild(monkeypatch):
    """A rebuilt remote backend keeps the session's last-known cwd anchor."""
    ops = _FakeRemoteFileOps()
    ops.cwd = "/"
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(
        ft,
        "_last_known_cwd_for",
        lambda task_id="default": "/home/myuser/project",
    )
    monkeypatch.setattr(ft, "_configured_terminal_cwd", lambda: None)

    assert (
        ft._lexical_path_for_task("nested/app.py", "remote-task", ops)
        == "/home/myuser/project/nested/app.py"
    )


def test_remote_restart_ignores_host_configured_terminal_cwd(monkeypatch):
    """A rebuilt backend must not join a host worktree cwd to remote paths."""
    ops = _FakeRemoteFileOps()
    ops.cwd = "/"
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(ft, "_last_known_cwd_for", lambda task_id="default": None)
    monkeypatch.setenv("TERMINAL_CWD", "/Users/dev/repo")

    assert (
        ft._lexical_path_for_task("nested/app.py", "remote-task", ops)
        == "/nested/app.py"
    )


def test_remote_read_and_write_use_same_backend_file_state_key(monkeypatch):
    """Remote reads and writes must share the same cross-agent file-state path."""
    import json

    ops = _FakeRemoteFileOps()
    recorded: list[tuple[str, bool]] = []
    checked: list[tuple[str, bool]] = []
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": Path("/System/Volumes/Data/home/u/app.py"),
    )
    monkeypatch.setattr(
        ft.file_state,
        "record_read",
        lambda task_id, path, partial=False, track_mtime=True: recorded.append(
            (path, track_mtime)
        ),
    )
    monkeypatch.setattr(
        ft.file_state,
        "check_stale",
        lambda task_id, path, track_mtime=True: checked.append(
            (path, track_mtime)
        ) or None,
    )

    assert not json.loads(
        ft.read_file_tool("/home/u/app.py", task_id="remote-task")
    ).get("error")
    assert not json.loads(
        ft.write_file_tool("/home/u/app.py", "new", task_id="remote-task")
    ).get("error")

    assert ops.read_paths == ["/home/u/app.py"]
    assert ops.write_paths == ["/home/u/app.py"]
    assert recorded == [("/home/u/app.py", False)]
    assert checked == [("/home/u/app.py", False)]


def test_remote_read_still_blocks_dotenv(monkeypatch):
    """Credential guards apply before a non-local backend reads project .env."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(
        ft.read_file_tool("/home/u/project/.env", task_id="remote-task")
    )

    assert "error" in out
    assert "secret-bearing environment file" in out["error"]
    assert ops.read_paths == []


def test_remote_read_uses_backend_cwd_without_host_resolve(monkeypatch):
    """Relative remote reads are joined lexically to the backend cwd."""
    import json

    ops = _FakeRemoteFileOps()
    ops.cwd = "/home/u/project"
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    def host_resolve_should_not_run(filepath, task_id="default"):
        raise AssertionError("remote reads must not host-resolve backend paths")

    monkeypatch.setattr(ft, "_resolve_path_for_task", host_resolve_should_not_run)

    out = json.loads(
        ft.read_file_tool("nested/app.py", task_id="ssh-task")
    )

    assert not out.get("error"), out
    assert ops.read_paths == ["/home/u/project/nested/app.py"]


def test_local_read_resolves_path_once(monkeypatch, tmp_path):
    """Host-backed reads do not repeat the filesystem resolution walk."""
    import json
    from tools.environments.local import LocalEnvironment

    ops = _FakeRemoteFileOps()
    ops.env = LocalEnvironment(cwd=str(tmp_path))
    calls: list[tuple[str, str]] = []
    resolved = tmp_path / "app.py"
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": calls.append((filepath, task_id)) or resolved,
    )

    out = json.loads(ft.read_file_tool("app.py", task_id="local-task"))

    assert not out.get("error"), out
    assert calls == [("app.py", "local-task")]


def test_remote_search_result_guard_does_not_host_resolve(monkeypatch):
    """Search-result filtering applies read guards to backend-lexical paths."""
    ops = _FakeRemoteFileOps()

    def host_resolve_should_not_run(filepath, task_id="default"):
        raise AssertionError("remote search guard must not host-resolve backend paths")

    monkeypatch.setattr(ft, "_resolve_path_for_task", host_resolve_should_not_run)

    error = ft._search_result_read_block_error(
        "/home/u/project/.env",
        "ssh-task",
        ops,
    )

    assert error is not None
    assert "secret-bearing environment file" in error


def test_remote_search_precheck_does_not_host_resolve(monkeypatch):
    """Search preflight and dispatch share the backend-lexical path."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    def host_resolve_should_not_run(filepath, task_id="default"):
        raise AssertionError("remote search precheck must not host-resolve backend paths")

    monkeypatch.setattr(ft, "_resolve_path_for_task", host_resolve_should_not_run)

    out = json.loads(
        ft.search_tool(
            pattern="needle",
            path="/home/u/project",
            task_id="ssh-task",
        )
    )

    assert not out.get("error"), out
    assert ops.search_paths == ["/home/u/project"]


def test_remote_search_reuses_backend_path_kind_for_result_guards(monkeypatch):
    """Search result filtering must not reflect on the backend per match."""
    import json

    from tools.file_operations import SearchMatch, SearchResult

    ops = _FakeRemoteFileOps()
    ops.search = lambda **kwargs: SearchResult(
        matches=[
            SearchMatch(path=f"/home/u/project/file-{index}.py", line_number=1, content="x")
            for index in range(20)
        ]
    )
    calls = 0

    def uses_host_paths(file_ops):
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(ft, "_file_ops_uses_host_paths", uses_host_paths)

    out = json.loads(
        ft.search_tool(
            pattern="needle",
            path="/home/u/project",
            task_id="ssh-task",
        )
    )

    assert not out.get("error"), out
    assert calls == 1


def test_remote_read_docx_does_not_use_host_extractor(monkeypatch, tmp_path):
    """Host document extraction should not run for non-local backend paths."""
    import json
    import tools.read_extract as read_extract

    ops = _FakeRemoteFileOps()
    calls: list[str] = []
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": tmp_path / "remote.docx",
    )

    def host_extract(path: str):
        calls.append(path)
        raise read_extract.ExtractionError("host extractor should not run")

    monkeypatch.setattr(read_extract, "extract_document_text", host_extract)

    out = json.loads(ft.read_file_tool("/home/u/remote.docx", task_id="remote-task"))

    assert not out.get("error"), out
    assert calls == []
    assert ops.read_paths == ["/home/u/remote.docx"]


def test_remote_file_state_records_backend_paths_with_real_registry(monkeypatch):
    """The registry should track backend paths even when host stat cannot see them."""
    import json

    ft.file_state.get_registry().clear()
    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(ft.read_file_tool("/remote/shared.txt", task_id="agent-a"))
    ft.file_state.note_write("agent-b", "/remote/shared.txt")
    warn = ft.file_state.check_stale("agent-a", "/remote/shared.txt")

    assert not out.get("error"), out
    assert "/remote/shared.txt" in ft.file_state.known_reads("agent-a")
    assert warn is not None
    assert "sibling subagent" in warn


def test_file_ops_without_real_env_are_host_backed():
    """MagicMock-style fakes should not fabricate a remote backend env."""
    from unittest.mock import MagicMock

    ops = MagicMock()

    assert ft._file_ops_uses_host_paths(ops)
    assert ft._resolve_path_for_file_ops("/tmp/out.txt", "default", ops) == "/tmp/out.txt"


def test_file_state_does_not_record_failed_remote_read(monkeypatch):
    """Failed backend reads must not become stale-write baselines."""
    import json
    from tools.file_operations import ReadResult

    class FailingRemoteOps(_FakeRemoteFileOps):
        def read_file(self, path: str, offset: int = 1, limit: int = 500):
            self.read_paths.append(path)
            return ReadResult(error=f"File not found: {path}")

    ft.file_state.get_registry().clear()
    ops = FailingRemoteOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(ft.read_file_tool("/remote/missing.txt", task_id="agent-a"))

    assert "error" in out
    assert ops.read_paths == ["/remote/missing.txt"]
    assert "/remote/missing.txt" not in ft.file_state.known_reads("agent-a")


def test_remote_read_does_not_host_resolve_before_backend_detection(monkeypatch):
    """Remote reads should classify paths after backend detection, not host resolve."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    def host_resolve_should_not_run(filepath, task_id="default"):
        raise AssertionError("remote read should not host-resolve before backend detection")

    monkeypatch.setattr(ft, "_resolve_path_for_task", host_resolve_should_not_run)

    out = json.loads(ft.read_file_tool("/home/u/app.py", task_id="remote-task"))

    assert not out.get("error"), out
    assert ops.read_paths == ["/home/u/app.py"]


def test_remote_write_after_remote_write_ignores_host_mtime(monkeypatch, tmp_path):
    """Remote writes must not seed per-task stale checks from host mtimes."""
    import json

    with ft._read_tracker_lock:
        ft._read_tracker.clear()
    ft.file_state.get_registry().clear()

    host_path = tmp_path / "home" / "u" / "app.py"
    host_path.parent.mkdir(parents=True)
    host_path.write_text("host v1\n")

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda filepath, task_id="default": host_path,
    )

    assert not json.loads(
        ft.write_file_tool("/home/u/app.py", "remote v1\n", task_id="remote-task")
    ).get("error")

    new_ts = host_path.stat().st_mtime + 10
    os.utime(host_path, (new_ts, new_ts))

    out = json.loads(
        ft.write_file_tool("/home/u/app.py", "remote v2\n", task_id="remote-task")
    )

    assert not out.get("error"), out
    assert "_warning" not in out
    assert ops.write_paths == ["/home/u/app.py", "/home/u/app.py"]


def test_read_binary_extension_does_not_create_backend(monkeypatch):
    """Cheap binary read rejection should not start a remote backend."""
    import json

    calls = []
    monkeypatch.setattr(
        ft,
        "_get_file_ops",
        lambda task_id="default": calls.append(task_id) or _FakeRemoteFileOps(),
    )

    out = json.loads(ft.read_file_tool("/tmp/picture.png", task_id="remote-task"))

    assert "binary file" in out["error"]
    assert calls == []


def test_remote_tilde_hermes_path_hits_container_mirror_guard(monkeypatch):
    """Remote '~/.hermes' writes should hit the Docker mirror soft guard."""
    import json

    calls = []
    monkeypatch.setattr(
        ft,
        "_get_file_ops",
        lambda task_id="default": calls.append(task_id) or _FakeRemoteFileOps(),
    )
    monkeypatch.setattr(
        ft,
        "_get_container_mirror_prefix_for_task",
        lambda task_id="default": "/root/.hermes",
    )
    monkeypatch.setattr(
        ft,
        "_expand_tilde",
        lambda path: path.replace("~", "/Users/host", 1),
    )

    out = json.loads(
        ft.write_file_tool("~/.hermes/SOUL.md", "x", task_id="remote-task")
    )

    assert "Sandbox-mirror write blocked" in out["error"]
    assert calls == []


def test_remote_sensitive_path_rejected_without_backend(monkeypatch):
    """An obvious sensitive target is rejected before backend creation."""
    import json

    calls = []
    monkeypatch.setattr(
        ft,
        "_get_file_ops",
        lambda task_id="default": calls.append(task_id),
    )

    out = json.loads(
        ft.write_file_tool("/etc/shadow", "x", task_id="remote-task")
    )

    assert "sensitive system path" in out["error"]
    assert calls == []


def test_remote_lexical_paths_stay_posix_on_windows_host(monkeypatch):
    """Remote lexical resolution must not inherit host path semantics."""
    ops = _FakeRemoteFileOps()
    ops.cwd = "/home/myuser/project"
    monkeypatch.setattr(ft.os, "path", ntpath)

    assert (
        ft._lexical_path_for_task("/home/myuser/app.py", "remote-task", ops)
        == "/home/myuser/app.py"
    )
    assert (
        ft._lexical_path_for_task("nested/app.py", "remote-task", ops)
        == "/home/myuser/project/nested/app.py"
    )


def test_remote_lexical_prefers_registered_cwd_when_live_cwd_owned_by_other_session(monkeypatch):
    """A stale shared backend cwd from another session must not shadow this session."""
    ops = _FakeRemoteFileOps()
    ops.env = _FakeOwnedEnv("/home/b/project", "sess-b")
    ops.cwd = "/home/b/project"
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    terminal_tool.register_task_env_overrides("sess-a", {"cwd": "/home/a/project"})

    assert (
        ft._lexical_path_for_task("nested/app.py", "sess-a", ops)
        == "/home/a/project/nested/app.py"
    )


def test_remote_registered_posix_cwd_survives_windows_host(monkeypatch):
    """A Linux backend cwd remains absolute even when the host path module is Windows."""
    ops = _FakeRemoteFileOps()
    ops.env = _FakeOwnedEnv("/home/b/project", "sess-b")
    ops.cwd = "/home/b/project"
    monkeypatch.setattr(ft.os, "path", ntpath)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    terminal_tool.register_task_env_overrides("sess-a", {"cwd": "/home/a/project"})

    assert (
        ft._lexical_path_for_task("nested/app.py", "sess-a", ops)
        == "/home/a/project/nested/app.py"
    )


def test_remote_search_uses_registered_cwd_when_live_cwd_owned_by_other_session(monkeypatch):
    """search_files should not inherit another session's shared backend cwd."""
    import json

    ops = _FakeRemoteFileOps()
    ops.env = _FakeOwnedEnv("/home/b/project", "sess-b")
    ops.cwd = "/home/b/project"
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    terminal_tool.register_task_env_overrides("sess-a", {"cwd": "/home/a/project"})

    out = json.loads(
        ft.search_tool(
            pattern="needle",
            path="nested",
            task_id="sess-a",
        )
    )

    assert not out.get("error"), out
    assert ops.search_paths == ["/home/a/project/nested"]


def test_local_search_keeps_caller_relative_path(monkeypatch, tmp_path):
    """Host-backed search preserves the caller path shape for local results."""
    import json
    from tools.environments.local import LocalEnvironment

    class RecordingLocalOps(_FakeRemoteFileOps):
        def __init__(self):
            super().__init__()
            self.env = LocalEnvironment(cwd=str(tmp_path))

    (tmp_path / "nested").mkdir()
    monkeypatch.chdir(tmp_path)
    ops = RecordingLocalOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(ft.search_tool(pattern="needle", path="nested", task_id="default"))

    assert not out.get("error"), out
    assert ops.search_paths == ["nested"]


def test_remote_write_file_does_not_expand_tilde_to_host_home(monkeypatch):
    """Remote '~' paths should be expanded by the backend, not the host."""
    import json

    ops = _FakeRemoteFileOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(ft, "_expand_tilde", lambda path: path.replace("~", "/Users/host", 1))

    out = json.loads(
        ft.write_file_tool(
            "~/now.txt",
            "stamp\n",
            task_id="remote-task",
        )
    )

    assert not out.get("error"), out
    assert "_warning" not in out
    assert ops.write_paths == ["~/now.txt"]
    assert out["resolved_path"] == "~/now.txt"
    assert out["files_modified"] == ["~/now.txt"]


def test_remote_lexical_tilde_paths_are_left_for_backend_expansion():
    """Do not normalize '~' paths; normpath can collapse them to non-tilde paths."""
    ops = _FakeRemoteFileOps()

    assert ft._lexical_path_for_task("~/../x", "remote-task", ops) == "~/../x"
    assert ft._lexical_path_for_task("~/a/../b", "remote-task", ops) == "~/a/../b"


def test_remote_patch_v4a_rewrites_relative_headers_to_backend_cwd(monkeypatch):
    """V4A headers are rewritten before reaching a non-local backend."""
    import json

    ops = _FakeRemoteFileOps()
    ops.cwd = "/home/myuser/project"
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(
        ft.patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Update File: nested/app.py\n"
                "@@\n"
                "-old\n"
                "+new\n"
                "*** End Patch\n"
            ),
            task_id="remote-task",
        )
    )

    assert not out.get("error"), out
    assert len(ops.patch_payloads) == 1
    assert "*** Update File: /home/myuser/project/nested/app.py" in ops.patch_payloads[0]
    assert out["files_modified"] == ["/home/myuser/project/nested/app.py"]


def test_remote_patch_v4a_reports_move_as_resolved_pair(monkeypatch):
    """Successful V4A moves should preserve the parser's 'src -> dst' shape."""
    import json

    ops = _FakeRemoteFileOps()
    ops.cwd = "/home/myuser/project"
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(
        ft.patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Move File: old.py -> nested/new.py\n"
                "*** End Patch\n"
            ),
            task_id="remote-task",
        )
    )

    assert not out.get("error"), out
    assert len(ops.patch_payloads) == 1
    assert (
        "*** Move File: /home/myuser/project/old.py -> "
        "/home/myuser/project/nested/new.py"
    ) in ops.patch_payloads[0]
    assert out["files_modified"] == [
        "/home/myuser/project/old.py -> /home/myuser/project/nested/new.py"
    ]


def test_remote_patch_v4a_move_marks_individual_paths_stale(monkeypatch):
    """Internal stale markers should receive real paths, not display strings."""
    import json

    ops = _FakeRemoteFileOps()
    ops.cwd = "/home/myuser/project"
    marked: list[str] = []
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)
    monkeypatch.setattr(
        ft,
        "_mark_verification_stale",
        lambda task_id, paths, session_id=None: marked.extend(paths),
    )

    out = json.loads(
        ft.patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Move File: old.py -> nested/new.py\n"
                "*** End Patch\n"
            ),
            task_id="remote-task",
        )
    )

    assert not out.get("error"), out
    assert out["files_modified"] == [
        "/home/myuser/project/old.py -> /home/myuser/project/nested/new.py"
    ]
    assert marked == [
        "/home/myuser/project/old.py",
        "/home/myuser/project/nested/new.py",
    ]


def test_patch_v4a_rejects_traversal_in_move_header(monkeypatch):
    """Traversal in either side of a V4A Move header must be blocked."""
    import json
    from unittest.mock import MagicMock

    ops = MagicMock()
    ops.env = object()
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": ops)

    out = json.loads(
        ft.patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Move File: ../../../etc/shadow -> safe.txt\n"
                "*** End Patch\n"
            ),
            task_id="remote-task",
        )
    )

    assert "error" in out
    assert "traversal" in out["error"].lower()
    ops.patch_v4a.assert_not_called()
