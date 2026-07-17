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
import json
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

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


class _DummyLocalEnvironment:
    cwd = "/Users/me/project"


class _DummySSHEnvironment:
    cwd = "/srv/host-a"
    host = "host-a.example"
    user = "deploy"
    port = 22
    key_path = "/keys/a"
    _persistent = True


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


@pytest.mark.parametrize(
    ("config", "stale_env", "expected_path"),
    [
        (
            {"env_type": "local", "cwd": "/opt/project"},
            _DummyDockerEnvironment(),
            "/opt/project/notes.txt",
        ),
        (
            {"env_type": "docker", "cwd": "/workspace"},
            _DummyLocalEnvironment(),
            "/workspace/notes.txt",
        ),
        (
            {
                "env_type": "ssh",
                "cwd": "/srv/host-b",
                "ssh_host": "host-b.example",
                "ssh_user": "deploy",
                "ssh_port": 22,
                "ssh_key": "/keys/b",
                "ssh_persistent": True,
            },
            _DummySSHEnvironment(),
            "/srv/host-b/notes.txt",
        ),
    ],
)
def test_relative_read_and_write_use_requested_backend_during_switch(
    monkeypatch,
    config,
    stale_env,
    expected_path,
):
    """Public file tools must resolve before cache replacement in the new namespace."""

    captured = {"read": [], "write": []}

    class _ReadResult:
        content = "1|hello"

        def to_dict(self):
            return {
                "content": self.content,
                "file_size": 5,
                "total_lines": 1,
                "truncated": False,
            }

    class _FileOps:
        def read_file(self, path, _offset, _limit):
            captured["read"].append(path)
            return _ReadResult()

        def write_file(self, path, _content):
            captured["write"].append(str(path))
            result = MagicMock()
            result.to_dict.return_value = {"status": "success"}
            return result

    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_docker_runtime_identity",
        lambda **_kwargs: ({}, "sha256:test", "requested-runtime", []),
    )
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": stale_env},
    )
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    monkeypatch.setattr(ft, "_read_tracker", {})
    monkeypatch.setattr(ft, "_get_file_ops", lambda _task_id: _FileOps())
    monkeypatch.setattr(ft, "_mark_verification_stale", lambda *args, **kwargs: None)

    read_result = json.loads(ft.read_file_tool("notes.txt"))
    write_result = json.loads(ft.write_file_tool("notes.txt", "updated\n"))

    assert "error" not in read_result
    assert "error" not in write_result
    assert captured == {
        "read": [expected_path],
        "write": [expected_path],
    }


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
    # macOS pytest roots live below /private/var, which the production safety
    # guard correctly treats as sensitive; this test exercises cwd dispatch.
    monkeypatch.setattr(ft, "_check_sensitive_path", lambda *_args, **_kwargs: None)

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
    monkeypatch.setattr(ft, "_check_sensitive_path", lambda *_args, **_kwargs: None)

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


def test_relative_file_dispatch_is_bound_to_each_shared_docker_session(
    _two_worktree_sessions, monkeypatch
):
    """Read/search/V4A dispatch uses each session's resolved absolute path."""

    wt_a, wt_b, _main = _two_worktree_sessions
    file_ops = MagicMock()

    read_result = MagicMock()
    read_result.content = "target\n"
    read_result.to_dict.return_value = {
        "content": "target\n",
        "total_lines": 1,
        "truncated": False,
    }
    file_ops.read_file.return_value = read_result

    search_result = MagicMock()
    search_result.matches = []
    search_result.to_dict.return_value = {
        "matches": [],
        "truncated": False,
    }
    file_ops.search.return_value = search_result

    patch_result = MagicMock()
    patch_result.to_dict.return_value = {"status": "ok"}
    file_ops.patch_v4a.return_value = patch_result

    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id: file_ops)
    monkeypatch.setattr(ft, "_read_tracker", {})
    monkeypatch.setattr(ft, "_check_sensitive_path", lambda *args, **kwargs: None)

    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: target.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )
    for task_id in ("sess-a", "sess-b"):
        ft.read_file_tool("target.py", task_id=task_id)
        ft.search_tool("target", path=".", task_id=task_id)
        ft.patch_tool(mode="patch", patch=patch_text, task_id=task_id)

    expected_targets = [str(wt_a / "target.py"), str(wt_b / "target.py")]
    assert [call.args[0] for call in file_ops.read_file.call_args_list] == (
        expected_targets
    )
    assert [call.kwargs["path"] for call in file_ops.search.call_args_list] == [
        str(wt_a),
        str(wt_b),
    ]
    rewritten_patches = [call.args[0] for call in file_ops.patch_v4a.call_args_list]
    assert f"*** Update File: {expected_targets[0]}" in rewritten_patches[0]
    assert f"*** Update File: {expected_targets[1]}" in rewritten_patches[1]


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
