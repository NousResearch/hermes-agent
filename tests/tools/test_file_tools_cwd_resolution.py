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

import json
import os
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


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/proc/1/task/1/root/dev/zero", True),
        ("/proc/self/root", False),
        ("/workspace/proc/self/root/dev/zero", False),
        ("/proc/١/root/dev/zero", False),
        ("/proc/1/task/١/root/dev/zero", False),
        # /proc/<pid>/exe is a symlink to the process's own executable —
        # reading it shares the same hang/leak surface as the other
        # /proc suffix family (environ, maps, ...), so it is blocked
        # outright rather than requiring cwd-style resolution.
        ("/proc/self/exe", True),
        ("/proc/12345/exe", True),
        ("/proc/self/task/1234/exe", True),
        ("/proc/self/root/proc/self/exe", True),
        # Suffix match must be exact ("/exe"), not a bare substring.
        ("/proc/self/exec", False),
        ("/proc/self/myexe", False),
    ],
)
def test_container_device_path_canonicalization_walk(path, expected):
    assert ft._is_blocked_container_device_path(path) is expected


def _empty_search_ops():
    mock_ops = MagicMock()
    result_obj = MagicMock(matches=[])
    result_obj.to_dict.return_value = {"matches": []}
    mock_ops.search.return_value = result_obj
    return mock_ops


@pytest.fixture
def _container_search_backend(monkeypatch):
    mock_ops = _empty_search_ops()
    file_ops_factory = MagicMock(return_value=mock_ops)
    host_check = MagicMock(
        side_effect=AssertionError("container search used a host device predicate")
    )
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "docker"})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setenv("TERMINAL_CWD", "/workspace")
    monkeypatch.setattr(ft, "get_read_block_error", lambda candidate: None)
    monkeypatch.setattr(ft, "_is_blocked_device_path", host_check)
    monkeypatch.setattr(ft, "_is_blocked_device", host_check)
    monkeypatch.setattr(ft, "_get_file_ops", file_ops_factory)
    return mock_ops, file_ops_factory


@pytest.mark.parametrize(
    "path, blocked",
    [
        ("/proc/self/root/../dev/zero", True),
        ("/proc/1/task/1/root/dev/zero", True),
        ("/proc/1/root/proc/self/root/dev/zero", True),
        ("/proc/thread-self/root/dev/zero", True),
        ("/dev/zero", True),
        ("/proc/self/environ", True),
        # /proc/<self|thread-self>/cwd is a symlink to the task's own cwd
        # (TERMINAL_CWD == "/workspace" in this fixture), so ".." segments
        # after the alias must resolve against that cwd, not be left as
        # literal path text.
        ("/proc/self/cwd/../dev/zero", True),
        ("/proc/self/cwd/../../dev/zero", True),
        ("/proc/thread-self/cwd/../dev/zero", True),
        # Per-thread cwd alias: /proc/self/task/<tid>/cwd -- "self" still
        # names this task's own process, so it resolves the same way.
        ("/proc/self/task/1234/cwd/../dev/zero", True),
        # Numeric pid/tid cwd aliases fail CLOSED regardless of what follows
        # -- this task cannot verify a different process's real cwd, so even
        # a benign-looking suffix (no ../ needed to reach a device) is
        # refused rather than approximated. See _resolve_container_cwd_alias.
        ("/proc/1/cwd/../dev/zero", True),
        ("/proc/1/cwd/src/main.py", True),
        ("/proc/1/cwd", True),
        ("/proc/1/task/2/cwd/../dev/zero", True),
        ("/proc/1/task/2/cwd", True),
        ("/proc/self/exe", True),
        ("/proc/1/task/2/exe", True),
        ("//dev/zero", True),
        ("///dev/zero", True),
        # Lexical bypass spellings must still be recognized as the alias
        # (duplicate "/" and literal "." are collapsed before the alias
        # regex runs -- see _normalize_slashes_and_dots_only), not silently
        # fall through to the literal-path walker, which doesn't know "cwd"
        # is a symlink at all.
        ("/proc/./self/cwd/../dev/zero", True),
        ("/proc//self/cwd/../dev/zero", True),
        ("//proc/self/cwd/../dev/zero", True),
        ("/proc/self/./cwd/../dev/zero", True),
        ("/proc/./1/cwd/src/main.py", True),
        ("//./etc/hosts", False),
        ("//./workspace/tools", False),
        ("/etc/hosts", False),
        ("tools", False),
        ("workspace/src", False),
        ("dev/zero", False),
        ("/workspace/proc/self/root/dev/zero", False),
        ("/proc/self/root", False),
        ("/proc/self/cwd/src/main.py", False),
        ("/proc/self/cwd", False),
        ("/proc/self/cwd/", False),
        ("/proc/self/task/1234/cwd", False),
        # Non-ASCII digits must not satisfy the pid/tid match (mirrors the
        # existing /proc/.../root canonicalization behavior).
        ("/proc/١/cwd/../dev/zero", False),
        # Non-numeric tid must not match the /task/<tid>/cwd alias either.
        ("/proc/self/task/abc/cwd/../dev/zero", False),
    ],
)
def test_container_search_classifies_paths_through_production_resolution(
    _container_search_backend, path, blocked
):
    """Use search_tool's real resolver with a configured Docker cwd."""
    mock_ops, file_ops_factory = _container_search_backend
    task_id = "container-production-resolution"
    result = json.loads(ft.search_tool(pattern="x", path=path, task_id=task_id))

    if blocked:
        assert set(result) == {"error"}
        assert "device file" in result["error"]
        file_ops_factory.assert_not_called()
    else:
        assert result == {"matches": []}
        file_ops_factory.assert_called_once_with(task_id)


@pytest.mark.parametrize(
    "cwd_alias",
    ["/proc/self/cwd", "/proc/self/task/2/cwd"],
    ids=["self-alias", "self-task-tid-alias"],
)
@pytest.mark.parametrize(
    "container_cwd, blocked",
    [
        ("/workspace", True),
        ("/a/b/c", False),
    ],
)
def test_container_search_self_cwd_alias_respects_cwd_depth(
    _container_search_backend, monkeypatch, cwd_alias, container_cwd, blocked
):
    """The self/thread-self proc cwd alias (plain or per-thread
    /task/<tid>/cwd form) must be resolved against the configured container
    cwd, not treated as literal path text -- a shallower cwd means
    "../dev/zero" no longer lands on the real device path, so a blocked
    verdict at one depth must NOT generalize to another (refusal must track
    the actual resolved path, not the alias string alone), and a
    non-blocked verdict must still dispatch to search (no over-blocking of a
    genuinely safe, deep cwd)."""
    mock_ops, file_ops_factory = _container_search_backend
    monkeypatch.setenv("TERMINAL_CWD", container_cwd)
    path = f"{cwd_alias}/../dev/zero"
    task_id = f"container-self-cwd-depth-{cwd_alias}-{blocked}"

    result = json.loads(ft.search_tool(pattern="x", path=path, task_id=task_id))

    if blocked:
        assert set(result) == {"error"}
        assert "device file" in result["error"]
        file_ops_factory.assert_not_called()
    else:
        assert result == {"matches": []}
        file_ops_factory.assert_called_once_with(task_id)
        assert mock_ops.search.call_args.kwargs["path"] == path


@pytest.mark.parametrize(
    "numeric_pid_alias",
    ["/proc/1/cwd", "/proc/1/task/2/cwd"],
    ids=["pid-alias", "task-tid-alias"],
)
@pytest.mark.parametrize("container_cwd", ["/workspace", "/a/b/c"])
def test_container_search_numeric_pid_cwd_alias_always_refused(
    _container_search_backend, monkeypatch, numeric_pid_alias, container_cwd
):
    """A numeric-pid /proc/<pid>/cwd alias (or its /task/<tid>/cwd form)
    names a DIFFERENT process's cwd that this task cannot verify -- there is
    no task-registry lookup here from pid to the process that actually owns
    it. Approximating it as this task's own cwd could bypass the guard (if
    the real target is deeper than this task's own cwd), so it fails
    closed: refused regardless of this task's own cwd depth, and even for a
    benign-looking suffix that never needs "../" to reach a device -- the
    alias itself is refused, proving there is no depth-dependent bypass."""
    mock_ops, file_ops_factory = _container_search_backend
    monkeypatch.setenv("TERMINAL_CWD", container_cwd)
    path = f"{numeric_pid_alias}/src/main.py"
    task_id = f"container-numeric-cwd-refusal-{numeric_pid_alias}-{container_cwd}"

    result = json.loads(ft.search_tool(pattern="x", path=path, task_id=task_id))

    assert set(result) == {"error"}
    assert "device file" in result["error"]
    file_ops_factory.assert_not_called()


def _seed_not_found_cache(resolved_path: str, task_id: str) -> None:
    """Pre-populate the search negative-result cache as if a prior call had
    legitimately missed on *resolved_path* -- used to prove enforcement runs
    before the cache is ever consulted, not to exercise the cache's own
    correctness."""
    ft._record_not_found(
        "search", resolved_path, task_id,
        json.dumps({"error": f"Path not found: {resolved_path}"}, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# Regression: rv-20260804-175517-r69403rr -- the negative-result cache
# lookup ran BEFORE the device/cwd-alias guard, so a stale "not found" hit
# under the key the generic (alias-unaware) top-of-function resolver
# produces could short-circuit the request and skip enforcement entirely.
# Each case below seeds a cache entry under exactly that generic key, then
# confirms the guard still fires -- i.e. the cache is never consulted until
# AFTER normalization, cwd-alias resolution, and the device-block decision.
# ---------------------------------------------------------------------------

def test_numeric_pid_cwd_alias_refusal_survives_stale_not_found_cache(
    _container_search_backend,
):
    """Numeric-pid cwd alias must stay refused (fail-closed) even when a
    stale not-found entry exists under the key the generic, alias-unaware
    resolver would have produced for this exact literal path (a plausible
    prior legitimate miss) -- a cache hit here would silently bypass the
    refusal."""
    mock_ops, file_ops_factory = _container_search_backend
    task_id = "cache-bypass-numeric-pid"
    path = "/proc/1/cwd/../dev/zero"
    # posixpath.normpath (the generic resolver) pops "cwd" lexically via
    # ".." without knowing it's a symlink, landing on "/proc/1/dev/zero".
    _seed_not_found_cache("/proc/1/dev/zero", task_id)

    result = json.loads(ft.search_tool(pattern="x", path=path, task_id=task_id))

    assert set(result) == {"error"}
    assert "device file" in result["error"]
    file_ops_factory.assert_not_called()


def test_bypass_spelling_cwd_alias_refusal_survives_stale_not_found_cache(
    _container_search_backend,
):
    """A lexical bypass spelling ("/proc/./self/cwd/...") must still be
    recognized as the cwd alias and blocked, even when a stale not-found
    entry exists under the key the generic resolver alone would produce."""
    mock_ops, file_ops_factory = _container_search_backend
    task_id = "cache-bypass-spelling"
    path = "/proc/./self/cwd/../dev/zero"
    # The generic resolver collapses both "./" and ".." lexically, popping
    # "cwd" without knowing it's a symlink, landing on "/proc/self/dev/zero".
    _seed_not_found_cache("/proc/self/dev/zero", task_id)

    result = json.loads(ft.search_tool(pattern="x", path=path, task_id=task_id))

    assert set(result) == {"error"}
    assert "device file" in result["error"]
    file_ops_factory.assert_not_called()


def test_plain_device_path_refusal_survives_stale_not_found_cache(
    _container_search_backend,
):
    """A plain (non-alias) device path must stay blocked even when a stale
    not-found entry exists under its own resolved key."""
    mock_ops, file_ops_factory = _container_search_backend
    task_id = "cache-bypass-plain-device"
    path = "/dev/zero"
    _seed_not_found_cache("/dev/zero", task_id)

    result = json.loads(ft.search_tool(pattern="x", path=path, task_id=task_id))

    assert set(result) == {"error"}
    assert "device file" in result["error"]
    file_ops_factory.assert_not_called()


def test_local_search_dispatches_posix_double_slash_path_after_resolution(monkeypatch):
    path = "//./etc/hosts"
    resolved_path = PurePosixPath("/etc/hosts")
    file_ops_factory = MagicMock(return_value=_empty_search_ops())
    local_device_check = MagicMock(return_value=False)

    assert ft._is_blocked_device_path(path)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(
        ft, "_resolve_path_for_task",
        lambda candidate, task_id="default": resolved_path,
    )
    monkeypatch.setattr(ft, "get_read_block_error", lambda candidate: None)
    monkeypatch.setattr(ft, "_is_blocked_device", local_device_check)
    monkeypatch.setattr(ft, "_get_file_ops", file_ops_factory)

    result = json.loads(ft.search_tool(pattern="x", path=path, task_id="local-posix"))

    assert result == {"matches": []}
    local_device_check.assert_called_once_with(str(resolved_path), base_dir=None)


class _DummyDockerEnvironment:
    cwd = "/workspace"
    cwd_owner = "default"


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


# ── Fix C: sentinel TERMINAL_CWD + empty-registry worktree anchoring ─────────
# (May 2026 follow-up: PR #35399 made misroutes visible via resolved_path but
# the divergence warning only fired when the live terminal cwd was known. A
# worktree session whose terminal registry is still empty — no `cd` run yet —
# got neither a worktree anchor nor a warning, so a relative edit silently
# landed in main. These tests pin the sentinel handling + empty-registry
# anchoring + early warning.)


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


# ── Fix A: write_file / patch report the resolved ABSOLUTE path ──────────────


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


def test_unregistered_session_never_inherits_another_sessions_record(
    _two_worktree_sessions, monkeypatch
):
    """Session C: no record, no override. Must NOT inherit A's or B's cwd."""
    wt_a, wt_b, main = _two_worktree_sessions
    resolved = ft._resolve_path_for_task("target.py", task_id="sess-c")
    assert not str(resolved).startswith(str(wt_a))
    assert not str(resolved).startswith(str(wt_b))
    assert resolved == (main / "target.py").resolve()


def test_v4a_patch_applies_to_resolved_workspace_not_backend_cwd(
    _isolated_cwd, monkeypatch
):
    """V4A patch must edit the path the tool layer resolved, not the shell cwd.

    Regression for the git-worktree cwd bug: ``patch_tool`` resolved header
    paths against the task workspace for locking/staleness/reporting, but the
    raw (relative) patch text was handed to ``file_ops.patch_v4a``, which
    re-resolved it against the backend env's own cwd. A relative header then
    landed in a different directory than everything the tool reported. The fix
    rewrites headers to the resolved absolute paths before apply.
    """
    import json

    workspace, decoy = _isolated_cwd
    task_id = "sess-v4a"

    # Tool layer resolves against the workspace (worktree registration path).
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(workspace)})

    # Backend file_ops lives in the DECOY dir — the divergence the fix closes.
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    env = LocalEnvironment(cwd=str(decoy))
    monkeypatch.setattr(
        ft, "_get_file_ops", lambda task_id="default": ShellFileOperations(env)
    )

    out = json.loads(
        ft.patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Update File: target.py\n"
                "@@\n"
                "-WORKSPACE_ORIGINAL\n"
                "+WORKSPACE_PATCHED\n"
                "*** End Patch\n"
            ),
            task_id=task_id,
        )
    )

    expected = str((workspace / "target.py").resolve())
    assert not out.get("error"), out
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    # The workspace file — which the tool locked and reported — was edited.
    assert (workspace / "target.py").read_text() == "WORKSPACE_PATCHED\n"
    # The decoy (backend cwd) was left untouched.
    assert (decoy / "target.py").read_text() == "DECOY_ORIGINAL\n"
