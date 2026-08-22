"""Terminal-level regression tests for elevated execution guard integration.

Verifies that elevated commands pass through the same _check_all_guards
decision path as ordinary terminal commands, and that the elevated executor
is never called when the guard returns blocked or pending-approval.
"""

import json
from pathlib import Path

import pytest

import tools.terminal_tool as terminal_tool


def _make_minimal_config(**overrides) -> dict:
    """Return a minimal _get_env_config()-shaped dict for testing."""
    config = {
        "env_type": "local",
        "cwd": "/tmp",
        "timeout": 30,
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "mount_docker_cwd": False,
        "docker_forward_env": [],
        "docker_volumes": [],
        "docker_env": {},
        "docker_extra_args": [],
        "container_cpu": 1.0,
        "container_memory": 5120,
        "container_disk": 51200,
        "forward_env": [],
        "always_forward_env": [],
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Elevated command blocked by guard — executor not called
# ---------------------------------------------------------------------------


def test_elevated_blocked_does_not_call_executor(monkeypatch):
    """When _check_all_guards returns blocked, elevated executor is not called."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    # Guard returns blocked
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {
            "approved": False,
            "status": "blocked",
            "description": "rm -rf / is hardline blocked",
            "message": "Command denied: some safety rule",
        },
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="rm -rf /",
        elevated=True,
    )
    data = json.loads(result)

    assert data["status"] == "blocked"
    assert data["exit_code"] == -1
    assert executor_calls == [], (
        f"execute_elevated should not have been called when blocked, "
        f"but was called {len(executor_calls)} time(s)"
    )


# ---------------------------------------------------------------------------
# Elevated command pending-approval — executor not called
# ---------------------------------------------------------------------------


def test_elevated_pending_approval_does_not_call_executor(monkeypatch):
    """When _check_all_guards returns pending_approval, executor is not called."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {
            "approved": False,
            "status": "pending_approval",
            "description": "command flagged",
            "command": cmd,
            "pattern_key": "dangerous_pattern",
        },
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="sudo rm -rf /var/log",
        elevated=True,
    )
    data = json.loads(result)

    assert data["status"] == "pending_approval"
    assert data.get("approval_pending") is True
    assert executor_calls == [], (
        f"execute_elevated should not have been called when pending_approval, "
        f"but was called {len(executor_calls)} time(s)"
    )


# ---------------------------------------------------------------------------
# Elevated command approved — executor IS called
# ---------------------------------------------------------------------------


def test_elevated_approved_calls_executor(monkeypatch):
    """When _check_all_guards approves, the elevated executor is called."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {
            "approved": True,
            "status": "approved",
            "description": "",
        },
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: executor_calls.append({
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
        }) or {"output": "admin result\n", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="whoami /priv",
        elevated=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == 0
    assert "admin result" in data["output"]
    assert len(executor_calls) == 1, (
        f"execute_elevated should have been called once when approved, "
        f"but was called {len(executor_calls)} time(s)"
    )
    assert executor_calls[0]["command"] == "whoami /priv"


# ---------------------------------------------------------------------------
# force=True skips guard but still dispatches elevated
# ---------------------------------------------------------------------------


def test_elevated_force_skips_guard_calls_executor(monkeypatch):
    """force=True should skip _check_all_guards and still dispatch elevated."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )

    guard_calls = []
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: guard_calls.append((cmd, env, kw)) or {"approved": True},
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: executor_calls.append(command)
        or {"output": "forced\n", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="format C: /fs:NTFS",
        elevated=True,
        force=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == 0
    assert guard_calls == [], "guard should not be called when force=True"
    assert executor_calls == ["format C: /fs:NTFS"]


# ---------------------------------------------------------------------------
# Fail-closed: non-local backend must block elevated and never call UAC
# ---------------------------------------------------------------------------


def _fake_env():
    class FakeEnv:
        env = {}
        cwd = "/tmp"

        def execute(self, command, **kwargs):
            raise AssertionError("FakeEnv.execute should not be called for elevated")

    return FakeEnv()


@pytest.mark.parametrize("env_type", ["docker", "ssh", "daytona", "vercel_sandbox"])
def test_elevated_non_local_backend_blocked(monkeypatch, env_type):
    """elevated=true with a non-local backend must fail closed (blocked)."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(env_type=env_type),
    )
    # Pre-populate an environment so the tool doesn't try to create the real
    # backend (docker/ssh/...) during the test.
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(command="whoami", elevated=True)
    data = json.loads(result)

    assert data["status"] == "blocked"
    assert "local" in data["error"]
    assert executor_calls == [], (
        f"execute_elevated must never be called for env_type={env_type}, "
        f"but was called {len(executor_calls)} time(s)"
    )


@pytest.mark.parametrize("env_type", ["docker", "ssh", "daytona", "vercel_sandbox"])
def test_elevated_non_local_fail_fast_no_backend_side_effects(monkeypatch, env_type):
    """Non-local elevated must fail BEFORE any backend is created/connected.

    This is the fail-fast contract: the elevated+env_type check must run
    before environment creation/connection/recovery. We spy on the backend
    constructor and the cleanup thread starter — neither may run.
    """
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(env_type=env_type),
    )

    # Spy on backend side-effect entry points. These would be invoked by the
    # normal path between config resolution and the (old) elevated gate.
    side_effects = []
    monkeypatch.setattr(
        terminal_tool, "_create_environment",
        lambda *a, **kw: side_effects.append(("_create_environment", a, kw)) or object(),
    )
    monkeypatch.setattr(
        terminal_tool, "_start_cleanup_thread",
        lambda: side_effects.append(("_start_cleanup_thread",)),
    )
    # resolve_task_overrides / container cwd normalization are pure reads, but
    # the fail-fast must occur before they even matter; spy the config cwd path
    # that container backends would touch.
    monkeypatch.setattr(
        terminal_tool, "_is_unusable_container_cwd",
        lambda cwd: side_effects.append(("_is_unusable_container_cwd", cwd)) or False,
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append(("execute_elevated", a, kw))
        or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(command="whoami", elevated=True)
    data = json.loads(result)

    assert data["status"] == "blocked"
    assert "local" in data["error"]
    # The fail-fast gate returns before ANY backend machinery runs.
    assert side_effects == [], (
        f"backend side effects must not run for non-local elevated; got {side_effects}"
    )
    assert executor_calls == [], "execute_elevated must not be called"

    # The elevated+background / elevated+pty combos must also fail before
    # side effects even on a LOCAL backend (they are unsupported).
    for combo in ({"background": True}, {"pty": True}):
        side_effects.clear()
        monkeypatch.setattr(
            terminal_tool, "_get_env_config",
            lambda: _make_minimal_config(env_type="local"),
        )
        result = terminal_tool.terminal_tool(command="whoami", elevated=True, **combo)
        data = json.loads(result)
        assert data["exit_code"] == -1
        assert side_effects == [], (
            f"backend side effects must not run for elevated+{combo}; got {side_effects}"
        )


# ---------------------------------------------------------------------------
# Fail-closed: injection-style workdir must be blocked before elevated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_workdir", [
    "C:/tmp/a&b",
    "C:/tmp/a|b",
    'C:/tmp/a"b',
    "C:/tmp/a<b",
    "C:/tmp/$(whoami)",
])
def test_elevated_injection_workdir_blocked(monkeypatch, bad_workdir):
    """A malicious workdir must be rejected before the elevated executor runs."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="whoami", elevated=True, workdir=bad_workdir,
    )
    data = json.loads(result)

    assert data["status"] == "blocked"
    assert "workdir" in data["error"].lower()
    assert executor_calls == [], (
        f"execute_elevated must not be called with injection workdir {bad_workdir!r}"
    )


# ---------------------------------------------------------------------------
# session_key CWD is honored by elevated execution
# ---------------------------------------------------------------------------


def test_elevated_uses_session_key_cwd(monkeypatch):
    """Elevated execution must resolve cwd via session_key, not raw env cwd."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(cwd="/tmp"),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )
    # A session with a recorded cwd different from config cwd.
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"default": "/session/workspace"})

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: executor_calls.append({
            "command": command, "cwd": cwd, "timeout": timeout,
        }) or {"output": "ok\n", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(command="whoami", elevated=True)
    data = json.loads(result)

    assert data["exit_code"] == 0
    assert executor_calls, "execute_elevated should have been called"
    assert executor_calls[0]["cwd"] == "/session/workspace", (
        f"elevated must honor session-scoped cwd, got {executor_calls[0]['cwd']!r}"
    )


def test_elevated_explicit_workdir_wins_over_session_cwd(monkeypatch):
    """An explicit workdir still wins over the recorded session cwd."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(cwd="/tmp"),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"default": "/session/workspace"})

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: executor_calls.append(cwd)
        or {"output": "ok\n", "exit_code": 0, "error": None},
    )

    terminal_tool.terminal_tool(
        command="whoami", elevated=True, workdir="/explicit/dir",
    )
    assert executor_calls == ["/explicit/dir"], executor_calls


# ---------------------------------------------------------------------------
# Gateway lifecycle guard cannot be bypassed by elevated
# ---------------------------------------------------------------------------


def test_elevated_cannot_bypass_gateway_lifecycle_guard(monkeypatch):
    """Elevated must not bypass the gateway lifecycle command guard."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="hermes gateway restart", elevated=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == 1
    assert "Blocked" in data["error"]
    assert executor_calls == [], (
        "execute_elevated must not be called when gateway lifecycle guard blocks"
    )


# ---------------------------------------------------------------------------
# Elevated output goes through the unified pipeline (redaction/truncation/spill)
# ---------------------------------------------------------------------------


def test_elevated_output_redacted(monkeypatch):
    """Elevated output must pass through secret redaction like normal output."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: {
            "output": "token=ghp_1234567890abcdef secret line\n",
            "exit_code": 0,
            "error": None,
        },
    )

    result = terminal_tool.terminal_tool(command="printenv", elevated=True)
    data = json.loads(result)

    assert data["exit_code"] == 0
    # ghp_ token must be redacted from elevated output.
    assert "ghp_1234567890abcdef" not in data["output"], (
        "elevated output must be redacted through the unified pipeline"
    )


def test_elevated_output_truncated_with_spill_metadata(monkeypatch, tmp_path):
    """Oversized elevated output must truncate and expose spill metadata.

    The elevated executor returns a bounded head/tail window plus a staged
    RAW file handle (output_total_chars / raw_output_path) — exactly the
    shape ``_execute_elevated_impl`` produces via ``_read_output_bounded`` +
    ``_stage_raw_output``.  The terminal pipeline must sanitize the raw file
    into a NEW durable spill path and report the metadata + truncation note.
    """
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )
    big_output = "X" * 200_000
    # The staged raw file on disk holds the FULL output; the executor only
    # loads the bounded window into the result dict.  It lives in a
    # throwaway temp dir, never the durable spill dir.
    raw_dir = tmp_path / "raw-staging"
    raw_dir.mkdir()
    raw_file = raw_dir / "output.raw"
    raw_file.write_bytes(big_output.encode("utf-8"))
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: {
            "output": big_output[:1000],
            "exit_code": 0,
            "error": None,
            "output_total_chars": len(big_output),
            "raw_output_path": str(raw_file),
        },
    )
    monkeypatch.setattr(
        "tools.tool_output_limits.get_max_bytes", lambda: 1000,
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path,
    )

    result = terminal_tool.terminal_tool(command="big", elevated=True)
    data = json.loads(result)

    assert data["exit_code"] == 0
    assert data.get("output_total_chars") == len(big_output)
    final_path = Path(data["full_output_path"])
    # Sanitized spill lives in the durable dir; raw staging is gone.
    assert final_path.parent == tmp_path / "cache" / "terminal-output"
    assert not raw_dir.exists()
    redacted = final_path.read_text(encoding="utf-8")
    assert redacted == big_output  # no secrets in fixture; file still intact
    # The visible output is the bounded window (no unbounded full read).
    assert len(data["output"]) <= 1000


# ---------------------------------------------------------------------------
# Fail-closed: background / pty combinations
# ---------------------------------------------------------------------------


def test_elevated_background_fail_closed(monkeypatch):
    """elevated=true + background=true must fail closed before any spawn."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="whoami", elevated=True, background=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == -1
    assert "background" in data["error"]
    assert executor_calls == [], "elevated executor must not run with background=True"


def test_elevated_pty_fail_closed(monkeypatch):
    """elevated=true + pty=true must fail closed."""
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: _make_minimal_config(),
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda cmd, env, **kw: {"approved": True},
    )

    executor_calls = []
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda *a, **kw: executor_calls.append((a, kw)) or {"output": "", "exit_code": 0, "error": None},
    )

    result = terminal_tool.terminal_tool(
        command="whoami", elevated=True, pty=True,
    )
    data = json.loads(result)

    assert data["exit_code"] == -1
    assert "pty" in data["error"]
    assert executor_calls == [], "elevated executor must not run with pty=True"


# ---------------------------------------------------------------------------
# Streaming spill redaction — bounded-memory full-file sanitization
# ---------------------------------------------------------------------------


def _make_spill(tmp_path, content, name="spill.log"):
    p = tmp_path / name
    p.write_bytes(content.encode("utf-8"))
    return p


class _ReadSpy:
    """Wraps a binary file object; records every read size (no-arg read fails)."""

    def __init__(self, real, reads):
        self._real = real
        self._reads = reads

    def read(self, size=-1):
        self._reads.append(size)
        if size == -1:
            raise AssertionError(
                "no-arg read() forbidden: streaming spill redaction must "
                "read in fixed-size chunks"
            )
        return self._real.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._real.close()


@pytest.fixture
def spy_stream_reads(monkeypatch):
    """Patch builtins.open to spy on binary reads.

    ``open`` is a builtin (not a module attribute of tools.terminal_tool), so
    the spy wraps it globally; non-binary opens pass straight through.
    """
    import builtins

    reads = []
    real_open = builtins.open

    def _fake_open(path, mode="r", **kwargs):
        f = real_open(path, mode, **kwargs)
        if "b" in mode:
            return _ReadSpy(f, reads)
        return f

    monkeypatch.setattr("builtins.open", _fake_open)
    return reads


def _stream_spill(
    monkeypatch,
    tmp_path,
    content,
    chunk=8,
    name="spill.log",
    dest_name=None,
    command="echo hi",
):
    """Run _stream_redact_spill(src, dest) with a forced small chunk size.

    ``dest`` defaults to the source path (in-place sanitization, the normal
    foreground-path semantics); pass ``dest_name`` for the elevated
    raw->new-durable-path semantics.
    """
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_SPILL_STREAM_CHUNK_BYTES", chunk)
    p = _make_spill(tmp_path, content, name=name)
    dest = str(tmp_path / (dest_name or name))
    final_path, total = tt._stream_redact_spill(str(p), dest, command)
    return p, final_path, total


def test_stream_spill_no_full_reads(spy_stream_reads, monkeypatch, tmp_path):
    """Spill redaction must never call Path.read_text or a no-arg read."""
    from pathlib import Path

    read_text_calls = []
    real_read_text = Path.read_text

    def _spy_read_text(self, *a, **kw):
        read_text_calls.append(self)
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _spy_read_text)

    content = ("L" * 10_000) + "END"
    p, final_path, total = _stream_spill(monkeypatch, tmp_path, content)
    # Only the spill file itself matters: unrelated process-initialization
    # reads (e.g. hermes_state's WAL-reset warning stamping .install_method
    # under the test HOME, added upstream in #77484) must not fail the
    # spill-only assertion.
    spill_reads = [c for c in read_text_calls if str(c) in (str(p), str(final_path))]
    assert spill_reads == [], "Path.read_text must never be used on the spill"
    assert spy_stream_reads, "binary streaming reads must have happened"
    assert all(0 < s <= 8 for s in spy_stream_reads)
    assert p.read_text(encoding="utf-8") == content  # round-trip intact
    assert total == len(content)


def test_stream_spill_reads_have_fixed_upper_bound(spy_stream_reads, monkeypatch, tmp_path):
    """Every read is bounded by the chunk constant — proves no full read."""
    content = ("R" * 5000) + "\n"
    _stream_spill(monkeypatch, tmp_path, content, chunk=64)
    assert spy_stream_reads
    assert all(0 < s <= 64 for s in spy_stream_reads)


def test_stream_spill_cross_chunk_secret_fully_redacted(monkeypatch, tmp_path):
    """A secret straddling many forced chunk boundaries is fully redacted."""
    secret = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0" * 3
    content = ("F" * 200) + " " + secret + "\n" + ("G" * 200)
    p, _, total = _stream_spill(monkeypatch, tmp_path, content)
    final = p.read_text(encoding="utf-8")
    assert secret not in final
    assert "A1b2C3d4E5f6G7h8I9j0A1b2C3d4E5f6G7h8I9j0" not in final
    assert ("F" * 200) in final and ("G" * 200) in final
    assert total == len(content)


def test_stream_spill_secret_spans_many_chunks(monkeypatch, tmp_path):
    """A secret starting in one chunk and terminating several chunks later
    leaves no head/middle/tail segment in the output."""
    secret = "ghp_" + "D" * 3000
    content = ("Z" * 50) + " " + secret + " " + ("M" * 50) + "\n"
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=64)
    final = p.read_text(encoding="utf-8")
    assert secret not in final
    assert ("D" * 40) not in final
    assert ("Z" * 50) in final and ("M" * 50) in final


def test_stream_spill_cross_chunk_ansi_stripped(monkeypatch, tmp_path):
    """ANSI escape sequences straddling chunk boundaries are stripped."""
    ansi = "\x1b[31mRED\x1b[0m"
    content = ("n" * 200) + ansi + ("m" * 200)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content)
    final = p.read_text(encoding="utf-8")
    assert "\x1b" not in final
    assert "RED" in final
    assert ("n" * 200) in final and ("m" * 200) in final


def test_stream_spill_multibyte_roundtrip(monkeypatch, tmp_path):
    """Chinese/emoji/ASCII mixed output crosses chunks without corruption."""
    unit = "中文😀A测试🚀"
    content = unit * 300
    p, _, total = _stream_spill(monkeypatch, tmp_path, content)
    final = p.read_text(encoding="utf-8")
    assert final == content
    assert "\ufffd" not in final
    assert total == len(content)
    assert len(content.encode("utf-8")) > len(content)  # bytes > chars


def test_stream_spill_char_count_is_chars_not_bytes(monkeypatch, tmp_path):
    """output_total_chars semantics: Unicode characters, never bytes."""
    content = ("中文" * 2000) + "tail"
    p, _, total = _stream_spill(monkeypatch, tmp_path, content)
    assert total == len(content)
    assert total < p.stat().st_size  # chars < bytes for multibyte content


def test_stream_spill_atomic_replace_no_leftovers(monkeypatch, tmp_path):
    """The spill is replaced atomically; no .tmp/.partial files remain."""
    content = ("C" * 5000) + "\n"
    p, final_path, _ = _stream_spill(monkeypatch, tmp_path, content)
    assert final_path == str(p)
    assert p.exists()
    assert p.read_text(encoding="utf-8") == content
    leftovers = [
        x for x in tmp_path.iterdir()
        if ".partial" in x.name or ".tmp-" in x.name
    ]
    assert leftovers == []


def test_stream_spill_plain_text_never_deleted(monkeypatch, tmp_path):
    """A long plain-text run (no secrets) streams through intact — the
    fail-closed machinery never deletes normal content."""
    line = "ordinary prose " * 2000
    content = line + "\n"
    p, _, total = _stream_spill(monkeypatch, tmp_path, content, chunk=256)
    final = p.read_text(encoding="utf-8")
    assert line in final
    assert total == len(content)


def test_stream_spill_oversize_token_no_leak(monkeypatch, tmp_path):
    """A >64 KiB sk- token (beyond any fixed window) is fully masked —
    no head/middle/tail segment of the secret survives."""
    token = "sk-proj-" + "B" * (80 * 1024)
    content = ("X" * 100) + " " + token + "\n" + ("Y" * 100)
    p, _, total = _stream_spill(monkeypatch, tmp_path, content, chunk=4096)
    final = p.read_text(encoding="utf-8")
    assert token not in final
    assert ("B" * 40) not in final            # middle segment gone
    assert ("sk-proj-" + "B" * 30) not in final  # head segment gone
    assert ("X" * 100) in final and ("Y" * 100) in final
    assert total == len(content)


def test_stream_spill_oversize_authorization_no_leak(monkeypatch, tmp_path):
    """An oversized Authorization: Bearer <token> value is fully masked."""
    token = "B" * (80 * 1024)
    content = ("H" * 100) + "\nAuthorization: Bearer " + token + "\n" + ("K" * 100)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=4096)
    final = p.read_text(encoding="utf-8")
    assert ("B" * 40) not in final
    assert "Bearer" in final
    assert ("H" * 100) in final and ("K" * 100) in final


def test_stream_spill_oversize_env_no_leak(monkeypatch, tmp_path):
    """An oversized KEY=value assignment is fully masked (env-dump command)."""
    value = "v" * (80 * 1024)
    content = ("Q" * 100) + "\nOPENAI_API_KEY=" + value + "\n" + ("W" * 100)
    p, _, _ = _stream_spill(
        monkeypatch, tmp_path, content, chunk=4096, command="env",
    )
    final = p.read_text(encoding="utf-8")
    assert ("v" * 40) not in final
    assert "OPENAI_API_KEY=" in final
    assert ("Q" * 100) in final and ("W" * 100) in final


def test_stream_spill_oversize_fail_closed_marker(monkeypatch, tmp_path):
    """A record longer than the confirm budget is replaced by a marker —
    its raw content is never partially emitted (fail-closed)."""
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_SPILL_CONFIRM_LIMIT_CHARS", 512)
    token = "sk-proj-" + "C" * 5000
    content = ("X" * 50) + " " + token + "\n" + ("Y" * 50)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=128)
    final = p.read_text(encoding="utf-8")
    assert "C" not in final
    assert "sk-proj" not in final
    assert "«redacted:secret…»" in final
    assert ("X" * 50) in final and ("Y" * 50) in final


def test_stream_spill_pem_across_chunks(monkeypatch, tmp_path):
    """A PEM block whose BEGIN/END markers span many chunks is replaced whole."""
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        + ("MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n" * 40)
        + "-----END PRIVATE KEY-----\n"
    )
    content = ("A" * 100) + "\n" + pem + ("B" * 100)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=7)
    final = p.read_text(encoding="utf-8")
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC" not in final
    assert "[REDACTED PRIVATE KEY]" in final
    assert ("A" * 100) in final and ("B" * 100) in final


def test_stream_spill_failure_cleans_temp_and_raises(monkeypatch, tmp_path):
    """A mid-stream redaction failure removes the temp file and re-raises."""
    import tools.terminal_tool as tt

    def _boom(text, command=None, **kw):
        raise RuntimeError("redaction exploded")

    monkeypatch.setattr("agent.redact.redact_terminal_output", _boom)
    # Content MUST contain a sensitive record so the confirm state actually
    # invokes redact_terminal_output mid-stream.
    p = _make_spill(tmp_path, "plain " * 20 + " sk-proj-" + "S" * 500 + " tail")
    dest = str(tmp_path / "dest.log")
    with pytest.raises(RuntimeError):
        tt._stream_redact_spill(str(p), dest, "echo hi")
    # Temp sanitized file removed; the raw source is left for the caller to
    # unlink (terminal_tool's except/finally does that).
    leftovers = [
        x for x in tmp_path.iterdir()
        if ".partial" in x.name or ".tmp-" in x.name
    ]
    assert leftovers == []
    assert p.exists()
    assert not (tmp_path / "dest.log").exists()


def test_stream_spill_replace_failure_cleans_temp(monkeypatch, tmp_path):
    """An os.replace failure leaves no partial file and no destination."""
    import os

    import tools.terminal_tool as tt

    def _boom_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("tools.terminal_tool.os.replace", _boom_replace)
    p = _make_spill(tmp_path, "S" * 500)
    dest = str(tmp_path / "dest.log")
    with pytest.raises(OSError):
        tt._stream_redact_spill(str(p), dest, "echo hi")
    assert not (tmp_path / "dest.log").exists()
    leftovers = [
        x for x in tmp_path.iterdir()
        if ".partial" in x.name or ".tmp-" in x.name
    ]
    assert leftovers == []


def test_elevated_spill_stream_redacted_cross_chunk(monkeypatch, tmp_path):
    """E2E: elevated raw output is sanitized into a NEW durable path.

    The staged raw file lives OUTSIDE the durable spill dir; after the call
    only the sanitized spill exists in the durable dir and the raw staging
    dir has been deleted.  A cross-chunk secret is fully masked.
    """
    from pathlib import Path

    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_get_env_config", lambda: _make_minimal_config())
    monkeypatch.setattr(tt, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        tt, "_check_all_guards", lambda cmd, env, **kw: {"approved": True},
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    secret = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0" * 3
    content = ("A" * (64 * 1024 - 50)) + " " + secret + "\n" + ("B" * 200_000)
    # Staged raw file: throwaway temp dir, never the durable spill dir.
    raw_dir = tmp_path / "raw-staging"
    raw_dir.mkdir()
    raw_file = raw_dir / "output.raw"
    raw_file.write_bytes(content.encode("utf-8"))

    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: {
            "output": content[:1000],
            "exit_code": 0,
            "error": None,
            "output_total_chars": len(content),
            "raw_output_path": str(raw_file),
        },
    )
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 1000)

    result = terminal_tool.terminal_tool(command="big", elevated=True)
    data = json.loads(result)

    assert data["exit_code"] == 0
    final_path = Path(data["full_output_path"])
    durable = tmp_path / "cache" / "terminal-output"
    assert final_path.parent == durable
    final = final_path.read_text(encoding="utf-8")
    assert secret not in final
    assert "A1b2C3d4E5f6G7h8I9j0A1b2C3d4E5f6G7h8I9j0" not in final
    assert ("A" * (64 * 1024 - 50)) in final
    assert ("B" * 200_000) in final
    assert data["output_total_chars"] == len(content)
    assert len(data["output"]) <= 1000
    # Raw staging dir removed; durable dir holds ONLY the sanitized spill.
    assert not raw_dir.exists(), "raw staged file must be deleted after sanitize"
    names = [x.name for x in durable.iterdir()]
    assert names == [final_path.name]


def test_elevated_spill_failure_drops_handle_and_unlinks_raw(monkeypatch, tmp_path):
    """If sanitization fails, the staged raw is removed and no durable spill
    (raw or partial) remains; no full_output_path is returned."""
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_get_env_config", lambda: _make_minimal_config())
    monkeypatch.setattr(tt, "_active_environments", {"default": _fake_env()})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        tt, "_check_all_guards", lambda cmd, env, **kw: {"approved": True},
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    raw_dir = tmp_path / "raw-staging"
    raw_dir.mkdir()
    raw_file = raw_dir / "output.raw"
    raw_file.write_bytes((("X" * 5000) + "\n").encode("utf-8"))

    def _boom(src_path, dest_path, command):
        raise RuntimeError("redaction exploded")

    monkeypatch.setattr(tt, "_stream_redact_spill", _boom)
    monkeypatch.setattr(
        "tools.admin_executor.execute_elevated",
        lambda command, cwd=None, timeout=120: {
            "output": "x" * 100,
            "exit_code": 0,
            "error": None,
            "output_total_chars": 5001,
            "raw_output_path": str(raw_file),
        },
    )

    result = terminal_tool.terminal_tool(command="big", elevated=True)
    data = json.loads(result)

    # Spill handle dropped: no spill metadata, no unredacted file anywhere.
    assert "full_output_path" not in data
    assert "output_total_chars" not in data
    assert not raw_dir.exists(), "staged raw must be cleaned on failure"
    durable = tmp_path / "cache" / "terminal-output"
    assert not durable.exists() or list(durable.iterdir()) == []
    leftovers = [
        x for x in tmp_path.iterdir()
        if ".partial" in x.name or ".tmp-" in x.name
    ]
    assert leftovers == []


# ---------------------------------------------------------------------------
# round-5: ANSI insertion bypass — stripping happens BEFORE redaction
# ---------------------------------------------------------------------------


def test_stream_spill_ansi_inserted_in_token_prefix(monkeypatch, tmp_path):
    """ANSI inserted between the token prefix and its body is stripped
    BEFORE redaction, so the reassembled secret is fully masked."""
    content = (
        ("A" * 50)
        + " sk-"
        + "\x1b[31m"
        + ("X" * 20)
        + ("Y" * 20)
        + "\x1b[0m\n"
        + ("B" * 50)
    )
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    assert ("X" * 20) not in final and ("Y" * 20) not in final
    assert ("A" * 50) in final and ("B" * 50) in final
    assert "\x1b" not in final


def test_stream_spill_ansi_inserted_in_token_body(monkeypatch, tmp_path):
    """ANSI inside a token body cannot split the secret."""
    content = ("C" * 30) + " ghp_" + ("D" * 10) + "\x1b[0m" + ("D" * 20) + "\n"
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    assert ("D" * 30) not in final and ("D" * 15) not in final
    assert "\x1b" not in final


def test_stream_spill_ansi_inserted_in_auth(monkeypatch, tmp_path):
    """ANSI inside an Authorization: Bearer value cannot split the token."""
    content = (
        ("H" * 20)
        + "\nAuthorization: Bearer "
        + ("E" * 10)
        + "\x1b[32m"
        + ("E" * 20)
        + "\x1b[0m\n"
        + ("K" * 20)
    )
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    assert ("E" * 30) not in final and ("E" * 15) not in final
    assert "Bearer" in final
    assert "\x1b" not in final


def test_stream_spill_ansi_inserted_in_env(monkeypatch, tmp_path):
    """ANSI inside a KEY=value assignment cannot split the secret (env dump)."""
    content = (
        ("Q" * 20)
        + "\nOPENAI_API_KEY="
        + ("v" * 10)
        + "\x1b[33m"
        + ("v" * 20)
        + "\x1b[0m\n"
        + ("W" * 20)
    )
    p, _, _ = _stream_spill(
        monkeypatch, tmp_path, content, chunk=8, command="env",
    )
    final = p.read_text(encoding="utf-8")
    assert ("v" * 30) not in final and ("v" * 15) not in final
    assert "OPENAI_API_KEY=" in final
    assert "\x1b" not in final


def test_stream_spill_ansi_inserted_in_pem_marker(monkeypatch, tmp_path):
    """ANSI inside the PEM BEGIN marker cannot hide the key block."""
    content = (
        ("Z" * 20)
        + "\n-----BEGIN "
        + "\x1b[35m"
        + "PRIVATE KEY"
        + "\x1b[0m"
        + "-----\nMIIE\n-----END PRIVATE KEY-----\n"
        + ("Y" * 20)
    )
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    assert "MIIE" not in final
    assert "[REDACTED PRIVATE KEY]" in final
    assert "\x1b" not in final


def test_stream_spill_ansi_csi_cross_chunk(monkeypatch, tmp_path):
    """A CSI sequence split across feed boundaries is stripped whole."""
    content = ("n" * 100) + "\x1b[3" + "1mRED\x1b[" + "0m" + ("m" * 100)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=4)
    final = p.read_text(encoding="utf-8")
    assert "\x1b" not in final
    assert "RED" in final
    assert ("n" * 100) in final and ("m" * 100) in final


def test_stream_spill_ansi_osc_cross_chunk(monkeypatch, tmp_path):
    """An OSC sequence (title string) split across boundaries is stripped."""
    content = ("o" * 50) + "\x1b]0;tit" + "le\x07" + ("p" * 50)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=5)
    final = p.read_text(encoding="utf-8")
    assert "\x1b" not in final
    assert "title" not in final
    assert ("o" * 50) in final and ("p" * 50) in final


# ---------------------------------------------------------------------------
# round-5: PEM EOF — unterminated blocks never leak body fragments
# ---------------------------------------------------------------------------


def test_stream_spill_pem_eof_unterminated_no_leak(monkeypatch, tmp_path):
    """A PEM block with no END marker at EOF yields exactly one marker —
    no head/middle/tail fragment of the key body survives."""
    pem_body = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
    content = ("W" * 20) + "\n" + pem_body
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=7)
    final = p.read_text(encoding="utf-8")
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC" not in final
    assert "MIIE" not in final
    assert "-----BEGIN" not in final
    assert final.count("[REDACTED PRIVATE KEY]") == 1
    assert ("W" * 20) in final


def test_stream_spill_pem_eof_truncated_end(monkeypatch, tmp_path):
    """A truncated END marker at EOF still drops the whole block."""
    content = (
        ("V" * 20)
        + "\n-----BEGIN PRIVATE KEY-----\nMIIEabc\n-----END PRIVATE\n"
    )
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=7)
    final = p.read_text(encoding="utf-8")
    assert "MIIEabc" not in final
    assert "[REDACTED PRIVATE KEY]" in final
    assert ("V" * 20) in final


def test_stream_spill_pem_multiple_blocks_then_eof(monkeypatch, tmp_path):
    """Multiple complete blocks followed by an unterminated block yield one
    marker per block (canonical parity) plus a fail-closed EOF marker."""
    content = (
        "-----BEGIN PRIVATE KEY-----\nMIIEone\n-----END PRIVATE KEY-----\n"
        "-----BEGIN PRIVATE KEY-----\nMIIEtwo\n-----END PRIVATE KEY-----\n"
        "-----BEGIN PRIVATE KEY-----\nMIIEthree"
    )
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=7)
    final = p.read_text(encoding="utf-8")
    assert "MIIEone" not in final
    assert "MIIEtwo" not in final
    assert "MIIEthree" not in final
    assert final.count("[REDACTED PRIVATE KEY]") == 3


def test_stream_spill_pem_eof_keeps_prefix_text(monkeypatch, tmp_path):
    """Text before an unterminated PEM block is preserved; only the block is
    dropped at EOF."""
    content = ("J" * 300) + "\n-----BEGIN PRIVATE KEY-----\n" + ("K" * 300)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    assert ("J" * 300) in final
    assert ("K" * 300) not in final
    assert "[REDACTED PRIVATE KEY]" in final


# ---------------------------------------------------------------------------
# round-5: form-urlencoded sensitive body keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sensitive_key,secret_value",
    [
        ("code", "ABC123"),
        ("session", "sess1"),
        ("signature", "sigval123"),
        ("x-amz-signature", "awsig123"),
        ("id_token", "zzz123"),
        ("token", "tok123"),
        ("access_token", "at123"),
        ("client_secret", "cs456"),
    ],
)
def test_stream_spill_form_body_sensitive_keys(
    monkeypatch, tmp_path, sensitive_key, secret_value,
):
    """form-urlencoded sensitive keys are masked; non-sensitive peers pass."""
    content = f"{sensitive_key}={secret_value}&state=xyz&next=abc"
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=4)
    final = p.read_text(encoding="utf-8")
    assert secret_value not in final
    assert "state=xyz" in final
    assert "next=abc" in final


def test_stream_spill_form_body_single_pair_not_masked(monkeypatch, tmp_path):
    """A single key=value pair is not a form body (canonical parity): the
    streaming path must not mask it either."""
    content = "code=ABC123"
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=4)
    final = p.read_text(encoding="utf-8")
    assert final == content


def test_stream_spill_form_body_multiline_is_safer(monkeypatch, tmp_path):
    """A form-body-shaped line inside a multi-line file is still masked by
    the streaming path (line-scoped), which is strictly safer than the
    canonical whole-text pass."""
    content = ("P" * 20) + "\ncode=ABC123&state=xyz\n" + ("R" * 20)
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    assert "ABC123" not in final


# ---------------------------------------------------------------------------
# round-5: durable spill is a force=True safety boundary
# ---------------------------------------------------------------------------


def test_stream_spill_force_true_with_redact_optout(monkeypatch, tmp_path):
    """Even with the global model-output redaction disabled, the durable
    spill is still sanitized (force=True on the persistent-file boundary)."""
    import agent.redact as R

    old = R._REDACT_ENABLED
    R._REDACT_ENABLED = False
    try:
        content = "sk-proj-" + ("F" * 30) + "\n"
        p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
        final = p.read_text(encoding="utf-8")
        assert ("F" * 30) not in final
        # The key label may survive in the mask head; the body must not.
        assert "sk-proj" not in final
    finally:
        R._REDACT_ENABLED = old


# ---------------------------------------------------------------------------
# round-5: canonical differential — stream == redact(strip_ansi(...), force)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,command",
    [
        ("plain text line with no secrets here\n", "echo hi"),
        ("sk-proj-" + ("A1b2C3d4E5f6G7h8I9j0" * 3) + "\n", "echo hi"),
        ("ghp_" + ("D" * 40) + "\n", "echo hi"),
        ("Authorization: Bearer abcdef0123456789abcdef\n", "curl x"),
        ("Proxy-Authorization: Basic dXNlcjpwYXNz\n", "curl x"),
        ("OPENAI_API_KEY=sk-secret1234567890abcd\n", "env"),
        ("MY_SERVICE_TOKEN=abc123randomstring\n", "env"),
        ('{"apiKey": "abc123def456", "token": "xyz"}\n', "cat config.json"),
        ("spring.datasource.password=hunter2secret\n", "cat application.properties"),
        ("password=secret123&user=alice\n", "env"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U\n", "echo hi"),
        ("bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALjAW\n", "echo hi"),
        ("+8613800138000\n", "echo hi"),
        ("postgresql://user:secretpass@db.example.com:5432/app\n", "echo hi"),
        ("https://user:secret@example.com/path\n", "echo hi"),
        ("git+ssh://git@github.com/repo.git\n", "echo hi"),
        ("code=ABC123&state=xyz&token=tok123", "echo hi"),
        ("session=sess1&signature=sigval123&state=xyz", "echo hi"),
        ("id_token=zzz123&client_secret=cs456&state=xyz", "echo hi"),
        ("x-amz-signature=awsig123&X-Amz-Date=20240101", "echo hi"),
        ("x-api-key: apikey1234567890\n", "curl x"),
        ("-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\n", "echo hi"),
        ("n" * 200 + "\x1b[31mRED\x1b[0m" + ("m" * 200) + "\n", "echo hi"),
        ("sk-" + "\x1b[31m" + ("X" * 20) + ("Y" * 20) + "\x1b[0m\n", "echo hi"),
        ("中文😀A测试🚀" * 50 + "\n", "echo hi"),
        ("A" * 5000 + "\n", "echo hi"),
    ],
)
def test_stream_spill_differential_canonical(monkeypatch, tmp_path, content, command):
    """Streaming spill output is byte-identical to the canonical
    ``redact_terminal_output(strip_ansi(content), command, force=True)``."""
    from agent.redact import redact_terminal_output
    from tools.ansi_strip import strip_ansi

    p, _, _ = _stream_spill(
        monkeypatch, tmp_path, content, chunk=8, command=command,
    )
    final = p.read_text(encoding="utf-8")
    canon = redact_terminal_output(strip_ansi(content), command, force=True)
    assert final == canon


def test_stream_spill_differential_canonical_oversize(monkeypatch, tmp_path):
    """A >64 KiB secret is buffered whole and masked byte-identically to the
    canonical pass.  Kept as a dedicated function because the parameterized
    id would exceed the Windows environment-variable limit for
    ``PYTEST_CURRENT_TEST``."""
    from agent.redact import redact_terminal_output
    from tools.ansi_strip import strip_ansi

    content = "sk-proj-" + ("B" * (80 * 1024)) + "\n"
    p, _, _ = _stream_spill(monkeypatch, tmp_path, content, chunk=8)
    final = p.read_text(encoding="utf-8")
    canon = redact_terminal_output(strip_ansi(content), "echo hi", force=True)
    assert final == canon
    assert ("B" * 40) not in final  # no middle segment survives
