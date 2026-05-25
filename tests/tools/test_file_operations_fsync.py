"""Regression coverage for #29786 — without a post-write fsync, an
interactive Hermes session that crashes (SSL ``[record layer
failure]``, parent SIGKILL, terminal closed) between
``ShellFileOperations.write_file`` returning success and the WSL2 9P
/ NFS / SMB / Docker bind-mount page cache flushing to physical
storage silently rolls back recent edits (e.g.
``session_summaries.md`` loses the last 2-3 entries).

These tests pin the contract from four angles:

* Helper purity — ``_is_fsync_disabled`` honours the truthy table
  (``1``/``true``/``yes``/``on`` + whitespace + case), and is False
  for everything else (empty, ``0``, ``no``, garbage).
* Wire-level — ``write_file`` plumbs a ``sync -f <escaped path>``
  shell command through to the terminal backend AFTER the
  ``cat > path`` and BEFORE the bytes-written probe.
* Patch tools — ``patch_replace`` (and by transitivity ``patch_v4a``)
  inherit the fsync because they delegate to ``write_file``.
* Opt-out — ``HERMES_DISABLE_FSYNC=1`` suppresses the sync entirely.
* Best-effort — a failing sync (non-zero exit, exception) must NOT
  turn a successful write into an error.
"""

from __future__ import annotations

import os
from typing import List, Optional
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper purity
# ---------------------------------------------------------------------------


class TestIsFsyncDisabledHelper:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Default + unset
            (None, False),
            ("", False),
            ("   ", False),
            ("\t\n", False),
            # Canonical truthy spellings
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("YES", True),
            ("on", True),
            ("On", True),
            # Whitespace tolerated
            ("  1  ", True),
            (" true ", True),
            ("\tyes\n", True),
            # Falsy spellings
            ("0", False),
            ("false", False),
            ("FALSE", False),
            ("no", False),
            ("off", False),
            # Garbage falls through to False (fail-safe — sync stays on)
            ("banana", False),
            ("2", False),
            ("maybe", False),
        ],
    )
    def test_truthy_table(self, monkeypatch, raw, expected):
        from tools.file_operations import _is_fsync_disabled
        if raw is None:
            monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        else:
            monkeypatch.setenv("HERMES_DISABLE_FSYNC", raw)
        assert _is_fsync_disabled() is expected


# ---------------------------------------------------------------------------
# Fake terminal backend — records every executed command and stdin
# ---------------------------------------------------------------------------


class _RecordingEnv:
    """Minimal terminal backend that records every executed command.

    Returns ``returncode=0`` and ``output=`` matched against
    ``responses`` (a list of ``(substring, response)`` pairs — first
    match wins; default if no match: empty string).
    """

    def __init__(self, responses: Optional[List[tuple]] = None):
        self.cwd = "/"
        self.calls: List[dict] = []
        self.responses = responses or []

    def execute(self, command: str, cwd: Optional[str] = None,
                stdin_data: Optional[str] = None, timeout: Optional[int] = None,
                **kw):
        self.calls.append({
            "command": command,
            "cwd": cwd,
            "stdin_data": stdin_data,
            "timeout": timeout,
        })
        for needle, resp in self.responses:
            if needle in command:
                return resp
        return {"output": "", "returncode": 0}

    def commands(self) -> List[str]:
        return [c["command"] for c in self.calls]

    def find(self, needle: str) -> List[dict]:
        return [c for c in self.calls if needle in c["command"]]


# ---------------------------------------------------------------------------
# Wire-level — ShellFileOperations.write_file emits sync after cat
# ---------------------------------------------------------------------------


class TestWriteFileEmitsSync:
    def _make(self, env):
        from tools.file_operations import ShellFileOperations
        ops = ShellFileOperations(env, cwd="/tmp")
        return ops

    def test_sync_command_is_issued_after_cat(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "report.md")
        env = _RecordingEnv(responses=[
            ("wc -c", {"output": "42\n", "returncode": 0}),
        ])
        ops = self._make(env)
        ops.write_file(target, "hello")

        commands = env.commands()
        cat_idx = next(
            (i for i, c in enumerate(commands)
             if c.startswith("cat > ") and target in c),
            None,
        )
        sync_idx = next(
            (i for i, c in enumerate(commands)
             if c.startswith("sync -f ") and target in c),
            None,
        )
        wc_idx = next(
            (i for i, c in enumerate(commands)
             if c.startswith("wc -c <") and target in c),
            None,
        )
        assert cat_idx is not None, f"no `cat > {target}` write in {commands}"
        assert sync_idx is not None, (
            f"no `sync -f {target}` after the write — fix regressed (#29786): "
            f"{commands}"
        )
        assert sync_idx > cat_idx, (
            "sync must come AFTER the write (otherwise it flushes nothing)"
        )
        if wc_idx is not None:
            assert sync_idx < wc_idx, (
                "sync should come BEFORE the bytes-written probe so the "
                "durable byte count is what we report to the agent"
            )

    def test_sync_command_falls_back_to_global_sync(self, monkeypatch, tmp_path):
        """``sync -f`` is Linux-only; macOS / BSD / non-GNU shells
        don't support it. The chain must end with ``|| sync`` so a
        backend without ``sync -f`` still flushes."""
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "log.md")
        env = _RecordingEnv()
        ops = self._make(env)
        ops.write_file(target, "x")

        sync_cmds = [c for c in env.commands() if c.startswith("sync -f ")]
        assert sync_cmds, "no sync command observed"
        assert sync_cmds[0].endswith("|| sync"), (
            f"sync chain must fall back to global `sync` for non-Linux "
            f"backends — got: {sync_cmds[0]!r}"
        )

    def test_sync_command_escapes_paths_with_spaces_and_quotes(
        self, monkeypatch, tmp_path
    ):
        """Path injection sanity — the sync command must single-quote
        the path so a filename like ``foo bar's file.md`` doesn't
        break out into a shell metachar."""
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "weird file's name.md")
        env = _RecordingEnv()
        ops = self._make(env)
        ops.write_file(target, "x")

        sync_cmds = [c for c in env.commands() if c.startswith("sync -f ")]
        assert sync_cmds
        # The first token after ``sync -f `` must be single-quoted
        # (the escape produces ``'…'`` with embedded quote escapes).
        body = sync_cmds[0][len("sync -f "):]
        assert body.startswith("'"), (
            f"unescaped path in sync chain — got: {sync_cmds[0]!r}"
        )

    def test_sync_command_has_bounded_timeout(self, monkeypatch, tmp_path):
        """A hung NFS / 9P round-trip shouldn't stall the whole agent.
        Verify the sync ``_exec`` is called with a finite timeout."""
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "x.md")
        env = _RecordingEnv()
        ops = self._make(env)
        ops.write_file(target, "x")

        sync_calls = env.find("sync -f")
        assert sync_calls
        timeout = sync_calls[0]["timeout"]
        assert timeout is not None and 0 < timeout <= 30, (
            f"sync must have a bounded timeout — got {timeout!r}"
        )

    def test_sync_skipped_when_env_var_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_DISABLE_FSYNC", "1")
        target = str(tmp_path / "y.md")
        env = _RecordingEnv()
        ops = self._make(env)
        ops.write_file(target, "x")
        assert not env.find("sync -f"), (
            "HERMES_DISABLE_FSYNC=1 must suppress the sync entirely"
        )
        assert not [c for c in env.commands() if c == "sync"], (
            "HERMES_DISABLE_FSYNC=1 must suppress the fallback sync too"
        )

    def test_sync_skipped_on_failed_write(self, monkeypatch, tmp_path):
        """If the ``cat > path`` write itself failed, the sync is
        pointless (no new bytes to flush). The fix returns the error
        before reaching the sync."""
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "z.md")
        env = _RecordingEnv(responses=[
            ("cat > ", {"output": "denied", "returncode": 1}),
        ])
        ops = self._make(env)
        result = ops.write_file(target, "x")
        assert result.error
        assert not env.find("sync -f"), (
            "failed writes must not trigger the post-write sync"
        )

    def test_sync_failure_does_not_break_write(self, monkeypatch, tmp_path):
        """Best-effort contract: a sync that exits non-zero (e.g.
        ``sync`` not on PATH at all, broken backend) must NOT turn
        the successful write into an error."""
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "best_effort.md")
        env = _RecordingEnv(responses=[
            ("sync -f", {"output": "sync: command not found",
                         "returncode": 127}),
            ("wc -c", {"output": "1\n", "returncode": 0}),
        ])
        ops = self._make(env)
        result = ops.write_file(target, "x")
        assert result.error is None, (
            f"failing sync must not surface as a write error — got "
            f"{result.error!r}"
        )

    def test_sync_exception_does_not_break_write(self, monkeypatch, tmp_path):
        """If the backend raises (e.g. SSH disconnected mid-call), the
        sync must swallow the exception and let the write report
        success — the bytes already left our process."""
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        from tools.file_operations import ShellFileOperations

        target = str(tmp_path / "x.md")
        env = _RecordingEnv()
        ops = ShellFileOperations(env, cwd="/tmp")

        real_exec = ops._exec

        def maybe_raise(cmd, *a, **kw):
            if cmd.startswith("sync -f"):
                raise RuntimeError("backend SSH dropped")
            return real_exec(cmd, *a, **kw)

        with patch.object(ops, "_exec", side_effect=maybe_raise):
            result = ops.write_file(target, "x")
        assert result.error is None, (
            f"sync exception must not surface as a write error — got "
            f"{result.error!r}"
        )


# ---------------------------------------------------------------------------
# patch_replace inherits the fsync via its delegation to write_file
# ---------------------------------------------------------------------------


class TestPatchReplaceInheritsSync:
    def _make_with_initial(self, env, target: str, body: str):
        """Pre-seed the recording env so that the ``cat <file>`` read
        at the start of ``patch_replace`` sees ``body``."""
        env.responses.append((f"cat '{target}'", {"output": body, "returncode": 0}))
        env.responses.append(("wc -c", {"output": str(len(body) + 4) + "\n", "returncode": 0}))
        from tools.file_operations import ShellFileOperations
        return ShellFileOperations(env, cwd="/tmp")

    def test_sync_runs_for_patch_replace(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        target = str(tmp_path / "doc.md")
        env = _RecordingEnv()
        ops = self._make_with_initial(env, target, "hello world\n")
        ops.patch_replace(target, "hello", "HELLO")
        assert env.find("sync -f"), (
            "patch_replace must inherit the fsync via its delegation "
            "to write_file (#29786)"
        )

    def test_sync_skipped_for_patch_replace_when_disabled(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_DISABLE_FSYNC", "true")
        target = str(tmp_path / "doc.md")
        env = _RecordingEnv()
        ops = self._make_with_initial(env, target, "hello world\n")
        ops.patch_replace(target, "hello", "HELLO")
        assert not env.find("sync -f")


# ---------------------------------------------------------------------------
# _sync_path helper — direct contract tests
# ---------------------------------------------------------------------------


class TestSyncPathHelper:
    def _make(self, env):
        from tools.file_operations import ShellFileOperations
        return ShellFileOperations(env, cwd="/tmp")

    def test_returns_none_quietly_on_success(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        env = _RecordingEnv()
        ops = self._make(env)
        assert ops._sync_path(str(tmp_path / "a.md")) is None

    def test_short_circuits_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_DISABLE_FSYNC", "yes")
        env = _RecordingEnv()
        ops = self._make(env)
        ops._sync_path(str(tmp_path / "a.md"))
        assert env.calls == [], (
            "disabled fsync must not even reach the backend"
        )

    def test_swallows_backend_exception(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_DISABLE_FSYNC", raising=False)
        env = _RecordingEnv()
        ops = self._make(env)
        with patch.object(
            ops, "_exec", side_effect=RuntimeError("backend died")
        ):
            ops._sync_path(str(tmp_path / "a.md"))  # must not raise


# ---------------------------------------------------------------------------
# Source guardrail — prevent silent regression
# ---------------------------------------------------------------------------


class TestSourceGuardrail:
    @pytest.fixture
    def source(self) -> str:
        from pathlib import Path
        path = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "file_operations.py"
        )
        return path.read_text(encoding="utf-8")

    def test_sync_path_helper_defined(self, source):
        assert "def _sync_path(" in source

    def test_sync_path_called_from_write_file(self, source):
        assert "self._sync_path(path)" in source, (
            "write_file must call self._sync_path after the cat > write "
            "(#29786) — otherwise patch_replace / patch_v4a also lose "
            "the durability guarantee"
        )

    def test_sync_path_uses_sync_minus_f_fallback(self, source):
        """The exact ``sync -f ... || sync`` shape is load-bearing for
        macOS / non-GNU coreutils backends."""
        assert "sync -f " in source
        assert "|| sync" in source

    def test_opt_out_env_var_is_named_correctly(self, source):
        assert "HERMES_DISABLE_FSYNC" in source

    def test_helper_uses_5_second_timeout(self, source):
        """Catch a future refactor that drops the bounded timeout —
        an unbounded sync on a hung NFS mount would stall every
        write_file forever."""
        assert "timeout=5" in source
