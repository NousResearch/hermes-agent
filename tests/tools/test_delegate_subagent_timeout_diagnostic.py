"""Regression tests for subagent timeout diagnostic dump (issue #14726).

When delegate_task's child subagent times out without having made any API
call, a structured diagnostic file is written under
``~/.hermes/logs/subagent-timeout-<sid>-<ts>.log``. This gives users a
concrete artifact to inspect (worker thread stack, system prompt size,
tool schema bytes, credential pool state, etc.) instead of the previous
opaque "subagent timed out" error.

These tests pin:
- the diagnostic writer's output format and content
- the timeout branch in _run_single_child only dumps when api_calls == 0
- the error message surfaces the diagnostic path
- api_calls > 0 timeouts do NOT write a dump (the old "stuck on slow API
  call" explanation still applies)
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class _StubChild:
    """Minimal stand-in for an AIAgent subagent."""
    def __init__(
        self,
        *,
        api_call_count: int = 0,
        hang_seconds: float = 5.0,
        subagent_id: str = "sa-0-stubabc",
        tool_schema=None,
    ):
        self._subagent_id = subagent_id
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self.model = "test/model"
        self.provider = "testprov"
        self.api_mode = "chat_completions"
        self.base_url = "https://example.test/v1"
        self.max_iterations = 30
        self.quiet_mode = True
        self.skip_memory = True
        self.skip_context_files = True
        self.platform = "cli"
        self.ephemeral_system_prompt = "sys prompt"
        self.enabled_toolsets = ["web", "terminal"]
        self.valid_tool_names = {"web_search", "terminal"}
        self.tools = tool_schema if tool_schema is not None else [
            {"name": "web_search", "description": "search"},
            {"name": "terminal", "description": "shell"},
        ]
        self._api_call_count = api_call_count
        self._hang = threading.Event()
        self._hang_seconds = hang_seconds

    def get_activity_summary(self):
        return {
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "current_tool": None,
            "seconds_since_activity": 60,
        }

    def run_conversation(self, user_message, task_id=None):
        self._hang.wait(self._hang_seconds)
        return {"final_response": "", "completed": False, "api_calls": self._api_call_count}

    def interrupt(self):
        self._hang.set()


# ── _dump_subagent_timeout_diagnostic ──────────────────────────────────

class TestDumpSubagentTimeoutDiagnostic:

    def test_writes_log_with_expected_sections(self, hermes_home):
        from tools.delegate_tool import _dump_subagent_timeout_diagnostic
        child = _StubChild(subagent_id="sa-7-abc123")

        worker = threading.Thread(
            target=lambda: child.run_conversation("test"),
            daemon=True,
        )
        worker.start()
        time.sleep(0.1)
        try:
            path = _dump_subagent_timeout_diagnostic(
                child=child,
                task_index=7,
                timeout_seconds=300.0,
                duration_seconds=300.01,
                worker_thread=worker,
                goal="Research something long",
            )
        finally:
            child.interrupt()
            worker.join(timeout=2.0)

        assert path is not None
        p = Path(path)
        assert p.is_file()
        # File lives under HERMES_HOME/logs/
        assert p.parent == hermes_home / "logs"
        assert p.name.startswith("subagent-timeout-sa-7-abc123-")
        assert p.suffix == ".log"

        content = p.read_text()
        # Header references the issue for future grep-ability
        assert "issue #14726" in content
        # Timeout facts
        assert "task_index:        7" in content
        assert "subagent_id:       sa-7-abc123" in content
        assert "configured_timeout: 300.0s" in content
        assert "actual_duration:   300.01s" in content
        # Goal
        assert "Research something long" in content
        # Child config
        assert "model: 'test/model'" in content
        assert "provider: 'testprov'" in content
        assert "base_url: 'https://example.test/v1'" in content
        assert "max_iterations: 30" in content
        # Toolsets
        assert "enabled_toolsets:  ['web', 'terminal']" in content
        assert "loaded tool count: 2" in content
        # Prompt / schema sizes
        assert "system_prompt_bytes:" in content
        assert "tool_schema_count: 2" in content
        assert "tool_schema_bytes:" in content
        # Activity summary
        assert "api_call_count: 0" in content
        # Worker stack
        assert "Worker thread stack at timeout" in content
        # The thread is parked inside _hang.wait → cond.wait → waiter.acquire
        assert "acquire" in content or "wait" in content

    def test_truncates_very_long_goal(self, hermes_home):
        from tools.delegate_tool import _dump_subagent_timeout_diagnostic
        child = _StubChild()
        huge_goal = "x" * 5000

        path = _dump_subagent_timeout_diagnostic(
            child=child,
            task_index=0,
            timeout_seconds=300.0,
            duration_seconds=300.0,
            worker_thread=None,
            goal=huge_goal,
        )
        child.interrupt()

        content = Path(path).read_text()
        assert "[truncated]" in content
        # Goal section trimmed to 1000 chars + suffix
        goal_block = content.split("## Goal", 1)[1].split("## Child config", 1)[0]
        assert len(goal_block) < 1200

    def test_missing_worker_thread_is_handled(self, hermes_home):
        from tools.delegate_tool import _dump_subagent_timeout_diagnostic
        child = _StubChild()
        path = _dump_subagent_timeout_diagnostic(
            child=child,
            task_index=0,
            timeout_seconds=300.0,
            duration_seconds=300.0,
            worker_thread=None,
            goal="x",
        )
        child.interrupt()
        content = Path(path).read_text()
        assert "<no worker thread handle>" in content

    def test_exited_worker_thread_is_handled(self, hermes_home):
        from tools.delegate_tool import _dump_subagent_timeout_diagnostic
        child = _StubChild()
        # A thread that has already finished
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        assert not t.is_alive()
        path = _dump_subagent_timeout_diagnostic(
            child=child,
            task_index=0,
            timeout_seconds=300.0,
            duration_seconds=300.0,
            worker_thread=t,
            goal="x",
        )
        child.interrupt()
        content = Path(path).read_text()
        assert "<worker thread already exited>" in content

    def test_returns_none_on_unwritable_logs_dir(self, tmp_path, monkeypatch):
        # Point HERMES_HOME at an unwritable path so logs/ can't be created
        # (simulates permission-denied). Helper must not raise.
        from tools.delegate_tool import _dump_subagent_timeout_diagnostic
        bogus = tmp_path / "does-not-exist" / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(bogus))
        child = _StubChild()

        # Make the logs dir itself unwritable by creating it as a FILE
        # so mkdir(exist_ok=True) → NotADirectoryError and we fall through.
        bogus.parent.mkdir(parents=True, exist_ok=True)
        bogus.mkdir()
        (bogus / "logs").write_text("not a dir")
        result = _dump_subagent_timeout_diagnostic(
            child=child,
            task_index=0,
            timeout_seconds=300.0,
            duration_seconds=300.0,
            worker_thread=None,
            goal="x",
        )
        child.interrupt()
        # Either None (mkdir failed) or a real path; must never raise.
        # We assert no exception propagates — the return value is advisory.
        assert result is None or Path(result).exists()


# ── _run_single_child timeout branch wiring ───────────────────────────

class TestRunSingleChildTimeoutDump:
    """The timeout branch in _run_single_child must emit the diagnostic
    dump when api_calls == 0, and must NOT emit it when api_calls > 0."""

    def _invoke_with_short_timeout(self, child, monkeypatch):
        """Run _run_single_child with a tiny timeout to force the timeout branch."""
        from tools import delegate_tool
        # Force a 0.3s timeout so the test is fast
        monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.3)

        parent = MagicMock()
        parent._touch_activity = MagicMock()
        parent._current_task_id = None
        return delegate_tool._run_single_child(
            task_index=0,
            goal="test goal",
            child=child,
            parent_agent=parent,
        )

    def test_zero_api_calls_writes_dump_and_surfaces_path(self, hermes_home, monkeypatch):
        child = _StubChild(api_call_count=0, hang_seconds=10.0)
        result = self._invoke_with_short_timeout(child, monkeypatch)

        assert result["status"] == "timeout"
        assert result["api_calls"] == 0
        assert result["diagnostic_path"] is not None
        dump_path = Path(result["diagnostic_path"])
        assert dump_path.is_file()
        assert dump_path.parent == hermes_home / "logs"

        # Error message surfaces the path and the "no API call" phrasing
        assert "without making any API call" in result["error"]
        assert "Diagnostic:" in result["error"]
        assert str(dump_path) in result["error"]

    def test_nonzero_api_calls_skips_dump_and_uses_old_message(self, hermes_home, monkeypatch):
        child = _StubChild(api_call_count=5, hang_seconds=10.0)
        result = self._invoke_with_short_timeout(child, monkeypatch)

        assert result["status"] == "timeout"
        assert result["api_calls"] == 5
        # No diagnostic file should be written for timeouts that made
        # actual API calls — the old generic "stuck on slow call" message
        # still applies.
        assert result.get("diagnostic_path") is None
        assert "stuck on a slow API call" in result["error"]
        # And no subagent-timeout-* file should exist under logs/
        logs_dir = hermes_home / "logs"
        if logs_dir.is_dir():
            dumps = list(logs_dir.glob("subagent-timeout-*.log"))
            assert dumps == []
