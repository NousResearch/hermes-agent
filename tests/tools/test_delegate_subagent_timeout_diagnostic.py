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

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

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

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
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


# ── #17308: N-API-call timeout structured diagnostics ─────────────────────

class _StubChildWithMessages(_StubChild):
    """_StubChild + a `_session_messages` view + configurable `current_tool`.

    Mirrors the real AIAgent: ``_session_messages`` is the chat-completions
    transcript that the assistant writes to as it works, and
    ``get_activity_summary()['current_tool']`` reflects whatever tool the
    main loop is currently dispatching.
    """

    def __init__(
        self,
        *,
        session_messages=None,
        current_tool=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session_messages = list(session_messages or [])
        self._current_tool = current_tool

    def get_activity_summary(self):
        s = super().get_activity_summary()
        s["current_tool"] = self._current_tool
        return s


class TestRunSingleChildTimeoutToolTrace:
    """#17308: when a subagent times out *after* making API calls, the result
    must surface ``tool_trace``, ``last_tool``, ``last_tool_status``, and
    ``current_tool`` so the lead agent can tell 'tool finished, next LLM call
    stuck' from 'tool itself hung'."""

    def _invoke(self, child, monkeypatch):
        from tools import delegate_tool
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

    def test_timeout_after_completed_tool_marks_status_ok(self, hermes_home, monkeypatch):
        """Tool finished cleanly, then the next LLM request hung. The trace
        must show ``status=ok`` on the last tool, and ``current_tool`` must
        be None — the lead agent knows the suspect is the LLM call."""
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": '{"q":"x"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "results: ..."},
        ]
        child = _StubChildWithMessages(
            api_call_count=2,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)

        assert result["status"] == "timeout"
        assert result["api_calls"] == 2
        assert result["last_tool"] == "web_search"
        assert result["last_tool_status"] == "ok"
        assert result["current_tool"] is None
        assert isinstance(result["tool_trace"], list) and len(result["tool_trace"]) == 1
        entry = result["tool_trace"][0]
        assert entry["tool"] == "web_search"
        assert entry["status"] == "ok"
        assert entry["result_bytes"] > 0
        # Error message references the last tool for the lead agent's eyes.
        assert "last_tool=web_search" in result["error"]
        assert "status=ok" in result["error"]
        # 0-API-call diagnostic dump must NOT fire on this branch.
        assert result.get("diagnostic_path") is None

    def test_timeout_inside_running_tool_marks_status_in_progress(
        self, hermes_home, monkeypatch
    ):
        """Assistant invoked a tool but no tool-role response was written
        before the timeout fired — the tool itself is hung. Status must be
        ``in_progress`` so the lead agent suspects the tool, not the LLM."""
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": '{"cmd":"sleep 9999"}'}},
                ],
            },
            # No tool-role reply yet — the tool is still running.
        ]
        child = _StubChildWithMessages(
            api_call_count=1,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool="terminal",
        )
        result = self._invoke(child, monkeypatch)

        assert result["status"] == "timeout"
        assert result["last_tool"] == "terminal"
        assert result["last_tool_status"] == "in_progress"
        assert result["current_tool"] == "terminal"
        assert result["tool_trace"][-1]["status"] == "in_progress"
        assert "last_tool=terminal" in result["error"]
        assert "status=in_progress" in result["error"]

    def test_timeout_with_tool_error_preserves_error_status(self, hermes_home, monkeypatch):
        """A tool returned an error, then the next LLM call hung. The trace
        must keep ``status=error`` on that entry — don't lie and call it ok."""
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Error: rate limit"},
        ]
        child = _StubChildWithMessages(
            api_call_count=1,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)
        assert result["last_tool"] == "web_search"
        assert result["last_tool_status"] == "error"
        assert result["tool_trace"][-1]["status"] == "error"

    def test_timeout_with_parallel_tool_calls_pairs_by_id(self, hermes_home, monkeypatch):
        """Parallel tool_calls must be paired by tool_call_id, not order."""
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "function": {"name": "web_search", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            # Replies arrive in reverse order — must still pair correctly.
            {"role": "tool", "tool_call_id": "b", "content": "shell ok"},
            {"role": "tool", "tool_call_id": "a", "content": "search ok"},
        ]
        child = _StubChildWithMessages(
            api_call_count=1,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)
        trace = result["tool_trace"]
        assert [e["tool"] for e in trace] == ["web_search", "terminal"]
        # web_search entry got the matching 'a' content; terminal got 'b'.
        assert all(e["status"] == "ok" for e in trace)
        # last_tool is the trailing assistant call — terminal in this case.
        assert result["last_tool"] == "terminal"

    def test_timeout_content_block_result_not_false_error(self, hermes_home, monkeypatch):
        """Content-block list results must flatten before classification.

        Mirror normal-path coverage in ``test_delegate.py``: a benign JSON body
        wrapped as OpenAI content blocks should stay ``status=ok``.
        """
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "image_generate",
                            "arguments": '{"prompt": "x"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": '{"success": true}'}],
            },
        ]
        child = _StubChildWithMessages(
            api_call_count=1,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)
        assert result["status"] == "timeout"
        assert result["last_tool"] == "image_generate"
        assert result["last_tool_status"] == "ok"
        assert result["tool_trace"][0]["status"] == "ok"
        assert result["tool_trace"][0]["result_bytes"] > 0

    def test_timeout_benign_output_containing_error_word_is_ok(
        self, hermes_home, monkeypatch
    ):
        """Conservative classifier: the substring ``error`` alone is not enough.

        Mirrors ``test_delegate.py`` normal-path intent — e.g. log lines like
        ``no error detected`` must not paint the tool red.
        """
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "scan complete: no error detected in module",
            },
        ]
        child = _StubChildWithMessages(
            api_call_count=1,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)
        assert result["last_tool"] == "terminal"
        assert result["last_tool_status"] == "ok"
        assert result["tool_trace"][-1]["status"] == "ok"

    def test_timeout_content_block_real_error_is_flagged(self, hermes_home, monkeypatch):
        """Block-wrapped classic error markers still classify as error."""
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "Error: command not found"}],
            },
        ]
        child = _StubChildWithMessages(
            api_call_count=1,
            hang_seconds=10.0,
            session_messages=msgs,
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)
        assert result["last_tool"] == "terminal"
        assert result["last_tool_status"] == "error"
        assert result["tool_trace"][-1]["status"] == "error"

    def test_zero_api_call_timeout_skips_tool_trace(self, hermes_home, monkeypatch):
        """0-API-call timeouts are covered by diagnostic_path (#15105). The
        new tool_trace fields must be empty/None on that branch — not stale."""
        child = _StubChildWithMessages(
            api_call_count=0,
            hang_seconds=10.0,
            session_messages=[{"role": "user", "content": "x"}],
            current_tool=None,
        )
        result = self._invoke(child, monkeypatch)
        assert result["status"] == "timeout"
        assert result["api_calls"] == 0
        assert result["diagnostic_path"] is not None
        assert result["tool_trace"] == []
        assert result["last_tool"] is None
        assert result["last_tool_status"] is None
        assert result["current_tool"] is None

    def test_timeout_with_no_session_messages_attr_does_not_crash(
        self, hermes_home, monkeypatch
    ):
        """Some agent shapes (mocks, exotic providers) won't expose
        _session_messages. Reconstruction must degrade to an empty trace."""
        child = _StubChild(api_call_count=2, hang_seconds=10.0)
        # Note: plain _StubChild does NOT define _session_messages.
        assert not hasattr(child, "_session_messages")
        result = self._invoke(child, monkeypatch)
        assert result["status"] == "timeout"
        assert result["api_calls"] == 2
        assert result["tool_trace"] == []
        assert result["last_tool"] is None
        assert result["last_tool_status"] is None
        # Error message stays the legacy 'stuck on a slow API call' — no last_tool=
        # suffix because we have nothing reliable to report.
        assert "stuck on a slow API call" in result["error"]
        assert "last_tool=" not in result["error"]


# ── _build_tool_trace_from_messages helper ────────────────────────────

class TestBuildToolTraceFromMessages:
    """Pin the shared helper used by normal completion + N-API-call timeouts.

    Error detection uses the conservative classifier, not substring matching.
    """

    def test_handles_non_list_input(self):
        from tools.delegate_tool import _build_tool_trace_from_messages
        assert _build_tool_trace_from_messages(None) == []
        assert _build_tool_trace_from_messages("not a list") == []
        assert _build_tool_trace_from_messages({"role": "user"}) == []

    def test_skips_non_dict_entries(self):
        from tools.delegate_tool import _build_tool_trace_from_messages
        msgs = [
            "junk",
            None,
            {"role": "assistant", "tool_calls": [
                {"id": "x", "function": {"name": "t", "arguments": "{}"}}
            ]},
        ]
        out = _build_tool_trace_from_messages(msgs)
        assert len(out) == 1
        assert out[0]["tool"] == "t"

    def test_assistant_with_no_tool_calls_is_ignored(self):
        from tools.delegate_tool import _build_tool_trace_from_messages
        msgs = [{"role": "assistant", "content": "hello"}]
        assert _build_tool_trace_from_messages(msgs) == []

    def test_tool_response_without_call_id_falls_back_to_last_entry(self):
        from tools.delegate_tool import _build_tool_trace_from_messages
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": None, "function": {"name": "web_search", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "ok"},
        ]
        out = _build_tool_trace_from_messages(msgs)
        assert out == [
            {
                "tool": "web_search",
                "args_bytes": 2,
                "result_bytes": 2,
                "status": "ok",
            }
        ]

    def test_content_blocks_and_benign_error_substring(self):
        from tools.delegate_tool import _build_tool_trace_from_messages
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "image_generate", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "text", "text": '{"success": true}'}],
            },
            {
                "role": "tool",
                "tool_call_id": "c2",
                "content": "no error detected",
            },
        ]
        out = _build_tool_trace_from_messages(msgs)
        assert out[0]["status"] == "ok"
        assert out[1]["status"] == "ok"
        assert out[0]["result_bytes"] > 0

