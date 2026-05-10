"""Tests for the Agent Status Panel."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_process(**overrides):
    """Build a process entry as it appears in collect_status() output."""
    base = {
        "name": "pytest -v",
        "type": "Process",
        "status": "running",
        "started_at": "2026-05-06T18:00:00",
        "elapsed": 42,
        "output_preview": "test_foo PASSED\ntest_bar PASSED",
        "session_id": "proc_abc123",
        "pid": 12345,
        "exit_code": None,
    }
    base.update(overrides)
    return base


def _make_cron_job(**overrides):
    """Build a cron job entry as it appears in collect_status() output."""
    base = {
        "name": "Daily Summary",
        "type": "Cron",
        "status": "scheduled",
        "schedule": "every day at 09:00",
        "next_run_at": "2026-05-07T09:00:00",
        "last_run_at": "2026-05-06T09:00:00",
        "last_status": "ok",
        "output_preview": "",
    }
    base.update(overrides)
    return base


def _make_subagent(**overrides):
    """Build a subagent entry as it appears in collect_status() output."""
    base = {
        "name": "Write unit tests",
        "type": "Subagent",
        "status": "running",
        "started_at": time.time() - 120,
        "elapsed": 120,
        "model": "openai/gpt-5.4",
        "subagent_id": "sub_001",
        "depth": 1,
        "output_preview": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: collect_status
# ---------------------------------------------------------------------------

class TestCollectStatus:
    """Tests for collect_status()."""

    def test_returns_all_keys(self):
        from tools.agent_status_panel import collect_status

        data = collect_status(session_id="s1", model="gpt-5", provider="openai")
        assert "session_id" in data
        assert "model" in data
        assert "provider" in data
        assert "title" in data
        assert "processes" in data
        assert "cron_jobs" in data
        assert "subagents" in data
        assert "gateway_agents" in data

    def test_metadata_passthrough(self):
        from tools.agent_status_panel import collect_status

        data = collect_status(
            session_id="sess-42",
            model="claude-4",
            provider="anthropic",
            title="My Task",
        )
        assert data["session_id"] == "sess-42"
        assert data["model"] == "claude-4"
        assert data["provider"] == "anthropic"
        assert data["title"] == "My Task"

    @patch("tools.agent_status_panel._collect_background_processes", return_value=[])
    @patch("tools.agent_status_panel._collect_cron_jobs", return_value=[])
    @patch("tools.agent_status_panel._collect_subagents", return_value=[])
    def test_empty_by_default(self, _sub, _cron, _proc):
        from tools.agent_status_panel import collect_status

        data = collect_status()
        assert data["processes"] == []
        assert data["cron_jobs"] == []
        assert data["subagents"] == []
        assert data["gateway_agents"] == []


# ---------------------------------------------------------------------------
# Tests: format_status_panel
# ---------------------------------------------------------------------------

class TestFormatStatusPanel:
    """Tests for format_status_panel()."""

    def test_empty_shows_friendly_message(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [],
            "cron_jobs": [],
            "subagents": [],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "No active tasks" in text
        assert "Agent Status Panel" in text

    def test_shows_session_metadata(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "abc-123",
            "model": "gpt-5.4",
            "provider": "openai",
            "title": "Debug session",
            "processes": [],
            "cron_jobs": [],
            "subagents": [],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "abc-123" in text
        assert "gpt-5.4" in text
        assert "openai" in text
        assert "Debug session" in text

    def test_shows_running_processes(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [_make_process(status="running")],
            "cron_jobs": [],
            "subagents": [],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "Running Processes" in text
        assert "pytest -v" in text
        assert "🔄" in text

    def test_shows_finished_processes_with_exit_code(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [
                _make_process(status="exited", exit_code=0, uptime_seconds=10),
                _make_process(
                    status="exited", exit_code=1, uptime_seconds=5,
                    command="failing-test", session_id="proc_fail1",
                ),
            ],
            "cron_jobs": [],
            "subagents": [],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "Recent Processes" in text
        assert "✅" in text  # exit 0
        assert "❌" in text  # exit 1

    def test_shows_cron_jobs(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [],
            "cron_jobs": [_make_cron_job()],
            "subagents": [],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "Cron Jobs" in text
        assert "Daily Summary" in text
        assert "📅" in text  # scheduled

    def test_shows_cron_job_with_error(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [],
            "cron_jobs": [_make_cron_job(
                status="error",
                last_status="error",
                output_preview="Timeout after 300s",
            )],
            "subagents": [],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "❌" in text
        assert "Timeout after 300s" in text

    def test_shows_subagents(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [],
            "cron_jobs": [],
            "subagents": [_make_subagent()],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        assert "Sub-agents" in text
        assert "Write unit tests" in text
        assert "🔄" in text

    def test_shows_gateway_agents(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [],
            "cron_jobs": [],
            "subagents": [],
            "gateway_agents": [{
                "name": "telegram:user123",
                "type": "Gateway Agent",
                "status": "running",
                "started_at": time.time() - 300,
                "elapsed": 300,
                "model": "gpt-5.4",
                "session_id": "gw-sess-1",
                "output_preview": "",
            }],
        }
        text = format_status_panel(data)
        assert "Gateway Agents" in text
        assert "telegram:user123" in text
        assert "5m 0s" in text

    def test_combined_panel_has_all_sections(self):
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "full-sess",
            "model": "claude-4",
            "provider": "anthropic",
            "title": "Full Test",
            "processes": [_make_process()],
            "cron_jobs": [_make_cron_job()],
            "subagents": [_make_subagent()],
            "gateway_agents": [{
                "name": "discord:ch1",
                "type": "Gateway Agent",
                "status": "starting",
                "started_at": time.time(),
                "elapsed": 0,
                "model": "",
                "session_id": "",
                "output_preview": "",
            }],
        }
        text = format_status_panel(data)
        assert "Agent Status Panel" in text
        assert "full-sess" in text
        assert "Gateway Agents" in text
        assert "Sub-agents" in text
        assert "Running Processes" in text
        assert "Cron Jobs" in text
        assert "⏳" in text  # starting emoji

    def test_truncates_long_goal(self):
        from tools.agent_status_panel import format_status_panel

        long_name = "A" * 100
        data = {
            "session_id": "",
            "model": "",
            "provider": "",
            "title": "",
            "processes": [],
            "cron_jobs": [],
            "subagents": [_make_subagent(name=long_name)],
            "gateway_agents": [],
        }
        text = format_status_panel(data)
        # Name should be truncated to ~40 chars
        assert "AAAA" in text
        assert "…" in text

    def test_output_is_mobile_friendly(self):
        """Ensure no individual line exceeds a reasonable width."""
        from tools.agent_status_panel import format_status_panel

        data = {
            "session_id": "s",
            "model": "m",
            "provider": "p",
            "title": "t",
            "processes": [_make_process()],
            "cron_jobs": [_make_cron_job()],
            "subagents": [_make_subagent()],
            "gateway_agents": [{
                "name": "tg:user",
                "type": "Gateway Agent",
                "status": "running",
                "started_at": time.time() - 3600,
                "elapsed": 3600,
                "model": "gpt-5.4",
                "session_id": "gw-1",
                "output_preview": "",
            }],
        }
        text = format_status_panel(data)
        for line in text.split("\n"):
            # Most mobile screens handle ~50-60 chars; allow some flex
            # for markdown bold markers
            assert len(line) <= 120, f"Line too long ({len(line)} chars): {line[:80]}"


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper functions."""

    def test_truncate_short_text(self):
        from tools.agent_status_panel import _truncate
        assert _truncate("hello", 10) == "hello"

    def test_truncate_long_text(self):
        from tools.agent_status_panel import _truncate
        result = _truncate("a" * 60, 50)
        assert len(result) == 51  # 50 chars + "…"
        assert result.endswith("…")

    def test_fmt_elapsed_seconds(self):
        from tools.agent_status_panel import _fmt_elapsed
        assert _fmt_elapsed(30) == "30s"

    def test_fmt_elapsed_minutes(self):
        from tools.agent_status_panel import _fmt_elapsed
        assert _fmt_elapsed(125) == "2m 5s"

    def test_fmt_elapsed_hours(self):
        from tools.agent_status_panel import _fmt_elapsed
        assert _fmt_elapsed(3661) == "1h 1m"

    def test_status_emoji_running(self):
        from tools.agent_status_panel import _status_emoji
        assert _status_emoji("running") == "🔄"
        assert _status_emoji("active") == "🔄"

    def test_status_emoji_ok(self):
        from tools.agent_status_panel import _status_emoji
        assert _status_emoji("ok") == "✅"
        assert _status_emoji("completed") == "✅"

    def test_status_emoji_error(self):
        from tools.agent_status_panel import _status_emoji
        assert _status_emoji("error") == "❌"
        assert _status_emoji("failed") == "❌"

    def test_status_emoji_paused(self):
        from tools.agent_status_panel import _status_emoji
        assert _status_emoji("paused") == "⏸️"

    def test_status_emoji_scheduled(self):
        from tools.agent_status_panel import _status_emoji
        assert _status_emoji("scheduled") == "📅"

    def test_status_emoji_unknown(self):
        from tools.agent_status_panel import _status_emoji
        assert _status_emoji("something_else") == "▪️"


# ---------------------------------------------------------------------------
# Tests: collection resilience
# ---------------------------------------------------------------------------

class TestCollectionResilience:
    """Ensure collectors don't crash when subsystems are unavailable."""

    def test_process_collection_handles_import_error(self):
        from tools.agent_status_panel import _collect_background_processes

        with patch.dict("sys.modules", {"tools.process_registry": None}):
            result = _collect_background_processes()
            assert result == []

    def test_cron_collection_handles_import_error(self):
        from tools.agent_status_panel import _collect_cron_jobs

        with patch.dict("sys.modules", {"cron.jobs": None}):
            result = _collect_cron_jobs()
            assert result == []

    def test_subagent_collection_handles_import_error(self):
        from tools.agent_status_panel import _collect_subagents

        with patch.dict("sys.modules", {"tools.delegate_tool": None}):
            result = _collect_subagents()
            assert result == []

    def test_gateway_collection_returns_empty_for_none(self):
        from tools.agent_status_panel import _collect_gateway_agents

        assert _collect_gateway_agents(None) == []

    def test_gateway_collection_handles_broken_gateway(self):
        from tools.agent_status_panel import _collect_gateway_agents

        broken = SimpleNamespace()  # no _running_agents attr
        result = _collect_gateway_agents(broken)
        assert result == []
