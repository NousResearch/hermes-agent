"""Messaging-gateway turns must not go deaf behind long foreground tool calls.

While a foreground tool call runs, that chat session cannot answer new messages. A profile that
raises the general foreground cap (TERMINAL_MAX_FOREGROUND_TIMEOUT=3600, sensible for long CLI
builds) otherwise lets a model hold a Discord/Telegram session for an hour, e.g. with a
``for i in $(seq 1 80); do gh pr view N; sleep 45; done`` status poll.

Pinned here:

1. Messaging-gateway turns get their own foreground cap
   (``terminal.gateway_max_foreground_timeout``, default 600 s) that holds even when the general
   cap was raised; a longer call is promoted to a tracked background process (the same shape as
   the general cap). A long configured default timeout is clamped. CLI/cron turns are unchanged.
2. Foreground polling loops (for/while/until + ``sleep`` >= 30 s, or ``watch``) are refused in
   messaging-gateway turns, naming the fix.
3. Every tool call dispatched through the registry that is still outstanding after the watchdog
   interval logs ``PHASE=tool_wait_long`` (grep-able, no py-spy needed).
"""
from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

INCIDENT_LOOP = "for i in $(seq 1 80); do gh pr view 978 --json state; sleep 45; done"


def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _bind(**kw):
    from gateway.session_context import set_session_vars

    kw.setdefault("cron_session", "")
    return set_session_vars(**kw)


@pytest.fixture
def discord_turn():
    """Bind a real messaging-gateway session context (as gateway/run.py does per turn)."""
    from gateway.session_context import clear_session_vars

    tokens = _bind(platform="discord", source="discord", chat_id="c1",
                   session_key="agent:main:discord:dm:c1")
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def cli_turn():
    from gateway.session_context import clear_session_vars

    tokens = _bind(platform="", source="cli")
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def cron_discord_turn():
    """Cron binds a delivery platform but has no human waiting on the session."""
    from gateway.session_context import clear_session_vars

    tokens = _bind(platform="discord", source="discord", cron_session="1")
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def raised_general_cap(monkeypatch):
    """A profile that raised the general foreground cap to an hour."""
    monkeypatch.setattr("tools.terminal_tool.FOREGROUND_MAX_TIMEOUT", 3600)
    monkeypatch.delenv("TERMINAL_GATEWAY_MAX_FOREGROUND_TIMEOUT", raising=False)


def _plan(command="echo hi", *, timeout=None, background=False, config=None):
    from tools.terminal_tool import _plan_execution

    with patch("tools.terminal_tool._get_env_config", return_value=config or _make_env_config()):
        return _plan_execution(command, task_id=None, timeout=timeout, background=background,
                               _host_local=False)


def _run_terminal(**kwargs):
    """Drive terminal_tool with a mocked backend; return (result, mock_env). Gateway turns key their
    env by session, so env creation is mocked too: nothing here may reach a real shell."""
    from tools.terminal_tool import terminal_tool

    config = kwargs.pop("_config", None) or _make_env_config()
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "ok", "returncode": 0}
    with patch("tools.terminal_tool._get_env_config", return_value=config), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
         patch("tools.terminal_tool._create_configured_env", return_value=mock_env), \
         patch("tools.terminal_tool._last_activity", {"default": 0}), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        result = json.loads(terminal_tool(**kwargs))
    return result, mock_env


# ---------------------------------------------------------------------------
# (1) messaging-gateway foreground cap
# ---------------------------------------------------------------------------
class TestGatewayForegroundCap:
    def test_gateway_turn_promotes_above_gateway_cap(self, discord_turn, raised_general_cap):
        plan = _plan(timeout=3600)
        assert plan.promoted_from_foreground_timeout == 3600
        assert plan.promoted_cap == 600

    def test_gateway_turn_keeps_foreground_at_gateway_cap(self, discord_turn, raised_general_cap):
        plan = _plan(timeout=600)
        assert plan.promoted_from_foreground_timeout is None
        assert plan.effective_timeout == 600

    def test_cli_turn_keeps_the_general_cap(self, cli_turn, raised_general_cap):
        plan = _plan(timeout=3600)
        assert plan.promoted_from_foreground_timeout is None
        assert plan.effective_timeout == 3600

    def test_cron_turn_keeps_the_general_cap(self, cron_discord_turn, raised_general_cap):
        plan = _plan(timeout=3600)
        assert plan.promoted_from_foreground_timeout is None

    def test_gateway_cap_is_configurable(self, discord_turn, raised_general_cap, monkeypatch):
        monkeypatch.setenv("TERMINAL_GATEWAY_MAX_FOREGROUND_TIMEOUT", "900")
        assert _plan(timeout=900).promoted_from_foreground_timeout is None
        assert _plan(timeout=901).promoted_cap == 900

    def test_gateway_cap_zero_disables_it(self, discord_turn, raised_general_cap, monkeypatch):
        monkeypatch.setenv("TERMINAL_GATEWAY_MAX_FOREGROUND_TIMEOUT", "0")
        assert _plan(timeout=3600).promoted_from_foreground_timeout is None

    def test_gateway_cap_never_raises_the_general_cap(self, discord_turn, monkeypatch):
        monkeypatch.setattr("tools.terminal_tool.FOREGROUND_MAX_TIMEOUT", 300)
        monkeypatch.setenv("TERMINAL_GATEWAY_MAX_FOREGROUND_TIMEOUT", "900")
        plan = _plan(timeout=400)
        assert plan.promoted_from_foreground_timeout == 400
        assert plan.promoted_cap == 300

    def test_gateway_turn_clamps_a_long_configured_default(self, discord_turn, raised_general_cap):
        """terminal.timeout: 3600 must not smuggle an hour-long wait past the cap."""
        plan = _plan(config=_make_env_config(timeout=3600))
        assert plan.promoted_from_foreground_timeout is None
        assert plan.effective_timeout == 600

    def test_cli_turn_keeps_a_long_configured_default(self, cli_turn, raised_general_cap):
        plan = _plan(config=_make_env_config(timeout=900))
        assert plan.effective_timeout == 900

    def test_promotion_note_names_the_gateway_cap(self):
        from tools.terminal_tool import _with_promoted_note

        spawned = json.dumps({"output": "Background process started", "notify_on_complete": True})
        note = json.loads(_with_promoted_note(spawned, 3600, 600))["promoted_from_foreground"]
        assert "3600s" in note and "600s cap" in note

    def test_config_key_is_bridged_everywhere(self):
        from pathlib import Path

        # Read gateway/run.py as text: importing it runs the dependency bootstrap.
        run_src = (Path(__file__).resolve().parents[2] / "gateway" / "run.py").read_text()
        from hermes_cli.cli_config_load import _TERMINAL_ENV_MAPPINGS
        from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        env_var = "TERMINAL_GATEWAY_MAX_FOREGROUND_TIMEOUT"
        assert TERMINAL_CONFIG_ENV_MAP["gateway_max_foreground_timeout"] == env_var
        assert _TERMINAL_ENV_MAPPINGS["gateway_max_foreground_timeout"] == env_var
        assert f'"gateway_max_foreground_timeout": "{env_var}"' in run_src
        assert DEFAULT_CONFIG["terminal"]["gateway_max_foreground_timeout"] == 600

    def test_schema_advertises_the_gateway_cap(self):
        from tools.terminal_tool import TERMINAL_SCHEMA

        desc = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
        assert "gateway_max_foreground_timeout" in desc


# ---------------------------------------------------------------------------
# (2) polling-loop guard
# ---------------------------------------------------------------------------
LOOPS_REFUSED = [
    INCIDENT_LOOP,
    "until gh pr checks 978 | grep -q pass; do sleep 60; done",
    "while true; do curl -s localhost:8080/health; sleep 30; done",
    "for i in {1..40}; do ls /tmp/x && break; sleep 1m; done",
    "watch -n 60 gh pr view 978",
    "cd /tmp && watch 'ls -la'",
]
LOOPS_ALLOWED = [
    "for i in 1 2 3; do curl -s x; sleep 5; done",           # short sleeps
    "sleep 45; gh pr view 978",                               # one wait, no loop
    'git commit -m "for i in $(seq 1 9); do sleep 60; done"',  # quoted data
    "gh run watch 123 --exit-status",                         # bounded; the cap covers it
    "for f in *.py; do ruff check $f; done",                  # loop, no sleep
]


class TestPollingLoopGuard:
    @pytest.mark.parametrize("cmd", LOOPS_REFUSED)
    def test_detector_flags_polling_loops(self, cmd):
        from tools.terminal_tool_guards import _gateway_polling_loop_guidance

        msg = _gateway_polling_loop_guidance(cmd)
        assert msg, cmd
        assert "background=true" in msg

    @pytest.mark.parametrize("cmd", LOOPS_ALLOWED)
    def test_detector_leaves_ordinary_commands_alone(self, cmd):
        from tools.terminal_tool_guards import _gateway_polling_loop_guidance

        assert _gateway_polling_loop_guidance(cmd) is None, cmd

    def test_gateway_turn_refuses_the_incident_loop(self, discord_turn):
        result, env = _run_terminal(command=INCIDENT_LOOP, timeout=600)
        assert result.get("status") == "error"
        assert "polling loop" in result["error"]
        env.execute.assert_not_called()

    def test_gateway_turn_refuses_even_when_over_cap(self, discord_turn, raised_general_cap):
        """Refused, not promoted: promoting would keep the poll's sleeps alive in the background."""
        result, env = _run_terminal(command=INCIDENT_LOOP, timeout=3600)
        assert "polling loop" in result.get("error", "")
        env.execute.assert_not_called()

    def test_background_calls_skip_the_guard(self, discord_turn):
        with patch("tools.terminal_tool._gateway_polling_loop_guidance") as guard:
            _plan(INCIDENT_LOOP, background=True)
        guard.assert_not_called()

    def test_cli_turn_is_unchanged(self, cli_turn):
        result, env = _run_terminal(command=INCIDENT_LOOP, timeout=600)
        assert not result.get("error"), result
        env.execute.assert_called_once()


# ---------------------------------------------------------------------------
# (3) PHASE=tool_wait_long watchdog
# ---------------------------------------------------------------------------
@pytest.fixture
def fast_watchdog(monkeypatch):
    import tools.tool_wait_watchdog as wd

    monkeypatch.setattr(wd, "TOOL_WAIT_LONG_INTERVAL_S", 0.2)
    return wd


class TestToolWaitLongWatchdog:
    def test_outstanding_call_logs_phase_line(self, fast_watchdog, caplog):
        wd = fast_watchdog
        caplog.set_level(logging.WARNING, logger="tools.tool_wait_watchdog")
        with wd.track_tool_call("terminal", {"command": "sleep 45\n  gh pr view 978"}):
            time.sleep(0.75)
        lines = [r.getMessage() for r in caplog.records if "PHASE=tool_wait_long" in r.getMessage()]
        assert lines, caplog.text
        first = lines[0]
        assert "tool=terminal" in first
        assert "cmd='sleep 45 gh pr view 978'" in first
        assert "\n" not in first
        # repeats at every interval multiple, then a single done line
        assert len([ln for ln in lines if "tool_wait_long " in ln]) >= 2
        assert any("PHASE=tool_wait_long_done" in ln for ln in lines)

    def test_fast_call_logs_nothing(self, fast_watchdog, caplog):
        wd = fast_watchdog
        caplog.set_level(logging.WARNING, logger="tools.tool_wait_watchdog")
        with wd.track_tool_call("terminal", {"command": "echo hi"}):
            pass
        time.sleep(0.4)
        assert "PHASE=tool_wait_long" not in caplog.text
        assert wd.outstanding_count() == 0

    def test_long_cmd_is_truncated(self, fast_watchdog):
        wd = fast_watchdog
        desc = wd.describe_call("execute_code", {"code": "x = 1\n" * 500})
        assert len(desc) <= wd.MAX_CMD_CHARS + 1
        assert "\n" not in desc

    def test_registry_dispatch_is_tracked(self, fast_watchdog, caplog):
        """The watchdog is wired into ToolRegistry.dispatch, the path every tool call takes."""
        from tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(
            name="slow_probe", toolset="test_probe",
            schema={"name": "slow_probe", "description": "x", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: (time.sleep(0.7), json.dumps({"ok": True}))[1],
        )
        caplog.set_level(logging.WARNING, logger="tools.tool_wait_watchdog")
        out = reg.dispatch("slow_probe", {"command": "probe-marker"})
        assert "ok" in (out if isinstance(out, str) else json.dumps(out))
        assert any(
            "PHASE=tool_wait_long " in r.getMessage() and "tool=slow_probe" in r.getMessage()
            and "probe-marker" in r.getMessage()
            for r in caplog.records
        ), caplog.text
        assert fast_watchdog.outstanding_count() == 0
