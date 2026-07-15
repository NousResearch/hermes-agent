"""Kanban worker approval policy must hold on the REAL worker path.

A worker is launched by ``hermes_cli.kanban_db._default_spawn`` as
``hermes -p <profile> --cli --accept-hooks ... chat -q "work kanban task <id>"``
with ``stdin=DEVNULL``. Two production facts that hand-scrubbed unit tests
miss (#55946 review, P1/P2):

* The child re-marks itself interactive at startup — ``cli.main()`` exports
  ``HERMES_INTERACTIVE=1`` even in quiet worker mode — so any approval branch
  nested under ``not _is_interactive_cli()`` is unreachable in a real worker.
* The dispatcher usually runs embedded in a gateway (or cron-tainted)
  process whose ambient markers (``HERMES_GATEWAY_SESSION``,
  ``HERMES_EXEC_ASK``, ``HERMES_CRON_SESSION``, ``HERMES_SESSION_PLATFORM``,
  ``HERMES_SESSION_KEY``, ``HERMES_YOLO_MODE``, ``HERMES_INTERACTIVE``) leak
  into the child through the full-env copy (#63183).

These tests therefore build the worker env with the REAL ``_default_spawn``
(``Popen`` intercepted), replay the child startup self-mark, and drive the
real gate entry points — no hand-curated env dicts, no marker deletion.
"""

from __future__ import annotations

import os
import subprocess

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    check_execute_code_guard,
    request_tool_approval,
)

DANGEROUS_CMD = "rm -rf /tmp/stuff"

# Ambient approval-context markers a dispatching parent process realistically
# exports (tui_gateway/server.py, gateway/run.py, cron/scheduler.py) and that
# `env = dict(os.environ)` in _default_spawn would copy into the child.
AMBIENT_PARENT_MARKERS = {
    "HERMES_INTERACTIVE": "1",        # tui_gateway _enable_gateway_prompts
    "HERMES_GATEWAY_SESSION": "1",    # tui_gateway _enable_gateway_prompts
    "HERMES_EXEC_ASK": "1",           # gateway/run.py (module import)
    "HERMES_CRON_SESSION": "1",       # cron/scheduler.py run_job
    "HERMES_SESSION_PLATFORM": "discord",
    "HERMES_SESSION_KEY": "discord:12345",
    "HERMES_YOLO_MODE": "1",          # parent launched with --yolo
}

# Every env marker the approval gates consult.
GATE_MARKER_VARS = ("HERMES_KANBAN_SESSION", *AMBIENT_PARENT_MARKERS)


def _make_task(kb):
    return kb.Task(
        id="t_real_path",
        title="real path approval",
        body=None,
        assignee="elias",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def _spawn_worker_env(monkeypatch, tmp_path, parent_markers: dict) -> dict:
    """Return the EXACT env a spawned worker receives.

    Seeds the parent process env with *parent_markers*, then runs the real
    ``_default_spawn`` with ``subprocess.Popen`` intercepted and returns the
    env it passed to ``Popen``.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var, value in parent_markers.items():
        monkeypatch.setenv(var, value)

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(_make_task(kb), str(workspace))
    return captured["env"]


def _enter_worker_process(monkeypatch, worker_env: dict) -> None:
    """Make this process's approval-relevant env match the spawned worker's.

    Applies exactly the marker state ``_default_spawn`` produced (present
    vars set, absent vars unset — never a hand-scrubbed approximation), then
    replays the one mutation the child performs at startup: ``cli.main()``
    exports ``HERMES_INTERACTIVE=1``. The replay is unconditional because
    ``_resolve_hermes_argv`` may pick up an older ``hermes`` build on PATH /
    ``$HERMES_BIN`` that still self-marks unconditionally — the gates must
    hold either way.
    """
    for var in GATE_MARKER_VARS:
        if var in worker_env:
            monkeypatch.setenv(var, worker_env[var])
        else:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("discord:12345")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("discord:12345")


@pytest.fixture
def human_surfaces(monkeypatch):
    """Spy on every path that would hand the decision to an absent human.

    A headless worker (stdin=DEVNULL, detached session) must never be routed
    to an interactive prompt, the smart-approval aux LLM, or a gateway
    pending queue that lives in the dispatcher process (#63183 wedge).
    """
    calls = {"prompt": 0, "smart": 0, "pending": 0}

    def fake_prompt(*args, **kwargs):
        calls["prompt"] += 1
        return "deny"

    def fake_smart(*args, **kwargs):
        calls["smart"] += 1
        return "escalate"

    def fake_pending(*args, **kwargs):
        calls["pending"] += 1

    monkeypatch.setattr(approval_module, "prompt_dangerous_approval", fake_prompt)
    monkeypatch.setattr(approval_module, "_smart_approve", fake_smart)
    monkeypatch.setattr(approval_module, "submit_pending", fake_pending)
    return calls


def _assert_no_human_surface_touched(calls):
    assert calls["prompt"] == 0, "worker was routed to an interactive prompt"
    assert calls["smart"] == 0, "worker was routed to smart (aux-LLM) approval"
    assert calls["pending"] == 0, "worker submitted a pending approval nobody can resolve"


# ---------------------------------------------------------------------------
# P2: spawn-env hygiene — ambient parent markers must not reach the worker
# ---------------------------------------------------------------------------

class TestSpawnEnvHygiene:
    def test_ambient_approval_markers_are_scrubbed(self, monkeypatch, tmp_path):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        assert env.get("HERMES_KANBAN_SESSION") == "1"
        assert env.get("HERMES_KANBAN_TASK") == "t_real_path"
        for var in AMBIENT_PARENT_MARKERS:
            assert var not in env, f"{var} leaked from the dispatcher into the worker env"

    def test_non_approval_parent_env_is_preserved(self, monkeypatch, tmp_path):
        """Sanitization is a denylist of approval markers, not an env rebuild."""
        env = _spawn_worker_env(
            monkeypatch, tmp_path,
            {**AMBIENT_PARENT_MARKERS, "SOME_PARENT_VAR": "keep-me"},
        )
        assert env.get("SOME_PARENT_VAR") == "keep-me"
        assert "PATH" in env


# ---------------------------------------------------------------------------
# P1: terminal gate (check_all_command_guards — the live terminal_tool gate)
# ---------------------------------------------------------------------------

class TestRealWorkerTerminalGate:
    def test_kanban_deny_blocks_despite_interactive_self_mark(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_all_command_guards(DANGEROUS_CMD, "local")

        assert result["approved"] is False
        assert "kanban_mode" in (result.get("message") or ""), (
            "worker must be denied by kanban POLICY (actionable remediation), "
            f"not by an accidental prompt failure — got: {result!r}"
        )
        _assert_no_human_surface_touched(human_surfaces)

    def test_safe_command_still_runs_under_kanban_deny(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_all_command_guards("echo hello", "local")

        assert result["approved"] is True
        _assert_no_human_surface_touched(human_surfaces)

    def test_kanban_approve_mode_is_honored_without_any_prompt(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        """approvals.kanban_mode: approve is the operator's explicit trust
        knob — it must actually take effect in a real worker instead of the
        command wedging in a prompt/pending queue."""
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "approve")

        result = check_all_command_guards(DANGEROUS_CMD, "local")

        assert result["approved"] is True
        _assert_no_human_surface_touched(human_surfaces)

    def test_inherited_cron_approve_cannot_preempt_kanban_deny(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        """A worker dispatched from a cron-tainted parent must get KANBAN
        policy: even if the cron marker reaches the child (older dispatcher,
        other spawner), cron_mode: approve must not override kanban deny."""
        env = _spawn_worker_env(
            monkeypatch, tmp_path, {"HERMES_CRON_SESSION": "1"}
        )
        _enter_worker_process(monkeypatch, env)
        # Simulate the marker surviving into the child regardless of spawn
        # scrubbing — gate precedence must be enough on its own.
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setattr(approval_module, "_get_cron_approval_mode", lambda: "approve")
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_all_command_guards(DANGEROUS_CMD, "local")

        assert result["approved"] is False
        assert "kanban_mode" in (result.get("message") or "")
        _assert_no_human_surface_touched(human_surfaces)


# ---------------------------------------------------------------------------
# P1: dangerous-command entry point (shared _run_approval_gate)
# ---------------------------------------------------------------------------

class TestRealWorkerDangerousCommandGate:
    def test_kanban_deny_blocks_despite_ambient_markers(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_dangerous_command(DANGEROUS_CMD, "local")

        assert result["approved"] is False
        assert "kanban_mode" in (result.get("message") or "")
        _assert_no_human_surface_touched(human_surfaces)

    def test_gateway_markers_do_not_reroute_worker_into_pending_queue(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        """#63183: a worker that self-identifies as gateway/ask submits a
        child-local pending approval that gateway /approve can never resolve.
        With kanban identity ranked first, that dead end is unreachable even
        when the markers are present in the worker process."""
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        # Force the gateway/ask markers back on top of the spawned env:
        # precedence must hold even when scrubbing didn't happen.
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_dangerous_command(DANGEROUS_CMD, "local")

        assert result["approved"] is False
        assert result.get("status") not in ("approval_required", "pending_approval")
        assert "kanban_mode" in (result.get("message") or "")
        _assert_no_human_surface_touched(human_surfaces)


# ---------------------------------------------------------------------------
# P1: plugin escalation entry point (request_tool_approval, fail-closed path)
# ---------------------------------------------------------------------------

class TestRealWorkerPluginGate:
    def test_kanban_deny_blocks_plugin_escalation(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = request_tool_approval("write_file", "writes to ~/.ssh")

        assert result["approved"] is False
        assert "kanban_mode" in (result.get("message") or "")
        _assert_no_human_surface_touched(human_surfaces)

    def test_kanban_approve_mode_allows_plugin_escalation(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "approve")

        result = request_tool_approval("write_file", "writes to ~/.ssh")

        assert result["approved"] is True
        _assert_no_human_surface_touched(human_surfaces)


# ---------------------------------------------------------------------------
# P2: execute_code gate — kanban identity must outrank an inherited cron marker
# ---------------------------------------------------------------------------

class TestRealWorkerExecuteCodeGate:
    def test_inherited_cron_approve_cannot_preempt_kanban_deny(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, {"HERMES_CRON_SESSION": "1"})
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setattr(approval_module, "_get_cron_approval_mode", lambda: "approve")
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_execute_code_guard("import os", "local")

        assert result["approved"] is False
        assert result.get("outcome") == "blocked"
        assert "kanban" in (result.get("message") or "").lower()
        _assert_no_human_surface_touched(human_surfaces)

    def test_kanban_deny_blocks_with_full_ambient_markers(
        self, monkeypatch, tmp_path, human_surfaces
    ):
        env = _spawn_worker_env(monkeypatch, tmp_path, AMBIENT_PARENT_MARKERS)
        _enter_worker_process(monkeypatch, env)
        monkeypatch.setattr(approval_module, "_get_cron_approval_mode", lambda: "approve")
        monkeypatch.setattr(approval_module, "_get_kanban_approval_mode", lambda: "deny")

        result = check_execute_code_guard("import os", "local")

        assert result["approved"] is False
        _assert_no_human_surface_touched(human_surfaces)


# ---------------------------------------------------------------------------
# Child startup: quiet kanban workers must not self-mark as interactive
# ---------------------------------------------------------------------------

class TestWorkerStartupSelfMark:
    def test_kanban_worker_startup_does_not_export_interactive(self, monkeypatch):
        import cli

        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        cli._export_interactive_marker()
        assert os.environ.get("HERMES_INTERACTIVE") is None

    def test_regular_cli_startup_still_exports_interactive(self, monkeypatch):
        import cli

        monkeypatch.delenv("HERMES_KANBAN_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        cli._export_interactive_marker()
        assert os.environ.get("HERMES_INTERACTIVE") == "1"
