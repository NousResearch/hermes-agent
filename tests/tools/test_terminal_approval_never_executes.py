"""Regression tests for #130890: approval-blocked terminal commands must never execute.

Bug: a terminal command requiring interactive approval was reported back as
blocked / not executed ('user has NOT consented') but had already executed and
produced side effects. The agent trusted the block message and reported that
nothing had changed.

Contract pinned here:
- Approval runs BEFORE any environment acquisition, pre-exec script reads,
  spawn, or execute. A denied/timed-out/cancelled command creates no env,
  spawns no subprocess/shell, and leaves no side effects.
- Denial/timeout/cancel aborts completely: nothing is dispatched or retried
  in the background.
- Approved commands still run (foreground + background); force=True still
  bypasses the dangerous-command check but not the unconditional hard blocks.

E2E style per tools/AGENTS.md: drive the REAL terminal_tool with real guards
against a temp HERMES_HOME where possible; unit-ordering tests mock only the
seams AFTER the gate to prove they are never reached.
"""

import json
from types import SimpleNamespace

import pytest

import tools.terminal_tool as tt
import tools.approval as approval_module
from tools import approval_context


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\n"
        "approvals:\n  mode: manual\n  timeout: 5\n  cron_mode: deny\n"
        "command_allowlist: []\n"
        "security:\n  tirith_enabled: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture
def cli_manual_context(monkeypatch):
    """Interactive CLI approval context: a human CAN answer via callback."""
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    # The test runner itself runs under a single-query kanban worker
    # (HERMES_SINGLE_QUERY_SESSION=1 outer env); clear it so the gate takes
    # the interactive CLI path instead of the unattended deny path.
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", "test-130890-session")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _cmd: {"action": "allow", "findings": [], "summary": ""},
    )
    approval_module.clear_session("test-130890-session")
    approval_module._permanent_approved.clear()
    tt.set_approval_callback(None)
    yield
    tt.set_approval_callback(None)
    approval_module.clear_session("test-130890-session")


def _fake_plan():
    return SimpleNamespace(
        config={},
        env_type="local",
        effective_task_id="130890-test",
        image="",
        cwd="/tmp",
        host_cwd=None,
        effective_timeout=30,
        promoted_from_foreground_timeout=None,
    )


def _denied_guard(message="BLOCKED: User denied this command."):
    return {
        "approved": False,
        "message": message,
        "pattern_key": "test-pattern",
        "description": "test description",
        "outcome": "denied",
        "user_consent": False,
        "user_summary": "Denied.",
    }


class TestDeniedNeverAcquiresNorExecutes:
    @pytest.mark.parametrize("background", [False, True])
    def test_deny_blocks_before_env_acquisition(self, monkeypatch, background):
        """Denied guard => no env, no pre-exec, no spawn, no foreground run."""
        calls = []
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())
        monkeypatch.setattr(
            tt, "_check_all_guards", lambda *_a, **_k: _denied_guard()
        )

        def _fail_acquire(*_a, **_k):
            calls.append("acquire")
            raise AssertionError("denied command must not acquire an environment")

        def _fail_pre_exec(*_a, **_k):
            calls.append("pre_exec")
            raise AssertionError("denied command must not run pre-exec guards")

        def _fail_foreground(*_a, **_k):
            calls.append("foreground")
            raise AssertionError("denied command must not run in foreground")

        def _fail_spawn(**_k):
            calls.append("background")
            raise AssertionError("denied command must not spawn background work")

        monkeypatch.setattr(tt, "_acquire_env", _fail_acquire)
        monkeypatch.setattr(tt, "_pre_exec_block", _fail_pre_exec)
        monkeypatch.setattr(tt, "_run_foreground", _fail_foreground)
        monkeypatch.setattr("tools.terminal_tool.spawn_background_process", _fail_spawn)

        result = json.loads(tt.terminal_tool("rm -rf /tmp/victim", background=background))
        assert result["status"] == "blocked"
        assert "NOT consented" in result["error"] or "denied" in result["error"].lower()
        assert calls == [], f"side-effecting stages ran for a denied command: {calls}"

    @pytest.mark.parametrize(
        "guard_result",
        [
            {"approved": False, "message": "BLOCKED: timed out", "pattern_key": "k",
             "description": "d", "outcome": "timeout", "user_consent": False,
             "user_summary": "Timed out."},
            {"approved": False, "message": "BLOCKED: cancelled", "pattern_key": "k",
             "description": "d", "outcome": "cancelled", "user_consent": False,
             "user_summary": "Cancelled."},
            {"approved": False, "message": "BLOCKED: notify failed"},
            {"approved": False, "message": "BLOCKED: unattended"},
        ],
        ids=["timeout", "cancelled", "notify_failed", "unattended_block"],
    )
    def test_all_non_consent_outcomes_never_execute(self, monkeypatch, guard_result):
        """Timeout / cancel / notify-failure / unattended block: same no-execution contract."""
        calls = []
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())
        monkeypatch.setattr(tt, "_check_all_guards", lambda *_a, **_k: dict(guard_result))
        monkeypatch.setattr(
            tt, "_acquire_env",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("non-consented command must not acquire env")),
        )
        monkeypatch.setattr(
            tt, "_run_foreground",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("non-consented command must not execute")),
        )
        monkeypatch.setattr(
            "tools.terminal_tool.spawn_background_process",
            lambda **_k: (_ for _ in ()).throw(
                AssertionError("non-consented command must not spawn")),
        )
        # Foreground and background both abort.
        for background in (False, True):
            result = json.loads(tt.terminal_tool("rm -rf /tmp/victim", background=background))
            assert result["status"] == "blocked", guard_result
            assert result["error"], guard_result

    def test_pending_approval_never_executes(self, monkeypatch):
        """Gateway ask-mode pending_approval is not consent: no execution."""
        calls = []
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda *_a, **_k: {
                "approved": False, "status": "pending_approval",
                "approval_pending": True, "command": "rm -rf /tmp/victim",
                "description": "recursive delete", "pattern_key": "k",
            },
        )
        monkeypatch.setattr(
            tt, "_acquire_env",
            lambda *_a, **_k: calls.append("acquire") or object(),
        )
        monkeypatch.setattr(
            tt, "_run_foreground",
            lambda *_a, **_k: calls.append("foreground") or "ran",
        )
        result = json.loads(tt.terminal_tool("rm -rf /tmp/victim"))
        assert result.get("status") == "pending_approval"
        assert result.get("approval_pending") is True
        assert calls == [], f"pending approval must not execute: {calls}"

    def test_approval_runs_before_env_acquisition(self, monkeypatch):
        """Call-order proof: the guard verdict lands before any env work."""
        order = []
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())

        def _guard(*_a, **_k):
            order.append("guard")
            return _denied_guard()

        def _acquire(*_a, **_k):
            order.append("acquire")
            return object()

        monkeypatch.setattr(tt, "_check_all_guards", _guard)
        monkeypatch.setattr(tt, "_acquire_env", _acquire)
        result = json.loads(tt.terminal_tool("rm -rf /tmp/victim"))
        assert result["status"] == "blocked"
        assert order == ["guard"], f"acquire ran despite denial: {order}"


class TestApprovedStillRuns:
    def test_approved_foreground_runs(self, monkeypatch):
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())
        monkeypatch.setattr(tt, "_check_all_guards", lambda *_a, **_k: {"approved": True})
        monkeypatch.setattr(tt, "_acquire_env", lambda *_a, **_k: object())
        monkeypatch.setattr(tt, "_pre_exec_block", lambda *_a, **_k: None)
        monkeypatch.setattr(
            tt, "_run_foreground", lambda *_a, **_k: json.dumps({"output": "ok", "exit_code": 0})
        )
        result = json.loads(tt.terminal_tool("echo ok"))
        assert result["exit_code"] == 0

    def test_approved_background_spawns(self, monkeypatch):
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())
        monkeypatch.setattr(tt, "_check_all_guards", lambda *_a, **_k: {"approved": True})
        monkeypatch.setattr(tt, "_acquire_env", lambda *_a, **_k: object())
        monkeypatch.setattr(tt, "_pre_exec_block", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "tools.terminal_tool.spawn_background_process",
            lambda **_k: json.dumps({"output": "Background process started", "exit_code": 0}),
        )
        result = json.loads(tt.terminal_tool("sleep 30", background=True))
        assert result["exit_code"] == 0

    def test_force_bypass_still_runs_without_prompt(self, monkeypatch):
        """force=True skips the dangerous-command check (user already confirmed)."""
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())

        def _no_prompt(*_a, **_k):
            raise AssertionError("force=True must not consult the approval gate")

        monkeypatch.setattr(tt, "_check_all_guards", _no_prompt)
        monkeypatch.setattr(tt, "_acquire_env", lambda *_a, **_k: object())
        monkeypatch.setattr(tt, "_pre_exec_block", lambda *_a, **_k: None)
        monkeypatch.setattr(
            tt, "_run_foreground", lambda *_a, **_k: json.dumps({"output": "ok", "exit_code": 0})
        )
        result = json.loads(tt.terminal_tool("echo ok", force=True))
        assert result["exit_code"] == 0


class TestEndToEndNoSideEffects:
    @pytest.mark.parametrize("background", [False, True])
    def test_denied_rm_rf_leaves_victim_intact(
        self, isolated_home, cli_manual_context, tmp_path, monkeypatch, background
    ):
        """Real guards + real tool: denying `rm -rf <dir>` leaves it intact.

        This is the reported bug shape: the tool result says blocked/not-run,
        and the filesystem must agree — no side effects.
        """
        from hermes_cli import config as hc
        hc._LOAD_CONFIG_CACHE.clear()
        monkeypatch.setattr(
            approval_context, "_get_approval_timeout", lambda: 5
        )
        tt.set_approval_callback(lambda _cmd, _desc, **_kw: "deny")
        try:
            victim = tmp_path / "victim-dir"
            victim.mkdir()
            (victim / "precious.txt").write_text("do not delete")
            sentinel = victim / "precious.txt"
            assert sentinel.exists()

            result = json.loads(
                tt.terminal_tool(f"rm -rf {victim}", background=background, task_id="e2e-130890")
            )
            assert result.get("status") in ("blocked", "error"), result
            err = result.get("error", "")
            assert "NOT consented" in err or "denied" in err.lower(), result
            # The command did NOT run: side effects absent.
            assert victim.exists(), "denied rm -rf executed anyway — side effects occurred"
            assert sentinel.exists()
            assert sentinel.read_text() == "do not delete"
        finally:
            tt.set_approval_callback(None)
            hc._LOAD_CONFIG_CACHE.clear()

    def test_timeout_leaves_victim_intact(
        self, isolated_home, cli_manual_context, tmp_path, monkeypatch
    ):
        """Approval timeout is not consent: no execution, no side effects."""
        from hermes_cli import config as hc
        hc._LOAD_CONFIG_CACHE.clear()
        tt.set_approval_callback(lambda _cmd, _desc, **_kw: "timeout")
        try:
            victim = tmp_path / "timeout-victim"
            victim.mkdir()
            (victim / "keep.txt").write_text("keep me")
            result = json.loads(tt.terminal_tool(f"rm -rf {victim}", task_id="e2e-130890-timeout"))
            assert result.get("status") == "blocked", result
            assert "NOT consented" in result.get("error", "") or "timed out" in result.get(
                "error", "").lower(), result
            assert victim.exists(), "timed-out command executed anyway"
        finally:
            tt.set_approval_callback(None)
            hc._LOAD_CONFIG_CACHE.clear()

    def test_denied_never_retried_in_background(
        self, isolated_home, cli_manual_context, tmp_path, monkeypatch
    ):
        """A denial must not dispatch or retry in the background: no session, no spawn."""
        from hermes_cli import config as hc
        hc._LOAD_CONFIG_CACHE.clear()
        tt.set_approval_callback(lambda _cmd, _desc, **_kw: "deny")
        spawns = []
        import tools.terminal_tool_background as bg
        real_spawn = bg._spawn
        monkeypatch.setattr(
            bg, "_spawn",
            lambda *a, **k: (spawns.append((a, k)), real_spawn(*a, **k))[1],
        )
        try:
            victim = tmp_path / "retry-victim"
            victim.mkdir()
            result = json.loads(
                tt.terminal_tool(
                    f"rm -rf {victim}", background=True, notify_on_complete=True,
                    task_id="e2e-130890-noretry",
                )
            )
            assert result.get("status") == "blocked", result
            assert spawns == [], f"denied command spawned background work: {spawns}"
            assert victim.exists()
        finally:
            tt.set_approval_callback(None)
            hc._LOAD_CONFIG_CACHE.clear()


class TestFailurePathErrorContract:
    """Failure SEAMS still surface the public error envelope through the real tool.

    force=True skips only the approval gate; planning, env acquisition and the
    foreground run are real. Only the failing seam is stubbed — no raw
    exception may escape terminal_tool(), and the monitoring wrapper
    (record_terminal_backend) still sees a well-formed envelope.
    """

    def test_acquire_env_failure_returns_error_envelope(self, monkeypatch):
        """_create_configured_env raising (daemon down) => fatal error envelope."""
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())

        def _boom(*_a, **_k):
            raise RuntimeError("docker daemon unreachable (connection refused)")

        monkeypatch.setattr(tt, "_create_configured_env", _boom)
        result = json.loads(tt.terminal_tool("echo ok", force=True, task_id="e2e-130890-acquire-fail"))
        assert result["status"] == "error"
        assert result["exit_code"] == -1
        assert "Failed to execute command" in result["error"]
        assert "docker daemon unreachable" in result["error"]

    def test_env_execute_throw_returns_timeout_contract(self, monkeypatch):
        """env.execute raising a connect timeout => the exit-124 timeout envelope."""
        monkeypatch.setattr(tt, "_plan_execution", lambda *_a, **_k: _fake_plan())

        class _DeadEnv:
            host_cwd = None

            def execute(self, *_a, **_k):
                raise ConnectionError("ssh connect timeout after 10s")

        monkeypatch.setattr(tt, "_create_configured_env", lambda *_a, **_k: _DeadEnv())
        result = json.loads(
            tt.terminal_tool("echo ok", force=True, task_id="e2e-130890-exec-timeout")
        )
        assert result["exit_code"] == 124
        assert "timed out" in result["error"].lower()
