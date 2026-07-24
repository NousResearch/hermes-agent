"""Tests for the generic, config-driven tool-approval gate.

Covers:
  * detection / default-off / bypass ladder (check_tool_approval)
  * mode selection across unattended / force_deferred / allow_inline / default
  * deferred staging: pending record + Kanban card + non-error "staged" result
  * inline (blocking) approval reuse of the dangerous-command engine
  * one-shot replay token, TTL refusal, double-execution guard
"""

import json
import os
import shutil
import tempfile
import threading
import time
from unittest.mock import MagicMock

import pytest

import tools.approval as approval_module
from tools import tool_gate
from tools import write_approval as wa


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_tg_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_state():
    approval_module._session_approved.clear()
    approval_module._permanent_approved.clear()
    approval_module._gateway_notify_cbs.clear()
    approval_module._gateway_queues.clear()
    saved = {}
    for k in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
              "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "HERMES_SESSION_KEY"):
        if k in os.environ:
            saved[k] = os.environ.pop(k)
    yield
    approval_module._session_approved.clear()
    approval_module._permanent_approved.clear()
    approval_module._gateway_notify_cbs.clear()
    approval_module._gateway_queues.clear()
    for k in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
              "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "HERMES_SESSION_KEY"):
        os.environ.pop(k, None)
    for k, v in saved.items():
        os.environ[k] = v


def _set_gate(home, **gate):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("approvals", {})["tool_gate"] = gate
    cfg.save_config(c)


# ---------------------------------------------------------------------------
# Detection / bypass ladder
# ---------------------------------------------------------------------------

class TestDetectionLadder:
    def test_no_config_allows(self, hermes_home):
        assert tool_gate.get_tool_gate_config() == {}
        assert approval_module.check_tool_approval("send_mail", {}) == {
            "approved": True, "message": None}

    def test_disabled_allows(self, hermes_home):
        _set_gate(hermes_home, enabled=False, require_approval=["send_mail"])
        assert approval_module.check_tool_approval("send_mail", {})["approved"] is True

    def test_unlisted_tool_allows(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["send_mail"])
        assert approval_module.check_tool_approval("read_file", {})["approved"] is True

    def test_glob_match_is_gated(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["send_*"],
                  force_deferred=["send_*"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("send_sms", {"to": "x"})
        assert r["status"] == "staged"

    def test_yolo_bypasses(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["send_mail"])
        os.environ["HERMES_YOLO_MODE"] = "1"
        # _YOLO_MODE_FROZEN is read at import; patch it directly.
        approval_module._YOLO_MODE_FROZEN = True
        try:
            assert approval_module.check_tool_approval("send_mail", {})["approved"] is True
        finally:
            approval_module._YOLO_MODE_FROZEN = False

    def test_mode_off_bypasses(self, hermes_home):
        import hermes_cli.config as cfg
        c = cfg.load_config()
        c.setdefault("approvals", {})["mode"] = "off"
        c["approvals"]["tool_gate"] = {"enabled": True, "require_approval": ["send_mail"]}
        cfg.save_config(c)
        assert approval_module.check_tool_approval("send_mail", {})["approved"] is True

    def test_prior_session_approval_allows(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["send_mail"])
        os.environ["HERMES_SESSION_KEY"] = "sk1"
        approval_module.approve_session("sk1", approval_module._tool_key("send_mail"))
        assert approval_module.check_tool_approval("send_mail", {})["approved"] is True


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

class TestModeSelection:
    def _cfg(self, **kw):
        base = {"enabled": True, "require_approval": ["t"], "default_mode": "deferred"}
        base.update(kw)
        return base

    def test_no_channel_is_deferred(self, hermes_home):
        assert approval_module._select_tool_mode("t", self._cfg(), "sk") == "deferred"

    def test_cron_is_deferred_even_with_channel(self, hermes_home):
        approval_module.register_gateway_notify("sk", lambda d: None)
        os.environ["HERMES_CRON_SESSION"] = "1"
        assert approval_module._select_tool_mode("t", self._cfg(allow_inline=["t"]), "sk") == "deferred"

    def test_force_deferred_overrides_inline(self, hermes_home):
        approval_module.register_gateway_notify("sk", lambda d: None)
        cfg = self._cfg(allow_inline=["t"], force_deferred=["t"])
        assert approval_module._select_tool_mode("t", cfg, "sk") == "deferred"

    def test_allow_inline_with_channel_is_inline(self, hermes_home):
        approval_module.register_gateway_notify("sk", lambda d: None)
        assert approval_module._select_tool_mode("t", self._cfg(allow_inline=["t"]), "sk") == "inline"

    def test_default_mode_inline_with_channel(self, hermes_home):
        approval_module.register_gateway_notify("sk", lambda d: None)
        assert approval_module._select_tool_mode("t", self._cfg(default_mode="inline"), "sk") == "inline"

    def test_default_mode_deferred_with_channel(self, hermes_home):
        approval_module.register_gateway_notify("sk", lambda d: None)
        assert approval_module._select_tool_mode("t", self._cfg(default_mode="deferred"), "sk") == "deferred"


# ---------------------------------------------------------------------------
# Deferred staging
# ---------------------------------------------------------------------------

class TestDeferredStaging:
    def test_stage_creates_pending_and_card(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"],
                  deferred={"pending_ttl_hours": 72})
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})

        assert r["approved"] is False
        assert r["status"] == "staged"
        assert "do NOT retry" in r["message"].lower() or "do not retry" in r["message"].lower()
        pid = r["pending_id"]
        assert pid

        # Pending record persisted with replayable payload + token + status.
        rec = wa.get_pending(tool_gate.SUBSYSTEM, pid)
        assert rec is not None
        assert rec["payload"]["tool_name"] == "echo_tool"
        assert rec["payload"]["args"] == {"msg": "hi"}
        assert rec["status"] == "pending"
        assert rec["token"]
        assert rec["expires_at"] > time.time()

        # Kanban approval card created and linked both ways.
        card_id = r["card_id"]
        assert card_id
        assert rec["card_id"] == card_id
        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        task = kb.get_task(conn, card_id)
        assert task is not None
        assert tool_gate.APPROVAL_MARKER in (task.body or "")
        assert pid in (task.body or "")

    def test_empty_board_assignee_uses_nonprofile_sentinel(self, hermes_home):
        # An empty board_assignee must NOT leave the card unassigned (a set
        # kanban.default_assignee could then auto-spawn it). It gets the
        # non-profile REVIEW_ASSIGNEE sentinel so the dispatcher skips it.
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"], deferred={"board_assignee": ""})
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        task = kb.get_task(conn, r["card_id"])
        assert task.assignee == tool_gate.REVIEW_ASSIGNEE
        from hermes_cli.profiles import profile_exists
        assert not profile_exists(task.assignee)  # never auto-dispatched

    def test_background_origin_defers(self, hermes_home, monkeypatch):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"])
        approval_module.register_gateway_notify("sk", lambda d: None)
        os.environ["HERMES_SESSION_KEY"] = "sk"
        monkeypatch.setattr(tool_gate, "_current_profile", lambda: "default")
        # Background review origin → deferred even though a channel exists.
        import tools.skill_provenance as sp
        tok = sp.set_current_write_origin("background_review")
        try:
            mode = approval_module._select_tool_mode("echo_tool",
                                                     tool_gate.get_tool_gate_config(), "sk")
        finally:
            sp.reset_current_write_origin(tok)
        assert mode == "deferred"


# ---------------------------------------------------------------------------
# Inline (blocking) approval
# ---------------------------------------------------------------------------

class TestInlineApproval:
    SK = "tg-inline-sk"

    def _run_in_thread(self, tool, args):
        holder = {}

        def _go():
            holder["r"] = approval_module.check_tool_approval(tool, args)

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        for _ in range(100):
            if approval_module._gateway_queues.get(self.SK):
                break
            time.sleep(0.02)
        return holder, t

    def test_inline_approve_once(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  allow_inline=["echo_tool"], default_mode="inline")
        os.environ["HERMES_SESSION_KEY"] = self.SK
        approval_module.register_gateway_notify(self.SK, lambda d: None)

        holder, t = self._run_in_thread("echo_tool", {"x": 1})
        approval_module.resolve_gateway_approval(self.SK, "once")
        t.join(timeout=5)
        assert holder["r"] == {"approved": True, "message": None}
        # "once" must NOT grant a session allowlist entry.
        assert not approval_module.is_approved(self.SK, approval_module._tool_key("echo_tool"))

    def test_inline_session_grants_allowlist(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  allow_inline=["echo_tool"], default_mode="inline")
        os.environ["HERMES_SESSION_KEY"] = self.SK
        approval_module.register_gateway_notify(self.SK, lambda d: None)

        holder, t = self._run_in_thread("echo_tool", {"x": 1})
        approval_module.resolve_gateway_approval(self.SK, "session")
        t.join(timeout=5)
        assert holder["r"]["approved"] is True
        assert approval_module.is_approved(self.SK, approval_module._tool_key("echo_tool"))

    def test_inline_deny_blocks(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  allow_inline=["echo_tool"], default_mode="inline")
        os.environ["HERMES_SESSION_KEY"] = self.SK
        approval_module.register_gateway_notify(self.SK, lambda d: None)

        holder, t = self._run_in_thread("echo_tool", {"x": 1})
        approval_module.resolve_gateway_approval(self.SK, "deny")
        t.join(timeout=5)
        r = holder["r"]
        assert r["approved"] is False
        assert r["status"] == "blocked"
        assert "do not retry" in r["message"].lower()

    def test_inline_notify_payload_carries_tool_fields(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  allow_inline=["echo_tool"], default_mode="inline")
        os.environ["HERMES_SESSION_KEY"] = self.SK
        seen = {}
        approval_module.register_gateway_notify(self.SK, lambda d: seen.update(d))

        holder, t = self._run_in_thread("echo_tool", {"x": 1})
        approval_module.resolve_gateway_approval(self.SK, "once")
        t.join(timeout=5)
        assert seen.get("kind") == "tool"
        assert seen.get("tool_name") == "echo_tool"
        assert "echo_tool" in seen.get("command", "")


# ---------------------------------------------------------------------------
# Replay token / TTL / idempotency
# ---------------------------------------------------------------------------

class TestReplay:
    def test_replay_token_allows_once(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"])
        tok = tool_gate.set_replay_token(
            {"pending_id": "p1", "tool_name": "echo_tool", "token": "t"})
        try:
            assert approval_module.check_tool_approval("echo_tool", {})["approved"] is True
            # Token is one-shot: a second call re-gates (deferred, no channel).
            r2 = approval_module.check_tool_approval("echo_tool", {})
            assert r2["approved"] is False
        finally:
            tool_gate.reset_replay_token(tok)

    def test_replay_token_wrong_tool_ignored(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"])
        tok = tool_gate.set_replay_token(
            {"pending_id": "p1", "tool_name": "other", "token": "t"})
        try:
            r = approval_module.check_tool_approval("echo_tool", {})
            assert r["approved"] is False  # token for a different tool → staged
        finally:
            tool_gate.reset_replay_token(tok)

    def test_replay_executes_and_discards(self, hermes_home, monkeypatch):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]

        calls = []

        def _fake_hfc(name, args):
            calls.append((name, args))
            return '{"ok": true}'

        monkeypatch.setattr("model_tools.handle_function_call", _fake_hfc)
        out = tool_gate.replay_pending_action(pid)
        assert out["ok"] is True
        assert calls == [("echo_tool", {"msg": "hi"})]
        # Pending discarded after success → cannot replay again.
        assert wa.get_pending(tool_gate.SUBSYSTEM, pid) is None
        out2 = tool_gate.replay_pending_action(pid)
        assert out2["ok"] is False

    def test_replay_refuses_expired(self, hermes_home, monkeypatch):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]
        wa.update_pending(tool_gate.SUBSYSTEM, pid, {"expires_at": time.time() - 1})

        called = []
        monkeypatch.setattr("model_tools.handle_function_call",
                            lambda n, a: called.append(1) or "x")
        out = tool_gate.replay_pending_action(pid)
        assert out["ok"] is False
        assert "expired" in out["message"].lower()
        assert not called

    def test_replay_refuses_double_execute(self, hermes_home, monkeypatch):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]
        # Simulate an in-flight execution (status flipped by another worker).
        wa.update_pending(tool_gate.SUBSYSTEM, pid, {"status": "executing"})
        monkeypatch.setattr("model_tools.handle_function_call", lambda n, a: "x")
        out = tool_gate.replay_pending_action(pid)
        assert out["ok"] is False
        assert "executing" in out["message"].lower()

    def test_replay_concurrent_race_executes_exactly_once(self, hermes_home, monkeypatch):
        """Two threads racing replay_pending_action must not both execute.

        The prior implementation read status then wrote "executing" as two
        separate steps, so two concurrent callers could both observe
        "pending" before either wrote — both then "won" and the tool ran
        twice. write_approval.claim_pending closes that window with a
        lock-guarded check-and-set.
        """
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]

        calls = []
        call_lock = threading.Lock()
        start_barrier = threading.Barrier(2)

        def _fake_hfc(name, args):
            with call_lock:
                calls.append((name, args))
            return '{"ok": true}'

        monkeypatch.setattr("model_tools.handle_function_call", _fake_hfc)

        results = []

        def _run():
            start_barrier.wait()
            results.append(tool_gate.replay_pending_action(pid))

        threads = [threading.Thread(target=_run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 1
        oks = [r for r in results if r["ok"]]
        not_oks = [r for r in results if not r["ok"]]
        assert len(oks) == 1
        assert len(not_oks) == 1
        assert "already" in not_oks[0]["message"].lower() or "executing" in not_oks[0]["message"].lower()

    def test_maybe_replay_kanban_task_not_an_exec_card(self, hermes_home):
        """A normal (non-exec-card) task must fall through to None so the
        caller proceeds with the regular LLM worker turn."""
        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        task_id = kb.create_task(
            conn, title="Normal work", body="Please do the normal thing.",
            assignee="default", created_by="default",
        )
        assert tool_gate.maybe_replay_kanban_task(task_id) is None

    def test_maybe_replay_kanban_task_success_completes_task(self, hermes_home, monkeypatch):
        """An exec card whose replay succeeds must call kanban_complete, not
        hand the replay off to an LLM turn."""
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]
        approve = tool_gate.approve_action(pid)
        exec_card_id = approve["exec_card_id"]

        calls = []
        import model_tools
        _real_hfc = model_tools.handle_function_call

        def _fake_hfc(name, args=None, **kwargs):
            calls.append((name, args))
            if name == "echo_tool":
                return '{"ok": true}'
            # kanban_complete/kanban_block must go through for real so the
            # task's board state is actually observable.
            return _real_hfc(name, args or {}, **kwargs)

        monkeypatch.setattr("model_tools.handle_function_call", _fake_hfc)
        os.environ["HERMES_KANBAN_TASK"] = exec_card_id
        try:
            outcome = tool_gate.maybe_replay_kanban_task(exec_card_id)
        finally:
            os.environ.pop("HERMES_KANBAN_TASK", None)

        assert outcome is not None
        assert outcome["ok"] is True
        called_names = [c[0] for c in calls]
        assert "echo_tool" in called_names
        assert "kanban_complete" in called_names
        assert "kanban_block" not in called_names

        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        task = kb.get_task(conn, exec_card_id)
        assert task.status == "done"

    def test_maybe_replay_kanban_task_failure_blocks_task(self, hermes_home, monkeypatch):
        """A replay that fails must call kanban_block, not silently vanish."""
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]
        approve = tool_gate.approve_action(pid)
        exec_card_id = approve["exec_card_id"]

        calls = []
        import model_tools
        _real_hfc = model_tools.handle_function_call

        def _fake_hfc(name, args=None, **kwargs):
            calls.append((name, args))
            if name == "echo_tool":
                raise RuntimeError("boom")
            return _real_hfc(name, args or {}, **kwargs)

        monkeypatch.setattr("model_tools.handle_function_call", _fake_hfc)
        os.environ["HERMES_KANBAN_TASK"] = exec_card_id
        try:
            outcome = tool_gate.maybe_replay_kanban_task(exec_card_id)
        finally:
            os.environ.pop("HERMES_KANBAN_TASK", None)

        assert outcome is not None
        assert outcome["ok"] is False
        called_names = [c[0] for c in calls]
        assert "kanban_block" in called_names
        assert "kanban_complete" not in called_names

        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        task = kb.get_task(conn, exec_card_id)
        assert task.status == "blocked"

    def test_approve_action_spawns_exec_card(self, hermes_home):
        _set_gate(hermes_home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]

        approval_card = r["card_id"]
        out = tool_gate.approve_action(pid)
        assert out["ok"] is True
        exec_card = out["exec_card_id"]
        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        task = kb.get_task(conn, exec_card)
        assert task is not None
        assert tool_gate.parse_replay_marker(task.body) == pid
        # Assigned to a real profile so the dispatcher will spawn it.
        assert task.assignee
        # Born READY (no blocking parent) so the dispatcher claims it — a parent
        # link would strand it in 'todo' until the review-only approval card is
        # 'done' (which never happens).
        assert task.status == "ready"
        # The human approval card is archived once approved (off the active board).
        approval = kb.get_task(conn, approval_card)
        assert approval is not None and approval.status == "archived"
        # Idempotent: approving again returns the same exec card (no dup).
        rec = wa.get_pending(tool_gate.SUBSYSTEM, pid)
        assert rec["status"] == "approved"
        out2 = tool_gate.approve_action(pid)
        assert out2["ok"] is False  # already approved


# ---------------------------------------------------------------------------
# Common dispatch choke point: model_tools.handle_function_call
# ---------------------------------------------------------------------------
# tool_executor.py (sequential) and agent_runtime_helpers.py (concurrent) are
# not the only ways a tool call reaches execution — the execute_code sandbox
# IPC handler and the MCP tool server both call
# ``model_tools.handle_function_call`` directly, bypassing those two choke
# points entirely. The gate must live inside handle_function_call itself so
# every dispatch path is covered, with a ``skip_tool_approval_gate`` escape
# hatch for callers (tool_executor.py, agent_runtime_helpers.py) that already
# ran the gate at their own choke point — otherwise a replay token consumed
# by the outer check would be gone by the time the inner check re-runs.

class TestHandleFunctionCallChokePoint:
    def test_direct_dispatch_is_gated(self, hermes_home, monkeypatch):
        """A caller that skips invoke_tool/execute_tool_calls_sequential
        entirely (like the sandbox IPC handler or the MCP server) and calls
        handle_function_call directly must still be gated."""
        import model_tools

        monkeypatch.setattr(
            "tools.approval.check_tool_approval",
            lambda name, args: {"approved": False, "status": "blocked",
                                "message": "BLOCKED: denied"},
        )
        result = model_tools.handle_function_call("definitely_not_a_real_tool", {})
        payload = json.loads(result)
        assert payload["status"] == "blocked"
        assert "denied" in payload["error"]

    def test_direct_dispatch_staged_short_circuits(self, hermes_home, monkeypatch):
        import model_tools

        monkeypatch.setattr(
            "tools.approval.check_tool_approval",
            lambda name, args: {"approved": False, "status": "staged",
                                "message": "Queued for approval", "pending_id": "p1"},
        )
        result = model_tools.handle_function_call("definitely_not_a_real_tool", {})
        payload = json.loads(result)
        assert payload["status"] == "staged"
        assert payload["message"] == "Queued for approval"

    def test_skip_flag_bypasses_gate(self, hermes_home, monkeypatch):
        """Callers that already gated (tool_executor.py, agent_runtime_helpers.py)
        pass skip_tool_approval_gate=True so the inner check — and any replay
        token it would try to consume — doesn't run a second time."""
        import model_tools

        gate = MagicMock(return_value={"approved": False, "status": "blocked",
                                       "message": "should not be reached"})
        monkeypatch.setattr("tools.approval.check_tool_approval", gate)
        # Would raise/error past the gate for an unregistered tool, but the
        # point here is only that the gate itself is never consulted.
        model_tools.handle_function_call(
            "definitely_not_a_real_tool", {}, skip_tool_approval_gate=True,
        )
        gate.assert_not_called()

    def test_allowed_dispatch_is_not_short_circuited(self, hermes_home, monkeypatch):
        import model_tools

        monkeypatch.setattr(
            "tools.approval.check_tool_approval",
            lambda name, args: {"approved": True, "message": None},
        )
        # An allowed, unregistered tool should fail past the gate with the
        # normal "unknown tool" style error, not a gate-shaped one.
        result = model_tools.handle_function_call("definitely_not_a_real_tool", {})
        payload = json.loads(result)
        assert "status" not in payload or payload.get("status") not in {"blocked", "staged"}

