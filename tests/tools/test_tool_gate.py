"""Tests for the generic, config-driven tool-approval gate.

Covers:
  * detection / default-off / bypass ladder (check_tool_approval)
  * mode selection across unattended / force_deferred / allow_inline / default
  * deferred staging: pending record + Kanban card + non-error "staged" result
  * inline (blocking) approval reuse of the dangerous-command engine
  * one-shot replay token, TTL refusal, double-execution guard
"""

import os
import shutil
import tempfile
import threading
import time

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


class TestMattermostCardCallback:
    """Cross-process deferred-card button resolution: the worker posts the
    buttons (persisting the secret to the pending record); the gateway validates
    the click against that on-disk record, NOT in-memory adapter state. Mirrors
    the closures the Mattermost adapter builds for ``kind == "card"``."""

    ALLOWED = {"u_alon"}

    def _stage(self, home):
        _set_gate(home, enabled=True, require_approval=["echo_tool"],
                  force_deferred=["echo_tool"])
        os.environ["HERMES_CRON_SESSION"] = "1"
        r = approval_module.check_tool_approval("echo_tool", {"msg": "hi"})
        pid = r["pending_id"]
        wa.update_pending(tool_gate.SUBSYSTEM, pid, {"mm_secret": "sekret"})
        return pid

    def _callback(self, pid, *, user, token, choice):
        from plugins.platforms.mattermost import approval as mm
        rec = wa.get_pending(tool_gate.SUBSYSTEM, pid)

        def secret_for(_ref):
            return (rec or {}).get("mm_secret")

        class Guard:
            def pop(self, _ref, _default=True):
                if rec is None or rec.get("mm_resolved"):
                    return True
                wa.update_pending(tool_gate.SUBSYSTEM, pid, {"mm_resolved": True})
                return False

        def approve(p):
            return tool_gate.approve_action(p)

        def discard(p):
            wa.discard_pending(tool_gate.SUBSYSTEM, p)

        ctx = {"kind": "card", "pending_id": pid, "token": token,
               "choice": choice, "post_ref": "r1"}
        return mm.handle_callback(ctx, user, allowed_users=self.ALLOWED,
                                  expected_secret_for=secret_for, resolved_store=Guard(),
                                  resolve_fn=None, approve_fn=approve, discard_fn=discard)

    def test_approve_resolves_and_spawns_exec(self, hermes_home):
        pid = self._stage(hermes_home)
        res = self._callback(pid, user="u_alon", token="sekret", choice="once")
        assert res["ok"] and res["status"] == "resolved"
        assert "Approved" in (res["update_text"] or "")
        assert wa.get_pending(tool_gate.SUBSYSTEM, pid)["status"] == "approved"

    def test_bad_token_rejected(self, hermes_home):
        pid = self._stage(hermes_home)
        res = self._callback(pid, user="u_alon", token="WRONG", choice="once")
        assert res["ok"] is False and res["status"] == "bad_token"
        assert wa.get_pending(tool_gate.SUBSYSTEM, pid)["status"] == "pending"

    def test_unauthorized_user_rejected(self, hermes_home):
        pid = self._stage(hermes_home)
        res = self._callback(pid, user="intruder", token="sekret", choice="once")
        assert res["ok"] is False and res["status"] == "unauthorized"

    def test_double_click_guarded(self, hermes_home):
        pid = self._stage(hermes_home)
        first = self._callback(pid, user="u_alon", token="sekret", choice="once")
        assert first["ok"]
        second = self._callback(pid, user="u_alon", token="sekret", choice="once")
        assert second["ok"] is False and second["status"] == "already_resolved"

    def test_deny_discards(self, hermes_home):
        pid = self._stage(hermes_home)
        res = self._callback(pid, user="u_alon", token="sekret", choice="deny")
        assert res["ok"] and res["status"] == "resolved"
        assert wa.get_pending(tool_gate.SUBSYSTEM, pid) is None
