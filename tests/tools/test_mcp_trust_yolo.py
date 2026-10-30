"""YOLO and ``approvals.mode: off`` against the MCP trust gate and the spend limit.

Invariants: when every approval is bypassed for the calling session (session ``/yolo``, process
``--yolo``, ``approvals.mode: off``), a write-capable call on a ``trust: untrusted`` server runs
without the trust prompt, and only for that session; the declared spend limit is a budget, not an
approval, so a call that would cross it still asks, and where nobody can answer it is refused.
"""

import json
from unittest.mock import patch

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import get_hermes_home
from tools import approval, mcp_tool, mcp_tool_costs
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools.approval_context import reset_current_session_key, set_current_session_key

PAID = {"trust": "untrusted", "tool_costs": {"paid_render": 40}, "spend_limit": 50}


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_session_yolo", set())
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_permanent_baseline_by_home", {})
    with patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
         patch.dict(mcp_tool._server_cost_policies, {}, clear=True), \
         patch.dict(mcp_tool_costs._spent, {}, clear=True), \
         patch.dict(mcp_tool_costs._ask_locks, {}, clear=True), \
         patch.object(mcp_tool_costs, "_unscoped_moved_to", None):
        yield


def _register(config):
    registration._record_tool_trust_metadata(
        "srv", config, [type("T", (), {"name": n, "annotations": None})() for n in ("write", "paid_render")])


class _Chat:
    """A messaging chat answering every approval card it is sent with ``choice``."""

    def __init__(self, session_key: str, choice: str = "deny", **session_vars):
        self.session_key, self.choice, self.cards, self.extra = session_key, choice, [], session_vars

    def __enter__(self):
        self._vars = set_session_vars(platform="telegram", session_key=self.session_key, **self.extra)
        self._key = set_current_session_key(self.session_key)
        approval.register_gateway_notify(self.session_key, self._answer)
        return self

    def __exit__(self, *exc):
        approval.unregister_gateway_notify(self.session_key)
        reset_current_session_key(self._key)
        clear_session_vars(self._vars)

    def _answer(self, data: dict) -> None:
        self.cards.append(dict(data))
        approval.resolve_gateway_approval(self.session_key, self.choice, request_id=data["request_id"])


def test_yolo_and_mode_off_skip_the_trust_prompt_for_the_calling_session_only():
    _register({"trust": "untrusted"})
    approval.enable_session_yolo("tg:yolo")
    with _Chat("tg:yolo") as yolo:
        assert handlers._trust_gate_check("srv", "write") is None
    with _Chat("tg:other") as other:  # another chat without /yolo is still asked
        assert handlers._trust_gate_check("srv", "write") is not None
    assert yolo.cards == [] and len(other.cards) == 1

    (get_hermes_home() / "config.yaml").write_text(json.dumps({"approvals": {"mode": "off"}}))
    with _Chat("tg:other") as other:
        assert handlers._trust_gate_check("srv", "write") is None
    assert other.cards == []


def test_spend_limit_still_asks_under_yolo_and_refuses_when_nobody_can_answer():
    _register(PAID)
    approval.enable_session_yolo("tg:yolo")
    with _Chat("tg:yolo") as chat:
        assert handlers._trust_gate_check("srv", "paid_render") is None  # 40 of 50: no prompt at all
        refused = handlers._trust_gate_check("srv", "paid_render")  # 40 + 40 > 50
    assert chat.cards and len(chat.cards) == 1
    assert "spend limit" in chat.cards[0]["command"] and "UNTRUSTED" not in chat.cards[0]["command"]
    assert "spend limit is 50 credits" in json.loads(refused)["error"]

    # A cron run under approvals.mode: off: the untrusted write runs unattended, the limit still refuses.
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"approvals": {"mode": "off"}}))
    tokens, key = set_session_vars(cron_session="1"), set_current_session_key("cron:job")
    try:
        assert handlers._trust_gate_check("srv", "write") is None
        assert handlers._trust_gate_check("srv", "paid_render") is None
        refused = handlers._trust_gate_check("srv", "paid_render")
    finally:
        reset_current_session_key(key)
        clear_session_vars(tokens)
    assert "spend limit is 50 credits" in json.loads(refused)["error"]
