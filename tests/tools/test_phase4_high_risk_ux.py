"""Phase 4: high-risk approval UX hard stop.

Verifies:
- Risk-classification mapping (description → low/medium/high).
- High-risk proposals carry risk_level='high', default_decision='deny'.
- display_text rendered for high-risk includes ALL required markers
  (HIGH RISK, default deny, command, context, reason, id, diff-or-warn).
- Medium-risk display_text does NOT carry HIGH RISK marker (UX gradient).
- Missing diff/summary surfaces an explicit warning, never silent.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tools import approval as approval_mod
from tools.approval import (
    _await_gateway_decision,
    _classify_pattern_risk,
    _render_approval_display_text,
    register_gateway_notify,
    set_default_approval_store,
)
from tools.approval_store_memory import InMemoryApprovalStore


@pytest.fixture(autouse=True)
def _reset():
    prev = approval_mod.get_default_approval_store()
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    set_default_approval_store(prev)


# ---------------------------------------------------------------------------
# Risk-classification mapping
# ---------------------------------------------------------------------------


class TestClassifyPatternRisk:

    @pytest.mark.parametrize("description,expected", [
        ("recursive delete of system directory", "high"),
        ("recursive delete", "high"),
        ("recursive world/other-writable", "high"),
        ("recursive chown to root", "high"),
        ("format filesystem", "high"),
        ("SQL DROP", "high"),
        ("SQL DELETE without WHERE", "high"),
        ("SQL TRUNCATE", "high"),
        ("overwrite system config", "high"),
        ("overwrite system file via tee", "high"),
        ("overwrite system file via redirection", "high"),
        ("write to block device", "high"),
        ("dd to raw block device", "high"),
        ("pipe remote content to shell", "high"),
        ("execute remote script via process substitution", "high"),
        ("kill all processes", "high"),
        # Lower-impact patterns stay medium:
        ("world/other-writable permissions", "medium"),
        ("disk copy", "medium"),
        ("stop/restart system service", "medium"),
        ("force kill processes", "medium"),
        ("script execution via -e/-c flag", "medium"),
        ("shell command via -c/-lc flag", "medium"),
    ])
    def test_known_descriptions_classify_correctly(self, description, expected):
        assert _classify_pattern_risk(description) == expected

    def test_empty_description_defaults_to_high(self):
        """Unknown/missing description fails closed: defensive high."""
        assert _classify_pattern_risk("") == "high"
        assert _classify_pattern_risk(None) == "high"


# ---------------------------------------------------------------------------
# Display-text rendering
# ---------------------------------------------------------------------------


class TestRenderApprovalDisplayText:

    def test_high_risk_includes_all_required_markers(self):
        text = _render_approval_display_text(
            approval_id="abc123",
            risk_level="high",
            command="rm -rf /home/user/data",
            risk_reason="recursive delete of home directory",
            cwd="/home/user",
            backend="bash",
            diff_summary="3 files would be deleted permanently",
        )
        # Spec-mandated markers:
        assert "HIGH RISK" in text
        assert "DENY" in text
        assert "abc123" in text                            # approval id
        assert "rm -rf /home/user/data" in text            # exact command
        assert "/home/user" in text                        # cwd
        assert "bash" in text                              # backend
        assert "recursive delete of home directory" in text   # pinned reason
        assert "3 files would be deleted permanently" in text  # diff/summary
        # /approve and /deny commands include the id:
        assert "/approve abc123" in text
        assert "/deny abc123" in text

    def test_high_risk_without_diff_includes_explicit_warning(self):
        """Spec: missing diff MUST be explicit, never silent."""
        text = _render_approval_display_text(
            approval_id="xyz789",
            risk_level="high",
            command="DROP DATABASE prod",
            risk_reason="SQL DROP",
        )
        assert "HIGH RISK" in text
        assert "NOT AVAILABLE" in text
        # The warning text must be self-explanatory:
        assert "independently reviewed" in text.lower() or \
               "review" in text.lower()

    def test_medium_risk_does_not_include_high_risk_marker(self):
        """UX gradient: only high-risk gets the loud marker."""
        text = _render_approval_display_text(
            approval_id="med1",
            risk_level="medium",
            command="systemctl restart nginx",
            risk_reason="stop/restart system service",
        )
        assert "HIGH RISK" not in text
        # But basic structure (command, id) still present:
        assert "systemctl restart nginx" in text
        assert "med1" in text


# ---------------------------------------------------------------------------
# End-to-end: high-risk proposal flowing through _await_gateway_decision
# ---------------------------------------------------------------------------


def test_high_risk_proposal_carries_correct_metadata():
    """A high-risk command produces a proposal with risk_level='high',
    default_decision='deny', and display_text with HIGH RISK marker."""
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    session_key = "phase4-test"
    register_gateway_notify(session_key, lambda data: None)

    captured: dict[str, Any] = {}
    barrier = threading.Event()

    def driver():
        _await_gateway_decision(
            session_key=session_key,
            notify_cb=lambda data: (captured.update(data=data), barrier.set()),
            approval_data={
                "command": "rm -rf /home/u/important",
                "description": "recursive delete of home directory",
                "pattern_key": "recursive-rm",
                "pattern_keys": ["recursive-rm"],
            },
            surface="test",
        )

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    assert barrier.wait(timeout=5)

    # approval_data passed to notify:
    data = captured["data"]
    assert data["risk_level"] == "high"
    assert data["default_decision"] == "deny"
    assert "HIGH RISK" in data["display_text"]
    assert "rm -rf /home/u/important" in data["display_text"]
    assert data["approval_id"] in data["display_text"]
    # NOT AVAILABLE warning since no diff provided:
    assert "NOT AVAILABLE" in data["display_text"]

    # And the persisted proposal mirrors all of it:
    proposal = store.get(data["approval_id"])
    assert proposal.risk_level == "high"
    assert proposal.default_decision == "deny"
    assert proposal.requires_explicit_approval is True
    assert "HIGH RISK" in (proposal.display_text or "")

    # Resolve to let the driver thread exit cleanly.
    from tools.approval import resolve_gateway_approval_by_id
    resolve_gateway_approval_by_id(session_key, data["approval_id"], "deny")
    t.join(timeout=5)


def test_medium_risk_proposal_does_not_include_high_marker():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    session_key = "phase4-med"
    register_gateway_notify(session_key, lambda data: None)

    captured: dict[str, Any] = {}
    barrier = threading.Event()

    def driver():
        _await_gateway_decision(
            session_key=session_key,
            notify_cb=lambda data: (captured.update(data=data), barrier.set()),
            approval_data={
                "command": "systemctl restart nginx",
                "description": "stop/restart system service",
                "pattern_key": "systemctl-restart",
                "pattern_keys": ["systemctl-restart"],
            },
            surface="test",
        )

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    assert barrier.wait(timeout=5)

    data = captured["data"]
    assert data["risk_level"] == "medium"
    assert "HIGH RISK" not in data["display_text"]
    # default_decision STILL deny (paranoid default for all approvals):
    assert data["default_decision"] == "deny"

    from tools.approval import resolve_gateway_approval_by_id
    resolve_gateway_approval_by_id(session_key, data["approval_id"], "deny")
    t.join(timeout=5)
