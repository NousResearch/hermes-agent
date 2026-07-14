"""Tests for deterministic command risk-level derivation and threading.

Phase-2 Task 1 of tiered remote approval: a command RISK LEVEL
(critical|high|medium) is derived from the available guard signals in
``check_all_command_guards`` and threaded through
``prompt_dangerous_approval`` into the approval callback so downstream
(CLI) can tier remote approval.

The callback threading must be backward compatible: older callbacks that
do NOT accept a ``risk_level`` kwarg (delegate auto-approve, subagent
callbacks) must still work via a TypeError fallback path.
"""
from tools.approval import _derive_risk_level, prompt_dangerous_approval


# ---------------------------------------------------------------------------
# _derive_risk_level mapping
# ---------------------------------------------------------------------------

def test_derive_risk_level_hardline_is_critical():
    assert _derive_risk_level({}, False, is_hardline=True) == "critical"


def test_derive_risk_level_tirith_critical():
    assert _derive_risk_level(
        {"action": "block", "findings": [{"severity": "critical"}]}, False
    ) == "critical"


def test_derive_risk_level_tirith_block_high():
    assert _derive_risk_level(
        {"action": "block", "findings": [{"severity": "high"}]}, False
    ) == "high"


def test_derive_risk_level_dangerous_pattern_high():
    assert _derive_risk_level({}, True) == "high"


def test_derive_risk_level_warn_medium():
    assert _derive_risk_level(
        {"action": "warn", "findings": [{"severity": "medium"}]}, False
    ) == "medium"


def test_derive_risk_level_empty_defaults_medium():
    assert _derive_risk_level({}, False) == "medium"


# ---------------------------------------------------------------------------
# prompt_dangerous_approval threading
# ---------------------------------------------------------------------------

def test_prompt_dangerous_approval_passes_risk_level():
    captured = {}

    def fake(cmd, desc, *, allow_permanent=True, risk_level=""):
        captured["cmd"] = cmd
        captured["desc"] = desc
        captured["allow_permanent"] = allow_permanent
        captured["risk_level"] = risk_level
        return "once"

    result = prompt_dangerous_approval(
        "echo hi", "desc", approval_callback=fake, risk_level="high"
    )
    assert result == "once"
    assert captured["risk_level"] == "high"


def test_prompt_dangerous_approval_legacy_callback_without_risk_level():
    """Older callbacks without a risk_level kwarg still work (TypeError fallback)."""
    captured = {}

    def fake(cmd, desc, *, allow_permanent=True):
        captured["cmd"] = cmd
        captured["allow_permanent"] = allow_permanent
        return "session"

    result = prompt_dangerous_approval(
        "echo hi", "desc", approval_callback=fake, risk_level="high"
    )
    assert result == "session"
    assert captured["cmd"] == "echo hi"
