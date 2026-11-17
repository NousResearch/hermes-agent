"""``--on-approval`` reaches the approval gate as the finite turn's committed ``on_approval``:
``deny`` / ``approve`` override ``approvals.single_query_mode`` for that turn only, ``ask`` sends the
flagged command to the viewer's gateway prompt instead of resolving it headless."""
from unittest.mock import patch

import pytest

import tools.approval as approval_mod
from gateway.session_finite import admit_finite, finite_turn_scope
from hermes_state_runtime import RuntimeStoreError
from tools import approval_context

SESSION = "test-finite-on-approval"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_mod, "_command_matches_permanent_allowlist", lambda command: False)
    for var in ("HERMES_CRON_SESSION", "HERMES_SINGLE_QUERY_SESSION", "HERMES_INTERACTIVE", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION)
    approval_mod.clear_session(SESSION)
    yield
    approval_mod.unregister_gateway_notify(SESSION)
    approval_mod.clear_session(SESSION)


def test_finite_on_approval_overrides_config_and_ask_reaches_the_viewer(monkeypatch):
    with patch("tools.approval_context._binary_approval_mode", return_value="approve"):
        with finite_turn_scope(True, on_approval="deny"):
            denied = approval_mod.check_all_command_guards("rm -rf ./build", "local")
    assert denied["approved"] is False and "single-query" in denied["message"]

    with patch("tools.approval_context._binary_approval_mode", return_value="deny"):
        with finite_turn_scope(True, on_approval="approve"):
            assert approval_mod.check_all_command_guards("rm -rf ./build", "local")["approved"] is True
        with finite_turn_scope(True):  # no flag: the config default still governs
            assert approval_mod.check_all_command_guards("rm -rf ./build", "local")["approved"] is False

    asked = []

    def human(*args, **kwargs):
        asked.append(kwargs["command"])
        return {"approved": False, "message": "denied by user"}

    monkeypatch.setattr(approval_mod, "_human_decision", human)
    approval_mod.register_gateway_notify(SESSION, lambda data: None)
    with finite_turn_scope(True, on_approval="ask"):
        assert approval_mod.check_all_command_guards("rm -rf ./build", "local")["approved"] is False
    assert asked == ["rm -rf ./build"]


@pytest.mark.parametrize("params", [
    {"on_approval": "ask"},                                        # not finite
    {"finite": True, "on_approval": "yes"},                        # unknown mode
    {"finite": True, "unattended": True, "on_approval": "deny"},   # contradicts -z auto-approval
])
def test_admission_refuses_malformed_on_approval(params):
    with pytest.raises(RuntimeStoreError):
        admit_finite(params)
    assert admit_finite({"finite": True, "on_approval": "ask"}) == {"finite": True, "on_approval": "ask"}
