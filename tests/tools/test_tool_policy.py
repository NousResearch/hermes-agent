"""Tests for the per-tool approval policy (``approvals.tools``).

Inspired by Perplexity Computer's per-connector Allow / Always ask / Deny
permission controls (Aug 2026). The policy is enforced at the tool
dispatcher so it covers built-in, MCP, and plugin tools alike.
"""

import json
from unittest.mock import patch


from tools import approval as approval_mod
from tools.approval import (
    _resolve_tool_policy,
    check_tool_policy,
)


def _with_policy(policy):
    """Patch the approvals config to carry the given tools policy map."""
    return patch.object(
        approval_mod, "_get_approval_config", return_value={"tools": policy}
    )


# ---------------------------------------------------------------------------
# _resolve_tool_policy — matching semantics
# ---------------------------------------------------------------------------

class TestResolveToolPolicy:
    def test_empty_policy_returns_none(self):
        with _with_policy({}):
            assert _resolve_tool_policy("send_message") is None

    def test_missing_key_returns_none(self):
        with patch.object(approval_mod, "_get_approval_config", return_value={}):
            assert _resolve_tool_policy("send_message") is None

    def test_malformed_policy_returns_none(self):
        with patch.object(
            approval_mod, "_get_approval_config",
            return_value={"tools": ["send_message"]},
        ):
            assert _resolve_tool_policy("send_message") is None

    def test_config_read_failure_fails_open(self):
        with patch.object(
            approval_mod, "_get_approval_config",
            side_effect=RuntimeError("boom"),
        ):
            assert _resolve_tool_policy("send_message") is None

    def test_exact_match(self):
        with _with_policy({"send_message": "ask"}):
            assert _resolve_tool_policy("send_message") == "ask"

    def test_exact_match_case_insensitive(self):
        with _with_policy({"Send_Message": "deny"}):
            assert _resolve_tool_policy("send_message") == "deny"

    def test_glob_match(self):
        with _with_policy({"mcp_gmail_*": "ask"}):
            assert _resolve_tool_policy("mcp_gmail_send_email") == "ask"
            assert _resolve_tool_policy("mcp_github_create_issue") is None

    def test_exact_beats_glob(self):
        with _with_policy({"mcp_gmail_*": "deny", "mcp_gmail_read": "allow"}):
            assert _resolve_tool_policy("mcp_gmail_read") == "allow"
            assert _resolve_tool_policy("mcp_gmail_send") == "deny"

    def test_verb_aliases(self):
        for raw, expected in [
            ("allow", "allow"), ("auto", "allow"),
            ("ask", "ask"), ("always", "ask"), ("always_ask", "ask"),
            ("always-ask", "ask"),
            ("deny", "deny"), ("never", "deny"), ("block", "deny"),
            ("ASK", "ask"), (" deny ", "deny"),
        ]:
            with _with_policy({"send_message": raw}):
                assert _resolve_tool_policy("send_message") == expected, raw

    def test_unknown_verb_ignored(self):
        with _with_policy({"send_message": "maybe"}):
            assert _resolve_tool_policy("send_message") is None

    def test_non_string_entries_ignored(self):
        with _with_policy({"send_message": 1, 2: "deny", None: "ask"}):
            assert _resolve_tool_policy("send_message") is None


# ---------------------------------------------------------------------------
# check_tool_policy — verb behavior
# ---------------------------------------------------------------------------

class TestCheckToolPolicy:
    def test_no_policy_is_noop(self):
        with _with_policy({}):
            assert check_tool_policy("send_message") is None

    def test_allow_is_noop(self):
        with _with_policy({"send_message": "allow"}):
            assert check_tool_policy("send_message") is None

    def test_deny_blocks(self):
        with _with_policy({"send_message": "deny"}):
            result = check_tool_policy("send_message")
        assert result is not None
        assert result["approved"] is False
        assert "approvals.tools" in result["message"]
        assert "send_message" in result["message"]

    def test_deny_beats_yolo(self):
        with _with_policy({"send_message": "deny"}), \
             patch.object(approval_mod, "_YOLO_MODE_FROZEN", True):
            result = check_tool_policy("send_message")
        assert result is not None and result["approved"] is False

    def test_ask_routes_through_shared_gate(self):
        captured = {}

        def _fake_gate(**kwargs):
            captured.update(kwargs)
            return {"approved": True, "message": None}

        with _with_policy({"send_message": "ask"}), \
             patch.object(approval_mod, "_run_approval_gate", _fake_gate):
            result = check_tool_policy("send_message")
        assert result == {"approved": True, "message": None}
        # Per-tool allowlist grain: [a]lways on one tool must never blanket
        # other tools.
        assert captured["pattern_key"] == "plugin_rule:tool_policy:send_message"
        # Explicit ask rules override the yolo bypass.
        assert captured["honor_yolo_bypass"] is False
        # An ask rule demands a human — no silent fall-through.
        assert captured["fail_closed_when_no_human"] is True

    def test_ask_fires_under_yolo(self):
        """An explicit ask rule must still prompt when yolo is active."""
        gate_calls = []

        def _fake_gate(**kwargs):
            gate_calls.append(kwargs)
            return {"approved": False, "message": "BLOCKED: denied by user"}

        with _with_policy({"send_message": "ask"}), \
             patch.object(approval_mod, "_YOLO_MODE_FROZEN", True), \
             patch.object(approval_mod, "_run_approval_gate", _fake_gate):
            result = check_tool_policy("send_message")
        assert gate_calls, "gate must be invoked even under yolo"
        assert result["approved"] is False

    def test_gate_yolo_param_actually_bypasses_when_honored(self):
        """Sanity: honor_yolo_bypass=True still short-circuits under yolo
        (the dangerous-command path's historical behavior is unchanged)."""
        with patch.object(approval_mod, "_YOLO_MODE_FROZEN", True):
            result = approval_mod._run_approval_gate(
                pattern_key="x", description="d", display_target="t",
                cron_deny_message="c", single_query_deny_message="s",
                autoapprove_log_prefix="p",
            )
        assert result == {"approved": True, "message": None}


# ---------------------------------------------------------------------------
# Dispatcher integration (model_tools.handle_function_call)
# ---------------------------------------------------------------------------

class TestDispatcherIntegration:
    def test_denied_tool_blocked_at_dispatch(self):
        import model_tools

        with _with_policy({"vision_analyze": "deny"}):
            result = model_tools.handle_function_call(
                "vision_analyze", {"image_url": "x"}
            )
        payload = json.loads(result)
        assert "error" in payload
        assert "approvals.tools" in payload["error"]

    def test_skip_pre_tool_call_hook_skips_policy(self):
        """The agent loop passes skip=True after running the policy itself —
        the dispatcher must not double-fire the gate."""
        import model_tools

        with _with_policy({"nonexistent_tool_xyz": "deny"}):
            result = model_tools.handle_function_call(
                "nonexistent_tool_xyz", {}, skip_pre_tool_call_hook=True
            )
        payload = json.loads(result)
        # Blocked message must be the unknown-tool error, NOT the policy
        # block (the policy path was skipped).
        assert "approvals.tools" not in payload.get("error", "")

    def test_unlisted_tool_unaffected(self):
        import model_tools

        with _with_policy({"vision_analyze": "deny"}):
            result = model_tools.handle_function_call(
                "definitely_not_a_real_tool", {}
            )
        payload = json.loads(result)
        assert "approvals.tools" not in payload.get("error", "")
