"""Provenance for silently auto-approved dangerous commands.

Port of the approval-provenance surfaced in Kilo-Org/kilocode#12728/#12995:
when a flagged command runs without a prompt because of a prior user decision
(session approval, permanent 'always' entry, or a command_allowlist glob),
the guard result must say WHY, so the tool result can explain the
auto-approval instead of looking like a silent bypass.
"""



import pytest

from tools import approval as mod
from tools.approval import (
    _approval_scope,
    _matching_permanent_allowlist_entry,
    approve_permanent,
    approve_session,
    check_all_command_guards,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(mod, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(mod, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(mod, "get_current_session_key", lambda default=None: "prov-sess")
    yield
    mod._session_approved.clear()
    mod._permanent_approved.clear()


def _tirith_allow(monkeypatch):
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _c: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )


class TestApprovalScope:
    def test_none_when_unapproved(self):
        assert _approval_scope("prov-sess", "recursive delete") is None

    def test_session_scope(self):
        approve_session("prov-sess", "recursive delete")
        assert _approval_scope("prov-sess", "recursive delete") == "session"

    def test_permanent_scope_wins(self):
        approve_session("prov-sess", "recursive delete")
        approve_permanent("recursive delete")
        assert _approval_scope("prov-sess", "recursive delete") == "permanent"


class TestAllowlistEntryLookup:
    def test_exact_match_returns_entry(self):
        mod.load_permanent({"podman ps"})
        assert _matching_permanent_allowlist_entry("podman ps") == "podman ps"

    def test_glob_match_returns_pattern(self):
        mod.load_permanent({"podman *"})
        assert _matching_permanent_allowlist_entry("podman rm x") == "podman *"

    def test_compound_command_never_matches(self):
        mod.load_permanent({"podman *"})
        assert _matching_permanent_allowlist_entry("podman ps && rm -rf /x") is None


class TestGuardProvenance:
    def test_session_preapproval_reports_provenance(self, monkeypatch):
        _tirith_allow(monkeypatch)
        monkeypatch.setattr(mod, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(mod, "_is_gateway_approval_context", lambda: False)
        is_dangerous, pattern_key, _desc = mod.detect_dangerous_command(
            "rm -rf /tmp/stuff"
        )
        assert is_dangerous
        approve_session("prov-sess", pattern_key)

        result = check_all_command_guards("rm -rf /tmp/stuff", "local")

        assert result["approved"] is True
        assert result.get("pre_approved") == "session"
        assert result.get("description")

    def test_permanent_preapproval_reports_provenance(self, monkeypatch):
        _tirith_allow(monkeypatch)
        monkeypatch.setattr(mod, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(mod, "_is_gateway_approval_context", lambda: False)
        _d, pattern_key, _desc = mod.detect_dangerous_command("rm -rf /tmp/stuff")
        approve_session("prov-sess", pattern_key)
        approve_permanent(pattern_key)

        result = check_all_command_guards("rm -rf /tmp/stuff", "local")

        assert result["approved"] is True
        assert result.get("pre_approved") == "permanent"

    def test_allowlist_match_reports_rule(self, monkeypatch):
        mod.load_permanent({"rm -rf /tmp/scratch"})

        result = check_all_command_guards("rm -rf /tmp/scratch", "local")

        assert result["approved"] is True
        assert result.get("pre_approved") == "allowlist"
        assert result.get("pre_approved_rule") == "rm -rf /tmp/scratch"

    def test_unflagged_command_has_no_provenance(self, monkeypatch):
        _tirith_allow(monkeypatch)
        monkeypatch.setattr(mod, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(mod, "_is_gateway_approval_context", lambda: False)

        result = check_all_command_guards("echo hello", "local")

        assert result["approved"] is True
        assert "pre_approved" not in result


class TestTerminalToolNote:
    """End-to-end: the terminal tool result carries an 'approval' note
    explaining a silent pre-approval."""

    def _run(self, monkeypatch, approval):
        import json

        from tools import terminal_tool as tt

        monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: approval)
        return json.loads(tt.terminal_tool(command="echo prov-ok"))

    def test_session_preapproval_note_in_result(self, monkeypatch):
        result = self._run(
            monkeypatch,
            {
                "approved": True,
                "message": None,
                "pre_approved": "session",
                "description": "recursive file deletion",
            },
        )
        assert result["exit_code"] == 0
        assert "approve for session" in result.get("approval", "")
        assert "recursive file deletion" in result["approval"]

    def test_permanent_preapproval_note_in_result(self, monkeypatch):
        result = self._run(
            monkeypatch,
            {
                "approved": True,
                "message": None,
                "pre_approved": "permanent",
                "description": "recursive file deletion",
            },
        )
        assert "'always' approval" in result.get("approval", "")

    def test_allowlist_note_names_matched_rule(self, monkeypatch):
        result = self._run(
            monkeypatch,
            {
                "approved": True,
                "message": None,
                "pre_approved": "allowlist",
                "pre_approved_rule": "podman *",
            },
        )
        assert "command_allowlist" in result.get("approval", "")
        assert "podman *" in result["approval"]

    def test_plain_approval_has_no_note(self, monkeypatch):
        result = self._run(monkeypatch, {"approved": True, "message": None})
        assert "approval" not in result
