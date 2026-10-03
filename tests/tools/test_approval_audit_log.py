"""Tests for the approval audit-log hook (``_log_approval_event`` and its
wiring into ``request_tool_approval`` / ``check_all_command_guards`` /
``check_execute_code_guard``).

Covers the review findings from PR #90186: redaction-failure fallback must
never leak the raw command, the session key must not leak PII, the log file
must resist a planted symlink and be non-world-readable, writes must never
raise into the safety-critical approval path, and the hook must actually be
reachable from the real production entry points (not just the unused
``check_dangerous_command`` path).
"""

from __future__ import annotations

import json
import os
import stat

import pytest

import tools.approval as approval


def _read_records(log_path):
    with open(log_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_KEY", "whatsapp:+15551234567")
    yield tmp_path


class TestRedactionFailureNeverLeaksRawCommand:
    def test_redact_exception_falls_back_to_placeholder_not_raw_text(self, tmp_path, monkeypatch):
        secret_command = "curl -H 'Authorization: Bearer sk-supersecrettoken12345' https://internal"

        def _boom(*a, **k):
            raise RuntimeError("redaction exploded")

        monkeypatch.setattr("agent.redact.redact_sensitive_text", _boom)

        approval._log_approval_event(
            pattern_key="pk", description="desc", command=secret_command,
            result={"approved": True}, surface="test",
        )

        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        assert len(records) == 1
        # The whole point of redaction: a failure must NOT dump raw command
        # text (which is exactly where a secret would live) to disk.
        assert "sk-supersecrettoken12345" not in records[0]["command"]
        assert records[0]["command"] == "[redaction unavailable, %d chars omitted]" % len(secret_command)

    def test_normal_redaction_path_still_masks_secrets(self, tmp_path):
        # Sanity check: the happy path (no forced failure) still redacts.
        approval._log_approval_event(
            pattern_key="pk", description="desc",
            command="curl -H 'Authorization: Bearer sk-live-abcdefghijklmnop' https://x",
            result={"approved": True}, surface="test",
        )
        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        assert "sk-live-abcdefghijklmnop" not in records[0]["command"]


class TestSessionKeyIsHashedNotRaw:
    def test_raw_session_key_never_written(self, tmp_path):
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )
        log_path = tmp_path / "logs" / "approvals.jsonl"
        raw_bytes = log_path.read_bytes()
        # The session key set by the fixture embeds a phone number — a raw
        # copy in this file would leak PII (gateway/slash_commands.py
        # applies the same precaution to Matrix session keys).
        assert b"+15551234567" not in raw_bytes
        assert b"whatsapp:" not in raw_bytes

    def test_session_key_hash_is_stable_and_prefixed(self, tmp_path):
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )
        approval._log_approval_event(
            pattern_key="pk2", description="desc2", command="pwd",
            result={"approved": True}, surface="test",
        )
        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        assert records[0]["session_key"].startswith("sha256:")
        # Same session -> same fingerprint, so mining can still correlate
        # events within a session without ever seeing the raw identity.
        assert records[0]["session_key"] == records[1]["session_key"]

    def test_empty_session_key_yields_empty_string_not_a_hash_of_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_KEY", "")
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )
        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        assert records[0]["session_key"] == ""


class TestFilePermissionsAndSymlinkConfinement:
    def test_log_file_created_with_0600(self, tmp_path):
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )
        log_path = tmp_path / "logs" / "approvals.jsonl"
        mode = stat.S_IMODE(os.stat(log_path).st_mode)
        assert mode == 0o600

    def test_preexisting_permissive_file_is_tightened(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        log_path = logs_dir / "approvals.jsonl"
        log_path.write_text("")
        os.chmod(log_path, 0o644)

        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )
        mode = stat.S_IMODE(os.stat(log_path).st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="POSIX symlinks only")
    def test_planted_symlink_is_not_followed(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        victim = tmp_path / "victim.txt"
        victim.write_text("do not touch me\n")
        symlink_path = logs_dir / "approvals.jsonl"
        os.symlink(victim, symlink_path)

        # Must never raise (approval is safety-critical, logging is not)...
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )
        # ...and must not have written through the symlink into the victim.
        assert victim.read_text() == "do not touch me\n"
        # The symlink itself must survive untouched too (O_NOFOLLOW refuses
        # the open outright rather than replacing the link).
        assert os.path.islink(symlink_path)


class TestNeverRaises:
    def test_write_failure_is_swallowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        # Must not raise — approval decisions must never be blocked by a
        # logging failure.
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result={"approved": True}, surface="test",
        )

    def test_non_dict_result_is_swallowed(self, tmp_path):
        approval._log_approval_event(
            pattern_key="pk", description="desc", command="ls",
            result=None, surface="test",
        )
        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        assert records[0]["approved"] is False


class TestAppendAtomicity:
    def test_concurrent_appends_all_survive_as_separate_lines(self, tmp_path):
        import threading

        def _write(i):
            approval._log_approval_event(
                pattern_key=f"pk{i}", description="desc", command=f"cmd {i}",
                result={"approved": True}, surface="test",
            )

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        # Exercise concurrent appends on the test filesystem. This is not a
        # portability guarantee for arbitrary record sizes or network filesystems.
        assert len(records) == 20
        assert {r["pattern_key"] for r in records} == {f"pk{i}" for i in range(20)}


class TestRequestToolApprovalWiresIntoAuditLog:
    def test_logs_the_gate_decision(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            approval, "_run_approval_gate",
            lambda **kw: {"approved": True, "message": None},
        )
        monkeypatch.setattr(
            approval, "_log_approval_event",
            lambda **kw: calls.append(kw),
        )
        result = approval.request_tool_approval("write_file", "writing ~/.ssh/authorized_keys")
        assert result == {"approved": True, "message": None}
        assert len(calls) == 1
        assert calls[0]["surface"] == "tool"
        assert calls[0]["result"] == {"approved": True, "message": None}


class TestCheckAllCommandGuardsWiresIntoAuditLog:
    """check_all_command_guards is the REAL production entry point
    (tools/terminal_tool.py imports it directly) — unlike
    check_dangerous_command, which has no production caller. The audit hook
    must live here for the log to ever receive real data.
    """

    def test_trivial_approve_is_not_logged(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            approval, "_check_all_command_guards_impl",
            lambda *a, **k: {"approved": True, "message": None},
        )
        monkeypatch.setattr(
            approval, "_log_approval_event",
            lambda **kw: calls.append(kw),
        )
        result = approval.check_all_command_guards("echo hi", "local")
        assert result == {"approved": True, "message": None}
        # A no-op decision for an ordinary command must not flood the audit
        # log with every single command ever run.
        assert calls == []

    def test_real_decision_is_logged(self, monkeypatch):
        calls = []
        decision = {
            "approved": False,
            "message": "BLOCKED",
            "pattern_key": "rm_rf",
            "description": "recursive delete",
            "outcome": "denied",
        }
        monkeypatch.setattr(
            approval, "_check_all_command_guards_impl",
            lambda *a, **k: decision,
        )
        monkeypatch.setattr(
            approval, "_log_approval_event",
            lambda **kw: calls.append(kw),
        )
        result = approval.check_all_command_guards("rm -rf /tmp/x", "local")
        assert result == decision
        assert len(calls) == 1
        assert calls[0]["pattern_key"] == "rm_rf"
        assert calls[0]["description"] == "recursive delete"
        assert calls[0]["command"] == "rm -rf /tmp/x"
        assert calls[0]["surface"] == "terminal"
        assert calls[0]["result"] == decision

    def test_end_to_end_writes_a_record_for_a_denied_command(self, tmp_path, monkeypatch):
        # No mocking of the guard internals: drive a real deny-rule block
        # through the actual entry point terminal_tool.py calls, and check
        # a record lands on disk.
        monkeypatch.setattr(approval, "_match_user_deny_rule", lambda command: "rm -rf *")
        result = approval.check_all_command_guards("rm -rf /", "local")
        assert result["approved"] is False
        log_path = tmp_path / "logs" / "approvals.jsonl"
        records = _read_records(log_path)
        assert len(records) == 1
        assert records[0]["surface"] == "terminal"
        assert records[0]["approved"] is False


class TestCheckExecuteCodeGuardWiresIntoAuditLog:
    def test_trivial_approve_is_not_logged(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            approval, "_check_execute_code_guard_impl",
            lambda *a, **k: {"approved": True, "message": None},
        )
        monkeypatch.setattr(
            approval, "_log_approval_event",
            lambda **kw: calls.append(kw),
        )
        result = approval.check_execute_code_guard("print('hi')", "local")
        assert result == {"approved": True, "message": None}
        assert calls == []

    def test_real_decision_is_logged(self, monkeypatch):
        calls = []
        decision = {
            "approved": False,
            "message": "BLOCKED",
            "pattern_key": "execute_code",
            "description": "arbitrary code",
            "outcome": "blocked",
        }
        monkeypatch.setattr(
            approval, "_check_execute_code_guard_impl",
            lambda *a, **k: decision,
        )
        monkeypatch.setattr(
            approval, "_log_approval_event",
            lambda **kw: calls.append(kw),
        )
        result = approval.check_execute_code_guard("import os; os.system('rm -rf /')", "local")
        assert result == decision
        assert len(calls) == 1
        assert calls[0]["pattern_key"] == "execute_code"
        assert calls[0]["surface"] == "execute_code"
        assert calls[0]["result"] == decision
