"""Tests for the append-only approval / guarded-command audit sink.

The sink is the answer to the security review's un-answerable question — "since
deployment, which attempts targeted a protected path?" — so these tests pin the
properties that make the answer trustworthy:

* the store is INSERT-only (UPDATE/DELETE fail, and a row edited behind the
  triggers' back is caught by the HMAC chain),
* exactly one row per approval decision, carrying trace_id and the classification
  key,
* benign commands leave nothing behind (the backpressure decision),
* ``security.audit.enabled: false`` writes nothing, ever,
* the query surface can filter down to protected-path attempts.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from tools import approval as mod
from tools import approval_audit
from tools.approval_audit import record_event, iter_events, verify_partitions
import tools.approval_context as approval_context

RAW_SECRETS_READ = "cat C:/Users/dangv/AppData/Local/hermes/profiles/default/.env"
BENIGN = "ls -la /tmp"


@pytest.fixture
def audit_store(tmp_path, monkeypatch):
    """Point the sink at a throwaway directory and reset its process-level caches."""
    store = tmp_path / "audit"
    monkeypatch.setattr(approval_audit, "audit_dir", lambda: store)
    monkeypatch.setattr(approval_audit, "_key_cache", None)
    monkeypatch.setattr(approval_audit, "_audit_config", lambda: {})
    approval_audit.clear_audit_config_cache()
    yield store
    approval_audit.clear_audit_config_cache()


@pytest.fixture
def clean_env(monkeypatch):
    """Non-interactive, non-gateway, non-cron, non-yolo baseline."""
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                "HERMES_INTERACTIVE", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)


def _rows(**kwargs):
    return list(iter_events(**kwargs))


class TestAppendOnly:
    def test_update_is_rejected_by_trigger(self, audit_store):
        record_event(surface="terminal", target=BENIGN, decision="deny",
                     class_key="probe")
        with pytest.raises(sqlite3.Error):
            for path in approval_audit.list_partitions():
                conn = sqlite3.connect(path)
                try:
                    conn.execute("UPDATE approval_events SET decision='allow'")
                    conn.commit()
                finally:
                    conn.close()

    def test_delete_is_rejected_by_trigger(self, audit_store):
        record_event(surface="terminal", target=BENIGN, decision="deny",
                     class_key="probe")
        with pytest.raises(sqlite3.Error):
            for path in approval_audit.list_partitions():
                conn = sqlite3.connect(path)
                try:
                    conn.execute("DELETE FROM approval_events")
                    conn.commit()
                finally:
                    conn.close()

    def test_edit_behind_the_triggers_is_caught_by_the_hash_chain(self, audit_store):
        """A tamperer who drops the triggers still cannot change a decision silently."""
        record_event(surface="terminal", target=BENIGN, decision="deny",
                     class_key="probe")
        path = approval_audit.list_partitions()[0]
        assert verify_partitions()[path.stem.split("-", 1)[1]] == "ok"

        conn = sqlite3.connect(path)
        try:
            conn.execute("DROP TRIGGER approval_events_no_update")
            conn.execute("UPDATE approval_events SET decision='allow'")
            conn.commit()
        finally:
            conn.close()
        report = verify_partitions()
        assert report[path.stem.split("-", 1)[1]] != "ok"

    def test_no_write_api_beyond_append(self, audit_store):
        for name in ("update_event", "delete_event", "edit_event"):
            assert not hasattr(approval_audit, name)


class TestWhatGetsRecorded:
    def test_flagged_command_writes_exactly_one_row(self, audit_store, clean_env):
        mod.check_dangerous_command(RAW_SECRETS_READ, "local", None, True)
        rows = _rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["decision"] in ("allow", "deny", "prompt")
        assert row["trace_id"], "trace_id must never be empty (Company OS §14)"
        expected_key, expected_desc = mod.detect_dangerous_command(RAW_SECRETS_READ)[1:]
        assert row["class_key"] == expected_desc
        assert row["pattern_id"] == expected_key
        assert row["surface"] == "terminal"

    def test_nested_gates_still_write_one_row(self, audit_store, clean_env):
        """check_dangerous_command -> _run_approval_gate is ONE decision, not two."""
        mod.check_dangerous_command(RAW_SECRETS_READ, "local", None, True)
        assert len(_rows()) == 1

    def test_hardline_block_is_recorded(self, audit_store, clean_env):
        result = mod.check_dangerous_command("rm -rf /", "local", None, True)
        assert result.get("approved") is False
        rows = _rows()
        assert len(rows) == 1
        assert rows[0]["outcome"] in ("hardline", "blocked", "user_deny")

    def test_execute_code_guard_is_recorded(self, audit_store, clean_env):
        mod.check_execute_code_guard("import os; os.system('id')", "local", True)
        rows = _rows(surface="execute_code")
        assert len(rows) == 1

    def test_benign_command_writes_nothing(self, audit_store, clean_env):
        """Not every call is a row: an allowlist/absent-match outcome is not an attempt."""
        assert mod.check_dangerous_command(BENIGN, "local", None, True).get("approved")
        assert _rows() == []

    def test_raw_command_never_reaches_disk(self, audit_store, clean_env):
        mod.check_dangerous_command(RAW_SECRETS_READ, "local", None, True)
        blob = b"".join(p.read_bytes() for p in approval_audit.list_partitions())
        assert RAW_SECRETS_READ.encode() not in blob
        assert b".env" not in blob
        row = _rows()[0]
        assert len(row["target_digest"]) == 16  # HMAC-SHA256[:8] in hex, keyed, not md5
        assert row["target_len"] == len(RAW_SECRETS_READ)


class TestConfig:
    def test_enabled_false_writes_nothing(self, audit_store, monkeypatch):
        monkeypatch.setattr(approval_audit, "_audit_config",
                            lambda: {"enabled": False})
        assert approval_audit.audit_enabled() is False
        assert record_event(surface="terminal", target=BENIGN, decision="deny") is False
        assert approval_audit.list_partitions() == []

    def test_config_key_is_registered_and_read(self, monkeypatch):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        approval_audit.clear_audit_config_cache()
        audit = DEFAULT_CONFIG["security"]["audit"]
        assert audit["enabled"] is True
        assert isinstance(audit["retention_days"], int) and audit["retention_days"] > 0

        calls = {}

        def fake_readonly():
            calls["read"] = True
            return {"security": {"audit": {"enabled": False, "retention_days": 7}}}

        monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_readonly)
        assert approval_audit.audit_enabled() is False
        assert approval_audit.retention_days() == 7
        assert calls.get("read"), "the reader must go through the config loader"


class TestRetention:
    @staticmethod
    def _make(days_ago: int):
        import datetime as dt
        day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)).strftime("%Y-%m-%d")
        path = approval_audit.partition_path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        approval_audit._open(path, create=True).close()

    def test_prune_removes_whole_stale_partitions_only(self, audit_store):
        for offset in (400, 300, 200):
            self._make(offset)
        assert len(approval_audit.list_partitions()) == 3
        removed = approval_audit.prune_partitions(180)
        assert len(removed) == 3
        assert approval_audit.list_partitions() == []

    def test_fresh_partition_survives_prune(self, audit_store):
        self._make(3)
        assert approval_audit.prune_partitions(180) == []
        assert len(approval_audit.list_partitions()) == 1

    def test_retention_zero_keeps_everything(self, audit_store):
        self._make(400)
        assert approval_audit.prune_partitions(0) == []
        assert len(approval_audit.list_partitions()) == 1


class TestQuerySurface:
    def _args(self, **overrides):
        base = dict(days=0, decision="", surface="", trace_id="", class_like="",
                    protected=False, limit=0, summary=False, verify=False, json=False)
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_lists_recorded_events(self, audit_store, clean_env, capsys):
        from hermes_cli.approvals_audit import approvals_audit_command
        mod.check_dangerous_command(RAW_SECRETS_READ, "local", None, True)
        assert approvals_audit_command(self._args()) == 0
        out = capsys.readouterr().out
        assert "access to Hermes secrets" in out

    def test_protected_filter_answers_the_review_question(self, audit_store, clean_env,
                                                          capsys):
        from hermes_cli.approvals_audit import approvals_audit_command
        mod.check_dangerous_command(RAW_SECRETS_READ, "local", None, True)
        mod.check_dangerous_command("rm -rf /", "local", None, True)
        capsys.readouterr()
        assert approvals_audit_command(self._args(protected=True)) == 0
        out = capsys.readouterr().out
        assert "access to Hermes secrets" in out
        assert "recursive delete" not in out

    def test_decision_and_trace_id_filters(self, audit_store, clean_env, capsys):
        """The decision filter must select on what was actually recorded — the gate's
        verdict itself depends on the surrounding context (single-query vs unattended),
        so the assertion follows the row rather than assuming deny."""
        from hermes_cli.approvals_audit import approvals_audit_command
        mod.check_dangerous_command(RAW_SECRETS_READ, "local", None, True)
        row = _rows()[0]
        capsys.readouterr()

        assert approvals_audit_command(self._args(decision=row["decision"])) == 0
        assert "access to Hermes secrets" in capsys.readouterr().out

        assert approvals_audit_command(self._args(trace_id=row["trace_id"])) == 0
        assert "access to Hermes secrets" in capsys.readouterr().out

        other = "deny" if row["decision"] != "deny" else "allow"
        assert approvals_audit_command(self._args(decision=other)) == 0
        assert "No approval audit events match." in capsys.readouterr().out

    def test_surface_filter(self, audit_store, clean_env, capsys):
        from hermes_cli.approvals_audit import approvals_audit_command
        mod.check_execute_code_guard("import os", "local", True)
        capsys.readouterr()
        assert approvals_audit_command(self._args(surface="terminal")) == 0
        assert "No approval audit events match." in capsys.readouterr().out

    def test_verify_reports_ok(self, audit_store, clean_env, capsys):
        from hermes_cli.approvals_audit import approvals_audit_command
        record_event(surface="terminal", target=BENIGN, decision="deny")
        capsys.readouterr()
        assert approvals_audit_command(self._args(verify=True)) == 0
        assert "0 not ok" in capsys.readouterr().out

    def test_empty_store_is_not_an_error(self, audit_store, capsys):
        from hermes_cli.approvals_audit import approvals_audit_command
        assert approvals_audit_command(self._args()) == 0
        assert "No approval audit events match." in capsys.readouterr().out


class TestFileWriteSurface:
    def test_approval_required_write_is_recorded_on_the_file_surface(self, audit_store,
                                                                     clean_env, monkeypatch):
        """The file-write gate passes its own ``audit_surface``; a write that needs
        approval lands one row under ``file_write`` with the gate's classification."""
        from agent import file_safety
        from tools.file_tools_write_guards import _check_approval_required_write

        # The path predicate is not what this card is under test for: stub it so the
        # gate runs on any candidate, on every platform, hermetically.
        monkeypatch.setattr(file_safety, "is_write_approval_required", lambda p: True)
        _check_approval_required_write(["/home/user/.ssh/config"])

        rows = _rows(surface="file_write")
        assert len(rows) == 1
        assert rows[0]["decision"] in ("allow", "deny", "prompt")
        assert rows[0]["trace_id"]
        assert rows[0]["pattern_id"] == "ssh_config_write"
