"""Tool analytics read the canonical tool-result history without writing telemetry."""

import pytest
import hashlib
import os
from pathlib import Path
import subprocess
import sys


from hermes_state import SessionDB
from tools.tool_usage import tool_summary
from utils import fast_safe_load


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session("session-1", source="cli")
    database.create_session("session-2", source="cli")
    yield database
    database.close()


def test_usage_counts_success_failure_and_output_estimate(db):
    db.append_message("session-1", "tool", content='{"ok": true}', tool_name="terminal")
    db.append_message("session-2", "tool", content='{"error": "failed"}', tool_name="terminal")
    db.append_message("session-2", "tool", content="[TOOL_ERROR] refused", tool_name="todo")

    report = tool_summary(db)

    assert report["total_calls"] == 3
    terminal = next(row for row in report["tools"] if row["tool"] == "terminal")
    assert terminal["calls"] == 2
    assert terminal["failures"] == 1
    assert terminal["sessions"] == 2
    assert terminal["estimated_output_tokens"] > 0
    assert next(row for row in report["tools"] if row["tool"] == "todo")["failures"] == 1


def test_compressed_away_results_still_count_as_historical_usage(db):
    db.append_message("session-1", "tool", content="result", tool_name="terminal")
    db._conn.execute("UPDATE messages SET active = 0 WHERE role = 'tool'")
    db._conn.commit()
    assert tool_summary(db)["total_calls"] == 1


@pytest.mark.parametrize("command", ["cmd_analytics", "cmd_prune"])
@pytest.mark.parametrize("value", ["false", "'false'"])
def test_analytics_commands_fail_closed_when_disabled(tmp_path, monkeypatch, capsys, command, value):
    from tools import tool_usage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"tools:\n  analytics: {value}\n", encoding="utf-8")
    getattr(tool_usage, command)()
    assert "disabled" in capsys.readouterr().out


def test_merged_config_keeps_analytics_disabled_and_other_tools_defaults(tmp_path, monkeypatch):
    from hermes_cli.config import load_config_readonly

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config_readonly()
    assert config["tools"]["analytics"] is False
    assert "tool_search" in config["tools"]


@pytest.mark.parametrize("source", ["cli", "local", "legacy"])
def test_prune_includes_never_called_enabled_tools_and_supported_config(db, tmp_path, monkeypatch, source):
    from tools.tool_usage import generate_prune_diff

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("platform_toolsets:\n  cli: [terminal, file, future-toolset]\n", encoding="utf-8")
    for index in range(10):
        sid = f"used-{index}"
        db.create_session(sid, source="cli" if source == "legacy" else source)
        db._conn.execute("UPDATE sessions SET created_source = 'cli' WHERE id = ?", (sid,))
        db._conn.commit()
        db.append_message(sid, "tool", content="ok", tool_name="terminal")

    if source == "legacy":
        db._conn.execute("DROP INDEX idx_sessions_local_created_started")
        db._conn.execute("ALTER TABLE sessions DROP COLUMN created_source")
        db._conn.commit()
    report = generate_prune_diff(db)
    assert {"tool": "read_file", "sessions": 0} in report["suggestions"]
    assert "file" in report["toolsets"]
    assert "terminal" not in report["toolsets"]
    guidance = fast_safe_load(report["diff_hint"].split("\n", 1)[1])
    assert set(guidance["platform_toolsets"]["cli"]) == {"terminal", "file", "future-toolset"}
    assert "future-toolset" not in report["toolsets"]
    assert ".disabled" not in report["diff_hint"]


def test_unnamed_cli_results_resolve_calls_without_double_counting(db):
    for session, name in [("session-1", "terminal"), ("session-2", "todo")]:
        db.append_message(session, "assistant", tool_calls=[
            {"id": "same-id", "type": "function", "function": {"name": name, "arguments": "{}"}}])
        db.append_message(session, "tool", content='{"error": "failed"}', tool_call_id="same-id")
    report = tool_summary(db)
    assert report["total_calls"] == 2
    assert {row["tool"] for row in report["tools"]} == {"terminal", "todo"}
    assert all(row["failures"] == 1 and row["estimated_output_tokens"] > 0 for row in report["tools"])


def test_assistant_only_and_named_result_histories_reconcile_per_session(db):
    for session in ["session-1", "session-2"]:
        db.append_message(session, "assistant", tool_calls=[
            {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}])
    db.append_message("session-1", "tool", content="ok", tool_name="terminal", tool_call_id="call-1")
    report = tool_summary(db)
    assert report["total_calls"] == 2
    assert report["tools"][0]["sessions"] == 2


def test_prune_does_not_treat_an_empty_history_as_evidence(db, tmp_path, monkeypatch):
    from tools.tool_usage import generate_prune_diff

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("platform_toolsets:\n  cli: [terminal, file]\n", encoding="utf-8")
    report = generate_prune_diff(db)
    assert report["suggestions"] == []
    assert report["toolsets"] == []


@pytest.mark.parametrize("command", ["analytics", "prune"])
def test_cli_reads_seeded_history_without_changing_database(tmp_path, command):
    path = tmp_path / "state.db"
    database = SessionDB(db_path=path)
    database.create_session("cli-seed", source="cli")
    for index in range(10):
        database.create_session(f"prior-{index}", source="cli")
    database.append_message("cli-seed", "assistant", tool_calls=[
        {"id": "seed-call", "function": {"name": "terminal", "arguments": "{}"}}])
    database.append_message("cli-seed", "tool", content='{"success": false}', tool_call_id="seed-call")
    database.close()
    (tmp_path / "config.yaml").write_text(
        "tools:\n  analytics: true\nplatform_toolsets:\n  cli: [terminal, file]\n", encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    config_before = (tmp_path / "config.yaml").read_bytes()
    env = {**os.environ, "HERMES_HOME": str(tmp_path)}
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "tools", command],
                            cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    if command == "analytics":
        assert "terminal: 1 calls, 1 failures" in result.stdout
        assert "estimated output tokens" in result.stdout
    else:
        assert "read_file: used in 0 recorded sessions" in result.stdout
        assert "platform_toolsets:" in result.stdout
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert (tmp_path / "config.yaml").read_bytes() == config_before
    assert not (tmp_path / "tool_usage.db").exists()
