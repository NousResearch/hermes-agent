"""Tests for Hermes-native large tool-result handles.

Large tool outputs should not be replayed verbatim in conversation history when
handle clearing is enabled: the full result is persisted under HERMES_HOME and a
compact preview+handle is returned to the agent loop.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import model_tools
from hermes_state import SessionDB


def _enable_handles(tmp_path, *, threshold=40, preview=12):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "tool_results:\n"
        "  handles_enabled: true\n"
        f"  handle_threshold_bytes: {threshold}\n"
        f"  preview_bytes: {preview}\n",
        encoding="utf-8",
    )
    return hermes_home


def test_large_tool_result_is_replaced_by_compact_handle(monkeypatch, tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=40, preview=10)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from tools.registry import registry

    original = "0123456789" * 20
    monkeypatch.setattr(registry, "dispatch", lambda name, args, **kw: original)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    out = model_tools.handle_function_call(
        "read_file",
        {"path": "big.txt"},
        task_id="task1",
        session_id="sess1",
        tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
    )

    payload = json.loads(out)
    assert payload["type"] == "hermes_tool_result_handle"
    assert payload["tool_name"] == "read_file"
    assert payload["bytes"] == len(original.encode("utf-8"))
    assert payload["preview"]["head"] == "0123456789"
    assert payload["preview"]["tail"] == "0123456789"
    assert original not in out

    assert "path" not in payload
    assert payload["relative_path"].startswith("tool-results/")

    artifact_path = hermes_home / payload["relative_path"]
    assert artifact_path.is_file()
    stored = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert stored["content"] == original
    assert stored["tool_name"] == "read_file"
    assert stored["session_id"] == "sess1"
    assert stored["tool_call_id"] == "tc1"
    assert stored["relative_path"] == payload["relative_path"]


def test_recover_tool_result_returns_exact_content_and_rejects_tampered_hash(monkeypatch, tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=40, preview=10)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.tool_result_handles import maybe_handle_tool_result, recover_tool_result

    original = "abcdef" * 50
    handle = maybe_handle_tool_result(
        original,
        tool_name="terminal",
        args={"command": "noisy"},
        task_id="task1",
        session_id="sess1",
        tool_call_id="tc1",
    )
    payload = json.loads(handle)

    assert recover_tool_result(handle) == original
    assert recover_tool_result(payload["id"]) == original

    payload["sha256"] = "0" * 64
    assert recover_tool_result(payload) is None


def test_state_db_e2e_persists_compact_handle_not_raw_output(monkeypatch, tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=40, preview=8)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.tool_result_handles import recover_tool_result
    from tools.registry import registry

    raw_output = "NOISY-LINE\n" * 200
    monkeypatch.setattr(registry, "dispatch", lambda name, args, **kw: raw_output)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    compact = model_tools.handle_function_call(
        "terminal",
        {"command": "make noise"},
        task_id="task1",
        session_id="sess-e2e",
        tool_call_id="tc-e2e",
        skip_pre_tool_call_hook=True,
    )
    payload = json.loads(compact)
    assert payload["type"] == "hermes_tool_result_handle"

    db = SessionDB(db_path=hermes_home / "state.db")
    db.create_session("sess-e2e", "cli")
    db.append_message(
        "sess-e2e",
        "tool",
        compact,
        tool_name="terminal",
        tool_call_id="tc-e2e",
    )

    stored_content = sqlite3.connect(hermes_home / "state.db").execute(
        "SELECT content FROM messages WHERE session_id = ? AND role = ?",
        ("sess-e2e", "tool"),
    ).fetchone()[0]

    assert json.loads(stored_content)["id"] == payload["id"]
    assert "NOISY-LINE" not in stored_content
    assert len(stored_content.encode("utf-8")) < len(raw_output.encode("utf-8"))
    assert recover_tool_result(stored_content) == raw_output


def test_cli_tool_result_get_recovers_full_content(monkeypatch, tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=40, preview=8)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.tool_result_handles import maybe_handle_tool_result

    raw_output = "CLI-RECOVER-ME\n" * 40
    compact = maybe_handle_tool_result(
        raw_output,
        tool_name="terminal",
        args={"command": "make cli noise"},
        session_id="sess-cli",
        tool_call_id="tc-cli",
    )
    handle_id = json.loads(compact)["id"]

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "tool-result", "get", handle_id],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == raw_output


def test_cli_tool_result_get_json_prints_artifact_metadata(monkeypatch, tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=40, preview=8)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.tool_result_handles import maybe_handle_tool_result

    raw_output = "CLI-JSON-ME\n" * 40
    compact = maybe_handle_tool_result(
        raw_output,
        tool_name="terminal",
        session_id="sess-cli-json",
        tool_call_id="tc-cli-json",
    )
    handle_id = json.loads(compact)["id"]

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "tool-result", "get", handle_id, "--json"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["id"] == handle_id
    assert payload["content"] == raw_output
    assert payload["tool_name"] == "terminal"
    assert payload["relative_path"].startswith("tool-results/")
    assert payload["path"] == str(hermes_home / payload["relative_path"])


def test_cli_tool_result_get_missing_handle_exits_with_not_found(tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=40, preview=8)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "tool-result", "get", "trh_missing"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "not found" in result.stderr.lower()


def test_small_tool_result_is_left_unchanged_when_handles_enabled(monkeypatch, tmp_path):
    hermes_home = _enable_handles(tmp_path, threshold=1000, preview=10)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from tools.registry import registry

    original = '{"ok": true}'
    monkeypatch.setattr(registry, "dispatch", lambda name, args, **kw: original)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    out = model_tools.handle_function_call(
        "read_file",
        {"path": "small.txt"},
        task_id="task1",
        session_id="sess1",
        tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
    )

    assert out == original
    assert not (hermes_home / "tool-results").exists()


def test_large_tool_result_is_left_unchanged_by_default(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from tools.registry import registry

    original = "x" * 200
    monkeypatch.setattr(registry, "dispatch", lambda name, args, **kw: original)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    out = model_tools.handle_function_call(
        "read_file",
        {"path": "big.txt"},
        task_id="t1",
        session_id="sess1",
        tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
    )

    assert out == original
    assert not (hermes_home / "tool-results").exists()
