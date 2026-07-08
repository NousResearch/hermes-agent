import json
from pathlib import Path

import model_tools


def _events_for(tmp_path: Path) -> list[dict]:
    event_files = sorted((tmp_path / "ops" / "events").glob("*.jsonl"))
    assert event_files
    return [json.loads(line) for line in event_files[-1].read_text(encoding="utf-8").splitlines()]


def test_handle_function_call_emits_privacy_preserving_tool_call_event(monkeypatch, tmp_path):
    from tools.registry import registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda name, args, **kw: json.dumps({"output": "safe result"}),
    )
    monkeypatch.setattr(registry, "get_toolset_for_tool", lambda name: "safe_toolset")
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    tool_name = "RAW_TOOL_NAME_DO_NOT_STORE"
    raw_secret_arg = "ARG_VALUE_DO_NOT_STORE_123"
    result = model_tools.handle_function_call(
        tool_name,
        {"secret_arg": raw_secret_arg, "count": 2},
        task_id="task-raw-id",
        session_id="session-raw-id",
        tool_call_id="tool-call-raw-id",
        turn_id="turn-raw-id",
        api_request_id="api-request-raw-id",
        skip_pre_tool_call_hook=True,
    )

    assert result == '{"output": "safe result"}'
    events = _events_for(tmp_path)
    event = events[-1]
    assert event["event_type"] == "tool_call"
    assert event["source"] == "model_tools.handle_function_call"
    assert event["status"] == "ok"
    payload = event["payload"]
    assert payload["tool_name_hash"]
    assert payload["toolset"] == "safe_toolset"
    assert payload["arg_count"] == 2
    assert payload["result_chars"] == len(result)
    assert payload["duration_ms"] >= 0
    assert payload["session_id_hash"]
    assert payload["tool_call_id_hash"]
    assert payload["turn_id_hash"]
    assert payload["api_request_id_hash"]
    assert payload["error_fingerprint"] is None

    raw_file = next((tmp_path / "ops" / "events").glob("*.jsonl")).read_text(encoding="utf-8")
    assert tool_name not in raw_file
    assert raw_secret_arg not in raw_file
    assert "task-raw-id" not in raw_file
    assert "session-raw-id" not in raw_file
    assert "tool-call-raw-id" not in raw_file
    assert "turn-raw-id" not in raw_file
    assert "api-request-raw-id" not in raw_file


def test_handle_function_call_tool_call_event_groups_errors_without_raw_error(monkeypatch, tmp_path):
    from tools.registry import registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda name, args, **kw: json.dumps({"error": "SECRET_ERROR_VALUE_DO_NOT_STORE"}),
    )
    monkeypatch.setattr(registry, "get_toolset_for_tool", lambda name: "dummy_toolset")
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    model_tools.handle_function_call(
        "dummy_error_tool",
        {"path": "/private/path/should/not/store"},
        skip_pre_tool_call_hook=True,
    )

    event = _events_for(tmp_path)[-1]
    assert event["event_type"] == "tool_call"
    assert event["status"] == "error"
    payload = event["payload"]
    assert payload["error_type"] == "tool_error"
    assert payload["error_fingerprint"]
    raw_file = next((tmp_path / "ops" / "events").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "SECRET_ERROR_VALUE_DO_NOT_STORE" not in raw_file
    assert "/private/path/should/not/store" not in raw_file
    assert "dummy_error_tool" not in raw_file
