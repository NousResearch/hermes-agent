"""Tests for orchestrator_packet_write content redaction in run_agent persistence."""
import json
import inspect
import pytest


# ---------------------------------------------------------------------------
# Task 3: _scrub_packet_tool_args helper — unit tests
# ---------------------------------------------------------------------------

def test_scrub_removes_content_from_packet_write_call():
    """content value must be replaced with redaction marker in orchestrator_packet_write arguments."""
    from run_agent import _scrub_packet_tool_args, _PACKET_CONTENT_REDACT_MARKER
    tool_calls = [
        {
            "name": "orchestrator_packet_write",
            "arguments": json.dumps({
                "filename": "pkt_20260425_081637_1519.md",
                "content": "TOP_SECRET_PACKET_BODY",
                "overwrite": False,
            }),
        }
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    result_str = json.dumps(scrubbed)
    assert "TOP_SECRET_PACKET_BODY" not in result_str, (
        "raw content must be absent from persisted tool_calls arguments"
    )
    args = json.loads(scrubbed[0]["arguments"])
    assert args["filename"] == "pkt_20260425_081637_1519.md"
    # content key must exist with the stable marker, not be deleted
    assert "content" in args, "content key must be preserved with redaction marker"
    assert args["content"] == _PACKET_CONTENT_REDACT_MARKER
    assert _PACKET_CONTENT_REDACT_MARKER in result_str


def test_scrub_leaves_other_tools_unchanged():
    """_scrub_packet_tool_args must not modify non-packet-write calls."""
    from run_agent import _scrub_packet_tool_args
    tool_calls = [
        {"name": "terminal", "arguments": json.dumps({"command": "ls -la"})},
        {"name": "read_file", "arguments": json.dumps({"path": "/tmp/x.txt"})},
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    assert json.dumps(scrubbed) == json.dumps(tool_calls)


def test_scrub_handles_empty_list():
    from run_agent import _scrub_packet_tool_args
    assert _scrub_packet_tool_args([]) == []
    assert _scrub_packet_tool_args(None) is None


def test_scrub_handles_malformed_arguments_safely():
    """Malformed JSON in arguments must be fully redacted, not crash."""
    from run_agent import _scrub_packet_tool_args
    tool_calls = [
        {
            "name": "orchestrator_packet_write",
            "arguments": "NOT_VALID_JSON{content: secret}",
        }
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    result_str = json.dumps(scrubbed)
    assert "secret" not in result_str


def test_scrub_handles_mixed_list():
    """A list containing both packet-write and other calls is handled correctly."""
    from run_agent import _scrub_packet_tool_args
    secret = "MIXED_LIST_SECRET"
    tool_calls = [
        {"name": "terminal", "arguments": json.dumps({"command": "echo hi"})},
        {
            "name": "orchestrator_packet_write",
            "arguments": json.dumps({"filename": "p.md", "content": secret}),
        },
        {"name": "read_file", "arguments": json.dumps({"path": "/tmp/y.txt"})},
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    result_str = json.dumps(scrubbed)
    assert secret not in result_str
    assert json.loads(scrubbed[0]["arguments"])["command"] == "echo hi"
    assert json.loads(scrubbed[2]["arguments"])["path"] == "/tmp/y.txt"


def test_scrub_handles_anthropic_content_blocks():
    """_scrub_packet_tool_args must also scrub Anthropic-format tool_use blocks."""
    from run_agent import _scrub_packet_tool_args, _PACKET_CONTENT_REDACT_MARKER
    secret = "ANTHROPIC_CONTENT_SECRET"
    blocks = [
        {
            "type": "tool_use",
            "id": "toolu_001",
            "name": "orchestrator_packet_write",
            "input": {"filename": "p.md", "content": secret},
        },
        {
            "type": "text",
            "text": "some text",
        },
    ]
    scrubbed = _scrub_packet_tool_args(blocks)
    result_str = json.dumps(scrubbed)
    assert secret not in result_str
    # filename must survive
    assert scrubbed[0]["input"]["filename"] == "p.md"
    # content key must exist with marker, not be deleted
    assert "content" in scrubbed[0]["input"], "content key must be preserved with redaction marker"
    assert scrubbed[0]["input"]["content"] == _PACKET_CONTENT_REDACT_MARKER
    assert _PACKET_CONTENT_REDACT_MARKER in result_str
    # non-packet block unchanged
    assert scrubbed[1]["text"] == "some text"


# ---------------------------------------------------------------------------
# Task 3: persistence integration — source inspection tests
#
# These tests use inspect.getsource to verify that _scrub_packet_tool_args
# is wired into the persistence functions. They:
# - FAIL before _scrub_packet_tool_args is added (ImportError)
# - FAIL after helper is added but before wiring (AssertionError: not in source)
# - PASS after wiring (helper appears in function source)
# ---------------------------------------------------------------------------


def test_flush_messages_sqlite_row_excludes_content():
    """_flush_messages_to_session_db source must reference _scrub_packet_tool_args."""
    from run_agent import _scrub_packet_tool_args, AIAgent
    source = inspect.getsource(AIAgent._flush_messages_to_session_db)
    assert "_scrub_packet_tool_args" in source, (
        "_scrub_packet_tool_args is not wired into _flush_messages_to_session_db. "
        "Add `tool_calls_data = _scrub_packet_tool_args(tool_calls_data)` after "
        "tool_calls_data is built (run_agent.py:3189)."
    )


def test_save_session_log_json_excludes_content():
    """_save_session_log source must reference _scrub_packet_tool_args."""
    from run_agent import _scrub_packet_tool_args, AIAgent
    source = inspect.getsource(AIAgent._save_session_log)
    assert "_scrub_packet_tool_args" in source, (
        "_scrub_packet_tool_args is not wired into _save_session_log. "
        "Add scrubbing call in the cleaned messages loop (run_agent.py:3704+)."
    )


def test_resume_path_delegates_to_scrubbing_helpers():
    """_persist_session must delegate to _flush_messages_to_session_db or _save_session_log."""
    from run_agent import _scrub_packet_tool_args, AIAgent
    source = inspect.getsource(AIAgent._persist_session)
    assert (
        "_flush_messages_to_session_db" in source
        or "_save_session_log" in source
    ), (
        "_persist_session must delegate to _flush_messages_to_session_db or "
        "_save_session_log — otherwise transcript redaction is bypassed on the resume path."
    )


# ---------------------------------------------------------------------------
# Finding 2: cli.py save_conversation and branch copy redaction
# ---------------------------------------------------------------------------

def _read_cli_source() -> str:
    """Read cli.py source without importing (avoids prompt_toolkit dependency)."""
    from pathlib import Path
    cli_path = Path(__file__).parents[2] / "cli.py"
    return cli_path.read_text(encoding="utf-8")


def test_save_conversation_redacts_packet_content():
    """HermesCLI.save_conversation source must reference _scrub_packet_tool_args."""
    source = _read_cli_source()
    # Verify the reference appears inside the save_conversation method
    start = source.find("def save_conversation(self):")
    assert start != -1, "save_conversation method not found in cli.py"
    # Find the next method definition to bound the search
    next_def = source.find("\n    def ", start + 1)
    snippet = source[start:next_def] if next_def != -1 else source[start:]
    assert "_scrub_packet_tool_args" in snippet, (
        "_scrub_packet_tool_args is not called in HermesCLI.save_conversation. "
        "Raw packet content would persist in exported conversation JSON files."
    )


def test_branch_command_redacts_packet_content():
    """HermesCLI._handle_branch_command source must reference _scrub_packet_tool_args."""
    source = _read_cli_source()
    start = source.find("def _handle_branch_command(self,")
    assert start != -1, "_handle_branch_command method not found in cli.py"
    next_def = source.find("\n    def ", start + 1)
    snippet = source[start:next_def] if next_def != -1 else source[start:]
    assert "_scrub_packet_tool_args" in snippet, (
        "_scrub_packet_tool_args is not called in HermesCLI._handle_branch_command. "
        "Raw packet content would persist in the branch session database copy."
    )


# ---------------------------------------------------------------------------
# Finding 4: recursive scrubbing of nested content arrays
# ---------------------------------------------------------------------------

def test_scrub_recurses_into_tool_result_content():
    """Nested tool_use blocks inside tool_result content arrays must be scrubbed."""
    from run_agent import _scrub_packet_tool_args, _PACKET_CONTENT_REDACT_MARKER
    secret = "NESTED_CONTENT_SECRET"
    blocks = [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_001",
            "content": [
                {
                    "type": "tool_use",
                    "name": "orchestrator_packet_write",
                    "input": {"filename": "p.md", "content": secret},
                }
            ],
        }
    ]
    scrubbed = _scrub_packet_tool_args(blocks)
    result_str = json.dumps(scrubbed)
    assert secret not in result_str, (
        "content inside a nested tool_use within tool_result must be scrubbed"
    )
    inner = scrubbed[0]["content"][0]
    assert inner["input"]["filename"] == "p.md"
    # content key must exist with marker, not be deleted
    assert "content" in inner["input"], "content key must be preserved with redaction marker"
    assert inner["input"]["content"] == _PACKET_CONTENT_REDACT_MARKER
    assert _PACKET_CONTENT_REDACT_MARKER in result_str


# ---------------------------------------------------------------------------
# OpenAI Responses API shape: {type: "function", function: {name, arguments}}
# ---------------------------------------------------------------------------

def test_scrub_responses_api_json_string_arguments():
    """Responses API shape with JSON-string arguments must have content replaced with marker."""
    from run_agent import _scrub_packet_tool_args, _PACKET_CONTENT_REDACT_MARKER
    secret = "RESPONSES_API_SECRET"
    tool_calls = [
        {
            "type": "function",
            "id": "call_abc123",
            "function": {
                "name": "orchestrator_packet_write",
                "arguments": json.dumps({
                    "filename": "pkt_test.md",
                    "content": secret,
                    "overwrite": False,
                }),
            },
        }
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    result_str = json.dumps(scrubbed)
    assert secret not in result_str, (
        "raw content must be absent from Responses API shape arguments"
    )
    fn = scrubbed[0]["function"]
    args = json.loads(fn["arguments"])
    assert args["filename"] == "pkt_test.md"
    # content key must exist with marker, not be deleted
    assert "content" in args, "content key must be preserved with redaction marker"
    assert args["content"] == _PACKET_CONTENT_REDACT_MARKER
    assert _PACKET_CONTENT_REDACT_MARKER in result_str
    # non-content fields on the call must be preserved
    assert scrubbed[0]["id"] == "call_abc123"
    assert scrubbed[0]["type"] == "function"


def test_scrub_responses_api_dict_arguments():
    """Responses API shape where arguments is already a dict must have content replaced with marker."""
    from run_agent import _scrub_packet_tool_args, _PACKET_CONTENT_REDACT_MARKER
    secret = "RESPONSES_API_DICT_SECRET"
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "orchestrator_packet_write",
                "arguments": {
                    "filename": "pkt_dict.md",
                    "content": secret,
                },
            },
        }
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    result_str = json.dumps(scrubbed)
    assert secret not in result_str
    fn = scrubbed[0]["function"]
    # arguments stays as dict when input was dict
    assert fn["arguments"]["filename"] == "pkt_dict.md"
    # content key must exist with marker, not be deleted
    assert "content" in fn["arguments"], "content key must be preserved with redaction marker"
    assert fn["arguments"]["content"] == _PACKET_CONTENT_REDACT_MARKER
    assert _PACKET_CONTENT_REDACT_MARKER in result_str


def test_scrub_responses_api_malformed_arguments():
    """Malformed JSON string in Responses API arguments must be fully redacted, not crash."""
    from run_agent import _scrub_packet_tool_args
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "orchestrator_packet_write",
                "arguments": "NOT_VALID_JSON{content: secret_value}",
            },
        }
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    result_str = json.dumps(scrubbed)
    assert "secret_value" not in result_str
    fn = scrubbed[0]["function"]
    args = json.loads(fn["arguments"])
    assert args.get("_redacted") is True


def test_scrub_responses_api_non_packet_call_unchanged():
    """Responses API shape for a non-packet-write tool must not be modified."""
    from run_agent import _scrub_packet_tool_args
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": json.dumps({"command": "ls -la"}),
            },
        }
    ]
    scrubbed = _scrub_packet_tool_args(tool_calls)
    assert json.dumps(scrubbed) == json.dumps(tool_calls)
