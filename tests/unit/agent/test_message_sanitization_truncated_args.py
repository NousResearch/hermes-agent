"""Test message sanitization, especially for truncated tool_call arguments.

This module tests `_repair_tool_call_arguments` behavior with:
- Truncated JSON that cannot be repaired
- Structured error responses for write_file/create_file
- Empty object fallback for other tools
"""
import json

from agent.message_sanitization import _repair_tool_call_arguments


def test_truncated_write_file_returns_structured_error():
    """Truncated write_file JSON returns structured error, not empty object."""
    # Simulate GLM-5.2 truncation: path present, content string cut off
    truncated = '{"path": "/tmp/script.py", "content": "#!/usr/bin/env python3\n# -*- coding: utf-8-*-\n"""'
    result = _repair_tool_call_arguments(truncated, 'write_file')
    parsed = json.loads(result)

    assert 'error' in parsed, f"Expected 'error' key, got: {parsed}"
    assert 'write_file' in parsed['error'], f"Error should mention write_file: {parsed['error']}"
    assert 'truncated' in parsed['error'], f"Error should mention truncation: {parsed['error']}"
    assert 're-emit' in parsed['error'], f"Error should prompt re-emission: {parsed['error']}"


def test_truncated_create_file_returns_structured_error():
    """Truncated create_file JSON returns structured error."""
    truncated = '{"content": "incomplete'
    result = _repair_tool_call_arguments(truncated, 'create_file')
    parsed = json.loads(result)

    assert 'error' in parsed, f"Expected 'error' key, got: {parsed}"
    assert 'create_file' in parsed['error']


def test_truncated_write_file_with_path_detects_missing_content():
    """When path is extractable, error mentions only content is missing."""
    truncated = '{"path": "/tmp/test.py", "content": "unfinished'
    result = _repair_tool_call_arguments(truncated, 'write_file')
    parsed = json.loads(result)

    assert 'error' in parsed
    # Path was extracted, so error should only mention content
    assert 'content' in parsed['error']
    # Path should not be in missing list
    error_msg = parsed['error']
    # Check that "content" appears but "path" doesn't appear in the "Missing required field(s)" part
    missing_part = error_msg[error_msg.index('Missing required field(s)'):]
    assert 'content' in missing_part
    assert 'path, content' not in missing_part  # Should not mention both


def test_truncated_write_file_without_path_detects_both_missing():
    """When path is not extractable, error mentions both path and content."""
    truncated = '{"content": "only content field'
    result = _repair_tool_call_arguments(truncated, 'write_file')
    parsed = json.loads(result)

    assert 'error' in parsed
    error_msg = parsed['error']
    missing_part = error_msg[error_msg.index('Missing required field(s)'):]
    # Both path and content should be mentioned
    assert 'path' in missing_part
    assert 'content' in missing_part


def test_other_tools_return_empty_object():
    """Non-write_file tools still return empty object on unrepairable args."""
    truncated = '{"incomplete": "json'
    result = _repair_tool_call_arguments(truncated, 'terminal')
    parsed = json.loads(result)

    assert parsed == {}, f"Expected empty object, got: {parsed}"


def test_repairable_truncation_still_works():
    """Repairable truncation (e.g., missing closing brace) is still fixed."""
    truncated = '{"path": "/tmp/test.py", "content": "hello"'  # Missing closing brace and quote
    result = _repair_tool_call_arguments(truncated, 'write_file')
    parsed = json.loads(result)

    # This should be repaired, not an error
    assert 'error' not in parsed, f"Unexpected error: {parsed}"
    assert parsed.get('path') == '/tmp/test.py'
    assert parsed.get('content') == 'hello'