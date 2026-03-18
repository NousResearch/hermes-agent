"""Tests for the audit_query tool handler."""

import json
import tempfile
from pathlib import Path

from agent.audit import (
    configure,
    start_session,
    end_session,
    log_tool_call,
    log_api_request,
)


def _setup_tmpdir(tmpdir):
    import agent.audit as audit
    old_dir = audit._AUDIT_DIR
    old_link = audit._LATEST_LINK
    audit._AUDIT_DIR = Path(tmpdir)
    audit._LATEST_LINK = Path(tmpdir) / "latest.jsonl"
    return old_dir, old_link


def _teardown(old_dir, old_link):
    import agent.audit as audit
    end_session()
    configure(enabled=False)
    audit._AUDIT_DIR = old_dir
    audit._LATEST_LINK = old_link


class TestAuditQueryTool:
    def test_query_action_returns_events(self):
        from tools.audit_tool import _handle_audit_query
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("aq-001", model="test")
                log_tool_call(tool_name="terminal", args={"command": "ls"}, result="ok")
                end_session()

                result = _handle_audit_query(action="query", session="aq-001")
                assert "terminal" in result
                assert "tool.call" in result
            finally:
                _teardown(old_dir, old_link)

    def test_summary_action(self):
        from tools.audit_tool import _handle_audit_query
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("aq-002", model="test")
                log_api_request(model="test", total_tokens=100, cost_usd=0.005)
                log_tool_call(tool_name="read_file", args={}, result="ok")
                end_session()

                result = _handle_audit_query(action="summary", session="aq-002")
                assert "Audit summary" in result
                assert "100" in result
            finally:
                _teardown(old_dir, old_link)

    def test_problems_action_clean(self):
        from tools.audit_tool import _handle_audit_query
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("aq-003", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                end_session()

                result = _handle_audit_query(action="problems", session="aq-003")
                assert "No problems" in result
            finally:
                _teardown(old_dir, old_link)

    def test_query_empty_returns_no_match(self):
        from tools.audit_tool import _handle_audit_query
        configure(enabled=False)
        result = _handle_audit_query(action="query", tool="nonexistent")
        assert "No matching" in result or "No audit" in result
