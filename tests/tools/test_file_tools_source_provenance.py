"""Exact source-provenance proofs from presented bounded file reads."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


def test_successful_bounded_read_issues_opaque_grant(tmp_path):
    from agent.source_provenance import SourceProvenanceRegistry, activate_source_provenance
    from tools.file_tools import read_file_tool

    source = tmp_path / "source.py"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = SourceProvenanceRegistry()
    with activate_source_provenance(
        registry,
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        policy_digest="policy-1",
    ):
        result = json.loads(read_file_tool(str(source), offset=2, limit=1))

    assert result["content"]
    grant = registry.grants_for_request("request-1")[0]
    assert (grant.line_start, grant.line_end) == (2, 2)
    assert grant.content_sha256 == __import__("hashlib").sha256(b"two\n").hexdigest()



def test_read_errors_or_untrusted_tool_text_cannot_issue_grants(tmp_path):
    from agent.source_provenance import SourceProvenanceRegistry, activate_source_provenance
    from tools.file_tools import read_file_tool

    registry = SourceProvenanceRegistry()
    with activate_source_provenance(
        registry,
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        policy_digest="policy-1",
    ):
        read_file_tool(str(tmp_path / "missing.py"), offset=1, limit=1)

    assert registry.grants_for_request("request-1") == ()



def test_symlink_read_cannot_issue_a_grant(tmp_path):
    from agent.source_provenance import SourceProvenanceRegistry, activate_source_provenance
    from tools.file_tools import read_file_tool

    source = tmp_path / "source.py"
    source.write_text("one\n", encoding="utf-8")
    linked = tmp_path / "linked.py"
    linked.symlink_to(source)
    registry = SourceProvenanceRegistry()

    with activate_source_provenance(
        registry,
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        policy_digest="policy-1",
    ):
        result = json.loads(read_file_tool(str(linked), offset=1, limit=1))

    assert result["content"]
    assert registry.grants_for_request("request-1") == ()



def test_symlinked_ancestor_read_cannot_issue_a_grant(tmp_path):
    from agent.source_provenance import SourceProvenanceRegistry, activate_source_provenance
    from tools.file_tools import read_file_tool

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    source = real_dir / "source.py"
    source.write_text("one\n", encoding="utf-8")
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    registry = SourceProvenanceRegistry()

    with activate_source_provenance(
        registry,
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        policy_digest="policy-1",
    ):
        result = json.loads(read_file_tool(str(linked_dir / "source.py"), offset=1, limit=1))

    assert result["content"]
    assert registry.grants_for_request("request-1") == ()



def test_line_display_truncation_cannot_issue_a_grant(tmp_path):
    from agent.source_provenance import SourceProvenanceRegistry, activate_source_provenance
    from tools.file_tools import read_file_tool

    source = tmp_path / "source.py"
    source.write_text("x" * 20_000 + "\n", encoding="utf-8")
    registry = SourceProvenanceRegistry()

    with activate_source_provenance(
        registry,
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        policy_digest="policy-1",
    ):
        result = json.loads(read_file_tool(str(source), offset=1, limit=1))

    assert result["content"]
    assert registry.grants_for_request("request-1") == ()



def test_forged_read_result_cannot_issue_a_grant(tmp_path, monkeypatch):
    from agent.source_provenance import SourceProvenanceRegistry, activate_source_provenance
    from tools.file_operations import ReadResult
    from tools.file_tools import read_file_tool

    source = tmp_path / "source.py"
    source.write_text("actual\n", encoding="utf-8")

    class ForgedFileOps:
        env = None

        @staticmethod
        def read_file(*_args):
            return ReadResult(content="1|forged", total_lines=1, file_size=7)

        @staticmethod
        def _add_line_numbers(content, offset):
            return f"{offset}|{content}"

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda _task_id: ForgedFileOps())
    registry = SourceProvenanceRegistry()
    with activate_source_provenance(
        registry,
        session_id="session-1",
        turn_id="turn-1",
        request_id="request-1",
        policy_digest="policy-1",
    ):
        read_file_tool(str(source), offset=1, limit=1)

    assert registry.grants_for_request("request-1") == ()
