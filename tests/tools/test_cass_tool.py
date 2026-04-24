"""Tests for tools/cass_tool.py — CASS CLI wrappers for session intelligence."""

import json
import subprocess
from unittest.mock import patch


class _RunRecorder:
    def __init__(self, returncode=0, stdout=None, stderr=""):
        self.calls = []
        self.returncode = returncode
        self.stdout = stdout if stdout is not None else json.dumps({"ok": True})
        self.stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def test_check_cass_requirements_uses_path_lookup():
    from tools.cass_tool import check_cass_requirements

    with patch("tools.cass_tool.shutil.which", return_value="/usr/local/bin/cass"):
        assert check_cass_requirements() is True

    with patch("tools.cass_tool.shutil.which", return_value=None):
        assert check_cass_requirements() is False


def test_cass_search_builds_robot_json_command_with_filters():
    from tools.cass_tool import cass_search

    runner = _RunRecorder(stdout=json.dumps({"hits": [{"title": "match"}]}))
    with (
        patch("tools.cass_tool.shutil.which", return_value="/usr/local/bin/cass"),
        patch("tools.cass_tool.subprocess.run", runner),
    ):
        result = json.loads(
            cass_search(
                "Sylveste Clavain",
                limit=3,
                mode="lexical",
                workspace="/home/mk/projects/Sylveste",
                agent="claude_code",
                since="2026-04-01",
                until="2026-04-23",
            )
        )

    cmd, kwargs = runner.calls[0]
    assert cmd == [
        "cass",
        "search",
        "--json",
        "--limit",
        "3",
        "--mode",
        "lexical",
        "--workspace",
        "/home/mk/projects/Sylveste",
        "--agent",
        "claude_code",
        "--since",
        "2026-04-01",
        "--until",
        "2026-04-23",
        "--max-content-length",
        "2000",
        "--",
        "Sylveste Clavain",
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert result["success"] is True
    assert result["command"] == "search"
    assert result["data"]["hits"][0]["title"] == "match"


def test_cass_search_rejects_empty_query_before_running_subprocess():
    from tools.cass_tool import cass_search

    with patch("tools.cass_tool.subprocess.run") as run:
        result = json.loads(cass_search("   "))

    run.assert_not_called()
    assert result["success"] is False
    assert "query" in result["error"].lower()


def test_cass_search_clamps_limit_to_bounded_positive_range():
    from tools.cass_tool import cass_search

    runner = _RunRecorder(stdout=json.dumps({"hits": []}))
    with (
        patch("tools.cass_tool.shutil.which", return_value="/usr/local/bin/cass"),
        patch("tools.cass_tool.subprocess.run", runner),
    ):
        json.loads(cass_search("not unbounded", limit=0))
        json.loads(cass_search("not huge", limit=5000))

    assert runner.calls[0][0][runner.calls[0][0].index("--limit") + 1] == "1"
    assert runner.calls[1][0][runner.calls[1][0].index("--limit") + 1] == "25"


def test_cass_search_uses_separator_for_hyphen_leading_query():
    from tools.cass_tool import cass_search

    runner = _RunRecorder(stdout=json.dumps({"hits": []}))
    with (
        patch("tools.cass_tool.shutil.which", return_value="/usr/local/bin/cass"),
        patch("tools.cass_tool.subprocess.run", runner),
    ):
        json.loads(cass_search("-h", limit=1))

    assert runner.calls[0][0][-2:] == ["--", "-h"]


def test_missing_cass_returns_clear_error():
    from tools.cass_tool import cass_status

    with patch("tools.cass_tool.shutil.which", return_value=None):
        result = json.loads(cass_status())

    assert result["success"] is False
    assert "cass" in result["error"].lower()
    assert "install" in result["hint"].lower()


def test_semantic_unavailable_error_suggests_lexical_fallback():
    from tools.cass_tool import cass_search

    payload = {
        "error": {
            "code": 15,
            "kind": "semantic-unavailable",
            "message": "Semantic search not available: vector index missing",
            "hint": "Run cass index --semantic --embedder hash, or use --mode lexical",
            "retryable": False,
        }
    }
    runner = _RunRecorder(returncode=15, stdout=json.dumps(payload))
    with (
        patch("tools.cass_tool.shutil.which", return_value="/usr/local/bin/cass"),
        patch("tools.cass_tool.subprocess.run", runner),
    ):
        result = json.loads(cass_search("intercore", mode="hybrid"))

    assert result["success"] is False
    assert result["code"] == 15
    assert result["kind"] == "semantic-unavailable"
    assert "--mode lexical" in result["hint"]


def test_cass_context_and_analytics_parse_json_results():
    from tools.cass_tool import cass_analytics, cass_context

    runner = _RunRecorder(stdout=json.dumps({"items": []}))
    with (
        patch("tools.cass_tool.shutil.which", return_value="/usr/local/bin/cass"),
        patch("tools.cass_tool.subprocess.run", runner),
    ):
        context_result = json.loads(cass_context("/tmp/session.jsonl", limit=2))
        analytics_result = json.loads(cass_analytics(kind="tokens", days=7, workspace="/repo"))

    assert context_result["success"] is True
    assert analytics_result["success"] is True
    assert runner.calls[0][0] == ["cass", "context", "--json", "--limit", "2", "--", "/tmp/session.jsonl"]
    assert runner.calls[1][0] == [
        "cass",
        "analytics",
        "tokens",
        "--json",
        "--days",
        "7",
        "--workspace",
        "/repo",
    ]
