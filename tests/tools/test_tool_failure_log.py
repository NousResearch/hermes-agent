"""Tests for tools/tool_failure_log.py and tools/_failure_log_store.py."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools._failure_log_store import (
    _MAX_ARGS_LEN,
    _MAX_ERROR_LEN,
    _next_id,
    append_record,
    auto_log,
    read_all,
    write_records,
)
from tools.tool_failure_log import (
    _MAX_FIX_LEN,
    _VALID_RESOLUTIONS,
    _parse_ids,
    check_requirements,
    tool_failure_log,
)


# =========================================================================
# Helpers
# =========================================================================


def _clean():
    """Remove the failures log so tests start from a known state."""
    from tools._failure_log_store import _log_path
    lp = _log_path()
    if lp.exists():
        lp.unlink()


# =========================================================================
# _failure_log_store
# =========================================================================


class TestNextId:
    def test_empty(self):
        assert _next_id([]) == 1

    def test_sequential(self):
        assert _next_id([{"id": 1}, {"id": 2}, {"id": 3}]) == 4

    def test_gaps(self):
        assert _next_id([{"id": 1}, {"id": 5}, {"id": 3}]) == 6

    def test_missing_id_field(self):
        assert _next_id([{"x": 1}, {"id": 7}]) == 8

    def test_non_int_id(self):
        assert _next_id([{"id": "abc"}, {"id": 3}]) == 4


class TestReadAll:
    def test_empty_log(self):
        _clean()
        assert read_all() == []

    def test_reads_and_reverses(self):
        _clean()
        append_record({"ts": "a", "t": "t1", "e": "e1"})
        append_record({"ts": "b", "t": "t2", "e": "e2"})
        records = read_all()
        assert len(records) == 2
        # newest first
        assert records[0]["t"] == "t2"
        assert records[1]["t"] == "t1"

    def test_skips_malformed_lines(self):
        _clean()
        lp = Path(os.environ["HERMES_HOME"]) / "tool_failures" / "failures.jsonl"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text('{"id":1,"t":"ok","e":"good"}\nnot json\n{"id":2,"t":"ok2","e":"good2"}\n', encoding="utf-8")
        records = read_all()
        assert len(records) == 2
        assert records[0]["id"] == 2

    def test_skips_empty_lines(self):
        _clean()
        lp = Path(os.environ["HERMES_HOME"]) / "tool_failures" / "failures.jsonl"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text('\n{"id":1,"t":"x","e":"y"}\n\n', encoding="utf-8")
        records = read_all()
        assert len(records) == 1


class TestWriteRecords:
    def test_atomic_rewrite(self):
        _clean()
        append_record({"ts": "a", "t": "x", "e": "old"})
        write_records([{"id": 1, "ts": "b", "t": "y", "e": "new", "r": "fixed"}])
        records = read_all()
        assert len(records) == 1
        assert records[0]["e"] == "new"
        assert records[0]["r"] == "fixed"


class TestAppendRecord:
    def test_assigns_id(self):
        _clean()
        rec = {"ts": "z", "t": "test", "e": "err", "r": "pending"}
        saved = append_record(rec)
        assert saved["id"] == 1
        assert saved["t"] == "test"

    def test_id_increments(self):
        _clean()
        r1 = append_record({"ts": "a", "t": "t1", "e": "e1"})
        r2 = append_record({"ts": "b", "t": "t2", "e": "e2"})
        assert r1["id"] == 1
        assert r2["id"] == 2

    def test_does_not_overwrite_caller_id(self):
        """Caller should not set id — append_record assigns it under lock."""
        _clean()
        rec = {"id": 999, "ts": "z", "t": "test", "e": "err"}
        saved = append_record(rec)
        # The lock-assigned id wins
        assert saved["id"] == 1


class TestAutoLog:
    def test_basic(self):
        _clean()
        nid = auto_log("web_search", "timeout", {"query": "test"}, "s1")
        assert nid == 1
        records = read_all()
        assert len(records) == 1
        r = records[0]
        assert r["t"] == "web_search"
        assert r["e"] == "timeout"
        assert r["a"] == '{"query": "test"}'
        assert r["s"] == "s1"
        assert r["f"] == ""
        assert r["l"] == []
        assert r["r"] == "pending"

    def test_args_truncation(self):
        _clean()
        big_args = {"q": "x" * 300}
        auto_log("t", "e", big_args)
        r = read_all()[0]
        assert len(r["a"]) <= _MAX_ARGS_LEN

    def test_error_truncation(self):
        _clean()
        auto_log("t", "e" * 300)
        r = read_all()[0]
        assert len(r["e"]) <= _MAX_ERROR_LEN

    def test_returns_none_on_failure(self, monkeypatch):
        _clean()
        # Break the filesystem so auto_log catches and returns None
        monkeypatch.setattr("tools._failure_log_store._data_dir", lambda: Path("/nonexistent/xyz"))
        result = auto_log("t", "e")
        assert result is None


# =========================================================================
# tool_failure_log — action handlers
# =========================================================================


class TestParseIds:
    def test_all(self):
        assert _parse_ids("all") is None

    def test_single_int(self):
        assert _parse_ids(5) == {5}

    def test_list(self):
        assert _parse_ids([1, 2, 3]) == {1, 2, 3}

    def test_invalid_returns_none(self):
        assert _parse_ids("abc") is None
        assert _parse_ids(None) is None

    def test_mixed_valid_invalid(self):
        assert _parse_ids([1, "x", 3]) == {1, 3}


class TestActionLog:
    def test_basic(self):
        _clean()
        r = json.loads(tool_failure_log(action="log", tool="web_search", error="timeout"))
        assert r["ok"] is True
        assert r["id"] == 1
        assert r["c"] == 1

    def test_missing_tool(self):
        _clean()
        r = json.loads(tool_failure_log(action="log", error="timeout"))
        assert "e" in r
        assert "tool is required" in r["e"]

    def test_missing_error(self):
        _clean()
        r = json.loads(tool_failure_log(action="log", tool="x"))
        assert "e" in r
        assert "error is required" in r["e"]

    def test_with_fix(self):
        _clean()
        tool_failure_log(action="log", tool="t", error="e", fix="used alternative")
        r = read_all()[0]
        assert r["f"] == "used alternative"

    def test_fix_truncation(self):
        _clean()
        tool_failure_log(action="log", tool="t", error="e", fix="x" * 500)
        r = read_all()[0]
        assert len(r["f"]) <= _MAX_FIX_LEN


class TestActionList:
    def test_empty(self):
        _clean()
        r = json.loads(tool_failure_log(action="list"))
        assert r["c"] == 0
        assert r["total"] == 0
        assert r["items"] == []

    def test_pagination(self):
        _clean()
        for i in range(5):
            tool_failure_log(action="log", tool="t", error=f"e{i}")
        r = json.loads(tool_failure_log(action="list", limit=2, offset=1))
        assert r["limit"] == 2
        assert r["offset"] == 1
        assert len(r["items"]) == 2
        assert r["c"] == 5

    def test_status_filter(self):
        _clean()
        tool_failure_log(action="log", tool="t", error="e1")
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e2"))["id"]
        tool_failure_log(action="resolve", ids=nid, resolution="fixed")
        r = json.loads(tool_failure_log(action="list", status="fixed"))
        assert r["c"] == 1
        assert r["items"][0]["r"] == "fixed"

    def test_tool_filter(self):
        _clean()
        tool_failure_log(action="log", tool="web_search", error="e1")
        tool_failure_log(action="log", tool="patch", error="e2")
        r = json.loads(tool_failure_log(action="list", tool="patch"))
        assert r["c"] == 1

    def test_limit_clamped(self):
        _clean()
        r = json.loads(tool_failure_log(action="list", limit=999))
        assert r["limit"] <= 100


class TestActionStats:
    def test_empty(self):
        _clean()
        r = json.loads(tool_failure_log(action="stats"))
        assert r["total"] == 0
        assert r["resolved"] == 0
        assert r["pending"] == 0

    def test_aggregates(self):
        _clean()
        tool_failure_log(action="log", tool="web_search", error="timeout")
        tool_failure_log(action="log", tool="web_search", error="timeout")
        tool_failure_log(action="log", tool="patch", error="no match")
        r = json.loads(tool_failure_log(action="stats"))
        assert r["total"] == 3
        assert r["pending"] == 3
        assert r["tools"] == 2
        assert r["by_tool"]["web_search"]["c"] == 2
        assert r["by_tool"]["web_search"]["top"]["timeout"] == 2

    def test_by_state(self):
        _clean()
        tool_failure_log(action="log", tool="a", error="e1")
        tool_failure_log(action="log", tool="b", error="e2")
        tool_failure_log(action="log", tool="c", error="e3")
        tool_failure_log(action="resolve", ids=1, resolution="fixed")
        tool_failure_log(action="resolve", ids=2, resolution="blocked")
        r = json.loads(tool_failure_log(action="stats"))
        assert r["resolved"] == 2
        assert r["pending"] == 1
        assert r["by_state"]["fixed"] == 1
        assert r["by_state"]["blocked"] == 1


class TestActionResolve:
    def test_single(self):
        _clean()
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e"))["id"]
        r = json.loads(tool_failure_log(action="resolve", ids=nid, resolution="fixed"))
        assert r["ok"] is True
        assert r["updated"] == 1

    def test_batch_list(self):
        _clean()
        ids = []
        for i in range(3):
            ids.append(json.loads(tool_failure_log(action="log", tool="t", error=f"e{i}"))["id"])
        r = json.loads(tool_failure_log(action="resolve", ids=ids, resolution="wontfix"))
        assert r["updated"] == 3

    def test_all_with_filter(self):
        _clean()
        tool_failure_log(action="log", tool="web_search", error="e1")
        tool_failure_log(action="log", tool="patch", error="e2")
        r = json.loads(tool_failure_log(
            action="resolve", ids="all", tool="web_search", status="pending", resolution="fixed"
        ))
        assert r["updated"] == 1

    def test_invalid_resolution(self):
        _clean()
        r = json.loads(tool_failure_log(action="resolve", ids=1, resolution="invalid"))
        assert "e" in r

    def test_no_matching_records(self):
        _clean()
        r = json.loads(tool_failure_log(action="resolve", ids=999))
        assert r["updated"] == 0
        assert "hint" in r


class TestActionUpdate:
    def test_update_fix(self):
        _clean()
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e"))["id"]
        r = json.loads(tool_failure_log(action="update", ids=nid, fix="found workaround"))
        assert r["ok"] is True
        assert r["updated"] == 1
        rec = read_all()[0]
        assert rec["f"] == "found workaround"

    def test_update_appends_fix(self):
        _clean()
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e", fix="first fix"))["id"]
        tool_failure_log(action="update", ids=nid, fix="second fix")
        rec = read_all()[0]
        assert "first fix" in rec["f"]
        assert "second fix" in rec["f"]

    def test_update_with_link_ids(self):
        _clean()
        nid1 = json.loads(tool_failure_log(action="log", tool="a", error="e1"))["id"]
        nid2 = json.loads(tool_failure_log(action="log", tool="b", error="e2"))["id"]
        tool_failure_log(action="update", ids=nid1, link_ids=[nid2])
        rec = [r for r in read_all() if r["id"] == nid1][0]
        assert nid2 in rec["l"]

    def test_no_fix_or_links(self):
        _clean()
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e"))["id"]
        r = json.loads(tool_failure_log(action="update", ids=nid))
        assert "e" in r


class TestActionLink:
    def test_bidirectional(self):
        _clean()
        ids = []
        for i in range(3):
            ids.append(json.loads(tool_failure_log(action="log", tool="t", error=f"e{i}"))["id"])
        r = json.loads(tool_failure_log(action="link", ids=ids))
        assert r["ok"] is True
        assert r["updated"] == 3
        assert sorted(r["linked"]) == ids

        # Verify bidirectional
        records = read_all()
        for rec in records:
            others = set(ids) - {rec["id"]}
            assert set(rec["l"]) == others

    def test_needs_at_least_two(self):
        _clean()
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e"))["id"]
        r = json.loads(tool_failure_log(action="link", ids=[nid]))
        assert "e" in r

    def test_non_list_ids(self):
        r = json.loads(tool_failure_log(action="link", ids=1))
        assert "e" in r


class TestUnknownAction:
    def test_returns_error(self):
        r = json.loads(tool_failure_log(action="nonexistent"))
        assert "e" in r
        assert "unknown action" in r["e"]


# =========================================================================
# check_requirements
# =========================================================================


class TestCheckRequirements:
    def test_always_true(self):
        assert check_requirements() is True


# =========================================================================
# Integration — auto_log → tool chain
# =========================================================================


class TestIntegration:
    def test_auto_log_then_update_then_resolve(self):
        _clean()
        # Simulate auto-log from registry.dispatch
        nid = auto_log("web_search", "timeout", {"q": "test"}, "sess_abc")
        assert nid == 1

        # Agent adds fix
        r = json.loads(tool_failure_log(action="update", ids=nid, fix="used browser"))
        assert r["updated"] == 1

        # Agent links to another auto-logged failure
        nid2 = auto_log("web_search", "HTTP 429", {"q": "test2"}, "sess_abc")
        r = json.loads(tool_failure_log(action="link", ids=[nid, nid2]))
        assert r["updated"] == 2

        # Resolve both
        r = json.loads(tool_failure_log(action="resolve", ids=[nid, nid2], resolution="fixed"))
        assert r["updated"] == 2

        # Verify final state
        records = read_all()
        assert len(records) == 2
        for rec in records:
            assert rec["r"] == "fixed"
            assert len(rec["l"]) == 1  # linked to each other

    def test_multiple_auto_logs_increment_ids(self):
        _clean()
        ids = []
        for i in range(10):
            ids.append(auto_log("t", f"e{i}"))
        assert ids == list(range(1, 11))
        assert len(read_all()) == 10


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_whitespace_tool_name_rejected(self):
        r = json.loads(tool_failure_log(action="log", tool="   ", error="e"))
        assert "e" in r

    def test_very_long_args_output(self):
        _clean()
        tool_failure_log(action="log", tool="t", error="e", args="x" * 500)
        r = read_all()[0]
        assert len(r["a"]) <= _MAX_ARGS_LEN

    def test_list_default_limit(self):
        _clean()
        for i in range(30):
            tool_failure_log(action="log", tool="t", error=f"e{i}")
        r = json.loads(tool_failure_log(action="list"))
        assert len(r["items"]) == 20  # default
        assert r["c"] == 30
        assert r["total"] == 30

    def test_resolve_ids_all_no_filter(self):
        _clean()
        for i in range(5):
            tool_failure_log(action="log", tool="t", error=f"e{i}")
        r = json.loads(tool_failure_log(action="resolve", ids="all", resolution="fixed"))
        assert r["updated"] == 5

    def test_resolve_blocked_state(self):
        _clean()
        nid = json.loads(tool_failure_log(action="log", tool="t", error="e"))["id"]
        r = json.loads(tool_failure_log(action="resolve", ids=nid, resolution="blocked"))
        assert r["updated"] == 1
        rec = read_all()[0]
        assert rec["r"] == "blocked"
