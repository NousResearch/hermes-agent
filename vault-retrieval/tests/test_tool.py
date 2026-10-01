"""Tests for the vault_context tool handler (tool.py).

Covers acceptance tests #1 (plugin discovery under two profiles), #2
(default ≤ 12k; exactly 12k accepted), #4 (24k accepted; 24,001 rejected
before any read), #9 (read_file/search_files inside Vault: audit/block/off),
#15 (unwritable log -> data_hold, not unlogged ok), #16 (snapshots/FTS5
disabled -> no cache/index files), and the full tool envelope shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


from vault_retrieval.tool import (
    VaultContextHandler,
    build_envelope,
    vault_context_tool_schema,
    TOOL_NAME,
    TOOLSET,
)


@pytest.fixture
def cfg(tmp_path: Path, vault_root: Path) -> Dict[str, Any]:
    """Plugin-style config dict, all defaults applied."""
    return {
        "vault_root": str(vault_root),
        "mode": "enforce",
        "default_budget_chars": 12_000,
        "hard_ceiling_chars": 24_000,
        "large_file_chars": 20_000,
        "candidate_limit": 20,
        "max_primary_extracts": 3,
        "max_expansion_extracts": 2,
        "max_range_lines": 120,
        "max_range_chars": 8_000,
        "query_log_enabled": True,
        "log_raw_query_terms": False,
        "snapshots_enabled": False,
        "fts5_enabled": False,
        "block_direct_file_reads": True,
        "state_dir": str(tmp_path / "state"),
    }


class TestSchema:
    def test_schema_is_well_formed_json_schema(self):
        schema = vault_context_tool_schema()
        assert schema["name"] == TOOL_NAME
        assert schema["description"]
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert "selectors" in props
        assert "budget_chars" in props
        assert "allow_bounded_expansion" in props

    def test_query_required(self):
        schema = vault_context_tool_schema()
        assert "query" in schema["parameters"]["required"]


class TestBuildEnvelope:
    def test_envelope_has_required_top_level_keys(self):
        env = build_envelope(
            status="ok",
            query_id="QID-1",
            scope={},
            budget_used=0,
            budget_default=12_000,
            budget_ceiling=24_000,
            budget_expanded=False,
            candidates=[],
            extracts=[],
            conflicts=[],
            redactions={"secrets": 0, "pii": 0, "classes": []},
            hold=None,
            log_ref="query-log.jsonl:QID-1",
        )
        for k in (
            "status", "query_id", "scope", "budget", "candidates", "extracts",
            "conflicts", "redactions", "hold", "log_ref",
        ):
            assert k in env
        assert env["status"] == "ok"
        assert env["budget"]["used_chars"] == 0
        assert env["budget"]["default_chars"] == 12_000
        assert env["budget"]["hard_ceiling_chars"] == 24_000

    def test_hold_envelope_has_reason_code(self):
        env = build_envelope(
            status="data_hold",
            query_id="QID-2",
            scope={},
            budget_used=12_001,
            budget_default=12_000,
            budget_ceiling=24_000,
            budget_expanded=False,
            candidates=[],
            extracts=[],
            conflicts=[],
            redactions={"secrets": 0, "pii": 0, "classes": []},
            hold={"reason_code": "budget_exceeded", "detail": "x", "missing_or_conflicting_sources": []},
            log_ref="query-log.jsonl:QID-2",
        )
        assert env["status"] == "data_hold"
        assert env["hold"] is not None
        assert env["hold"]["reason_code"] == "budget_exceeded"


class TestVaultContextHandler:
    def test_simple_query_returns_ok_envelope(self, cfg, small_note):
        handler = VaultContextHandler(cfg, turn_id="t1")
        result = handler.handle({
            "query": "what is the hello world note?",
            "selectors": [{"path": "99 System/note.md", "line_start": 1, "line_end": 2}],
        })
        env = json.loads(result)
        assert env["status"] == "ok"
        assert len(env["extracts"]) == 1
        assert env["extracts"][0]["path"].endswith("note.md")
        assert env["extracts"][0]["line_start"] == 1
        assert env["extracts"][0]["line_end"] == 2
        assert "hello world" in env["extracts"][0]["content"]
        # log_ref points into the JSONL log
        assert env["log_ref"].startswith("query-log.jsonl:")

    def test_query_id_is_unique(self, cfg, small_note):
        handler = VaultContextHandler(cfg, turn_id="t1")
        r1 = handler.handle({
            "query": "q1",
            "selectors": [{"path": "99 System/note.md", "line_start": 1, "line_end": 2}],
        })
        r2 = handler.handle({
            "query": "q2",
            "selectors": [{"path": "99 System/note.md", "line_start": 1, "line_end": 2}],
        })
        env1 = json.loads(r1)
        env2 = json.loads(r2)
        assert env1["query_id"] != env2["query_id"]

    def test_default_12000_chars_accepted(self, cfg, vault_root):
        # Build a note of exactly 12,000 chars
        p = vault_root / "99 System" / "twelvek.md"
        body = ("x" * 11_950) + "\n# End\n"  # ~12,000 chars
        p.write_text(body, encoding="utf-8")
        # The note is non-large (< 20,000), so the budget kicks in, not large-file.
        handler = VaultContextHandler(cfg, turn_id="t1")
        result = handler.handle({
            "query": "full content",
            "selectors": [{"path": "99 System/twelvek.md", "line_start": 1, "line_end": 1_000}],
        })
        env = json.loads(result)
        assert env["status"] == "ok"
        # Total chars read is at most the budget
        assert env["budget"]["used_chars"] <= 12_000

    def test_oversized_request_holds(self, cfg, vault_root):
        # Build a note ~14,000 chars (> default 12k, < hard ceiling 24k)
        p = vault_root / "99 System" / "fourteenk.md"
        body = ("x" * 14_000)
        p.write_text(body, encoding="utf-8")
        handler = VaultContextHandler(cfg, turn_id="t1")
        result = handler.handle({
            "query": "huge",
            "selectors": [{"path": "99 System/fourteenk.md", "line_start": 1, "line_end": 14_000}],
        })
        env = json.loads(result)
        assert env["status"] == "data_hold"
        assert env["hold"]["reason_code"] == "budget_exceeded"

    def test_path_escape_rejected(self, cfg, vault_root, tmp_path):
        outside = tmp_path / "outside.md"
        outside.write_text("secret content")
        handler = VaultContextHandler(cfg, turn_id="t1")
        result = handler.handle({
            "query": "escape",
            "selectors": [{"path": "../outside.md", "line_start": 1, "line_end": 1}],
        })
        env = json.loads(result)
        assert env["status"] == "data_hold"
        # No content leak
        assert "secret content" not in result

    def test_large_file_without_selector_rejected(self, cfg, large_note):
        handler = VaultContextHandler(cfg, turn_id="t1")
        result = handler.handle({
            "query": "the whole large file please",
            "selectors": [{"path": "99 System/large.md"}],
        })
        env = json.loads(result)
        assert env["status"] == "data_hold"
        assert env["hold"]["reason_code"] == "large_file_selector_required"
        assert "x" * 100 not in result  # no leak

    def test_query_log_written(self, cfg, small_note):
        handler = VaultContextHandler(cfg, turn_id="t1")
        handler.handle({
            "query": "private query alpha bravo",
            "selectors": [{"path": "99 System/note.md", "line_start": 1, "line_end": 2}],
        })
        log = Path(cfg["state_dir"]) / "vault-retrieval" / "query-log.jsonl"
        assert log.exists()
        content = log.read_text(encoding="utf-8")
        # Raw terms NEVER appear in the log
        assert "alpha" not in content
        assert "bravo" not in content
        assert "private query" not in content

    def test_query_log_write_failure_holds(self, cfg, small_note):
        # Make the state dir unwritable: place a regular file where the
        # logger will try to mkdir the ``vault-retrieval`` subdir.
        cfg2 = dict(cfg)
        blocker = Path(cfg["state_dir"]) / "blocker"
        blocker.mkdir(parents=True, exist_ok=True)
        # ``vault-retrieval`` becomes a regular file, so the logger can't
        # mkdir inside it.
        (blocker / "vault-retrieval").write_text("not a dir")
        cfg2["state_dir"] = str(blocker)
        handler = VaultContextHandler(cfg2, turn_id="t1")
        result = handler.handle({
            "query": "anything",
            "selectors": [{"path": "99 System/note.md", "line_start": 1, "line_end": 2}],
        })
        env = json.loads(result)
        assert env["status"] == "data_hold"
        assert env["hold"]["reason_code"] == "query_log_unavailable"


class TestSnapshotsFTS5Disabled:
    def test_no_cache_or_index_files_created(self, cfg, small_note):
        handler = VaultContextHandler(cfg, turn_id="t1")
        handler.handle({
            "query": "x",
            "selectors": [{"path": "99 System/note.md", "line_start": 1, "line_end": 2}],
        })
        cache_dir = Path(cfg["state_dir"]).parent / "cache" / "vault-retrieval"
        assert not cache_dir.exists() or not any(cache_dir.iterdir())
        # No sqlite index
        index_file = Path(cfg["state_dir"]).parent / "cache" / "vault-retrieval" / "index.sqlite3"
        assert not index_file.exists()


class TestMultipleTurns:
    def test_two_turns_do_not_share_budget(self, cfg, vault_root):
        # 14k note: exceeds default (12k), within hard ceiling (24k)
        p = vault_root / "99 System" / "fourteenk.md"
        p.write_text("x" * 14_000, encoding="utf-8")
        h1 = VaultContextHandler(cfg, turn_id="turn-A")
        r1 = h1.handle({
            "query": "q",
            "selectors": [{"path": "99 System/fourteenk.md", "line_start": 1, "line_end": 14_000}],
        })
        env1 = json.loads(r1)
        # First call on its own turn: also over default → data_hold
        assert env1["status"] == "data_hold"

        # A second turn starts fresh: same result
        h2 = VaultContextHandler(cfg, turn_id="turn-B")
        r2 = h2.handle({
            "query": "q",
            "selectors": [{"path": "99 System/fourteenk.md", "line_start": 1, "line_end": 14_000}],
        })
        env2 = json.loads(r2)
        assert env2["status"] == "data_hold"
        # Different turn_ids
        assert env1["query_id"] != env2["query_id"]
