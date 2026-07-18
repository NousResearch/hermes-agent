"""Tests for the memory classification gate + mutation ledger.

The classification gate is the semantic counterpart to the threat-pattern
security scan: it rejects/warns on content that pollutes durable memory
(task state, imperatives, raw clinical/genomic payloads, secrets), and the
ledger records every mutation with provenance.
"""

import json

import pytest

from tools import memory_classification as mc
from tools import memory_ledger as ml
from tools.memory_tool import MemoryStore, memory_tool


# =========================================================================
# Tier 1 — hard rejects
# =========================================================================

class TestHardRejects:
    @pytest.mark.parametrize("content", [
        "Fixed auth bug in commit d59b79fad12",                     # 12-char SHA
        "See PR #4821 for the fix",                                  # issue ref
        "JIRA ticket PROJ-1234 tracks the migration",                # ticket ID
        "Deployed the new pipeline on 2026-07-18",                   # ISO date
        "The meeting is on 7/18/2026",                               # calendar date
        "Patient has rs429358 variant",                              # rsID
        "Variant at chr19:44908684",                                 # genomic coordinate
        "MRN: 00452318 admitted last week",                          # clinical PII
        "The key is sk-a1b2c3d4e5f6g7h8",                            # credential shape
        "Config lives at C:\\Users\\amiwe\\AppData\\Local\\hermes",  # file path
    ])
    def test_hard_reject_no_override(self, content):
        verdict, messages = mc.evaluate(content, override=True,
                                        rationale="I really want this")
        assert verdict == "reject"
        assert messages and "memory policy reject" in messages[0]

    @pytest.mark.parametrize("content", [
        "User prefers concise responses that lead with the conclusion",
        "Project uses pytest with xdist for parallel test runs",
        "Hermes terminal on Windows is git-bash (MSYS), not PowerShell",
        "The genomics pipeline targets 30x whole-genome sequencing",
        "User's first-priority project is the WGS pipeline",
    ])
    def test_clean_content_passes(self, content):
        verdict, messages = mc.evaluate(content)
        assert verdict == "pass"
        assert messages == []


# =========================================================================
# Tier 2 — warn / override
# =========================================================================

class TestTierTwo:
    @pytest.mark.parametrize("content", [
        "Always run the test suite before committing",       # imperative start
        "Never store credentials in memory",                  # imperative start
        "Fixed the login bug in the gateway",                 # completed work
        "Currently the gateway runs on port 8642",            # transient status
        "Phase 2 of the migration is in progress",            # progress state
    ])
    def test_warn_without_override(self, content):
        verdict, messages = mc.evaluate(content)
        assert verdict == "warn"
        assert messages

    def test_override_without_rationale_still_warns(self):
        verdict, _ = mc.evaluate("Always run tests first", override=True,
                                 rationale=None)
        assert verdict == "warn"

    def test_override_with_rationale_passes(self):
        verdict, messages = mc.evaluate(
            "Always run tests first", override=True,
            rationale="Stable team convention the user asked me to enforce")
        assert verdict == "override"
        assert messages  # warnings are still reported for the ledger


# =========================================================================
# Store integration — gate + ledger
# =========================================================================

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    s = MemoryStore()
    s.load_from_disk()
    return s


def _ledger_lines(tmp_path):
    p = tmp_path / "memories" / "LEDGER.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class TestStoreGate:
    def test_hard_reject_blocks_add(self, store):
        result = store.add("memory", "Fixed bug in commit d59b79fad12")
        assert result["success"] is False
        assert result.get("rejected_by") == "classification_gate"
        assert store.memory_entries == []

    def test_warn_requires_override(self, store):
        result = store.add("memory", "Always verify with real commands")
        assert result["success"] is False
        assert result.get("needs_override") is True
        assert store.memory_entries == []

        ok = store.add("memory", "Always verify with real commands",
                       override=True,
                       rationale="Core operating convention the user set")
        assert ok["success"] is True
        assert store.memory_entries == ["Always verify with real commands"]

    def test_gate_disabled_allows_write(self, store):
        store.classification_gate = False
        result = store.add("memory", "Fixed bug in commit d59b79fad12")
        assert result["success"] is True

    def test_batch_rejected_by_gate(self, store):
        result = store.apply_batch("memory", [
            {"action": "add", "content": "User prefers dark mode"},
            {"action": "add", "content": "Deployed on 2026-07-18"},
        ])
        assert result["success"] is False
        assert "Operation 2" in result["error"]
        assert store.memory_entries == []


class TestLedger:
    def test_add_writes_ledger_line(self, store, tmp_path):
        store.add("memory", "User prefers dark mode")
        lines = _ledger_lines(tmp_path)
        assert len(lines) == 1
        assert lines[0]["action"] == "add"
        assert lines[0]["target"] == "memory"
        assert lines[0]["new_sha256"]
        assert "ts" in lines[0]

    def test_override_recorded_with_rationale(self, store, tmp_path):
        store.add("memory", "Always verify with real commands", override=True,
                  rationale="User-set convention")
        lines = _ledger_lines(tmp_path)
        assert lines[0]["override"] is True
        assert lines[0]["rationale"] == "User-set convention"
        assert lines[0]["warnings"]

    def test_remove_and_replace_ledgered(self, store, tmp_path):
        store.add("memory", "User prefers dark mode")
        store.replace("memory", "dark mode", "User prefers light mode")
        store.remove("memory", "light mode")
        actions = [l["action"] for l in _ledger_lines(tmp_path)]
        assert actions == ["add", "replace", "remove"]

    def test_rejected_writes_not_ledgered(self, store, tmp_path):
        store.add("memory", "Fixed bug in commit d59b79fad12")
        assert _ledger_lines(tmp_path) == []


class TestMemoryToolDispatch:
    def test_override_flow_through_tool(self, store):
        raw = memory_tool(action="add", target="memory",
                          content="Never send emails without permission",
                          store=store)
        assert json.loads(raw)["success"] is False

        raw = memory_tool(action="add", target="memory",
                          content="Never send emails without permission",
                          store=store, override=True,
                          rationale="Standing safety rule from the user")
        assert json.loads(raw)["success"] is True

    def test_ledger_failure_does_not_break_write(self, store, monkeypatch):
        import tools.memory_tool as mt

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(mt._ledger, "record", boom)
        result = store.add("memory", "User prefers dark mode")
        assert result["success"] is True
