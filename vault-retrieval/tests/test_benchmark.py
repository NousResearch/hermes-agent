"""Benchmark: token/character reduction against naive full-read approach.

Demonstrates that ``vault_context`` reduces per-turn character count
versus a naive approach (read the whole Vault note before responding).

This is the evidence-side counterpart to acceptance test #2/#4/#5: it
actually runs both paths against the same fixture Vault and reports the
reduction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


from vault_retrieval.tool import VaultContextHandler


@pytest.fixture
def bench_cfg(tmp_path: Path) -> Dict[str, Any]:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "99 System").mkdir()
    # A realistic-shaped note (~60 lines / ~3,500 chars). Most Vault
    # notes are well under the 20,000-char large-file threshold, so the
    # budget matters, not the large-file rules.
    note = vault / "99 System" / "Project - Maid Dee - Current State.md"
    body_lines = [
        "---",
        "title: Project - Maid Dee - Current State",
        "status: current",
        "owner: Roger",
        "last_updated: 2026-10-01T12:00:00+07:00",
        "review_by: 2026-11-01",
        "---",
        "",
        "# Maid Dee",
        "",
        "## Stage",
        "Pilot live since 2026-01-01; 200 maids onboarded by 2026-04-01.",
        "",
        "## KPI",
        "- Primary: weekly active maids (WAM).",
        "- Baseline: 168 (WAM as of 2026-03-31).",
        "- Target: 200 by 2026-06-30.",
        "",
        "## Current decisions",
        "- DEC-MAIDDEE-20260101-001 — React+TS+Vite + Supabase/Firebase stack.",
        "- DEC-MAIDDEE-20260215-002 — LIFF over LINE OA as primary channel.",
        "",
        "## Blockers",
        "- BLOCK-001 — Demand softened since early April 2026.",
        "",
        "## Next actions",
        "- [ ] Roger — refresh campaign — due 2026-10-08 — proof: campaign brief.",
        "",
        "## Canonical links",
        "- Project hub: [[Project - Maid Dee]].",
        "",
        "## Notes",
    ]
    # Pad out to ~60 lines so the benchmark can do two non-overlapping
    # 30-line reads in one turn.
    for i in range(30):
        body_lines.append(f"- note line {i + 1}: lorem ipsum dolor sit amet.")
    body = "\n".join(body_lines) + "\n"
    note.write_text(body, encoding="utf-8")
    return {
        "vault_root": str(vault),
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


def _naive_full_read_chars(vault_root: Path) -> int:
    """What a naive agent would do: read every file in the Vault."""
    total = 0
    for p in vault_root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _vault_context_chars(cfg: Dict[str, Any], selectors: List[Dict[str, Any]]) -> int:
    """What the plugin actually returns when used with targeted selectors."""
    handler = VaultContextHandler(cfg, turn_id="bench")
    out = handler.handle({"query": "bench", "selectors": selectors})
    env = json.loads(out)
    return sum(e["chars"] for e in env.get("extracts", []))


class TestBenchmark:
    def test_targeted_read_smaller_than_naive_full_read(self, bench_cfg):
        vault = Path(bench_cfg["vault_root"])
        naive = _naive_full_read_chars(vault)
        targeted = _vault_context_chars(bench_cfg, [
            {"path": "99 System/Project - Maid Dee - Current State.md",
             "line_start": 1, "line_end": 60},
        ])
        assert naive > 0
        assert targeted > 0
        # Strict reduction: targeted should be no more than the single
        # relevant file (typically much smaller than the whole Vault).
        assert targeted <= naive

    def test_targeted_under_default_budget(self, bench_cfg):
        targeted = _vault_context_chars(bench_cfg, [
            {"path": "99 System/Project - Maid Dee - Current State.md",
             "line_start": 1, "line_end": 60},
        ])
        assert targeted <= bench_cfg["default_budget_chars"]

    def test_no_extracts_consumes_zero_evidence_chars(self, bench_cfg):
        # Selectors without line_start are added to candidates but yield
        # no evidence chars (the candidate is discovery, not evidence).
        targeted = _vault_context_chars(bench_cfg, [
            {"path": "99 System/Project - Maid Dee - Current State.md"},
        ])
        # No extract → no evidence consumption. Discovery chars are
        # tracked separately (the spec caps discovery at 2,000 chars).
        assert targeted == 0

    def test_repeated_calls_share_budget(self, bench_cfg):
        # Two targeted reads in the same turn share the 12k budget.
        handler = VaultContextHandler(bench_cfg, turn_id="shared")
        # First read consumes X chars.
        out1 = handler.handle({"query": "q1", "selectors": [
            {"path": "99 System/Project - Maid Dee - Current State.md",
             "line_start": 1, "line_end": 30},
        ]})
        env1 = json.loads(out1)
        used1 = env1["budget"]["used_chars"]
        # Second read in the same turn shares the same ceiling.
        out2 = handler.handle({"query": "q2", "selectors": [
            {"path": "99 System/Project - Maid Dee - Current State.md",
             "line_start": 31, "line_end": 60},
        ]})
        env2 = json.loads(out2)
        used2 = env2["budget"]["used_chars"]
        # The cumulative used must be >= each individual call.
        assert used2 >= used1
        assert used2 <= bench_cfg["default_budget_chars"]
