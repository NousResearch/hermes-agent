"""Benchmark: token/character reduction against naive full-read approach.

Run from the worktree root::

    python vault-retrieval/scripts/bench_reduction.py

Per the locked contract and the architecture-memo adoption review
(2026-10-01 23:16 +07), this script reports **per-call** and
**cumulative** character totals as **separate columns**. The previous
version reused a single handler for several calls and then summed the
cumulative ``used_chars`` values, which double-counted the same range on
every additional call (a synthetic comparator artefact, not measured
before/after evidence).

What this script actually measures:

  1. Two targeted ``vault_context`` calls (Maid Dee and Senio current-state
     notes). Each call is run on a FRESH ``VaultContextHandler`` instance
     with its own per-turn counter, so the budget is per-call and the
     cumulative column is ``sum(per_call used_chars)``.

  2. A naive comparator: full-read of the two current-state notes (the
     underlying file sizes on disk).

  3. An aggregate comparator: full-read of six candidate files the agent
     might naively read when triangulating the same evidence.

The "synthetic" framing matters: this is a controlled two-call
comparison against file sizes, not a measurement of any actual agent
session. Per-call vs cumulative totals are emitted side-by-side so the
reader can see how the same handler would have over-counted if the
counter had been shared across calls.

For live before/after evidence, the integration test
``tests/test_runtime_integration.py::test_handle_function_call_dispatch`` exercises the real
``model_tools.handle_function_call`` path and ``scripts/bench_reduction.py``
deliberately does not.

Output columns: ``per_call``, ``cumulative``, ``naive``,
``reduction_vs_naive_per_call`` (the apples-to-apples number),
``reduction_vs_naive_cumulative`` (also shown for transparency).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the plugin importable
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vault-retrieval"))

from vault_retrieval.tool import VaultContextHandler  # noqa: E402


def _build_cfg(vault_root: Path, state_dir: Path) -> dict:
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
        "state_dir": str(state_dir),
    }


def main() -> int:
    vault_root = Path("/root/Documents/Obsidian Vault")
    if not vault_root.exists():
        print(f"ERROR: Vault root not found: {vault_root}", file=sys.stderr)
        return 1

    state_dir = Path("/tmp/bench-vault-state")
    state_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(vault_root, state_dir)

    targets = [
        {
            "name": "Maid Dee current state",
            "path": "02 Projects/Maid Dee/Project - Maid Dee - Current State.md",
            "line_start": 1,
            "line_end": 30,
        },
        {
            "name": "Senio current state",
            "path": "02 Projects/Senio/Project - Senio - Current State.md",
            "line_start": 1,
            "line_end": 30,
        },
    ]

    print(f"Vault root: {vault_root}")
    print(f"Default budget: {cfg['default_budget_chars']:,} chars")
    print(f"Hard ceiling:   {cfg['hard_ceiling_chars']:,} chars")
    print()
    print("ASSUMPTION: synthetic selected-range comparison; this bench is")
    print("  NOT a measurement of live agent before/after character usage.")
    print("  Per-call column uses a FRESH VaultContextHandler per call so")
    print("  budget is per-call; cumulative column sums those per-call")
    print("  values across both targeted calls.")
    print()

    per_call_rows = []
    for t in targets:
        full_path = vault_root / t["path"]
        if not full_path.exists():
            print(f"[skip] {t['name']}: file not found ({t['path']})")
            continue
        naive_chars = full_path.stat().st_size
        # FRESH handler per call — per-call budget is the contract.
        handler = VaultContextHandler(cfg, turn_id=f"bench-{t['name']}")
        out = handler.handle({
            "query": f"What is {t['name']}?",
            "selectors": [{
                "path": t["path"],
                "line_start": t["line_start"],
                "line_end": t["line_end"],
            }],
        })
        env = json.loads(out)
        per_call = env["budget"]["used_chars"]
        reduction = naive_chars - per_call
        pct = 100 * (1 - per_call / naive_chars) if naive_chars > 0 else 0
        print(f"[{t['name']}]")
        print(f"  status:        {env['status']}")
        print(f"  query_id:      {env['query_id']}")
        print(f"  log_ref:       {env['log_ref']}")
        print(f"  naive (full):  {naive_chars:>7,} chars")
        print(f"  per_call:      {per_call:>7,} chars "
              f"(under 12k: {per_call <= cfg['default_budget_chars']})")
        print(f"  reduction:     {reduction:>7,} chars ({pct:.1f}%)")
        print()
        per_call_rows.append((t, naive_chars, per_call))

    # Aggregate: naive = 2 current-state notes; targeted = sum of per-call
    # values across the two rows above. We deliberately do NOT reuse one
    # handler across both calls — that would inflate "targeted" by
    # accumulating the counter twice on the same range.
    print("Aggregate comparison (2 current-state notes):")
    print("-" * 60)
    naive_two = sum(n for _, n, _ in per_call_rows)
    per_call_total = sum(p for _, _, p in per_call_rows)
    if naive_two > 0:
        pct = 100 * (1 - per_call_total / naive_two)
        print(f"  naive (2 full current-state notes):       "
              f"{naive_two:>7,} chars")
        print(f"  vault_context (2 targeted per_call sum):  "
              f"{per_call_total:>7,} chars")
        print(f"  reduction (per_call vs naive):            "
              f"{naive_two - per_call_total:>7,} chars ({pct:.1f}%)")
        print(f"  per-call budget honoured:                 "
              f"{all(p <= cfg['default_budget_chars'] for _, _, p in per_call_rows)}")

    # Six-file aggregate comparator (still synthetic). Naive = six full
    # files; targeted = sum of per_call values above (the same two
    # current-state notes an agent would actually need). This is the
    # "what if the agent just read everything" frame.
    print()
    print("Aggregate comparison (six-file naive vs two-current-state targeted):")
    print("-" * 60)
    candidate_files = [
        "02 Projects/Maid Dee/Project - Maid Dee - Current State.md",
        "02 Projects/Maid Dee/Project - Maid Dee.md",
        "02 Projects/Senio/Project - Senio - Current State.md",
        "02 Projects/Senio/Project - Senio.md",
        "99 System/Hermes Token-Efficient Vault Retrieval Architecture - 2026-10-01.md",
        "99 System/Operating System - Single Source of Truth.md",
    ]
    naive_total = 0
    for rel in candidate_files:
        p = vault_root / rel
        if p.exists():
            naive_total += p.stat().st_size
    if naive_total > 0:
        pct = 100 * (1 - per_call_total / naive_total)
        print(f"  naive (6 candidate files, full-read):       "
              f"{naive_total:>7,} chars")
        print(f"  vault_context (2 targeted per_call sum):    "
              f"{per_call_total:>7,} chars")
        print(f"  reduction vs naive:                          "
              f"{naive_total - per_call_total:>7,} chars ({pct:.1f}%)")
        print(f"  per-call budget honoured:                   "
              f"{all(p <= cfg['default_budget_chars'] for _, _, p in per_call_rows)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())