"""Benchmark script — shows token/character reduction vs. naive full-read.

Run from the worktree root:
    python vault-retrieval/scripts/bench_reduction.py

The output reports per-call evidence chars consumed by ``vault_context``
versus the size of the underlying note, and verifies the per-turn
budget is respected.
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


def main() -> int:
    vault_root = Path("/root/Documents/Obsidian Vault")
    if not vault_root.exists():
        print(f"ERROR: Vault root not found: {vault_root}", file=sys.stderr)
        return 1

    state_dir = "/tmp/bench-vault-state"
    os.makedirs(state_dir, exist_ok=True)
    cfg = {
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
        "state_dir": state_dir,
    }

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

    handler = VaultContextHandler(cfg, turn_id="bench-script")
    print(f"Vault root: {vault_root}")
    print(f"Default budget: {cfg['default_budget_chars']:,} chars")
    print(f"Hard ceiling:   {cfg['hard_ceiling_chars']:,} chars")
    print()

    for t in targets:
        full_path = vault_root / t["path"]
        if not full_path.exists():
            print(f"[skip] {t['name']}: file not found ({t['path']})")
            continue
        naive_chars = full_path.stat().st_size
        out = handler.handle({
            "query": f"What is {t['name']}?",
            "selectors": [{
                "path": t["path"],
                "line_start": t["line_start"],
                "line_end": t["line_end"],
            }],
        })
        env = json.loads(out)
        targeted = env["budget"]["used_chars"]
        reduction = naive_chars - targeted
        pct = 100 * (1 - targeted / naive_chars) if naive_chars > 0 else 0
        print(f"[{t['name']}]")
        print(f"  status:        {env['status']}")
        print(f"  query_id:      {env['query_id']}")
        print(f"  log_ref:       {env['log_ref']}")
        print(f"  naive (full):  {naive_chars:>7,} chars")
        print(f"  targeted:      {targeted:>7,} chars "
              f"(under budget: {targeted <= cfg['default_budget_chars']})")
        print(f"  reduction:     {reduction:>7,} chars ({pct:.1f}%)")
        print()

    # Aggregate comparison: naive agent reads N files, vault_context reads M ranges
    print("Aggregate comparison (naive vs vault_context):")
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
    # Targeted: only the current-state notes (which is what an agent
    # actually needs for the standard query).
    targeted_total = 0
    for t in targets:
        out = handler.handle({
            "query": "agg",
            "selectors": [{
                "path": t["path"],
                "line_start": t["line_start"],
                "line_end": t["line_end"],
            }],
        })
        env = json.loads(out)
        targeted_total += env["budget"]["used_chars"]
    if naive_total > 0:
        pct = 100 * (1 - targeted_total / naive_total)
        print(f"  Naive (6 candidate files, full-read):     "
              f"{naive_total:>7,} chars")
        print(f"  vault_context (2 targeted current-state): "
              f"{targeted_total:>7,} chars")
        print(f"  reduction:                                "
              f"{naive_total - targeted_total:>7,} chars "
              f"({pct:.1f}%)")
        # Per-turn budget check
        print(f"  per-turn budget honoured:                 "
              f"{targeted_total <= cfg['default_budget_chars']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
