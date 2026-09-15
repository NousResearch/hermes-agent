from __future__ import annotations

import json
from pathlib import Path

from workstation.workload_benchmark import run_workload_benchmark


def test_provider_free_workload_matches_versioned_baseline(tmp_path):
    actual = run_workload_benchmark(tmp_path)
    baseline = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "workload_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert actual == baseline
