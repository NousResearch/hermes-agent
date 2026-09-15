"""Synthetic benchmark comparing Old Mode (Turn-by-Turn LLM) vs New Mode (Durable Task Runner).

Measures:
- Tool calls exposed to LLM
- Messages/context events
- Approximate output payload size sent to model context
- Duration
- Retries
- Raw data sent to reasoning context
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workstation.artifacts import ArtifactStore
from workstation.batch_runner import DurableBatchRunner
from workstation.durable_tasks import DurableTaskStore
from workstation.tool_verbosity import VerbosityLevel, format_tool_output


def run_benchmark(item_count: int = 86, suspect_count: int = 2) -> Dict[str, Any]:
    # Mock data generation
    raw_item_text = (
        "Insights da publicação: Visualizações: 115839, Alcance: 98200, "
        "Interações com a publicação: 3450, Curtidas: 3200, Comentários: 120, "
        "Compartilhamentos: 85, Salvamentos: 45. Atividade do perfil: 210 visitas."
    )

    items = [
        {
            "id": i,
            "media_id": f"media_{3982372324279313500 + i}",
            "text": raw_item_text if i not in (23, 71) else "Curtidas: 0, 0:01",  # 2 anomalies
            "views": 115839 if i not in (23, 71) else 10,
            "likes": 3200 if i not in (23, 71) else 50,  # 23 and 71 have views < likes
        }
        for i in range(1, item_count + 1)
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SIMULATE OLD MODE: Turn-by-Turn LLM
    # Each item requires:
    # 1. open_preview (tool call + turn)
    # 2. sleep/wait (terminal or wait tool)
    # 3. read_preview (returns full 700 chars into context!)
    # 4. write_file (saves item to disk)
    # Total tool calls: ~3-4 per item. All raw text enters conversational context!
    # ──────────────────────────────────────────────────────────────────────────
    old_start = time.monotonic()
    old_tool_calls = 0
    old_context_bytes = 0
    old_messages = 0

    for item in items:
        # Step 1: open_preview
        old_tool_calls += 1
        old_messages += 2  # assistant tool_call + tool result
        old_context_bytes += len(f'{{"success": true, "url": "https://instagram.com/p/{item["media_id"]}"}}')

        # Step 2: read_preview (returns full raw text!)
        old_tool_calls += 1
        old_messages += 2
        old_context_bytes += len(item["text"])

        # Step 3: write_file
        old_tool_calls += 1
        old_messages += 2
        old_context_bytes += len(f'{{"bytes_written": {len(item["text"])}, "path": "raw/{item["media_id"]}.json"}}')

    old_duration = time.monotonic() - old_start

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SIMULATE NEW MODE: Durable Task Runner + Batching + Artifacts + Exception Escalation
    # 1 high-level batch tool call
    # Execution happens deterministically in software outside LLM
    # Raw items are stored in ArtifactStore (Data Plane)
    # Only suspect anomalies (2 items) return to Reasoning Plane!
    # ──────────────────────────────────────────────────────────────────────────
    new_start = time.monotonic()
    artifact_store = ArtifactStore()
    task_store = DurableTaskStore()
    runner = DurableBatchRunner("bench_task_86", task_store=task_store, artifact_store=artifact_store)

    def worker(inp, work_item):
        return {"views": inp["views"], "likes": inp["likes"], "text": inp["text"]}

    def validator(out, inp):
        if out["views"] < out["likes"]:
            return {"valid": False, "suspect": True, "reason": "views < likes (corrupted or unhydrated)"}
        return {"valid": True, "suspect": False}

    summary = runner.execute_batch("Benchmark 86 Posts", items, worker_fn=worker, validator_fn=validator)

    new_tool_calls = 1  # 1 batch execution call
    new_messages = 2  # 1 tool call + 1 summary result

    # Compact summary returned to LLM
    model_response_str = json.dumps(summary.to_dict(), ensure_ascii=False)
    new_context_bytes = len(model_response_str)
    new_duration = time.monotonic() - new_start

    reduction_tool_calls = (old_tool_calls - new_tool_calls) / old_tool_calls * 100
    reduction_context = (old_context_bytes - new_context_bytes) / old_context_bytes * 100

    return {
        "item_count": item_count,
        "old_mode": {
            "tool_calls": old_tool_calls,
            "messages": old_messages,
            "context_bytes": old_context_bytes,
            "duration_seconds": round(old_duration, 4),
            "raw_data_in_context": True,
        },
        "new_mode": {
            "tool_calls": new_tool_calls,
            "messages": new_messages,
            "context_bytes": new_context_bytes,
            "duration_seconds": round(new_duration, 4),
            "raw_data_in_context": False,
            "anomalies_escalated": len(summary.anomalies),
            "success_in_data_plane": summary.success_count,
        },
        "improvements": {
            "tool_call_reduction_percent": round(reduction_tool_calls, 1),
            "context_traffic_reduction_percent": round(reduction_context, 1),
        },
    }


if __name__ == "__main__":
    res = run_benchmark(86, 2)
    print(json.dumps(res, indent=2, ensure_ascii=False))
