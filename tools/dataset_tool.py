#!/usr/bin/env python3
"""Dataset synthesis, curation, and validation tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.config import get_research_config
from research.manager import assess_dataset_rows
from research.models import DatasetRecord, now_utc
from research.state_store import ResearchStateStore
from tools.registry import registry


def check_dataset_requirements() -> bool:
    """Dataset tool is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def _load_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _get_auxiliary_client(task: str = "dataset_synthesis"):
    """Get an auxiliary LLM client for dataset operations."""
    from agent.auxiliary_client import get_text_auxiliary_client

    client, _model = get_text_auxiliary_client(task=task)
    return client


def _run_python_json(code: str, args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "subprocess failed")
    return json.loads(result.stdout)


def dataset_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform dataset synthesis, curation, and validation."""
    store = ResearchStateStore(cwd=cwd)
    action = (action or "").strip()
    cfg = get_research_config()
    ds_cfg = cfg.get("dataset", {})

    if action == "synthesize_dataset":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        source_path = kwargs.get("source_path")
        output_path = kwargs.get("output_path")
        if not source_path:
            raise ValueError("source_path is required")
        if not output_path:
            raise ValueError("output_path is required")

        num_rows = min(
            int(kwargs.get("num_rows", 100)),
            int(ds_cfg.get("max_synthesis_rows", 5000)),
        )
        method = kwargs.get("method", "sft")

        source_rows = _load_jsonl_rows(source_path)
        sample_text = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in source_rows[:5]
        )

        if method == "sft":
            synthesis_prompt = (
                f"Given these example rows from a dataset:\n{sample_text}\n\n"
                f"Generate {num_rows} new instruction-completion pairs in JSONL format. "
                "Each line must be a valid JSON object with 'prompt' and 'completion' fields. "
                "Output ONLY the JSONL lines, no other text."
            )
        elif method == "dpo":
            synthesis_prompt = (
                f"Given these example rows from a dataset:\n{sample_text}\n\n"
                f"Generate {num_rows} preference pairs in JSONL format. "
                "Each line must be a valid JSON object with 'prompt', 'chosen', and 'rejected' fields. "
                "Output ONLY the JSONL lines, no other text."
            )
        else:
            synthesis_prompt = (
                f"Given these example rows from a dataset:\n{sample_text}\n\n"
                f"Generate {num_rows} training examples in JSONL format. "
                "Each line must be a valid JSON object with 'prompt' and 'completion' fields. "
                "Output ONLY the JSONL lines, no other text."
            )

        try:
            client = _get_auxiliary_client(ds_cfg.get("auxiliary_model_task", "dataset_synthesis"))
            response = client.chat.completions.create(
                model=os.getenv("AUXILIARY_DATASET_MODEL", "google/gemini-2.0-flash-001"),
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            return json.dumps({"success": False, "error": f"LLM synthesis failed: {e}"}, ensure_ascii=False, indent=2)

        generated_rows: list[dict[str, Any]] = []
        for line in raw_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                generated_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        _write_jsonl(output_path, generated_rows)

        # Detect columns
        columns = sorted(set().union(*(row.keys() for row in generated_rows))) if generated_rows else []
        quality_report = assess_dataset_rows(generated_rows, method=method)

        record = DatasetRecord(
            project_id=project_id,
            name=kwargs.get("name", Path(output_path).stem),
            path=output_path,
            format="jsonl",
            num_rows=len(generated_rows),
            columns=columns,
            source_action="synthesize_dataset",
            parent_dataset_id=kwargs.get("dataset_id"),
            source_paths=[str(source_path)],
            quality_report=quality_report,
        )
        store.save_dataset_record(project_id, record)

        return json.dumps(
            {"success": True, "record": record.model_dump(mode="json"), "rows_generated": len(generated_rows)},
            ensure_ascii=False,
            indent=2,
        )

    if action == "curate_dataset":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        source_path = kwargs.get("source_path")
        output_path = kwargs.get("output_path")
        if not source_path or not output_path:
            raise ValueError("source_path and output_path are required")

        rows = _load_jsonl_rows(source_path)
        original_count = len(rows)

        # Deduplicate by exact match on serialized row
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for row in rows:
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                deduped.append(row)

        # Filter by length heuristics
        filtered: list[dict[str, Any]] = []
        for row in deduped:
            text_values = [str(v) for v in row.values() if isinstance(v, str)]
            total_len = sum(len(v) for v in text_values)
            if total_len < 5:
                continue
            if total_len > 50000:
                continue
            # Skip rows with mostly empty fields
            empty_count = sum(1 for v in row.values() if not v and v != 0)
            if empty_count > len(row) / 2:
                continue
            filtered.append(row)

        _write_jsonl(output_path, filtered)

        columns = sorted(set().union(*(row.keys() for row in filtered))) if filtered else []
        record = DatasetRecord(
            project_id=project_id,
            name=kwargs.get("name", Path(output_path).stem),
            path=output_path,
            format="jsonl",
            num_rows=len(filtered),
            columns=columns,
            source_action="curate_dataset",
            parent_dataset_id=kwargs.get("dataset_id"),
            source_paths=[str(source_path)],
            quality_report={
                "original_rows": original_count,
                "after_dedup": len(deduped),
                "after_filter": len(filtered),
                "duplicates_removed": original_count - len(deduped),
                "filtered_out": len(deduped) - len(filtered),
                **assess_dataset_rows(filtered, method="dpo" if {"chosen", "rejected"}.issubset(columns) else "sft"),
            },
        )
        store.save_dataset_record(project_id, record)

        return json.dumps(
            {"success": True, "record": record.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
        )

    if action == "generate_preferences":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        source_path = kwargs.get("source_path")
        output_path = kwargs.get("output_path")
        checkpoint_path = kwargs.get("checkpoint_path")
        num_completions = int(kwargs.get("num_completions", 4))

        if not source_path or not output_path:
            raise ValueError("source_path and output_path are required")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for generate_preferences")

        prompts_data = _load_jsonl_rows(source_path)
        prompts = [row.get("prompt", "") for row in prompts_data if row.get("prompt")]

        preference_rows: list[dict[str, Any]] = []
        for prompt in prompts:
            # Generate completions via tinker sampling
            sample_code = """
import json, sys
import tinker

model_path, prompt, n, temp = sys.argv[1:5]
service = tinker.ServiceClient()
sampler = service.create_sampling_client(model_path=model_path)
params = tinker.SamplingParams(max_tokens=512, temperature=float(temp))
completions = []
for _ in range(int(n)):
    raw = sampler.sample(prompt=prompt, sampling_params=params).result()
    text = getattr(raw, "text", None)
    if text is None and hasattr(raw, "sequences") and raw.sequences:
        text = getattr(raw.sequences[0], "text", None)
    if text is None:
        text = str(raw)
    completions.append(text)
print(json.dumps({"completions": completions}))
"""
            try:
                result = _run_python_json(
                    sample_code,
                    [str(checkpoint_path), prompt, str(num_completions), str(kwargs.get("temperature", 0.7))],
                )
                completions = result.get("completions", [])
            except Exception:
                continue

            if len(completions) < 2:
                continue

            # Use auxiliary LLM to rank completions
            try:
                judge_cfg = cfg.get("judge", {})
                client = _get_auxiliary_client(judge_cfg.get("auxiliary_model_task", "judge"))
                ranking_prompt = (
                    f"Rank the following completions for the prompt: {prompt}\n\n"
                    + "\n".join(f"[{i}] {c}" for i, c in enumerate(completions))
                    + "\n\nReturn a JSON object with 'ranking' as a list of indices from best to worst."
                )
                response = client.chat.completions.create(
                    model=os.getenv("AUXILIARY_JUDGE_MODEL", "google/gemini-2.0-flash-001"),
                    messages=[{"role": "user", "content": ranking_prompt}],
                    temperature=0.1,
                    max_tokens=256,
                )
                rank_text = response.choices[0].message.content or ""
                rank_data = json.loads(rank_text)
                ranking = rank_data.get("ranking", list(range(len(completions))))
            except Exception:
                ranking = list(range(len(completions)))

            if len(ranking) >= 2:
                chosen = completions[ranking[0]] if ranking[0] < len(completions) else completions[0]
                rejected = completions[ranking[-1]] if ranking[-1] < len(completions) else completions[-1]
                preference_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        _write_jsonl(output_path, preference_rows)

        columns = ["prompt", "chosen", "rejected"]
        record = DatasetRecord(
            project_id=project_id,
            name=kwargs.get("name", Path(output_path).stem),
            path=output_path,
            format="jsonl",
            num_rows=len(preference_rows),
            columns=columns,
            source_action="generate_preferences",
            source_paths=[str(source_path), str(checkpoint_path)],
            quality_report=assess_dataset_rows(preference_rows, method="dpo"),
        )
        store.save_dataset_record(project_id, record)

        return json.dumps(
            {"success": True, "record": record.model_dump(mode="json"), "pairs_generated": len(preference_rows)},
            ensure_ascii=False,
            indent=2,
        )

    if action == "validate_dataset":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        source_path = kwargs.get("source_path")
        if not source_path:
            dataset_id = kwargs.get("dataset_id")
            if dataset_id:
                rec = store.load_dataset_record(project_id, dataset_id)
                source_path = rec.get("path")
        if not source_path:
            raise ValueError("source_path or dataset_id required")

        method = kwargs.get("method", "sft")
        rows = _load_jsonl_rows(source_path)

        required_fields: dict[str, list[str]] = {
            "sft": ["prompt", "completion"],
            "dpo": ["prompt", "chosen", "rejected"],
            "ppo": ["prompt", "completion", "logprobs"],
        }
        required = required_fields.get(method, ["prompt"])
        field_map = kwargs.get("field_map") or {}

        issues: list[str] = []
        empty_fields = 0
        for idx, row in enumerate(rows):
            for field in required:
                mapped_field = field_map.get(field, field)
                if mapped_field not in row:
                    issues.append(f"Row {idx}: missing field '{mapped_field}'")
                elif not row[mapped_field] and row[mapped_field] != 0:
                    empty_fields += 1

        all_columns = sorted(set().union(*(row.keys() for row in rows))) if rows else []

        quality_report = {
            "path": source_path,
            "method": method,
            "num_rows": len(rows),
            "columns": all_columns,
            "required_fields": required,
            "missing_field_issues": len(issues),
            "empty_fields": empty_fields,
            "issues": issues[:20],
            "valid": len(issues) == 0,
        }

        return json.dumps({"success": True, "quality_report": quality_report}, ensure_ascii=False, indent=2)

    raise ValueError(f"Unsupported dataset action: {action}")


DATASET_SCHEMA = {
    "name": "dataset",
    "description": (
        "Synthesize, curate, and validate datasets for post-training. "
        "Supports SFT instruction-completion generation, DPO preference pair creation, "
        "deduplication, quality filtering, and format validation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["synthesize_dataset", "curate_dataset", "generate_preferences", "validate_dataset"],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "source_path": {"type": "string"},
            "output_path": {"type": "string"},
            "method": {"type": "string", "enum": ["sft", "dpo", "ppo"]},
            "num_rows": {"type": "integer"},
            "checkpoint_path": {"type": "string"},
            "num_completions": {"type": "integer"},
            "field_map": {"type": "object"},
            "dataset_id": {"type": "string"},
            "name": {"type": "string"},
            "temperature": {"type": "number"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="dataset",
    toolset="research",
    schema=DATASET_SCHEMA,
    handler=lambda args, **kw: dataset_tool(**args),
    check_fn=check_dataset_requirements,
)
