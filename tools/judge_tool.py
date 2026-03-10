#!/usr/bin/env python3
"""Reward model / judge integration tool."""

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


def check_judge_requirements() -> bool:
    """Judge tool is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


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


def _get_auxiliary_client(task: str = "judge"):
    """Get an auxiliary LLM client for judge operations."""
    from agent.auxiliary_client import get_text_auxiliary_client

    client, _model = get_text_auxiliary_client(task=task)
    return client


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform judge / reward model operations."""
    store = ResearchStateStore(cwd=cwd)
    action = (action or "").strip()
    cfg = get_research_config()
    judge_cfg = cfg.get("judge", {})

    if action == "generate_completions":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        prompts = kwargs.get("prompts") or []
        checkpoint_path = kwargs.get("checkpoint_path")
        if not prompts:
            raise ValueError("prompts list is required")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        num_completions = min(
            int(kwargs.get("num_completions", 4)),
            int(judge_cfg.get("max_completions_per_prompt", 5)),
        )
        temperature = float(kwargs.get("temperature", judge_cfg.get("default_temperature", 0.3)))
        max_tokens = int(kwargs.get("max_tokens", 512))

        all_results: list[dict[str, Any]] = []
        for prompt in prompts:
            sample_code = """
import json, sys
import tinker

model_path, prompt, n, temp, max_tok = sys.argv[1:6]
service = tinker.ServiceClient()
sampler = service.create_sampling_client(model_path=model_path)
params = tinker.SamplingParams(max_tokens=int(max_tok), temperature=float(temp))
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
                    [str(checkpoint_path), prompt, str(num_completions), str(temperature), str(max_tokens)],
                )
                all_results.append({
                    "prompt": prompt,
                    "completions": result.get("completions", []),
                })
            except Exception as e:
                all_results.append({
                    "prompt": prompt,
                    "completions": [],
                    "error": str(e),
                })

        return json.dumps({"success": True, "results": all_results}, ensure_ascii=False, indent=2)

    if action == "rank_completions":
        prompt = kwargs.get("prompt", "")
        completions = kwargs.get("completions") or []
        if not prompt or not completions:
            raise ValueError("prompt and completions are required")

        temperature = float(kwargs.get("temperature", judge_cfg.get("default_temperature", 0.3)))

        ranking_prompt = (
            f"You are a judge evaluating the quality of completions for a given prompt.\n\n"
            f"Prompt: {prompt}\n\n"
            "Completions:\n"
            + "\n".join(f"[{i}] {c}" for i, c in enumerate(completions))
            + "\n\nRank these completions from best to worst. Return a JSON object with:\n"
            '- "ranking": list of indices from best to worst\n'
            '- "scores": dict mapping index to score (0-10)\n'
            '- "reasoning": brief explanation for the ranking\n'
            "Return ONLY valid JSON."
        )

        try:
            client = _get_auxiliary_client(judge_cfg.get("auxiliary_model_task", "judge"))
            response = client.chat.completions.create(
                model=os.getenv("AUXILIARY_JUDGE_MODEL", "google/gemini-2.0-flash-001"),
                messages=[{"role": "user", "content": ranking_prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            raw_text = response.choices[0].message.content or ""
            # Try to extract JSON from the response
            rank_data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: try to find JSON in the response
            import re
            match = re.search(r"\{[^}]+\}", raw_text, re.DOTALL)
            if match:
                rank_data = json.loads(match.group())
            else:
                rank_data = {"ranking": list(range(len(completions))), "scores": {}, "reasoning": "Failed to parse judge response"}
        except Exception as e:
            rank_data = {"ranking": list(range(len(completions))), "scores": {}, "reasoning": f"Judge error: {e}"}

        return json.dumps({"success": True, "ranking": rank_data}, ensure_ascii=False, indent=2)

    if action == "build_preference_dataset":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        prompts = kwargs.get("prompts") or []
        checkpoint_path = kwargs.get("checkpoint_path")
        output_path = kwargs.get("output_path")

        if not prompts:
            raise ValueError("prompts list is required")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if not output_path:
            raise ValueError("output_path is required")

        num_completions = min(
            int(kwargs.get("num_completions", 4)),
            int(judge_cfg.get("max_completions_per_prompt", 5)),
        )
        temperature = float(kwargs.get("temperature", judge_cfg.get("default_temperature", 0.3)))
        max_tokens = int(kwargs.get("max_tokens", 512))

        preference_rows: list[dict[str, Any]] = []
        for prompt in prompts:
            # Generate completions
            sample_code = """
import json, sys
import tinker

model_path, prompt, n, temp, max_tok = sys.argv[1:6]
service = tinker.ServiceClient()
sampler = service.create_sampling_client(model_path=model_path)
params = tinker.SamplingParams(max_tokens=int(max_tok), temperature=float(temp))
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
                    [str(checkpoint_path), prompt, str(num_completions), str(temperature), str(max_tokens)],
                )
                completions = result.get("completions", [])
            except Exception:
                continue

            if len(completions) < 2:
                continue

            # Rank via judge
            try:
                client = _get_auxiliary_client(judge_cfg.get("auxiliary_model_task", "judge"))
                ranking_prompt = (
                    f"Rank these completions for the prompt: {prompt}\n\n"
                    + "\n".join(f"[{i}] {c}" for i, c in enumerate(completions))
                    + '\n\nReturn JSON: {"ranking": [indices best to worst]}'
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
                chosen_idx = ranking[0] if ranking[0] < len(completions) else 0
                rejected_idx = ranking[-1] if ranking[-1] < len(completions) else len(completions) - 1
                preference_rows.append({
                    "prompt": prompt,
                    "chosen": completions[chosen_idx],
                    "rejected": completions[rejected_idx],
                })

        _write_jsonl(output_path, preference_rows)

        record = DatasetRecord(
            project_id=project_id,
            name=kwargs.get("name", Path(output_path).stem),
            path=output_path,
            format="jsonl",
            num_rows=len(preference_rows),
            columns=["prompt", "chosen", "rejected"],
            source_action="build_preference_dataset",
            source_paths=[str(checkpoint_path)],
            quality_report=assess_dataset_rows(preference_rows, method="dpo"),
        )
        store.save_dataset_record(project_id, record)

        return json.dumps(
            {"success": True, "record": record.model_dump(mode="json"), "pairs_generated": len(preference_rows)},
            ensure_ascii=False,
            indent=2,
        )

    raise ValueError(f"Unsupported judge action: {action}")


JUDGE_SCHEMA = {
    "name": "judge",
    "description": (
        "Generate completions from model checkpoints, rank them using an auxiliary LLM judge, "
        "and build DPO preference datasets. Integrates Tinker sampling with LLM-as-judge ranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["generate_completions", "rank_completions", "build_preference_dataset"],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "prompts": {"type": "array", "items": {"type": "string"}},
            "checkpoint_path": {"type": "string"},
            "completions": {"type": "array", "items": {"type": "string"}},
            "prompt": {"type": "string"},
            "num_completions": {"type": "integer"},
            "output_path": {"type": "string"},
            "temperature": {"type": "number"},
            "max_tokens": {"type": "integer"},
            "name": {"type": "string"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="judge",
    toolset="research",
    schema=JUDGE_SCHEMA,
    handler=lambda args, **kw: judge_tool(**args),
    check_fn=check_judge_requirements,
)
