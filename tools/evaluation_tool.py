#!/usr/bin/env python3
"""Automated evaluation harness tool."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from research.config import get_research_config
from research.knowledge_graph import KnowledgeGraph
from research.manager import link_lineage_edges
from research.models import EvalResult, now_utc
from research.reporting import render_comparison_table
from research.state_store import ResearchStateStore
from tools.registry import registry


def check_evaluation_requirements() -> bool:
    """Evaluation tool is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def evaluation_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform evaluation operations."""
    store = ResearchStateStore(cwd=cwd)
    action = (action or "").strip()
    cfg = get_research_config()
    eval_cfg = cfg.get("evaluation", {})

    if action == "run_eval":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        checkpoint_path = kwargs.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for run_eval")

        benchmarks = kwargs.get("benchmarks") or eval_cfg.get("default_benchmarks", ["hellaswag"])
        num_fewshot = kwargs.get("num_fewshot", eval_cfg.get("default_num_fewshot", 0))
        batch_size = kwargs.get("batch_size", eval_cfg.get("default_batch_size", 32))
        model_backend = kwargs.get("model_backend", eval_cfg.get("model_backend", "hf"))

        if model_backend == "vllm":
            model_args = f"pretrained={checkpoint_path},backend=vllm"
        else:
            model_args = f"pretrained={checkpoint_path}"

        results = []
        kg = KnowledgeGraph(store, project_id)
        for benchmark in benchmarks:
            eval_result = EvalResult(
                project_id=project_id,
                run_id=kwargs.get("run_id"),
                experiment_id=kwargs.get("experiment_id"),
                benchmark=benchmark,
                num_fewshot=num_fewshot,
                model_args=model_args,
                status="running",
            )

            cmd = [
                sys.executable, "-m", "lm_eval",
                "--model", model_backend,
                "--model_args", model_args,
                "--tasks", benchmark,
                "--num_fewshot", str(num_fewshot),
                "--batch_size", str(batch_size),
                "--output_path", str(store.layout(project_id)["runtime"] / f"eval_{eval_result.eval_id}"),
            ]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                stdout, stderr = proc.communicate(timeout=3600)
                if proc.returncode == 0:
                    try:
                        output = json.loads(stdout)
                        metrics = {}
                        task_results = output.get("results", {})
                        for task_name, task_metrics in task_results.items():
                            for metric_name, value in task_metrics.items():
                                if isinstance(value, (int, float)):
                                    metrics[f"{task_name}/{metric_name}"] = value
                        eval_result.metrics = metrics
                        eval_result.status = "completed"
                    except (json.JSONDecodeError, KeyError):
                        eval_result.metrics = {"raw_output": stdout[:2000]}
                        eval_result.status = "completed"
                else:
                    eval_result.status = "failed"
                    eval_result.error = stderr[:2000]
            except subprocess.TimeoutExpired:
                proc.kill()
                eval_result.status = "failed"
                eval_result.error = "Evaluation timed out after 3600 seconds"
            except FileNotFoundError:
                eval_result.status = "failed"
                eval_result.error = "lm_eval not installed. Install with: pip install lm-eval"

            store.append_jsonl(project_id, "evaluations", eval_result.model_dump(mode="json"))
            link_lineage_edges(
                kg,
                experiment_id=kwargs.get("experiment_id"),
                run_id=kwargs.get("run_id"),
                eval_id=eval_result.eval_id,
            )
            results.append(eval_result.model_dump(mode="json"))

        return json.dumps({"success": True, "results": results}, ensure_ascii=False, indent=2)

    if action == "get_eval_results":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        evals = store.load_evaluations(project_id, run_id=kwargs.get("run_id"))
        benchmark = kwargs.get("benchmark")
        experiment_id = kwargs.get("experiment_id")
        if benchmark:
            evals = [e for e in evals if e.get("benchmark") == benchmark]
        if experiment_id:
            evals = [e for e in evals if e.get("experiment_id") == experiment_id]
        return json.dumps({"success": True, "results": evals}, ensure_ascii=False, indent=2)

    if action == "compare_evals":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        eval_ids = kwargs.get("eval_ids") or []
        metrics_filter = kwargs.get("metrics")
        all_evals = store.load_evaluations(project_id)
        if eval_ids:
            selected = [e for e in all_evals if e.get("eval_id") in eval_ids]
        else:
            selected = all_evals
        experiments = store.list_experiments(project_id)
        metric_names = list(metrics_filter) if metrics_filter else None
        table = render_comparison_table(experiments, selected, metric_names)
        return json.dumps({"success": True, "comparison": table}, ensure_ascii=False, indent=2)

    raise ValueError(f"Unsupported evaluation action: {action}")


EVALUATION_SCHEMA = {
    "name": "evaluation",
    "description": (
        "Run automated evaluations on model checkpoints using lm-eval-harness. "
        "Supports running benchmarks, retrieving results, and comparing evaluations across experiments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run_eval", "get_eval_results", "compare_evals"],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "run_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "checkpoint_path": {"type": "string"},
            "benchmarks": {"type": "array", "items": {"type": "string"}},
            "num_fewshot": {"type": "integer"},
            "batch_size": {"type": "integer"},
            "model_backend": {"type": "string", "enum": ["hf", "vllm"]},
            "eval_ids": {"type": "array", "items": {"type": "string"}},
            "metrics": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    },
}


registry.register(
    name="evaluation",
    toolset="research",
    schema=EVALUATION_SCHEMA,
    handler=lambda args, **kw: evaluation_tool(**args),
    check_fn=check_evaluation_requirements,
)
