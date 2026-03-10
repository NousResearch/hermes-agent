#!/usr/bin/env python3
"""Research decision-management tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.knowledge_graph import KnowledgeGraph
from research.manager import (
    assess_dataset_rows,
    build_recovery_plan,
    link_lineage_edges,
    prune_search_queue,
    queue_recovery_retry,
    rank_runs,
    suggest_next_step,
    triage_literature,
)
from research.reporting import render_inbox_markdown, render_research_memo
from research.state_store import ResearchStateStore
from tools.registry import registry


def check_research_manager_requirements() -> bool:
    """Research management is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def _load_rows(path: str) -> list[dict[str, Any]]:
    payload = Path(path)
    if not payload.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    if payload.suffix == ".json":
        parsed = json.loads(payload.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        raise ValueError("JSON dataset must contain a list of objects")
    for line in payload.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def research_manager_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform research-management operations."""
    store = ResearchStateStore(cwd=cwd)
    project_id = _resolve_project_id(store, kwargs.get("project_id"))
    action = (action or "").strip()

    if action == "triage_literature":
        project = store.load_project(project_id)
        records = store.load_jsonl(project_id, "literature")
        summary = triage_literature(
            records,
            objective=str(kwargs.get("objective") or project.objective or project.user_brief),
            top_k=int(kwargs.get("top_k", 10)),
        )
        edges = []
        if kwargs.get("auto_link"):
            kg = KnowledgeGraph(store, project_id)
            paper_ids = [paper.get("paper_id") or paper.get("record_id") for paper in summary["ranked_papers"] if paper.get("paper_id") or paper.get("record_id")]
            edges = link_lineage_edges(
                kg,
                paper_ids=paper_ids[: int(kwargs.get("link_top_k", 3))],
                idea_id=kwargs.get("idea_id"),
                hypothesis_id=kwargs.get("hypothesis_id"),
                experiment_id=kwargs.get("experiment_id"),
            )
        return json.dumps({"success": True, "summary": summary, "edges": edges}, ensure_ascii=False, indent=2)

    if action == "assess_dataset":
        method = str(kwargs.get("method") or "sft")
        dataset_id = kwargs.get("dataset_id")
        record = None
        source_path = kwargs.get("source_path")
        if dataset_id:
            record = store.load_dataset_record(project_id, str(dataset_id))
            source_path = source_path or record.get("path")
        if not source_path:
            raise ValueError("source_path or dataset_id is required")

        rows = _load_rows(str(source_path))
        report = assess_dataset_rows(rows, method=method, field_map=kwargs.get("field_map") or {})
        if record is not None:
            record["quality_report"] = report
            store.save_dataset_record(project_id, record)
        return json.dumps({"success": True, "report": report, "dataset_id": dataset_id}, ensure_ascii=False, indent=2)

    if action == "rank_runs":
        rankings = rank_runs(
            store.list_runs(project_id),
            store.load_evaluations(project_id),
            preferred_metrics=list(kwargs.get("preferred_metrics") or []),
            baseline_run_id=kwargs.get("baseline_run_id"),
        )
        return json.dumps({"success": True, **rankings}, ensure_ascii=False, indent=2)

    if action == "plan_next_step":
        project = store.load_project(project_id)
        literature_summary = triage_literature(
            store.load_jsonl(project_id, "literature"),
            objective=project.objective or project.user_brief,
            top_k=int(kwargs.get("top_k", 10)),
        )
        rankings = rank_runs(
            store.list_runs(project_id),
            store.load_evaluations(project_id),
            preferred_metrics=list(kwargs.get("preferred_metrics") or []),
            baseline_run_id=kwargs.get("baseline_run_id"),
        )
        decision = suggest_next_step(
            project=project,
            status_summary=store.status_summary(project_id),
            runs=store.list_runs(project_id),
            eval_results=store.load_evaluations(project_id),
            datasets=store.list_dataset_records(project_id),
            literature_summary=literature_summary,
            rankings=rankings,
        )
        return json.dumps(
            {
                "success": True,
                "decision": decision,
                "best_run": rankings.get("best_run"),
                "focus_areas": literature_summary.get("focus_areas", []),
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "prune_search":
        loop_state = store.load_loop_state(project_id)
        queue = list(loop_state.get("hyperparam_queue") or [])
        result = prune_search_queue(
            queue=queue,
            runs=store.list_runs(project_id),
            max_queue=int(kwargs.get("max_queue", store.config.get("manager", {}).get("search_max_queue", 10))),
        )
        loop_state["hyperparam_queue"] = result["kept"]
        store.save_loop_state(project_id, loop_state)
        return json.dumps(
            {
                "success": True,
                "kept": len(result["kept"]),
                "dropped": len(result["dropped"]),
                "dropped_configs": result["dropped"],
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "recover_run":
        run_id = str(kwargs.get("run_id") or "")
        if not run_id:
            raise ValueError("run_id is required for recover_run")
        run = store.load_run(project_id, run_id)
        runtime_status = {}
        runtime_dir = Path(run.get("runtime_dir") or "")
        status_path = runtime_dir / "status.json" if runtime_dir else None
        if status_path and status_path.exists():
            runtime_status = json.loads(status_path.read_text(encoding="utf-8"))
        recovery = build_recovery_plan(run, runtime_status)
        if kwargs.get("queue_retry") and recovery.get("config_patch"):
            loop_state = store.load_loop_state(project_id)
            loop_state = queue_recovery_retry(loop_state, run, recovery)
            store.save_loop_state(project_id, loop_state)
            recovery["queued_retry"] = True
        return json.dumps({"success": True, "recovery": recovery}, ensure_ascii=False, indent=2)

    if action == "write_research_memo":
        project = store.load_project(project_id)
        status_summary = store.status_summary(project_id)
        literature_summary = triage_literature(
            store.load_jsonl(project_id, "literature"),
            objective=project.objective or project.user_brief,
            top_k=int(kwargs.get("top_k", 10)),
        )
        rankings = rank_runs(
            store.list_runs(project_id),
            store.load_evaluations(project_id),
            preferred_metrics=list(kwargs.get("preferred_metrics") or []),
            baseline_run_id=kwargs.get("baseline_run_id"),
        )
        decision = suggest_next_step(
            project=project,
            status_summary=status_summary,
            runs=store.list_runs(project_id),
            eval_results=store.load_evaluations(project_id),
            datasets=store.list_dataset_records(project_id),
            literature_summary=literature_summary,
            rankings=rankings,
        )
        memo = render_research_memo(
            project=project,
            status_summary=status_summary,
            decision=decision,
            ranked_runs=rankings.get("rankings") or [],
            literature_summary=literature_summary,
            dataset_reports=[
                {
                    "dataset_id": row.get("dataset_id"),
                    "name": row.get("name"),
                    **(row.get("quality_report") or {}),
                }
                for row in store.list_dataset_records(project_id)
            ],
            recovery_notes=list(kwargs.get("recovery_notes") or []),
        )
        filename = str(kwargs.get("filename") or f"research-memo-{len(list(store.layout(project_id)['reports'].glob('research-memo-*.md'))) + 1}.md")
        path = store.write_named_report(project_id, filename, memo)
        inbox_body = render_inbox_markdown(
            title=f"Research memo: {project.name}",
            summary=decision.get("summary", "Research memo written."),
            details=memo,
            next_steps=list(decision.get("next_steps") or []),
        )
        inbox_path = store.add_inbox_item(
            project_id,
            {
                "project_id": project_id,
                "title": f"Research memo: {project.name}",
                "summary": decision.get("summary", "Research memo written."),
                "details": inbox_body,
            },
        )
        return json.dumps(
            {
                "success": True,
                "path": str(path),
                "inbox_path": str(inbox_path),
                "decision": decision,
                "best_run": rankings.get("best_run"),
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "link_artifacts":
        kg = KnowledgeGraph(store, project_id)
        edges = link_lineage_edges(
            kg,
            paper_ids=list(kwargs.get("paper_ids") or []),
            idea_id=kwargs.get("idea_id"),
            hypothesis_id=kwargs.get("hypothesis_id"),
            experiment_id=kwargs.get("experiment_id"),
            run_id=kwargs.get("run_id"),
            eval_id=kwargs.get("eval_id"),
        )
        return json.dumps({"success": True, "edges": edges}, ensure_ascii=False, indent=2)

    raise ValueError(f"Unsupported research_manager action: {action}")


RESEARCH_MANAGER_SCHEMA = {
    "name": "research_manager",
    "description": (
        "Decision-management layer for Hermes Research Agent. "
        "Use it to triage literature, score dataset quality, rank runs, plan the next step, "
        "prune search queues, build recovery plans, write research memos, and link research artifacts into a graph."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "triage_literature",
                    "assess_dataset",
                    "rank_runs",
                    "plan_next_step",
                    "prune_search",
                    "recover_run",
                    "write_research_memo",
                    "link_artifacts",
                ],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "objective": {"type": "string"},
            "top_k": {"type": "integer"},
            "auto_link": {"type": "boolean"},
            "link_top_k": {"type": "integer"},
            "paper_ids": {"type": "array", "items": {"type": "string"}},
            "idea_id": {"type": "string"},
            "hypothesis_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
            "eval_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "source_path": {"type": "string"},
            "method": {"type": "string", "enum": ["sft", "dpo", "ppo"]},
            "field_map": {"type": "object"},
            "preferred_metrics": {"type": "array", "items": {"type": "string"}},
            "baseline_run_id": {"type": "string"},
            "max_queue": {"type": "integer"},
            "queue_retry": {"type": "boolean"},
            "filename": {"type": "string"},
            "recovery_notes": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["action"],
    },
}


registry.register(
    name="research_manager",
    toolset="research",
    schema=RESEARCH_MANAGER_SCHEMA,
    handler=lambda args, **kw: research_manager_tool(**args),
    check_fn=check_research_manager_requirements,
)
