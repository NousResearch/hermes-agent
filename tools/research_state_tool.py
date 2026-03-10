#!/usr/bin/env python3
"""Structured research state tool."""

from __future__ import annotations

import json
from typing import Any

from research.knowledge_graph import KnowledgeGraph
from research.loop_policy import detect_spec_level
from research.models import ExperimentSpec, Hypothesis, LiteratureRecord, now_utc
from research.reporting import render_comparison_table, render_iteration_report
from research.state_store import ResearchStateStore
from tools.registry import registry


def check_research_state_requirements() -> bool:
    """Research state is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def research_state_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform filesystem-backed research state operations."""
    store = ResearchStateStore(cwd=cwd)
    action = (action or "").strip()

    if action == "init_project":
        brief = str(kwargs.get("brief", "") or "")
        project = store.init_project(
            name=kwargs.get("name"),
            brief=brief,
            project_id=kwargs.get("project_id"),
            objective=kwargs.get("objective"),
            spec_level=kwargs.get("spec_level") or detect_spec_level(brief),
            metadata=kwargs.get("metadata") or {},
        )
        return json.dumps(
            {
                "success": True,
                "project": project.model_dump(mode="json"),
                "layout": {key: str(path) for key, path in store.layout(project.project_id).items()},
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "get_project":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        return json.dumps(
            {
                "success": True,
                "project": project.model_dump(mode="json"),
                "status": store.status_summary(project_id),
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "list_projects":
        return json.dumps(
            {
                "success": True,
                "projects": [project.model_dump(mode="json") for project in store.list_projects()],
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "record_literature":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        record = LiteratureRecord(
            paper_id=str(kwargs.get("paper_id") or kwargs.get("note_id") or f"lit_{now_utc().replace(':', '').replace('-', '')}"),
            title=str(kwargs.get("title") or "Literature note"),
            abstract=str(kwargs.get("abstract") or kwargs.get("summary") or kwargs.get("note") or ""),
            url=str(kwargs.get("url") or ""),
            source=str(kwargs.get("source") or "manual"),
            tags=list(kwargs.get("tags") or []),
            summary=str(kwargs.get("summary") or kwargs.get("note") or ""),
            year=kwargs.get("year"),
            citation_count=kwargs.get("citation_count"),
            authors=list(kwargs.get("authors") or []),
            metadata=kwargs.get("metadata") or {},
        )
        store.append_jsonl(project_id, "literature", record.model_dump(mode="json"))
        kg = KnowledgeGraph(store, project_id)
        edges = []
        if kwargs.get("linked_idea_id"):
            edges.append(kg.add_edge("paper", record.paper_id, "idea", str(kwargs["linked_idea_id"]), "inspires"))
        if kwargs.get("linked_hypothesis_id"):
            edges.append(kg.add_edge("paper", record.paper_id, "hypothesis", str(kwargs["linked_hypothesis_id"]), "supports"))
        if kwargs.get("linked_experiment_id"):
            edges.append(kg.add_edge("paper", record.paper_id, "experiment", str(kwargs["linked_experiment_id"]), "motivates"))
        return json.dumps({"success": True, "record": record.model_dump(mode="json"), "edges": edges}, ensure_ascii=False, indent=2)

    if action == "record_hypothesis":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        hypothesis = Hypothesis(
            statement=str(kwargs.get("statement") or ""),
            success_metric=str(kwargs.get("success_metric") or ""),
            stop_condition=str(kwargs.get("stop_condition") or ""),
            rationale=str(kwargs.get("rationale") or ""),
            linked_idea_id=kwargs.get("linked_idea_id"),
            status=str(kwargs.get("status") or "draft"),
        )
        store.append_jsonl(project_id, "hypotheses", hypothesis.model_dump(mode="json"))
        edges = []
        if hypothesis.linked_idea_id:
            kg = KnowledgeGraph(store, project_id)
            edges.append(kg.add_edge("idea", str(hypothesis.linked_idea_id), "hypothesis", hypothesis.hypothesis_id, "motivates"))
        return json.dumps({"success": True, "hypothesis": hypothesis.model_dump(mode="json"), "edges": edges}, ensure_ascii=False, indent=2)

    if action == "record_experiment_plan":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        experiment_payload = {
            "title": str(kwargs.get("title") or "Experiment"),
            "hypothesis_id": str(kwargs.get("hypothesis_id") or ""),
            "objective": str(kwargs.get("objective") or ""),
            "method": str(kwargs.get("method") or "sft"),
            "dataset_path": str(kwargs.get("dataset_path") or ""),
            "evaluation_plan": list(kwargs.get("evaluation_plan") or []),
            "success_metric": str(kwargs.get("success_metric") or ""),
            "stop_condition": str(kwargs.get("stop_condition") or ""),
            "status": str(kwargs.get("status") or "draft"),
            "metadata": kwargs.get("metadata") or {},
        }
        if kwargs.get("experiment_id"):
            experiment_payload["experiment_id"] = kwargs["experiment_id"]
        experiment = ExperimentSpec(**experiment_payload)
        path = store.save_experiment(project_id, experiment.experiment_id, experiment.model_dump(mode="json"))
        edges = []
        if experiment.hypothesis_id:
            kg = KnowledgeGraph(store, project_id)
            edges.append(kg.add_edge("hypothesis", experiment.hypothesis_id, "experiment", experiment.experiment_id, "tests"))
        return json.dumps(
            {"success": True, "experiment": experiment.model_dump(mode="json"), "path": str(path), "edges": edges},
            ensure_ascii=False,
            indent=2,
        )

    if action == "record_result":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        result = {
            "result_id": kwargs.get("result_id") or f"result_{now_utc().replace(':', '').replace('-', '')}",
            "experiment_id": kwargs.get("experiment_id"),
            "run_id": kwargs.get("run_id"),
            "summary": kwargs.get("summary", ""),
            "metrics": kwargs.get("metrics") or {},
            "next_steps": kwargs.get("next_steps") or [],
            "status": kwargs.get("status") or "recorded",
            "created_at": now_utc(),
        }
        store.append_result(project_id, result)
        edges = []
        kg = KnowledgeGraph(store, project_id)
        if result.get("experiment_id"):
            edges.append(kg.add_edge("experiment", str(result["experiment_id"]), "result", str(result["result_id"]), "produced"))
        if result.get("run_id"):
            edges.append(kg.add_edge("run", str(result["run_id"]), "result", str(result["result_id"]), "produced"))
        return json.dumps({"success": True, "result": result, "edges": edges}, ensure_ascii=False, indent=2)

    if action == "set_phase":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        project.phase = str(kwargs.get("phase") or project.phase)
        project.updated_at = now_utc()
        store.save_project(project)
        return json.dumps({"success": True, "project": project.model_dump(mode="json")}, ensure_ascii=False, indent=2)

    if action == "generate_report":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        status = store.status_summary(project_id)
        report = render_iteration_report(
            project=project,
            status_summary=status,
            summary=str(kwargs.get("summary") or "Research iteration completed."),
            next_steps=list(kwargs.get("next_steps") or []),
            run_rows=store.list_runs(project_id),
        )
        path = store.write_report(project_id, report, iteration=kwargs.get("iteration"))
        return json.dumps({"success": True, "path": str(path), "report": report}, ensure_ascii=False, indent=2)

    if action == "compare_experiments":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        experiment_ids = kwargs.get("experiment_ids") or []
        experiments = store.list_experiments(project_id)
        if experiment_ids:
            experiments = [e for e in experiments if e.get("experiment_id") in experiment_ids]
        eval_results = store.load_evaluations(project_id)
        metrics_filter = kwargs.get("metrics")
        metric_names = list(metrics_filter) if metrics_filter else None
        table = render_comparison_table(experiments, eval_results, metric_names)
        return json.dumps({"success": True, "comparison": table}, ensure_ascii=False, indent=2)

    if action == "query_graph":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        kg = KnowledgeGraph(store, project_id)
        graph_action = str(kwargs.get("graph_action", "summary")).strip()

        if graph_action == "summary":
            return json.dumps({"success": True, "graph": kg.summary()}, ensure_ascii=False, indent=2)

        if graph_action == "neighbors":
            node_type = str(kwargs.get("node_type") or "")
            node_id = str(kwargs.get("node_id") or "")
            if not node_type or not node_id:
                raise ValueError("node_type and node_id are required for neighbors")
            neighbors = kg.get_neighbors(
                node_type, node_id,
                direction=str(kwargs.get("direction", "both")),
                relation=kwargs.get("relation"),
            )
            return json.dumps({"success": True, "neighbors": neighbors}, ensure_ascii=False, indent=2)

        if graph_action == "lineage":
            node_type = str(kwargs.get("node_type") or "")
            node_id = str(kwargs.get("node_id") or "")
            if not node_type or not node_id:
                raise ValueError("node_type and node_id are required for lineage")
            lineage = kg.get_lineage(node_type, node_id)
            return json.dumps({"success": True, "lineage": lineage}, ensure_ascii=False, indent=2)

        if graph_action == "subgraph":
            node_type = str(kwargs.get("node_type") or "")
            node_id = str(kwargs.get("node_id") or "")
            if not node_type or not node_id:
                raise ValueError("node_type and node_id are required for subgraph")
            depth = int(kwargs.get("depth", 2))
            subgraph = kg.get_subgraph(node_type, node_id, depth=depth)
            return json.dumps({"success": True, "subgraph": subgraph}, ensure_ascii=False, indent=2)

        if graph_action == "add_edge":
            source_type = str(kwargs.get("source_type") or "")
            source_id = str(kwargs.get("source_id") or "")
            target_type = str(kwargs.get("target_type") or "")
            target_id = str(kwargs.get("target_id") or "")
            relation = str(kwargs.get("relation") or "")
            if not all([source_type, source_id, target_type, target_id, relation]):
                raise ValueError("source_type, source_id, target_type, target_id, and relation are all required")
            edge = kg.add_edge(source_type, source_id, target_type, target_id, relation)
            return json.dumps({"success": True, "edge": edge}, ensure_ascii=False, indent=2)

        raise ValueError(f"Unknown graph_action: {graph_action}")

    raise ValueError(f"Unsupported research_state action: {action}")


RESEARCH_STATE_SCHEMA = {
    "name": "research_state",
    "description": (
        "Manage persistent research project state under .hermes-research/. "
        "Use it to initialize projects, record literature and hypotheses, save experiment plans, "
        "track results, update project phase, and generate iteration reports."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "init_project",
                    "get_project",
                    "record_literature",
                    "record_hypothesis",
                    "record_experiment_plan",
                    "record_result",
                    "set_phase",
                    "generate_report",
                    "list_projects",
                    "compare_experiments",
                    "query_graph",
                ],
            },
            "cwd": {"type": "string", "description": "Working directory to anchor the research workspace."},
            "project_id": {"type": "string"},
            "name": {"type": "string"},
            "brief": {"type": "string"},
            "objective": {"type": "string"},
            "spec_level": {"type": "string", "enum": ["zero_spec", "partial_spec", "full_spec"]},
            "title": {"type": "string"},
            "paper_id": {"type": "string"},
            "url": {"type": "string"},
            "source": {"type": "string"},
            "abstract": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "year": {"type": "integer"},
            "citation_count": {"type": "integer"},
            "summary": {"type": "string"},
            "note": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "linked_idea_id": {"type": "string"},
            "linked_hypothesis_id": {"type": "string"},
            "linked_experiment_id": {"type": "string"},
            "statement": {"type": "string"},
            "success_metric": {"type": "string"},
            "stop_condition": {"type": "string"},
            "rationale": {"type": "string"},
            "hypothesis_id": {"type": "string"},
            "method": {"type": "string", "enum": ["sft", "dpo", "ppo"]},
            "dataset_path": {"type": "string"},
            "evaluation_plan": {"type": "array", "items": {"type": "string"}},
            "phase": {"type": "string"},
            "run_id": {"type": "string"},
            "metrics": {"type": "object"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "iteration": {"type": "integer"},
            "metadata": {"type": "object"},
            "experiment_ids": {"type": "array", "items": {"type": "string"}},
            "graph_action": {"type": "string", "enum": ["summary", "neighbors", "lineage", "subgraph", "add_edge"]},
            "node_type": {"type": "string"},
            "node_id": {"type": "string"},
            "source_type": {"type": "string"},
            "source_id": {"type": "string"},
            "target_type": {"type": "string"},
            "target_id": {"type": "string"},
            "relation": {"type": "string"},
            "direction": {"type": "string", "enum": ["in", "out", "both"]},
            "depth": {"type": "integer"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="research_state",
    toolset="research",
    schema=RESEARCH_STATE_SCHEMA,
    handler=lambda args, **kw: research_state_tool(**args),
    check_fn=check_research_state_requirements,
)
