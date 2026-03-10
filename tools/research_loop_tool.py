#!/usr/bin/env python3
"""Long-running research loop orchestration tool."""

from __future__ import annotations

import json
import os
import signal
from typing import Any

from cron.jobs import create_job
from research.loop_policy import (
    bootstrap_experiment_template,
    bootstrap_hypothesis_template,
    bootstrap_idea_candidates,
    compose_continuation_prompt,
    compose_monitor_prompt,
    detect_spec_level,
    diagnose_run_failure,
    generate_search_configs,
)
from research.manager import rank_runs, suggest_next_step, triage_literature
from research.models import InboxItem, LoopCheckpoint, now_utc
from research.reporting import (
    render_failure_report,
    render_inbox_markdown,
    render_iteration_report,
    render_research_memo,
    render_session_changelog,
)
from research.state_store import ResearchStateStore
from tools.registry import registry


def check_research_loop_requirements() -> bool:
    """Research loop orchestration is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def _current_origin() -> dict[str, Any] | None:
    platform = os.getenv("HERMES_SESSION_PLATFORM")
    chat_id = os.getenv("HERMES_SESSION_CHAT_ID")
    if platform and chat_id:
        return {
            "platform": platform,
            "chat_id": chat_id,
            "chat_name": os.getenv("HERMES_SESSION_CHAT_NAME"),
        }
    return None


def _pid_is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _schedule_prompt(name: str, prompt: str, schedule: str) -> dict[str, Any]:
    return create_job(prompt=prompt, schedule=schedule, name=name, origin=_current_origin())


def _collect_session_changes(store: ResearchStateStore, project_id: str, loop_state: dict[str, Any]) -> dict[str, Any]:
    """Compare current project state against the last saved snapshot."""
    last = loop_state.get("last_status_snapshot", {})
    if not last:
        return {}

    current = store.status_summary(project_id)
    changes: dict[str, Any] = {}

    for key in ("ideas", "literature_notes", "hypotheses", "experiments"):
        old_val = int(last.get(key, 0))
        new_val = int(current.get(key, 0))
        delta = new_val - old_val
        if delta != 0:
            changes[key] = {"added": max(delta, 0), "removed": max(-delta, 0)}

    old_run_count = int(last.get("run_count", 0))
    new_run_count = int(current.get("run_count", 0))
    if new_run_count != old_run_count:
        changes["runs"] = {"added": max(new_run_count - old_run_count, 0), "removed": 0}

    old_active = set(last.get("active_run_ids", []))
    new_active = set(current.get("active_run_ids", []))
    started = new_active - old_active
    finished = old_active - new_active
    if started:
        changes["runs_started"] = list(started)
    if finished:
        changes["runs_finished"] = list(finished)

    return changes


def research_loop_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform research loop orchestration operations."""
    store = ResearchStateStore(cwd=cwd)
    action = (action or "").strip()

    if action == "bootstrap_project":
        brief = str(kwargs.get("brief") or "")
        spec_level = str(kwargs.get("spec_level") or detect_spec_level(brief))
        project = store.init_project(
            name=kwargs.get("name"),
            brief=brief,
            project_id=kwargs.get("project_id"),
            objective=kwargs.get("objective"),
            spec_level=spec_level,
            metadata=kwargs.get("metadata") or {},
        )

        ideas = []
        hypotheses = []
        experiment = None
        if spec_level == "zero_spec":
            ideas = bootstrap_idea_candidates(project.project_id, brief)
            ideas[0].chosen = True
            for item in ideas:
                store.append_jsonl(project.project_id, "ideas", item.model_dump(mode="json"))
            hypothesis = bootstrap_hypothesis_template(project.project_id, chosen_idea_id=ideas[0].idea_id)
            store.append_jsonl(project.project_id, "hypotheses", hypothesis.model_dump(mode="json"))
            hypotheses.append(hypothesis.model_dump(mode="json"))
            experiment = bootstrap_experiment_template(project.project_id, hypothesis.hypothesis_id)
            store.save_experiment(project.project_id, experiment.experiment_id, experiment.model_dump(mode="json"))
            project.phase = "literature"
        elif spec_level == "partial_spec":
            hypothesis = bootstrap_hypothesis_template(project.project_id)
            store.append_jsonl(project.project_id, "hypotheses", hypothesis.model_dump(mode="json"))
            hypotheses.append(hypothesis.model_dump(mode="json"))
            experiment = bootstrap_experiment_template(project.project_id, hypothesis.hypothesis_id, method=str(kwargs.get("method") or "sft"))
            store.save_experiment(project.project_id, experiment.experiment_id, experiment.model_dump(mode="json"))
            project.phase = "planning"
        else:
            project.phase = "planning"

        store.save_project(project)
        loop_state = {
            "project_id": project.project_id,
            "status": "active",
            "spec_level": spec_level,
            "current_chunk": 0,
            "next_action": "literature_scan" if spec_level == "zero_spec" else "execute_spec",
            "checkpoints": [],
        }
        store.save_loop_state(project.project_id, loop_state)
        return json.dumps(
            {
                "success": True,
                "project": project.model_dump(mode="json"),
                "ideas": [item.model_dump(mode="json") for item in ideas],
                "hypotheses": hypotheses,
                "experiment": experiment.model_dump(mode="json") if experiment else None,
                "loop_state": loop_state,
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "resume_project":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        unread_items = []
        for path in store.unread_inbox(project_id):
            unread_items.append({"path": str(path), "preview": path.read_text(encoding="utf-8")[:500]})
        return json.dumps(
            {
                "success": True,
                "project": project.model_dump(mode="json"),
                "status": store.status_summary(project_id),
                "unread_inbox": unread_items,
                "experiments": store.list_experiments(project_id),
                "runs": store.list_runs(project_id),
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "checkpoint_loop":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        loop_state = store.load_loop_state(project_id)
        checkpoint = LoopCheckpoint(
            project_id=project_id,
            reason=str(kwargs.get("reason") or "checkpoint"),
            summary=str(kwargs.get("summary") or ""),
            next_action=str(kwargs.get("next_action") or ""),
            metadata=kwargs.get("metadata") or {},
        )
        job = None
        if kwargs.get("schedule_next", True):
            status_summary = store.status_summary(project_id)
            prompt = compose_continuation_prompt(project, status_summary, checkpoint.reason, checkpoint.next_action)
            schedule = str(kwargs.get("schedule") or "1m")
            job = _schedule_prompt(f"research-continue-{project_id}", prompt, schedule)
            checkpoint.continuation_job_id = job["id"]
        loop_state.setdefault("checkpoints", []).append(checkpoint.model_dump(mode="json"))
        loop_state["last_checkpoint"] = checkpoint.model_dump(mode="json")
        loop_state["status"] = "waiting" if job else "checkpointed"
        if checkpoint.next_action:
            loop_state["next_action"] = checkpoint.next_action

        # Session summary: diff current state against last snapshot
        session_changes = _collect_session_changes(store, project_id, loop_state)
        if session_changes:
            changelog = render_session_changelog(project, checkpoint.model_dump(mode="json"), session_changes)
            loop_state.setdefault("session_changelogs", []).append(changelog)

        # Save current snapshot for future diffs
        current_summary = store.status_summary(project_id)
        loop_state["last_status_snapshot"] = {
            "ideas": current_summary.get("ideas", 0),
            "literature_notes": current_summary.get("literature_notes", 0),
            "hypotheses": current_summary.get("hypotheses", 0),
            "experiments": current_summary.get("experiments", 0),
            "run_count": current_summary.get("run_count", 0),
            "active_run_ids": current_summary.get("active_run_ids", []),
        }

        store.save_loop_state(project_id, loop_state)
        return json.dumps(
            {
                "success": True,
                "checkpoint": checkpoint.model_dump(mode="json"),
                "job": job,
                "session_changes": session_changes,
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "schedule_continuation":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        status_summary = store.status_summary(project_id)
        prompt = compose_continuation_prompt(
            project=project,
            status_summary=status_summary,
            reason=str(kwargs.get("reason") or "scheduled continuation"),
            next_action=str(kwargs.get("next_action") or ""),
        )
        job = _schedule_prompt(
            f"research-continue-{project_id}",
            prompt,
            str(kwargs.get("schedule") or "1m"),
        )
        return json.dumps({"success": True, "job": job, "prompt": prompt}, ensure_ascii=False, indent=2)

    if action == "monitor_run":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        run_id = str(kwargs.get("run_id") or "")
        if not run_id:
            raise ValueError("run_id is required for monitor_run")
        project = store.load_project(project_id)
        run = store.load_run(project_id, run_id)
        runtime_dir = store.layout(project_id)["runtime"] / run_id
        status_path = runtime_dir / "status.json"
        status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        pid = run.get("pid")
        is_alive = _pid_is_alive(pid)
        effective_status = status_payload.get("status") or run.get("status") or "unknown"
        if effective_status == "running" and not is_alive:
            effective_status = "stopped"
        run["status"] = effective_status
        run["updated_at"] = now_utc()
        if status_payload:
            run["runtime_status"] = status_payload
        store.save_run(run)

        response: dict[str, Any] = {
            "success": True,
            "run": run,
            "runtime_status": status_payload,
            "pid_alive": is_alive,
        }

        if effective_status in {"running", "queued", "pending"}:
            schedule = f"{int(store.config.get('loop_poll_interval_minutes', 30))}m"
            job = _schedule_prompt(
                f"research-monitor-{run_id}",
                compose_monitor_prompt(project, run_id),
                schedule,
            )
            response["job"] = job
        elif effective_status in {"completed", "failed", "stopped"}:
            project.active_run_ids = [item for item in project.active_run_ids if item != run_id]
            store.save_project(project)
            loop_state = store.load_loop_state(project_id)
            loop_state["hyperparam_active"] = [
                item for item in list(loop_state.get("hyperparam_active") or [])
                if item.get("run_id") != run_id
            ]
            store.save_loop_state(project_id, loop_state)
            title = f"Run {run_id} {effective_status}"
            summary = status_payload.get("error") if effective_status == "failed" else f"Run {run_id} finished with status {effective_status}."
            details = render_inbox_markdown(
                title=title,
                summary=summary,
                details=json.dumps(status_payload, indent=2, ensure_ascii=False) if status_payload else "",
                next_steps=["Review results and decide the next experiment."],
            )
            item = InboxItem(project_id=project_id, title=title, summary=summary, details=details)
            inbox_path = store.add_inbox_item(project_id, item)
            response["inbox_path"] = str(inbox_path)
            if effective_status == "completed":
                job = _schedule_prompt(
                    f"research-followup-{project_id}",
                    compose_continuation_prompt(project, store.status_summary(project_id), "run completed", "evaluate_and_report"),
                    "1m",
                )
                response["job"] = job
            elif loop_state.get("hyperparam_queue"):
                job = _schedule_prompt(
                    f"research-search-next-{project_id}",
                    compose_continuation_prompt(project, store.status_summary(project_id), "hyperparameter search continuation", "launch_next_search_run"),
                    "1m",
                )
                response["job"] = job
        return json.dumps(response, ensure_ascii=False, indent=2)

    if action == "finish_project":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        project.phase = "completed"
        project.status = "completed"
        store.save_project(project)
        status_summary = store.status_summary(project_id)
        literature_summary = triage_literature(
            store.load_jsonl(project_id, "literature"),
            objective=project.objective or project.user_brief,
            top_k=int(store.config.get("manager", {}).get("literature_top_k", 10)),
        )
        rankings = rank_runs(store.list_runs(project_id), store.load_evaluations(project_id))
        decision = suggest_next_step(
            project=project,
            status_summary=status_summary,
            runs=store.list_runs(project_id),
            eval_results=store.load_evaluations(project_id),
            datasets=store.list_dataset_records(project_id),
            literature_summary=literature_summary,
            rankings=rankings,
        )
        report = render_iteration_report(
            project=project,
            status_summary=status_summary,
            summary=str(kwargs.get("summary") or "Project finished."),
            next_steps=list(kwargs.get("next_steps") or []),
            run_rows=store.list_runs(project_id),
        )
        report_path = store.write_report(project_id, report, iteration=kwargs.get("iteration"))
        memo = render_research_memo(
            project=project,
            status_summary=status_summary,
            decision=decision,
            ranked_runs=(rankings.get("rankings") or [])[: int(store.config.get("manager", {}).get("memo_top_k_runs", 5))],
            literature_summary=literature_summary,
            dataset_reports=[
                {"dataset_id": row.get("dataset_id"), "name": row.get("name"), **(row.get("quality_report") or {})}
                for row in store.list_dataset_records(project_id)
            ],
        )
        memo_path = store.write_named_report(project_id, f"research-memo-final-{project_id}.md", memo)
        inbox = InboxItem(
            project_id=project_id,
            title=f"Project complete: {project.name}",
            summary=str(kwargs.get("summary") or "The research loop completed and left artifacts in the workspace."),
            details=memo,
        )
        inbox_path = store.add_inbox_item(project_id, inbox)
        return json.dumps(
            {
                "success": True,
                "project": project.model_dump(mode="json"),
                "report_path": str(report_path),
                "memo_path": str(memo_path),
                "inbox_path": str(inbox_path),
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "get_status":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        return json.dumps({"success": True, "status": store.status_summary(project_id)}, ensure_ascii=False, indent=2)

    if action == "auto_advance":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        project = store.load_project(project_id)
        literature_summary = triage_literature(
            store.load_jsonl(project_id, "literature"),
            objective=project.objective or project.user_brief,
            top_k=int(kwargs.get("top_k", store.config.get("manager", {}).get("literature_top_k", 10))),
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
            {"success": True, "decision": decision, "best_run": rankings.get("best_run"), "focus_areas": literature_summary.get("focus_areas", [])},
            ensure_ascii=False,
            indent=2,
        )

    if action == "hyperparam_search":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        strategy = str(kwargs.get("strategy", "grid")).strip()
        param_grid = dict(kwargs.get("param_grid") or {})
        base_config = dict(kwargs.get("base_config") or {})
        max_concurrent = int(kwargs.get("max_concurrent", store.config.get("max_concurrent_runs", 3)))

        if not param_grid:
            raise ValueError("param_grid is required for hyperparam_search")

        configs = generate_search_configs(strategy, param_grid, base_config, max_configs=50)
        loop_state = store.load_loop_state(project_id)
        loop_state.setdefault("hyperparam_queue", [])
        loop_state.setdefault("hyperparam_active", [])

        for idx, cfg in enumerate(configs):
            cfg["_search_index"] = idx
            loop_state["hyperparam_queue"].append(cfg)

        store.save_loop_state(project_id, loop_state)

        return json.dumps(
            {
                "success": True,
                "strategy": strategy,
                "total_configs": len(configs),
                "max_concurrent": max_concurrent,
                "queued": len(configs),
                "message": f"Generated {len(configs)} configs via {strategy} search. Use start_run to launch them.",
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "diagnose_failure":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        run_id = str(kwargs.get("run_id") or "")
        if not run_id:
            raise ValueError("run_id is required for diagnose_failure")

        run = store.load_run(project_id, run_id)
        runtime_dir = store.layout(project_id)["runtime"] / run_id
        status_path = runtime_dir / "status.json"
        runtime_status: dict[str, Any] = {}
        if status_path.exists():
            runtime_status = json.loads(status_path.read_text(encoding="utf-8"))

        diagnosis = diagnose_run_failure(run, runtime_status)
        recovery_options = diagnosis.pop("recovery_options", [])
        report = render_failure_report(run, diagnosis, recovery_options)

        return json.dumps(
            {
                "success": True,
                "diagnosis": diagnosis,
                "recovery_options": recovery_options,
                "report": report,
            },
            ensure_ascii=False,
            indent=2,
        )

    raise ValueError(f"Unsupported research_loop action: {action}")


RESEARCH_LOOP_SCHEMA = {
    "name": "research_loop",
    "description": (
        "Coordinate durable autonomous research loops. Use it to bootstrap projects, resume saved state, "
        "checkpoint long-running work, schedule fresh-session continuations, monitor Tinker runs, finish projects, "
        "and inspect project status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "bootstrap_project",
                    "resume_project",
                    "checkpoint_loop",
                    "schedule_continuation",
                    "monitor_run",
                    "finish_project",
                    "get_status",
                    "auto_advance",
                    "hyperparam_search",
                    "diagnose_failure",
                ],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "name": {"type": "string"},
            "brief": {"type": "string"},
            "objective": {"type": "string"},
            "spec_level": {"type": "string", "enum": ["zero_spec", "partial_spec", "full_spec"]},
            "reason": {"type": "string"},
            "summary": {"type": "string"},
            "next_action": {"type": "string"},
            "schedule": {"type": "string"},
            "schedule_next": {"type": "boolean"},
            "run_id": {"type": "string"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "iteration": {"type": "integer"},
            "metadata": {"type": "object"},
            "strategy": {"type": "string", "enum": ["grid", "random"]},
            "param_grid": {"type": "object", "description": "Dict of param name to list of values for hyperparameter search"},
            "base_config": {"type": "object", "description": "Base run configuration to overlay search params on"},
            "max_concurrent": {"type": "integer"},
            "preferred_metrics": {"type": "array", "items": {"type": "string"}},
            "baseline_run_id": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="research_loop",
    toolset="research",
    schema=RESEARCH_LOOP_SCHEMA,
    handler=lambda args, **kw: research_loop_tool(**args),
    check_fn=check_research_loop_requirements,
)
