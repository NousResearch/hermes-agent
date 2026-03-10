#!/usr/bin/env python3
"""Direct Tinker post-training lifecycle tool."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from cron.jobs import create_job
from research.approvals import ApprovalStore
from research.config import get_research_config, get_research_mode
from research.knowledge_graph import KnowledgeGraph
from research.loop_policy import compose_monitor_prompt
from research.manager import link_lineage_edges
from research.models import now_utc
from research.state_store import ResearchStateStore
from research.tinker_client import build_runner_script, build_runtime_paths, estimate_run_cost, launch_background_run, normalize_run_spec
from tools.registry import registry


def check_tinker_posttrain_requirements() -> bool:
    """Expose the tool even before Tinker is configured."""
    return True


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


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def _pid_is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _runtime_status(store: ResearchStateStore, project_id: str, run_id: str) -> tuple[dict[str, Any], Path]:
    runtime_dir = store.layout(project_id)["runtime"] / run_id
    status_path = runtime_dir / "status.json"
    if not status_path.exists():
        return {}, status_path
    return json.loads(status_path.read_text(encoding="utf-8")), status_path


def _schedule_monitor(store: ResearchStateStore, project_id: str, run_id: str) -> dict[str, Any]:
    project = store.load_project(project_id)
    schedule = f"{int(store.config.get('loop_poll_interval_minutes', 30))}m"
    prompt = compose_monitor_prompt(project, run_id)
    return create_job(
        prompt=prompt,
        schedule=schedule,
        name=f"research-monitor-{run_id}",
        origin=_current_origin(),
    )


def _require_tinker_key() -> None:
    cfg = get_research_config()
    env_name = str(cfg.get("tinker", {}).get("api_key_env", "TINKER_API_KEY"))
    if not os.getenv(env_name):
        raise ValueError(f"{env_name} is not set")


def _check_cost_budget(store: ResearchStateStore, project_id: str) -> None:
    """Raise if cumulative estimated run costs exceed the configured budget."""
    cfg = get_research_config()
    max_cost = float(cfg.get("max_total_cost_usd", 0) or 0)
    if max_cost <= 0:
        return
    runs = store.list_runs(project_id)
    total = sum(float(r.get("estimated_cost_usd") or 0) for r in runs)
    if total >= max_cost:
        raise ValueError(
            f"Cost budget exhausted: ${total:.2f} spent of ${max_cost:.2f} limit. "
            f"Increase research.max_total_cost_usd in config to continue."
        )


def _peek_search_config(store: ResearchStateStore, project_id: str) -> dict[str, Any] | None:
    """Return the next queued hyperparameter-search config without consuming it."""
    loop_state = store.load_loop_state(project_id)
    queue = list(loop_state.get("hyperparam_queue") or [])
    if not queue:
        return None
    return dict(queue[0])


def _dequeue_search_config(store: ResearchStateStore, project_id: str) -> dict[str, Any] | None:
    """Pull the next queued hyperparameter-search config, if any."""
    loop_state = store.load_loop_state(project_id)
    queue = list(loop_state.get("hyperparam_queue") or [])
    if not queue:
        return None
    next_config = dict(queue.pop(0))
    loop_state["hyperparam_queue"] = queue
    store.save_loop_state(project_id, loop_state)
    return next_config


def _register_search_run(store: ResearchStateStore, project_id: str, run_id: str, search_index: Any) -> None:
    """Track a run that was launched from the hyperparameter queue."""
    if search_index is None:
        return
    loop_state = store.load_loop_state(project_id)
    active = list(loop_state.get("hyperparam_active") or [])
    active.append({"run_id": run_id, "search_index": search_index})
    loop_state["hyperparam_active"] = active
    store.save_loop_state(project_id, loop_state)


def _complete_search_run(store: ResearchStateStore, project_id: str, run_id: str) -> dict[str, Any]:
    """Remove a finished/stopped run from the active hyperparameter search set."""
    loop_state = store.load_loop_state(project_id)
    active = list(loop_state.get("hyperparam_active") or [])
    loop_state["hyperparam_active"] = [item for item in active if item.get("run_id") != run_id]
    store.save_loop_state(project_id, loop_state)
    return loop_state


def _should_gate(action: str, mode_override: str | None = None) -> bool:
    cfg = get_research_config()
    mode = get_research_mode(mode_override)
    if mode == "auto":
        return False
    return action in set(cfg.get("require_approval_for", []))


def _handle_approval(
    approvals: ApprovalStore,
    project_id: str,
    action: str,
    mode_override: str | None,
    payload: dict[str, Any],
    approval_id: str | None,
    approved: bool,
) -> dict[str, Any] | None:
    if not _should_gate(action, mode_override):
        return None
    if approval_id and approvals.is_approved(project_id, action, approval_id):
        return None
    if approval_id and approved:
        approvals.approve(project_id, approval_id)
        return None
    record = approvals.request(project_id, action, payload, reason=f"{action} requires approval in research mode")
    return {
        "success": True,
        "status": "pending_approval",
        "approval_id": record["approval_id"],
        "message": f"{action} is waiting for approval. Retry with approval_id and approved=true to proceed.",
    }


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


def tinker_posttrain_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform Tinker lifecycle operations."""
    store = ResearchStateStore(cwd=cwd)
    approvals = ApprovalStore(store)
    action = (action or "").strip()

    if action == "validate_config":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        raw_config = dict(kwargs.get("config") or {})
        if not raw_config:
            raw_config = {key: value for key, value in kwargs.items() if key not in {"cwd", "project_id", "action"}}
        spec = normalize_run_spec(raw_config, project_id=project_id, cwd=store.cwd, experiment_id=kwargs.get("experiment_id"))
        return json.dumps({"success": True, "config": spec.model_dump(mode="json")}, ensure_ascii=False, indent=2)

    if action == "start_run":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        raw_config = dict(kwargs.get("config") or {})
        used_search_queue = False
        if not raw_config:
            queued_config = _peek_search_config(store, project_id)
            if queued_config:
                raw_config = dict(queued_config)
                used_search_queue = True
        if not raw_config:
            raw_config = {key: value for key, value in kwargs.items() if key not in {"cwd", "project_id", "action", "mode", "approval_id", "approved"}}
        gated = _handle_approval(
            approvals=approvals,
            project_id=project_id,
            action=action,
            mode_override=kwargs.get("mode"),
            payload=raw_config,
            approval_id=kwargs.get("approval_id"),
            approved=bool(kwargs.get("approved")),
        )
        if gated:
            return json.dumps(gated, ensure_ascii=False, indent=2)
        if used_search_queue:
            consumed = _dequeue_search_config(store, project_id)
            if consumed:
                raw_config = dict(consumed)

        _require_tinker_key()
        _check_cost_budget(store, project_id)
        spec = normalize_run_spec(raw_config, project_id=project_id, cwd=store.cwd, experiment_id=kwargs.get("experiment_id"))
        if spec.estimated_cost_usd is None:
            spec.estimated_cost_usd = estimate_run_cost(spec)
        project = store.load_project(project_id)
        paths = build_runtime_paths(store.project_dir(project_id), spec.run_id)
        paths.config_path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.runner_path.write_text(build_runner_script(spec, paths), encoding="utf-8")
        os.chmod(paths.runner_path, 0o755)
        paths.status_path.write_text(
            json.dumps({"run_id": spec.run_id, "project_id": project_id, "status": "queued", "step": spec.resume_step}, indent=2),
            encoding="utf-8",
        )
        process = launch_background_run(paths, cwd=project.cwd)
        monitor_job = _schedule_monitor(store, project_id, spec.run_id)
        record = spec.model_dump(mode="json")
        record.update(
            {
                "status": "running",
                "pid": process.pid,
                "runtime_dir": str(paths.runtime_dir),
                "config_path": str(paths.config_path),
                "runner_path": str(paths.runner_path),
                "status_path": str(paths.status_path),
                "log_path": str(paths.log_path),
                "monitor_job_id": monitor_job["id"],
                "updated_at": now_utc(),
            }
        )
        search_index = raw_config.pop("_search_index", None)
        if search_index is not None:
            metadata = dict(record.get("metadata") or {})
            metadata["search_index"] = search_index
            record["metadata"] = metadata
        store.save_run(record)
        _register_search_run(store, project_id, spec.run_id, search_index)
        project.active_run_ids = sorted(set(project.active_run_ids + [spec.run_id]))
        project.phase = "training"
        store.save_project(project)
        if spec.experiment_id:
            link_lineage_edges(
                KnowledgeGraph(store, project_id),
                experiment_id=spec.experiment_id,
                run_id=spec.run_id,
            )
        return json.dumps({"success": True, "run": record, "monitor_job": monitor_job}, ensure_ascii=False, indent=2)

    if action == "status":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        run = store.load_run(project_id, str(kwargs.get("run_id") or ""))
        status_payload, status_path = _runtime_status(store, project_id, run["run_id"])
        pid_alive = _pid_is_alive(run.get("pid"))
        effective_status = status_payload.get("status") or run.get("status")
        if effective_status == "running" and not pid_alive:
            effective_status = "stopped"
        run["status"] = effective_status
        run["updated_at"] = now_utc()
        run["runtime_status"] = status_payload
        store.save_run(run)
        return json.dumps(
            {
                "success": True,
                "run": run,
                "runtime_status": status_payload,
                "status_path": str(status_path),
                "pid_alive": pid_alive,
            },
            ensure_ascii=False,
            indent=2,
        )

    if action == "list_runs":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        return json.dumps({"success": True, "runs": store.list_runs(project_id)}, ensure_ascii=False, indent=2)

    if action == "resume_run":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        run = store.load_run(project_id, str(kwargs.get("run_id") or ""))
        gated = _handle_approval(
            approvals=approvals,
            project_id=project_id,
            action=action,
            mode_override=kwargs.get("mode"),
            payload=run,
            approval_id=kwargs.get("approval_id"),
            approved=bool(kwargs.get("approved")),
        )
        if gated:
            return json.dumps(gated, ensure_ascii=False, indent=2)

        _require_tinker_key()
        _check_cost_budget(store, project_id)
        status_payload, _ = _runtime_status(store, project_id, run["run_id"])
        last_checkpoint_path = Path(run["runtime_dir"]) / "last_checkpoint.txt"
        resume_from = kwargs.get("resume_from") or status_payload.get("last_checkpoint")
        if not resume_from and last_checkpoint_path.exists():
            resume_from = last_checkpoint_path.read_text(encoding="utf-8").strip()
        if not resume_from:
            raise ValueError("No checkpoint available to resume from")

        raw_config = dict(run)
        raw_config["resume_from"] = resume_from
        raw_config["resume_step"] = int(status_payload.get("step") or 0)
        spec = normalize_run_spec(raw_config, project_id=project_id, cwd=store.cwd, experiment_id=run.get("experiment_id"), run_id=run["run_id"])
        project = store.load_project(project_id)
        paths = build_runtime_paths(store.project_dir(project_id), spec.run_id)
        paths.config_path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.runner_path.write_text(build_runner_script(spec, paths), encoding="utf-8")
        os.chmod(paths.runner_path, 0o755)
        process = launch_background_run(paths, cwd=project.cwd)
        monitor_job = _schedule_monitor(store, project_id, spec.run_id)
        run.update(
            {
                "status": "running",
                "pid": process.pid,
                "resume_from": resume_from,
                "resume_step": spec.resume_step,
                "monitor_job_id": monitor_job["id"],
                "updated_at": now_utc(),
            }
        )
        store.save_run(run)
        return json.dumps({"success": True, "run": run, "monitor_job": monitor_job}, ensure_ascii=False, indent=2)

    if action == "stop_run":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        run = store.load_run(project_id, str(kwargs.get("run_id") or ""))
        gated = _handle_approval(
            approvals=approvals,
            project_id=project_id,
            action=action,
            mode_override=kwargs.get("mode"),
            payload=run,
            approval_id=kwargs.get("approval_id"),
            approved=bool(kwargs.get("approved")),
        )
        if gated:
            return json.dumps(gated, ensure_ascii=False, indent=2)

        pid = run.get("pid")
        if pid and _pid_is_alive(pid):
            os.kill(pid, signal.SIGTERM)
        run["status"] = "stopped"
        run["updated_at"] = now_utc()
        store.save_run(run)
        project = store.load_project(project_id)
        project.active_run_ids = [item for item in project.active_run_ids if item != run["run_id"]]
        store.save_project(project)
        _complete_search_run(store, project_id, run["run_id"])
        return json.dumps({"success": True, "run": run}, ensure_ascii=False, indent=2)

    if action == "sample_checkpoint":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        model_path = kwargs.get("model_path")
        if not model_path and kwargs.get("run_id"):
            run = store.load_run(project_id, str(kwargs["run_id"]))
            sampler_path = Path(run["runtime_dir"]) / "last_sampler.txt"
            if sampler_path.exists():
                model_path = sampler_path.read_text(encoding="utf-8").strip()
        if not model_path:
            raise ValueError("model_path or run_id with sampler path is required")
        prompt = str(kwargs.get("prompt") or "")
        code = """
import json, sys
import tinker

model_path, prompt, max_tokens, temperature = sys.argv[1:5]
service = tinker.ServiceClient()
sampler = service.create_sampling_client(model_path=model_path)
params = tinker.SamplingParams(max_tokens=int(max_tokens), temperature=float(temperature))
raw = sampler.sample(prompt=prompt, sampling_params=params).result()
text = getattr(raw, "text", None)
if text is None and hasattr(raw, "sequences") and raw.sequences:
    text = getattr(raw.sequences[0], "text", None)
if text is None:
    text = str(raw)
print(json.dumps({"model_path": model_path, "output": text}))
"""
        payload = _run_python_json(code, [str(model_path), prompt, str(kwargs.get("max_tokens", 256)), str(kwargs.get("temperature", 0.7))])
        return json.dumps({"success": True, **payload}, ensure_ascii=False, indent=2)

    if action == "download_checkpoint":
        project_id = _resolve_project_id(store, kwargs.get("project_id"))
        model_path = kwargs.get("model_path")
        if not model_path and kwargs.get("run_id"):
            run = store.load_run(project_id, str(kwargs["run_id"]))
            checkpoint_path = Path(run["runtime_dir"]) / "last_checkpoint.txt"
            if checkpoint_path.exists():
                model_path = checkpoint_path.read_text(encoding="utf-8").strip()
        if not model_path:
            raise ValueError("model_path or run_id with checkpoint path is required")
        output_path = kwargs.get("output_path") or str((store.cwd / "checkpoint.tar.gz").resolve())
        code = """
import json, sys, urllib.request
import tinker

model_path, output_path = sys.argv[1:3]
service = tinker.ServiceClient()
rest = service.create_rest_client()
response = rest.get_checkpoint_archive_url_from_tinker_path(model_path).result()
urllib.request.urlretrieve(response.url, output_path)
print(json.dumps({"model_path": model_path, "output_path": output_path, "download_url": response.url}))
"""
        payload = _run_python_json(code, [str(model_path), str(output_path)])
        return json.dumps({"success": True, **payload}, ensure_ascii=False, indent=2)

    raise ValueError(f"Unsupported tinker_posttrain action: {action}")


TINKER_POSTTRAIN_SCHEMA = {
    "name": "tinker_posttrain",
    "description": (
        "Manage Tinker-based post-training runs for Hermes Research Agent. "
        "Use it to validate configs, start and resume runs, check status, stop runs, sample checkpoints, "
        "download checkpoints, and list project runs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "validate_config",
                    "start_run",
                    "status",
                    "list_runs",
                    "resume_run",
                    "stop_run",
                    "sample_checkpoint",
                    "download_checkpoint",
                ],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
            "mode": {"type": "string", "enum": ["approval", "auto"]},
            "approval_id": {"type": "string"},
            "approved": {"type": "boolean"},
            "config": {"type": "object"},
            "method": {"type": "string", "enum": ["sft", "dpo", "ppo"]},
            "base_model": {"type": "string"},
            "dataset_path": {"type": "string"},
            "field_map": {"type": "object"},
            "batch_size": {"type": "integer"},
            "learning_rate": {"type": "number"},
            "lora_rank": {"type": "integer"},
            "num_epochs": {"type": "integer"},
            "max_steps": {"type": "integer"},
            "save_every": {"type": "integer"},
            "reference_model_name": {"type": "string"},
            "beta": {"type": "number"},
            "sampling_params": {"type": "object"},
            "optimizer": {"type": "object"},
            "metadata": {"type": "object"},
            "prompt": {"type": "string"},
            "max_tokens": {"type": "integer"},
            "temperature": {"type": "number"},
            "model_path": {"type": "string"},
            "output_path": {"type": "string"},
            "resume_from": {"type": "string"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="tinker_posttrain",
    toolset="research",
    schema=TINKER_POSTTRAIN_SCHEMA,
    handler=lambda args, **kw: tinker_posttrain_tool(**args),
    check_fn=check_tinker_posttrain_requirements,
)
