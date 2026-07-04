"""
Workflow DAG Engine — Dependency-Aware Multi-Step Task Execution.

Extends ``delegate_task`` with a ``workflow`` parameter that accepts
a list of step definitions with explicit dependencies (``needs``).
The engine:

1. Validates the DAG (cycle detection, missing deps)
2. Topologically sorts steps
3. Executes ready steps in parallel, respecting concurrency limits
4. Passes upstream results into downstream step context automatically
5. Optionally runs a ``synthesis`` aggregator after all steps complete

Designed to be called from ``delegate_tool.py`` which supplies the
internal functions for child agent construction and execution.

Example usage (model-visible)::

    delegate_task(
        workflow=[
            {"id": "research", "goal": "Research Stripe API",
             "toolsets": ["web"]},
            {"id": "backend",  "goal": "Implement payment client",
             "needs": ["research"], "toolsets": ["terminal", "file"]},
            {"id": "test",     "goal": "Write integration tests",
             "needs": ["backend"], "toolsets": ["terminal"]},
        ],
        synthesis="Compile all results into a final report"
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG Validation
# ---------------------------------------------------------------------------


class WorkflowValidationError(ValueError):
    """Raised when a workflow definition is invalid."""


def _validate_workflow(steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Validate a workflow definition and return a step map keyed by id.

    Checks:
    - All steps have unique 'id'
    - All 'needs' references point to existing steps
    - No circular dependencies
    - No duplicate ids

    Returns {step_id: step_dict} on success.
    Raises WorkflowValidationError on failure.
    """
    if not steps:
        raise WorkflowValidationError("Workflow must contain at least one step")

    step_map: Dict[str, Dict[str, Any]] = {}

    for i, step in enumerate(steps):
        step_id = step.get("id")
        if not step_id or not isinstance(step_id, str) or not step_id.strip():
            raise WorkflowValidationError(
                f"Step {i}: missing or invalid 'id' (must be a non-empty string)"
            )
        step_id = step_id.strip()
        if step_id in step_map:
            raise WorkflowValidationError(
                f"Duplicate step id: '{step_id}'"
            )
        if not step.get("goal"):
            raise WorkflowValidationError(
                f"Step '{step_id}': missing 'goal'"
            )
        step_map[step_id] = step

    # Validate needs references
    all_ids = set(step_map.keys())
    for step_id, step in step_map.items():
        needs = step.get("needs")
        if isinstance(needs, list):
            for dep in needs:
                if dep not in all_ids:
                    raise WorkflowValidationError(
                        f"Step '{step_id}' depends on '{dep}' which does not exist. "
                        f"Available ids: {sorted(all_ids)}"
                    )
        elif needs is not None:
            raise WorkflowValidationError(
                f"Step '{step_id}': 'needs' must be a list of step ids, got {type(needs).__name__}"
            )

    # Cycle detection using Kahn's algorithm
    in_degree: Dict[str, int] = {sid: 0 for sid in all_ids}
    adj: Dict[str, List[str]] = {sid: [] for sid in all_ids}

    for step_id, step in step_map.items():
        for dep in step.get("needs") or []:
            adj[dep].append(step_id)
            in_degree[step_id] = in_degree.get(step_id, 0) + 1

    queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
    visited = 0

    while queue:
        sid = queue.popleft()
        visited += 1
        for neighbor in adj.get(sid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(all_ids):
        # Find the cycle: look at nodes still with in_degree > 0
        cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
        raise WorkflowValidationError(
            f"Circular dependency detected involving steps: {cycle_nodes}"
        )

    return step_map


# ---------------------------------------------------------------------------
# Topological Sort (Kahn's)
# ---------------------------------------------------------------------------


def _topological_sort(step_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return step ids in topological order (deps before dependents)."""
    in_degree: Dict[str, int] = {sid: 0 for sid in step_map}
    adj: Dict[str, List[str]] = {sid: [] for sid in step_map}

    for step_id, step in step_map.items():
        for dep in step.get("needs") or []:
            adj[dep].append(step_id)
            in_degree[step_id] = in_degree.get(step_id, 0) + 1

    queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
    result = []

    while queue:
        sid = queue.popleft()
        result.append(sid)
        for neighbor in adj.get(sid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


# ---------------------------------------------------------------------------
# Context Assembly for Downstream Steps
# ---------------------------------------------------------------------------


def _build_step_context(
    step_id: str,
    step: Dict[str, Any],
    completed_results: Dict[str, Dict[str, Any]],
) -> str:
    """Assemble the context string for a step from its upstream results.

    Injects the summaries of all completed dependencies so the child agent
    has full context without needing to call context_read.
    """
    needs = step.get("needs") or []
    if not needs:
        # No dependencies — use the step's own context if provided
        return step.get("context") or ""

    parts: List[str] = []
    if step.get("context"):
        parts.append(step.get("context", ""))

    for dep_id in needs:
        dep_result = completed_results.get(dep_id)
        if dep_result is None:
            continue
        summary = dep_result.get("summary", "")
        if summary:
            parts.append(f"[Results from step '{dep_id}']:\n{summary}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Run Single Workflow Step
# ---------------------------------------------------------------------------


def _run_workflow_step(
    step_id: str,
    step: Dict[str, Any],
    combined_context: str,
    build_child_fn,
    run_child_fn,
    parent_agent,
    task_index: int,
    total_steps: int,
    max_iterations: int,
    creds: dict,
    toolsets: Optional[List[str]] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    role: str = "leaf",
) -> Dict[str, Any]:
    """Build and run a single workflow step, returning its result dict.

    Uses the same _build_child_agent and _run_single_child functions
    that delegate_task uses.
    """
    context = combined_context or step.get("context")

    try:
        child = build_child_fn(
            task_index=task_index,
            goal=step["goal"],
            context=context,
            toolsets=step.get("toolsets") or toolsets,
            model=creds.get("model"),
            max_iterations=max_iterations,
            task_count=total_steps,
            parent_agent=parent_agent,
            override_provider=creds.get("provider"),
            override_base_url=creds.get("base_url"),
            override_api_key=creds.get("api_key"),
            override_api_mode=creds.get("api_mode"),
            override_acp_command=step.get("acp_command") or acp_command or creds.get("command"),
            override_acp_args=step.get("acp_args") or acp_args or creds.get("args"),
            role=step.get("role") or role,
        )
        result = run_child_fn(
            task_index=task_index,
            goal=step["goal"],
            child=child,
            parent_agent=parent_agent,
        )
    except Exception as exc:
        logger.error("Workflow step '%s' failed to execute: %s", step_id, exc)
        return {
            "step_id": step_id,
            "status": "failed",
            "error": str(exc),
        }

    # Parse the result (run_child_fn returns a dict)
    if isinstance(result, dict):
        result["step_id"] = step_id
        return result

    # String result — wrap it
    return {
        "step_id": step_id,
        "status": "completed",
        "summary": str(result),
    }


# ---------------------------------------------------------------------------
# Main Workflow Runner
# ---------------------------------------------------------------------------


def run_workflow(
    *,
    workflow: List[Dict[str, Any]],
    synthesis: Optional[str] = None,
    parent_agent=None,
    toolsets: Optional[List[str]] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    role: str = "leaf",
    max_iterations: int,
    creds: dict,
    build_child_fn,
    run_child_fn,
    max_concurrent: int,
    agent_name: str = "workflow",
) -> str:
    """Execute a dependency-aware workflow.

    Parameters
    ----------
    workflow : list of dict
        Each dict: {id, goal, needs?, context?, toolsets?, role?, acp_command?, acp_args?}
    synthesis : str or None
        Optional synthesis goal — an aggregator step that receives all results.
    parent_agent : AIAgent
        The parent agent context.
    build_child_fn : callable
        _build_child_agent from delegate_tool.py
    run_child_fn : callable
        _run_single_child from delegate_tool.py
    max_concurrent : int
        Max parallel children (delegation.max_concurrent_children).
    creds : dict
        Resolved delegation credentials from _resolve_delegation_credentials.
    agent_name : str
        Name for logging/diagnostics.

    Returns
    -------
    str
        JSON string with {status, steps, synthesis_result?, workflow_id, total_duration_ms}
    """
    workflow_start = time.monotonic()
    step_results: Dict[str, Dict[str, Any]] = {}
    # Track which steps have been started to avoid duplicates when
    # multiple downstream steps share the same dependency
    started: Set[str] = set()
    errors: List[str] = []

    # 1. Validate
    logger.info("Workflow: validating %d step(s)", len(workflow))
    try:
        step_map = _validate_workflow(workflow)
    except WorkflowValidationError as exc:
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "steps": workflow,
        })

    # 2. Get topological order
    topo_order = _topological_sort(step_map)
    total = len(topo_order)
    logger.info("Workflow: topo order: %s", topo_order)

    # 3. Build dependency tracking
    deps_remaining: Dict[str, Set[str]] = {}
    for step_id, step in step_map.items():
        deps = set(step.get("needs") or [])
        deps_remaining[step_id] = deps

    # 4. Execute in waves
    step_index = 0
    pending_futures: Dict[Any, str] = {}  # future -> step_id

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all root steps (no deps) immediately
        for step_id in topo_order:
            if not deps_remaining[step_id]:
                step_index += 1
                context = _build_step_context(step_id, step_map[step_id], step_results)
                future = executor.submit(
                    _run_workflow_step,
                    step_id=step_id,
                    step=step_map[step_id],
                    combined_context=context,
                    build_child_fn=build_child_fn,
                    run_child_fn=run_child_fn,
                    parent_agent=parent_agent,
                    task_index=step_index,
                    total_steps=total,
                    max_iterations=max_iterations,
                    creds=creds,
                    toolsets=toolsets,
                    acp_command=acp_command,
                    acp_args=acp_args,
                    role=role,
                )
                pending_futures[future] = step_id
                started.add(step_id)
                logger.debug("Workflow: submitted root step '%s'", step_id)

        # Process futures as they complete
        while pending_futures:
            done_set = set()
            for future in list(pending_futures.keys()):
                if not future.done():
                    continue
                step_id = pending_futures[future]
                done_set.add(future)

                try:
                    result = future.result()
                    step_results[step_id] = result
                    status = result.get("status", "completed")
                    if status == "failed" or "error" in result:
                        err_msg = result.get("error", result.get("summary", "unknown error"))
                        errors.append(f"Step '{step_id}': {err_msg}")
                        logger.warning("Workflow step '%s' failed: %s", step_id, err_msg)

                    logger.info(
                        "Workflow step '%s' %s (%.1fs)",
                        step_id, status,
                        time.monotonic() - workflow_start,
                    )
                except Exception as exc:
                    errors.append(f"Step '{step_id}': {exc}")
                    step_results[step_id] = {
                        "step_id": step_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                    logger.error("Workflow step '%s' raised: %s", step_id, exc)

            # Remove done futures
            for f in done_set:
                del pending_futures[f]

            # Check for newly-ready steps
            for step_id in topo_order:
                if step_id in started:
                    continue
                # Are all deps satisfied?
                deps = deps_remaining[step_id]
                all_deps_done = all(
                    dep in step_results
                    for dep in deps
                )
                if all_deps_done:
                    step_index += 1
                    context = _build_step_context(step_id, step_map[step_id], step_results)
                    future = executor.submit(
                        _run_workflow_step,
                        step_id=step_id,
                        step=step_map[step_id],
                        combined_context=context,
                        build_child_fn=build_child_fn,
                        run_child_fn=run_child_fn,
                        parent_agent=parent_agent,
                        task_index=step_index,
                        total_steps=total,
                        max_iterations=max_iterations,
                        creds=creds,
                        toolsets=toolsets,
                        acp_command=acp_command,
                        acp_args=acp_args,
                        role=role,
                    )
                    pending_futures[future] = step_id
                    started.add(step_id)
                    logger.debug("Workflow: submitted step '%s' (deps satisfied)", step_id)

            # Brief sleep to avoid busy-wait
            if pending_futures:
                time.sleep(0.05)

    # 5. Build summary
    total_ms = int((time.monotonic() - workflow_start) * 1000)
    step_summaries = []
    for step_id in topo_order:
        result = step_results.get(step_id, {})
        step_summaries.append({
            "step_id": step_id,
            "status": result.get("status", "unknown"),
            "summary": result.get("summary", ""),
            "error": result.get("error"),
        })

    overall_status = "failed" if errors else "completed"
    workflow_id = f"wf-{int(time.time())}"

    response = {
        "status": overall_status,
        "workflow_id": workflow_id,
        "step_count": total,
        "steps": step_summaries,
        "errors": errors if errors else None,
        "total_duration_ms": total_ms,
    }

    # 6. Optional synthesis step
    if synthesis and overall_status != "failed":
        logger.info("Workflow: running synthesis step")
        synthesis_context_parts = []
        for step_id in topo_order:
            result = step_results.get(step_id, {})
            summary = result.get("summary", "")
            if summary:
                synthesis_context_parts.append(
                    f"=== Step '{step_id}' ===\n{summary}"
                )

        synthesis_context = (
            "Synthesize the following results from a multi-step workflow.\n\n"
            + "\n\n".join(synthesis_context_parts)
        )

        try:
            synth_child = build_child_fn(
                task_index=total + 1,
                goal=synthesis,
                context=synthesis_context,
                toolsets=toolsets,
                model=creds.get("model"),
                max_iterations=max_iterations,
                task_count=total + 1,
                parent_agent=parent_agent,
                override_provider=creds.get("provider"),
                override_base_url=creds.get("base_url"),
                override_api_key=creds.get("api_key"),
                override_api_mode=creds.get("api_mode"),
                override_acp_command=acp_command or creds.get("command"),
                override_acp_args=acp_args or creds.get("args"),
                role="leaf",
            )
            synth_result = run_child_fn(
                task_index=total + 1,
                goal=synthesis,
                child=synth_child,
                parent_agent=parent_agent,
            )
            if isinstance(synth_result, dict):
                response["synthesis"] = {
                    "summary": synth_result.get("summary", ""),
                }
            else:
                response["synthesis"] = {"summary": str(synth_result)}
        except Exception as exc:
            logger.error("Workflow synthesis step failed: %s", exc)
            response["synthesis"] = {"error": str(exc)}

    return json.dumps(response, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Synthesis-only runner (parallel batch + aggregation)
# ---------------------------------------------------------------------------


def run_synthesis(
    *,
    synthesis: str,
    tasks: List[Dict[str, Any]],
    parent_agent=None,
    toolsets: Optional[List[str]] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    role: str = "leaf",
    max_iterations: int,
    creds: dict,
    build_child_fn,
    run_child_fn,
    max_concurrent: int,
) -> str:
    """Run parallel tasks then aggregate via a synthesis step.

    This is the 'convoy mode' from Gas Town — parallel execution of
    independent tasks followed by a synthesis agent that combines results.
    """
    from tools.delegate_tool import _normalize_role

    start = time.monotonic()
    child_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            context = task.get("context")
            effective_role = _normalize_role(task.get("role") or role)
            future = executor.submit(
                _run_workflow_step,
                step_id=task.get("id", f"task_{i}"),
                step=task,
                combined_context=context or "",
                build_child_fn=build_child_fn,
                run_child_fn=run_child_fn,
                parent_agent=parent_agent,
                task_index=i,
                total_steps=len(tasks),
                max_iterations=max_iterations,
                creds=creds,
                toolsets=task.get("toolsets") or toolsets,
                acp_command=task.get("acp_command") or acp_command,
                acp_args=task.get("acp_args") or acp_args,
                role=effective_role,
            )
            futures[future] = task.get("id", f"task_{i}")

        for future in futures:
            try:
                result = future.result()
                child_results.append(result)
            except Exception as exc:
                task_id = futures[future]
                child_results.append({
                    "step_id": task_id,
                    "status": "failed",
                    "error": str(exc),
                })

    total_ms = int((time.monotonic() - start) * 1000)

    # Build synthesis context
    synth_parts = []
    for result in child_results:
        sid = result.get("step_id", "?")
        summary = result.get("summary", "")
        if summary:
            synth_parts.append(f"=== Task '{sid}' ===\n{summary}")

    if not synth_parts:
        return json.dumps({
            "status": "completed",
            "tasks": child_results,
            "total_duration_ms": total_ms,
        })

    synthesis_context = (
        "Synthesize the following results from parallel tasks.\n\n"
        + "\n\n".join(synth_parts)
    )

    try:
        synth_child = build_child_fn(
            task_index=len(tasks),
            goal=synthesis,
            context=synthesis_context,
            toolsets=toolsets,
            model=creds.get("model"),
            max_iterations=max_iterations,
            task_count=len(tasks) + 1,
            parent_agent=parent_agent,
            override_provider=creds.get("provider"),
            override_base_url=creds.get("base_url"),
            override_api_key=creds.get("api_key"),
            override_api_mode=creds.get("api_mode"),
            override_acp_command=acp_command or creds.get("command"),
            override_acp_args=acp_args or creds.get("args"),
            role="leaf",
        )
        synth_result = run_child_fn(
            task_index=len(tasks),
            goal=synthesis,
            child=synth_child,
            parent_agent=parent_agent,
        )
        synthesis_output = synth_result.get("summary", "") if isinstance(synth_result, dict) else str(synth_result)
    except Exception as exc:
        synthesis_output = f"<synthesis failed: {exc}>"

    return json.dumps({
        "status": "completed",
        "tasks": child_results,
        "synthesis": synthesis_output,
        "total_duration_ms": total_ms,
    }, ensure_ascii=False, default=str)
