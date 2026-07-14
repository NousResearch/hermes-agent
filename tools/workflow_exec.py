#!/usr/bin/env python3
"""
Visual Multi-Agent Workflow Builder - Execution Engine

This module implements parallel, topologically-sorted execution of workflow
graphs containing React Flow nodes and edges. Each node represents a task
delegated to a subagent via the delegate_task tool.
"""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Callable, Set, Tuple

from tools.delegate_tool import delegate_task
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approval-gate runtime registry — bridges HTTP endpoints to running executors
# ---------------------------------------------------------------------------
_RUNNING_EXECUTORS: Dict[str, "WorkflowExecutor"] = {}
_RUN_LOCK = threading.Lock()


def register_run(run_id: str, executor: "WorkflowExecutor") -> None:
    with _RUN_LOCK:
        _RUNNING_EXECUTORS[run_id] = executor


def unregister_run(run_id: str) -> None:
    with _RUN_LOCK:
        _RUNNING_EXECUTORS.pop(run_id, None)


def get_run(run_id: str) -> Optional["WorkflowExecutor"]:
    with _RUN_LOCK:
        return _RUNNING_EXECUTORS.get(run_id)


class WorkflowExecutor:
    """
    Manages the parallel execution of a workflow graph with topological sorting,
    cycle detection, error propagation, and progress callbacks.
    """

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        parent_agent=None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_workers: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        self.nodes = {str(n["id"]): n for n in nodes}
        self.edges = edges
        self.parent_agent = parent_agent
        self.progress_callback = progress_callback
        self.max_workers = max_workers or min(32, len(nodes) + 1)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        # Build adjacency lists and compute in-degrees
        # For an edge source -> target: target depends on source.
        self.adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        self.rev_adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        self.in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}

        for edge in edges:
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            if src in self.nodes and tgt in self.nodes:
                self.adj[src].append(tgt)
                self.rev_adj[tgt].append(src)
                self.in_degree[tgt] += 1

        self.states: Dict[str, str] = {nid: "pending" for nid in self.nodes}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.running_count = 0
        self.executor: Optional[ThreadPoolExecutor] = None
        self.start_times: Dict[str, float] = {}
        # Approval-gate state: node_id -> threading.Event + decision
        self.gate_events: Dict[str, threading.Event] = {}
        self.gate_decisions: Dict[str, str] = {}
        # Unique run identifier for gate-resolve routing
        self._run_id = run_id or f"local-{id(self):x}"

    def has_cycle(self) -> bool:
        """
        Detects if the dependency graph contains any cycles using Kahn's algorithm.
        """
        in_deg = dict(self.in_degree)
        queue = [nid for nid, deg in in_deg.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in self.adj[curr]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)

        return visited_count < len(self.nodes)

    def _notify_status(self, node_id: str, status: str, result: Optional[Dict[str, Any]] = None):
        """
        Relays status transition progress events via callbacks and console loggers.
        """
        node = self.nodes.get(node_id, {})
        goal = node.get("data", {}).get("goal", "")
        event = {
            "event": "node_status_changed",
            "node_id": node_id,
            "goal": goal,
            "status": status,
            "timestamp": time.time(),
            "result": result,
        }

        # 1. Custom callback
        if self.progress_callback:
            try:
                self.progress_callback(event)
            except Exception as e:
                logger.debug("Custom progress callback failed: %s", e)

        # 2. Parent agent's tool progress callback (for gateway and UI integration)
        if self.parent_agent and hasattr(self.parent_agent, "tool_progress_callback") and self.parent_agent.tool_progress_callback:
            try:
                self.parent_agent.tool_progress_callback(
                    "workflow.node_status",
                    tool_name="execute_workflow",
                    preview=f"Node {node_id} ({goal[:30]}...) transitioned to {status}",
                    args=None,
                    node_id=node_id,
                    status=status,
                    result=result,
                )
            except Exception as e:
                logger.debug("Parent agent progress callback failed: %s", e)

        # 3. Print to spinner or console
        spinner = getattr(self.parent_agent, "_delegate_spinner", None)
        icon = {
            "running": "⏳",
            "completed": "✓",
            "failed": "✗",
            "aborted": "⚠",
        }.get(status, "•")

        elapsed_str = ""
        if status in ("completed", "failed") and node_id in self.start_times:
            elapsed = round(time.monotonic() - self.start_times[node_id], 2)
            elapsed_str = f" ({elapsed}s)"

        msg = f"{icon} [Node {node_id}] {status.upper()}: {goal[:50]}...{elapsed_str}"
        if spinner:
            try:
                spinner.print_above(msg)
            except Exception:
                print(f"  {msg}")
        else:
            print(f"  {msg}")

    def _abort_downstream(self, failed_node_id: str):
        """
        Transitively marks all downstream nodes depending on a failed node as aborted.
        """
        queue = list(self.adj[failed_node_id])
        while queue:
            curr = queue.pop(0)
            if self.states[curr] == "pending":
                self.states[curr] = "aborted"
                self.results[curr] = {
                    "status": "aborted",
                    "error": f"Aborted due to failure of prerequisite node '{failed_node_id}'."
                }
                self._notify_status(curr, "aborted", result=self.results[curr])
                # Enqueue children of this aborted node
                for child_id in self.adj[curr]:
                    queue.append(child_id)

    def _await_gate_decision(self, node_id: str, data: dict) -> str:
        """
        Blocks until the approval resolver returns a decision.
        Default: waits on a per-node threading.Event (set by resolve_gate).
        """
        event = threading.Event()
        self.gate_events[node_id] = event
        try:
            prompt = data.get("prompt") or data.get("label") or f"Gate {node_id}"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Gate %s waiting for decision: %s", node_id, prompt)
            event.wait()
            return self.gate_decisions.get(node_id, "denied")
        finally:
            self.gate_events.pop(node_id, None)

    def resolve_gate(self, node_id: str, decision: str) -> bool:
        """
        Wake a blocked approval gate. Returns True if a gate was waiting.
        Called by the HTTP endpoint after the user clicks approve/deny.
        """
        event = self.gate_events.get(node_id)
        if event is None:
            return False
        self.gate_decisions[node_id] = decision if decision in ("approved", "denied") else "denied"
        event.set()
        return True

    def _run_node(self, node_id: str):
        """
        Executes a single node task. Invoked in a thread pool worker.
        """
        node = self.nodes[node_id]

        with self.lock:
            if self.states[node_id] != "pending":
                return
            self.states[node_id] = "running"
            self.running_count += 1
            self.start_times[node_id] = time.monotonic()
            self._notify_status(node_id, "running")

        node_type = node.get("type", "agent")

        # Synthetic control nodes (start/end) — pure plumbing, never submitted.
        if node_type in ("start", "end"):
            nodes_to_submit = []
            with self.lock:
                self.states[node_id] = "completed"
                self.results[node_id] = {"status": "completed"}
                self.running_count -= 1
                self._notify_status(node_id, "completed", result=self.results[node_id])
                for child_id in self.adj[node_id]:
                    if self.states[child_id] == "pending":
                        self.in_degree[child_id] -= 1
                        if self.in_degree[child_id] == 0:
                            nodes_to_submit.append(child_id)
                # We handle this below — skip the rest of the method
            if self.executor:
                for next_id in nodes_to_submit:
                    self.executor.submit(self._run_node, next_id)
            return

        # Approval gate — blocks until resolved by the user
        if node_type == "gate":
            with self.lock:
                self.states[node_id] = "pending"
                self._notify_status(node_id, "pending")
            decision = self._await_gate_decision(node_id, node.get("data", {}))
            with self.lock:
                self.running_count -= 1
                if decision == "denied":
                    self.states[node_id] = "failed"
                    self.results[node_id] = {
                        "status": "failed",
                        "error": "Approval denied by user.",
                    }
                    self._notify_status(node_id, "failed", result=self.results[node_id])
                    self._abort_downstream(node_id)
                    if self.running_count == 0 and not any(s == "running" for s in self.states.values()):
                        self.condition.notify_all()
                    return
                # approved:
                self.states[node_id] = "completed"
                self.results[node_id] = {"status": "completed"}
                self._notify_status(node_id, "completed", result=self.results[node_id])
                nodes_to_submit = []
                for child_id in self.adj[node_id]:
                    if self.states[child_id] == "pending":
                        self.in_degree[child_id] -= 1
                        if self.in_degree[child_id] == 0:
                            nodes_to_submit.append(child_id)
                if self.running_count == 0 and not any(s == "running" for s in self.states.values()):
                    self.condition.notify_all()
            for next_id in nodes_to_submit:
                if self.executor:
                    self.executor.submit(self._run_node, next_id)
            return

        # Unknown node type — fail fast rather than silently complete
        if node_type not in ("agent",):
            with self.lock:
                self.states[node_id] = "failed"
                self.results[node_id] = {"status": "failed", "error": f"Unknown node type '{node_type}'."}
                self.running_count -= 1
                self._notify_status(node_id, "failed", result=self.results[node_id])
                self._abort_downstream(node_id)
                if self.running_count == 0 and not any(s == "running" for s in self.states.values()):
                    self.condition.notify_all()
            return

        # Agent node — delegate to subagent
        data = node.get("data", {})
        goal = data.get("goal") or data.get("prompt")
        context = data.get("context")
        role = data.get("role")

        status = "failed"
        result_data = {}

        try:
            res_str = delegate_task(
                goal=goal,
                context=context,
                role=role,
                parent_agent=self.parent_agent,
            )
            res = json.loads(res_str)

            if "error" in res:
                status = "failed"
                result_data = {"status": "failed", "error": res["error"]}
            elif "results" in res and res["results"]:
                task_res = res["results"][0]
                task_status = task_res.get("status")
                if task_status == "completed":
                    status = "completed"
                    result_data = {
                        "status": "completed",
                        "summary": task_res.get("summary", ""),
                        "duration_seconds": task_res.get("duration_seconds", 0),
                        "api_calls": task_res.get("api_calls", 0),
                    }
                else:
                    status = "failed"
                    result_data = {
                        "status": task_status or "failed",
                        "error": task_res.get("error", "Subagent did not complete successfully."),
                        "summary": task_res.get("summary"),
                    }
            else:
                status = "failed"
                result_data = {"status": "failed", "error": "Invalid response format from delegate_task."}
        except Exception as e:
            status = "failed"
            result_data = {"status": "failed", "error": str(e)}

        nodes_to_submit = []

        with self.lock:
            self.states[node_id] = status
            self.results[node_id] = result_data
            self.running_count -= 1
            self._notify_status(node_id, status, result=result_data)

            if status == "completed":
                # Decrement in-degree of all targets
                for child_id in self.adj[node_id]:
                    if self.states[child_id] == "pending":
                        self.in_degree[child_id] -= 1
                        if self.in_degree[child_id] == 0:
                            nodes_to_submit.append(child_id)
            else:
                # Node failed, abort all downstream dependencies
                self._abort_downstream(node_id)

            # Check if all nodes are in terminal state
            if self.running_count == 0 and not any(s == "running" for s in self.states.values()):
                self.condition.notify_all()

        # Submit next ready nodes outside the lock to avoid lock contention
        for next_id in nodes_to_submit:
            if self.executor:
                self.executor.submit(self._run_node, next_id)

    def _is_finished(self) -> bool:
        """
        Returns True if all nodes have reached a terminal state.
        """
        terminal_states = {"completed", "failed", "aborted"}
        return all(self.states[nid] in terminal_states for nid in self.nodes)

    def _interrupt_all(self):
        """
        Aborts all pending nodes and logs the interrupt.
        """
        for nid in self.nodes:
            if self.states[nid] == "pending":
                self.states[nid] = "aborted"
                self.results[nid] = {"status": "aborted", "error": "Workflow execution was interrupted."}
                self._notify_status(nid, "aborted", result=self.results[nid])

    def execute(self) -> Dict[str, Any]:
        """
        Starts the topologically sorted parallel execution of workflow tasks.
        """
        if self.has_cycle():
            raise ValueError("Cyclic dependency detected in workflow graph.")

        start_nodes = [nid for nid, deg in self.in_degree.items() if deg == 0]
        if not start_nodes and self.nodes:
            raise ValueError("No start nodes found (all nodes have dependencies, possible cycle).")

        overall_start = time.monotonic()

        # Register this executor for gate-resolve routing from the HTTP layer
        register_run(self._run_id, self)
        try:
            # Execute using a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                self.executor = executor
                for node_id in start_nodes:
                    executor.submit(self._run_node, node_id)

                # Wait until all tasks are complete, while checking for interrupts
                with self.lock:
                    while not self._is_finished():
                        if self.parent_agent and getattr(self.parent_agent, "_interrupt_requested", False):
                            self._interrupt_all()
                            break
                        # Wait on condition with a timeout to check for interrupts periodically
                        self.condition.wait(timeout=0.2)
        finally:
            unregister_run(self._run_id)

        duration = round(time.monotonic() - overall_start, 2)
        success = all(self.states[nid] == "completed" for nid in self.nodes)

        return {
            "status": "completed" if success else "failed",
            "results": self.results,
            "duration_seconds": duration,
        }


def parse_workflow_graph(graph_input: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses and extracts nodes and edges from string or dictionary input.
    """
    if not graph_input:
        return [], []

    if isinstance(graph_input, str):
        try:
            parsed = json.loads(graph_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for workflow graph: {str(e)}")
    else:
        parsed = graph_input

    if isinstance(parsed, dict):
        nodes = parsed.get("nodes", [])
        edges = parsed.get("edges", [])
        return nodes, edges
    elif isinstance(parsed, list):
        # Fallback: Treat list of nodes directly with no edges
        return parsed, []
    else:
        raise ValueError(f"Workflow graph must be a dict or list, got {type(parsed).__name__}")


def validate_workflow_structure(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Optional[str]:
    """
    Validates the structure of nodes and edges, checking for required keys and valid references.
    """
    if not isinstance(nodes, list):
        return "nodes must be a list."
    if not isinstance(edges, list):
        return "edges must be a list."

    node_ids = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            return f"Node at index {idx} must be an object/dict."
        nid = node.get("id")
        if nid is None or str(nid).strip() == "":
            return f"Node at index {idx} is missing a valid 'id'."
        node_ids.add(str(nid))

        node_type = node.get("type", "agent")
        if node_type == "agent":
            data = node.get("data")
            if not isinstance(data, dict):
                return f"Node '{nid}' is missing a valid 'data' object."
            goal = data.get("goal") or data.get("prompt")
            if not goal or not isinstance(goal, str) or not goal.strip():
                return f"Node '{nid}' must have a non-empty string 'goal' or 'prompt' in data."

    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            return f"Edge at index {idx} must be an object/dict."
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            return f"Edge at index {idx} is missing 'source' or 'target'."
        if str(src) not in node_ids:
            return f"Edge at index {idx} references non-existent source node '{src}'."
        if str(tgt) not in node_ids:
            return f"Edge at index {idx} references non-existent target node '{tgt}'."

    return None


def execute_workflow_tool(graph: Any, parent_agent=None) -> str:
    """
    Main tool handler for execute_workflow.
    """
    if parent_agent is None:
        return tool_error("execute_workflow requires a parent agent context.")

    try:
        nodes, edges = parse_workflow_graph(graph)
    except Exception as e:
        return tool_error(f"Failed to parse graph: {str(e)}")

    validation_err = validate_workflow_structure(nodes, edges)
    if validation_err:
        return tool_error(f"Validation failed: {validation_err}")

    try:
        executor = WorkflowExecutor(nodes, edges, parent_agent=parent_agent)
        if executor.has_cycle():
            return tool_error("Validation failed: Cyclic dependency (loop) detected in workflow graph.")

        result = executor.execute()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return tool_error(f"Workflow execution failed: {str(e)}")


def execute_workflow_graph(graph: Any, log_callback: Callable[[str], None], run_id: Optional[str] = None) -> bool:
    """
    Main entry point for executing a workflow graph from a thread with log feedback.
    """
    try:
        nodes, edges = parse_workflow_graph(graph)
    except Exception as e:
        log_callback(f"[Error] Failed to parse workflow graph: {e}")
        return False

    validation_err = validate_workflow_structure(nodes, edges)
    if validation_err:
        log_callback(f"[Error] Validation failed: {validation_err}")
        return False

    # Initialize a parent agent context
    try:
        from run_agent import AIAgent
        parent_agent = AIAgent(
            skip_memory=True,
            skip_context_files=True,
            quiet_mode=True,
        )
    except Exception as e:
        log_callback(f"[Error] Failed to initialize agent context: {e}")
        return False

    def progress_callback(event):
        status = event.get("status")
        node_id = event.get("node_id")
        goal = event.get("goal", "")
        log_callback(f"[{status.upper()}] Node {node_id}: {goal}")
        result = event.get("result", {})
        if status == "completed":
            summary = result.get("summary", "")
            if summary:
                log_callback(f"  Result Summary: {summary}")
        elif status == "failed":
            error = result.get("error", "Unknown error")
            log_callback(f"  Failure Reason: {error}")
        elif status == "aborted":
            error = result.get("error", "Aborted due to pre-requisite failure")
            log_callback(f"  Aborted: {error}")

    try:
        executor = WorkflowExecutor(
            nodes,
            edges,
            parent_agent=parent_agent,
            progress_callback=progress_callback,
            run_id=run_id,
        )
        if executor.has_cycle():
            log_callback("[Error] Cyclic dependency (loop) detected in workflow graph.")
            return False

        result = executor.execute()
        return result.get("status") == "completed"
    except Exception as e:
        log_callback(f"[Error] Execution failed: {e}")
        return False
    finally:
        try:
            parent_agent.close()
        except Exception:
            pass


def check_requirements() -> bool:
    return True


# --- Schema and Registry ---

EXECUTE_WORKFLOW_SCHEMA = {
    "name": "execute_workflow",
    "description": (
        "Execute a workflow of multi-agent tasks in parallel with topological dependency ordering. "
        "The workflow is described as a React Flow graph JSON containing nodes (each representing an agent task) "
        "and edges representing dependencies (source depends on target / source executes before target). "
        "Tasks are run via subagent delegation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "graph": {
                "type": "string",
                "description": (
                    "JSON string representing the workflow graph. "
                    "Must contain 'nodes' (each with 'id' and 'data' containing 'goal') and "
                    "'edges' (each with 'source' and 'target' references)."
                ),
            }
        },
        "required": ["graph"],
    },
}

registry.register(
    name="execute_workflow",
    toolset="delegation",
    schema=EXECUTE_WORKFLOW_SCHEMA,
    handler=lambda args, **kw: execute_workflow_tool(
        graph=args.get("graph", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_requirements,
    emoji="⛓️",
)
