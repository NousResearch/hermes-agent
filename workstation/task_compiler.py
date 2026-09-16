"""Compile decided, structured work into the existing durable execution path.

No model is called here. Tool execution remains in the agent's scoped dispatcher.
Natural-language ambiguity stays with the planner; this layer never guesses writes.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
import hashlib
import json
import time
import re
from typing import Any, Callable

from agent.tool_guardrails import classify_tool_failure
from workstation.artifacts import ArtifactStore
from workstation.batch_runner import DurableBatchRunner
from workstation.durable_tasks import DurableTaskStore, WorkItem
from workstation.routing import ConstraintViolation, require_allowed_route
from workstation.reference_plane import ReadCache, blob_references, content_reference
from workstation.tool_verbosity import VerbosityLevel, normalize_verbosity


class WorkClass(str, Enum):
    INTERACTIVE_REASONING = "INTERACTIVE_REASONING"
    DETERMINISTIC_SINGLE = "DETERMINISTIC_SINGLE"
    DETERMINISTIC_BATCH = "DETERMINISTIC_BATCH"
    BROWSER_TRANSACTION = "BROWSER_TRANSACTION"
    PROMPT_QUEUE = "PROMPT_QUEUE"
    EXCEPTION_REQUIRES_REASONING = "EXCEPTION_REQUIRES_REASONING"


_dispatch_context: ContextVar[Any] = ContextVar("workstation_work_dispatch", default=None)
_constraints: ContextVar[Any] = ContextVar("workstation_work_constraints", default=None)
_execution_active: ContextVar[bool] = ContextVar("workstation_durable_active", default=False)
READ_TOOLS = frozenset({"read_file", "search_files", "browser_snapshot"})


def batch_intent(prompt: Any) -> bool:
    """Conservative structural signal; only requires compilation, never infers writes."""
    if not isinstance(prompt, str):
        return False
    lower = prompt.lower()
    operation = re.search(r"\b(cri(?:ar|e)|ger(?:ar|e)|process(?:ar|e)|atualiz(?:ar|e)|execut(?:ar|e)|create|generate|process|update|execute|apply)\b", lower)
    count = re.search(r"\b([2-9]|[1-9][0-9]+)\s+(?:registros|arquivos|cartões|cartoes|imagens|itens|entradas|entidades|prompts|records|files|cards|images|items|rows|entities)\b", lower)
    structured = False
    for candidate in [prompt] + re.findall(r"```(?:json)?\s*(.*?)```", prompt, re.S):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        entries = payload.get("items", payload.get("records", [])) if isinstance(payload, dict) else payload
        if isinstance(entries, list) and len(entries) > 1 and all(isinstance(i, dict) for i in entries):
            structured = all(set(i) == set(entries[0]) for i in entries)
            if isinstance(payload, dict) and payload.get("steps"):
                return True
    return bool(operation and (count or structured))


def user_constraints(prompt: Any) -> dict:
    """Read explicit structured route constraints from the user, never tool text."""
    if not isinstance(prompt, str):
        return {}
    candidates = [prompt] + re.findall(r"```(?:json)?\s*(.*?)```", prompt, re.S)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            constraints = parsed.get("constraints", parsed)
            if isinstance(constraints, dict):
                return {k: constraints[k] for k in ("allowed_routes", "forbidden_routes") if k in constraints}
    lower = prompt.lower()
    constraints = {}
    forbidden = []
    for route, phrase in (("openai_api", r"openai[_ ]api|api[^\n]{0,20}openai"),
                          ("browserclaw", r"browserclaw")):
        if (re.search(r"(?:não use|nao use|do not use|never use)[^\n]{0,60}(?:" + phrase + ")", lower)
                or re.search(re.escape(route) + r"\s*(?:[:=]\s*)?(?:forbidden|proibido)", lower)):
            forbidden.append(route)
    if forbidden:
        constraints["forbidden_routes"] = forbidden
    if re.search(r"(?:somente|apenas|only)\s+(?:o |a |the )?(?:navegador|browser|chatgpt web)", lower):
        constraints["allowed_routes"] = ["native_browser"]
    return constraints


def merge_constraints(user: dict, compiled: dict) -> dict:
    combined = {k: compiled[k] for k in ("allowed_routes", "forbidden_routes") if k in compiled}
    combined["forbidden_routes"] = sorted(set(user.get("forbidden_routes", [])) | set(compiled.get("forbidden_routes", [])))
    if "allowed_routes" in user:
        if "allowed_routes" not in compiled:
            combined["allowed_routes"] = user["allowed_routes"]
        else:
            combined["allowed_routes"] = sorted({
                a if a == b or a.startswith(b + ".") else b
                for a in user["allowed_routes"] for b in compiled["allowed_routes"]
                if a == b or a.startswith(b + ".") or b.startswith(a + ".")})
    return combined


def requires_compilation(agent: Any, calls: list) -> bool:
    if not getattr(agent, "_work_batch_candidate", False) or "work_execute" not in agent.valid_tool_names:
        return False
    allowed = READ_TOOLS | {"tool_search", "tool_describe", "clarify", "work_execute"}
    for call in calls:
        name = call.function.name
        if name == "tool_call":
            try:
                name = json.loads(call.function.arguments).get("name", "")
            except (ValueError, AttributeError):
                return True
        if name not in allowed:
            return True
    return False


@contextmanager
def execution_context(dispatch: Callable, session_id: str, progress: Callable | None = None,
                      constraints: dict | None = None, provider_usage: dict | None = None):
    token = _dispatch_context.set((dispatch, session_id, {}, [], progress, provider_usage))
    constraint_token = _constraints.set(constraints or {})
    try:
        yield
    finally:
        _constraints.reset(constraint_token)
        _dispatch_context.reset(token)


def active_constraints() -> dict:
    return _constraints.get() or {}


def durable_execution_active() -> bool:
    return _execution_active.get()


def capture_raw_result(call_id: str, raw: Any) -> None:
    context = _dispatch_context.get()
    if context is not None and durable_execution_active():
        context[2][call_id] = raw


def take_raw_result(call_id: str, fallback: Any) -> Any:
    context = _dispatch_context.get()
    return context[2].pop(call_id, fallback) if context is not None else fallback


def operational_references() -> list:
    context = _dispatch_context.get()
    return list(context[3]) if context else []


def classify(request: dict) -> WorkClass:
    if request.get("exception"):
        return WorkClass.EXCEPTION_REQUIRES_REASONING
    if not request.get("steps") or not isinstance(request.get("items"), list):
        return WorkClass.INTERACTIVE_REASONING
    if request.get("kind") == "prompt_queue":
        return WorkClass.PROMPT_QUEUE
    if request.get("kind") == "browser_transaction" and any(
        str(s.get("tool", "")).startswith("browser_") for s in request["steps"]
    ):
        return WorkClass.BROWSER_TRANSACTION
    if all(str(s.get("tool", "")).startswith("browser_") for s in request["steps"]):
        return WorkClass.BROWSER_TRANSACTION
    return WorkClass.DETERMINISTIC_BATCH if len(request["items"]) > 1 else WorkClass.DETERMINISTIC_SINGLE


def _bind(value: Any, item: dict) -> Any:
    if isinstance(value, str) and value.startswith("$item."):
        result: Any = item
        for field in value[6:].split("."):
            result = result[field]
        return result
    if isinstance(value, dict):
        return {k: _bind(v, item) for k, v in value.items()}
    if isinstance(value, list):
        return [_bind(v, item) for v in value]
    return value


def _validate(output: Any, expected: dict) -> bool:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return False
    for path, value in expected.items():
        current = output
        try:
            for key in path.split("."):
                current = current[key]
        except (KeyError, TypeError):
            return False
        if current != value:
            return False
    return True


class TaskCompiler:
    def __init__(self, store: DurableTaskStore | None = None, artifacts: ArtifactStore | None = None):
        self.store = store or DurableTaskStore()
        self.artifacts = artifacts or ArtifactStore()

    def execute(self, request: dict, *, task_id: str, session_id: str, dispatch: Callable,
                progress: Callable | None = None, provider_usage: dict | None = None) -> dict:
        if request.get("items_ref"):
            if "items" in request:
                raise ValueError("Use items or items_ref, not both")
            dataset = self.artifacts.read_json(request["items_ref"])
            if isinstance(dataset, dict):
                dataset = dataset.get("items")
            if not isinstance(dataset, list) or not all(isinstance(i, dict) for i in dataset):
                raise ValueError("items_ref must resolve to structured records")
            request = {**request, "items": dataset, "source_items_ref": request["items_ref"]}
            request.pop("items_ref")
        kind = classify(request)
        if kind in {WorkClass.INTERACTIVE_REASONING, WorkClass.EXCEPTION_REQUIRES_REASONING}:
            raise ValueError("Work requires a decided operation, structured items and verifiable steps")
        # Scope work identity to the owning session and a planner-supplied operation key.
        key = str(request.get("operation_key") or "")
        if not key or not session_id:
            raise ValueError("operation_key and owning session are required")
        identity = hashlib.sha256(f"{session_id}:{task_id}:{key}".encode()).hexdigest()
        durable_id = f"work_{identity}"
        constraints = merge_constraints(active_constraints(), request.get("constraints", {}))
        request = {**request, "constraints": constraints}
        steps = request["steps"]
        if kind == WorkClass.PROMPT_QUEUE and not any(s.get("wait") for s in steps):
            raise ValueError("Prompt queue requires a bounded completion wait before advancing")
        if not steps or len(steps) > 64 or len(request["items"]) > 10000:
            raise ValueError("Work exceeds bounded plan limits")
        for step in steps:
            tool = step["tool"]
            if tool == "work_execute":
                raise ValueError("Nested durable execution is prohibited")
            if tool in {"delegate_task", "clarify", "tool_call"}:
                raise ValueError("Cognitive/indirect steps must be resolved before compiling work")
            route = "native_browser" if tool.startswith("browser_") else f"tool.{tool}"
            if (constraints.get("forbidden_routes") and tool not in READ_TOOLS
                    and not tool.startswith("browser_")
                    and any(not r.startswith(("tool.", "native_browser")) for r in constraints["forbidden_routes"])):
                raise ConstraintViolation(f"Cannot prove provider route for constrained tool: {tool}")
            require_allowed_route(route, constraints)
            if tool not in READ_TOOLS and not step.get("expect"):
                raise ValueError(f"Mutation {tool} requires an explicit result verifier")
            if step.get("wait") and (tool not in READ_TOOLS or not step.get("expect")):
                raise ValueError("Completion waits require a read-only probe and explicit verifier")
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        plan = self.store.get_plan(durable_id)
        if plan and (plan.session_id != session_id or plan.metadata.get("fingerprint") != fingerprint):
            raise ValueError("Operation identity conflicts with the persisted plan")
        objective = self.artifacts.store(durable_id, "objective.json", request)
        metadata = {"classification": kind.value, "constraints": constraints,
                    "objective_ref": objective.ref, "fingerprint": fingerprint,
                    "browser_task_id": task_id, "verbosity": normalize_verbosity(request.get("verbosity")).value}
        metrics = {"tool_calls": 0, "LLM_interventions": 0, "artifact_bytes": 0,
                   "inline_context_bytes": 0, "work_items_completed": 0, "replans": 0,
                   "cache_hits": 0, "cache_misses": 0, "no_progress_calls": 0}
        metrics.update({"tool_input_bytes": 0, "tool_output_bytes": 0, "latency_ms": 0,
                        "compactions": 0, "state_transitions": 0, "duplicate_calls_blocked": 0,
                        "usage_status": "unknown"})
        metrics_uri = f"artifact://tasks/{durable_id}/metrics.json"
        if self.artifacts.resolve_ref(metrics_uri):
            metrics.update(self.artifacts.read_json(metrics_uri))
        for usage_field in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens",
                            "tokens_per_successful_state_transition", "token_usage_scope"):
            metrics.pop(usage_field, None)
        metrics["usage_status"] = "unknown"
        prior_transitions = metrics["state_transitions"]
        if plan:
            metrics["duplicate_calls_blocked"] += sum(
                i.status.value == "completed" for i in self.store.get_work_items(plan.id)) * len(steps)

        def persist_metrics():
            self.artifacts.store(durable_id, "metrics.json", metrics)
        reads = ReadCache(self.artifacts, durable_id)

        def worker(payload: dict, item: WorkItem) -> dict:
            results = []
            for index, step in enumerate(steps):
                # Persist each step before advancing; restart never replays a committed step.
                current = self.store.get_item(item.id)
                committed = current.checkpoints.get(f"step_{index}_meta", {})
                if committed.get("result_ref"):
                    metrics["duplicate_calls_blocked"] += 1
                    results.append({"artifact_ref": committed["result_ref"], "verified": True})
                    continue
                if current.checkpoints.get(f"step_{index}_dispatch") and step["tool"] not in READ_TOOLS:
                    return {"valid": False, "code": "uncertain_mutation_requires_review", "results": results}
                args = _bind(step.get("args", {}), payload)
                self.store.update_item_checkpoint(item.id, f"step_{index}_dispatch")
                metrics["tool_calls"] += 1
                metrics["tool_input_bytes"] += len(json.dumps(args).encode())
                started = time.monotonic()
                try:
                    raw = dispatch(step["tool"], args, task_id, f"{item.id}_{index}")
                    wait = step.get("wait", {})
                    deadline = started + min(300, max(0, float(wait.get("timeout_seconds", 30))))
                    max_polls = min(100, max(1, int(wait.get("max_polls", 20))))
                    polls = 1
                    while wait and not _validate(raw, _bind(step["expect"], payload)):
                        if time.monotonic() >= deadline or polls >= max_polls:
                            break
                        # Intermediate waiting observations belong to the Data Plane.
                        content_reference(self.artifacts, durable_id, blob_references(self.artifacts, durable_id, raw))
                        time.sleep(min(1, max(0, float(wait.get("interval_seconds", 0.2)))))
                        metrics["tool_calls"] += 1
                        raw = dispatch(step["tool"], args, task_id, f"{item.id}_{index}_poll_{polls}")
                        polls += 1
                except InterruptedError:
                    raise
                except Exception:
                    if step["tool"] in READ_TOOLS:
                        raise
                    return {"valid": False, "code": "mutation_failed_requires_review", "results": results}
                finally:
                    metrics["latency_ms"] += int((time.monotonic() - started) * 1000)
                    persist_metrics()
                output_bytes = len((raw if isinstance(raw, str) else json.dumps(raw)).encode("utf-8", "surrogatepass"))
                normalized = blob_references(self.artifacts, durable_id, raw)
                try:
                    projection = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(projection, dict) and projection.get("cache_hit") and projection.get("status") == "unchanged":
                        raw = self.artifacts.read_json(projection["artifact_ref"])
                except (ValueError, FileNotFoundError):
                    pass
                ref = reads.project(args, normalized) if step["tool"] == "read_file" else content_reference(
                    self.artifacts, durable_id, normalized)
                metrics["artifact_bytes"] += ref["size_bytes"] if not ref["cache_hit"] else 0
                metrics["cache_hits" if ref["cache_hit"] else "cache_misses"] += 1
                metrics["tool_output_bytes"] += output_bytes
                text = raw if isinstance(raw, str) else json.dumps(raw)
                failed, _ = classify_tool_failure(step["tool"], text)
                verified = not failed and (not step.get("expect") or _validate(raw, _bind(step["expect"], payload)))
                results.append({"artifact_ref": ref["artifact_ref"], "verified": verified})
                if not verified:
                    metrics["no_progress_calls"] += 1
                    return {"valid": False, "results": results, "code": "unexpected_state"}
                self.store.update_item_checkpoint(item.id, f"step_{index}", metadata={"result_ref": ref["artifact_ref"]})
                metrics["state_transitions"] += 1
                persist_metrics()
                if progress is not None:
                    progress()
            return {"valid": True, "results": results}

        token = _constraints.set(constraints)
        execution_token = _execution_active.set(True)
        try:
            runner = DurableBatchRunner(durable_id, task_store=self.store, artifact_store=self.artifacts,
                                        max_retries=2, backoff_seconds=0)
            summary = runner.execute_batch(request.get("title", key), request["items"],
                worker_fn=worker, validator_fn=lambda raw, _: {
                    "valid": raw.get("valid") is True, "reason": raw.get("code", ""),
                    "expected_delta": "verified_step", "actual_delta": raw.get("valid") is True},
                session_id=session_id, metadata=metadata,
                stop_on_exception=kind in {WorkClass.BROWSER_TRANSACTION, WorkClass.PROMPT_QUEUE})
        finally:
            _execution_active.reset(execution_token)
            _constraints.reset(token)
        envelope = summary.to_dict()
        envelope["results_ref"] = summary.summary_artifact_ref
        envelope["ledger"] = self.store.operational_ledger(durable_id)
        metrics["work_items_completed"] = envelope["completed"]
        metrics["replans"] = int(envelope["needs_reasoning"] > 0)
        metrics["tool_calls_per_state_transition"] = metrics["tool_calls"] / max(1, metrics["state_transitions"])
        metrics["artifact_bytes"] = sum(a.size_bytes for a in self.artifacts.list_artifacts(durable_id)
                                         if a.name != "metrics.json")
        if provider_usage is not None:
            metrics.update({"usage_status": "reported", "token_usage_scope": "compile_request",
                            "input_tokens": provider_usage.get("input_tokens"),
                            "cached_input_tokens": provider_usage.get("cache_read_tokens"),
                            "output_tokens": provider_usage.get("output_tokens"),
                            "reasoning_tokens": provider_usage.get("reasoning_tokens")})
            transitions = metrics["state_transitions"] - prior_transitions
            if transitions and provider_usage.get("total_tokens") is not None:
                metrics["tokens_per_successful_state_transition"] = provider_usage["total_tokens"] / transitions
        envelope["metrics"] = metrics
        if normalize_verbosity(request.get("verbosity")) == VerbosityLevel.FULL:
            manifest = self.artifacts.read_json(summary.summary_artifact_ref)
            envelope["full_results"] = [self.artifacts.read_json(i["result_ref"])
                                        for i in manifest["results"] if i["result_ref"]]
        for _ in range(3):
            metrics["inline_context_bytes"] = len(json.dumps(envelope).encode())
            metrics["bytes_avoided_by_refs"] = max(0, metrics["tool_output_bytes"] - metrics["inline_context_bytes"])
        persist_metrics()
        return envelope

    def resume(self, plan_id: str, *, session_id: str, dispatch: Callable,
               progress: Callable | None = None, provider_usage: dict | None = None,
               status_only: bool = False) -> dict:
        plan = self.store.get_plan(plan_id)
        if plan is None or plan.id != plan_id or plan.session_id != session_id:
            raise ValueError("Plan not found in the owning conversation")
        if status_only:
            return self.store.operational_ledger(plan_id)
        objective = self.artifacts.read_json(plan.metadata["objective_ref"])
        return self.execute(objective, task_id=plan.metadata["browser_task_id"], session_id=session_id,
                            dispatch=dispatch, progress=progress, provider_usage=provider_usage)


def execute_compiled_work(args: dict, **kwargs) -> str:
    context = _dispatch_context.get()
    if context is None:
        return json.dumps({"error": "Durable work requires the scoped agent dispatcher"})
    dispatch, session_id, _, references, progress, usage = context
    compiler = TaskCompiler()
    if args.get("plan_id"):
        envelope = compiler.resume(args["plan_id"], session_id=session_id, dispatch=dispatch,
                                   progress=progress, provider_usage=usage,
                                   status_only=args.get("action") == "status")
        if args.get("action") == "status":
            references.append({"trusted": True, "version": 1, "source": "KanbanRun", "kind": "durable_work",
                               "id": envelope["plan_id"], "task_id": envelope["task_id"],
                               "owner_session_id": session_id})
            return json.dumps(envelope, ensure_ascii=False)
    else:
        envelope = compiler.execute(args, task_id=str(kwargs.get("task_id") or session_id),
                                    session_id=session_id, dispatch=dispatch, progress=progress,
                                    provider_usage=usage)
    references.append({"trusted": True, "version": 1, "source": "KanbanRun", "kind": "durable_work",
                       "id": envelope["plan_id"], "task_id": envelope["task_id"],
                       "owner_session_id": session_id, "result_ref": envelope["results_ref"]})
    return json.dumps(envelope, ensure_ascii=False)
