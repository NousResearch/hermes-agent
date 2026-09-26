"""Gemini/Antigravity routing seam for ``delegate_task``.

The normal AIAgent child remains the authority and optional fallback. This
module only builds the bounded output-only adapter and copies its route truth
into parent-visible result records.
"""

from __future__ import annotations

import contextlib
from pathlib import Path, PurePath
from typing import Any, Dict


_ROUTE_METADATA_KEYS = (
    "route",
    "route_reason",
    "receipt_id",
    "gemini_error_code",
    "worker_route",
    "worker_provider",
    "worker_model_requested",
    "route_receipt_id",
    "fallback_used",
)

_DEFAULT_ROUTE_RECEIPT_DB = "routing/gemini-routing.sqlite3"
_RUN_KINDS = {"production", "canary", "synthetic", "evaluation"}
_ROUTE_REQUESTS = {"auto", "gemini", "sol"}


def frontier_receipt_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return the small validated config used by every Frontier receipt."""
    source = config if isinstance(config, dict) else {}
    receipt_db = source.get("receipt_db")
    if not isinstance(receipt_db, str) or not receipt_db.strip():
        receipt_db = _DEFAULT_ROUTE_RECEIPT_DB
    receipt_path = PurePath(receipt_db)
    if receipt_path.is_absolute() or ".." in receipt_path.parts:
        receipt_db = _DEFAULT_ROUTE_RECEIPT_DB

    run_kind = source.get("default_run_kind")
    if run_kind not in _RUN_KINDS:
        run_kind = "production"
    default_route = source.get("default_route")
    if default_route not in _ROUTE_REQUESTS:
        default_route = "auto"
    classification = source.get("default_data_classification")
    if not isinstance(classification, str) or not classification.strip():
        classification = "standard"
    return {
        "receipt_db": receipt_db,
        "default_run_kind": run_kind,
        "default_route": default_route,
        "default_data_classification": classification,
    }


def merge_child_route_metadata(
    entry: Dict[str, Any], child: Any, result: Any = None
) -> None:
    """Copy current child routing truth, then overlay terminal result truth."""
    route_metadata = getattr(child, "_route_metadata", None)
    for source in (route_metadata, result):
        if not isinstance(source, dict):
            continue
        for key in _ROUTE_METADATA_KEYS:
            if key in source:
                entry[key] = source[key]


def active_profile_name() -> str:
    """Return the profile whose Hermes home owns routing policy and receipts."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name()
    except Exception:
        return "default"


def _run_kind(task: Dict[str, Any], routing_cfg: Dict[str, Any]) -> str:
    value = str(task.get("run_kind") or routing_cfg.get("default_run_kind") or "production")
    if value not in _RUN_KINDS:
        raise ValueError("run_kind must be production, canary, synthetic, or evaluation")
    return value


def wrap_frontier_delegate_child(
    *,
    child: Any,
    parent_agent: Any,
    task_index: int,
    task: Dict[str, Any],
    routing_cfg: Dict[str, Any],
    route_reason: str,
    receipt_path: Path | None = None,
) -> Any:
    """Wrap one normal delegation child with the route-neutral receipt producer."""
    from agent.gemini_route_receipts import resolve_profile_receipt_path
    from agent.route_receipts import RouteReceiptChild, RouteReceiptStore
    from hermes_constants import get_hermes_home

    path = receipt_path or resolve_profile_receipt_path(
        get_hermes_home(), Path(routing_cfg["receipt_db"])
    )
    wrapped = RouteReceiptChild(
        child=child,
        store=RouteReceiptStore(path),
        parent_session_id=str(getattr(parent_agent, "session_id", "") or ""),
        parent_turn_id=str(getattr(parent_agent, "_current_turn_id", "") or ""),
        task_index=task_index,
        run_kind=_run_kind(task, routing_cfg),
        route_requested=str(task.get("route") or routing_cfg.get("default_route") or "auto"),
        route_reason=route_reason,
        data_classification=str(
            task.get("data_classification")
            or routing_cfg.get("default_data_classification")
            or "standard"
        ),
        output_contract=str(task.get("output_contract") or "text"),
    )
    _replace_registered_child(parent_agent, child, wrapped)
    return wrapped


def _replace_registered_child(parent_agent: Any, old_child: Any, new_child: Any) -> None:
    """Replace one parent-owned child without leaving duplicate lifecycle owners."""
    active = getattr(parent_agent, "_active_children", None)
    if isinstance(active, list):
        lock = getattr(parent_agent, "_active_children_lock", None)
        lock_context = lock if lock is not None else contextlib.nullcontext()
        with lock_context:
            try:
                index = active.index(old_child)
            except ValueError:
                active.append(new_child)
            else:
                active[index] = new_child


class RoutingInitializationFailureChild:
    """AIAgent-compatible fail-closed result when Gemini setup cannot start."""

    def __init__(
        self,
        *,
        task_index: int,
        routing_cfg: Dict[str, Any],
        route_reason: str,
        route: str = "gemini",
        worker_provider: str = "antigravity-subscription",
        worker_model: str | None = None,
        unstarted_child: Any | None = None,
    ) -> None:
        self.session_id = f"{route}-receipt-init-failed-{task_index}"
        self.model = str(
            worker_model or routing_cfg.get("model") or "gemini-3.8-flash-low"
        )
        self.provider = worker_provider
        self._unstarted_child = unstarted_child
        self._error = (
            "Gemini receipt initialization failed"
            if route == "gemini"
            else "Delegation receipt initialization failed"
        )
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names: list[str] = []
        self._credential_pool = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.tool_progress_callback: Any = None
        self._live_transcript_path = ""
        self._route_metadata = {
            "route": route,
            "route_reason": route_reason,
            "gemini_error_code": "receipt_initialization_failed",
            "worker_route": route,
            "worker_provider": self.provider,
            "worker_model_requested": self.model,
            "route_receipt_id": None,
            "fallback_used": False,
        }

    def run_conversation(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "final_response": "",
            "completed": False,
            "api_calls": 0,
            "messages": [],
            "error": self._error,
            **self._route_metadata,
        }

    def get_activity_summary(self) -> Dict[str, Any]:
        return {"api_call_count": 0, "max_iterations": 0, "current_tool": None}

    def close(self) -> None:
        child = self._unstarted_child
        self._unstarted_child = None
        close = getattr(child, "close", None)
        if callable(close):
            close()


def build_antigravity_delegate_child(
    *,
    task_index: int,
    task: Dict[str, Any],
    fallback_child: Any | None,
    routing_cfg: Dict[str, Any],
    route_reason: str,
    parent_agent: Any,
) -> Any:
    """Build the output-only adapter without publishing project capabilities."""
    from agent.antigravity_delegate import AntigravityDelegateChild
    from agent.antigravity_worker import AntigravityWorker
    from agent.gemini_route_receipts import GeminiReceiptStore, resolve_profile_receipt_path
    from agent.route_receipts import RouteReceiptChild, RouteReceiptStore
    from hermes_constants import get_hermes_home

    receipt_path = resolve_profile_receipt_path(
        get_hermes_home(), Path(routing_cfg["receipt_db"])
    )
    output_contract = str(task.get("output_contract") or "text")
    output_schema = task.get("output_schema")
    if output_contract == "json" and output_schema is None:
        output_schema = {}

    worker = AntigravityWorker(
        command=routing_cfg["command"],
        model=routing_cfg["model"],
        effort=routing_cfg["effort"],
        timeout_seconds=float(routing_cfg["timeout_seconds"]),
        max_input_bytes=routing_cfg["max_input_bytes"],
        max_output_bytes=routing_cfg["max_output_bytes"],
        extra_args=list(routing_cfg.get("extra_args", [])),
    )
    try:
        if fallback_child is not None and not isinstance(
            fallback_child, RouteReceiptChild
        ):
            fallback_child = wrap_frontier_delegate_child(
                child=fallback_child,
                parent_agent=parent_agent,
                task_index=task_index,
                task=task,
                routing_cfg=routing_cfg,
                route_reason=route_reason,
                receipt_path=receipt_path,
            )
        child = AntigravityDelegateChild(
            worker=worker,
            fallback_child=fallback_child,
            store=GeminiReceiptStore(receipt_path),
            route_store=RouteReceiptStore(receipt_path),
            run_kind=_run_kind(task, routing_cfg),
            task_index=task_index,
            goal=task["goal"],
            context=str(task.get("context") or ""),
            output_schema=output_schema,
            output_contract=output_contract,
            route_requested=str(
                task.get("route") or routing_cfg.get("default_route") or "auto"
            ),
            route_reason=route_reason,
            data_classification=str(
                task.get("data_classification")
                or routing_cfg.get("default_data_classification")
                or "standard"
            ),
            requested_provider="antigravity-subscription",
            requested_model=routing_cfg["model"],
            requested_effort=routing_cfg["effort"],
            parent_session_id=str(getattr(parent_agent, "session_id", "") or ""),
            parent_turn_id=str(
                getattr(parent_agent, "_current_turn_id", "") or ""
            ),
        )
        child.prepare_receipt()
    except Exception:
        worker.close()
        raise

    # The fallback was registered for parent interrupt propagation. Replace that
    # exact object with the adapter so cancellation reaches both the subprocess
    # and a fallback that has actually started.
    if fallback_child is not None and hasattr(parent_agent, "_active_children"):
        lock = getattr(parent_agent, "_active_children_lock", None)
        lock_context = lock if lock is not None else contextlib.nullcontext()
        with lock_context:
            try:
                index = parent_agent._active_children.index(fallback_child)
            except ValueError:
                parent_agent._active_children.append(child)
            else:
                parent_agent._active_children[index] = child
    return child
