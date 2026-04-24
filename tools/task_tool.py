from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent.task_store import PersistentTaskRecord, TaskStatus, TaskStore
from tools.registry import registry, tool_error, tool_result


_MUTATING_ACTIONS = {
    "create",
    "update_metadata",
    "add_dependency",
    "remove_dependency",
    "cancel",
    "retry",
    "reconcile",
}

_SENSITIVE_KEY_TOKENS = ("api_key", "token", "secret", "password", "authorization")
_REDACTED = "<redacted>"


class TaskAction(str, Enum):
    create = "create"
    list = "list"
    get = "get"
    update_metadata = "update_metadata"
    add_dependency = "add_dependency"
    remove_dependency = "remove_dependency"
    cancel = "cancel"
    retry = "retry"
    reconcile = "reconcile"
    runnable = "runnable"


class TaskToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: TaskAction
    task_id: str | None = None
    task_ids: list[str] | None = None
    goal: str | None = None
    context: str | None = None
    activeForm: str | None = None
    owner: str | None = None
    threadID: str | None = None
    owner_session_id: str | None = None
    metadata: dict[str, Any] | None = None
    launch_spec: dict[str, Any] | None = None
    dependency_id: str | None = None
    statuses: list[str] | None = None

    @field_validator("task_id", "goal", "context", "activeForm", "owner", "threadID", "owner_session_id", "dependency_id")
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("task_ids")
    @classmethod
    def _normalize_task_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            task_id = str(item or "").strip()
            if not task_id or task_id in seen:
                continue
            normalized.append(task_id)
            seen.add(task_id)
        return normalized

    @field_validator("statuses")
    @classmethod
    def _normalize_statuses(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [TaskStatus(str(item).strip()).value for item in value]

    @model_validator(mode="after")
    def _check_action_requirements(self) -> "TaskToolArgs":
        if self.action == TaskAction.create and not self.goal:
            raise ValueError("goal is required for create")
        if self.action in {
            TaskAction.get,
            TaskAction.update_metadata,
            TaskAction.add_dependency,
            TaskAction.remove_dependency,
            TaskAction.cancel,
            TaskAction.retry,
        } and not self.task_id:
            raise ValueError(f"task_id is required for {self.action.value}")
        if self.action in {TaskAction.add_dependency, TaskAction.remove_dependency} and not self.dependency_id:
            raise ValueError(f"dependency_id is required for {self.action.value}")
        if self.action == TaskAction.update_metadata and self.metadata is None:
            raise ValueError("metadata is required for update_metadata")
        if self.action == TaskAction.reconcile and not self.task_id and not self.task_ids:
            raise ValueError("task_id or task_ids is required for reconcile")
        return self


class TaskMutationDecision(BaseModel):
    allowed: bool
    code: str
    message: str


def default_task_mutation_guard(*, action: str, read_only_agent: bool = False, can_mutate_tasks: bool = False, permission_granted: bool = False, **_: Any) -> TaskMutationDecision:
    if action not in _MUTATING_ACTIONS:
        return TaskMutationDecision(allowed=True, code="ok", message="ok")
    if read_only_agent:
        return TaskMutationDecision(
            allowed=False,
            code="task_mutation_read_only",
            message="Read-only agents cannot mutate persistent tasks",
        )
    if not (can_mutate_tasks and permission_granted):
        return TaskMutationDecision(
            allowed=False,
            code="task_mutation_not_allowed",
            message="Task mutation requires explicit approval",
        )
    return TaskMutationDecision(allowed=True, code="ok", message="ok")


def _is_sensitive_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    return any(token in normalized for token in _SENSITIVE_KEY_TOKENS)


def _redact_sensitive_value(value: Any, *, key_hint: str | None = None) -> Any:
    if key_hint and _is_sensitive_key(key_hint):
        return _REDACTED
    if isinstance(value, dict):
        return {
            str(key): _redact_sensitive_value(nested, key_hint=str(key))
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_value(item, key_hint=key_hint) for item in value]
    return value


def _serialize_task(record: PersistentTaskRecord) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    payload["launch_spec"] = _redact_sensitive_value(payload.get("launch_spec") or {}, key_hint="launch_spec")
    return payload


def _serialize_tasks(records: list[PersistentTaskRecord]) -> list[dict[str, Any]]:
    return [_serialize_task(record) for record in records]


def _validation_error_payload(exc: ValidationError) -> dict[str, Any]:
    details: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", [])) or "args"
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return {"error": "Invalid task tool arguments", "details": details}


def task_tool(
    *,
    action: str,
    task_id: str | None = None,
    task_ids: list[str] | None = None,
    goal: str | None = None,
    context: str | None = None,
    activeForm: str | None = None,
    owner: str | None = None,
    threadID: str | None = None,
    owner_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    launch_spec: dict[str, Any] | None = None,
    dependency_id: str | None = None,
    statuses: list[str] | None = None,
    store: TaskStore | None = None,
    mutation_guard=default_task_mutation_guard,
    process_registry=None,
    **context_kwargs: Any,
) -> str:
    try:
        parsed = TaskToolArgs.model_validate(
            {
                "action": action,
                "task_id": task_id,
                "task_ids": task_ids,
                "goal": goal,
                "context": context,
                "activeForm": activeForm,
                "owner": owner,
                "threadID": threadID,
                "owner_session_id": owner_session_id,
                "metadata": metadata,
                "launch_spec": launch_spec,
                "dependency_id": dependency_id,
                "statuses": statuses,
            }
        )
    except ValidationError as exc:
        return tool_result(_validation_error_payload(exc))

    decision = mutation_guard(action=parsed.action.value, **context_kwargs)
    if not decision.allowed:
        return tool_error(decision.message, code=decision.code)

    task_store = store or TaskStore()

    try:
        if parsed.action == TaskAction.create:
            record = task_store.create_task(
                goal=parsed.goal or "",
                context=parsed.context,
                activeForm=parsed.activeForm or "",
                owner=parsed.owner or "",
                threadID=parsed.threadID or "",
                owner_session_id=parsed.owner_session_id,
                metadata=parsed.metadata,
                launch_spec=parsed.launch_spec,
            )
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(record))

        if parsed.action == TaskAction.list:
            records = task_store.list_tasks(owner_session_id=parsed.owner_session_id, statuses=parsed.statuses)
            return tool_result(success=True, action=parsed.action.value, tasks=_serialize_tasks(records))

        if parsed.action == TaskAction.get:
            record = task_store.get_task(parsed.task_id or "")
            if record is None:
                return tool_error(f"task not found: {parsed.task_id}", code="task_not_found")
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(record))

        if parsed.action == TaskAction.update_metadata:
            record = task_store.require_task(parsed.task_id or "")
            record.metadata = dict(parsed.metadata or {})
            saved = task_store.save_task(record)
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(saved))

        if parsed.action == TaskAction.add_dependency:
            record = task_store.require_task(parsed.task_id or "")
            task_store.require_task(parsed.dependency_id or "")
            record.blockedBy = list(record.blockedBy) + [parsed.dependency_id or ""]
            saved = task_store.save_task(record)
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(saved))

        if parsed.action == TaskAction.remove_dependency:
            record = task_store.require_task(parsed.task_id or "")
            dependency_id_value = parsed.dependency_id or ""
            record.blockedBy = [item for item in record.blockedBy if item != dependency_id_value]
            saved = task_store.save_task(record)
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(saved))

        if parsed.action == TaskAction.cancel:
            saved = task_store.transition_task(parsed.task_id or "", TaskStatus.cancelled)
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(saved))

        if parsed.action == TaskAction.retry:
            saved = task_store.prepare_for_retry(parsed.task_id or "")
            return tool_result(success=True, action=parsed.action.value, task=_serialize_task(saved))

        if parsed.action == TaskAction.reconcile:
            records = task_store.reconcile_tasks(
                owner_session_id=parsed.owner_session_id,
                task_ids=[parsed.task_id] if parsed.task_id else parsed.task_ids,
                process_registry=process_registry,
            )
            if parsed.task_id and records:
                return tool_result(success=True, action=parsed.action.value, task=_serialize_task(records[0]))
            return tool_result(success=True, action=parsed.action.value, tasks=_serialize_tasks(records))

        if parsed.action == TaskAction.runnable:
            records = task_store.get_runnable_tasks(owner_session_id=parsed.owner_session_id, statuses=parsed.statuses)
            return tool_result(success=True, action=parsed.action.value, tasks=_serialize_tasks(records))

        return tool_error(f"Unsupported task action: {parsed.action.value}", code="unsupported_task_action")
    except KeyError as exc:
        return tool_error(str(exc), code="task_not_found")
    except ValueError as exc:
        return tool_error(str(exc), code="invalid_task_operation")


TASK_SCHEMA = {
    "name": "task",
    "description": (
        "Manage persistent TaskStore tasks. Read operations: list, get, runnable. "
        "Mutation operations: create, update_metadata, add_dependency, remove_dependency, "
        "cancel, retry, reconcile. Sensitive launch_spec fields are redacted in tool output."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": [action.value for action in TaskAction],
                "description": "Task operation to perform.",
            },
            "task_id": {"type": "string", "description": "Target task id for single-task operations."},
            "task_ids": {
                "type": "array",
                "description": "Optional task ids for reconcile.",
                "items": {"type": "string"},
            },
            "goal": {"type": "string", "description": "Goal text for create."},
            "context": {"type": "string", "description": "Optional task context for create."},
            "activeForm": {"type": "string", "description": "Optional activeForm for create."},
            "owner": {"type": "string", "description": "Optional owner label for create."},
            "threadID": {"type": "string", "description": "Optional thread identifier for create."},
            "owner_session_id": {"type": "string", "description": "Optional session filter for list, reconcile, and runnable."},
            "metadata": {"type": "object", "description": "Metadata dict for create or update_metadata."},
            "launch_spec": {"type": "object", "description": "Launch configuration for create; sensitive fields are redacted in output."},
            "dependency_id": {"type": "string", "description": "Dependency task id for add_dependency/remove_dependency."},
            "statuses": {
                "type": "array",
                "description": "Optional status filter for list or runnable.",
                "items": {
                    "type": "string",
                    "enum": [status.value for status in TaskStatus],
                },
            },
        },
        "required": ["action"],
    },
}


def check_task_requirements() -> bool:
    return True


def _task_handler(args: dict, **kwargs: Any) -> str:
    schema_keys = set(TaskToolArgs.model_fields)
    context_kwargs = {key: value for key, value in kwargs.items() if key not in schema_keys}
    return task_tool(
        action=args.get("action"),
        task_id=args.get("task_id"),
        task_ids=args.get("task_ids"),
        goal=args.get("goal"),
        context=args.get("context"),
        activeForm=args.get("activeForm"),
        owner=args.get("owner"),
        threadID=args.get("threadID"),
        owner_session_id=args.get("owner_session_id"),
        metadata=args.get("metadata"),
        launch_spec=args.get("launch_spec"),
        dependency_id=args.get("dependency_id"),
        statuses=args.get("statuses"),
        **context_kwargs,
    )


registry.register(
    name="task",
    toolset="task",
    schema=TASK_SCHEMA,
    handler=_task_handler,
    check_fn=check_task_requirements,
    emoji="🗂️",
)
