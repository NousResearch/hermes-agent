"""Workflow graph tools for opt-in orchestrator profiles."""
from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

import yaml

from hermes_cli import workflows_db as wfdb
from hermes_cli.config import load_config
from hermes_cli.workflows_spec import WorkflowSpec, validate_graph
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


def _check_workflow_mode() -> bool:
    if os.environ.get("HERMES_WORKFLOW_CONTEXT"):
        return True
    try:
        cfg = load_config()
        return "workflow" in (cfg.get("toolsets", []) or [])
    except Exception:
        return False


setattr(_check_workflow_mode, "__hermes_no_check_fn_cache__", True)


@contextlib.contextmanager
def _connect_initialized():
    wfdb.init_db()
    with contextlib.closing(wfdb.connect()) as conn:
        yield conn


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    return str(exc)


def _parse_version(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        version = int(value)
    except (TypeError, ValueError):
        raise ValueError("version must be an integer") from None
    if version < 1:
        raise ValueError("version must be >= 1")
    return version


def _spec_from_definition_text(text: Any) -> WorkflowSpec:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("definition_text must be a non-empty string")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("definition_text must decode to a YAML/JSON object")
    spec = WorkflowSpec.model_validate(loaded)
    validate_graph(spec)
    return spec


def _definition_record(conn, workflow_id: str, version: int | None = None):
    records = [
        record
        for record in wfdb.list_definitions(conn)
        if record.workflow_id == workflow_id and (version is None or record.version == version)
    ]
    if not records:
        if version is None:
            raise KeyError(f"workflow definition not found: {workflow_id}")
        raise KeyError(f"workflow definition not found: {workflow_id} v{version}")
    return max(records, key=lambda record: record.version)


def _definition_to_dict(record, *, include_spec: bool = False) -> dict[str, Any]:
    payload = {
        "checksum": record.checksum,
        "created_at": record.created_at,
        "created_by": record.created_by,
        "enabled": record.enabled,
        "id": record.workflow_id,
        "name": record.name,
        "version": record.version,
        "workflow_id": record.workflow_id,
    }
    if include_spec:
        payload["spec"] = record.spec.model_dump(mode="json", by_alias=True)
    return payload


def _execution_to_dict(execution: wfdb.WorkflowExecution) -> dict[str, Any]:
    return {
        "context": execution.context,
        "created_at": execution.created_at,
        "execution_id": execution.execution_id,
        "input": execution.input,
        "status": execution.status,
        "trigger_id": execution.trigger_id,
        "trigger_type": execution.trigger_type,
        "updated_at": execution.updated_at,
        "version": execution.version,
        "workflow_id": execution.workflow_id,
    }


def _parse_input_json(value: Any) -> dict[str, Any]:
    data = json.loads("{}" if value in (None, "") else str(value))
    if not isinstance(data, dict):
        raise ValueError("input_json must decode to a JSON object")
    return data


def _handle_list(args: dict, **_kw) -> str:
    try:
        with _connect_initialized() as conn:
            definitions = [_definition_to_dict(record) for record in wfdb.list_definitions(conn)]
        return tool_result({"definitions": definitions})
    except Exception as exc:
        logger.exception("workflow_list failed")
        return tool_error(f"workflow_list: {_error_text(exc)}")


def _handle_show(args: dict, **_kw) -> str:
    workflow_id = (args.get("workflow_id") or "").strip()
    if not workflow_id:
        return tool_error("workflow_id is required")
    try:
        version = _parse_version(args.get("version"))
        with _connect_initialized() as conn:
            record = _definition_record(conn, workflow_id, version)
        return tool_result(_definition_to_dict(record, include_spec=True))
    except Exception as exc:
        logger.exception("workflow_show failed")
        return tool_error(f"workflow_show: {_error_text(exc)}")


def _handle_validate(args: dict, **_kw) -> str:
    try:
        spec = _spec_from_definition_text(args.get("definition_text"))
        return tool_result({
            "valid": True,
            "workflow_id": spec.id,
            "definition": spec.model_dump(mode="json", by_alias=True),
        })
    except Exception as exc:
        logger.exception("workflow_validate failed")
        return tool_error(f"workflow_validate: {_error_text(exc)}")


def _handle_deploy(args: dict, **_kw) -> str:
    try:
        spec = _spec_from_definition_text(args.get("definition_text"))
        created_by = (
            str(args.get("created_by") or "workflow_tool").strip() or "workflow_tool"
        )
        with _connect_initialized() as conn:
            wfdb.deploy_definition(conn, spec, created_by=created_by)
            record = _definition_record(conn, spec.id, spec.version)
        return tool_result(_definition_to_dict(record, include_spec=True))
    except Exception as exc:
        logger.exception("workflow_deploy failed")
        return tool_error(f"workflow_deploy: {_error_text(exc)}")


def _handle_run(args: dict, **_kw) -> str:
    workflow_id = (args.get("workflow_id") or "").strip()
    if not workflow_id:
        return tool_error("workflow_id is required")
    try:
        input_data = _parse_input_json(args.get("input_json", "{}"))
        with _connect_initialized() as conn:
            execution_id = wfdb.start_execution(
                conn,
                workflow_id,
                input_data=input_data,
                trigger_type="manual",
            )
            execution = wfdb.get_execution(conn, execution_id)
        return tool_result({
            "execution_id": execution.execution_id,
            "input": execution.input,
            "status": execution.status,
            "version": execution.version,
            "workflow_id": execution.workflow_id,
        })
    except Exception as exc:
        logger.exception("workflow_run failed")
        return tool_error(f"workflow_run: {_error_text(exc)}")


def _handle_execution_show(args: dict, **_kw) -> str:
    execution_id = (args.get("execution_id") or "").strip()
    if not execution_id:
        return tool_error("execution_id is required")
    try:
        with _connect_initialized() as conn:
            execution = wfdb.get_execution(conn, execution_id)
        return tool_result(_execution_to_dict(execution))
    except Exception as exc:
        logger.exception("workflow_execution_show failed")
        return tool_error(f"workflow_execution_show: {_error_text(exc)}")


def _handle_cancel(args: dict, **_kw) -> str:
    execution_id = (args.get("execution_id") or "").strip()
    if not execution_id:
        return tool_error("execution_id is required")
    try:
        with _connect_initialized() as conn:
            execution, cancelled = wfdb.cancel_execution(
                conn, execution_id, source="workflow_cancel"
            )
            payload = _execution_to_dict(execution)
            payload["cancelled"] = cancelled
            return tool_result(payload)
    except Exception as exc:
        logger.exception("workflow_cancel failed")
        return tool_error(f"workflow_cancel: {_error_text(exc)}")


_WORKFLOW_LIST_SCHEMA = {
    "name": "workflow_list",
    "description": "List deployed workflow graph definitions.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_WORKFLOW_SHOW_SCHEMA = {
    "name": "workflow_show",
    "description": "Show one deployed workflow graph definition and spec.",
    "parameters": {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string", "description": "Workflow id."},
            "version": {"type": "integer", "description": "Optional version; latest when omitted."},
        },
        "required": ["workflow_id"],
    },
}

_WORKFLOW_VALIDATE_SCHEMA = {
    "name": "workflow_validate",
    "description": "Validate workflow YAML/JSON definition text without deploying it.",
    "parameters": {
        "type": "object",
        "properties": {
            "definition_text": {
                "type": "string",
                "description": "Workflow YAML or JSON definition text.",
            },
        },
        "required": ["definition_text"],
    },
}

_WORKFLOW_DEPLOY_SCHEMA = {
    "name": "workflow_deploy",
    "description": "Validate and deploy workflow YAML/JSON definition text.",
    "parameters": {
        "type": "object",
        "properties": {
            "definition_text": {
                "type": "string",
                "description": "Workflow YAML or JSON definition text.",
            },
            "created_by": {
                "type": "string",
                "description": "Optional deployment source; defaults to workflow_tool.",
            },
        },
        "required": ["definition_text"],
    },
}

_WORKFLOW_RUN_SCHEMA = {
    "name": "workflow_run",
    "description": "Start a manual workflow execution with JSON object input.",
    "parameters": {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string", "description": "Workflow id to run."},
            "input_json": {"type": "string", "description": "JSON object input. Defaults to {}."},
        },
        "required": ["workflow_id"],
    },
}

_WORKFLOW_EXECUTION_SHOW_SCHEMA = {
    "name": "workflow_execution_show",
    "description": "Show a workflow execution's status, input, and context.",
    "parameters": {
        "type": "object",
        "properties": {
            "execution_id": {"type": "string", "description": "Workflow execution id."},
        },
        "required": ["execution_id"],
    },
}

_WORKFLOW_CANCEL_SCHEMA = {
    "name": "workflow_cancel",
    "description": "Cancel a non-terminal workflow execution idempotently.",
    "parameters": {
        "type": "object",
        "properties": {
            "execution_id": {"type": "string", "description": "Workflow execution id."},
        },
        "required": ["execution_id"],
    },
}


registry.register(
    name="workflow_list",
    toolset="workflow",
    schema=_WORKFLOW_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_show",
    toolset="workflow",
    schema=_WORKFLOW_SHOW_SCHEMA,
    handler=_handle_show,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_validate",
    toolset="workflow",
    schema=_WORKFLOW_VALIDATE_SCHEMA,
    handler=_handle_validate,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_deploy",
    toolset="workflow",
    schema=_WORKFLOW_DEPLOY_SCHEMA,
    handler=_handle_deploy,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_run",
    toolset="workflow",
    schema=_WORKFLOW_RUN_SCHEMA,
    handler=_handle_run,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_execution_show",
    toolset="workflow",
    schema=_WORKFLOW_EXECUTION_SHOW_SCHEMA,
    handler=_handle_execution_show,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_cancel",
    toolset="workflow",
    schema=_WORKFLOW_CANCEL_SCHEMA,
    handler=_handle_cancel,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
