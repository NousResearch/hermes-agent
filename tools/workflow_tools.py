"""Workflow graph tools for opt-in orchestrator profiles."""
from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

import yaml

from hermes_cli import workflows_assistant, workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.config import load_config
from hermes_cli.workflows_capabilities import (
    require_available_profiles,
    require_implemented_primitives,
)
from hermes_cli.workflows_redaction import redact_sensitive
from hermes_cli.workflows_spec import (
    WorkflowSpec,
    reject_unknown_spec_fields,
    validate_graph,
)
from tools.registry import registry, tool_error, tool_result
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _check_workflow_mode() -> bool:
    if os.environ.get("HERMES_WORKFLOW_CONTEXT"):
        return True
    try:
        cfg = load_config()
        toolsets = cfg.get("toolsets") or []
        if isinstance(toolsets, (list, tuple, set)) and "workflow" in toolsets:
            return True
        platform_toolsets = cfg.get("platform_toolsets") or {}
        if isinstance(platform_toolsets, dict):
            return any(
                "workflow" in toolsets
                for toolsets in platform_toolsets.values()
                if isinstance(toolsets, (list, tuple, set))
            )
        return False
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
        return str(redact_sensitive(str(exc.args[0])))
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    return str(redact_sensitive(str(exc)))


def _assistant_runtime_error(tool_name: str) -> str:
    return tool_error(f"{tool_name}: workflow assistant failed")


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


def _spec_from_args(args: dict) -> WorkflowSpec:
    if "definition" in args:
        raw = args["definition"]
        if not isinstance(raw, dict):
            raise ValueError("definition must be a workflow object")
    else:
        text = args.get("definition_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("definition or definition_text is required")
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("definition_text must decode to a YAML/JSON object")
    return _spec_from_object(raw)


def _spec_from_object(value: Any) -> WorkflowSpec:
    if not isinstance(value, dict):
        raise ValueError("spec must be an object")
    reject_unknown_spec_fields(value)
    spec = WorkflowSpec.model_validate(value)
    validate_graph(spec)
    require_implemented_primitives(spec)
    from hermes_cli import profiles as profiles_mod
    available = {p.name for p in profiles_mod.list_profiles()}
    require_available_profiles(spec, available)
    return spec


def _dispatch_in_gateway_enabled() -> bool:
    try:
        cfg = load_config()
    except Exception:
        return True
    workflow_cfg = cfg.get("workflow") if isinstance(cfg, dict) else {}
    if not isinstance(workflow_cfg, dict):
        workflow_cfg = {}
    return is_truthy_value(workflow_cfg.get("dispatch_in_gateway"), default=True)


def _definition_record(conn, workflow_id: str, version: int | None = None):
    return wfdb.get_definition_record(conn, workflow_id, version)


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
        payload["spec"] = redact_sensitive(record.spec.model_dump(mode="json", by_alias=True))
    return payload


def _event_to_dict(event: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(event)
    payload = redacted.get("payload")
    if isinstance(payload, (dict, list, str)):
        redacted["payload"] = redact_sensitive(payload)
    return redacted


def _node_run_to_dict(node_run: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(node_run)
    for key in ("input", "output", "payload", "error"):
        if key in redacted:
            redacted[key] = redact_sensitive(redacted[key])
    return redacted


def _execution_to_dict(execution: wfdb.WorkflowExecution) -> dict[str, Any]:
    return {
        "context": redact_sensitive(execution.context),
        "created_at": execution.created_at,
        "execution_id": execution.execution_id,
        "input": redact_sensitive(execution.input),
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


def _input_from_args(args: dict) -> dict[str, Any]:
    if "input" in args:
        data = args.get("input")
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("input must be an object")
        return data
    return _parse_input_json(args.get("input_json", "{}"))


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


def _handle_draft(args: dict, **_kw) -> str:
    try:
        goal = str(args.get("goal") or "").strip()
        if not goal:
            raise ValueError("goal is required")
        result = workflows_assistant.draft_workflow_with_default_runner(goal)
        return tool_result(result.to_dict())
    except (workflows_assistant.AssistantValidationError, ValueError) as exc:
        logger.exception("workflow_draft failed")
        return tool_error(f"workflow_draft: {_error_text(exc)}")
    except Exception:
        logger.exception("workflow_draft failed")
        return _assistant_runtime_error("workflow_draft")


def _handle_refine(args: dict, **_kw) -> str:
    try:
        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("instruction is required")
        if args.get("spec") is not None:
            spec = _spec_from_object(args.get("spec"))
        else:
            workflow_id = str(args.get("workflow_id") or "").strip()
            if not workflow_id:
                raise ValueError("spec or workflow_id is required")
            version = _parse_version(args.get("version"))
            with _connect_initialized() as conn:
                spec = _definition_record(conn, workflow_id, version).spec
        result = workflows_assistant.refine_workflow_with_default_runner(spec, instruction)
        return tool_result(result.to_dict())
    except (KeyError, ValueError, workflows_assistant.AssistantValidationError) as exc:
        logger.exception("workflow_refine failed")
        return tool_error(f"workflow_refine: {_error_text(exc)}")
    except Exception:
        logger.exception("workflow_refine failed")
        return _assistant_runtime_error("workflow_refine")


def _handle_validate(args: dict, **_kw) -> str:
    try:
        spec = _spec_from_args(args)
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
        spec = _spec_from_args(args)
        created_by = (
            str(args.get("created_by") or "workflow_tool").strip() or "workflow_tool"
        )
        auto_bump = bool(args.get("auto_bump", False))
        with _connect_initialized() as conn:
            deployed_version = wfdb.deploy_definition(
                conn, spec, created_by=created_by, auto_bump=auto_bump
            )
            record = _definition_record(conn, spec.id, deployed_version)
        return tool_result(_definition_to_dict(record, include_spec=True))
    except Exception as exc:
        logger.exception("workflow_deploy failed")
        return tool_error(f"workflow_deploy: {_error_text(exc)}")


def _handle_run(args: dict, **_kw) -> str:
    workflow_id = (args.get("workflow_id") or "").strip()
    if not workflow_id:
        return tool_error("workflow_id is required")
    try:
        input_data = _input_from_args(args)
        with _connect_initialized() as conn:
            execution_id = wfdb.start_manual_execution(
                conn,
                workflow_id,
                input_data=input_data,
            )
        # Advance cheap nodes inline (same as CLI run / dashboard Run) so
        # simple graphs finish immediately and agent_task nodes materialize
        # their Kanban tasks without waiting for the next dispatcher tick.
        try:
            workflows_dispatcher.tick(limit=1)
        except Exception:
            logger.debug("workflow_run initial tick failed", exc_info=True)
        with _connect_initialized() as conn:
            execution = wfdb.get_execution(conn, execution_id)
        payload = {
            "execution_id": execution.execution_id,
            "input": redact_sensitive(execution.input),
            "status": execution.status,
            "version": execution.version,
            "workflow_id": execution.workflow_id,
        }
        if execution.status in {"queued", "waiting"} and not _dispatch_in_gateway_enabled():
            payload["dispatcher_hint"] = (
                "workflow.dispatch_in_gateway is off — nothing advances this "
                "execution automatically. Call workflow_tick (or run `hermes "
                "workflow tick`) to advance it."
            )
        return tool_result(payload)
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
            payload = _execution_to_dict(execution)
            if args.get("include_node_runs"):
                payload["node_runs"] = [
                    _node_run_to_dict(row) for row in wfdb.list_node_runs(conn, execution_id)
                ]
            if args.get("include_events"):
                payload["events"] = [
                    _event_to_dict(event) for event in wfdb.list_events(conn, execution_id)
                ]
        return tool_result(payload)
    except Exception as exc:
        logger.exception("workflow_execution_show failed")
        return tool_error(f"workflow_execution_show: {_error_text(exc)}")


def _handle_tick(args: dict, **_kw) -> str:
    try:
        raw_limit = args.get("limit", 10)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            raise ValueError("limit must be an integer") from None
        if limit < 1:
            raise ValueError("limit must be >= 1")
        wfdb.init_db()
        processed = workflows_dispatcher.tick(limit=limit)
        return tool_result({"processed": processed})
    except Exception as exc:
        logger.exception("workflow_tick failed")
        return tool_error(f"workflow_tick: {_error_text(exc)}")


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

_WORKFLOW_DRAFT_SCHEMA = {
    "name": "workflow_draft",
    "description": "Draft a validated workflow graph spec from a plain-language goal.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Plain-language workflow goal."},
        },
        "required": ["goal"],
    },
}

_WORKFLOW_REFINE_SCHEMA = {
    "name": "workflow_refine",
    "description": "Refine a workflow using a plain-language instruction; callers must provide either spec or workflow_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {"type": "object", "description": "Inline workflow spec object."},
            "workflow_id": {"type": "string", "description": "Deployed workflow id to refine."},
            "version": {"type": "integer", "description": "Optional deployed workflow version; latest when omitted."},
            "instruction": {"type": "string", "description": "Plain-language refinement instruction."},
        },
        "required": ["instruction"],
    },
}

_WORKFLOW_VALIDATE_SCHEMA = {
    "name": "workflow_validate",
    "description": "Validate a workflow definition object without deploying it; legacy definition_text YAML/JSON is also accepted.",
    "parameters": {
        "type": "object",
        "properties": {
            "definition": {
                "type": "object",
                "description": "Workflow definition object (preferred).",
            },
            "definition_text": {
                "type": "string",
                "description": "Legacy workflow YAML or JSON definition text.",
            },
        },
        "required": [],
    },
}

_WORKFLOW_DEPLOY_SCHEMA = {
    "name": "workflow_deploy",
    "description": "Validate and deploy a workflow definition object; legacy definition_text YAML/JSON is also accepted.",
    "parameters": {
        "type": "object",
        "properties": {
            "definition": {
                "type": "object",
                "description": "Workflow definition object (preferred).",
            },
            "definition_text": {
                "type": "string",
                "description": "Legacy workflow YAML or JSON definition text.",
            },
            "created_by": {
                "type": "string",
                "description": "Optional deployment source; defaults to workflow_tool.",
            },
            "auto_bump": {
                "type": "boolean",
                "description": "On checksum conflict, redeploy as the next version instead of erroring. Defaults to false.",
            },
        },
        "required": [],
    },
}

_WORKFLOW_RUN_SCHEMA = {
    "name": "workflow_run",
    "description": "Start a manual workflow execution with an input object; legacy input_json is also accepted.",
    "parameters": {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string", "description": "Workflow id to run."},
            "input": {"type": "object", "description": "Execution input object (preferred). Defaults to {}."},
            "input_json": {"type": "string", "description": "Legacy JSON object input. Defaults to {}."},
        },
        "required": ["workflow_id"],
    },
}

_WORKFLOW_EXECUTION_SHOW_SCHEMA = {
    "name": "workflow_execution_show",
    "description": "Show a workflow execution's status, input, and context; optionally include per-node runs and the event timeline.",
    "parameters": {
        "type": "object",
        "properties": {
            "execution_id": {"type": "string", "description": "Workflow execution id."},
            "include_node_runs": {
                "type": "boolean",
                "description": "Include per-node run records (status, output, errors, linked Kanban tasks). Defaults to false.",
            },
            "include_events": {
                "type": "boolean",
                "description": "Include the execution's event timeline. Defaults to false.",
            },
        },
        "required": ["execution_id"],
    },
}

_WORKFLOW_TICK_SCHEMA = {
    "name": "workflow_tick",
    "description": "Advance queued workflow executions (fires due schedules, resumes waits/retries, collects finished agent tasks). Returns the number processed.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max executions to advance this call. Defaults to 10.",
            },
        },
        "required": [],
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
    name="workflow_draft",
    toolset="workflow",
    schema=_WORKFLOW_DRAFT_SCHEMA,
    handler=_handle_draft,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
registry.register(
    name="workflow_refine",
    toolset="workflow",
    schema=_WORKFLOW_REFINE_SCHEMA,
    handler=_handle_refine,
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
registry.register(
    name="workflow_tick",
    toolset="workflow",
    schema=_WORKFLOW_TICK_SCHEMA,
    handler=_handle_tick,
    check_fn=_check_workflow_mode,
    emoji="🔁",
)
