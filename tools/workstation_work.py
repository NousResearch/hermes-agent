"""Session-gated durable work capability; never part of the core toolset."""
from tools.registry import registry
from workstation.task_compiler import execute_compiled_work


def _runtime_enabled():
    from workstation.config import load_workstation_config
    return load_workstation_config().enabled


registry.register(
    name="work_execute", toolset="desktop_ui", check_fn=_runtime_enabled,
    handler=execute_compiled_work,
    schema={"name": "work_execute", "description": (
        "Execute decided repetitive work as a durable plan, with no model call between items. "
        "For homogeneous records, browser transactions or prompt queues, compile once into items "
        "and steps instead of calling individual tools per item. Bind args using $item.field. "
        "Use a stable operation_key for restart/resume. Mutations require expect: dotted JSON "
        "result paths mapped to expected values. "
        "Use plan_id with action=resume/status to reconstruct directly from durable state. "
        "Every step uses the existing scoped dispatcher. "
        "Prompt queues must include submit, bounded completion wait and capture steps. "
        "Returns compact refs, operational ledger and exceptions; never raw item outputs."),
        "parameters": {"type": "object", "properties": {
            "operation_key": {"type": "string"}, "title": {"type": "string"},
            "plan_id": {"type": "string"}, "action": {"type": "string", "enum": ["execute", "resume", "status"]},
            "verbosity": {"type": "string", "enum": ["minimal", "summary", "full"]},
            "kind": {"type": "string", "enum": ["batch", "browser_transaction", "prompt_queue"]},
            "items": {"type": "array", "items": {"type": "object"}},
            "items_ref": {"type": "string", "description": "ArtifactStore dataset reference instead of inline records."},
            "steps": {"type": "array", "items": {"type": "object", "properties": {
                "tool": {"type": "string"}, "args": {"type": "object"}, "expect": {"type": "object"},
                "wait": {"type": "object", "properties": {
                    "timeout_seconds": {"type": "number"}, "interval_seconds": {"type": "number"},
                    "max_polls": {"type": "integer"}}}},
                "required": ["tool", "args"]}},
            "constraints": {"type": "object", "properties": {
                "allowed_routes": {"type": "array", "items": {"type": "string"}},
                "forbidden_routes": {"type": "array", "items": {"type": "string"}}}},
        }, "anyOf": [{"required": ["operation_key", "steps"]}, {"required": ["plan_id"]}]}},
)
