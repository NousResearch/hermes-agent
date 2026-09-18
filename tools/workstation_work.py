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
        "Use action=discover with at most 8 structured read-only preflight probes before a plan is known; no mutations or arbitrary JS/shell. Successful GET does not authorize PUT. "
        "Use action=contract for bounded examples, recipe_key + items/items_ref to reuse a verified graph, "
        "or setup_steps/steps/finalize_steps with unique IDs and depends_on. "
        "Use $setup.id.field and $steps.id.field for prior results. "
        "Unknown mutable batches automatically gate item 1 as canary. Add a read/discovery "
        "step with verifies=[mutation_id] and expect proving persisted state. "
        "Confirmed mutations survive restart; uncertain mutations require human review, never blind retry. "
        "SUMMARY is default; FULL explicitly loads content. "
        "Use a stable operation_key for restart/resume. Mutations require expect: dotted JSON "
        "result paths mapped to expected values. "
        "Use plan_id with action=resume/status to reconstruct directly from durable state. "
        "Every step uses the existing scoped dispatcher. "
        "Prompt queues must include submit, bounded completion wait and capture steps. "
        "Returns compact refs, operational ledger and exceptions; never raw item outputs."),
        "parameters": {"type": "object", "properties": {
            "operation_key": {"type": "string"}, "title": {"type": "string"},
            "mutation_target": {"type": "object", "description": "Intended resource scope (external/local), provider, kind and field. Declare external tasks so preparation-only plans fail before dispatch."},
            "plan_id": {"type": "string"}, "action": {"type": "string", "enum": ["execute", "resume", "status", "contract", "discover"]},
            "recipe_key": {"type": "string"}, "recipe_scope": {"type": "object", "description": "Stable route/host/path_family; no secrets or item IDs."},
            "operation_fingerprint": {"type": "string", "description": "Exact verified fingerprint for automatic reuse; scope, target and preconditions must agree."},
            "routine_preconditions": {"type": "array", "items": {"type": "string"}, "description": "Exact structured precondition contracts for harness-selected promoted routines."},
            "preflight": {"type": "array", "maxItems": 8, "items": {"type": "object"}, "description": "Read-only capability probes with expect; run once before cached fan-out."},
            "verbosity": {"type": "string", "enum": ["minimal", "summary", "full"]},
            "kind": {"type": "string", "enum": ["batch", "browser_transaction", "prompt_queue"]},
            "transaction_contract": {"type": "string", "description": "Owner-registered UI operation contract; caller phase labels alone never relax evidence."},
            "items": {"type": "array", "items": {"type": "object"}},
            "items_ref": {"type": "string", "description": "ArtifactStore dataset reference instead of inline records."},
            "setup_steps": {"type": "array", "items": {"type": "object"}, "description": "Shared steps executed once; bind results with $setup.step_id.field."},
            "finalize_steps": {"type": "array", "items": {"type": "object"}, "description": "Durable fan-in after all items complete; $items_ref contains item result refs."},
            "steps": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "depends_on": {"type": "array", "items": {"type": "string"}},
                "verifies": {"type": "array", "items": {"type": "string"}, "description": "A read/discovery step with expect proves these mutation IDs."},
                "readback": {"type": "object", "description": "Map intended resource field to a different serialized result path: field + path; expect must compare that path."},
                "tool": {"type": "string"}, "args": {"type": "object"}, "expect": {"type": "object"},
                "transaction_phase": {"type": "string", "enum": ["PREPARE", "INTERACT", "COMMIT", "VERIFY"]},
                "semantic_anchor": {"type": "object"},
                "wait": {"type": "object", "properties": {
                    "timeout_seconds": {"type": "number"}, "interval_seconds": {"type": "number"},
                    "max_polls": {"type": "integer"}}}},
                "required": ["tool", "args"]}},
            "capability_id": {"type": "string", "description": "Deterministic Operational Capability identifier to execute or compose."},
            "capability_version": {"type": "string", "description": "Optional version constraint for the capability."},
            "capability_inputs": {"type": "object", "description": "Input arguments mapped to the capability schema."},
            "constraints": {"type": "object", "properties": {
                "mutation_allowed_routes": {"type": "array", "items": {"type": "string"}, "description": "Mutation channel authority only; independent verification reads remain permitted."},
                "mutation_forbidden_routes": {"type": "array", "items": {"type": "string"}},
                "allowed_routes": {"type": "array", "items": {"type": "string"}},
                "forbidden_routes": {"type": "array", "items": {"type": "string"}}}},
        }, "anyOf": [{"required": ["operation_key", "steps"]}, {"required": ["recipe_key"]}, {"required": ["capability_id"]}, {"required": ["operation_fingerprint", "recipe_scope"]}, {"required": ["plan_id"]}, {"properties": {"action": {"enum": ["contract", "discover"]}}, "required": ["action"]}]}},
)
