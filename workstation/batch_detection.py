"""Per-turn structural fan-out detector; scalar item values do not affect shape."""
import hashlib
import json

from tools.effects import WRITE_EFFECTS, tool_effect, unwrap_call
from agent.tool_guardrails import classify_tool_failure


def call_key(name, args):
    return hashlib.sha256(json.dumps([name, args], sort_keys=True).encode()).hexdigest()


def structural_signature(name, args):
    def shape(value, field=""):
        if isinstance(value, dict):
            return {k: shape(v, k) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return sorted({json.dumps(shape(v), sort_keys=True) for v in value})
        # Operation selectors are semantic, not item-specific values.
        return value if field in {"action", "operation", "method"} else type(value).__name__
    return call_key(name, shape(args))


def detects_fan_out(agent, calls):
    if "work_execute" not in getattr(agent, "valid_tool_names", ()):
        return False
    counts = dict(getattr(agent, "_work_mutation_shapes", {}))
    seen = set(getattr(agent, "_work_completed_mutations", {}))
    for call in calls:
        name, args = unwrap_call(call)
        if name == "work_execute" or tool_effect(name) not in WRITE_EFFECTS:
            continue
        key = call_key(name, args)
        if key in seen:
            continue  # Identical replays remain the responsibility of guardrails.
        seen.add(key)
        signature = structural_signature(name, args)
        counts[signature] = counts.get(signature, 0) + 1
        if counts[signature] >= 3:
            return True
    return False


def record_mutation(agent, name, args, raw, *, dispatched=True):
    from workstation.task_compiler import durable_execution_active
    if "work_execute" not in agent.valid_tool_names:
        return
    if durable_execution_active() or not dispatched or name == "work_execute":
        return
    if name == "tool_call":
        name, args = args.get("name", ""), args.get("arguments", {})
    text = raw if isinstance(raw, str) else json.dumps(raw)
    if tool_effect(name) not in WRITE_EFFECTS or classify_tool_failure(name, text)[0]:
        return
    # Raw result belongs to the existing ArtifactStore, never to replan text.
    from workstation.artifacts import ArtifactStore
    from workstation.reference_plane import content_reference, blob_references
    store = ArtifactStore()
    owner = agent._conversation_root_id() or agent.session_id
    ref = content_reference(store, owner, blob_references(store, owner, raw))["artifact_ref"]
    completed = getattr(agent, "_work_completed_mutations", {})
    key = call_key(name, args)
    if key not in completed:
        shapes = getattr(agent, "_work_mutation_shapes", {})
        signature = structural_signature(name, args)
        shapes[signature] = shapes.get(signature, 0) + 1
        agent._work_mutation_shapes = shapes
        completed[key] = ref
        agent._work_completed_mutations = completed
