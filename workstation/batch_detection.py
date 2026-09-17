"""Per-turn structural fan-out detector; scalar item values do not affect shape."""
import hashlib
import json
from datetime import datetime, timezone

from tools.effects import WRITE_EFFECTS, tool_effect, unwrap_call
from agent.tool_guardrails import classify_tool_failure


def mutation_identity(name, args, *, task_id=None, run_id=None):
    """Bounded identity from owner metadata; unknown effects never claim persistence."""
    from tools.effects import tool_contract
    effect, contract = tool_contract(name)
    target = contract.get("mutation_target") or {}
    scope = target.get("scope", "external" if name.startswith("browser_") else
                       "local" if name in {"write_file", "patch"} else "unknown")
    identifier = args.get(target.get("identifier_field", "path" if scope == "local" else "id"))
    return {"operation_id": call_key(name, args), "task_id": task_id, "run_id": run_id,
            "tool": name, "tool_effect": effect.value, "effect": scope + "_mutation",
            "scope": scope, "external": scope == "external" if scope != "unknown" else None,
            "provider": target.get("provider"), "field": target.get("field"), "target_kind": target.get("kind", "file" if scope == "local" else "unknown"),
            "target_identifier": identifier, "operation_semantic": target.get("operation", args.get("operation", args.get("action", name))),
            "timestamp": datetime.now(timezone.utc).isoformat()}


def mutation_summary(agent):
    records = list(getattr(agent, "_work_mutation_evidence", {}).values())
    counts = {}
    for record in records:
        counts[record["effect"]] = counts.get(record["effect"], 0) + 1
    return {"completed_mutation_count": len(getattr(agent, "_work_completed_mutations", {})),
            "completed_result_refs": list(getattr(agent, "_work_completed_mutations", {}).values())[:8],
            "mutation_identities": records[:8], "effect_summary": counts,
            "identities_truncated": len(records) > 8,
            "persistence_warning": "Tool completion is not persisted external state. Only independent readback confirms a target; unknown/local effects never imply a provider resource was updated."}


def call_key(name, args):
    from tools.effects import tool_contract
    target = tool_contract(name)[1].get("mutation_target") or {}
    fields = target.get("identity_fields")
    if (isinstance(fields, list) and fields and all(isinstance(f, str) and f in args for f in fields)
            and target.get("provider") and target.get("operation")):
        # Owner-declared semantic identity ignores incidental selectors/scripts,
        # while including the desired value/version to distinguish a new update.
        return hashlib.sha256(json.dumps([target["provider"], target.get("kind"), target["operation"],
            {f: args[f] for f in fields}], sort_keys=True).encode()).hexdigest()
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
    seen = set(getattr(agent, "_work_completed_mutations", {})) | set(getattr(agent, "_work_mutation_evidence", {}))
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
    from tools.effects import observe_capability
    capabilities = getattr(agent, "_work_capabilities", {})
    observe_capability(capabilities, name, args, raw)
    agent._work_capabilities = capabilities
    text = raw if isinstance(raw, str) else json.dumps(raw)
    if tool_effect(name) not in WRITE_EFFECTS:
        return
    failed = classify_tool_failure(name, text)[0]
    # Raw result belongs to the existing ArtifactStore, never to replan text.
    from workstation.artifacts import ArtifactStore
    from workstation.reference_plane import content_reference, blob_references
    store = ArtifactStore()
    owner = agent._conversation_root_id() or agent.session_id
    ref = content_reference(store, owner, blob_references(store, owner, raw))["artifact_ref"]
    completed = getattr(agent, "_work_completed_mutations", {})
    key = call_key(name, args)
    evidence = getattr(agent, "_work_mutation_evidence", {})
    if key not in evidence:
        shapes = getattr(agent, "_work_mutation_shapes", {})
        signature = structural_signature(name, args)
        shapes[signature] = shapes.get(signature, 0) + 1
        agent._work_mutation_shapes = shapes
    record = mutation_identity(name, args, task_id=getattr(agent, "_canonical_work_task_id", None),
                               run_id=getattr(agent, "_canonical_work_run_id", None))
    record.update({"evidence_ref": ref, "verifier_status": "not_verified",
                   "persisted": None, "status": "uncertain" if failed else "executed_unverified"})
    evidence[key] = record
    agent._work_mutation_evidence = evidence
    store.store(owner, "mutation_" + key + ".json", record)
    if failed:
        return
    if key not in completed:
        completed[key] = ref
        agent._work_completed_mutations = completed
