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
    scope = target.get("scope", "external" if name in {"browser_click", "browser_type", "browser_press"} else
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
    from tools.effects import tool_contract
    contract = tool_contract(name)[1]
    target = contract.get('mutation_target') or {}
    def shape(value, field=""):
        if isinstance(value, dict):
            return {k: shape(v, k) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return sorted({json.dumps(shape(v), sort_keys=True) for v in value})
        # Operation selectors are semantic, not item-specific values.
        return value if field in {"action", "operation", "method"} else type(value).__name__
    family_fields = target.get('target_family_fields', [])
    family = {k: args.get(k) for k in family_fields[:8] if isinstance(k, str)} if isinstance(family_fields, list) else {}
    selectors = {k: args[k] for k in ('action', 'operation', 'method') if k in args}
    return hashlib.sha256(json.dumps([call_key(name, shape(args)), target.get('provider'),
        target.get('kind'), target.get('scope'), contract.get('routes', []), family, selectors], sort_keys=True).encode()).hexdigest()


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


def record_mutation(agent, name, args, raw, *, dispatched=True, duration_ms=None):
    from workstation.task_compiler import durable_execution_active
    if "work_execute" not in agent.valid_tool_names:
        return
    if durable_execution_active() or not dispatched or name == "work_execute":
        return
    if name == "tool_call":
        name, args = args.get("name", ""), args.get("arguments", {})
    from workstation.procedure_trace import record_trace
    record_trace(agent, name, args, raw, duration_ms=duration_ms)
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
    from workstation.execution_policy import semantic_operation_fingerprint
    signature = structural_signature(name, args)
    sem_fp = semantic_operation_fingerprint(name, args)
    op_fp = sem_fp or signature
    if key not in evidence:
        shapes = getattr(agent, "_work_mutation_shapes", {})
        shapes[signature] = shapes.get(signature, 0) + 1
        agent._work_mutation_shapes = shapes
        if sem_fp:
            sem_families = getattr(agent, "_work_mutation_sem_families", {})
            sem_families[sem_fp] = sem_families.get(sem_fp, 0) + 1
            agent._work_mutation_sem_families = sem_families
    record = mutation_identity(name, args, task_id=getattr(agent, "_canonical_work_task_id", None),
                               run_id=getattr(agent, "_canonical_work_run_id", None))
    record.update({"evidence_ref": ref, "verifier_status": "not_verified",
                   "persisted": None, "status": "uncertain" if failed else "executed_unverified",
                   "operation_fingerprint": op_fp,
                   "semantic_fingerprint": sem_fp})
    evidence[key] = record
    agent._work_mutation_evidence = evidence
    store.store(owner, "mutation_" + key + ".json", record)
    if failed:
        return
    if key not in completed:
        completed[key] = ref
        agent._work_completed_mutations = completed


def prepare_mutation(agent, name, args):
    """Record mutable dispatch before crossing I/O, using existing artifacts.

    A process death before acknowledgement leaves uncertainty durably visible.
    Compiled steps already have their own WorkItem dispatch checkpoints.
    """
    from workstation.task_compiler import durable_execution_active
    if durable_execution_active() or 'work_execute' not in getattr(agent, 'valid_tool_names', ()) or name == 'work_execute':
        return
    if name == 'tool_call':
        name, args = args.get('name', ''), args.get('arguments', {})
    if tool_effect(name) not in WRITE_EFFECTS:
        return
    # Middleware may have rewritten arguments after the model's admission check.
    # Recheck the final operation before persisting or crossing mutable I/O.
    from types import SimpleNamespace
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    decision = decisions_for_calls(agent, [call])[0]
    if decision in {CompilationDecision.REQUIRE_COMPILE, CompilationDecision.REQUIRE_HUMAN}:
        raise RuntimeError(decision.value + ': final mutable operation requires compiler or human review')
    task_id, run_id = getattr(agent, '_canonical_work_task_id', None), getattr(agent, '_canonical_work_run_id', None)
    if task_id:
        from hermes_cli import kanban_db
        conn = kanban_db.connect()
        try:
            task = kanban_db.get_task(conn, task_id)
            if not task or task.current_run_id != run_id or task.status in {'done', 'cancelled'}:
                raise RuntimeError('stale_task_run: mutation authority no longer belongs to this run')
        finally:
            conn.close()
    from workstation.artifacts import ArtifactStore
    from workstation.recipes import sanitize
    key = call_key(name, args)
    owner = agent._conversation_root_id() or agent.session_id
    store = ArtifactStore()
    uri = f'artifact://tasks/{owner}/mutation_{key}.json'
    if store.resolve_ref(uri):
        prior = store.read_json(uri)
        if prior.get('status') == 'uncertain':
            raise RuntimeError('uncertain_mutation_requires_review')
    record = sanitize(mutation_identity(name, args, task_id=task_id, run_id=run_id))
    from workstation.execution_policy import semantic_operation_fingerprint
    signature = structural_signature(name, args)
    sem_fp = semantic_operation_fingerprint(name, args)
    op_fp = sem_fp or signature
    record.update({'status': 'uncertain', 'persisted': None, 'verifier_status': 'pending',
                   'operation_fingerprint': op_fp,
                   'semantic_fingerprint': sem_fp})
    store.store(owner, 'mutation_' + key + '.json', record)
    evidence = getattr(agent, '_work_mutation_evidence', {})
    if key not in evidence:
        shapes = getattr(agent, '_work_mutation_shapes', {})
        shapes[signature] = shapes.get(signature, 0) + 1
        agent._work_mutation_shapes = shapes
    evidence[key] = record
    agent._work_mutation_evidence = evidence
