"""Bounded deterministic durable handoff and provider-only planner projection.

The persisted transcript is never edited. Only runtime-authenticated plan refs
with a matching canonical owner may enable this projection.
"""
import copy
import json

from workstation.recipes import digest


def serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_durable_handoff(store, plan_id):
    ledger = store.operational_ledger(plan_id)
    if ledger.get("found") is False:
        raise ValueError("Durable plan not found")
    plan = store.get_plan(plan_id)
    handoff = {k: ledger[k] for k in ("task_id", "plan_id", "objective_ref", "phase", "constraints", "recipe",
                "canary", "setup", "items", "finalize", "next_action") if k in ledger}
    handoff["recipe"] = {k: handoff.get("recipe", {})[k] for k in ("recipe_id", "status", "fingerprint")
                         if k in handoff.get("recipe", {})}
    handoff["constraints"] = {k: handoff.get("constraints", {})[k] for k in ("allowed_routes", "forbidden_routes")
                              if k in handoff.get("constraints", {})}
    if "items" not in handoff:
        summary = store.get_progress_summary(plan.task_id)
        handoff["items"] = {"completed": summary["completed"], "total": summary["total_items"],
                            "pending": ledger["pending"], "failed": ledger["failed"],
                            "needs_reasoning": ledger["blocker_count"], "uncertain": 0}
    handoff["blockers"] = ledger["blockers"][:3]
    handoff["blocker_count"] = ledger["blocker_count"]
    handoff["artifact_refs"] = ledger["artifacts"][:3]
    handoff["artifact_count"] = ledger["artifact_count"]
    handoff["latest_verified_state"] = {"completed_items": handoff["items"]["completed"],
                                         "canary_verified": handoff.get("canary", {}).get("verified", False)}
    # Constraints cannot be silently dropped. Oversized constraints get an
    # artifact ref to the already persisted objective, not a partial policy.
    if len(serialized(handoff).encode()) > 4096:
        handoff["constraints"] = {"source_ref": ledger["objective_ref"], "policy": "enforce_persisted_constraints"}
    if len(serialized(handoff).encode()) > 8192:
        raise ValueError("Durable handoff exceeds hard ceiling")
    return handoff


def authenticated_handoff(agent, messages):
    from agent.context_compressor import _validated_operational_refs, is_user_originated_turn
    from workstation.durable_tasks import DurableTaskStore
    owners = {getattr(agent, "session_id", None)}
    root = getattr(agent, "_conversation_root_id", None)
    if callable(root):
        owners.add(root())
    latest_user = next((n for n in range(len(messages)-1, -1, -1) if is_user_originated_turn(messages[n])), -1)
    for n in range(len(messages)-1, -1, -1):
        message = messages[n]
        for ref in reversed(_validated_operational_refs(message)):
            if ref["kind"] != "durable_work" or ref["source"] != "KanbanRun" or ref.get("owner_session_id") not in owners:
                continue
            store = DurableTaskStore()
            try:
                plan = store.get_plan(ref["id"])
                if plan is None or plan.id != ref["id"] or plan.session_id != ref["owner_session_id"]:
                    continue
                if plan.status == "completed" and n < latest_user:
                    continue  # A finished old task cannot shrink an unrelated new turn.
                return build_durable_handoff(store, plan.id)
            finally:
                store.close()
    return None


def planner_projection(messages, handoff):
    from agent.context_compressor import is_user_originated_turn, is_compaction_summary_message, user_originated_turn_view
    latest = next((n for n in range(len(messages)-1, -1, -1)
                   if is_user_originated_turn(messages[n]) and not messages[n].get("_durable_handoff")), None)
    if latest is None:
        return None
    result = [copy.deepcopy(m) for m in messages[:latest] if m.get("role") == "system"]
    user = copy.deepcopy(user_originated_turn_view(messages[latest]) or messages[latest])
    # A previous projection is never considered raw history or live user text.
    content = user.get("content", "")
    text = "[DURABLE HANDOFF — STATE ONLY]\n" + serialized(handoff)
    if isinstance(content, str):
        user["content"] = content + "\n\n" + text
    elif isinstance(content, list):
        user["content"] = [*content, {"type": "text", "text": text}]
    else:
        return None
    user["_durable_handoff"] = digest(handoff)
    result.append(user)
    covered_calls = set()
    for message in messages[latest+1:]:
        if message.get("_durable_handoff") or message.get("_compressed_summary") or is_compaction_summary_message(message):
            continue
        calls = message.get("tool_calls", [])
        if calls:
            covered_calls.update(c["id"] for c in calls if c.get("function", {}).get("name") == "work_execute")
            active_calls = [c for c in calls if c["id"] not in covered_calls]
            if not active_calls:
                continue
            if len(active_calls) != len(calls):
                message = {**message, "tool_calls": active_calls}
        if message.get("role") == "tool" and message.get("tool_call_id") in covered_calls:
            continue
        # Preserve a live unrelated exchange intact. Never truncate tool args.
        result.append(copy.deepcopy(message))
    return result


def project_for_provider(agent, messages, authenticated_messages=None):
    try:
        handoff = authenticated_handoff(agent, authenticated_messages or messages)
        if handoff is None:
            return messages
        projection = planner_projection(messages, handoff)
        if projection is None:
            return messages
        telemetry = getattr(agent, "_durable_context_metrics", {})
        from workstation.artifacts import ArtifactStore
        artifacts = ArtifactStore()
        metrics_ref = f"artifact://tasks/{handoff['task_id']}/metrics.json"
        prior = artifacts.read_json(metrics_ref) if artifacts.resolve_ref(metrics_ref) else {}
        if not telemetry:
            telemetry["duplicate_compaction_bytes_suppressed"] = prior.get("duplicate_compaction_bytes_suppressed", 0)
        size = len(serialized(handoff).encode())
        state_hash = digest(handoff)
        if getattr(agent, "_durable_handoff_hash", prior.get("last_handoff_hash")) == state_hash:
            telemetry["duplicate_compaction_bytes_suppressed"] = telemetry.get("duplicate_compaction_bytes_suppressed", 0) + size
        agent._durable_handoff_hash = state_hash
        telemetry.update({"handoff_bytes": size, "compaction_bytes": size,
                          "inline_context_bytes": len(serialized(projection).encode()),
                          "projection_source_bytes": len(serialized(messages).encode())})
        agent._durable_context_metrics = telemetry
        artifacts.store(handoff["task_id"], "handoff.json", handoff, schema="durable_handoff_v1")
        artifacts.store(handoff["task_id"], "metrics.json", {**prior,
            **{k: telemetry[k] for k in ("handoff_bytes", "compaction_bytes", "duplicate_compaction_bytes_suppressed")},
            "wire_context_bytes": telemetry["inline_context_bytes"], "last_handoff_hash": state_hash})
        return projection
    except Exception:
        # Missing/corrupt state never breaks ordinary conversation/compression.
        return messages


def durable_compaction(agent, messages):
    projection = project_for_provider(agent, messages)
    if projection is messages:
        return False
    agent._durable_provider_projection = projection
    return True
