"""Read-only cockpit/lineage projection over canonical Kanban state."""
import json

from hermes_cli import kanban_db
from workstation.journal import ExecutionJournal


def task_cockpit(conn, task_id: str) -> dict:
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise ValueError("Canonical task not found")
    run = kanban_db.latest_run(conn, task_id)
    metadata = run.metadata if run and isinstance(run.metadata, dict) else {}
    ws = metadata.get("workstation", {})
    outcome = ws.get("outcome", {})
    events = ExecutionJournal(task.id, task.session_id or "").read_events()
    if not outcome:
        outcome = next((e.metadata["outcome"] for e in reversed(events) if isinstance(e.metadata.get("outcome"), dict)), {})
    accepted = task.status == "done" and ws.get("acceptance_approved") is True and outcome.get("status") == "verified_completed"
    outcome_status = outcome.get("status", "uncertain")
    if outcome_status == "verified_completed" and not accepted:
        outcome_status = "uncertain"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    plans = []
    if "work_plans" in tables:
        for row in conn.execute("SELECT * FROM work_plans"):
            record = dict(row)
            plan_meta = json.loads(record.get("metadata") or "{}")
            if record["task_id"] == task_id or plan_meta.get("canonical_task_id") == task_id:
                plans.append({
                    "id": record["id"],
                    "status": record["status"],
                    "metadata": plan_meta,
                    "run_id": record.get("run_id"),
                    "execution_key": record.get("execution_key"),
                })
    card = None
    if "hybrid_card_delegations" in tables:
        row = conn.execute("SELECT human_card_id FROM hybrid_card_delegations WHERE agent_task_id=? ORDER BY attempt DESC LIMIT 1", (task_id,)).fetchone()
        if row:
            card = {"id": row[0]}
    evidence = outcome.get("evidence_refs", [])
    handoffs = [p["metadata"]["handoff"] for p in plans
                if p["metadata"].get("handoff", {}).get("status") == "waiting-for-human"]
    completed_items = total_items = 0
    if "work_items" in tables:
        for plan in plans:
            for status, count in conn.execute("SELECT status, COUNT(*) FROM work_items WHERE plan_id=? GROUP BY status", (plan["id"],)):
                total_items += count
                if status == "completed":
                    completed_items += count
    browser = sorted({p["metadata"]["browser_task_id"] for p in plans if p["metadata"].get("browser_task_id")})
    workers = sorted({e.metadata["worker_id"] for e in events if e.metadata.get("worker_id")})
    processes = sorted({e.metadata["process_id"] for e in events if e.metadata.get("process_id")})
    active_run_id = (run.id if run else None) or getattr(task, "current_run_id", None)
    execution_keys = sorted({p.get("execution_key") for p in plans if p.get("execution_key")})
    operation_ids = sorted({
        e.metadata.get("operation_id") for e in events if e.metadata.get("operation_id")
    } | ({outcome.get("operation_id")} if outcome.get("operation_id") else set()))
    lineage = {
        "task_id": task.id,
        "run_id": active_run_id,
        "execution_key": execution_keys[0] if len(execution_keys) == 1 else (execution_keys or None),
        "execution_keys": execution_keys,
        "operation_id": operation_ids[0] if len(operation_ids) == 1 else (operation_ids or None),
        "operation_ids": operation_ids,
        "workplan_id": plans[0]["id"] if plans else None,
        "human_card_id": card["id"] if card else None,
        "session_id": task.session_id,
        "human_card": card,
        "agent_task_id": task.id,
        "workplans": plans,
        "browser_tasks": browser,
        "workers": workers,
        "processes": processes,
        "evidence": evidence,
        "deliverables": outcome.get("deliverables", []),
        "outcome": outcome,
        "acceptance_approved": accepted,
        "acceptance_status": outcome_status,
    }
    return {"task_id": task.id, "objective": task.body or task.title, "human_card": card,
            "outcome_status": "waiting_for_human" if handoffs else outcome_status,
            "acceptance_approved": accepted,
            "verified_progress": completed_items / total_items if total_items else 0,
            "blockers": outcome.get("pending_items", []) + [h["reason"] for h in handoffs], "deliverables": lineage["deliverables"],
            "evidence_count": len(evidence), "workplans": plans, "browser_tasks": browser,
            "workers": workers, "current_activity": events[-1].message if events else None,
            "last_activity": events[-1].timestamp if events else None,
            "planned_handoff": outcome.get("planned_handoffs", []) + handoffs, "usage": outcome.get("metrics", {}),
            "next_action": "deliver_result" if accepted else "reconcile_or_verify",
            "lineage": lineage}
