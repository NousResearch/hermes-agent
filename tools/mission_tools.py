"""Fred's shared mission ownership tools; all surfaces share Kanban's board database."""
from __future__ import annotations

import json
import os

from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli import mission_ownership as missions
from tools.registry import registry, tool_error


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required or [], "additionalProperties": False}}


def _check_fred() -> bool:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return str(get_active_profile_name() or "").casefold() == "fred"
    except Exception:
        return False


def _actor(kwargs: dict) -> tuple[str, str]:
    task_id = str(os.environ.get("HERMES_KANBAN_TASK", "")).strip()
    if task_id:
        return "task", task_id
    return "session", str(kwargs.get("session_id") or "").strip()


def _call(fn):
    def handler(args: dict, **kwargs):
        try:
            return json.dumps(fn(args or {}, **kwargs), ensure_ascii=False)
        except Exception as exc:
            return tool_error(f"Mission operation refused ({type(exc).__name__}): {exc}")
    return handler


def _list(_args: dict, **_kwargs):
    with connect_closing() as conn:
        return {"missions": missions.list_active(conn)}


def _show(args: dict, **_kwargs):
    with connect_closing() as conn:
        mission = missions.get(conn, str(args.get("mission_id", "")))
    if mission is None:
        raise missions.MissionError("mission not found")
    return mission


def _progress(args: dict, **kwargs):
    kind, oid = _actor(kwargs)
    with connect_closing() as conn:
        missions.record_progress(
            conn, mission_id=str(args["mission_id"]), owner_kind=kind, owner_id=oid,
            generation=int(args["generation"]), progress=args["progress"],
            child_ids=args.get("child_ids"), recovery=args.get("recovery"),
        )
    return {"ok": True, "mission_id": args["mission_id"]}


def _update(args: dict, **kwargs):
    kind, oid = _actor(kwargs)
    with connect_closing() as conn:
        version = missions.update_contract(
            conn, mission_id=str(args["mission_id"]), expected_owner_kind=kind,
            expected_owner_id=oid, expected_generation=int(args["generation"]),
            correction=str(args["correction"]), contract=args["contract"],
        )
    return {"ok": True, "contract_version": version}


def _handoff(args: dict, **kwargs):
    kind, oid = _actor(kwargs)
    with connect_closing() as conn:
        new_owner_id = str(args.get("new_owner_id") or "").strip()
        if args["new_owner_kind"] == "task" and not new_owner_id:
            title = str(args.get("target_title") or "").strip()
            body = str(args.get("target_body") or "").strip()
            assignee = str(args.get("target_assignee") or "").strip()
            if not title or not body or not assignee:
                raise missions.MissionError(
                    "task handoff needs new_owner_id for an existing triage card, or target_title, target_body, and target_assignee to create one"
                )
            if assignee.casefold() != "fred":
                raise missions.MissionError(
                    "a root-ownership task must run as Fred; create specialist work as a child task and keep the current mission owner"
                )
            mission = missions.get(conn, str(args["mission_id"]))
            if mission is None or (mission["owner_kind"], mission["owner_id"], mission["owner_generation"]) != (kind, oid, int(args["generation"])):
                raise missions.StaleOwnerError("caller is not the current mission owner")
            packet = {
                "mission_id": mission["id"], "original_request": mission["original_request"],
                "contract": mission["contract"], "contract_version": mission["contract_version"],
                "progress": mission["progress"], "children": mission["children"],
                "recovery": mission["recovery"],
            }
            body = "Authoritative root mission handoff packet:\n" + json.dumps(packet, ensure_ascii=False, indent=2) + "\n\n" + body
            from hermes_cli import kanban_db as kb
            new_owner_id = kb.create_task(
                conn, title=title, body=body, assignee=assignee,
                created_by="fred:mission_handoff", triage=True,
            )
        generation = missions.handoff(
            conn, mission_id=str(args["mission_id"]), expected_owner_kind=kind,
            expected_owner_id=oid, expected_generation=int(args["generation"]),
            new_owner_kind=str(args["new_owner_kind"]), new_owner_id=new_owner_id,
            progress=args["progress"], child_ids=args.get("child_ids"), recovery=args.get("recovery"),
        )
    return {"ok": True, "generation": generation,
            "owner": [args["new_owner_kind"], new_owner_id]}


def _complete(args: dict, **kwargs):
    kind, oid = _actor(kwargs)
    with connect_closing() as conn:
        missions.complete(
            conn, mission_id=str(args["mission_id"]), owner_kind=kind, owner_id=oid,
            generation=int(args["generation"]), contract_version=int(args["contract_version"]),
            evidence=args["evidence"],
        )
    return {"ok": True, "mission_id": args["mission_id"], "status": "completed"}


_TOOLS = (
    ("mission_list", _schema("mission_list", "List active Fred missions across interfaces. Listing never takes ownership.", {}), _call(_list), "🧭"),
    ("mission_show", _schema("mission_show", "Read a mission contract, owner, progress, child tasks and recovery details.",
        {"mission_id": {"type": "string"}}, ["mission_id"]), _call(_show), "🧭"),
    ("mission_progress", _schema("mission_progress", "Record progress and recovery details as the current mission owner.",
        {"mission_id": {"type": "string"}, "generation": {"type": "integer"},
         "progress": {"type": "object"}, "child_ids": {"type": "array", "items": {"type": "string"}},
         "recovery": {"type": "object"}}, ["mission_id", "generation", "progress"]), _call(_progress), "🧭"),
    ("mission_update", _schema("mission_update", "Append a user correction and replace the current mission contract. Old completion receipts become stale.",
        {"mission_id": {"type": "string"}, "generation": {"type": "integer"},
         "correction": {"type": "string"}, "contract": {"type": "object"}},
        ["mission_id", "generation", "correction", "contract"]), _call(_update), "🧭"),
    ("mission_handoff", _schema("mission_handoff", "Transfer root mission ownership. Task targets must be assigned triage cards; they are made runnable only in the same transaction as the transfer.",
        {"mission_id": {"type": "string"}, "generation": {"type": "integer"},
         "new_owner_kind": {"type": "string", "enum": ["session", "task"]},
         "new_owner_id": {"type": "string"}, "target_title": {"type": "string"},
         "target_body": {"type": "string"}, "target_assignee": {"type": "string"},
         "progress": {"type": "object"},
         "child_ids": {"type": "array", "items": {"type": "string"}}, "recovery": {"type": "object"}},
        ["mission_id", "generation", "new_owner_kind", "progress"]), _call(_handoff), "🧭"),
    ("mission_complete", _schema("mission_complete", "Complete the current mission against its current version. Implementation missions require a current code artifact hash and a successful check receipt.",
        {"mission_id": {"type": "string"}, "generation": {"type": "integer"},
         "contract_version": {"type": "integer"}, "evidence": {"type": "object"}},
        ["mission_id", "generation", "contract_version", "evidence"]), _call(_complete), "✅"),
)

for _name, _schema_obj, _handler, _emoji in _TOOLS:
    registry.register(name=_name, toolset="mission", schema=_schema_obj, handler=_handler,
                      emoji=_emoji, check_fn=_check_fred)
