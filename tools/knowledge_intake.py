"""Knowledge intake classifier and Obsidian router.

Tool name: capture_knowledge
"""

from __future__ import annotations

import json
from typing import Optional

from agent.knowledge_intake import (
    DESTINATIONS,
    classify_knowledge,
    list_intake_notes,
    resolve_vault_path,
    sync_obsidian_maps,
    write_intake_note,
)
from tools.registry import registry


def capture_knowledge(
    action: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    source_url: Optional[str] = None,
    source_title: Optional[str] = None,
    source_project: Optional[str] = None,
    preferred_destination: Optional[str] = None,
    destination: Optional[str] = None,
    write_to_obsidian: bool = True,
    enqueue_review: bool = True,
    sync_maps: bool = False,
    task_id: str = None,
) -> str:
    """Classify and route knowledge into the right review queue.

    Actions:
      classify  - return classification only
      capture   - classify and write a pending Obsidian intake note
      list      - list intake notes
      sync_maps - sync Skill/Agent/Relation maps into Obsidian
    """
    action = (action or "classify").strip().lower()

    if action == "sync_maps":
        try:
            result = sync_obsidian_maps(write_runtime_db=True)
            return json.dumps({"success": True, "action": action, "result": result})
        except Exception as exc:
            return json.dumps({"success": False, "action": action, "error": str(exc)})

    if action == "list":
        notes = list_intake_notes(destination=destination)
        return json.dumps({
            "success": True,
            "action": action,
            "count": len(notes),
            "items": notes,
        })

    if action not in {"classify", "capture"}:
        return json.dumps({
            "success": False,
            "error": "Invalid action. Valid actions: classify, capture, list, sync_maps",
        })

    if not title or not (content or source_url):
        return json.dumps({
            "success": False,
            "error": "classify/capture require title and either content or source_url",
        })

    if preferred_destination and preferred_destination not in DESTINATIONS:
        return json.dumps({
            "success": False,
            "error": f"Invalid preferred_destination: {preferred_destination}",
            "valid_destinations": sorted(DESTINATIONS),
        })

    classification = classify_knowledge(
        title=title,
        content=content or source_url or "",
        source_project=source_project or "",
        preferred_destination=preferred_destination,
        source_url=source_url,
        source_title=source_title,
    )
    result = {
        "success": True,
        "action": action,
        "classification": classification.to_dict(),
    }

    if action == "capture" and write_to_obsidian:
        try:
            note_path = write_intake_note(
                classification,
                content or "",
                source_url=source_url,
                source_title=source_title,
            )
            result["intake_note_path"] = str(note_path)
        except Exception as exc:
            result["success"] = False
            result["error"] = str(exc)
            return json.dumps(result)

        if classification.destination == "domain_knowledge" and enqueue_review:
            try:
                from tools.knowledge_review import review_knowledge

                review_result = review_knowledge(
                    action="add",
                    title=title,
                    content=content or source_url or "",
                    source_project=source_project or "",
                    target_domain=(classification.domains[0] if classification.domains else "backend"),
                    summary="Captured by Knowledge Intake Router",
                )
                result["review_queue"] = json.loads(review_result)
            except Exception as exc:
                result["review_queue"] = {"success": False, "error": str(exc)}

    if sync_maps:
        try:
            result["sync_maps"] = sync_obsidian_maps(write_runtime_db=True)
        except Exception as exc:
            result["sync_maps"] = {"success": False, "error": str(exc)}

    return json.dumps(result)


def check_requirements() -> bool:
    """The tool is available when the HermesAgent vault exists."""
    return resolve_vault_path().exists()


registry.register(
    name="capture_knowledge",
    toolset="knowledge",
    schema={
        "name": "capture_knowledge",
        "description": (
            "Classify raw knowledge as skill_candidate, agent_candidate, domain_knowledge, "
            "workspace_knowledge, playbook_candidate, or project_note; optionally write "
            "an Obsidian intake note and sync Skill/Agent/workspace relation maps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["classify", "capture", "list", "sync_maps"],
                    "description": "Operation to perform.",
                },
                "title": {"type": "string", "description": "Short title for the knowledge."},
                "content": {"type": "string", "description": "Raw markdown/text knowledge to classify."},
                "source_url": {"type": "string", "description": "Original URL source for lineage and duplicate detection."},
                "source_title": {"type": "string", "description": "Optional title of the source page/post."},
                "source_project": {"type": "string", "description": "Optional project slug where this came from."},
                "preferred_destination": {
                    "type": "string",
                    "enum": sorted(DESTINATIONS),
                    "description": "Optional override when the user already knows the destination.",
                },
                "destination": {
                    "type": "string",
                    "enum": sorted(DESTINATIONS),
                    "description": "Filter for list action.",
                },
                "write_to_obsidian": {"type": "boolean", "description": "For capture: write an intake note."},
                "enqueue_review": {"type": "boolean", "description": "For domain_knowledge: add to review queue."},
                "sync_maps": {"type": "boolean", "description": "Also sync Skill/Agent relation maps."},
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: capture_knowledge(
        action=args.get("action", "classify"),
        title=args.get("title"),
        content=args.get("content"),
        source_url=args.get("source_url"),
        source_title=args.get("source_title"),
        source_project=args.get("source_project"),
        preferred_destination=args.get("preferred_destination"),
        destination=args.get("destination"),
        write_to_obsidian=args.get("write_to_obsidian", True),
        enqueue_review=args.get("enqueue_review", True),
        sync_maps=args.get("sync_maps", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
