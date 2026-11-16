"""Derived checklist progress; completion still belongs to the authored task state."""


def _rooted(item_id: str, parents: dict[str, str]) -> bool:
    seen = set()
    while item_id in parents:
        if item_id in seen:
            return False
        seen.add(item_id)
        item_id = parents[item_id]
    return True


def todo_progress(todos: list[dict]) -> dict:
    """Partition phases and subtasks without counting a checked parent over unfinished work.

    Malformed replay entries are ignored, while cycles degrade to visible root phases.
    Cancellation removes only that item from counts, never its still-active descendants.
    """
    by_id = {item["id"]: item for item in todos if isinstance(item, dict)
             and isinstance(item.get("id"), str) and item["id"]
             and item.get("status") in {"pending", "in_progress", "completed", "cancelled"}}
    parents = {key: item["parent"] for key, item in by_id.items()
               if isinstance(item.get("parent"), str) and item["parent"] in by_id and item["parent"] != key}
    active = {key for key, item in by_id.items() if item["status"] in {"pending", "in_progress"}}
    unfinished = set(active)
    for key in active:
        seen = {key}
        while key in parents and parents[key] not in seen:
            key = parents[key]
            seen.add(key)
            unfinished.add(key)
    progress = {"phases": {"total": 0, "completed": 0},
                "subtasks": {"total": 0, "completed": 0}, "unresolved": len(active)}
    for key, item in by_id.items():
        if item["status"] == "cancelled":
            continue
        counts = progress["subtasks" if key in parents and _rooted(key, parents) else "phases"]
        counts["total"] += 1
        counts["completed"] += int(item["status"] == "completed" and key not in unfinished)
    return progress
