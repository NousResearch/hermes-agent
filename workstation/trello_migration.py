"""Non-destructive, idempotent Trello legacy reconciliation in canonical Kanban."""
import json

from hermes_cli import hybrid_kanban as hybrid
from hermes_cli import kanban_db


def reconcile_trello_legacy(conn, records: list[dict], *, dry_run: bool = True) -> list[dict]:
    from contextlib import nullcontext
    with nullcontext() if dry_run else kanban_db.write_txn(conn):
        return _reconcile_trello_legacy(conn, records, dry_run=dry_run)


def _reconcile_trello_legacy(conn, records: list[dict], *, dry_run: bool) -> list[dict]:
    """Records must carry explicit legacy identity and remote board/list/card IDs.

    No status or title inference grants provenance. Existing Agent Tasks remain
    untouched; human columns never auto-delegate migrated cards.
    """
    preview = []
    for record in records:
        task = kanban_db.get_task(conn, record["agent_task_id"])
        if task is None or task.created_by != "trello-sync":
            raise ValueError("legacy Trello provenance mismatch")
        if any(not record.get(k) for k in ("board_id", "list_id", "card_id", "board_name", "list_name")):
            raise ValueError("explicit Trello source identity is required")
        key = f"trello:{record['board_id']}:{record['card_id']}"
        existing = next((dict(r) for r in conn.execute("SELECT * FROM hybrid_cards")
                         if json.loads(r["metadata"] or "{}").get("idempotency_key") == key), None)
        preview.append({"agent_task_id": task.id, "idempotency_key": key,
                        "title": task.title, "description": task.body, "url": record.get("url"),
                        "action": "existing" if existing else "project", "human_card_id": existing["id"] if existing else None})
        if dry_run or existing:
            continue
        source = f"trello-board:{record['board_id']}"
        board = next((b for b in hybrid.list_boards(conn) if b["description"] == source), None)
        if board is None:
            board = hybrid.create_board(conn, name=record["board_name"], description=source, source="trello-migration")
        board = hybrid.get_board(conn, board["id"])
        list_source = f"trello-list:{record['board_id']}:{record['list_id']}"
        projected_list = conn.execute("SELECT column_id FROM hybrid_activity WHERE board_id=? AND kind='column_created' AND source=? ORDER BY id LIMIT 1",
                                      (board["id"], list_source)).fetchone()
        column = next((c for c in board["columns"] if projected_list and c["id"] == projected_list[0]), None)
        if column is None:
            column = hybrid.create_column(conn, board_id=board["id"], name=record["list_name"], source=list_source)
        card = hybrid.create_card(conn, board_id=board["id"], column_id=column["id"], title=task.title,
                                  description=task.body or "", metadata={"idempotency_key": key, "origin": record,
                                  "legacy_agent_task_id": task.id}, source="trello-migration")
        preview[-1]["human_card_id"] = card["id"]
    return preview


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON manifest with explicit Trello source identities")
    parser.add_argument("--board")
    parser.add_argument("--apply", action="store_true", help="Create human projections; default is preview")
    args = parser.parse_args()
    conn = kanban_db.connect(board=args.board)
    try:
        print(json.dumps(reconcile_trello_legacy(conn, json.loads(args.input.read_text(encoding="utf-8")),
                                                dry_run=not args.apply), ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
