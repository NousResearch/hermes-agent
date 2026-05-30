#!/usr/bin/env python3
"""Metadata-only Discord thread context diagnostic.

This tool is intentionally read-only and requires explicit local paths. It does
not call Discord, restart the gateway, inspect memory files, or query transcript
content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.discord_thread_mapping_diagnostic import inspect_discord_thread_mapping


def build_discord_thread_session_key(thread_id: str, *, chat_id: str | None = None) -> str:
    """Return the stable default Discord thread session key."""
    effective_chat_id = str(chat_id or thread_id)
    return f"agent:main:discord:thread:{effective_chat_id}:{thread_id}"


def inspect_discord_thread_context(
    *,
    thread_id: str,
    state_root: str | Path | None = None,
    sessions_json: str | Path | None = None,
    state_db: str | Path | None = None,
    chat_id: str | None = None,
    candidate_limit: int = 10,
) -> dict[str, Any]:
    """Return a JSON-serializable metadata-only Discord thread context report."""
    if state_root is not None:
        root = Path(state_root)
        sessions_path = Path(sessions_json) if sessions_json is not None else root / "sessions" / "sessions.json"
        state_db_path = Path(state_db) if state_db is not None else root / "state.db"
    else:
        sessions_path = Path(sessions_json) if sessions_json is not None else None
        state_db_path = Path(state_db) if state_db is not None else None

    if sessions_path is None or state_db_path is None:
        raise ValueError("Provide --state-root or both --sessions-json and --state-db.")

    effective_chat_id = str(chat_id or thread_id)
    expected_session_key = build_discord_thread_session_key(
        str(thread_id),
        chat_id=effective_chat_id,
    )
    mapping_report = inspect_discord_thread_mapping(
        sessions_json=sessions_path,
        state_db=state_db_path,
        session_key=expected_session_key,
        candidate_limit=candidate_limit,
    )

    exact_orphans = mapping_report.get("exact_orphan_sessions") or []
    candidate_orphans = mapping_report.get("candidate_orphan_sessions") or []
    missing_mapping_would_warn = (
        not bool(mapping_report.get("mapping", {}).get("exists"))
        and bool(exact_orphans)
    )

    return {
        "platform": "discord",
        "chat_type": "thread",
        "chat_id": effective_chat_id,
        "thread_id": str(thread_id),
        "expected_session_key": expected_session_key,
        "sessions_json": mapping_report["sessions_json"],
        "state_db": mapping_report["state_db"],
        "mapping": mapping_report["mapping"],
        "active_session": mapping_report["active_session"],
        "orphan_summary": {
            "exact_candidate_count": len(exact_orphans),
            "candidate_only_count": len(candidate_orphans),
        },
        "exact_orphan_sessions": exact_orphans,
        "candidate_orphan_sessions": candidate_orphans,
        "diagnostic": {
            "missing_mapping_diagnostic_would_fire": missing_mapping_would_warn,
        },
        "errors": mapping_report.get("errors", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Discord thread routing metadata without reading transcript content.",
    )
    parser.add_argument("--thread-id", required=True)
    parser.add_argument(
        "--chat-id",
        help="Optional chat_id override. Defaults to the Discord thread id.",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        help="Directory containing state.db and sessions/sessions.json.",
    )
    parser.add_argument(
        "--sessions-json",
        type=Path,
        help="Explicit sessions.json path. Use with --state-db or as a --state-root override.",
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        help="Explicit state.db path. Use with --sessions-json or as a --state-root override.",
    )
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Required safety marker; transcript and memory contents are never printed.",
    )
    args = parser.parse_args(argv)

    if not args.no_content:
        parser.error("--no-content is required; this tool never prints transcript or memory content.")

    try:
        report = inspect_discord_thread_context(
            thread_id=args.thread_id,
            chat_id=args.chat_id,
            state_root=args.state_root,
            sessions_json=args.sessions_json,
            state_db=args.state_db,
            candidate_limit=args.candidate_limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
