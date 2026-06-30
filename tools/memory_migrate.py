#!/usr/bin/env python3
"""Migrate a profile's flat memory (MEMORY.md / USER.md) into the scoped graph.

Goal: move project/workflow/environment/safety/vocabulary facts out of the
small, always-loaded flat stores and into SCOPED_MEMORY.jsonl as typed nodes,
leaving only tiny global facts in the flat fallback. Then flip the config flags
so retrieval uses the graph.

Safety guarantees:
  - Back up MEMORY.md, USER.md, config.yaml, and any existing
    SCOPED_MEMORY.jsonl BEFORE writing anything. Never silently delete data.
  - Idempotent: re-running does not duplicate nodes (dedup by content) and
    does not re-migrate entries already moved out of the flat files.
  - Heuristic, deterministic classification — no model calls.

Usage (CLI):
    python3 -m tools.memory_migrate [--dry-run]

Tests monkeypatch get_memory_dir / get_config_path to a temp HERMES_HOME, so
this never touches a real profile during testing.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_config_path
from tools.memory_tool import (
    ENTRY_DELIMITER,
    SCOPED_MEMORY_FILENAME,
    MemoryStore,
    get_memory_dir,
)

# Project / entity names this workspace cares about. Lowercased for matching.
KNOWN_ENTITIES = [
    "Botson",
    "Rough Cut",
    "TermFleet",
    "Watchpost",
    "Hermes",
    "Codex",
    "Claude Code",
    "OpenCode",
    "MASTER_PLAN",
]

# Words that signal a safety / confirmation / destructive-guard rule.
_SAFETY_HINTS = (
    "confirm", "confirmation", "never", "must not", "do not", "don't",
    "danger", "destructive", "delete", "approval", "safe", "safety",
    "irreversible", "production",
)

# Words that signal a how-to / procedure / workflow.
_WORKFLOW_HINTS = (
    "deploy", "build", "test", "run", "workflow", "pipeline", "steps",
    "process", "release", "commit", "branch", "migrate", "ci", "table",
    "keep", "use ", "always", "preserve",
)

# Words that signal a global communication / user preference (kept flat).
_GLOBAL_PREF_HINTS = (
    "prefer", "concise", "brief", "terse", "tone", "respond", "reply",
    "language", "communicat", "style", "verbose", "short answer",
)


def _find_entities(text: str) -> List[str]:
    low = text.lower()
    found = []
    for name in KNOWN_ENTITIES:
        if name.lower() in low:
            found.append(name)
    return found


def _classify(text: str, from_user_file: bool) -> Tuple[str, List[str], bool]:
    """Return (node_type, entities, is_global) for a flat entry.

    is_global=True means the entry should remain in the flat fallback file
    (truly global; not project-scoped).
    """
    low = text.lower()
    entities = _find_entities(text)

    # Project-scoped safety rule.
    if entities and any(h in low for h in _SAFETY_HINTS):
        return "safety_rule", entities, False
    # Project-scoped workflow/procedure.
    if entities and any(h in low for h in _WORKFLOW_HINTS):
        return "workflow", entities, False
    # Project mention without a clear verb -> project node.
    if entities:
        return "project", entities, False

    # No project entity: communication preference stays flat & global.
    if any(h in low for h in _GLOBAL_PREF_HINTS):
        return "user_preference", entities, True

    # Generic non-project fact: keep as a global environment fallback.
    return "environment_fact", entities, True


def _backup_file(path: Path, ts: int) -> Optional[str]:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(bak)


def _load_flat_entries(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]


def _write_flat_entries(path: Path, entries: List[str]):
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def _enable_config(config_path: Path, char_limit: int) -> None:
    """Set memory.scoped_memory_enabled/char_limit in config.yaml (round-trip)."""
    data: Dict[str, Any] = {}
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    mem = data.get("memory")
    if not isinstance(mem, dict):
        mem = {}
        data["memory"] = mem
    mem["scoped_memory_enabled"] = True
    mem["scoped_memory_char_limit"] = char_limit
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def migrate_profile_to_scoped(
    char_limit: int = 4000,
    timestamp: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Migrate the active profile's flat memory into the scoped graph.

    Returns a report dict: {success, backups, scoped_path, migrated, kept_flat,
    config_path}.
    """
    mem_dir = get_memory_dir()
    config_path = get_config_path()
    mem_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp if timestamp is not None else int(time.time())

    memory_path = mem_dir / "MEMORY.md"
    user_path = mem_dir / "USER.md"
    scoped_path = mem_dir / SCOPED_MEMORY_FILENAME

    # 1) Back up everything that exists, before any write.
    backups: Dict[str, str] = {}
    for label, path in (
        ("MEMORY.md", memory_path),
        ("USER.md", user_path),
        ("config.yaml", config_path),
        (SCOPED_MEMORY_FILENAME, scoped_path),
    ):
        bak = _backup_file(path, ts)
        if bak:
            backups[label] = bak

    # 2) Load existing scoped nodes (for idempotency) + flat entries.
    existing_nodes = MemoryStore._load_scoped_raw(scoped_path)
    existing_contents = {str(n.get("content", "")).strip() for n in existing_nodes}
    existing_ids = {str(n.get("id")) for n in existing_nodes}

    memory_entries = _load_flat_entries(memory_path)
    user_entries = _load_flat_entries(user_path)

    nodes = list(existing_nodes)
    migrated: List[str] = []
    kept_memory: List[str] = []
    kept_user: List[str] = []
    id_helper = MemoryStore()  # only used for its deterministic id/slug helpers

    def _consider(entry: str, from_user_file: bool):
        node_type, entities, is_global = _classify(entry, from_user_file)
        if is_global:
            # Stays in the flat fallback file it came from.
            (kept_user if from_user_file else kept_memory).append(entry)
            return
        if entry.strip() in existing_contents:
            # Already migrated in a prior run — drop from flat (it lives in graph).
            return
        new_id = id_helper._generate_scoped_id(node_type, entities, entry, existing_ids)
        existing_ids.add(new_id)
        existing_contents.add(entry.strip())
        nodes.append({
            "id": new_id,
            "type": node_type,
            "content": entry.strip(),
            "entities": entities,
            "project_paths": [],
            "sources": [],
            "edges": [],
            "global": False,
        })
        migrated.append(entry)

    for entry in memory_entries:
        _consider(entry, from_user_file=False)
    for entry in user_entries:
        _consider(entry, from_user_file=True)

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "backups": backups,
            "scoped_path": str(scoped_path),
            "config_path": str(config_path),
            "would_migrate": migrated,
            "would_keep_flat": kept_memory + kept_user,
        }

    # 3) Persist: scoped graph, shrunken flat files, config flags.
    MemoryStore._write_scoped_nodes(scoped_path, nodes)
    _write_flat_entries(memory_path, kept_memory)
    _write_flat_entries(user_path, kept_user)
    _enable_config(config_path, char_limit)

    return {
        "success": True,
        "backups": backups,
        "scoped_path": str(scoped_path),
        "config_path": str(config_path),
        "migrated": migrated,
        "kept_flat": kept_memory + kept_user,
        "node_count": len(nodes),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate flat memory into the scoped graph.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument("--char-limit", type=int, default=4000, help="scoped_memory_char_limit to set.")
    args = parser.parse_args(argv)

    report = migrate_profile_to_scoped(char_limit=args.char_limit, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
