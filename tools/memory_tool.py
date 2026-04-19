#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from agent.memory_inspection import explain_archive, explain_conflict, explain_expired, explain_write
from agent.memory_policy import (
    ConflictDecision,
    WriteClass,
    assign_topic_key,
    classify_write_candidate,
    is_scoped_refinement,
    resolve_conflict,
    transition_freshness,
)
from agent.memory_records import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RecordStatus,
    normalize_legacy_entry,
    records_from_sidecar_payload,
    records_to_sidecar_payload,
)

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live prompt-projected state derived from
        the record store and legacy exports on disk.
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.records: List[MemoryRecord] = []
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """Load record-backed memory, project legacy exports, and capture the prompt snapshot."""
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        with self._file_lock(self._records_path()):
            self._reload_records(render_exports=True)
            self._system_prompt_snapshot = {
                "memory": self._render_block("memory", self.memory_entries),
                "user": self._render_block("user", self.user_entries),
            }

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety."""
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        fd = open(lock_path, "r+" if msvcrt else "a+")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _records_path(self) -> Path:
        return get_memory_dir() / "records.json"

    def _load_sidecar_records(self, raw_payload: Any, path: Path) -> Optional[List[MemoryRecord]]:
        if isinstance(raw_payload, list):
            payload = {"version": 1, "records": raw_payload}
        elif isinstance(raw_payload, dict):
            payload = raw_payload
            if "records" not in payload:
                logger.warning("Ignoring memory sidecar %s because dict payload is missing required 'records' key", path)
                return None
        else:
            logger.warning("Ignoring memory sidecar %s with unsupported payload type %s", path, type(raw_payload).__name__)
            return None

        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            logger.warning(
                "Ignoring memory sidecar %s because 'records' is %s instead of a list",
                path,
                type(raw_records).__name__,
            )
            return None

        records: List[MemoryRecord] = []
        skipped_records = 0
        version = payload.get("version", 1)
        for index, item in enumerate(raw_records):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping malformed memory sidecar record %s[%d]: expected object, got %s",
                    path,
                    index,
                    type(item).__name__,
                )
                skipped_records += 1
                continue
            try:
                records.extend(records_from_sidecar_payload({"version": version, "records": [item]}))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed memory sidecar record %s[%d]: %s", path, index, exc)
                skipped_records += 1

        if skipped_records and raw_records and not records:
            logger.warning("Falling back to legacy memory exports because %s contained no usable records", path)
            return None

        return records

    def _load_records(self) -> List[MemoryRecord]:
        path = self._records_path()
        if path.exists():
            try:
                raw_payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, IOError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load memory sidecar %s: %s", path, exc)
            else:
                sidecar_records = self._load_sidecar_records(raw_payload, path)
                if sidecar_records is not None:
                    return sidecar_records

        imported: List[MemoryRecord] = []
        created_at = _utc_now_iso()
        for target in ("memory", "user"):
            for entry in self._read_file(self._path_for(target)):
                record = normalize_legacy_entry(target=target, content=entry, created_at=created_at)
                record.metadata["target"] = target
                imported.append(record)
        return imported

    def _save_records(self, records: List[MemoryRecord]) -> None:
        payload = records_to_sidecar_payload(records)
        self._write_json_file(self._records_path(), payload)

    def _review_explanation_for_transition(
        self,
        original: MemoryRecord,
        transitioned: MemoryRecord,
    ) -> Optional[Dict[str, Any]]:
        if transitioned.status is original.status:
            return None
        if transitioned.status is RecordStatus.EXPIRED:
            return explain_expired(transitioned, "review_window_elapsed")
        if transitioned.status is RecordStatus.ARCHIVED:
            return explain_archive(transitioned, "archive_on_review")
        return None

    def _review_freshness(
        self,
        records: List[MemoryRecord],
        *,
        now: Optional[str] = None,
    ) -> tuple[List[MemoryRecord], bool, List[Dict[str, Any]]]:
        reviewed: List[MemoryRecord] = []
        changed = False
        explanations: List[Dict[str, Any]] = []
        review_time = now or _utc_now_iso()

        for record in records:
            transitioned = transition_freshness(record, now=review_time)
            reviewed.append(transitioned)
            if transitioned.to_dict() != record.to_dict():
                changed = True
            explanation = self._review_explanation_for_transition(record, transitioned)
            if explanation:
                explanations.append(explanation)

        deduped = self._deduplicate_records(reviewed)
        if len(deduped) != len(reviewed):
            changed = True
        return deduped, changed, explanations

    def _reload_records(self, *, render_exports: bool = False) -> List[Dict[str, Any]]:
        loaded_records = self._deduplicate_records(self._load_records())
        self.records, freshness_changed, review_explanations = self._review_freshness(loaded_records)
        if freshness_changed:
            self._save_records(self.records)
        self._sync_live_entries()
        if render_exports or freshness_changed:
            self._render_legacy_exports()
        return review_explanations

    def read(self, target: str) -> Dict[str, Any]:
        with self._file_lock(self._records_path()):
            review_explanations = self._reload_records()
        return self._success_response(target, "Current entries.", review_explanations=review_explanations)

    def save_to_disk(self, target: str | None = None):
        """Persist the record sidecar and projected legacy exports."""
        del target
        self._persist_state()

    def _persist_state(self) -> None:
        self._save_records(self.records)
        self._sync_live_entries()
        self._render_legacy_exports()

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _records_for_target(self, target: str) -> List[MemoryRecord]:
        return [record for record in self.records if self._target_for_record(record) == target]

    def _active_records_for_target(self, target: str) -> List[MemoryRecord]:
        return [record for record in self._records_for_target(target) if record.status is RecordStatus.ACTIVE]

    def _active_contents_for_target(self, target: str) -> List[str]:
        return self._project_contents_for_records(self._active_records_for_target(target))

    def _project_contents_for_records(self, records: List[MemoryRecord]) -> List[str]:
        contents: List[str] = []
        seen: set[str] = set()
        for record in records:
            if record.content in seen:
                continue
            seen.add(record.content)
            contents.append(record.content)
        return contents

    def _sync_live_entries(self) -> None:
        self.memory_entries = self._active_contents_for_target("memory")
        self.user_entries = self._active_contents_for_target("user")

    def _target_for_record(self, record: MemoryRecord) -> str:
        target = record.metadata.get("target")
        if target in {"memory", "user"}:
            return target
        if record.scope in {MemoryScope.OPERATOR, MemoryScope.PROFILE}:
            return "user"
        return "memory"

    def _deduplicate_records(self, records: List[MemoryRecord]) -> List[MemoryRecord]:
        deduped: List[MemoryRecord] = []
        seen: set[tuple[str, str, Optional[str], str, Optional[str]]] = set()
        for record in records:
            key = (
                self._target_for_record(record),
                record.content,
                record.topic_key,
                record.status.value,
                record.supersedes,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def _scope_for_target(self, target: str) -> MemoryScope:
        return MemoryScope.OPERATOR if target == "user" else MemoryScope.WORKSPACE

    def _source_kind_for_target(self, target: str) -> str:
        return "explicit_user_statement" if target == "user" else "tool_observation"

    def _validate_char_limit(self, target: str, projected_records: List[MemoryRecord], added_content: str, mode: str) -> Optional[Dict[str, Any]]:
        entries = self._project_contents_for_records(
            [record for record in projected_records if self._target_for_record(record) == target and record.status is RecordStatus.ACTIVE]
        )
        total = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        limit = self._char_limit(target)
        if total <= limit:
            return None

        current = self._char_count(target)
        if mode == "add":
            return {
                "success": False,
                "error": (
                    f"Memory at {current:,}/{limit:,} chars. "
                    f"Adding this entry ({len(added_content)} chars) would exceed the limit. "
                    f"Replace or remove existing entries first."
                ),
                "current_entries": self._entries_for(target),
                "usage": f"{current:,}/{limit:,}",
            }

        return {
            "success": False,
            "error": (
                f"Replacement would put memory at {total:,}/{limit:,} chars. "
                f"Shorten the new content or remove other entries first."
            ),
        }

    def _resolve_single_match(self, target: str, old_text: str) -> Dict[str, Any] | MemoryRecord:
        matches = [record for record in self._active_records_for_target(target) if old_text in record.content]
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}

        if len(matches) > 1:
            unique_texts = {record.content for record in matches}
            if len(unique_texts) > 1:
                previews = [content[:80] + ("..." if len(content) > 80 else "") for content in unique_texts]
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": previews,
                }

        return matches[0]

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new record-backed entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._records_path()):
            review_explanations = self._reload_records()

            if any(record.content == content for record in self._active_records_for_target(target)):
                return self._success_response(
                    target,
                    "Entry already exists (no duplicate added).",
                    review_explanations=review_explanations,
                )

            decision = classify_write_candidate(
                target=target,
                content=content,
                source_kind=self._source_kind_for_target(target),
                explicit_remember=content.lower().startswith("remember this:"),
                explicit_correction=False,
            )
            if decision.write_class is WriteClass.DO_NOT_WRITE:
                return {"success": False, "error": "Policy rejected this memory candidate as ephemeral or unsafe."}

            # The rule-based classifier still marks ambiguous content as MAY_WRITE so
            # passive/background extraction can defer it, but an explicit memory tool
            # invocation is itself an operator-selected write and should retain the
            # long-standing synchronous persistence behavior.
            scope = self._scope_for_target(target)
            record = MemoryRecord(
                record_id=f"rec-{uuid.uuid4()}",
                memory_type=MemoryType.PROFILE,
                scope=scope,
                topic_key=assign_topic_key(target=target, content=content, scope=scope),
                content=content,
                source="memory_tool:add",
                source_kind=self._source_kind_for_target(target),
                created_at=_utc_now_iso(),
                trust_tier=decision.trust_tier,
                salience_tier=decision.salience_tier,
                status=RecordStatus.ACTIVE,
                metadata={"target": target},
            )

            limit_error = self._validate_char_limit(target, self.records + [record], content, mode="add")
            if limit_error:
                return limit_error

            self.records.append(record)
            self.records = self._deduplicate_records(self.records)
            self._persist_state()

        return self._success_response(
            target,
            "Entry added.",
            explanations=[explain_write(record, decision.reason)],
            review_explanations=review_explanations,
        )

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Supersede the record containing old_text with a new record-backed replacement."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._records_path()):
            review_explanations = self._reload_records()

            match = self._resolve_single_match(target, old_text)
            if isinstance(match, dict):
                return match
            old_record = match

            topic_key = old_record.topic_key or assign_topic_key(
                target=target,
                content=new_content,
                scope=old_record.scope,
            )
            new_record = MemoryRecord(
                record_id=f"rec-{uuid.uuid4()}",
                memory_type=old_record.memory_type,
                scope=old_record.scope,
                topic_key=topic_key,
                content=new_content,
                source="memory_tool:replace",
                source_kind=old_record.source_kind,
                created_at=_utc_now_iso(),
                trust_tier=old_record.trust_tier,
                salience_tier=old_record.salience_tier,
                status=RecordStatus.ACTIVE,
                supersedes=old_record.record_id,
                metadata=dict(old_record.metadata),
            )

            if old_record.topic_key is None:
                superseded_record = MemoryRecord.from_dict(old_record.to_dict())
                superseded_record.status = RecordStatus.SUPERSEDED
                conflict = ConflictDecision(
                    winner=new_record,
                    loser=superseded_record,
                    loser_status=superseded_record.status,
                    reason="explicit_replace_supersedes_topicless_match",
                )
            else:
                explicit_correction = not is_scoped_refinement(old_record.content, new_content)
                conflict = resolve_conflict(old_record, new_record, explicit_correction=explicit_correction)
            projected_records: List[MemoryRecord] = []
            for record in self.records:
                if record.record_id == old_record.record_id:
                    if conflict.winner.record_id == old_record.record_id:
                        projected_records.append(conflict.winner)
                    elif conflict.loser.record_id == old_record.record_id:
                        projected_records.append(conflict.loser)
                    else:
                        projected_records.append(record)
                else:
                    projected_records.append(record)

            incoming_record = conflict.winner if conflict.winner.record_id == new_record.record_id else conflict.loser
            projected_records.append(incoming_record)

            limit_error = self._validate_char_limit(target, projected_records, new_content, mode="replace")
            if limit_error:
                return limit_error

            self.records = self._deduplicate_records(projected_records)
            self._persist_state()

        return self._success_response(
            target,
            "Entry replaced.",
            explanations=[explain_conflict(conflict)],
            review_explanations=review_explanations,
        )

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Archive the record containing old_text."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._records_path()):
            review_explanations = self._reload_records()

            match = self._resolve_single_match(target, old_text)
            if isinstance(match, dict):
                return match
            record = match

            updated_records: List[MemoryRecord] = []
            archived_record: Optional[MemoryRecord] = None
            for existing in self.records:
                if existing.record_id == record.record_id:
                    archived_record = MemoryRecord.from_dict(existing.to_dict())
                    archived_record.status = RecordStatus.ARCHIVED
                    archived_record.revision = existing.revision + 1
                    updated_records.append(archived_record)
                else:
                    updated_records.append(existing)

            self.records = self._deduplicate_records(updated_records)
            self._persist_state()

        return self._success_response(
            target,
            "Entry removed.",
            explanations=[explain_archive(archived_record or record, "operator_requested_archive")],
            review_explanations=review_explanations,
        )

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    def _success_response(
        self,
        target: str,
        message: str = None,
        explanations: Optional[List[Dict[str, Any]]] = None,
        review_explanations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        entries = self._active_contents_for_target(target)
        current = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        records = self._records_for_target(target)

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "records": [record.to_dict() for record in records],
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
            "record_count": len(records),
        }
        target_record_ids = {record.record_id for record in records}
        scoped_review_explanations = [
            explanation
            for explanation in (review_explanations or [])
            if explanation.get("record_id") in target_record_ids
        ]
        if scoped_review_explanations:
            resp["review_explanations"] = scoped_review_explanations
        if explanations:
            resp["explanations"] = explanations
        if message:
            resp["message"] = message
        return resp

    def _render_legacy_exports(self) -> None:
        self._write_file(self._path_for("memory"), self.memory_entries)
        self._write_file(self._path_for("user"), self.user_entries)

    def _render_block(self, target: str, entries: List[str]) -> str:
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        entries = [entry.strip() for entry in raw.split(ENTRY_DELIMITER)]
        return [entry for entry in entries if entry]

    @staticmethod
    def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
        MemoryStore._write_text_file(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        MemoryStore._write_text_file(path, content)

    @staticmethod
    def _write_text_file(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".mem_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Failed to write memory file {path}: {exc}")


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    elif action == "read":
        result = store.read(target)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove, read", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it), read (inspect current state and record statuses).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "read"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




