from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from hermes_constants import get_hermes_home


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProceduralMemoryResult(str, Enum):
    FOUND = "found"
    MISS = "miss"
    BROKEN = "broken"


class ProcedureLifecycle(str, Enum):
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    RETIRED = "retired"


class MemoryKind(str, Enum):
    TASK_CONTEXT = "task_context"
    FACT = "fact"
    DECISION = "decision"
    USER_PREFERENCE = "user_preference"
    PROCEDURE = "procedure"
    EVIDENCE = "evidence"


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    kind: MemoryKind
    content: str
    workspace_id: str | None = None
    source: str | None = None
    created_at: str = field(default_factory=_utc_now)
    observed_at: str = field(default_factory=_utc_now)
    last_verified_at: str | None = None
    confidence: float = 1.0
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["record_type"] = "memory"
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=str(data["record_id"]),
            kind=MemoryKind(data.get("kind", MemoryKind.TASK_CONTEXT.value)),
            content=str(data.get("content", "")),
            workspace_id=data.get("workspace_id"),
            source=data.get("source"),
            created_at=str(data.get("created_at", _utc_now())),
            observed_at=str(data.get("observed_at", data.get("created_at", _utc_now()))),
            last_verified_at=data.get("last_verified_at"),
            confidence=float(data.get("confidence", 1.0)),
            expires_at=data.get("expires_at"),
        )

    def is_stale(self, *, as_of: str | None = None, max_age_seconds: float | None = None) -> bool:
        now = datetime.fromisoformat(as_of or _utc_now())
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if self.expires_at:
            expiry = datetime.fromisoformat(self.expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if now >= expiry:
                return True
        if max_age_seconds is None:
            return False
        observed = datetime.fromisoformat(self.observed_at)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return (now - observed).total_seconds() > max_age_seconds


@dataclass(slots=True)
class ProcedureStep:
    action: str
    target: str = ""
    value: str = ""
    expectation: str = ""
    required: bool = True
    anchor_type: str = "selector"
    fallback_anchors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "value": self.value,
            "expectation": self.expectation,
            "required": self.required,
            "anchor_type": self.anchor_type,
            "fallback_anchors": self.fallback_anchors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcedureStep:
        return cls(
            action=str(data.get("action", "")),
            target=str(data.get("target", "")),
            value=str(data.get("value", "")),
            expectation=str(data.get("expectation", "")),
            required=bool(data.get("required", True)),
            anchor_type=str(data.get("anchor_type", "selector")),
            fallback_anchors=list(data.get("fallback_anchors", [])),
        )

    def resolve_anchor(self, available_elements: list[dict[str, Any]]) -> str | None:
        """Resolve the best target identifier across multi-facet fallback anchors.

        Prioritizes semantic stability: testid -> role_name -> text content -> CSS selector.
        """
        if not available_elements:
            return self.target or None

        # 1. Search by testid if present
        for anchor in self.fallback_anchors:
            if anchor.get("type") == "testid":
                val = anchor.get("value", "").lower()
                for el in available_elements:
                    attrs = el.get("attributes", {})
                    tid = str(attrs.get("data-testid", attrs.get("data-test", el.get("testid", "")))).lower()
                    if val and val == tid:
                        return str(el.get("ref", el.get("selector", self.target)))

        # 2. Search by role_name
        for anchor in self.fallback_anchors:
            if anchor.get("type") == "role_name":
                val = anchor.get("value", "").lower()
                role_filter, _, name_filter = val.partition(":")
                for el in available_elements:
                    el_role = str(el.get("role", el.get("tag", ""))).lower()
                    el_name = str(el.get("name", el.get("text", ""))).lower()
                    if role_filter in el_role and name_filter in el_name:
                        return str(el.get("ref", el.get("selector", self.target)))

        # 3. Search by semantic text
        for anchor in self.fallback_anchors:
            if anchor.get("type") == "text":
                val = anchor.get("value", "").lower()
                for el in available_elements:
                    el_name = str(el.get("name", el.get("text", ""))).lower()
                    if val and val in el_name:
                        return str(el.get("ref", el.get("selector", self.target)))

        # 4. Fallback to original target selector
        for el in available_elements:
            el_sel = str(el.get("selector", el.get("path", "")))
            if self.target and self.target == el_sel:
                return str(el.get("ref", self.target))

        return self.target or None


@dataclass(slots=True)
class WebProcedure:
    id: str
    name: str
    site: str
    intent: str
    side_effect: str = "read_write"
    steps: list[ProcedureStep] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    success_count: int = 1
    failure_count: int = 0
    confidence: float = 1.0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    version: int = 1
    lifecycle: ProcedureLifecycle = ProcedureLifecycle.DISCOVERED
    validation_evidence: list[dict[str, Any]] = field(default_factory=list)
    validated_at: str | None = None
    promoted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "site": self.site,
            "intent": self.intent,
            "side_effect": self.side_effect,
            "steps": [s.to_dict() for s in self.steps],
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "lifecycle": self.lifecycle.value,
            "validation_evidence": self.validation_evidence,
            "validated_at": self.validated_at,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebProcedure:
        steps = [ProcedureStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            id=str(data.get("id", uuid4())),
            name=str(data.get("name", "Unnamed Procedure")),
            site=str(data.get("site", "")),
            intent=str(data.get("intent", "")),
            side_effect=str(data.get("side_effect", "read_write")),
            steps=steps,
            preconditions=list(data.get("preconditions", [])),
            postconditions=list(data.get("postconditions", [])),
            success_count=int(data.get("success_count", 1)),
            failure_count=int(data.get("failure_count", 0)),
            confidence=float(data.get("confidence", 1.0)),
            created_at=str(data.get("created_at", _utc_now())),
            updated_at=str(data.get("updated_at", _utc_now())),
            version=int(data.get("version", 1)),
            lifecycle=ProcedureLifecycle(data.get("lifecycle", ProcedureLifecycle.DISCOVERED.value)),
            validation_evidence=[item for item in data.get("validation_evidence", []) if isinstance(item, dict)],
            validated_at=data.get("validated_at"),
            promoted_at=data.get("promoted_at"),
        )


class ProceduralMemory:
    """V2 procedural web memory: discover -> run -> explore -> learn.

    Manages persistent reusable multi-step browser procedures indexed by
    domain and intent, with confidence estimation and decay/reinforcement.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        if storage_path is None:
            storage_path = get_hermes_home() / "workstation" / "memory" / "procedures.json"
        self.storage_path = Path(storage_path)
        self._procedures: dict[str, WebProcedure] = {}
        self._records: dict[str, MemoryRecord] = {}
        self._loaded_record_ids: set[str] = set()
        self._deleted_record_ids: set[str] = set()
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict):
                items = list(raw.get("procedures", [])) + list(raw.get("records", []))
            else:
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("record_type") == "memory" and item.get("record_id"):
                    record = MemoryRecord.from_dict(item)
                    self._records[record.record_id] = record
                elif "id" in item:
                    proc = WebProcedure.from_dict(item)
                    self._procedures[proc.id] = proc
            self._loaded_record_ids = set(self._records)
        except Exception as exc:
            # Corrupted storage resets to empty; atomic write prevents half-written state.
            self._procedures = {}
            self._records = {}
            self._loaded_record_ids.clear()
            self.load_error = str(exc)

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge on write: read external updates from disk without dropping our in-memory changes
        disk_records: dict[str, MemoryRecord] = {}
        disk_record_ids: set[str] = set()
        disk_read_ok = not self.storage_path.exists()
        if self.storage_path.exists():
            try:
                disk_raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
                disk_items = disk_raw if isinstance(disk_raw, list) else (
                    list(disk_raw.get("procedures", [])) + list(disk_raw.get("records", []))
                    if isinstance(disk_raw, dict) else []
                )
                for item in disk_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("record_type") == "memory" and item.get("record_id"):
                        disk_record = MemoryRecord.from_dict(item)
                        disk_record_ids.add(disk_record.record_id)
                        if disk_record.record_id not in self._deleted_record_ids:
                            disk_records[disk_record.record_id] = disk_record
                    elif "id" in item:
                        disk_id = str(item["id"])
                        if disk_id not in self._procedures:
                            self._procedures[disk_id] = WebProcedure.from_dict(item)
                disk_read_ok = True
            except Exception:
                pass

        # A writer may have loaded records before another writer compacted
        # them. Reconcile only IDs known to have come from the prior disk
        # snapshot; locally-created records remain eligible for this write.
        if disk_read_ok:
            for record_id in self._loaded_record_ids - disk_record_ids:
                self._records.pop(record_id, None)
        for record_id, record in disk_records.items():
            self._records.setdefault(record_id, record)

        temp_file = self.storage_path.with_suffix(".tmp")
        payload = [proc.to_dict() for proc in self._procedures.values()]
        payload.extend(record.to_dict() for record in self._records.values())
        temp_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(self.storage_path)
        self._loaded_record_ids = set(self._records)
        self._deleted_record_ids.clear()
        self.load_error = None

    def discover(self, site: str, goal: str) -> list[WebProcedure]:
        """Find procedures that match the given site and goal keywords."""
        site_norm = site.strip().lower().replace("https://", "").replace("http://", "").split("/")[0]
        goal_tokens = set(goal.strip().lower().split())

        candidates: list[tuple[float, WebProcedure]] = []
        for proc in self._procedures.values():
            proc_site = proc.site.strip().lower().replace("https://", "").replace("http://", "").split("/")[0]
            if site_norm and proc_site and site_norm not in proc_site and proc_site not in site_norm:
                continue

            # Compute intent relevance score
            proc_tokens = set(proc.intent.strip().lower().split()) | set(proc.name.strip().lower().split())
            overlap = len(goal_tokens & proc_tokens)
            relevance = (overlap / max(1, len(goal_tokens))) * proc.confidence

            if relevance > 0.1 or not goal_tokens:
                candidates.append((relevance, proc))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [proc for _, proc in candidates]

    def record_success(
        self,
        site: str,
        goal: str,
        steps: list[dict[str, Any] | ProcedureStep] | None = None,
        name: str | None = None,
        journal_id: str | None = None,
    ) -> WebProcedure:
        """Learn or reinforce a successful procedure."""
        parsed_steps: list[ProcedureStep] = []
        if steps:
            for s in steps:
                if isinstance(s, ProcedureStep):
                    parsed_steps.append(s)
                elif isinstance(s, dict):
                    parsed_steps.append(ProcedureStep.from_dict(s))

        # Check if an existing procedure matches site + intent closely
        existing = self.discover(site, goal)
        if existing and existing[0].confidence >= 0.5 and existing[0].lifecycle != ProcedureLifecycle.PROMOTED:
            proc = existing[0]
            proc.success_count += 1
            proc.confidence = min(1.0, proc.confidence + 0.05)
            proc.updated_at = _utc_now()
            if parsed_steps:
                proc.steps = parsed_steps
            self._persist()
            return proc

        # Otherwise create a new procedure
        proc_id = str(uuid4())
        proc_name = name or f"Workflow for {goal[:40]}"
        proc = WebProcedure(
            id=proc_id,
            name=proc_name,
            site=site,
            intent=goal,
            steps=parsed_steps,
            success_count=1,
            failure_count=0,
            confidence=0.8,
        )
        self._procedures[proc.id] = proc
        self._persist()
        return proc

    def record_failure(self, procedure_id: str, step_index: int = 0, reason: str = "") -> bool:
        """Record a failure during procedure replay, reducing its confidence."""
        proc = self._procedures.get(procedure_id)
        if not proc:
            return False
        proc.failure_count += 1
        proc.confidence = max(0.1, proc.confidence - 0.2)
        proc.updated_at = _utc_now()
        self._persist()
        return True

    def get_procedure(self, procedure_id: str) -> WebProcedure | None:
        return self._procedures.get(procedure_id)

    def update_procedure(self, procedure: WebProcedure) -> WebProcedure:
        if procedure.id not in self._procedures:
            raise KeyError(f"procedure '{procedure.id}' not found")
        self._procedures[procedure.id] = procedure
        self._persist()
        return procedure

    def record_memory(
        self,
        kind: MemoryKind,
        content: str,
        *,
        workspace_id: str | None = None,
        source: str | None = None,
        created_at: str | None = None,
        observed_at: str | None = None,
        last_verified_at: str | None = None,
        confidence: float = 1.0,
        expires_at: str | None = None,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("memory content is required")
        record = MemoryRecord(
            record_id=f"memory-{uuid4().hex}",
            kind=kind,
            content=content,
            workspace_id=workspace_id,
            source=source,
            created_at=created_at or _utc_now(),
            observed_at=observed_at or created_at or _utc_now(),
            last_verified_at=last_verified_at,
            confidence=min(1.0, max(0.0, confidence)),
            expires_at=expires_at,
        )
        self._records[record.record_id] = record
        self._persist()
        return record

    def compact_memory(
        self,
        *,
        max_records: int,
        workspace_id: str | None = None,
        kind: MemoryKind | None = None,
    ) -> int:
        """Retain the newest bounded set of records in an explicit scope.

        Compaction is opt-in so callers can choose retention policy without
        silently deleting durable user memory. Procedures are never affected;
        only records matching ``workspace_id`` and ``kind`` are considered.
        The deletion set is carried through the merge-on-write path so a
        concurrent writer cannot resurrect records intentionally compacted by
        this instance.
        """

        if max_records < 0:
            raise ValueError("max_records must be non-negative")

        candidates = [
            record
            for record in self._records.values()
            if (workspace_id is None or record.workspace_id == workspace_id)
            and (kind is None or record.kind == kind)
        ]
        if len(candidates) <= max_records:
            return 0

        newest = sorted(
            candidates,
            key=lambda record: (record.observed_at, record.created_at, record.record_id),
            reverse=True,
        )[:max_records]
        keep_ids = {record.record_id for record in newest}
        removed_ids = {
            record.record_id for record in candidates if record.record_id not in keep_ids
        }
        for record_id in removed_ids:
            self._records.pop(record_id, None)
        self._deleted_record_ids.update(removed_ids)
        self._persist()
        return len(removed_ids)

    def get_memory(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def list_memory(
        self,
        *,
        kind: MemoryKind | None = None,
        workspace_id: str | None = None,
        as_of: str | None = None,
        max_age_seconds: float | None = None,
        include_stale: bool = False,
    ) -> list[MemoryRecord]:
        records = []
        for record in self._records.values():
            if kind is not None and record.kind != kind:
                continue
            if workspace_id is not None and record.workspace_id != workspace_id:
                continue
            if not include_stale and record.is_stale(as_of=as_of, max_age_seconds=max_age_seconds):
                continue
            records.append(record)
        return records

    def snapshot(self, destination: Path | None = None) -> Path:
        """Create an explicit recoverable copy of the canonical memory projection."""
        if not self.storage_path.exists():
            self._persist()
        target = Path(destination or self.storage_path.with_suffix(self.storage_path.suffix + ".snapshot"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(self.storage_path, temp)
        temp.replace(target)
        return target

    def restore_snapshot(self, snapshot_path: Path) -> None:
        """Restore a validated snapshot without accepting malformed JSON."""
        source = Path(snapshot_path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, list) and not (
            isinstance(raw, dict) and ("procedures" in raw or "records" in raw)
        ):
            raise ValueError("memory snapshot has an invalid representation")
        temp = self.storage_path.with_suffix(self.storage_path.suffix + ".restore.tmp")
        shutil.copy2(source, temp)
        temp.replace(self.storage_path)
        self._procedures = {}
        self._records = {}
        self._loaded_record_ids.clear()
        self._deleted_record_ids.clear()
        self._load()

    def delete_procedure(self, procedure_id: str) -> bool:
        if procedure_id in self._procedures:
            del self._procedures[procedure_id]
            self._persist()
            return True
        return False

    def list_procedures(self, site: str | None = None) -> list[WebProcedure]:
        if not site:
            return list(self._procedures.values())
        site_norm = site.strip().lower()
        return [p for p in self._procedures.values() if site_norm in p.site.lower()]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "storage_path": str(self.storage_path),
            "procedure_count": len(self._procedures),
            "memory_count": len(self._records),
            "load_error": self.load_error,
        }
