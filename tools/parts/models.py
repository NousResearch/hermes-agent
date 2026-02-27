import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class SuggestionResult:
    predicted_result: str
    predicted_result_confidence: str
    predicted_result_timeframe_seconds: int
    your_suggestion: str
    timestamp: str
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuggestionResult":
        return cls(**data)


@dataclass
class OriginatingEvent:
    timestamp: str
    result: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OriginatingEvent":
        return cls(**data)


@dataclass
class Part:
    name: str
    description: str
    suggestions_and_results: List[SuggestionResult] = field(default_factory=list)
    core_part: bool = False
    needs_evaluation: bool = False
    originating_event: Optional[OriginatingEvent] = None
    conclude_when: str = ""
    emotion: str = ""
    intensity: str = ""
    personality: str = ""
    triggers: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    wants: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["suggestions_and_results"] = [s.to_dict() for s in self.suggestions_and_results]
        if self.originating_event:
            data["originating_event"] = self.originating_event.to_dict()
        else:
            data["originating_event"] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Part":
        suggestions = [SuggestionResult.from_dict(s) for s in data.get("suggestions_and_results", [])]
        orig_event = None
        if data.get("originating_event"):
            orig_event = OriginatingEvent.from_dict(data["originating_event"])
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data["description"],
            suggestions_and_results=suggestions,
            core_part=data.get("core_part", False),
            needs_evaluation=data.get("needs_evaluation", False),
            originating_event=orig_event,
            conclude_when=data.get("conclude_when", ""),
            emotion=data.get("emotion", ""),
            intensity=data.get("intensity", ""),
            personality=data.get("personality", ""),
            triggers=data.get("triggers", []),
            phrases=data.get("phrases", []),
            responses=data.get("responses", []),
            wants=data.get("wants", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            archived=data.get("archived", False),
        )

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()

    def add_suggestion_result(self, suggestion: SuggestionResult):
        self.suggestions_and_results.append(suggestion)
        self.updated_at = datetime.now().isoformat()

    def evaluate_prediction(self, result: str):
        if self.suggestions_and_results:
            latest = self.suggestions_and_results[-1]
            if latest.result is None:
                latest.result = result
                self.needs_evaluation = False
                self.updated_at = datetime.now().isoformat()
                return True
        return False


class PartsStore:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._parts: Dict[str, Part] = {}
        if storage_path and storage_path.exists():
            self._load()

    def _load(self):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for part_data in data.get("parts", []):
                part = Part.from_dict(part_data)
                self._parts[part.id] = part

    def _save(self):
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "parts": [p.to_dict() for p in self._parts.values()],
            "version": "1.0"
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create(self, part: Part) -> Part:
        self._parts[part.id] = part
        self._save()
        return part

    def get(self, part_id: str) -> Optional[Part]:
        return self._parts.get(part_id)

    def get_by_name(self, name: str) -> Optional[Part]:
        for part in self._parts.values():
            if part.name.lower() == name.lower():
                return part
        return None

    def list_all(self, include_archived: bool = False) -> List[Part]:
        parts = list(self._parts.values())
        if not include_archived:
            parts = [p for p in parts if not p.archived]
        return sorted(parts, key=lambda x: x.created_at, reverse=True)

    def update(self, part_id: str, **kwargs) -> Optional[Part]:
        part = self._parts.get(part_id)
        if part:
            part.update(**kwargs)
            self._save()
        return part

    def delete(self, part_id: str) -> bool:
        if part_id in self._parts:
            del self._parts[part_id]
            self._save()
            return True
        return False

    def archive(self, part_id: str) -> Optional[Part]:
        return self.update(part_id, archived=True)

    def unarchive(self, part_id: str) -> Optional[Part]:
        return self.update(part_id, archived=False)

    def get_due_evaluations(self) -> List[Part]:
        now = datetime.now()
        due_parts = []
        for part in self._parts.values():
            if part.archived:
                continue
            for suggestion in part.suggestions_and_results:
                if suggestion.result is None and suggestion.predicted_result_timeframe_seconds > 0:
                    try:
                        suggestion_time = datetime.fromisoformat(suggestion.timestamp)
                        elapsed = (now - suggestion_time).total_seconds()
                        if elapsed >= suggestion.predicted_result_timeframe_seconds:
                            due_parts.append(part)
                            break
                    except ValueError:
                        continue
        return due_parts

    def search_by_trigger(self, trigger: str) -> List[Part]:
        matching = []
        trigger_lower = trigger.lower()
        for part in self._parts.values():
            if part.archived:
                continue
            for t in part.triggers:
                if trigger_lower in t.lower():
                    matching.append(part)
                    break
        return matching

    def get_stats(self) -> Dict[str, Any]:
        all_parts = list(self._parts.values())
        active = [p for p in all_parts if not p.archived]
        archived = [p for p in all_parts if p.archived]
        due_evaluations = self.get_due_evaluations()
        
        return {
            "total_parts": len(all_parts),
            "active_parts": len(active),
            "archived_parts": len(archived),
            "parts_needing_evaluation": len(due_evaluations),
            "core_parts": len([p for p in active if p.core_part]),
        }
