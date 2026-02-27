import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import importlib.util

# Avoid triggering tools/__init__.py by loading directly
_parts_models_path = Path(__file__).parent / "models.py"
spec = importlib.util.spec_from_file_location("parts_models", _parts_models_path)
_parts_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_parts_models)

Part = _parts_models.Part
SuggestionResult = _parts_models.SuggestionResult
OriginatingEvent = _parts_models.OriginatingEvent

logger = logging.getLogger(__name__)

PARTS_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "parts"
PARTS_FILE = PARTS_DIR / "parts.json"


class PartsStorage:
    """
    Persistent storage for Dynamic Parts.
    Provides JSON file-based CRUD operations with automatic backup.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or PARTS_FILE
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write_empty()

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted parts file: {e}")
            return {"parts": [], "version": "1.0"}
        except Exception as e:
            logger.error(f"Failed to read parts file: {e}")
            return {"parts": [], "version": "1.0"}

    def _write(self, data: Dict[str, Any]):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_empty(self):
        self._write({"parts": [], "version": "1.0"})

    def _backup(self):
        if self.storage_path.exists():
            backup_path = self.storage_path.with_suffix(".json.bak")
            import shutil
            shutil.copy2(self.storage_path, backup_path)

    def create_part(self, part: Part) -> Part:
        data = self._read()
        part.id = part.id or self._generate_id()
        part.created_at = part.created_at or datetime.now().isoformat()
        part.updated_at = part.updated_at or datetime.now().isoformat()
        
        data["parts"].append(part.to_dict())
        self._backup()
        self._write(data)
        logger.info(f"Created part: {part.name} ({part.id})")
        return part

    def get_part(self, part_id: str) -> Optional[Part]:
        data = self._read()
        for part_data in data["parts"]:
            if part_data.get("id") == part_id:
                return Part.from_dict(part_data)
        return None

    def get_part_by_name(self, name: str) -> Optional[Part]:
        data = self._read()
        for part_data in data["parts"]:
            if part_data.get("name", "").lower() == name.lower():
                return Part.from_dict(part_data)
        return None

    def update_part(self, part_id: str, updates: Dict[str, Any]) -> Optional[Part]:
        data = self._read()
        for i, part_data in enumerate(data["parts"]):
            if part_data.get("id") == part_id:
                part = Part.from_dict(part_data)
                part.update(**updates)
                data["parts"][i] = part.to_dict()
                self._backup()
                self._write(data)
                logger.info(f"Updated part: {part.name} ({part.id})")
                return part
        return None

    def delete_part(self, part_id: str) -> bool:
        data = self._read()
        original_len = len(data["parts"])
        data["parts"] = [p for p in data["parts"] if p.get("id") != part_id]
        if len(data["parts"]) < original_len:
            self._backup()
            self._write(data)
            logger.info(f"Deleted part: {part_id}")
            return True
        return False

    def list_parts(self, include_archived: bool = False) -> List[Part]:
        data = self._read()
        parts = [Part.from_dict(p) for p in data["parts"]]
        if not include_archived:
            parts = [p for p in parts if not p.archived]
        return sorted(parts, key=lambda x: x.created_at, reverse=True)

    def archive_part(self, part_id: str) -> Optional[Part]:
        return self.update_part(part_id, {"archived": True})

    def unarchive_part(self, part_id: str) -> Optional[Part]:
        return self.update_part(part_id, {"archived": False})

    def search_parts(self, query: str) -> List[Part]:
        data = self._read()
        query_lower = query.lower()
        results = []
        
        for part_data in data["parts"]:
            if part_data.get("archived"):
                continue
            part = Part.from_dict(part_data)
            
            if (query_lower in part.name.lower() or 
                query_lower in part.description.lower() or
                any(query_lower in t.lower() for t in part.triggers)):
                results.append(part)
        
        return results

    def get_due_evaluations(self) -> List[Part]:
        data = self._read()
        now = datetime.now()
        due_parts = []
        
        for part_data in data["parts"]:
            if part_data.get("archived"):
                continue
            part = Part.from_dict(part_data)
            
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

    def add_suggestion(self, part_id: str, suggestion: SuggestionResult) -> Optional[Part]:
        part = self.get_part(part_id)
        if part:
            part.add_suggestion_result(suggestion)
            self.update_part(part_id, {"suggestions_and_results": [s.to_dict() for s in part.suggestions_and_results]})
            return part
        return None

    def get_stats(self) -> Dict[str, Any]:
        parts = self.list_parts(include_archived=True)
        active = [p for p in parts if not p.archived]
        archived = [p for p in parts if p.archived]
        due = self.get_due_evaluations()
        
        return {
            "total_parts": len(parts),
            "active_parts": len(active),
            "archived_parts": len(archived),
            "parts_needing_evaluation": len(due),
            "core_parts": len([p for p in active if p.core_part]),
        }

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())


def get_parts_storage() -> PartsStorage:
    return PartsStorage()
