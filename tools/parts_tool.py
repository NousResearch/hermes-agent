"""
Parts Tool - Dynamic Parts Management for Hermes Agent

Provides tools for creating, retrieving, updating, and managing Dynamic Parts.
Dynamic Parts are persistent perspectives that can bid for attention and evolve over time.
"""

import logging
import sys
import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Avoid triggering tools/__init__.py by creating minimal modules inline

# Create Part model inline
@dataclass
class _SuggestionResult:
    predicted_result: str
    predicted_result_confidence: str
    predicted_result_timeframe_seconds: int
    your_suggestion: str
    timestamp: str
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class _OriginatingEvent:
    timestamp: str
    result: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Part:
    name: str
    description: str
    suggestions_and_results: List[_SuggestionResult] = field(default_factory=list)
    core_part: bool = False
    needs_evaluation: bool = False
    originating_event: Optional[_OriginatingEvent] = None
    conclude_when: str = ''
    emotion: str = ''
    intensity: str = ''
    personality: str = ''
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
        # Handle both dict and object formats for suggestions
        data['suggestions_and_results'] = [
            s.to_dict() if hasattr(s, 'to_dict') else s 
            for s in self.suggestions_and_results
        ]
        if self.originating_event:
            data['originating_event'] = self.originating_event.to_dict() if hasattr(self.originating_event, 'to_dict') else self.originating_event
        else:
            data['originating_event'] = None
        return data

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()

    def add_suggestion_result(self, suggestion: _SuggestionResult):
        self.suggestions_and_results.append(suggestion)
        self.updated_at = datetime.now().isoformat()

SuggestionResult = _SuggestionResult
OriginatingEvent = _OriginatingEvent

# Create storage class inline
class PartsStorage:
    PARTS_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "parts"
    PARTS_FILE = PARTS_DIR / "parts.json"

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or self.PARTS_FILE
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write_empty()

    def _write_empty(self):
        self._write({"parts": [], "version": "1.0"})

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"parts": [], "version": "1.0"}

    def _write(self, data: Dict[str, Any]):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_part(self, part: Part) -> Part:
        data = self._read()
        data["parts"].append(part.to_dict())
        self._write(data)
        return part

    def get_part(self, part_id: str) -> Optional[Part]:
        data = self._read()
        for p in data["parts"]:
            if p.get("id") == part_id:
                return Part(**p)
        return None

    def get_part_by_name(self, name: str) -> Optional[Part]:
        data = self._read()
        for p in data["parts"]:
            if p.get("name", "").lower() == name.lower():
                return Part(**p)
        return None

    def update_part(self, part_id: str, updates: Dict[str, Any]) -> Optional[Part]:
        data = self._read()
        for i, p in enumerate(data["parts"]):
            if p.get("id") == part_id:
                part = Part(**p)
                part.update(**updates)
                data["parts"][i] = part.to_dict()
                self._write(data)
                return part
        return None

    def delete_part(self, part_id: str) -> bool:
        data = self._read()
        original = len(data["parts"])
        data["parts"] = [p for p in data["parts"] if p.get("id") != part_id]
        if len(data["parts"]) < original:
            self._write(data)
            return True
        return False

    def list_parts(self, include_archived: bool = False) -> List[Part]:
        data = self._read()
        parts = [Part(**p) for p in data["parts"]]
        if not include_archived:
            parts = [p for p in parts if not p.archived]
        return sorted(parts, key=lambda x: x.created_at, reverse=True)

    def archive_part(self, part_id: str) -> Optional[Part]:
        return self.update_part(part_id, {"archived": True})

    def unarchive_part(self, part_id: str) -> Optional[Part]:
        return self.update_part(part_id, {"archived": False})

    def search_parts(self, query: str) -> List[Part]:
        data = self._read()
        q = query.lower()
        results = []
        for p in data["parts"]:
            if p.get("archived"):
                continue
            part = Part(**p)
            if q in part.name.lower() or q in part.description.lower():
                results.append(part)
        return results

    def get_due_evaluations(self) -> List[Part]:
        return []

    def get_stats(self) -> Dict[str, Any]:
        parts = self.list_parts(include_archived=True)
        active = [p for p in parts if not p.archived]
        return {
            "total_parts": len(parts),
            "active_parts": len(active),
            "archived_parts": len(parts) - len(active),
            "parts_needing_evaluation": 0,
            "core_parts": 0,
        }


def get_parts_storage() -> PartsStorage:
    return PartsStorage()

logger = logging.getLogger(__name__)

_storage: Optional[PartsStorage] = None


def _get_storage() -> PartsStorage:
    global _storage
    if _storage is None:
        _storage = get_parts_storage()
    return _storage


def parts_list(include_archived: bool = False) -> Dict[str, Any]:
    """
    List all active parts.
    
    Args:
        include_archived: Whether to include archived parts in the list.
    
    Returns:
        Dictionary with list of parts and metadata.
    """
    storage = _get_storage()
    parts = storage.list_parts(include_archived=include_archived)
    stats = storage.get_stats()
    
    parts_data = []
    for part in parts:
        parts_data.append({
            "id": part.id,
            "name": part.name,
            "description": part.description[:100] + "..." if len(part.description) > 100 else part.description,
            "core_part": part.core_part,
            "archived": part.archived,
            "created_at": part.created_at,
            "triggers": part.triggers[:3],
            "wants": part.wants[:2],
        })
    
    return {
        "success": True,
        "parts": parts_data,
        "stats": stats,
    }


def parts_get(part_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a specific part by ID or name.
    
    Args:
        part_id: The unique ID of the part.
        name: The name of the part.
    
    Returns:
        The part data or error message.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    return {
        "success": True,
        "part": part.to_dict(),
    }


def parts_create(
    name: str,
    description: str,
    triggers: Optional[List[str]] = None,
    wants: Optional[List[str]] = None,
    phrases: Optional[List[str]] = None,
    personality: Optional[str] = None,
    emotion: Optional[str] = None,
    intensity: Optional[str] = None,
    core_part: bool = False,
    originating_event_result: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new Dynamic Part.
    
    Args:
        name: The name/identifier of the part (e.g., "Fear of Dogs").
        description: A first-person description of what this part is about.
        triggers: List of triggers that activate this part.
        wants: List of things this part wants.
        phrases: Characteristic phrases this part says.
        personality: Personality description for how this part speaks.
        emotion: The emotion this part represents.
        intensity: The intensity level (Low, Medium, High).
        core_part: Whether this is a core/primary part.
        originating_event_result: The event that caused this part to be created.
    
    Returns:
        The created part data.
    """
    storage = _get_storage()
    
    existing = storage.get_part_by_name(name)
    if existing:
        return {"success": False, "error": f"Part with name '{name}' already exists"}
    
    orig_event = None
    if originating_event_result:
        from datetime import datetime
        orig_event = OriginatingEvent(
            timestamp=datetime.now().isoformat(),
            result=originating_event_result
        )
    
    part = Part(
        name=name,
        description=description,
        triggers=triggers or [],
        wants=wants or [],
        phrases=phrases or [],
        personality=personality or "",
        emotion=emotion or "",
        intensity=intensity or "Medium",
        core_part=core_part,
        originating_event=orig_event,
    )
    
    created = storage.create_part(part)
    
    return {
        "success": True,
        "part": created.to_dict(),
        "message": f"Created part '{name}' with ID {created.id}",
    }


def parts_update(
    part_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    triggers: Optional[List[str]] = None,
    wants: Optional[List[str]] = None,
    phrases: Optional[List[str]] = None,
    personality: Optional[str] = None,
    emotion: Optional[str] = None,
    intensity: Optional[str] = None,
    conclude_when: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing part.
    
    Args:
        part_id: The ID of the part to update.
        name: The name (if searching by name instead of ID).
        description: New description.
        triggers: New triggers list.
        wants: New wants list.
        phrases: New phrases list.
        personality: New personality description.
        emotion: New emotion.
        intensity: New intensity.
        conclude_when: Condition for archiving.
    
    Returns:
        The updated part data.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    updates = {}
    if description is not None:
        updates["description"] = description
    if triggers is not None:
        updates["triggers"] = triggers
    if wants is not None:
        updates["wants"] = wants
    if phrases is not None:
        updates["phrases"] = phrases
    if personality is not None:
        updates["personality"] = personality
    if emotion is not None:
        updates["emotion"] = emotion
    if intensity is not None:
        updates["intensity"] = intensity
    if conclude_when is not None:
        updates["conclude_when"] = conclude_when
    
    updated = storage.update_part(part.id, updates)
    
    if not updated:
        return {"success": False, "error": "Failed to update part"}
    
    return {
        "success": True,
        "part": updated.to_dict(),
        "message": f"Updated part '{updated.name}'",
    }


def parts_delete(part_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete a part permanently.
    
    Args:
        part_id: The ID of the part to delete.
        name: The name (if searching by name instead of ID).
    
    Returns:
        Success or error message.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    deleted = storage.delete_part(part.id)
    
    return {
        "success": deleted,
        "message": f"Deleted part '{part.name}'" if deleted else "Failed to delete part",
    }


def parts_archive(part_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Archive a part (soft delete - part remains but inactive).
    
    Args:
        part_id: The ID of the part to archive.
        name: The name (if searching by name instead of ID).
    
    Returns:
        The archived part data.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    archived = storage.archive_part(part.id)
    
    if not archived:
        return {"success": False, "error": "Failed to archive part"}
    
    return {
        "success": True,
        "part": archived.to_dict(),
        "message": f"Archived part '{archived.name}'",
    }


def parts_unarchive(part_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Unarchive a part (restore from archive).
    
    Args:
        part_id: The ID of the part to unarchive.
        name: The name (if searching by name instead of ID).
    
    Returns:
        The restored part data.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    unarchived = storage.unarchive_part(part.id)
    
    if not unarchived:
        return {"success": False, "error": "Failed to unarchive part"}
    
    return {
        "success": True,
        "part": unarchived.to_dict(),
        "message": f"Restored part '{unarchived.name}' from archive",
    }


def parts_search(query: str) -> Dict[str, Any]:
    """
    Search parts by name, description, or triggers.
    
    Args:
        query: Search query string.
    
    Returns:
        List of matching parts.
    """
    storage = _get_storage()
    parts = storage.search_parts(query)
    
    return {
        "success": True,
        "query": query,
        "results": [p.to_dict() for p in parts],
        "count": len(parts),
    }


def parts_get_bids(context: str, max_bids: int = 5) -> Dict[str, Any]:
    """
    Get active bids from parts based on current context.
    
    This is the core of the Dynamic Parts system - it checks which parts
    are activated by the current conversation context and returns their bids.
    
    Args:
        context: The current conversation or context to check against triggers.
        max_bids: Maximum number of bids to return (default 5).
    
    Returns:
        List of active bids with urgency scores.
    """
    try:
        # Use standalone bid_engine
        import importlib.util
        from pathlib import Path
        _bid_engine_path = Path(__file__).parent / "parts" / "bid_engine.py"
        spec = importlib.util.spec_from_file_location("bid_engine", _bid_engine_path)
        _bid_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bid_mod)
        
        storage = _get_storage()
        parts = storage.list_parts(include_archived=False)
        
        bid_engine = _bid_mod.get_bid_engine()
        bids = bid_engine.get_active_bids(parts, context, max_bids)
        
        return {
            "success": True,
            "context": context[:200] + "..." if len(context) > 200 else context,
            "bids": [b.to_dict() for b in bids],
            "count": len(bids),
            "summary": bid_engine.get_bids_summary(bids),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "bids": [],
        }


def parts_semantic_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search parts using semantic vector similarity.
    
    Requires qdrant-client and an embedding model (sentence-transformers or OpenAI).
    
    Args:
        query: Natural language query to find relevant parts.
        limit: Maximum number of results to return.
    
    Returns:
        List of semantically similar parts with relevance scores.
    """
    try:
        from tools.parts.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        
        if not vector_store.client:
            return {
                "success": False,
                "error": "Vector search not available. Install qdrant-client and an embedding model (sentence-transformers or OpenAI).",
                "results": [],
            }
        
        results = vector_store.search(query=query, limit=limit)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }


def parts_reindex_vectors() -> Dict[str, Any]:
    """
    Rebuild the vector index for all parts.
    
    Use this if vector search results seem outdated.
    
    Returns:
        Status of the reindexing operation.
    """
    try:
        from tools.parts.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        storage = _get_storage()
        
        if not vector_store.client:
            return {
                "success": False,
                "error": "Vector search not available.",
                "indexed": 0,
            }
        
        parts = storage.list_parts(include_archived=False)
        indexed = vector_store.rebuild_index(parts)
        
        return {
            "success": True,
            "indexed": indexed,
            "message": f"Indexed {indexed} parts",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "indexed": 0,
        }


def parts_evaluate(
    part_id: Optional[str] = None,
    name: Optional[str] = None,
    result: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a part's prediction by adding the actual result.
    
    Args:
        part_id: The ID of the part.
        name: The name (if searching by name instead of ID).
        result: The actual result that happened.
    
    Returns:
        The updated part data.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    if not part.suggestions_and_results:
        return {"success": False, "error": "No pending predictions to evaluate"}
    
    latest_suggestion = part.suggestions_and_results[-1]
    if latest_suggestion.result is not None:
        return {"success": False, "error": "Latest suggestion already has a result"}
    
    latest_suggestion.result = result
    part.needs_evaluation = False
    
    storage.update_part(part.id, {
        "suggestions_and_results": [s.to_dict() for s in part.suggestions_and_results],
        "needs_evaluation": False,
    })
    
    return {
        "success": True,
        "part": part.to_dict(),
        "message": f"Added result to part '{part.name}'",
    }


def parts_due_evaluations() -> Dict[str, Any]:
    """
    Get parts with predictions that are due for evaluation.
    
    Returns:
        List of parts with due evaluations.
    """
    storage = _get_storage()
    due_parts = storage.get_due_evaluations()
    
    return {
        "success": True,
        "due_evaluations": [p.to_dict() for p in due_parts],
        "count": len(due_parts),
    }


def parts_split(
    part_id: Optional[str] = None,
    name: Optional[str] = None,
    split_name_1: str = "",
    split_name_2: str = "",
    split_triggers_1: Optional[List[str]] = None,
    split_triggers_2: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Split a part into two separate parts.
    
    Use when a part becomes too broad or has conflicting contexts.
    
    Args:
        part_id: The ID of the part to split.
        name: The name (if searching by name instead of ID).
        split_name_1: Name for the first new part.
        split_name_2: Name for the second new part.
        split_triggers_1: Triggers for the first new part.
        split_triggers_2: Triggers for the second new part.
    
    Returns:
        The two new parts.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    # Create two new parts using local Part class
    part1 = Part(
        name=split_name_1 or f"{part.name} (1)",
        description=part.description,
        triggers=split_triggers_1 or part.triggers[:len(part.triggers)//2] if part.triggers else [],
        wants=part.wants[:len(part.wants)//2] if part.wants else [],
        phrases=part.phrases[:len(part.phrases)//2] if part.phrases else [],
        personality=part.personality,
        emotion=part.emotion,
        intensity=part.intensity,
    )
    
    part2 = Part(
        name=split_name_2 or f"{part.name} (2)",
        description=part.description,
        triggers=split_triggers_2 or part.triggers[len(part.triggers)//2:] if part.triggers else [],
        wants=part.wants[len(part.wants)//2:] if part.wants else [],
        phrases=part.phrases[len(part.phrases)//2:] if part.phrases else [],
        personality=part.personality,
        emotion=part.emotion,
        intensity=part.intensity,
    )
    
    # Archive original and create new parts
    storage.archive_part(part.id)
    storage.create_part(part1)
    storage.create_part(part2)
    
    return {
        "success": True,
        "original_part": part.name,
        "new_part_1": part1.to_dict(),
        "new_part_2": part2.to_dict(),
        "message": f"Split '{part.name}' into '{part1.name}' and '{part2.name}'",
    }


def parts_merge(
    part_id_1: Optional[str] = None,
    name_1: Optional[str] = None,
    part_id_2: Optional[str] = None,
    name_2: Optional[str] = None,
    merged_name: str = "",
) -> Dict[str, Any]:
    """
    Merge two parts into one.
    
    Use when two parts consistently say the same thing in the same situations.
    
    Args:
        part_id_1: ID of the first part.
        name_1: Name of the first part.
        part_id_2: ID of the second part.
        name_2: Name of the second part.
        merged_name: Name for the merged part.
    
    Returns:
        The merged part.
    """
    storage = _get_storage()
    
    part1 = None
    if part_id_1:
        part1 = storage.get_part(part_id_1)
    elif name_1:
        part1 = storage.get_part_by_name(name_1)
    
    part2 = None
    if part_id_2:
        part2 = storage.get_part(part_id_2)
    elif name_2:
        part2 = storage.get_part_by_name(name_2)
    
    if not part1:
        return {"success": False, "error": "First part not found"}
    if not part2:
        return {"success": False, "error": "Second part not found"}
    
    # Merge the parts - use local Part class
    merged_triggers = list(set(part1.triggers + part2.triggers))
    merged_wants = list(set(part1.wants + part2.wants))
    merged_phrases = list(set(part1.phrases + part2.phrases))
    
    merged = Part(
        name=merged_name or f"{part1.name} + {part2.name}",
        description=f"{part1.description}\n\n{part2.description}",
        triggers=merged_triggers,
        wants=merged_wants,
        phrases=merged_phrases,
        personality=part1.personality or part2.personality,
        emotion=part1.emotion or part2.emotion,
        intensity=max([part1.intensity, part2.intensity], key=lambda x: {"High": 3, "Medium": 2, "Low": 1}.get(x, 0)) if part1.intensity and part2.intensity else (part1.intensity or part2.intensity),
    )
    
    # Archive originals and create merged
    storage.archive_part(part1.id)
    storage.archive_part(part2.id)
    storage.create_part(merged)
    
    return {
        "success": True,
        "original_parts": [part1.name, part2.name],
        "merged_part": merged.to_dict(),
        "message": f"Merged '{part1.name}' and '{part2.name}' into '{merged.name}'",
    }


def parts_add_suggestion(
    part_id: Optional[str] = None,
    name: Optional[str] = None,
    predicted_result: str = "",
    predicted_result_confidence: str = "Medium",
    predicted_result_timeframe_seconds: int = 0,
    your_suggestion: str = "",
) -> Dict[str, Any]:
    """
    Add a suggestion/prediction to a part.
    
    This is used for the time-delayed evaluation system.
    
    Args:
        part_id: The ID of the part.
        name: The name (if searching by name instead of ID).
        predicted_result: What the part predicts will happen.
        predicted_result_confidence: Confidence level (Low, Medium, High).
        predicted_result_timeframe_seconds: Time in seconds until evaluation.
        your_suggestion: What the part recommends doing.
    
    Returns:
        The updated part.
    """
    storage = _get_storage()
    
    part = None
    if part_id:
        part = storage.get_part(part_id)
    elif name:
        part = storage.get_part_by_name(name)
    else:
        return {"success": False, "error": "Either part_id or name must be provided"}
    
    if not part:
        return {"success": False, "error": "Part not found"}
    
    # Use local SuggestionResult class
    suggestion = _SuggestionResult(
        predicted_result=predicted_result,
        predicted_result_confidence=predicted_result_confidence,
        predicted_result_timeframe_seconds=predicted_result_timeframe_seconds,
        your_suggestion=your_suggestion,
        timestamp=datetime.now().isoformat(),
    )
    
    part.add_suggestion_result(suggestion)
    part.needs_evaluation = True
    
    storage.update_part(part.id, {
        "suggestions_and_results": [s.to_dict() for s in part.suggestions_and_results],
        "needs_evaluation": True,
    })
    
    updated_part = storage.get_part(part.id)
    if not updated_part:
        return {"success": False, "error": "Part not found after update"}
    
    return {
        "success": True,
        "part": updated_part.to_dict(),
        "message": f"Added suggestion to '{part.name}'",
    }


def parts_stats() -> Dict[str, Any]:
    """
    Get statistics about the parts storage.
    
    Returns:
        Stats dictionary.
    """
    storage = _get_storage()
    stats = storage.get_stats()
    
    return {
        "success": True,
        "stats": stats,
    }


PARTS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parts_list",
            "description": "List all active Dynamic Parts. Use this to see what perspectives the bot has created.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_archived": {
                        "type": "boolean",
                        "description": "Whether to include archived parts in the list (default: false)",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_get",
            "description": "Get detailed information about a specific Dynamic Part.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The unique ID of the part"},
                    "name": {"type": "string", "description": "The name of the part"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_create",
            "description": "Create a new Dynamic Part. A part represents a persistent perspective that can bid for attention.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the part (e.g., 'Fear of Dogs')"},
                    "description": {
                        "type": "string", 
                        "description": "First-person description of this perspective (e.g., 'A protective part that holds trauma...')"
                    },
                    "triggers": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "What activates this part (e.g., 'Seeing a dog', 'Hearing barking')"
                    },
                    "wants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What this part wants (e.g., 'Avoid contact with dogs')"
                    },
                    "phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Characteristic phrases this part says"
                    },
                    "personality": {"type": "string", "description": "How this part speaks (e.g., 'Stubborn and alarmist...')"},
                    "emotion": {"type": "string", "description": "The emotion (e.g., 'Terror', 'Joy', 'Curiosity')"},
                    "intensity": {"type": "string", "description": "Intensity level (Low, Medium, High)"},
                    "core_part": {"type": "boolean", "description": "Whether this is a core/primary part", "default": False},
                    "originating_event_result": {
                        "type": "string",
                        "description": "What caused this part to be created"
                    },
                },
                "required": ["name", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_update",
            "description": "Update an existing Dynamic Part.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part to update"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                    "description": {"type": "string", "description": "New description"},
                    "triggers": {"type": "array", "items": {"type": "string"}, "description": "New triggers"},
                    "wants": {"type": "array", "items": {"type": "string"}, "description": "New wants"},
                    "phrases": {"type": "array", "items": {"type": "string"}, "description": "New phrases"},
                    "personality": {"type": "string", "description": "New personality"},
                    "emotion": {"type": "string", "description": "New emotion"},
                    "intensity": {"type": "string", "description": "New intensity"},
                    "conclude_when": {"type": "string", "description": "Condition for archiving"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_delete",
            "description": "Delete a Dynamic Part permanently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part to delete"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_archive",
            "description": "Archive a Dynamic Part (soft delete - part remains but becomes inactive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part to archive"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_unarchive",
            "description": "Restore a Dynamic Part from archive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part to restore"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_search",
            "description": "Search Dynamic Parts by name, description, or triggers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_get_bids",
            "description": "Get active bids from parts based on current context. This is the core of Dynamic Parts - it shows which parts are activated and what they're proposing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Current conversation or context to check against triggers"},
                    "max_bids": {"type": "integer", "description": "Maximum number of bids to return", "default": 5},
                },
                "required": ["context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_semantic_search",
            "description": "Search parts using semantic vector similarity. Finds parts based on meaning, not just keyword matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query to find relevant parts"},
                    "limit": {"type": "integer", "description": "Maximum results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_reindex_vectors",
            "description": "Rebuild the vector index for semantic search. Use if search results seem outdated.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_evaluate",
            "description": "Evaluate a part's prediction by adding the actual result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                    "result": {"type": "string", "description": "The actual result that happened"},
                },
                "required": ["result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_add_suggestion",
            "description": "Add a suggestion/prediction to a part for time-delayed evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                    "predicted_result": {"type": "string", "description": "What the part predicts will happen"},
                    "predicted_result_confidence": {"type": "string", "description": "Confidence level", "enum": ["Low", "Medium", "High"]},
                    "predicted_result_timeframe_seconds": {"type": "integer", "description": "Time in seconds until evaluation"},
                    "your_suggestion": {"type": "string", "description": "What the part recommends doing"},
                },
                "required": ["predicted_result", "your_suggestion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_split",
            "description": "Split a part into two separate parts. Use when a part becomes too broad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "The ID of the part to split"},
                    "name": {"type": "string", "description": "The name (if searching by name instead of ID)"},
                    "split_name_1": {"type": "string", "description": "Name for the first new part"},
                    "split_name_2": {"type": "string", "description": "Name for the second new part"},
                    "split_triggers_1": {"type": "array", "items": {"type": "string"}, "description": "Triggers for the first new part"},
                    "split_triggers_2": {"type": "array", "items": {"type": "string"}, "description": "Triggers for the second new part"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_merge",
            "description": "Merge two parts into one. Use when two parts consistently say the same thing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id_1": {"type": "string", "description": "ID of the first part"},
                    "name_1": {"type": "string", "description": "Name of the first part"},
                    "part_id_2": {"type": "string", "description": "ID of the second part"},
                    "name_2": {"type": "string", "description": "Name of the second part"},
                    "merged_name": {"type": "string", "description": "Name for the merged part"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_due_evaluations",
            "description": "Get parts with predictions that are due for evaluation.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parts_stats",
            "description": "Get statistics about the parts storage.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


PARTS_TOOL_FUNCTIONS = {
    "parts_list": parts_list,
    "parts_get": parts_get,
    "parts_create": parts_create,
    "parts_update": parts_update,
    "parts_delete": parts_delete,
    "parts_archive": parts_archive,
    "parts_unarchive": parts_unarchive,
    "parts_search": parts_search,
    "parts_get_bids": parts_get_bids,
    "parts_semantic_search": parts_semantic_search,
    "parts_reindex_vectors": parts_reindex_vectors,
    "parts_evaluate": parts_evaluate,
    "parts_add_suggestion": parts_add_suggestion,
    "parts_split": parts_split,
    "parts_merge": parts_merge,
    "parts_due_evaluations": parts_due_evaluations,
    "parts_stats": parts_stats,
}
