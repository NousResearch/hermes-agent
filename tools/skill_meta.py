#!/usr/bin/env python3
"""
Local skill confidence metadata management.

Manages per-skill metadata in ~/.hermes/skills/.skills_meta.json.
Provides:
  - Read/write confidence levels: untested → trial → verified → default
  - Usage counting and tracking
  - Default skill pool management (auto-loaded at session start)
  - Grade history audit trail

Usage:
  from tools.skill_meta import SkillMetaDB
  
  db = SkillMetaDB()
  
  # Register a skill
  db.register("obsidian")
  
  # Upgrade confidence
  db.grade("obsidian", "trial", reason="used 3 times, no issues")
  db.grade("obsidian", "verified", reason="stable across multiple projects")
  
  # Add to default pool (auto-loaded)
  db.add_to_default_pool("obsidian")
  
  # Check status
  meta = db.get("obsidian")
  print(meta.confidence)  # "verified"
  print(meta.default_skill_pool)  # True
  
  # List all skills
  for name, m in db.list_all():
      print(f"{name}: {m.confidence} (uses={m.usage_count})")
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Confidence levels — ordered from lowest to highest
CONFIDENCE_ORDER = ["untested", "trial", "verified", "default"]
VALID_LEVELS = set(CONFIDENCE_ORDER)


@dataclass
class SkillMeta:
    """Metadata for a single skill."""
    confidence: str = "untested"
    usage_count: int = 0
    last_used: Optional[str] = None
    default_skill_pool: bool = False
    grade_history: list = field(default_factory=list)
    notes: str = ""


class SkillMetaDB:
    """Thread-safe-ish JSON-backed metadata store for local skills."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Respect HERMES_HOME like the rest of the codebase does
            hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
            self.db_path = Path(hermes_home) / "skills" / ".skills_meta.json"
        else:
            self.db_path = Path(db_path)
        self._data: dict = {}
        self._load()

    # ── Core I/O ──────────────────────────────────────────────────

    def _load(self):
        """Load metadata from JSON file. Create default if missing."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = raw.get("skills", {})
            except (json.JSONDecodeError, KeyError):
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        """Write current state to JSON file atomically."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert all SkillMeta objects to dicts for JSON serialization
        serializable = {}
        for name, meta in self._data.items():
            if isinstance(meta, SkillMeta):
                serializable[name] = asdict(meta)
            else:
                serializable[name] = meta
        payload = {
            "_version": 1,
            "_description": "Per-skill metadata for local skills.",
            "skills": serializable,
        }
        # Atomic write via temp file
        tmp_path = self.db_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(self.db_path))

    # ── Registration ──────────────────────────────────────────────

    def register(self, name: str, *, reason: str = "new skill") -> SkillMeta:
        """Register a new skill. Returns the created SkillMeta."""
        name = self._normalize(name)
        if name not in self._data:
            meta = SkillMeta(confidence="untested", grade_history=[])
            self._data[name] = meta
            self._add_grade_history(name, None, "untested", reason, "system")
            self._save()
        return self._data[name]

    def get(self, name: str) -> Optional[SkillMeta]:
        """Get metadata for a skill. Returns None if not registered."""
        name = self._normalize(name)
        raw = self._data.get(name)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return SkillMeta(**raw)
        return raw

    # ── Confidence grading ────────────────────────────────────────

    def grade(
        self,
        name: str,
        level: str,
        *,
        reason: str = "",
        grader: str = "user",
    ) -> Optional[SkillMeta]:
        """
        Upgrade (or downgrade) a skill's confidence level.
        
        level must be in: untested, trial, verified, default
        Cannot go backwards — only upgrades allowed.
        """
        name = self._normalize(name)
        if level not in VALID_LEVELS:
            raise ValueError(f"Invalid confidence level: {level}. Must be one of {CONFIDENCE_ORDER}")

        # Ensure registered
        if name not in self._data:
            self.register(name, reason=reason)

        meta = self._data[name]
        if isinstance(meta, dict):
            meta = SkillMeta(**meta)
            self._data[name] = meta

        old_level = meta.confidence
        old_idx = CONFIDENCE_ORDER.index(old_level)
        new_idx = CONFIDENCE_ORDER.index(level)

        if new_idx <= old_idx:
            raise ValueError(
                f"Cannot downgrade {old_level} → {level}. "
                f"Levels are: {' → '.join(CONFIDENCE_ORDER)}"
            )

        meta.confidence = level
        self._add_grade_history(name, old_level, level, reason, grader)
        
        # Auto-add to default pool when promoted to "default"
        if level == "default":
            meta.default_skill_pool = True

        self._data[name] = meta
        self._save()
        return meta

    # ── Usage tracking ────────────────────────────────────────────

    def record_usage(self, name: str):
        """Increment usage count and update last_used timestamp."""
        name = self._normalize(name)
        if name not in self._data:
            self.register(name)

        meta = self._data[name]
        if isinstance(meta, dict):
            meta = SkillMeta(**meta)
            self._data[name] = meta

        meta.usage_count += 1
        meta.last_used = datetime.now(timezone.utc).isoformat()

        # Auto-promote from untested → trial after 3 uses
        if meta.confidence == "untested" and meta.usage_count >= 3:
            meta.confidence = "trial"
            self._add_grade_history(
                name, "untested", "trial",
                "Auto-promoted after 3 uses", "auto"
            )

        self._data[name] = meta
        self._save()

    # ── Default pool management ───────────────────────────────────

    def add_to_default_pool(self, name: str) -> SkillMeta:
        """Add a skill to the default preload list."""
        name = self._normalize(name)
        meta = self.get(name)
        if meta is None:
            raise KeyError(f"Skill '{name}' not registered. Run register() first.")

        if isinstance(meta, dict):
            meta = SkillMeta(**meta)
            self._data[name] = meta

        meta.default_skill_pool = True
        if meta.confidence == "untested":
            meta.confidence = "trial"
            self._add_grade_history(name, "untested", "trial",
                                    "Added to default pool", "user")
        self._data[name] = meta
        self._save()
        return meta

    def remove_from_default_pool(self, name: str) -> SkillMeta:
        """Remove a skill from the default preload list."""
        name = self._normalize(name)
        meta = self.get(name)
        if meta is None:
            raise KeyError(f"Skill '{name}' not registered.")

        if isinstance(meta, dict):
            meta = SkillMeta(**meta)
            self._data[name] = meta

        meta.default_skill_pool = False
        self._data[name] = meta
        self._save()
        return meta

    def list_default_pool(self) -> list[str]:
        """Return list of skill names in the default preload pool."""
        result = []
        for name, meta in self._data.items():
            if isinstance(meta, dict):
                if meta.get("default_skill_pool"):
                    result.append(name)
            else:
                if meta.default_skill_pool:
                    result.append(name)
        return result

    # ── Listing ───────────────────────────────────────────────────

    def list_all(self) -> list[tuple[str, SkillMeta]]:
        """Return all registered skills with their metadata."""
        result = []
        for name, raw in sorted(self._data.items()):
            if isinstance(raw, dict):
                result.append((name, SkillMeta(**raw)))
            else:
                result.append((name, raw))
        return result

    def list_by_confidence(self, level: str) -> list[tuple[str, SkillMeta]]:
        """Filter skills by confidence level."""
        return [
            (n, m) for n, m in self.list_all()
            if m.confidence == level
        ]

    def list_by_default_pool(self) -> list[tuple[str, SkillMeta]]:
        """Return skills in the default pool."""
        return [
            (n, m) for n, m in self.list_all()
            if m.default_skill_pool
        ]

    def remove(self, name: str) -> bool:
        """Remove a skill from the database. Returns True if removed."""
        name = self._normalize(name)
        if name in self._data:
            del self._data[name]
            self._save()
            return True
        return False

    # ── Report ────────────────────────────────────────────────────

    def report(self) -> str:
        """Generate a human-readable summary of all skills."""
        lines = []
        lines.append(f"Local Skill Confidence Database ({len(self._data)} skills)\n")
        lines.append(f"{'='*60}\n")

        for level in CONFIDENCE_ORDER:
            skills = self.list_by_confidence(level)
            if not skills:
                continue
            lines.append(f"\n[{level.upper()}] ({len(skills)} skill{'s' if len(skills) > 1 else ''})")
            for name, meta in skills:
                pool = " [DEFAULT]" if meta.default_skill_pool else ""
                lines.append(f"  • {name} — uses={meta.usage_count}, last={meta.last_used or 'never'}{pool}")
                if meta.notes:
                    lines.append(f"    note: {meta.notes}")

        # Default pool section
        defaults = self.list_by_default_pool()
        if defaults:
            lines.append(f"\n{'='*60}")
            lines.append(f"\n[DEFAULT POOL — auto-loaded] ({len(defaults)} skills)")
            for name, meta in defaults:
                lines.append(f"  ✓ {name}")

        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize skill name to lowercase, no hyphens/underscores variance."""
        return name.strip().lower().replace("_", "-")

    def _add_grade_history(
        self,
        name: str,
        old_level: Optional[str],
        new_level: str,
        reason: str,
        grader: str,
    ):
        """Append an entry to the grade history."""
        meta = self._data.get(name)
        if isinstance(meta, dict):
            meta = SkillMeta(**meta)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": old_level or "initial",
            "to": new_level,
            "reason": reason,
            "grader": grader,
        }
        if meta is not None:
            if isinstance(meta, SkillMeta):
                meta.grade_history.append(entry)
                self._data[name] = meta
            else:
                meta["grade_history"].append(entry)
                self._data[name] = meta


def get_default_pool_names() -> list[str]:
    """Convenience: get all skills that should be auto-loaded."""
    db = SkillMetaDB()
    return db.list_default_pool()
