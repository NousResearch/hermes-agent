"""Replay cache for successful decide-loop trajectories (RFC #112639 autoresearch)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def _normalize_goal(goal: str) -> str:
    return re.sub(r"\s+", " ", (goal or "").strip().lower())


def _cache_key(goal: str, app: str | None) -> str:
    raw = f"{_normalize_goal(goal)}|{(app or '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CachedStep:
    action: str
    target_element: int | None = None
    keys: str | None = None
    backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.action}
        if self.target_element is not None:
            out["target_element"] = self.target_element
        if self.keys:
            out["keys"] = self.keys
        if self.backend:
            out["backend"] = self.backend
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedStep | None:
        action = (data.get("action") or "").strip().lower()
        if not action:
            return None
        target = data.get("target_element")
        return cls(
            action=action,
            target_element=int(target) if target is not None else None,
            keys=data.get("keys"),
            backend=data.get("backend"),
        )


def _cache_dir() -> Path:
    return get_hermes_home() / "cache" / "computer_use" / "trajectories"


def _cache_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def load_trajectory(goal: str, app: str | None = None) -> list[CachedStep] | None:
    path = _cache_path(_cache_key(goal, app))
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            return None
        parsed = [CachedStep.from_dict(s) for s in steps if isinstance(s, dict)]
        out = [s for s in parsed if s is not None]
        return out or None
    except Exception:
        return None


def save_trajectory(goal: str, app: str | None, steps: list[CachedStep]) -> str | None:
    if not steps:
        return None
    key = _cache_key(goal, app)
    path = _cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{key}.json"
    out.write_text(
        json.dumps(
            {
                "goal": _normalize_goal(goal),
                "app": (app or "").strip().lower() or None,
                "steps": [s.to_dict() for s in steps],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return str(out)


def invalidate_trajectory(goal: str, app: str | None = None) -> None:
    path = _cache_path(_cache_key(goal, app))
    if path.is_file():
        path.unlink(missing_ok=True)


def cached_step_to_decision(step: CachedStep) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "action": step.action,
        "backend": step.backend or "cache",
        "confidence": 1.0,
        "from_cache": True,
    }
    if step.target_element is not None:
        decision["target_element"] = step.target_element
        decision["target_ref"] = str(step.target_element)
    if step.keys:
        decision["keys"] = step.keys
    if step.action == "done":
        decision["done"] = True
    return decision
