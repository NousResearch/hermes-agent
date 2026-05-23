"""Observation ingestion helpers for VRChat autonomy loops."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.openclaw.vrchat_autonomy import (
    DEFAULT_OBSERVATION_QUEUE_NAME,
    VALID_OBSERVATION_SOURCES,
    enqueue_observation,
    get_hermes_home,
    normalize_observations,
)


def build_observation(
    *,
    source: str,
    text: str = "",
    summary: str = "",
    content: str = "",
    trust: str = "untrusted",
    timestamp: str = "",
) -> dict[str, Any]:
    """Build one bounded observation object from an external sensor or bridge."""
    return {
        "source": str(source or "").strip(),
        "text": str(text or summary or content or "").strip(),
        "trust": str(trust or "untrusted").strip() or "untrusted",
        "timestamp": str(timestamp or "").strip(),
    }


def build_observation_from_osc(
    address: str,
    args: list[Any] | tuple[Any, ...],
    *,
    allow_avatar_parameters: bool = False,
) -> dict[str, Any]:
    """Convert one incoming VRChat OSC event into a non-actuating observation."""
    address = str(address or "").strip()
    values = list(args or [])
    if address == "/chatbox/input":
        text = str(values[0] if values else "").strip()
        return {
            "success": bool(text),
            "observation": build_observation(source="textBox", text=text, trust="vrchat_osc"),
            "ignored": "" if text else "empty_chatbox",
        }
    if address.startswith("/avatar/parameters/"):
        if not allow_avatar_parameters:
            return {
                "success": False,
                "observation": None,
                "ignored": "avatar_parameter_observation_disabled",
            }
        parameter = address.rsplit("/", 1)[-1]
        value = values[0] if values else None
        return {
            "success": True,
            "observation": build_observation(
                source="system",
                text=f"Avatar parameter changed: {parameter}={value!r}",
                trust="vrchat_osc",
            ),
            "ignored": "",
        }
    return {"success": False, "observation": None, "ignored": f"unsupported_osc_address:{address}"}


def ingest_observations(
    observations: list[dict[str, Any]],
    *,
    queue_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Validate and queue multiple external observations for a later autonomy tick."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    queued: list[dict[str, Any]] = []
    for index, observation in enumerate(observations or []):
        normalized = normalize_observations([observation])
        if not normalized["accepted"]:
            reason = normalized["rejected"][0]["reason"] if normalized["rejected"] else "empty_observation"
            rejected.append({"index": str(index), "reason": reason})
            continue
        accepted_observation = normalized["accepted"][0]
        accepted.append(accepted_observation)
        queued_result = enqueue_observation(accepted_observation, queue_path=queue_path, persist=persist)
        if queued_result["success"]:
            queued.append(queued_result["observation"])
        else:
            rejected.append({"index": str(index), "reason": "queue_rejected"})

    return {
        "success": not rejected,
        "accepted": accepted,
        "queued": queued,
        "rejected": rejected,
        "queue_path": str(_queue_path(queue_path)) if persist else None,
        "persisted": persist,
    }


def observation_queue_status(
    *,
    queue_path: str | Path | None = None,
    max_preview: int = 5,
) -> dict[str, Any]:
    """Return a read-only preview of the persisted observation queue."""
    path = _queue_path(queue_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {
            "success": True,
            "exists": False,
            "path": str(path),
            "count": 0,
            "preview": [],
            "rejected_lines": 0,
        }

    preview_limit = max(0, int(max_preview))
    preview: list[dict[str, Any]] = []
    rejected = 0
    count = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            rejected += 1
            continue
        if not isinstance(item, dict):
            rejected += 1
            continue
        count += 1
        if len(preview) < preview_limit:
            preview.append(item)
    return {
        "success": True,
        "exists": True,
        "path": str(path),
        "count": count,
        "preview": preview,
        "rejected_lines": rejected,
    }


def parse_jsonl_observation(line: str) -> dict[str, Any]:
    """Parse a JSONL stdin event into a supported observation object."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return {"success": False, "observation": None, "error": "invalid_json"}
    if not isinstance(payload, dict):
        return {"success": False, "observation": None, "error": "event_must_be_object"}
    if "observation" in payload and isinstance(payload["observation"], dict):
        payload = payload["observation"]
    source = str(payload.get("source") or payload.get("type") or "").strip()
    if source not in VALID_OBSERVATION_SOURCES:
        return {"success": False, "observation": None, "error": f"unsupported_source:{source}"}
    observation = build_observation(
        source=source,
        text=str(payload.get("text") or ""),
        summary=str(payload.get("summary") or ""),
        content=str(payload.get("content") or ""),
        trust=str(payload.get("trust") or "untrusted"),
        timestamp=str(payload.get("timestamp") or int(time.time())),
    )
    return {"success": True, "observation": observation, "error": ""}


def _queue_path(queue_path: str | Path | None) -> Path:
    if queue_path:
        return Path(queue_path).expanduser()
    return get_hermes_home() / "state" / DEFAULT_OBSERVATION_QUEUE_NAME
