"""Personal STT correction/glossary helpers.

This module stores small deterministic phrase corrections for speech-to-text
output. It intentionally does not call an LLM: corrections are fast, local, and
safe to run before the agent sees the transcript.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_CORRECTIONS_FILE = "stt_corrections.json"
_VERSION = 1
_SEPARATORS = ("=>", "->", "=")


def _corrections_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return get_hermes_home() / _CORRECTIONS_FILE


def _load(path: str | Path | None = None) -> dict[str, Any]:
    p = _corrections_path(path)
    if not p.exists():
        return {"version": _VERSION, "corrections": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": _VERSION, "corrections": []}
    if not isinstance(data, dict):
        return {"version": _VERSION, "corrections": []}
    corrections = data.get("corrections")
    if not isinstance(corrections, list):
        corrections = []
    clean: list[dict[str, Any]] = []
    for item in corrections:
        if not isinstance(item, dict):
            continue
        wrong = str(item.get("wrong") or "").strip()
        right = str(item.get("right") or "").strip()
        if not wrong or not right:
            continue
        clean.append(
            {
                "wrong": wrong,
                "right": right,
                "case_sensitive": bool(item.get("case_sensitive", False)),
                "created_at": str(item.get("created_at") or ""),
            }
        )
    return {"version": _VERSION, "corrections": clean}


def _save(data: dict[str, Any], path: str | Path | None = None) -> None:
    p = _corrections_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _VERSION,
        "corrections": data.get("corrections", []),
    }
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_correction_args(args: str) -> tuple[str, str]:
    """Parse `/stt-correct wrong => right` style arguments."""
    raw = (args or "").strip()
    for sep in _SEPARATORS:
        if sep in raw:
            wrong, right = raw.split(sep, 1)
            wrong = wrong.strip().strip("\"'“”‘’")
            right = right.strip().strip("\"'“”‘’")
            if wrong and right:
                return wrong, right
    raise ValueError("Format attendu : /stt-correct phrase entendue => phrase correcte")


def list_corrections(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return stored corrections in insertion order."""
    return list(_load(path).get("corrections", []))


def add_correction(
    wrong: str,
    right: str,
    *,
    case_sensitive: bool = False,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Add or replace a phrase correction."""
    wrong = (wrong or "").strip()
    right = (right or "").strip()
    if not wrong or not right:
        raise ValueError("La phrase source et la correction doivent être non vides.")

    data = _load(path)
    corrections = data.setdefault("corrections", [])
    key = wrong if case_sensitive else wrong.casefold()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "wrong": wrong,
        "right": right,
        "case_sensitive": bool(case_sensitive),
        "created_at": now,
    }
    for idx, item in enumerate(corrections):
        existing_key = item["wrong"] if item.get("case_sensitive") else item["wrong"].casefold()
        if existing_key == key:
            corrections[idx] = entry
            _save(data, path)
            return {"action": "updated", **entry, "index": idx + 1}
    corrections.append(entry)
    _save(data, path)
    return {"action": "added", **entry, "index": len(corrections)}


def remove_correction(index: int, path: str | Path | None = None) -> dict[str, Any]:
    """Remove a correction by one-based index."""
    data = _load(path)
    corrections = data.setdefault("corrections", [])
    if index < 1 or index > len(corrections):
        raise IndexError("Index de correction STT invalide.")
    removed = corrections.pop(index - 1)
    _save(data, path)
    return removed


def clear_corrections(path: str | Path | None = None) -> int:
    """Remove all corrections and return how many were cleared."""
    data = _load(path)
    count = len(data.get("corrections", []))
    _save({"version": _VERSION, "corrections": []}, path)
    return count


def apply_stt_corrections(transcript: str, path: str | Path | None = None) -> str:
    """Apply deterministic phrase corrections to a transcript."""
    text = transcript or ""
    corrections = list_corrections(path)
    # Longer phrases first prevents short glossary terms from disrupting a
    # longer, more specific correction.
    corrections.sort(key=lambda item: len(item["wrong"]), reverse=True)
    for item in corrections:
        wrong = item["wrong"]
        right = item["right"]
        flags = 0 if item.get("case_sensitive") else re.IGNORECASE
        text = re.sub(re.escape(wrong), lambda _m: right, text, flags=flags)
    return text


def format_corrections(path: str | Path | None = None) -> str:
    corrections = list_corrections(path)
    if not corrections:
        return "Aucune correction STT enregistrée."
    lines = ["Corrections STT enregistrées :"]
    for idx, item in enumerate(corrections, start=1):
        lines.append(f"{idx}. « {item['wrong']} » → « {item['right']} »")
    return "\n".join(lines)


def usage_text() -> str:
    return (
        "Usage :\n"
        "• /stt-correct phrase entendue => phrase correcte\n"
        "• /stt-correct list\n"
        "• /stt-correct remove 1\n"
        "• /stt-correct clear"
    )
