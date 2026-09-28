#!/usr/bin/env python3
"""Taste Tool — decaying, corroborated preference learning for the review fork.

Thin integration surface over the stable computation engine
(``agent/taste_decay.py`` + ``agent/taste_corroboration.py`` — DO NOT MODIFY
those files). Pure stdlib, offline, fail-open: every entry point catches its
own exceptions and returns an ``{"ok": False, ...}`` dict instead of raising
into the agent loop.

Commands (single ``taste`` tool with an ``action`` parameter, mirroring the
``memory`` / ``skill_manage`` pattern):

- ``learn``   — record a candidate preference + confidence from a session.
- ``forget``  — line-level forget from ``taste.md`` (mirrors CC
  ``forgetTasteLearning``) + drop the candidate's engine state.
- ``summary`` — dump the escalation queue (staleness / conflict).
- ``write``   — flush established candidates to ``taste.md`` in the
  CC-compatible format (the ``confidence:`` token the CC parser reads).

State model: one :class:`CorroborationEngine` per candidate (preference id),
held in a module-level registry and persisted to a JSON sidecar
(``.taste_state.json``) inside ``taste_dir`` so corroboration accumulates
across processes. The human-facing ``taste.md`` always carries the decaying
score, so the ``confidence:`` token is always current.
"""

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_FILENAME = ".taste_state.json"
TASTE_FILENAME = "taste.md"

# Regex mirroring CC's own parser: confidence:\s*([0-9.]+)
CONFIDENCE_RE = re.compile(r"confidence:\s*([0-9.]+)")

DEFAULT_TASTE_CFG: Dict[str, Any] = {
    "enabled": True,
    "half_life_days": 14.0,
    "escalate_stale_after_days": 21,
    "conflict_epsilon": 0.15,
    "min_observations_for_write": 3,
    "auto_ack_observations": 10,
    "taste_dir": ".commandcode/taste",
}


def resolve_taste_dir(taste_dir: Optional[str] = None) -> Path:
    """Resolve the taste sidecar directory (created on write, never on read)."""
    raw = taste_dir or DEFAULT_TASTE_CFG["taste_dir"]
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _state_path(taste_dir: Optional[str] = None) -> Path:
    return resolve_taste_dir(taste_dir) / STATE_FILENAME


def _taste_md_path(taste_dir: Optional[str] = None) -> Path:
    return resolve_taste_dir(taste_dir) / TASTE_FILENAME


# ---------------------------------------------------------------------------
# Engine registry (candidate-keyed: one engine per preference id)
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, Any] = {}
_ENGINES_LOCK = threading.Lock()


def _merged_cfg(taste_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_TASTE_CFG)
    if isinstance(taste_cfg, dict):
        for k, v in taste_cfg.items():
            if v is not None:
                cfg[k] = v
    return cfg


def get_engine(
    preference_id: str,
    label: Optional[str] = None,
    taste_cfg: Optional[Dict[str, Any]] = None,
    taste_dir: Optional[str] = None,
):
    """Return the canonical engine for ``preference_id`` (create + restore)."""
    from agent.taste_corroboration import CorroborationEngine
    from agent.taste_decay import DecayConfig

    if not preference_id:
        raise ValueError("preference_id is required")
    cfg = _merged_cfg(taste_cfg)
    with _ENGINES_LOCK:
        engine = _ENGINES.get(preference_id)
        if engine is None:
            engine = CorroborationEngine(
                id=preference_id,
                label=label or preference_id,
                decay=DecayConfig(half_life_days=cfg["half_life_days"]),
            )
            stored = _load_state(taste_dir).get(preference_id)
            if isinstance(stored, dict):
                try:
                    engine.restore(stored)
                except Exception:
                    logger.debug(
                        "Taste state restore failed for %r; starting fresh",
                        preference_id,
                        exc_info=True,
                    )
            _ENGINES[preference_id] = engine
        return engine


def _load_state(taste_dir: Optional[str] = None) -> Dict[str, Any]:
    try:
        raw = _state_path(taste_dir).read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(taste_dir: Optional[str] = None) -> None:
    try:
        path = _state_path(taste_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _ENGINES_LOCK:
            payload = {
                pid: eng.snapshot()
                for pid, eng in _ENGINES.items()
                if hasattr(eng, "snapshot")
            }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.debug("Taste state save failed (fail-open)", exc_info=True)


def _drop_engine(preference_id: str) -> None:
    with _ENGINES_LOCK:
        _ENGINES.pop(preference_id, None)


# ---------------------------------------------------------------------------
# taste.md interop (CC-compatible format)
# ---------------------------------------------------------------------------

def _format_taste_line(label: str, score: float, n_obs: int,
                       weight: float, conflicts: int, stale: bool) -> str:
    return (
        f"- {label}. confidence: {score:.2f}\n"
        f"  (weight: {weight:.2f}, n_obs: {n_obs}, "
        f"last_conflict: {conflicts}, stale: {str(stale).lower()})"
    )


def write_taste_md(snapshot: Dict[str, Any],
                   taste_dir: Optional[str] = None) -> Path:
    """Write one established candidate into ``taste.md`` (upsert by label).

    The ``confidence:`` token is what CC's ``confidence:\\s*([0-9.]+)`` parser
    reads (interop); the parenthetical companion fields are Hermes' machine
    side. The decaying score is what gets written, so the token is always
    current — the upgrade over CC's inert write.
    """
    path = _taste_md_path(taste_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    label = str(snapshot.get("label") or snapshot.get("id") or "preference")
    line = _format_taste_line(
        label=label,
        score=float(snapshot.get("score", 0.0)),
        n_obs=int(snapshot.get("n_obs", 0)),
        weight=float(snapshot.get("raw_weight",
                                  snapshot.get("weight", 0.0))),
        conflicts=int(snapshot.get("conflicts", 0)),
        stale=bool(snapshot.get("stale", False)),
    )
    existing: List[str] = []
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            existing = []
    # Line-level upsert: replace the block whose first line names this label.
    out: List[str] = []
    replaced = False
    skip_next = False
    for ln in existing:
        if skip_next and ln.startswith("  ("):
            skip_next = False
            continue
        skip_next = False
        if ln.strip() == f"- {label}." or ln.startswith(f"- {label}."):
            out.append(line)
            replaced = True
            skip_next = True  # drop the old companion line
            continue
        out.append(ln)
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _read_taste_md(taste_dir: Optional[str] = None) -> List[str]:
    try:
        return _taste_md_path(taste_dir).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def taste_learn(preference_id: str, label: Optional[str] = None,
                confidence: float = 0.5, project: str = "",
                taste_cfg: Optional[Dict[str, Any]] = None,
                taste_dir: Optional[str] = None) -> Dict[str, Any]:
    """Record one candidate preference + confidence observation."""
    try:
        cfg = _merged_cfg(taste_cfg)
        engine = get_engine(preference_id, label or preference_id,
                            cfg, taste_dir)
        result = engine.observe(float(confidence))
        _save_state(taste_dir)
        wrote = False
        if engine.should_write() and result.established:
            snap = engine.snapshot()
            snap["score"] = result.score
            snap["label"] = result.label
            write_taste_md(snap, taste_dir)
            wrote = True
        return {
            "ok": True,
            "id": result.id,
            "score": result.score,
            "established": result.established,
            "auto_ack": result.auto_ack,
            "conflict": result.conflict,
            "n_obs": result.n_obs,
            "wrote_taste_md": wrote,
        }
    except Exception as e:  # fail-open: never raise into the agent loop
        logger.debug("taste learn failed (fail-open)", exc_info=True)
        return {"ok": False, "error": str(e)}


def taste_forget(target: str,
                 taste_dir: Optional[str] = None) -> Dict[str, Any]:
    """Line-level forget: remove the ``taste.md`` block naming ``target``.

    Mirrors CC ``forgetTasteLearning`` but scoped to one line (per-project),
    so forgetting one preference never destroys others sharing the file.
    """
    try:
        if not target:
            return {"ok": False, "error": "target is required"}
        # The taste.md line carries the human label, not the stable id — so
        # resolve an id to its label(s) via the live registry + sidecar
        # before line-matching. Match strings = the raw target plus any
        # known label for that id.
        match_strings = {target}
        try:
            with _ENGINES_LOCK:
                for pid, eng in _ENGINES.items():
                    if pid == target and getattr(eng, "label", ""):
                        match_strings.add(str(eng.label))  # type: ignore[union-attr]
            for pid, stored in _load_state(taste_dir).items():
                if pid == target and isinstance(stored, dict) and stored.get("label"):
                    match_strings.add(str(stored["label"]))
        except Exception:
            logger.debug("Taste forget label resolve failed (fail-open)",
                         exc_info=True)
        path = _taste_md_path(taste_dir)
        lines = _read_taste_md(taste_dir)
        out: List[str] = []
        removed = 0
        skip_next = False
        for ln in lines:
            if skip_next and ln.startswith("  ("):
                skip_next = False
                continue
            skip_next = False
            if ln.startswith("- ") and any(m in ln for m in match_strings):
                removed += 1
                skip_next = True
                continue
            out.append(ln)
        if removed and path.exists():
            path.write_text("\n".join(out) + ("\n" if out else ""),
                            encoding="utf-8")
        # Drop engine state for the forgotten candidate (match by id or label).
        with _ENGINES_LOCK:
            for pid in [p for p, e in _ENGINES.items()
                        if p == target or getattr(e, "label", "") == target]:
                _ENGINES.pop(pid, None)
        _save_state(taste_dir)
        # Also prune the persisted sidecar entry.
        try:
            sp = _state_path(taste_dir)
            if sp.exists():
                data = _load_state(taste_dir)
                pruned = {k: v for k, v in data.items()
                          if k != target and not (
                              isinstance(v, dict) and v.get("label") == target)}
                if pruned != data:
                    tmp = sp.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(pruned, indent=1),
                                   encoding="utf-8")
                    os.replace(tmp, sp)
        except Exception:
            logger.debug("Taste sidecar prune failed (fail-open)",
                         exc_info=True)
        return {"ok": True, "removed": removed}
    except Exception as e:
        logger.debug("taste forget failed (fail-open)", exc_info=True)
        return {"ok": False, "error": str(e)}


def taste_summary(taste_dir: Optional[str] = None,
                  taste_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dump the escalation queue (staleness / conflict) for human review."""
    try:
        cfg = _merged_cfg(taste_cfg)
        now = time.time()
        stale_after = float(cfg["escalate_stale_after_days"])
        queue: List[Dict[str, Any]] = []
        with _ENGINES_LOCK:
            engines = list(_ENGINES.items())
        for pid, engine in engines:
            try:
                peek = engine.to_result_now()
            except Exception:
                continue
            reasons: List[str] = []
            if getattr(engine, "is_conflicting", lambda: False)():
                reasons.append("conflict")
            try:
                age_days = (now - engine._state.last_obs_epoch) / 86400.0
            except Exception:
                age_days = 0.0
            if age_days > stale_after and peek.n_obs >= 2:
                reasons.append("staleness")
            if peek.escalated is not None and not reasons:
                reasons.append(getattr(peek.escalated, "reason", "escalated"))
            if reasons:
                queue.append({
                    "id": pid,
                    "label": peek.label,
                    "reasons": reasons,
                    "score": peek.score,
                    "n_obs": peek.n_obs,
                    "age_days": round(max(0.0, age_days), 1),
                })
        return {"ok": True, "escalations": queue, "count": len(queue)}
    except Exception as e:
        logger.debug("taste summary failed (fail-open)", exc_info=True)
        return {"ok": False, "error": str(e)}


def taste_write(taste_dir: Optional[str] = None,
                taste_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flush all established, escalation-free candidates to ``taste.md``."""
    try:
        cfg = _merged_cfg(taste_cfg)
        _ = cfg
        written: List[str] = []
        with _ENGINES_LOCK:
            engines = list(_ENGINES.items())
        for pid, engine in engines:
            try:
                if not (engine.should_write()):
                    continue
                peek = engine.to_result_now()
                if not peek.established:
                    continue
                snap = engine.snapshot()
                snap["score"] = peek.score
                snap["label"] = peek.label
                write_taste_md(snap, taste_dir)
                written.append(pid)
            except Exception:
                logger.debug("taste write skipped %r (fail-open)", pid,
                             exc_info=True)
                continue
        return {"ok": True, "written": written, "count": len(written)}
    except Exception as e:
        logger.debug("taste write failed (fail-open)", exc_info=True)
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool-loop entry point (registry handler)
# ---------------------------------------------------------------------------

def taste_tool(action: str = "", **kwargs) -> Dict[str, Any]:
    """Dispatch one ``taste`` tool call from the Hermes tool loop."""
    act = str(action or "").strip().lower()
    if act == "learn":
        return taste_learn(
            preference_id=str(kwargs.get("preference_id")
                              or kwargs.get("id") or ""),
            label=kwargs.get("label"),
            confidence=float(kwargs.get("confidence", 0.5)),
            project=str(kwargs.get("project", "")),
            taste_cfg=kwargs.get("taste_cfg"),
            taste_dir=kwargs.get("taste_dir"),
        )
    if act == "forget":
        return taste_forget(
            target=str(kwargs.get("target") or kwargs.get("preference_id")
                       or kwargs.get("label") or ""),
            taste_dir=kwargs.get("taste_dir"),
        )
    if act == "summary":
        return taste_summary(
            taste_dir=kwargs.get("taste_dir"),
            taste_cfg=kwargs.get("taste_cfg"),
        )
    if act == "write":
        return taste_write(
            taste_dir=kwargs.get("taste_dir"),
            taste_cfg=kwargs.get("taste_cfg"),
        )
    return {"ok": False,
            "error": f"unknown taste action {action!r} "
                     f"(expected learn|forget|summary|write)"}


TASTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Decaying, corroborated preference learning (taste). Record candidate "
        "preferences with confidence (learn), remove one line (forget), review "
        "the staleness/conflict escalation queue (summary), or flush "
        "established candidates to taste.md in CC-compatible format (write). "
        "Scores decay by half-life and need repeated corroboration before "
        "they are written — conflicting or stale candidates are escalated, "
        "never silently overwritten."
    ),
    "properties": {
        "action": {
            "type": "string",
            "enum": ["learn", "forget", "summary", "write"],
            "description": "The taste command to run.",
        },
        "preference_id": {
            "type": "string",
            "description": "Stable id for the candidate preference "
                           "(learn; also accepted as the forget target).",
        },
        "label": {
            "type": "string",
            "description": "Human-readable preference rule, written to "
                           "taste.md (learn).",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Assessed confidence for this observation, 0..1 "
                           "(learn).",
        },
        "target": {
            "type": "string",
            "description": "Preference id or label substring to forget "
                           "(forget).",
        },
        "project": {
            "type": "string",
            "description": "Project scope for the observation (learn).",
        },
        "taste_dir": {
            "type": "string",
            "description": "Override for the taste sidecar directory.",
        },
    },
    "required": ["action"],
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="taste",
    toolset="taste",
    schema=TASTE_SCHEMA,
    handler=lambda args, **kw: taste_tool(
        action=args.get("action", ""),
        preference_id=args.get("preference_id"),
        label=args.get("label"),
        confidence=args.get("confidence", 0.5),
        target=args.get("target"),
        project=args.get("project", ""),
        taste_dir=args.get("taste_dir"),
        store=kw.get("store"),
    ),
    emoji="👅",
)


__all__ = [
    "taste_tool",
    "taste_learn",
    "taste_forget",
    "taste_summary",
    "taste_write",
    "write_taste_md",
    "get_engine",
    "resolve_taste_dir",
    "TASTE_SCHEMA",
    "DEFAULT_TASTE_CFG",
]
