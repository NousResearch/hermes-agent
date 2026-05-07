"""Extractor — the main orchestration module for Phase 2 auto-memory.

Public entry points (called from run_agent.py / cli.py):
  * ``on_turn_end(session_id, user_msg, assistant_msg)``
  * ``on_pre_compress(session_id, messages)``
  * ``on_session_end(session_id, messages, *, interactive=False)``
  * ``flush_buffer(session_id)``
  * ``is_enabled()``

All entry points are best-effort — they catch every exception, log it,
and return. Extraction failures must never break the agent loop.

LLM routing: uses ``auxiliary_client.call_llm`` with task name
``memory_extraction``. User can override model / provider / timeout via
``auxiliary.memory_extraction.*`` in ``config.yaml``. Default model is
``claude-haiku-4-5``.

Concurrency: per-turn extraction runs in a background thread so it
doesn't block the agent loop. Pre-compress runs inline (it's already on
a slow path — compression itself is a multi-second LLM call). Session-end
runs inline (the user is exiting; blocking briefly is fine).

Telemetry: every extraction call's input/output token counts are logged
to ``$HERMES_HOME/logs/memory_extraction.log`` so we can tune prompts
later. Format: one JSON object per line (jsonlines).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.memory_extraction import buffer as _buffer
from tools.memory_extraction import conflict as _conflict
from tools.memory_extraction import prompts as _prompts

logger = logging.getLogger(__name__)

# Default model. User can override via ``auxiliary.memory_extraction.model``.
_DEFAULT_MODEL = "claude-haiku-4-5"

# Background thread pool — small, daemonized, reuses threads to avoid spawn cost.
_per_turn_lock = threading.Lock()
_per_turn_thread: Optional[threading.Thread] = None

# Telemetry log handle (lazy)
_telemetry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Config / enable check
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Return True when auto-extraction is configured ON.

    Reads ``memory.auto_extract`` from config.yaml. Default: ``False``
    (Phase 1 ships without auto-extract; user opts in).
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        mem_cfg = cfg.get("memory", {}) or {}
        return bool(mem_cfg.get("auto_extract", False))
    except Exception:
        return False


def _get_extraction_config() -> Dict[str, Any]:
    """Read auxiliary.memory_extraction.* config with defaults."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        aux = (cfg.get("auxiliary", {}) or {}).get("memory_extraction", {}) or {}
        return {
            "model": aux.get("model", _DEFAULT_MODEL),
            "provider": aux.get("provider"),
            "timeout": aux.get("timeout", 30),
            "max_tokens_per_turn": aux.get("max_tokens_per_turn", 1024),
            "max_tokens_session_end": aux.get("max_tokens_session_end", 2048),
            "include_pre_compress": aux.get("include_pre_compress", True),
            "auto_commit_session_end": aux.get("auto_commit_session_end", False),
        }
    except Exception:
        return {
            "model": _DEFAULT_MODEL,
            "provider": None,
            "timeout": 30,
            "max_tokens_per_turn": 1024,
            "max_tokens_session_end": 2048,
            "include_pre_compress": True,
            "auto_commit_session_end": False,
        }


# ---------------------------------------------------------------------------
# LLM dispatch
# ---------------------------------------------------------------------------

def _call_extraction_llm(
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
    timeout: Optional[int] = None,
) -> str:
    """Call the auxiliary LLM client with extraction-task hints.

    Returns the response text. Raises on transport failures so callers
    can fall back / log.
    """
    from agent.auxiliary_client import call_llm
    cfg = _get_extraction_config()
    call_kwargs: Dict[str, Any] = {
        "task": "memory_extraction",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    if cfg.get("model"):
        call_kwargs["model"] = cfg["model"]
    if cfg.get("provider"):
        call_kwargs["provider"] = cfg["provider"]
    if timeout is not None:
        call_kwargs["timeout"] = timeout
    elif cfg.get("timeout"):
        call_kwargs["timeout"] = cfg["timeout"]

    response = call_llm(**call_kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        content = str(content) if content else ""
    # Telemetry: log token usage
    _log_telemetry({
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "max_tokens": max_tokens,
        "input_chars": len(system) + len(user),
        "output_chars": len(content),
        "usage": _maybe_extract_usage(response),
    })
    return content.strip()


def _maybe_extract_usage(response: Any) -> Optional[Dict[str, int]]:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    except Exception:
        return None


def _log_telemetry(record: Dict[str, Any]) -> None:
    """Append a one-line jsonl record to memory_extraction.log."""
    try:
        from hermes_constants import get_hermes_home
        log_path = get_hermes_home() / "logs" / "memory_extraction.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _telemetry_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-turn extraction
# ---------------------------------------------------------------------------

def on_turn_end(
    session_id: str,
    user_msg: Any,
    assistant_msg: Any,
) -> None:
    """Per-turn extraction. Runs in a background thread so we don't block.

    Writes proposals to the session buffer. Final commit happens at
    session-end.
    """
    if not is_enabled() or not session_id:
        return
    if not user_msg and not assistant_msg:
        return

    def _run():
        try:
            cfg = _get_extraction_config()
            response_text = _call_extraction_llm(
                system=_prompts.PER_TURN_SYSTEM,
                user=_prompts.per_turn_user(
                    user_msg=str(user_msg or ""),
                    assistant_msg=str(assistant_msg or ""),
                ),
                max_tokens=int(cfg["max_tokens_per_turn"]),
            )
            entries = _prompts.parse_extraction_response(response_text)
            if entries:
                appended = _buffer.append(session_id, entries, source="per_turn")
                if appended:
                    logger.debug(
                        "memory extraction: per_turn appended %d entries to session %s",
                        appended, session_id,
                    )
        except Exception as e:
            logger.debug("memory extraction per_turn failed: %s", e)

    # Wait for the previous per-turn extraction (if still running) to
    # avoid backing up the LLM client. Best-effort, short timeout.
    global _per_turn_thread
    with _per_turn_lock:
        if _per_turn_thread and _per_turn_thread.is_alive():
            _per_turn_thread.join(timeout=2.0)
        _per_turn_thread = threading.Thread(
            target=_run,
            name=f"mem-extract-{session_id[:8]}",
            daemon=True,
        )
        _per_turn_thread.start()


# ---------------------------------------------------------------------------
# Pre-compress extraction
# ---------------------------------------------------------------------------

def on_pre_compress(
    session_id: str,
    messages: List[Dict[str, Any]],
) -> None:
    """Pre-compress extraction. Runs inline on the compression slow path.

    Extracts facts from the slice that's about to be compressed/discarded.
    """
    if not is_enabled() or not session_id or not messages:
        return
    cfg = _get_extraction_config()
    if not cfg.get("include_pre_compress", True):
        return

    try:
        response_text = _call_extraction_llm(
            system=_prompts.PRE_COMPRESS_SYSTEM,
            user=_prompts.pre_compress_user(messages),
            max_tokens=int(cfg["max_tokens_per_turn"]),
        )
        entries = _prompts.parse_extraction_response(response_text)
        if entries:
            appended = _buffer.append(session_id, entries, source="pre_compress")
            logger.info(
                "memory extraction: pre_compress appended %d entries to session %s",
                appended, session_id,
            )
    except Exception as e:
        logger.debug("memory extraction pre_compress failed: %s", e)


# ---------------------------------------------------------------------------
# Session-end extraction + commit
# ---------------------------------------------------------------------------

def on_session_end(
    session_id: str,
    messages: List[Dict[str, Any]],
    *,
    interactive: bool = False,
    confirm_callback: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Session-end extraction + commit.

    Args:
        session_id: id of the session that just ended
        messages: final conversation state (post-compression)
        interactive: when True, calls ``confirm_callback`` with the proposed
            entry list and uses the returned list. When False, the
            ``auto_commit_session_end`` config flag decides whether entries
            are auto-committed.
        confirm_callback: required when interactive=True — receives a list
            of entry dicts and returns the user-approved subset.

    Returns a summary dict:
        {
          "session_id": str,
          "buffered": int,           # entries from per-turn / pre-compress
          "final_proposed": int,     # entries after session-end LLM pass
          "committed": int,          # actually written to warm tier
          "skipped": int,            # rejected by user / dedup'd / errored
          "actions": [...]           # per-entry verdict + outcome
        }

    Failures degrade gracefully — on any error the buffer is preserved
    so the next session can retry.
    """
    summary: Dict[str, Any] = {
        "session_id": session_id,
        "buffered": 0,
        "final_proposed": 0,
        "committed": 0,
        "skipped": 0,
        "actions": [],
    }
    if not is_enabled() or not session_id:
        return summary

    buffered = _buffer.get_session_entries(session_id)
    summary["buffered"] = len(buffered)

    cfg = _get_extraction_config()

    # Step 1: final extraction pass — reconcile buffer + final messages.
    final_entries: List[Dict[str, Any]] = []
    try:
        response_text = _call_extraction_llm(
            system=_prompts.SESSION_END_SYSTEM,
            user=_prompts.session_end_user(messages or [], buffered),
            max_tokens=int(cfg["max_tokens_session_end"]),
        )
        final_entries = _prompts.parse_extraction_response(response_text)
        summary["final_proposed"] = len(final_entries)
    except Exception as e:
        logger.warning("memory extraction session_end failed: %s — falling back to buffer", e)
        # Fall back to buffer contents so we don't lose proposals.
        final_entries = buffered
        summary["final_proposed"] = len(buffered)

    if not final_entries:
        # Nothing to commit. Clear the buffer to free space.
        _buffer.clear_session(session_id)
        return summary

    # Step 2: confirm UI (interactive) or auto-commit
    auto_commit = bool(cfg.get("auto_commit_session_end", False))
    if interactive and confirm_callback is not None:
        try:
            approved = confirm_callback(final_entries)
        except Exception as e:
            logger.warning("memory extraction confirm callback failed: %s", e)
            approved = []
    elif auto_commit:
        approved = final_entries
    else:
        # Default safe path: skip auto-commit when the user isn't watching.
        # Stash proposals back into the buffer so they survive. The next
        # interactive session can pick them up via a "memory pending" prompt.
        _buffer.replace_session_entries(session_id, final_entries)
        summary["skipped"] = len(final_entries)
        return summary

    # Step 3: dispatch each approved entry through conflict resolution
    #
    # IMPORTANT: when the proposal already carries a ``verdict`` field
    # (because the confirm UI ran ``_classify_proposals`` and showed it
    # to the user), we MUST reuse that exact verdict here. Re-classifying
    # at commit time would:
    #   1. Lie to the user — they approved based on the displayed verdict;
    #      the LLM is non-deterministic on edge cases and a second roll can
    #      flip DUPLICATE → NEW (or vice versa), polluting the warm store
    #      with duplicates the user thought were being deduped.
    #   2. Double the LLM cost on the slow path (one classify per proposal
    #      in the UI, one again here).
    # Only classify fresh when the verdict isn't pre-attached — i.e. the
    # non-interactive auto-commit path that bypasses the confirm UI.
    from tools.memory_extraction.conflict import ConflictVerdict
    for proposal in approved:
        try:
            attached = proposal.get("verdict")
            if isinstance(attached, ConflictVerdict):
                verdict = attached
            else:
                verdict = _conflict.classify(proposal["content"])
            outcome = _conflict.apply_verdict(verdict, proposal, auto_commit=False)
            summary["actions"].append({
                "content": proposal["content"][:120],
                "verdict": verdict.verdict,
                "outcome": outcome.get("action"),
                "fact_id": outcome.get("fact_id"),
            })
            if outcome.get("action") in (
                "stored", "refined", "deduplicated", "superseded",
            ):
                summary["committed"] += 1
            else:
                summary["skipped"] += 1
        except Exception as e:
            logger.warning("memory extraction commit failed: %s", e)
            summary["skipped"] += 1

    # Step 4: clear the buffer — proposals are now committed (or surfaced)
    _buffer.clear_session(session_id)
    return summary


def flush_buffer(session_id: str) -> int:
    """Drop a session's buffer without committing. Used on /reset."""
    return _buffer.clear_session(session_id)
