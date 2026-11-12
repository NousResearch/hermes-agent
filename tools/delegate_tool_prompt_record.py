#!/usr/bin/env python3
"""Record the operative prompt of every delegated child.

The gap this closes. A child's system prompt is NOT the cached base prompt that
``sessions.system_prompt_hash`` points at. ``_build_child_system_prompt()`` composes a
focused prompt -- base identity, then a ``CONTEXT:`` block, then the workspace's project
context files -- and passes it to the child as ``ephemeral_system_prompt``. That text is
composed at API-call time and is never persisted, so two children given different
contracts are indistinguishable in the record.

Measured on 2026-11-02 (contract-propagation test, delegation ``deleg_01e48cbc``):

  arm A (contract in `context`)  -> refused, no file written
  arm B (no contract)            -> wrote the file
  both arms                      -> identical ``system_prompt_hash`` and length

The record could not explain the difference because the deciding text was not stored. This
module stores it: the composed prompt content-addressed on disk, plus one append-only index
line per child, so "what governed this child?" survives the scratch transcript.

Contract: this must never raise. A recorder that can break delegation is worse than a
missing record, so every failure is swallowed and warned about once per process.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Set once the recorder has failed in this process, so the warning is emitted once and not
# per delegated child.
_warned = False

INDEX_NAME = "index.jsonl"

# Long contexts exist; keep the artefact bounded and say when it was cut.
MAX_TEXT_BYTES = 262_144

# Booleans recorded per prompt. A marker here is evidence the prompt NAMED the thing; it is
# not by itself proof that the child obeyed it -- see the note in the invariance project.
MARKERS = {
    "sentinel": "Forge governance contract not loaded",
    "human_gates": "human_gates",
    "permissions": "permissions",
    "constraints": "constraints",
    "ratification": "ratification token",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _resolve_state_dir(state_dir: str | Path | None) -> Path:
    """Profile-scoped by default. ``state_dir`` exists so tests never touch the live record."""
    if state_dir is not None:
        return Path(state_dir).expanduser()
    return Path(get_hermes_home()) / "state" / "delegated-prompts"


def record_child_prompt(
    *,
    goal: str | None,
    context: str | None,
    child_prompt: str | None,
    subagent_id: str | None = None,
    task_index: int | None = None,
    parent_session_id: str | None = None,
    workspace_path: str | None = None,
    state_dir: str | None = None,
) -> dict | None:
    """Persist the composed child prompt + one index line. Returns the record, or None.

    ``goal`` is the child's first user turn (the delegate tool keeps it out of the system
    prompt deliberately), so ``goal_sha256`` is the join key between this index and the
    session row.
    """
    try:
        # Resolved per call, not at import: a long-lived gateway process can serve more than
        # one profile, so a module constant would freeze the launching profile's home.
        state = _resolve_state_dir(state_dir)
        state.mkdir(parents=True, exist_ok=True)
        text = child_prompt or ""
        digest = _sha(text)

        stored, truncated = text, False
        encoded = text.encode("utf-8", "replace")
        if len(encoded) > MAX_TEXT_BYTES:
            half = MAX_TEXT_BYTES // 2
            head = encoded[:half].decode("utf-8", "ignore")
            tail = encoded[-half:].decode("utf-8", "ignore")
            stored, truncated = f"{head}\n…[truncated {len(encoded) - MAX_TEXT_BYTES} bytes]…\n{tail}", True

        artefact = state / f"{digest}.txt"
        if not artefact.exists():
            artefact.write_text(stored, encoding="utf-8")

        record = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "subagent_id": subagent_id,
            "task_index": task_index,
            "parent_session_id": parent_session_id,
            "workspace_path": workspace_path,
            "prompt_sha256": digest,
            "prompt_len": len(text),
            "truncated": truncated,
            "goal_sha256": _sha(goal or ""),
            "goal_len": len(goal or ""),
            "context_len": len(context or ""),
            "context_sha256": _sha(context) if context else None,
            "markers": {name: (needle in text) for name, needle in MARKERS.items()},
        }
        index_path = state / INDEX_NAME
        with open(index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
    except Exception:                                    # never break delegation
        # Fail-open, but not silently: a recorder that fails invisibly is a control whose
        # failure reads as success. Warn once per process, then stay quiet.
        global _warned
        if not _warned:
            _warned = True
            logger.warning("ephemeral prompt record failed; delegated prompts are not being "
                           "recorded this process", exc_info=True)
        return None
