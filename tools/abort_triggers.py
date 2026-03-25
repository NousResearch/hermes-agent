"""Natural-language abort trigger detection.

Detects when a user sends a message that clearly means "stop what you're
doing" — even without using a slash command.

Background: On 2026-02-23 an AI agent deleted a user's entire Gmail
inbox after context compaction discarded the "confirm before acting"
instruction.  The user typed "STOP" in the chat but the agent kept
going because only the /stop slash command was recognised.  This module
ensures plain-text abort requests are caught before they reach the
agent loop.

Usage:
    from tools.abort_triggers import is_abort_request

    if is_abort_request(user_text):
        agent.interrupt()
        return ABORT_REPLY
"""

import re

# ── Reply message ──────────────────────────────────────────────────────
ABORT_REPLY = "🚫 Agent aborted."

# ── Single-word triggers ──────────────────────────────────────────────
# If the *entire* normalized message is one of these words, it's an abort.
ABORT_TRIGGERS: frozenset[str] = frozenset({
    "stop",
    "abort",
    "cancel",
    "halt",
    "quit",
})

# ── Multi-word trigger phrases ────────────────────────────────────────
# If the normalized message matches one of these exactly, it's an abort.
ABORT_TRIGGER_PHRASES: frozenset[str] = frozenset({
    "stop hermes",
    "stop agent",
    "stop please",
    "stop now",
    "stop it",
    "stop that",
    "stop action",
    "stop run",
    "stop everything",
    "stop all",
})

# ── Normalization ─────────────────────────────────────────────────────
_TRAILING_PUNCT = re.compile(r"[!?.,:;…]+$")
_MULTI_SPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Normalize user input for abort matching.

    Pipeline:
    1. Strip whitespace
    2. Lowercase
    3. Strip trailing punctuation
    4. Collapse multiple spaces
    5. Strip leading/trailing whitespace again
    """
    text = text.strip().lower()
    text = _TRAILING_PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def is_abort_request(text: str) -> bool:
    """Return True if *text* is a natural-language abort request.

    Does NOT match slash commands (/stop) — those are handled by the
    existing command dispatch.  This function is for *plain text* abort
    detection only.
    """
    if not text or text.startswith("/"):
        return False

    normalized = _normalize(text)
    if not normalized:
        return False

    # Single-word match
    if normalized in ABORT_TRIGGERS:
        return True

    # Multi-word phrase match
    if normalized in ABORT_TRIGGER_PHRASES:
        return True

    return False
