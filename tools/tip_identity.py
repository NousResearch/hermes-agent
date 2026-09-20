"""Stable content identity for agent-authored tips (issue #117216).

An agent tip used to travel with no ``tipId``, so the renderer's seen/retired
ledgers never recorded it and every new conversation showed the same bubble
again. ``agent_tip_id`` derives a stable id from what the tip says and points
at, so "the same tip" always carries the same id and a manual ✕ can mean
"never again".
"""

import hashlib


def agent_tip_id(selector: str, text: str, title: str = "") -> str:
    """A stable content id for an agent tip: same content → same id.

    ``agent-`` prefixed so a renderer can tell agent ids from catalog ids at a
    glance; the hash is truncated to 12 hex chars, which is collision-proof for
    the handful of tips a conversation ever mints.
    """
    raw = "\x1f".join((selector, text, title))
    return "agent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
