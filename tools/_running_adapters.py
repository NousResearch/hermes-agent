"""Module-level registry for *running* platform-adapter instances.

Stateless adapters (Telegram, Discord, Feishu, ...) instantiate fresh
per outbound call from ``send_message_tool.py``. That pattern silently
breaks for **webhook-receive** platforms — Teams Bot Framework being
the first instance, with Webex / Zoom Apps / Google Chat all following
the same shape — because those adapters hold per-process state
(``_pending_uploads``, ``_conv_refs``, message-id maps, ...) that is
the rendezvous point between an *outbound* action and a later
*inbound* webhook from the platform. A fresh instance has empty
state, so the inbound webhook lands on the running instance whose
state was never seeded — and the action silently fails.

This registry lets the gateway publish the live adapter when it
connects, and lets ``send_message_tool.py`` reach it from outbound
code paths. Single Python process — gateway, ``run_agent.py``, and
tool dispatch all share memory in the standard topology.

If a future deployment splits the gateway and tool runners across a
process boundary, this registry needs an RPC backing instead — but
that is not the world we live in.

Architecture context: see the ``hermes-agent-pilot`` skill, reference
``outbound-media-wiring-by-send-model.md`` for why webhook-receive
platforms cannot use the stateless / client-pull patterns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Module-level state. Keyed by ``platform.value`` (the lowercase string
# form, e.g. ``"teams"``) so callers don't need to import ``Platform``.
_RUNNING_ADAPTERS: Dict[str, Any] = {}


def set_running_adapter(platform: str, adapter: Any) -> None:
    """Publish ``adapter`` as the live instance for ``platform``.

    Called by the gateway after a successful ``connect()``. Idempotent:
    re-registering replaces the previous entry, which is the right
    behavior on adapter reconnect.
    """
    _RUNNING_ADAPTERS[platform] = adapter


def get_running_adapter(platform: str) -> Optional[Any]:
    """Return the running adapter for ``platform``, or ``None`` if none.

    Returning ``None`` (rather than raising) lets the caller emit a
    clear platform-level error message ("Teams not connected — cannot
    send proactive media") instead of leaking a ``KeyError`` traceback
    through the agent's tool result.
    """
    return _RUNNING_ADAPTERS.get(platform)


def clear_running_adapter(platform: str) -> None:
    """Drop the entry for ``platform``. No-op if not present.

    Called when the gateway disconnects an adapter, so a stale instance
    isn't left in the registry pointing at a closed connection.
    """
    _RUNNING_ADAPTERS.pop(platform, None)


def clear_running_adapters() -> None:
    """Drop every entry. Test fixture only — production should use the
    per-platform ``clear_running_adapter``."""
    _RUNNING_ADAPTERS.clear()
