"""Configurable busy-input acknowledgement templates (issue #26024).

When a user sends a follow-up message while the gateway is already running
an agent for that session, the gateway sends a short acknowledgement so
the user knows the input was received. Three messages exist today:

  * ``interrupt`` — "⚡ Interrupting current task{status_detail}. I'll respond
    to your message shortly."
  * ``queue``     — "⏳ Queued for the next turn{status_detail}. I'll respond
    once the current task finishes."
  * ``steer``     — "⏩ Steered into current run{status_detail}. Your message
    arrives after the next tool call."

The text used to be hardcoded. Deployments that want quieter / non-English /
brand-voiced acknowledgements had no clean knob; the only ways out were:

  1. Set ``display.busy_ack_enabled: false`` -- removes feedback entirely,
     leaving the user wondering whether the bot is dead.
  2. Patch the strings in a local fork -- breaks on upgrade.

This module adds a config-driven template system that keeps the current
defaults but lets operators override per mode. The schema lives under
``display.busy_ack_templates``::

    display:
      busy_input_mode: queue
      busy_ack_enabled: true
      busy_ack_templates:
        queue: "Queued for the next turn{status_detail}. Will reply once done."
        interrupt: "Interrupting{status_detail}. New reply incoming."
        steer: "Steered{status_detail}. Will fold this into the next turn."

Each template may contain a single ``{status_detail}`` placeholder. The
runtime fills it with a short status string like
`` (3 min elapsed, iteration 17/60, running: terminal)`` when an agent is
actively reporting progress, or with an empty string when no status is
available. Templates without a ``{status_detail}`` placeholder are
rendered as-is (the status string is computed but discarded).

The defaults match the strings that shipped before this feature so
existing deployments see no behaviour change.

Two integration shapes are supported on the input side:

  1. **Dict from config** -- the gateway calls
     :func:`resolve_busy_ack_template` with the parsed config dict and the
     mode name. This is the canonical path used by ``GatewayRunner``.
  2. **JSON env var** -- when the templates are bridged into a subprocess
     (cron jobs, spawned tools, dashboard worker), the gateway encodes
     ``display.busy_ack_templates`` into ``HERMES_GATEWAY_BUSY_ACK_TEMPLATES``
     as JSON. :func:`load_templates_from_env` decodes and returns the same
     dict shape the resolver expects.

This module is intentionally dependency-free (stdlib only) so it can be
imported safely from both the agent-side code and the gateway runtime
without dragging in heavier modules.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Defaults
# =========================================================================

# Default templates -- exact strings the gateway has shipped since
# busy-ack was introduced. Preserve verbatim so any deployment that does
# NOT override templates sees zero behaviour change.
DEFAULT_TEMPLATES: Dict[str, str] = {
    "interrupt": (
        "⚡ Interrupting current task{status_detail}. "
        "I'll respond to your message shortly."
    ),
    "queue": (
        "⏳ Queued for the next turn{status_detail}. "
        "I'll respond once the current task finishes."
    ),
    "steer": (
        "⏩ Steered into current run{status_detail}. "
        "Your message arrives after the next tool call."
    ),
}

# Valid modes -- the resolver rejects anything else so a typo in
# config.yaml (e.g. ``steered:`` instead of ``steer:``) doesn't silently
# get accepted into the templates map.
VALID_MODES = frozenset(DEFAULT_TEMPLATES.keys())

# Env var name used to bridge templates into subprocesses. Matches the
# existing ``HERMES_GATEWAY_BUSY_*`` naming convention.
ENV_VAR_TEMPLATES = "HERMES_GATEWAY_BUSY_ACK_TEMPLATES"


# =========================================================================
# Resolver
# =========================================================================

def resolve_busy_ack_template(
    templates: Optional[Mapping[str, Any]],
    mode: str,
) -> str:
    """Pick the right template string for ``mode``.

    Args:
        templates: The raw value of ``display.busy_ack_templates`` from
            config (a mapping ``{mode: template_string}``). May be ``None``
            (no override), empty, or partial -- missing modes fall back to
            the default. Non-string values are ignored with a warning so a
            malformed entry can't crash the gateway during a busy turn.
        mode: One of ``"interrupt"``, ``"queue"``, ``"steer"``. Any other
            value falls back to ``"interrupt"`` (matches the runtime's
            existing default in :func:`gateway.run._handle_active_session_busy_message`).

    Returns:
        The template string -- either the override or the documented
        default. Always returns a non-empty string; never raises.
    """
    if mode not in VALID_MODES:
        # Defensive fallback: the gateway already coerces unknown modes
        # to "interrupt" before reaching us, but external callers (tests,
        # custom plugins) may pass anything.
        mode = "interrupt"

    if not templates:
        return DEFAULT_TEMPLATES[mode]

    if not isinstance(templates, Mapping):
        # YAML 1.1 quirks aside, a top-level non-dict is malformed config.
        # Log once and fall back so the busy-turn path keeps working.
        logger.warning(
            "display.busy_ack_templates is not a mapping (%s); ignoring",
            type(templates).__name__,
        )
        return DEFAULT_TEMPLATES[mode]

    override = templates.get(mode)
    if override is None:
        return DEFAULT_TEMPLATES[mode]

    if not isinstance(override, str):
        logger.warning(
            "display.busy_ack_templates[%s] is %s, expected string; using default",
            mode, type(override).__name__,
        )
        return DEFAULT_TEMPLATES[mode]

    # Whitespace-only overrides are treated as "I want to suppress this
    # specific mode entirely"; we honour that by returning the empty
    # string. Callers must guard against empty messages downstream
    # (gateway.run already short-circuits empty sends).
    return override


def render_busy_ack(
    templates: Optional[Mapping[str, Any]],
    mode: str,
    status_detail: str = "",
) -> str:
    """Resolve + render the busy-ack message in one call.

    Wraps :func:`resolve_busy_ack_template` and substitutes the single
    documented placeholder ``{status_detail}``. Unknown placeholders in a
    user-supplied template degrade gracefully -- they're rendered as
    literal text rather than raising ``KeyError`` and dropping the ack
    entirely.
    """
    template = resolve_busy_ack_template(templates, mode)
    if not template:
        return ""
    try:
        return template.format(status_detail=status_detail)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(
            "display.busy_ack_templates[%s] failed to format (%s); using default",
            mode, exc,
        )
        return DEFAULT_TEMPLATES[mode].format(status_detail=status_detail)


# =========================================================================
# Env-var bridge (gateway → subprocess)
# =========================================================================

def encode_templates_for_env(templates: Optional[Mapping[str, Any]]) -> Optional[str]:
    """JSON-encode ``templates`` for transport through ``HERMES_GATEWAY_BUSY_ACK_TEMPLATES``.

    Returns ``None`` when there is nothing to bridge (empty / missing /
    malformed input) so the caller can decide whether to set or unset the
    env var. Only string values for known modes survive the encode;
    everything else is dropped silently to keep the env shape predictable.
    """
    if not templates or not isinstance(templates, Mapping):
        return None
    clean: Dict[str, str] = {}
    for mode in VALID_MODES:
        value = templates.get(mode)
        if isinstance(value, str):
            clean[mode] = value
    if not clean:
        return None
    try:
        return json.dumps(clean, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - JSON of strs is safe
        logger.warning("failed to encode busy_ack_templates for env: %s", exc)
        return None


def load_templates_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Read templates back out of the env var; reverse of :func:`encode_templates_for_env`.

    Returns an empty dict when the env var is unset or malformed -- the
    caller can pass the result straight into :func:`resolve_busy_ack_template`
    which already handles ``{}`` correctly (falls through to defaults).
    """
    source = env if env is not None else os.environ
    raw = source.get(ENV_VAR_TEMPLATES, "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "%s contains malformed JSON (%s); ignoring", ENV_VAR_TEMPLATES, exc,
        )
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            "%s is not a JSON object (%s); ignoring",
            ENV_VAR_TEMPLATES, type(parsed).__name__,
        )
        return {}
    out: Dict[str, str] = {}
    for mode in VALID_MODES:
        value = parsed.get(mode)
        if isinstance(value, str):
            out[mode] = value
    return out
