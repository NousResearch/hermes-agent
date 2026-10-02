"""Shared validation for the two desktop settings mirrored server-side:
``hermes.desktop.pluginDecisions.v2`` and ``hermes.desktop.keybinds``.

Used by ``config.set`` to validate the shape of server-side desktop settings
without logging or echoing their values.

Never log or return the decoded value; callers that need presence/shape only
should catch ``ValueError`` and report the message, not the payload.
"""
from __future__ import annotations

# Upper bound on a mirrored setting payload.
MAX_BYTES = 1_000_000


def _check_size(obj: object, label: str) -> None:
    # A cheap, good-enough bound: real repr length tracks the eventual JSON
    # size closely enough to catch a runaway payload before it is written.
    if len(repr(obj)) > MAX_BYTES:
        raise ValueError(f'{label} payload too large')


def validate_plugin_decisions(obj: object) -> dict[str, bool]:
    """``{plugin_id: bool}`` — explicit enable/disable choices. Absence of a key
    means \"no choice\", so this never invents a default; it only validates shape."""
    if not isinstance(obj, dict):
        raise ValueError('pluginDecisions must be a JSON object')
    _check_size(obj, 'pluginDecisions')
    for key, value in obj.items():
        if not isinstance(key, str) or not key:
            raise ValueError('pluginDecisions keys must be non-empty strings')
        if not isinstance(value, bool):
            raise ValueError(f'pluginDecisions[{key!r}] must be a boolean')
    return dict(obj)


def validate_keybinds(obj: object) -> dict[str, list[str]]:
    """``{action_id: [combo, ...]}`` — only the diff from shipped defaults is ever
    stored (desktop keybinds.ts), so an empty list is a deliberately cleared binding,
    not a missing one; this validator accepts that shape as-is."""
    if not isinstance(obj, dict):
        raise ValueError('keybinds must be a JSON object')
    _check_size(obj, 'keybinds')
    for key, value in obj.items():
        if not isinstance(key, str) or not key:
            raise ValueError('keybinds keys must be non-empty strings')
        if not isinstance(value, list) or any(not isinstance(c, str) for c in value):
            raise ValueError(f'keybinds[{key!r}] must be a list of strings')
    return {k: list(v) for k, v in obj.items()}
