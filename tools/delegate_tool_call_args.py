"""Validation of model-supplied ``delegate_task`` call arguments.

Both model-facing dispatch sites (``run_agent.AIAgent._dispatch_delegate_task`` and the registry
handler) forward a fixed field list picked out of the LLM's argument dict with ``args.get(...)``.
Any key outside that list evaporates before routing ever sees it: under
``delegation.backend: model_map`` a model that sent ``option`` at the TOP level (beside ``goal``
instead of inside a ``tasks`` entry) spawned a child on the backend's default route while the
call itself reported success — the route silently changed under the caller (sibling of #118825,
which is the same swallow one level down on per-task fields). A stray key must be refused with an
actionable error, never silently dropped.

Two exclusions keep their deliberate silent-drop: ``acp_command`` / ``acp_args`` are a trust
boundary (a model's transport pin is stripped, pinned by ``test_delegate.py``) rather than an
input typo, and an error there would advertise the trusted fields' shape.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Optional

#: Top-level keys intentionally dropped rather than refused: model-supplied ACP transport pins
#: must never reach a child, and the drop is the tested contract (tests/tools/test_delegate.py).
#: Single source of truth: tools/delegate_tool.py strips the same set from per-task entries.
NEVER_FORWARDED_KEYS = frozenset({"acp_command", "acp_args"})

#: delegate_task parameters that exist for internal callers, never for the model.
_INTERNAL_PARAMS = frozenset({"parent_agent", "credentials_cfg"})

#: Routing fields a backend may legitimately declare per-task: `option` (model_map), `profile`
#: (profile_routes), `service_tier`. NOT raw route fields (model/provider/base_url/api_mode) —
#: those are RESERVED_TASK_PARAMETERS, never model-facing at any level, and hinting at them
#: would only earn a second refusal from _route_one.
_ROUTING_HINT_KEYS = frozenset({"option", "profile", "service_tier"})


def model_top_level_keys(delegate_fn: Callable) -> frozenset[str]:
    """The names ``delegate_task`` can receive: its signature minus internal-only kwargs. Built
    ONCE at import from the real function (before any monkeypatch can shadow it) and passed to
    :func:`stray_call_argument_error`, so a patched callee can never desync the gate."""
    params = inspect.signature(delegate_fn).parameters
    return frozenset(
        name for name, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and name not in _INTERNAL_PARAMS
    )


def _stray_hint(strays: set) -> str:
    """A routing field is the common case and the most confusing mistake: it IS valid, just one
    level deeper. Tell the caller exactly where it belongs — but only when EVERY stray is one of
    those fields; with a reserved field mixed in (e.g. `model`), the per-task hint would steer the
    caller into a second refusal from _route_one. The refusal itself lists every stray key."""
    if strays and strays <= _ROUTING_HINT_KEYS:
        if len(strays) == 1:
            field = next(iter(strays))
            example = f'{{"goal": "...", "{field}": "<route value>"}}'
            return (
                " Routing fields are per-task: send them inside a tasks=[...] entry — "
                f"tasks=[{example}] — not at the top level."
                " A single task is a one-entry array."
            )
        return (
            " Routing fields are per-task: send them inside a tasks=[...] entry, "
            "not at the top level. A single task is a one-entry array."
        )
    return ""


def stray_call_argument_error(function_args: Mapping[str, Any], accepted: frozenset[str]) -> Optional[str]:
    """Refusal message for keys a model passed that ``delegate_task`` can never receive, else None.
    Runs at both dispatch sites, where the raw model argument dict is visible; both sites receive
    a dict by construction (``_parse_tool_arguments`` / registry dispatch reject non-object
    arguments before the handler runs)."""
    strays = {k for k in function_args if k not in accepted and k not in NEVER_FORWARDED_KEYS}
    if not strays:
        return None
    listed = ", ".join(repr(k) for k in sorted(strays))
    return (
        f"delegate_task received unknown top-level field(s): {listed}. No subagent was started."
        f"{_stray_hint(strays)} Accepted top-level fields: {', '.join(sorted(accepted))}."
    )
