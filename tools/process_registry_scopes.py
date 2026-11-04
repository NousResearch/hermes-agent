"""The naming and parsing of Hermes' transient per-worker systemd scopes.

``tools.process_registry`` mints a background child's scope as
``hermes-worker-<suffix>`` (``_build_systemd_scope_argv``); the kanban dispatcher
passes ``kanban-<task id>-run-<run id>``, so its scopes are
``hermes-worker-kanban-t_<8 hex>-run-<n>.scope``. Both the minting side and the
dispatcher's orphan-scope sweep need that shape, and ``process_registry`` sits at
its ``FILE_LINES`` cap, so the shared vocabulary lives here instead.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

WORKER_SCOPE_PREFIX = "hermes-worker-"

# Same shape ``_restart_safe_worker_argv`` mints: 12 hex chars (``t_`` + 4 hex
# bytes), then the run id. Anchored, so a hand-named or legacy unit is ignored
# rather than guessed at.
_KANBAN_WORKER_SCOPE_RE = re.compile(
    r"^" + re.escape(WORKER_SCOPE_PREFIX) + r"kanban-(t_[0-9a-f]{8})-run-(\d+)\.scope$"
)


def parse_kanban_worker_scope(unit: str) -> Optional[tuple[str, int]]:
    """``(task_id, run_id)`` from a kanban worker scope name, or ``None``.

    ``None`` for anything that is not exactly
    ``hermes-worker-kanban-t_<8 hex>-run-<n>.scope`` — including non-kanban Hermes
    worker scopes, hand-named units (``t_oauth_deploy``) and operator-created
    units. Callers must treat ``None`` as "not mine to touch", never as "parse
    loosely": the only safe failure mode for a sweep that stops systemd units is
    to leave an unfamiliar one alone.
    """
    match = _KANBAN_WORKER_SCOPE_RE.match(unit or "")
    if match is None:
        return None
    return match.group(1), int(match.group(2))
