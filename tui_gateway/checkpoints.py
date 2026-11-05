"""Checkpoint-enablement resolution for TUI-created agents (issue #79625).

Desktop sessions (``hermes serve`` → TUI gateway) never set ``HERMES_TUI_CHECKPOINTS``
(that env var is only set by the ``hermes --tui --checkpoints`` CLI path), so agent
creation previously gated filesystem checkpointing on the env var alone and silently
ignored ``checkpoints.enabled`` in config.yaml. These helpers resolve enablement the
way the messaging gateway does (``gateway/run.py`` ``_checkpoint_agent_kwargs``):
the env var (explicit CLI opt-in) wins when set, otherwise the ``checkpoints``
config section applies with ``DEFAULT_CONFIG`` defaults.
"""

import os

from utils import is_truthy_value


def _load_checkpoints_enabled(cfg: dict | None = None) -> bool:
    """Whether filesystem checkpoints are enabled for TUI-created agents.

    Keeps the legacy ``checkpoints: true`` bool form working and falls back to the
    ``DEFAULT_CONFIG`` default when the section is absent or malformed.
    """
    cp_cfg = (cfg or {}).get("checkpoints", {})
    if isinstance(cp_cfg, bool):
        cp_cfg = {"enabled": cp_cfg}
    elif not isinstance(cp_cfg, dict):
        cp_cfg = {}
    from hermes_cli.config import DEFAULT_CONFIG
    return bool(cp_cfg.get("enabled", DEFAULT_CONFIG["checkpoints"]["enabled"]))


def resolve_checkpoints_enabled(cfg: dict | None = None) -> bool:
    """Resolve checkpoint enablement with the env-var override first.

    ``HERMES_TUI_CHECKPOINTS`` (set by ``hermes --tui --checkpoints``) is the
    explicit opt-in and wins when present; otherwise fall back to the
    ``checkpoints`` config section (issue #79625 — the desktop backend never
    sets the env var, so config was being silently ignored).
    """
    env_val = os.environ.get("HERMES_TUI_CHECKPOINTS")
    if env_val is not None:
        return is_truthy_value(env_val)
    return _load_checkpoints_enabled(cfg)


def revert_undone_files(session: dict, live_view: dict) -> list[str]:
    """``/undo``'s file half on the TUI sidecar: put back the files the undone turns wrote, via the
    session agent's checkpoint manager. A persisted session addresses the turn by the rewound row's
    durable id (``undo_row_files``, the canonical path's helper: it follows a compaction restatement
    back to the turn's original stamp); a session without a key by the live row's timestamp. Lines
    naming what was reverted/kept; empty with checkpoints off or no file writes."""
    from tools.checkpoint_turns import summarize, undo_row_files, undo_turn_files
    from tui_gateway.server import _session_cwd, _session_db, _session_profile_runtime_scope
    agent = session.get("agent")
    mgr = getattr(agent, "_checkpoint_mgr", None)
    sessions = (getattr(agent, "session_id", None), session.get("session_key"))
    try:
        # The session's profile, not the launch one: the checkpoint store lives under its home.
        with _session_profile_runtime_scope(session, hydrate_secrets=False), _session_db(session) as db:
            row_id = live_view.get("_rewound_row_id")
            result = (undo_row_files(mgr, db, row_id, {s for s in sessions if s}) if type(row_id) is int
                      else undo_turn_files(mgr, live_view.get("timestamp"), db, *sessions))
    except Exception as exc:  # health: allow BLE001 -- the transcript undo already committed; report it
        result = {"errors": [str(exc)]}
    return summarize(result, _session_cwd(session))


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    """``/rollback N`` style number → the checkpoint's hash; a real hash passes through."""
    try:
        checkpoints = mgr.list_checkpoints(cwd)
        idx = int(ref) - 1
    except ValueError:
        return ref
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx].get("hash", ref)
    raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")
