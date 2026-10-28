"""Scoped toolset for the Jackson (Historian / Record Keeper) role.

Source of truth for which tools the Jackson profile may use when resolved
through the ``jackson`` entry of ``TOOLSETS``.  ``toolsets.py`` imports the
resulting definition from here at import time, mirroring the referee and
role-work-state OSM toolsets; this module imports nothing from toolsets.py
to avoid circular imports.

Capability surface (Phase-3 Jackson commissioning, report
``jackson-discovery-and-recovery-plan-2026-10-02.md`` §3.1 precondition 1,
kanban card t_42c7eb46):
- filesystem: read_file, write_file, search_files (knowledge/profile scope;
  source edits go to Clockwork)
- bounded terminal: terminal + process_manage (bounded in practice by the
  per-task workspace dir and output limits; not an unrestricted shell)
- web: web_search, web_extract
- vision: vision_analyze
- session recall: session_search
- persistent memory: memory
- Kanban lifecycle: the full ``kanban_*`` lifecycle surface.  Board-admin
  guards (non-orchestrator profile / delegated-child checks) remain a
  runtime concern enforced inside the kanban tools, so they are NOT
  hard-stripped here — a dispatcher-spawned Jackson run is never stranded,
  and the guard still applies to jackson.
- role-work-state: deliberately NOT listed here.  The dispatcher
  (hermes_cli/kanban_db_dispatch.py::_worker_argv) injects the registered
  ``role-work-state`` toolset for jackson/model-lab workers; listing the
  tool directly would duplicate it.  Reachability is verified at the
  profile-resolution layer, not in this static list.
- bounded Git: runs through terminal (clone/fetch/branch/commit/push) plus
  the registered read-only git evidence tools when present in the live
  registry; Jackson publishes knowledge himself, but holds no
  production-source authority.

Deliberate exclusions (matching SOUL.md boundaries): no patch (general
source-editing lane), no execute_code, no cronjob, no computer_use, no
board-admin kanban mutations at the config layer, no MCP administration,
no config-file service control, no browser automation, no skills
management, no image/video generation, no TTS, no Home Assistant, no
platform messaging, no desktop project/MCP setup, no clarification UI,
no delegation.
"""

from __future__ import annotations

from typing import Any

try:
    from tools.registry import registry
except Exception:  # registry unavailable (early boot) — degrade to static only
    registry = None

# Tools deliberately excluded from Jackson.
_EXCLUDED_TOOLS: set[str] = {
    # General source-editing lane (source edits route to Clockwork)
    "patch",
    # Arbitrary code execution
    "execute_code",
    # Scheduled tasks
    "cronjob_manage",
    # Desktop control
    "computer_use",
    # Board-admin kanban mutations (config-layer belt; the runtime
    # board-mutation guard still applies to non-stewart profiles)
    "kanban_create", "kanban_link", "kanban_unblock",
    # MCP administration / config-file service control
    "manage_connections", "setup_mcp", "desktop_project",
    # Browser automation
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images",
    "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
    "browser_vault_list", "browser_vault_unlock", "browser_vault_fill",
    "browser_vault_save_login", "browser_vault_enter_code", "browser_exec",
    # Skills management
    "skills_list", "skill_view", "skill_manage",
    # Media generation
    "image_generate", "video_generate", "xai_video_edit", "xai_video_extend",
    "text_to_speech",
    # Home Assistant
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
    # Clarification / delegation UI
    "clarify", "delegate_task",
    # Platform messaging
    "discord", "discord_admin", "feishu_doc_read", "feishu_drive",
    "spotify_playback", "spotify_devices", "spotify_queue", "spotify_search",
    "spotify_playlists", "spotify_albums", "spotify_library",
}

# Read-only Git evidence tools granted only when actually registered in the
# live registry (the git tools are registered by an OSM overlay; degrading to
# [] here keeps the resolver stable on stock installs where they are absent).
_GIT_EVIDENCE_TOOLS: set[str] = {
    "git_status", "git_diff", "git_batch_paths", "git_show", "git_log", "git_rev_parse",
}


def _registered_git_tools() -> list[str]:
    if registry is None:
        return []
    try:
        names = {e.name for e in registry.get_all_entries()}
    except Exception:
        return []
    return sorted(_GIT_EVIDENCE_TOOLS & names)


def _jackson_tools() -> list[str]:
    tools = {
        # Core knowledge work (filesystem)
        "read_file", "write_file", "search_files",
        # Bounded terminal (workspace-bounded; not unrestricted)
        "terminal", "process_manage",
        # Web research and extraction
        "web_search", "web_extract",
        # Vision
        "vision_analyze",
        # Session recall and persistent memory
        "session_search", "memory",
        # Kanban lifecycle (full surface; board-admin guard is a runtime check)
        "kanban_show", "kanban_list", "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes", "kanban_heartbeat",
        "kanban_comment", "kanban_attach", "kanban_attach_url", "kanban_attachments",
    }
    tools.update(_registered_git_tools())
    return sorted(tools)


def get_jackson_toolset() -> dict[str, Any]:
    """Return the Jackson toolset dict, compatible with TOOLSETS entries."""
    return {
        "description": (
            "Dr. Daniel Jackson — Historian / Record Keeper. Scoped knowledge "
            "write and bounded Git for knowledge repositories "
            "(openseasonmedia/osm-knowledge); bounded terminal; web, vision, "
            "session search, memory, and Kanban lifecycle. NOT a source author "
            "and NOT a system administrator. Role-work-state is injected by "
            "the dispatcher, not duplicated here."
        ),
        "tools": _jackson_tools(),
        "includes": [],
        "posture": True,
        "_excluded_from_jackson": sorted(_EXCLUDED_TOOLS),
    }


# Module-level constants for convenience and testing.
JACKSON_EXCLUDED_TOOLS = _EXCLUDED_TOOLS
JACKSON_TOOLSET = get_jackson_toolset()
