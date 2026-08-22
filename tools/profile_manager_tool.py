#!/usr/bin/env python3
"""
Profile Manager Tool -- Agent-Managed Profile (Bot) Creation & Configuration

Lets the agent create and configure Hermes profiles from inside a session, so
"spin up a bot for each of these workstreams" resolves through the governed
tool loop instead of a shell-out or a `computer_use` run against the desktop
New Agent dialog.

A profile *is* a Bot (see the Bot Mode user guide): isolated config, memory,
skills, credentials, and chat history under ``<profiles_root>/<name>/``. This
tool is a thin, validated wrapper over the same ``hermes_cli.profiles``
primitives the CLI uses -- it does not reimplement profile creation.

Actions:
  list       -- Enumerate existing profiles (check for a fit before creating)
  create     -- Create a profile, optionally with a title, description, SOUL.md
  configure  -- Update title / description / SOUL.md on an existing profile

Deliberately NOT included: profile deletion. Removing a profile destroys its
sessions, memory, and credentials; that stays behind the CLI's and desktop's
destructive-confirmation flows.

Off by default. The agent only sees this tool when the profile explicitly
enables the ``profiles`` toolset in config.yaml:

    toolsets:
      - profiles

which mirrors how the ``kanban`` toolset opts in (tools/kanban_tools.py).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# SOUL.md is a persona/standing-instruction document, not a data dump. Bound it
# so a runaway generation cannot write an unbounded file into the profile dir.
MAX_SOUL_CHARS = 100_000

VALID_ACTIONS = ("list", "create", "configure")


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------

def _profile_has_profiles_toolset() -> bool:
    """True when the active profile opts into the ``profiles`` toolset.

    Uses ``load_config()`` which is mtime-cached, and the registry TTL-caches
    check_fn results (~30s), so this is cheap to call per schema build.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        toolsets = cfg.get("toolsets", [])
        return "profiles" in toolsets
    except Exception:
        return False


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _check_profile_manage_mode() -> bool:
    """Expose ``profile_manage`` only to explicitly opted-in top-level agents.

    ``delegate_task`` children are excluded: they run in the parent's process
    and inherit its environment, so a subagent given a narrow research goal
    should not be able to mint persistent profiles as a side effect. The
    parent can create the profile and delegate into it.
    """
    if _is_delegated_child_context():
        return False
    return _profile_has_profiles_toolset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_profile_name(name: str) -> str:
    """Normalize + validate a profile id, raising ValueError when unusable.

    Normalization happens before validation so title-cased input from a model
    ("Researcher") resolves to the on-disk id ("researcher") rather than being
    rejected -- the same ingress rule the dashboard and CLI follow (#18498).
    """
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    return canon


def _write_soul(profile_dir: Path, soul: str) -> None:
    """Write SOUL.md inside *profile_dir*.

    ``profile_dir`` comes from ``get_profile_dir()`` on an already-validated
    id, so the path is not attacker-influenced; the join is a fixed filename.
    """
    (profile_dir / "SOUL.md").write_text(soul, encoding="utf-8")


def _profile_row(info) -> dict:
    """Serialize a ``ProfileInfo`` into the shape the agent sees."""
    return {
        "name": info.name,
        "display_name": info.display_name or "",
        "description": info.description or "",
        "model": info.model,
        "provider": info.provider,
        "skill_count": info.skill_count,
        "is_default": info.is_default,
    }


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _action_list() -> str:
    from tools.registry import tool_result
    from hermes_cli.profiles import list_profiles

    profiles = [_profile_row(p) for p in list_profiles()]
    return tool_result({"profiles": profiles, "count": len(profiles)})


def _action_create(
    name: str,
    display_name: Optional[str],
    description: Optional[str],
    soul: Optional[str],
    clone_from: Optional[str],
    no_skills: bool,
) -> str:
    from tools.registry import tool_error, tool_result
    from hermes_cli.profiles import (
        create_profile,
        profile_exists,
        write_profile_meta,
    )

    canon = _resolve_profile_name(name)

    if canon == "default":
        return tool_error(
            "Cannot create a profile named 'default' -- it is the built-in "
            "profile. Pick a distinct name."
        )
    if profile_exists(canon):
        return tool_error(
            f"Profile '{canon}' already exists. Use action='configure' to "
            f"change it, or pick a different name."
        )

    clone_canon = None
    if clone_from:
        clone_canon = _resolve_profile_name(clone_from)
        if not profile_exists(clone_canon):
            return tool_error(f"Clone source profile '{clone_canon}' does not exist.")

    # create_profile rejects this combination itself, but failing before any
    # directory is created keeps the error clean and leaves no partial profile.
    if no_skills and clone_canon:
        return tool_error(
            "no_skills is mutually exclusive with clone_from -- cloning "
            "explicitly copies the source profile's skills."
        )

    profile_dir = create_profile(
        canon,
        clone_from=clone_canon,
        clone_config=bool(clone_canon),
        no_skills=no_skills,
    )

    # Identity metadata and SOUL.md are applied after the directory exists.
    # A failure here leaves a usable profile, so report it rather than raising:
    # the agent can retry with action='configure' instead of re-creating.
    warnings = []
    if display_name or description:
        try:
            write_profile_meta(
                profile_dir,
                description=description if description is not None else None,
                description_auto=False if description else None,
                display_name=display_name if display_name is not None else None,
            )
        except Exception as e:
            warnings.append(f"profile created but metadata not written: {e}")

    if soul:
        try:
            _write_soul(profile_dir, soul)
        except Exception as e:
            warnings.append(f"profile created but SOUL.md not written: {e}")

    result = {
        "created": canon,
        "path": str(profile_dir),
        "display_name": display_name or "",
        "description": description or "",
        "soul_written": bool(soul) and not any("SOUL.md" in w for w in warnings),
        "cloned_from": clone_canon,
        "chat_command": f"hermes -p {canon} chat",
    }
    if warnings:
        result["warnings"] = warnings
    return tool_result(result)


def _action_configure(
    name: str,
    display_name: Optional[str],
    description: Optional[str],
    soul: Optional[str],
) -> str:
    from tools.registry import tool_error, tool_result
    from hermes_cli.profiles import (
        get_profile_dir,
        profile_exists,
        write_profile_meta,
    )

    canon = _resolve_profile_name(name)
    if not profile_exists(canon):
        return tool_error(
            f"Profile '{canon}' does not exist. Use action='create' first, or "
            f"action='list' to see what exists."
        )
    if display_name is None and description is None and soul is None:
        return tool_error(
            "configure requires at least one of: display_name, description, soul."
        )

    profile_dir = get_profile_dir(canon)
    applied = []

    if display_name is not None or description is not None:
        write_profile_meta(
            profile_dir,
            description=description,
            # Only stamp description_auto when a description is actually set;
            # None leaves the stored value untouched.
            description_auto=False if description is not None else None,
            display_name=display_name,
        )
        if display_name is not None:
            applied.append("display_name")
        if description is not None:
            applied.append("description")

    if soul is not None:
        _write_soul(profile_dir, soul)
        applied.append("soul")

    return tool_result({"configured": canon, "applied": applied})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def profile_manage(
    action: str = "",
    name: str = "",
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    soul: Optional[str] = None,
    clone_from: Optional[str] = None,
    no_skills: bool = False,
    **kwargs,
) -> str:
    """Create, configure, or list Hermes profiles (Bots). Returns JSON."""
    from tools.registry import tool_error

    action = (action or "").strip().lower()
    if action not in VALID_ACTIONS:
        return tool_error(
            f"Unknown action {action!r}. Valid actions: {', '.join(VALID_ACTIONS)}."
        )

    if action in ("create", "configure") and not (name or "").strip():
        return tool_error(f"action='{action}' requires a profile 'name'.")

    if soul is not None and len(soul) > MAX_SOUL_CHARS:
        return tool_error(
            f"soul is {len(soul)} chars, over the {MAX_SOUL_CHARS} limit. "
            f"SOUL.md holds persona and standing instructions -- put reference "
            f"material in a skill instead."
        )

    try:
        if action == "list":
            return _action_list()
        if action == "create":
            return _action_create(
                name=name,
                display_name=display_name,
                description=description,
                soul=soul,
                clone_from=clone_from,
                no_skills=bool(no_skills),
            )
        return _action_configure(
            name=name,
            display_name=display_name,
            description=description,
            soul=soul,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        # Expected, actionable failures: bad name, name collision, missing dir.
        return tool_error(str(e))
    except PermissionError as e:
        return tool_error(f"Permission denied: {e}")
    except Exception as e:
        logger.exception("profile_manage failed: action=%s name=%s", action, name)
        return tool_error(f"profile_manage failed: {e}")


PROFILE_MANAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "profile_manage",
        "description": (
            "Create, configure, and list Hermes profiles (Bots). A profile is a "
            "Bot: its own config, memory, skills, and chat history. Use this to "
            "turn a discussion into a roster of specialist agents -- one profile "
            "per workstream -- instead of driving the desktop UI.\n\n"
            "Call action='list' FIRST to check whether a suitable profile already "
            "exists; prefer configuring an existing Bot over creating a near-"
            "duplicate. Give each new profile a 'description' (one or two "
            "sentences on what it is good at) so task routers can dispatch to it "
            "by role, and a 'soul' capturing the persona and standing "
            "instructions agreed in the conversation.\n\n"
            "A new profile is scaffolding, not a running Bot: it starts with "
            "no API credentials, no provider/model config of its own, and no "
            "running gateway, so its first chat cannot reach a provider until "
            "a human adds credentials via `hermes -p <name> auth add` or the "
            "desktop Bots pane. Say so when reporting a profile you created, "
            "rather than implying it is ready to talk. The exception is "
            "'clone_from', which copies the source profile's config.yaml and "
            ".env -- a clone inherits that Bot's provider settings and keys.\n\n"
            "Profiles cannot be deleted through this tool. Chat with a created "
            "profile via `hermes -p <name> chat`, or open it in the desktop "
            "Bots pane."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(VALID_ACTIONS),
                    "description": (
                        "list: enumerate existing profiles. create: make a new "
                        "one. configure: update an existing one."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Profile id, lowercase [a-z0-9][a-z0-9_-]{0,63} (mixed "
                        "case is lowercased). Required for create/configure."
                    ),
                },
                "display_name": {
                    "type": "string",
                    "description": (
                        "Human-facing title shown in the roster, e.g. 'Research "
                        "Lead'. Empty string clears it."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "One or two sentences on what this Bot is good at. Used "
                        "to route work by role rather than by name alone."
                    ),
                },
                "soul": {
                    "type": "string",
                    "description": (
                        "Full SOUL.md content -- the Bot's persona and standing "
                        "instructions. Overwrites any existing SOUL.md."
                    ),
                },
                "clone_from": {
                    "type": "string",
                    "description": (
                        "Existing profile to copy config, skills, and SOUL.md "
                        "from. Also copies the source's .env, so the clone "
                        "inherits its credentials. Omit for a fresh profile "
                        "with bundled skills and no credentials."
                    ),
                },
                "no_skills": {
                    "type": "boolean",
                    "description": (
                        "Create a minimal profile with no bundled skills. "
                        "Mutually exclusive with clone_from."
                    ),
                    "default": False,
                },
            },
            "required": ["action"],
        },
    },
}


# --- Registry ---
from tools.registry import registry  # noqa: E402

registry.register(
    name="profile_manage",
    toolset="profiles",
    schema=PROFILE_MANAGE_SCHEMA,
    handler=lambda args, **kw: profile_manage(
        action=args.get("action", ""),
        name=args.get("name", ""),
        display_name=args.get("display_name"),
        description=args.get("description"),
        soul=args.get("soul"),
        clone_from=args.get("clone_from"),
        no_skills=args.get("no_skills", False),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
    ),
    check_fn=_check_profile_manage_mode,
    emoji="🤖",
)
