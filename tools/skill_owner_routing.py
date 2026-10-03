"""Profile ownership routing for agent-created skills.

This module keeps cross-profile admission policy out of ``skill_manager_tool``
so the manager stays below Hermes' 2k production-file boundary. The policy is
owned by the fleet Default profile and is opt-in. When enabled, a new skill can
declare ``metadata.hermes.owner_profile``; Default may route the create into
that registered profile, while named profiles may only create for themselves.
"""

from __future__ import annotations

import logging
from contextvars import Token
from pathlib import Path
from typing import Any, Callable

from agent.skill_utils import parse_frontmatter
from hermes_constants import (
    get_default_hermes_root,
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.config import cfg_get
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_DEFAULT_POLICY: dict[str, bool] = {
    "enabled": False,
    "require_owner_metadata": True,
    "route_from_default": True,
}


def owner_routing_policy() -> dict[str, bool]:
    """Read owner-routing policy from the fleet Default profile.

    ``get_default_hermes_root()`` is the canonical root resolver for standard,
    custom-root, Docker, and profile-scoped deployments. Reading policy under
    a context-local override prevents a specialist's local config from
    weakening the fleet-level cross-profile admission rule.
    """
    try:
        from hermes_cli.config import load_config

        token = set_hermes_home_override(get_default_hermes_root())
        try:
            cfg = load_config() or {}
        finally:
            reset_hermes_home_override(token)

        raw = cfg_get(cfg, "skills", "owner_routing", default={})
        if isinstance(raw, bool):
            return {**_DEFAULT_POLICY, "enabled": raw}
        if not isinstance(raw, dict):
            return dict(_DEFAULT_POLICY)
        return {
            "enabled": is_truthy_value(raw.get("enabled"), default=False),
            "require_owner_metadata": is_truthy_value(
                raw.get("require_owner_metadata"), default=True
            ),
            "route_from_default": is_truthy_value(
                raw.get("route_from_default"), default=True
            ),
        }
    except Exception:
        # Cross-profile authority must fail closed. An unreadable or malformed
        # Default config means routing is disabled, never delegated to the
        # specialist's own config.
        logger.debug("Could not resolve skill owner-routing policy", exc_info=True)
        return dict(_DEFAULT_POLICY)


def declared_skill_owner(content: str) -> str | None:
    """Return ``metadata.hermes.owner_profile`` from SKILL.md frontmatter."""
    try:
        frontmatter, _body = parse_frontmatter(content)
    except Exception:
        return None
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return None
    hermes_meta = metadata.get("hermes")
    if not isinstance(hermes_meta, dict):
        return None
    raw = hermes_meta.get("owner_profile")
    if raw is None:
        return None
    owner = str(raw).strip()
    return owner or None


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def active_profile_from_home() -> str:
    """Derive the active profile from the live context-scoped HERMES_HOME.

    This intentionally does not consult the sticky ``active_profile`` file.
    Long-lived Dashboard/TUI/Desktop processes bind a profile per request via
    ``set_hermes_home_override``; the live home is therefore the authority for
    which profile is performing this mutation.
    """
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    current = _resolved(get_hermes_home())
    root = _resolved(get_default_hermes_root())
    if current == root:
        return "default"

    profiles_root = root / "profiles"
    try:
        relative = current.relative_to(profiles_root)
    except ValueError as exc:
        raise ValueError(
            f"active HERMES_HOME {current} is outside the fleet profile root {profiles_root}"
        ) from exc
    if len(relative.parts) != 1:
        raise ValueError(
            f"active HERMES_HOME {current} is not a direct named-profile home"
        )

    active = normalize_profile_name(relative.parts[0])
    validate_profile_name(active)
    return active


def _validated_owner(raw_owner: str) -> str:
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    owner = normalize_profile_name(raw_owner)
    # Validation happens before any profile filesystem lookup. It rejects path
    # separators, traversal-like values, and reserved identifiers while keeping
    # the special ``default`` root profile valid.
    validate_profile_name(owner)
    return owner


def create_skill_with_owner_routing(
    *,
    name: str,
    content: str,
    category: str | None,
    create_fn: Callable[[str, str, str | None], dict[str, Any]],
    policy: dict[str, bool] | None = None,
) -> tuple[dict[str, Any], Token | None]:
    """Create a skill under the declared owner and return a held scope token.

    ``create_fn`` owns the actual skill-management transaction. For a successful
    cross-profile route the returned token deliberately remains active until the
    adapter has finished that transaction, so the write, audit ledger, prompt
    cache, usage/provenance telemetry, and sync hook all observe the same owner
    profile. Failed and local creates return ``None`` and leave the caller scope
    unchanged.
    """
    resolved_policy = dict(policy) if policy is not None else owner_routing_policy()
    if not resolved_policy.get("enabled"):
        return create_fn(name, content, category), None

    raw_owner = declared_skill_owner(content)
    if not raw_owner:
        if resolved_policy.get("require_owner_metadata", True):
            return (
                {
                    "success": False,
                    "error": (
                        "Skill owner routing is enabled. New skills must declare "
                        "metadata.hermes.owner_profile in SKILL.md frontmatter. "
                        "Choose the profile that owns the capability's primary "
                        "output, not the profile that happened to discover it."
                    ),
                },
                None,
            )
        return create_fn(name, content, category), None

    try:
        owner = _validated_owner(raw_owner)
        active = active_profile_from_home()
    except Exception as exc:
        return (
            {
                "success": False,
                "error": f"Could not resolve skill owner profile {raw_owner!r}: {exc}",
            },
            None,
        )

    from hermes_cli.profiles import get_profile_dir, profile_exists

    if owner != "default" and not profile_exists(owner):
        return (
            {
                "success": False,
                "error": f"Declared skill owner profile {owner!r} is not registered.",
            },
            None,
        )

    if active == owner:
        result = create_fn(name, content, category)
        if result.get("success"):
            result["owner_profile"] = owner
        return result, None

    if active != "default":
        return (
            {
                "success": False,
                "error": (
                    f"Skill {name!r} declares owner_profile={owner!r}, but the active "
                    f"profile is {active!r}. Named specialists may not write skills "
                    "sideways into another specialist. Hand the skill creation to "
                    f"the {owner!r} profile."
                ),
                "owner_profile": owner,
                "active_profile": active,
            },
            None,
        )

    if not resolved_policy.get("route_from_default", True):
        return (
            {
                "success": False,
                "error": (
                    f"Skill {name!r} belongs to {owner!r}. Default owner-routing is "
                    "configured to refuse cross-profile creation; delegate the create."
                ),
                "owner_profile": owner,
            },
            None,
        )

    target_home = get_profile_dir(owner)
    token = set_hermes_home_override(target_home)
    try:
        result = create_fn(name, content, category)
    except BaseException:
        reset_hermes_home_override(token)
        raise

    if not result.get("success"):
        reset_hermes_home_override(token)
        return result, None

    result["owner_profile"] = owner
    result["routed_from_profile"] = "default"
    result["message"] = (
        f"Skill {name!r} created in owner profile {owner!r} (routed from default)."
    )
    result["hint"] = (
        f"Further edits, patches, or supporting-file writes for {name!r} belong "
        f"to profile {owner!r}; hand those mutations to that profile."
    )
    return result, token


def reset_owner_routing_scope(token: Token | None) -> None:
    """Restore the caller's HERMES_HOME after the routed transaction finishes."""
    if token is not None:
        reset_hermes_home_override(token)
