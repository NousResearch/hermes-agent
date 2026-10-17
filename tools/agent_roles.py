"""Bounded agent roles for ``delegate_task``.

Ported semantic from OpenAI Codex's agent-role system
(``codex-rs/core/src/agent/role.rs``): a role is a *bounded override* applied
to a delegated child agent. A role may customize the child's instructions or
trim its capabilities, but it may never raise the child above the parent's
authority.

Codex guarantees this with a projected config layer that only *reduces*
(the comment in ``role.rs``: "Roles may customize the child or reduce its
capabilities, but never replace the parent session's authority"). Hermes
mirrors the invariant with three checks:

- **Instructions** are appended to (never replace) the child system prompt.
- **Model** override only applies when the caller also passed one or the
  parent's model is otherwise routable; a role cannot mint credentials the
  parent does not have (``override_*`` credential params stay exclusive to
  delegation config, unchanged).
- **enabled_toolsets** are intersected with the parent's toolsets — a child
  must never gain tools the parent lacks (same rule ``_build_child_agent``
  already applies to caller-supplied toolsets).

Roles are configured in ``config.yaml``::

    delegation:
      roles:
        explorer:
          instructions: "You are a fast codebase explorer. Return concise answers with file:line citations."
          enabled_toolsets: [terminal, file]
        reviewer:
          instructions: "You are a security reviewer. Flag risks with severity."
          model: "gpt-5.6-sol"

The ``delegate_task`` ``role`` parameter accepts a role name in addition to
the built-in ``leaf`` / ``orchestrator``. Unknown names still coerce to
``leaf`` (backward compatible).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentRole:
    """One configured bounded role for delegated children."""

    name: str
    instructions: str = ""
    model: Optional[str] = None
    enabled_toolsets: Optional[List[str]] = None

    @property
    def is_builtin(self) -> bool:
        return self.name in ("leaf", "orchestrator")


def _load_delegation_config() -> dict:
    """Load the delegation config block (same path delegate_tool uses)."""
    try:
        from hermes_cli.config import load_config_readonly

        full = load_config_readonly()
        cfg = full.get("delegation") or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        try:
            from cli import CLI_CONFIG

            cfg = CLI_CONFIG.get("delegation") or {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}


def get_agent_roles() -> Dict[str, AgentRole]:
    """Return configured bounded roles keyed by name.

    Reads ``delegation.roles`` from the active config. Malformed entries are
    skipped with a warning (a broken role must never break delegation).
    """
    cfg = _load_delegation_config()
    roles_raw = cfg.get("roles") or {}
    if not isinstance(roles_raw, dict):
        return {}
    roles: Dict[str, AgentRole] = {}
    for name, raw in roles_raw.items():
        if not isinstance(raw, dict):
            logger.warning("delegation.roles.%s: expected a mapping, skipping", name)
            continue
        try:
            roles[str(name)] = AgentRole(
                name=str(name),
                instructions=str(raw.get("instructions") or ""),
                model=(
                    str(raw["model"]) if isinstance(raw.get("model"), str) and raw["model"] else None
                ),
                enabled_toolsets=(
                    [str(t) for t in raw["enabled_toolsets"]]
                    if isinstance(raw.get("enabled_toolsets"), list)
                    else None
                ),
            )
        except Exception as e:
            logger.warning("delegation.roles.%s: malformed (%s), skipping", name, e)
    return roles


def resolve_role(role_name: Optional[str]) -> Optional[AgentRole]:
    """Resolve a caller-provided role name to a configured role.

    Built-ins (``leaf`` / ``orchestrator``) and unknown names return None —
    they are handled by the built-in delegate path unchanged.
    """
    if not role_name:
        return None
    name = str(role_name).strip().lower()
    if name in ("leaf", "orchestrator"):
        return None
    return get_agent_roles().get(name)


def apply_role_instructions(base_prompt: str, role: Optional[AgentRole]) -> str:
    """Append a role's instructions to the child system prompt (never replace)."""
    if role is None or not role.instructions:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"## Agent role: {role.name}\n"
        f"{role.instructions}"
    )


def apply_role_toolsets(
    child_toolsets: List[str],
    role: Optional[AgentRole],
) -> List[str]:
    """Intersect a role's enabled_toolsets with the child's (never widen).

    The caller has already intersected with the parent's toolsets; intersecting
    again with the role's list can only narrow the child — the bounded-override
    invariant.
    """
    if role is None or not role.enabled_toolsets:
        return child_toolsets
    allowed = set(role.enabled_toolsets)
    return [t for t in child_toolsets if t in allowed]


def apply_role_model(
    role: Optional[AgentRole],
    caller_model: Optional[str],
    parent_model: Optional[str],
) -> Optional[str]:
    """Resolve the child model under a role override.

    Precedence: caller-supplied model > role model > parent model. A role can
    only point at a model string (provider/credentials are inherited), so it
    can never mint auth the parent does not have.
    """
    if caller_model:
        return caller_model
    if role is not None and role.model:
        return role.model
    return parent_model
