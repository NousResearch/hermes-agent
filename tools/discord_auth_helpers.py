"""Shared auth helpers for Discord component views.

Extracted from ``plugins.platforms.discord.adapter._component_check_auth``
and ``tools.discord_interactive_views._component_check_auth`` to eliminate
duplication.  Both modules import this via alias to preserve the private
``_component_check_auth`` name at their existing call sites.
"""

from typing import Any, Optional


def component_check_auth(
    interaction: Any,
    allowed_user_ids: Optional[set],
    allowed_role_ids: Optional[set],
) -> bool:
    """User-or-role OR semantics for component interactions.

    Behaviour:

      - both allowlists empty → allow (no-allowlist deployments)
      - user is in user allowlist → allow
      - role allowlist set + user has a matching role → allow
      - role allowlist set + user has no resolvable ``roles`` attribute
        → reject (fail closed — e.g. DM context with a role policy active)
      - otherwise → reject
    """
    user_set = allowed_user_ids or set()
    role_set = allowed_role_ids or set()
    has_users = bool(user_set)
    has_roles = bool(role_set)

    if not has_users and not has_roles:
        return True

    user = getattr(interaction, "user", None)
    if user is None:
        return False

    if has_users:
        try:
            uid = str(user.id)
        except AttributeError:
            uid = ""
        if uid and uid in user_set:
            return True

    if has_roles:
        roles_attr = getattr(user, "roles", None)
        if roles_attr is None:
            # Role policy is configured but the interaction doesn't
            # carry role data (DM-context Member, raw User payload).
            # Fail closed: a user without a resolvable role list cannot
            # satisfy a role allowlist.
            return False
        try:
            user_role_ids = {getattr(r, "id", None) for r in roles_attr}
        except TypeError:
            return False
        if user_role_ids & role_set:
            return True

    return False
