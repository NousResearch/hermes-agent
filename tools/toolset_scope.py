"""Platform scope rules for configured toolsets."""

import ast
from typing import Set, List, Optional


# Toolsets without a restriction entry are available on every platform.
_TOOLSET_PLATFORM_RESTRICTIONS = {"discord": {"discord"}, "discord_admin": {"discord"}}


def toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return whether ``ts_key`` is available on ``platform``."""
    allowed: Set[str] | None = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed


def parse_platform_toolsets_value(value: object) -> Optional[List[str]]:
    """The toolset list a saved ``platform_toolsets.<platform>`` value encodes, or None.

    Older ``hermes config set`` builds stored a bare ``[...]`` argument as a plain string, so an
    explicit selection like ``'["browser", "terminal"]'`` parses as str, not list (#115866).
    Every reader and writer of the section goes through this one parser so the runtime,
    ``hermes doctor`` and ``hermes plugins enable`` agree on what the user configured. Any other
    shape (null, scalar, unparseable string) is None: the caller decides how to report it.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None

