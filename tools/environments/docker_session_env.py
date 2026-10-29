"""Per-exec session identity for shared Docker sandboxes."""

import os


def session_exec_env() -> tuple[dict[str, str], set[str]]:
    """Use the local spawn bridge's authority rules, never a container-time identity.

    Absent names must be explicitly unset inside the container: omitting ``-e``
    alone would inherit a stale value baked into an existing sandbox.
    """
    from gateway.session_context import _VAR_MAP
    from tools.environments.local import _inject_session_context_env

    values = {name: os.environ[name] for name in _VAR_MAP if name in os.environ}
    _inject_session_context_env(values)
    return values, set(_VAR_MAP) - values.keys()
