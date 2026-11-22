"""The deployment's launch stamps and the env of a gateway this process starts.

Its own module, not ``local``: a process started on v0.23.3 keeps the ``local`` and
``local_env_policy`` it loaded and imports this lazily after ``hermes update`` moved the
checkout (#158723), so every name here must resolve on those old modules too.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

# URLs a hosting deployment stamps into the gateway's launch environment, each with its credential:
# the status report URL and the brokered sleep URL. Tier 1, so the scrub drops them (the terminal
# tool, credentialed CLIs) and ``served_profile_child_env`` drops them whatever its base; the
# gateway launch builders (``gateway_child_env``) put them back.
DEPLOYMENT_STAMP_ENV: frozenset[str] = frozenset({"GATEWAY_STATUS_REPORT_URL", "GATEWAY_RELAY_SLEEP_URL"})


def drop_deployment_stamps(env: dict[str, str]) -> dict[str, str]:
    """Remove ``DEPLOYMENT_STAMP_ENV`` from *env* in place, any casing (a Windows env block is not
    case-sensitive), for a child built from an unscrubbed base that is not a gateway."""
    for name in [key for key in env if key.upper() in DEPLOYMENT_STAMP_ENV]:
        del env[name]
    return env


def keep_deployment_stamps(env: dict, source: Mapping[str, str] | None = None) -> dict:
    """Put the deployment's launch stamps (``DEPLOYMENT_STAMP_ENV``) from *source* (the process env)
    into the env of a gateway this process starts, where the scrub removed them. A value already in
    *env* (the target profile's own ``GATEWAY_RELAY_SLEEP_URL``) is kept."""
    source = os.environ if source is None else source
    for name in DEPLOYMENT_STAMP_ENV:
        if source.get(name) and not env.get(name):
            env[name] = source[name]
    return env


def gateway_child_env(target_home: str | Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Child env for a gateway this process starts on *target_home*: that profile's environment,
    with each launch stamp it lacks filled from *base* (``keep_deployment_stamps``)."""
    from tools.environments import local  # local imports this module

    # v0.23.3's served_profile_child_env is _profile_child_env under its old name: it drops no stamps.
    profile_env = getattr(local, "_profile_child_env", None)
    env = (profile_env(base, target_home, True) if profile_env is not None
           else local.served_profile_child_env(base, target_home=target_home, inherit_credentials=True))
    return keep_deployment_stamps(env, os.environ if base is None else base)
