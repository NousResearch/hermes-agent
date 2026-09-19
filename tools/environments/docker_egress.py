"""Egress-proxy (iron-proxy) plumbing for the Docker backend.

Builds the mount/env/host args that route a sandbox through the host-side
credential firewall, and guards the three config surfaces (docker_forward_env,
docker_env, docker_extra_args) that could otherwise weaken or bypass it.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger("tools.environments.docker")

_EGRESS_LABEL_KEY = "hermes-egress"
_CONTAINER_CA = "/etc/ssl/certs/hermes-egress-ca.crt"
_NODE_OPTIONS_SENTINEL = "_HERMES_EGRESS_NODE_OPTIONS_APPEND"
_CA_MODE_FLAGS = {"--use-openssl-ca", "--use-bundled-ca"}

# Env names whose override would weaken or bypass enforced egress.
_PROXY_CONTROL_ENV = frozenset({
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
    "NO_PROXY", "no_proxy",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS"})


def _egress_proxy_args_for_docker() -> tuple[list[str], dict[str, str], list[str]]:
    """``(volume_args, env_overrides, host_args)`` for routing through iron-proxy; all empty
    when the proxy is disabled/unconfigured/not running. Under ``proxy.enforce_on_docker``
    (default) any half-configured state raises so the sandbox refuses to start unprotected;
    otherwise it warns and continues. Only ImportError is swallowed — a broken config must
    fail visibly rather than silently disable enforcement."""
    try:
        from hermes_cli.config import load_config
        from agent.proxy_sources import iron_proxy as ip
    except ImportError as exc:
        logger.debug("Egress proxy plumbing unavailable: %s", exc)
        return ([], {}, [])

    proxy_cfg = load_config().get("proxy") or {}
    if not proxy_cfg.get("enabled"):
        return ([], {}, [])

    status = ip.get_status()
    enforce = bool(proxy_cfg.get("enforce_on_docker", True))

    def _degraded(msg: str):
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    if not status.configured:
        return _degraded(
            "proxy.enabled is true but iron-proxy is not configured. "
            "Run `hermes egress setup` to mint tokens and write proxy.yaml.")
    if not (status.pid and status.listening):
        return _degraded(
            f"iron-proxy is enabled but not running on port {status.tunnel_port}. "
            "Start it with `hermes egress start`.")
    if status.ca_cert_path is None or not status.ca_cert_path.exists():
        # Configured a moment ago but the trust anchor vanished: proxy env vars
        # without the CA would make every TLS handshake fail.
        return _degraded(
            f"iron-proxy CA cert vanished from {status.ca_cert_path}. "
            "Re-run `hermes egress setup` to regenerate it.")
    # Empty/corrupt mappings look like an upstream outage from inside the
    # sandbox (every request 403s); refuse rather than ship a broken sandbox.
    mappings = ip.load_mappings()
    if not mappings:
        return _degraded(
            "iron-proxy is configured but mappings.json is empty or "
            "corrupt.  Re-run `hermes egress setup` to mint provider "
            "tokens before starting a sandbox.")

    volume_args = ["-v", f"{status.ca_cert_path}:{_CONTAINER_CA}:ro"]

    # tunnel_port serves CONNECT (HTTPS); the plain-HTTP forward listener is on +1.
    proxy_url = f"http://host.docker.internal:{status.tunnel_port}"
    plain_http_url = f"http://host.docker.internal:{status.tunnel_port + 1}"
    env_overrides: dict[str, str] = {
        # Both casings: some tools only read one.
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": plain_http_url,
        "http_proxy": plain_http_url,
        # Loopback-only so in-sandbox dev servers/local LLMs bypass the proxy.
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
        # CA bundles: Python/curl vars REPLACE the system store, NODE_EXTRA_CA_CERTS
        # only ADDS to it. NODE_OPTIONS=--use-openssl-ca narrows that asymmetry
        # but must be APPENDED to the operator's NODE_OPTIONS, not clobber it —
        # so it travels in a sentinel key that merge_egress_env() resolves.
        "REQUESTS_CA_BUNDLE": _CONTAINER_CA,
        "SSL_CERT_FILE": _CONTAINER_CA,
        "CURL_CA_BUNDLE": _CONTAINER_CA,
        "NODE_EXTRA_CA_CERTS": _CONTAINER_CA,
        "HERMES_EGRESS_PROXY": "1",  # lets the in-sandbox agent know it is proxy-aware
        _NODE_OPTIONS_SENTINEL: "--use-openssl-ca"}

    # Proxy tokens under the standard provider env names (and their aliases) so
    # SDKs work unchanged; HERMES_PROXY_TOKEN_* copies are for diagnostics.
    for m in mappings:
        env_overrides[m.real_env_name] = m.proxy_token
        env_overrides[f"HERMES_PROXY_TOKEN_{m.real_env_name}"] = m.proxy_token
        for alias in getattr(m, "alias_env_names", ()) or ():
            env_overrides[alias] = m.proxy_token

    # Linux needs an explicit host-gateway mapping; Docker Desktop already has it.
    host_args = ["--add-host", "host.docker.internal:host-gateway"]
    return (volume_args, env_overrides, host_args)


def _egress_reuse_fingerprint(
    volume_args: list[str], env_overrides: dict[str, str], host_args: list[str]) -> str:
    """Stable Docker-label value for the egress posture of a container."""
    if not (volume_args or env_overrides or host_args):
        return "off"
    payload = json.dumps(
        {"volume_args": volume_args, "env_overrides": env_overrides, "host_args": host_args},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _egress_enforce_on_docker(default: bool = True) -> bool:
    """Read proxy.enforce_on_docker; any config failure fails safe to *default*."""
    try:
        from hermes_cli.config import load_config
        return bool((load_config().get("proxy") or {}).get("enforce_on_docker", default))
    except (ImportError, OSError):
        return default
    except Exception as exc:  # malformed config.yaml etc.
        logger.warning("Could not read proxy config for egress collision check: %s", exc)
        return default


def _critical_egress_env_names(env_overrides: dict[str, str]) -> set[str]:
    """Env names that would weaken or bypass enforced egress if overridden."""
    critical = set(_PROXY_CONTROL_ENV) | {"NODE_OPTIONS"}
    critical.update(k for k in env_overrides if k.endswith("_API_KEY") or k.endswith("_TOKEN"))
    return critical


def _extra_args_egress_collisions(extra_args: list[str], critical_names: set[str]) -> list[str]:
    """Return docker_extra_args entries that can override egress controls."""
    collisions: list[str] = []
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        flag, sep, inline_value = arg.partition("=")  # ``-e NAME=v`` vs ``-e=NAME=v`` / ``--env-file=f``
        if flag in ("-e", "--env", "--env-file"):
            value = inline_value if sep else (extra_args[i + 1] if i + 1 < len(extra_args) else "")
            name = value.split("=", 1)[0]
            if flag == "--env-file":
                collisions.append(flag)
            elif name in critical_names:
                collisions.append(name)
            i += 1 if sep else 2
            continue
        if flag in ("--network", "--net"):
            collisions.append(arg)
        i += 1
    return sorted(set(collisions))


def _collision_guard(msg: str, *, enforce: bool, remedy: str, consequence: str) -> None:
    """Fail loud under enforcement, otherwise warn and let the user's config win."""
    msg = f"{msg}; enforce_on_docker is {'enabled' if enforce else 'disabled'}."
    if enforce:
        raise RuntimeError(f"{msg}  {remedy}")
    logger.warning("%s  %s", msg, consequence)


def check_forward_env_collisions(forward_env: list[str], critical: set[str], enforce: bool) -> None:
    collisions = sorted(k for k in forward_env if k in critical)
    if collisions:
        _collision_guard(
            f"docker_forward_env would inject real egress-protected variables {collisions}",
            enforce=enforce,
            remedy="Remove these names from docker_forward_env or disable enforce_on_docker "
                   "to opt out of egress isolation.",
            consequence="Explicit docker_forward_env values will override egress tokens.")


def check_docker_env_collisions(user_env: dict[str, str], egress_env: dict[str, str], enforce: bool) -> None:
    """Reject docker_env overrides of proxy-control vars or real provider keys (read from the
    live token mappings; for those ANY override collides since a real key bypasses the swap)."""
    provider_keys: set[str] = set()
    try:
        from agent.proxy_sources import iron_proxy as ip
        provider_keys = {m.real_env_name for m in ip.load_mappings()}
    except Exception:  # best-effort
        pass

    def _collides(k: str) -> bool:
        if k not in user_env:
            return False
        if k in provider_keys:
            return k not in egress_env or user_env[k] != egress_env[k]
        return k in egress_env and user_env[k] != egress_env[k]

    collisions = sorted(k for k in (_PROXY_CONTROL_ENV | provider_keys) if _collides(k))
    if collisions:
        _collision_guard(
            f"docker_env in config.yaml overrides egress-proxy variables {collisions}",
            enforce=enforce,
            remedy="Remove these keys from docker_env or disable enforce_on_docker to opt out "
                   "of egress isolation.",
            consequence="Falling back to docker_env values; sandbox traffic will NOT route "
                        "through the proxy.")


def check_extra_args_collisions(extra_args: list[str], critical: set[str], enforce: bool) -> None:
    collisions = _extra_args_egress_collisions(extra_args, critical)
    if collisions:
        _collision_guard(
            f"docker_extra_args would override egress-proxy controls {collisions}",
            enforce=enforce,
            remedy="Remove these args or disable enforce_on_docker to opt out of egress isolation.",
            consequence="Extra Docker args may bypass egress isolation.")


def merge_egress_env(user_env: dict[str, str], egress_env: dict[str, str], enforce: bool) -> dict[str, str]:
    """Merge docker_env with egress overrides (egress wins under enforcement, docker_env
    otherwise) and resolve the NODE_OPTIONS sentinel: the flag is APPENDED to the operator's
    NODE_OPTIONS after stripping conflicting CA-mode flags so it wins deterministically."""
    merged = {**user_env, **egress_env} if enforce and egress_env else {**egress_env, **user_env}

    raw_append = merged.pop(_NODE_OPTIONS_SENTINEL, None)
    if raw_append:
        append_token = raw_append.strip()
        tokens = merged.get("NODE_OPTIONS", "").split()
        if append_token in _CA_MODE_FLAGS:
            dropped = [t for t in tokens if t in _CA_MODE_FLAGS and t != append_token]
            if dropped:
                logger.warning(
                    "Overriding conflicting NODE_OPTIONS CA-mode flag(s) %s "
                    "with egress-required %s to keep Node routed through the "
                    "egress CA store.", dropped, append_token)
            tokens = [t for t in tokens if t not in _CA_MODE_FLAGS or t == append_token]
        if append_token not in tokens:
            tokens.append(append_token)
        merged["NODE_OPTIONS"] = " ".join(tokens).strip()
        if not merged["NODE_OPTIONS"]:
            merged.pop("NODE_OPTIONS", None)
    return merged
