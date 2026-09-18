"""Backend-neutral egress-proxy (iron-proxy) pieces shared by the Docker and SSH backends.

Every sandbox backend that routes through the credential firewall needs the same two things:
a readiness verdict (proxy enabled, configured, running, CA present, tokens minted) and the
env block that points SDKs at the proxy with opaque tokens under the real provider names.
Only the transport differs (Docker mounts + host-gateway vs. SSH reverse forward + scp).
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("tools.environments.egress")

_NODE_OPTIONS_SENTINEL = "_HERMES_EGRESS_NODE_OPTIONS_APPEND"
_CA_MODE_FLAGS = {"--use-openssl-ca", "--use-bundled-ca"}

# Env names whose override would weaken or bypass enforced egress.
_PROXY_CONTROL_ENV = frozenset({
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
    "NO_PROXY", "no_proxy",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS"})


def ready_egress(degraded: Callable[[str], None]):
    """``(status, mappings)`` when iron-proxy is fully usable, else ``None`` after calling
    *degraded(msg)* (which may raise). Returns ``None`` silently when ``proxy.enabled`` is false."""
    try:
        from hermes_cli.config import load_config
        from agent.proxy_sources import iron_proxy as ip
    except ImportError as exc:
        logger.debug("Egress proxy plumbing unavailable: %s", exc)
        return None

    proxy_cfg = load_config().get("proxy") or {}
    if not proxy_cfg.get("enabled"):
        return None

    status = ip.get_status()
    if not status.configured:
        degraded("proxy.enabled is true but iron-proxy is not configured. "
                 "Run `hermes egress setup` to mint tokens and write proxy.yaml.")
        return None
    if not (status.pid and status.listening):
        degraded(f"iron-proxy is enabled but not running on port {status.tunnel_port}. "
                 "Start it with `hermes egress start`.")
        return None
    if status.ca_cert_path is None or not status.ca_cert_path.exists():
        # Configured a moment ago but the trust anchor vanished: proxy env vars
        # without the CA would make every TLS handshake fail.
        degraded(f"iron-proxy CA cert vanished from {status.ca_cert_path}. "
                 "Re-run `hermes egress setup` to regenerate it.")
        return None
    # Empty/corrupt mappings look like an upstream outage from inside the
    # sandbox (every request 403s); refuse rather than ship a broken sandbox.
    mappings = ip.load_mappings()
    if not mappings:
        degraded("iron-proxy is configured but mappings.json is empty or "
                 "corrupt.  Re-run `hermes egress setup` to mint provider "
                 "tokens before starting a sandbox.")
        return None
    return status, mappings


def egress_env_overrides(proxy_host: str, tunnel_port: int, ca_path: str, mappings) -> dict[str, str]:
    """Proxy-control vars plus proxy tokens under the standard provider env names (and aliases)
    so SDKs work unchanged; ``HERMES_PROXY_TOKEN_*`` copies are for diagnostics."""
    # tunnel_port serves CONNECT (HTTPS); the plain-HTTP forward listener is on +1.
    proxy_url = f"http://{proxy_host}:{tunnel_port}"
    plain_http_url = f"http://{proxy_host}:{tunnel_port + 1}"
    env: dict[str, str] = {
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
        # so it travels in a sentinel key each backend resolves its own way.
        "REQUESTS_CA_BUNDLE": ca_path,
        "SSL_CERT_FILE": ca_path,
        "CURL_CA_BUNDLE": ca_path,
        "NODE_EXTRA_CA_CERTS": ca_path,
        "HERMES_EGRESS_PROXY": "1",  # lets the in-sandbox agent know it is proxy-aware
        _NODE_OPTIONS_SENTINEL: "--use-openssl-ca"}
    for m in mappings:
        env[m.real_env_name] = m.proxy_token
        env[f"HERMES_PROXY_TOKEN_{m.real_env_name}"] = m.proxy_token
        for alias in getattr(m, "alias_env_names", ()) or ():
            env[alias] = m.proxy_token
    return env


def critical_egress_env_names(env_overrides: dict[str, str]) -> set[str]:
    """Env names that would weaken or bypass enforced egress if overridden."""
    critical = set(_PROXY_CONTROL_ENV) | {"NODE_OPTIONS"}
    critical.update(k for k in env_overrides if k.endswith("_API_KEY") or k.endswith("_TOKEN"))
    return critical
