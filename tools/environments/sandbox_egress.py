"""Outbound allowlist for cloud sandboxes (``terminal.egress`` in config.yaml).

``terminal.egress: {domains: [...], cidrs: [...]}`` limits where commands inside the sandbox can
connect. It is enforced by the sandbox provider's own network layer, never by Hermes: Modal's
``outbound_domain_allowlist`` / ``outbound_cidr_allowlist`` / ``block_network``, Daytona's
``domain_allow_list`` / ``network_allow_list`` / ``network_block_all``, Vercel's ``NetworkPolicy``.
A backend with no native allowlist (local, docker, ssh, managed Modal, plugins) refuses to start
while the key is set, so a configured allowlist is never silently unenforced.

Absent (or ``{}``) = open network, the legacy default. Present with both lists empty = no outbound
network at all. Entries are normalized once here: domains lowercase, ``*.`` only as a leading
wildcard, no scheme/port/path, no IP literals; CIDRs canonical, a bare address gets /32 or /128.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Any, Callable

MAX_ENTRIES = 64
CONFIG_KEY = "terminal.egress"

_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")


@dataclass(frozen=True)
class SandboxEgress:
    """A parsed allowlist. Both lists empty means every outbound connection is blocked."""

    domains: tuple[str, ...]
    cidrs: tuple[str, ...]

    @property
    def blocks_all(self) -> bool:
        return not self.domains and not self.cidrs


def _domain(value: str) -> str:
    lowered = value.strip().lower().rstrip(".")
    wildcard = lowered.startswith("*.")
    host = lowered[2:] if wildcard else lowered
    labels = host.split(".")
    if (len(host) > 253 or len(labels) < 2 or not all(_LABEL.match(label) for label in labels)
            or labels[-1].isdigit()):
        raise ValueError(f"{CONFIG_KEY}.domains: {value!r} is not a hostname (use example.com or "
                         "*.example.com; no scheme, port, path or IP address: put addresses in cidrs)")
    return f"*.{host}" if wildcard else host


def _cidr(value: str) -> str:
    try:
        return str(ipaddress.ip_network(value.strip(), strict=False))
    except ValueError:
        raise ValueError(f"{CONFIG_KEY}.cidrs: {value!r} is not an IP address or CIDR "
                         "(e.g. 203.0.113.7 or 10.0.0.0/8)") from None


def _entries(raw: Any, field: str, normalize: Callable[[str], str]) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
        raise ValueError(f"{CONFIG_KEY}.{field} must be a list of strings")
    if len(raw) > MAX_ENTRIES:
        raise ValueError(f"{CONFIG_KEY}.{field} has {len(raw)} entries; the limit is {MAX_ENTRIES}")
    return tuple(sorted({normalize(entry) for entry in raw}))


def parse_sandbox_egress(raw: Any) -> SandboxEgress | None:
    """``terminal.egress`` -> :class:`SandboxEgress`, ``None`` when absent/empty; ValueError when invalid."""
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{CONFIG_KEY} must be a mapping with domains and/or cidrs lists")
    unknown = set(raw) - {"domains", "cidrs"}
    if unknown:
        raise ValueError(f"{CONFIG_KEY}: unknown key(s) {', '.join(sorted(unknown))} (only domains and cidrs)")
    return SandboxEgress(_entries(raw.get("domains"), "domains", _domain),
                         _entries(raw.get("cidrs"), "cidrs", _cidr))


def sandbox_egress_from_config() -> SandboxEgress | None:
    """The active profile's allowlist (``load_config`` follows the bound profile home)."""
    from hermes_cli.config import load_config_readonly

    return parse_sandbox_egress((load_config_readonly().get("terminal") or {}).get("egress"))


# Backends whose provider enforces the allowlist; every other backend (plugins included) refuses.
ENFORCED_BACKENDS = frozenset({"modal", "daytona", "vercel_sandbox"})
_REFUSAL_REASONS = {
    "local": "commands run directly on this host",
    "docker": "for Docker use proxy.enabled with iron-proxy, or docker_network: false",
    "ssh": "the remote host's own firewall decides its egress",
}


def refuse_unenforceable(backend: str, why: str) -> None:
    raise ValueError(f"{CONFIG_KEY} is set, but the {backend} terminal backend has no provider-enforced "
                     f"network allowlist ({why}). Use backend modal (direct), daytona or vercel_sandbox, "
                     f"or remove {CONFIG_KEY} from config.yaml.")


def egress_for_backend(env_type: str, *, hosted: bool = False) -> SandboxEgress | None:
    """The allowlist *env_type* must enforce, ``None`` when unset; ValueError where it cannot be."""
    egress = sandbox_egress_from_config()
    if egress is None:
        return None
    if hosted:
        refuse_unenforceable(f"hosted {env_type}", "a hosted scope's sandbox is created by its plugin")
    if env_type not in ENFORCED_BACKENDS:
        refuse_unenforceable(env_type, _REFUSAL_REASONS.get(env_type, "plugin backends declare no allowlist"))
    return egress


def modal_create_kwargs(egress: SandboxEgress) -> dict[str, Any]:
    """``modal.Sandbox.create`` kwargs. Modal matches domains on TLS SNI (port 443 only); CIDRs
    allow any port. The two lists are additive."""
    if egress.blocks_all:
        return {"block_network": True}
    return {"outbound_domain_allowlist": list(egress.domains), "outbound_cidr_allowlist": list(egress.cidrs)}


def daytona_create_kwargs(egress: SandboxEgress) -> dict[str, Any]:
    """``CreateSandboxFromImageParams`` / ``update_network_settings`` kwargs. Daytona takes one
    list per sandbox and IPv4 CIDRs only, so anything it cannot enforce exactly is refused."""
    if egress.domains and egress.cidrs:
        raise ValueError(f"{CONFIG_KEY}: Daytona enforces either a domain list or a CIDR list, not both. "
                         "Keep one of domains or cidrs.")
    v6 = [c for c in egress.cidrs if ipaddress.ip_network(c).version == 6]
    if v6:
        raise ValueError(f"{CONFIG_KEY}.cidrs: Daytona accepts IPv4 ranges only; remove {', '.join(v6)}")
    if egress.domains:
        return {"domain_allow_list": ",".join(egress.domains)}
    if egress.cidrs:
        return {"network_allow_list": ",".join(egress.cidrs)}
    return {"network_block_all": True}


def vercel_network_policy(egress: SandboxEgress) -> Any:
    """``vercel.sandbox.NetworkPolicy`` for the allowlist (SDK imported by the caller first)."""
    from vercel.sandbox import NetworkPolicy, NetworkPolicySubnets

    if egress.blocks_all:
        return NetworkPolicy.deny_all()
    subnets = NetworkPolicySubnets(allow=egress.cidrs) if egress.cidrs else None
    return NetworkPolicy.custom(allow={domain: () for domain in egress.domains}, subnets=subnets)
